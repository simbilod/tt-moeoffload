"""GLM's MLA query side as one program (tt/kernels/glm_fmq_*.cpp), real blk.3 weights:
hn -> q_a | kv_a | i_proj | i_k | i_gate (A), the q_a norm's rsqrt on every core and q_b | i_q_b (B), the absorb into
the flash-MLA input tiles beside the indexer's pool scores (C), the host's top 512 at barrier 3 with the 2,048 token
ids written back for the gather. It replaces g3_mla_layer.py's stock projections, the indexer's inputs and tail and
its separate launch.

Checked per phase against float64 on the card's weights from the card's own previous outputs (check tensors, the
D2H page), and q_abs end to end against G2's float32 formulas; the selection against float64's top 512. Timed traced
with the host serving the round trip, against the stock route's 435.8 + 94.1 + 102.5 us + the indexer's launch.

  python3 tt_moeoffload/launch.py --cpus all --memory-max $((24 << 30)) --engine tt/glm/g3_fmq.py
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
TT = os.path.dirname(HERE)
KDIR = os.path.join(TT, "kernels")
sys.path.insert(0, HERE)
sys.path.insert(1, TT)
sys.path.append(os.path.join(TT, "flashnext"))                      # u1_sockets only
from g2_layers import CTX, IH, IHD, ITOP, KPOOL, MLA_LAYER, MODEL, OUT, RMS_EPS, Gguf, pcc  # noqa: E402
assert os.path.dirname(sys.modules["g2_layers"].__file__) == HERE
from fused.grid import NOC1_ROWS, Grid, split, split_weighted  # noqa: E402
from fused.mrows import blocks, mp_of, pad_rows, step_words, unblocks  # noqa: E402
from g3_fmo import stream  # noqa: E402
from g3_idx import _LIB  # noqa: E402
from u1_sockets import pinned  # noqa: E402

D, NH, HK, KVL, QL = 4096, 64, 256, 512, 1536
NA, NB, NQA = 73, 640, 48
IDLE, W_PAGE = 0xFFFFFFFF, 1088
BATCH, DEPTH = 8, 8                              # the ring's batch divides every K here (128, 48, 8)
NSEL = ITOP // KPOOL
NS = 2112                                        # the selection's slots: 2,048 + the open pool's 3, to k chunks of 64
A2_BYTES, IQ_BYTES = (NA - NQA) * 128, IH * IHD * 2
RB = NS * 4 + 1024 + 64 + 512                    # the reply (glm_loop.cpp's mla_step)
_LIB.gl_mla_new.argtypes = [ctypes.c_void_p] * 4
_LIB.gl_mla_step.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_LIB.gl_pools.argtypes = [ctypes.c_void_p]
_LIB.gl_mla_rows.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


def mla_state(r) -> tuple[int, list]:
    """A new MLA layer's host decode state at position 0 (glm_loop.cpp's gl_mla_new) -> its handle and the weight
    arrays it points at (keep them alive)."""
    ws = [np.ascontiguousarray(r[k].float().numpy().reshape(-1), np.float32)
          for k in ("kv_a_norm", "i_k_norm", "i_k_norm_b", "i_ape")]
    return _LIB.gl_mla_new(*(w.ctypes.data for w in ws)), ws


def last_pools(lib=_LIB) -> np.ndarray:
    """the last SEL exchange's selected pools (Python's serve or the C++ loop's)"""
    p = np.zeros(NSEL, np.uint32)
    return p[:lib.gl_pools(p.ctypes.data)]
REPS = int(os.environ.get("FMQ_REPS", 50))


class MlaQuery:
    """r: the layer's tensors as llama.cpp holds them (G2's MlaCard.ref names); K: the pooled indexer keys (bf16 tiles
    [n_pool, 128], n_pool the capacity); hn: the input (row-major bf16 [1, 1, 1, 4096]); qg: the two flash-MLA query
    groups it writes (bf16 tiles [1, 1, 32, 512]); sel: the selection's NS token ids (row-major uint32 [1, NS]); cpos:
    the attention's cur_pos (row-major int32 [1]); lat: the latent cache (row-major bf16 [n_pool * 4, 512]); state:
    the host decode state's handle (mla_state), or -1: the stateless top 512 of all n_pool, no cache writes. Its own
    socket pair on core (0, 0): build it before any trace capture. serve() is the host's side of a launch."""

    def __init__(self, dev, r, K, n_pool, hn, qg, sel, cpos, lat, state=-1, link=None, check=True, pf=None,
                 m_max=1, step=None):
        """link: another MlaQuery's (h2d, d2h), shared (the MLA layers' pages are the same size); pf: (an L1
        prefetch buffer, pages): phase A's first pages come from it (g3_fkda.Mixer's pf). m_max > 1: the M-row build
        (glm_fmq_reader.cpp): hn in M_MAX-row blocks, qg [1, M_MAX, 32, 512], sel [M_MAX, NS], cpos
        [M_MAX], the pass's M from step (the step tensor, tt/fused/mrows.py)."""
        self.dev, self.n_pool, self.hn, self.qg, self.sel, self.state = dev, n_pool, hn, qg, sel, state
        self.m_max, self.M = m_max, 1
        self.row_bytes = n_pool * 4 + A2_BYTES + IQ_BYTES
        self.page_bytes = m_max * self.row_bytes
        if link is None:
            where = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
            h2d = pinned(lambda: ttnn.H2DSocket(dev, where, ttnn.BufferType.L1, 4 * m_max * RB,
                                                ttnn.H2DMode.DEVICE_PULL))
            h2d.set_page_size(m_max * RB)
            d2h = pinned(lambda: ttnn.D2HSocket(dev, where, 2 * self.page_bytes))
            d2h.set_page_size(self.page_bytes)
            link = (h2d, d2h)
        self.h2d, self.d2h = link
        wa = torch.cat([r["q_a"].T, r["kv_a_mqa"].T, r["i_proj"].T / math.sqrt(IHD * IH), r["i_attn_k"].T,
                        r["i_gate"].T], dim=1)                                          # [4096, 2336]
        self.wa, self.waq = stream(dev, wa, check)
        wb = r["q_a_norm"][:, None] * torch.cat([r["q_b"].T, r["i_attn_q_b"].T], dim=1)   # [1536, 20480]
        self.wb, self.wbq = stream(dev, wb, check)
        kbs = [stream(dev, r["k_b"][h].T.contiguous()) for h in range(NH)]               # [256 k, 512 n] each
        self.kbq = torch.stack([q for _, q in kbs]) if check else None
        self.kb = ttnn.from_torch(torch.cat([ttnn.to_torch(t).reshape(-1, 32) for t, _ in kbs]).reshape(1, 1, -1, 32),
                                  dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        for t, _ in kbs:
            ttnn.deallocate(t)
        self.K = K
        nan = lambda n, dt: ttnn.from_torch(torch.full((1, 1, 1, n), float("nan")), dtype=dt,  # noqa: E731
                                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.check_a, self.check_b = nan(NA * 32 * m_max, ttnn.float32), nan(NB * 32 * m_max, ttnn.bfloat16)
        g = self.grid = Grid(dev)
        speeds = g.speeds(NOC1_ROWS)
        sa, sb = split_weighted(NA, speeds), split_weighted(NB, speeds)
        self.sa = sa
        sp = [(0, 0)] * NH + split(n_pool // 32, len(g) - NH)
        self.rx, self.r1 = 0, m_max * KVL * 16
        self.r2 = self.r1 + (7104 if m_max == 1 else NQA * m_max * 80 + (NA - NQA) * m_max * 128)
        self.iqt = self.r2          # M rows: iq's heads x dims tiles per row (4 m_max x 2 KB) just before R2: barrier 2
        if m_max > 1:               # broadcasts both
            self.r2 += 4 * m_max * 2048
        self.sc = self.r2 + (NB * 64 * m_max)
        self.reply = self.sc + self.page_bytes
        # shortcut: SC holds M_MAX row pages (4 x 19.4 KB at 8K context); past ~32K context they outgrow L1 at 4 rows,
        # then send the row pages one at a time
        arena = self.reply + m_max * RB + (64 if m_max > 1 else 0)
        rargs, wargs, crt = {}, {}, ttnn.RuntimeArgs()
        for idx, (cx, cy) in enumerate(g.cores):
            head = idx if idx < NH else IDLE
            (a0, na), (b0, nb), (p0, np_) = sa[idx], sb[idx], sp[idx]
            rargs[(cx, cy)] = g.args(idx) + [hn.buffer_address(), self.wa.buffer_address(), self.wb.buffer_address(),
                                             self.kb.buffer_address(), K.buffer_address(), sel.buffer_address(),
                                             cpos.buffer_address(), lat.buffer_address(), a0, na, b0, nb, head, p0, np_, self.rx, self.r1, self.r2, self.sc,
                                             self.reply] + self.pf_args(idx, pf) + (
                [step.buffer_address(), self.iqt] if m_max > 1 else [])
            wargs[(cx, cy)] = g.args(idx) + [a0, na, b0, nb, head, p0, np_, self.check_a.buffer_address(),
                                             self.check_b.buffer_address(), qg[0].buffer_address(),
                                             qg[1].buffer_address(), self.r1, self.r2, self.sc] + (
                [step.buffer_address(), self.row_bytes, n_pool * 4, self.iqt] if m_max > 1 else [])
            crt[cx][cy] = [na, nb, head, np_]
        acc = list(ttnn.TensorAccessorArgs(self.wa).get_compile_time_args())
        defines = [("H2D_CONFIG", str(self.h2d.get_config_buffer_address())),
                   ("D2H_CONFIG", str(self.d2h.get_config_buffer_address()))] + (
            [("M_MAX", str(m_max))] if m_max > 1 else [])
        kernels = g.dm_kernels(os.path.join(KDIR, "glm_fmq_reader.cpp"), os.path.join(KDIR, "glm_fmq_writer.cpp"),
                               [BATCH, DEPTH, n_pool] + acc, acc, rargs, wargs, defines=defines)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        mode[2] = ttnn.UnpackToDestMode.UnpackToDestFp32                         # r, exact
        cfg.unpack_to_dest_mode = mode
        kernels.append(ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "glm_fmq_compute.cpp"),
                                             core_ranges=g.ranges, compile_time_args=[BATCH], runtime_args=crt,
                                             config=cfg, defines=[("M_MAX", str(m_max))] if m_max > 1 else []))
        f32, bf, b8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
        spec = [(0, b8, W_PAGE, BATCH * DEPTH), (1, bf, 2048, 2 * BATCH), (2, f32, 4096, 1), (3, bf, 2048, 8),
                (4, bf, 2048, 4 * m_max), (5, bf, 2048, m_max), (6, f32, 4096, 1), (7, f32, 4096, 1),
                (8, bf, 2048, 4 * m_max), (16, f32, 4096, 2), (17, bf, 2048, 2), (30, ttnn.uint8, arena, 1),
                (31, ttnn.uint8, 1024 if m_max == 1 else 6144, 1)]
        self.cb_kb = sum(s * n for _, _, s, n in spec) / 1024
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=[g.cb(*x) for x in spec])
        self.ios = [hn, self.wa, self.wb, self.kb, K, sel, cpos, lat, self.check_a, self.check_b, qg[0], qg[1]] + \
            ([pf[0]] if pf is not None else []) + ([step] if m_max > 1 else [])
        self.page = torch.zeros(self.page_bytes, dtype=torch.uint8)
        self.reply_t = torch.zeros(m_max * RB, dtype=torch.uint8)
        self.page_np, self.reply_np = self.page.numpy(), self.reply_t.numpy()
        self.pools = np.zeros(NSEL, np.uint32)

    def __call__(self):
        ttnn.generic_op(self.ios, self.prog)

    @property
    def a_src(self):
        return self.wa

    def a_ranges(self, idx: int, n: int) -> list[tuple[int, int]]:
        """core idx's first n pages of phase A's weights (tiles a0 .. a0 + na, KA pages each)"""
        a0, na = self.sa[idx]
        return [(a0 * 128, min(n, na * 128))] if na and n > 0 else []

    def pf_args(self, idx: int, pf) -> list[int]:
        if pf is None:
            return [0, 0]
        return [pf[0].buffer_address(), sum(c for _, c in self.a_ranges(idx, pf[1]))]

    def serve(self):
        """The host's side (glm_loop.cpp's mla_step): the scores and this token's indexer inputs in; the selection,
        cur_pos and the caches' new rows out."""
        self.d2h.read_tensor(self.page)
        if self.m_max > 1:   # the pass's M rows (glm_loop.cpp's mla_rows); the last row's pools kept for the checks
            if _LIB.gl_mla_rows(self.state, self.page_np.ctypes.data, self.n_pool, self.M,
                                self.reply_np.ctypes.data) < 0:
                raise RuntimeError("gl_mla_rows: past the cache's capacity")
            self.pools = last_pools()
            self.h2d.write_tensor(self.reply_t)
            return
        if _LIB.gl_mla_step(self.state, self.page_np.ctypes.data, self.n_pool, self.reply_np.ctypes.data) < 0:
            raise RuntimeError("gl_mla_step: past the cache's capacity")
        self.pools = last_pools()
        self.h2d.write_tensor(self.reply_t)

    def errors(self, hn_host, K_host) -> dict:
        """Each phase against float64 on the card's weights from the card's own previous outputs."""
        rel = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
        a = ttnn.to_torch(self.check_a).double().flatten()
        res = {"A": rel(a, hn_host.double() @ self.waq)}
        qa = a[:QL]
        r = 1 / math.sqrt(float(qa.pow(2).mean()) + RMS_EPS)
        yb = r * (qa.to(torch.bfloat16).double() @ self.wbq)
        b = ttnn.to_torch(self.check_b).double().flatten()
        res["B"] = rel(b, yb)
        q = b[:NH * HK].reshape(NH, HK)
        q_abs = torch.einsum("hk,hkn->hn", q, self.kbq)
        got = torch.cat([ttnn.to_torch(t).double().reshape(32, KVL) for t in self.qg])
        res["C_absorb"] = rel(got, q_abs)
        iq = b[NH * HK:].reshape(IH, IHD)
        w = a[64 * 32:65 * 32].to(torch.bfloat16).double()
        s_ref = (torch.relu(K_host.double() @ iq.T) * w).sum(1)
        s = torch.from_numpy(self.page_np.view(np.float32)[:self.n_pool].copy()).double()
        res["C_scores"] = rel(s, s_ref)
        res["top512_agree"] = len(set(self.pools.tolist()) & set(torch.topk(s_ref, NSEL).indices.tolist()))
        a2 = torch.from_numpy(self.page_np.view(np.float32)[self.n_pool:self.n_pool + A2_BYTES // 4].copy()).double()
        res["page_a2"] = rel(a2, a[NQA * 32:NA * 32])
        return res


    def errors_m(self, hn_rows, K_host) -> dict:
        """M rows: each phase of each row against float64 from the card's own previous outputs; the stateless
        selection (state -1) of each row against float64's top 512 of that row's scores."""
        rel = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
        M, mp = self.M, mp_of(self.M)
        a = unblocks(ttnn.to_torch(self.check_a).double().flatten(), NA, mp)[:M]
        b = unblocks(ttnn.to_torch(self.check_b).double().flatten(), NB, mp)[:M]
        res = {"A": rel(a, hn_rows.double() @ self.waq)}
        qa = a[:, :QL]
        r = 1 / torch.sqrt(qa.pow(2).mean(1, keepdim=True) + RMS_EPS)
        res["B"] = rel(b, r * (qa.to(torch.bfloat16).double() @ self.wbq))
        q = b[:, :NH * HK].reshape(M, NH, HK)
        q_abs = torch.einsum("mhk,hkn->mhn", q, self.kbq)
        got = torch.cat([ttnn.to_torch(t).double().reshape(self.m_max, 32, KVL)[:M] for t in self.qg], dim=1)
        res["C_absorb"] = rel(got, q_abs)
        pg = self.page_np.view(np.uint8)
        worst_s, agree, a2e, iqe, sel_ok = 0.0, [], 0.0, 0.0, True
        sel = ttnn.to_torch(self.sel).reshape(self.m_max, NS)[:M].long()
        for m in range(M):
            row = pg[m * self.row_bytes:(m + 1) * self.row_bytes]
            iq = b[m, NH * HK:].reshape(IH, IHD)
            w = a[m, 64 * 32:65 * 32].to(torch.bfloat16).double()
            s_ref = (torch.relu(K_host.double() @ iq.T) * w).sum(1)
            sc = torch.from_numpy(row[:self.n_pool * 4].view(np.float32).copy()).double()
            worst_s = max(worst_s, rel(sc, s_ref))
            pools = (sel[m, :ITOP].reshape(NSEL, KPOOL)[:, 0] // KPOOL).tolist()
            agree.append(len(set(pools) & set(torch.topk(s_ref, NSEL).indices.tolist())))
            sel_ok &= bool(torch.equal(sel[m, :ITOP].reshape(NSEL, KPOOL)[:, 1:] - sel[m, :ITOP].reshape(NSEL, KPOOL)[:, :1],
                                       torch.arange(1, KPOOL).expand(NSEL, -1)))
            a2 = torch.from_numpy(row[self.n_pool * 4:self.n_pool * 4 + A2_BYTES].view(np.float32).copy()).double()
            a2e = max(a2e, rel(a2, a[m, NQA * 32:NA * 32]))
            iqr = torch.from_numpy(row[self.n_pool * 4 + A2_BYTES:].view(np.uint16).astype(np.int32) << 16)
            iqe = max(iqe, rel(iqr.view(torch.float32).double(), b[m, NH * HK:].to(torch.bfloat16).double()))
        res.update({"C_scores": worst_s, "top512_agree": agree, "page_a2": a2e, "page_iq": iqe, "sel_pools_ok": sel_ok,
                    "cur_pos": ttnn.to_torch(self.cpos_t).flatten()[:M].tolist()})
        return res


def main_m() -> None:
    """The query side at M rows (FMQ_M = M_MAX), the stateless selection: every row's phases and selection."""
    g = Gguf(MODEL)
    p = f"blk.{MLA_LAYER}."
    r = {x: g(p + f"attn_{x}.weight") for x in ("q_a", "q_b", "kv_a_mqa", "k_b", "q_a_norm")}
    for x in ("attn_k", "attn_q_b", "proj"):
        r["i_" + x] = g(p + f"indexer.{x}.weight")
    r["i_gate"] = g(p + "indexer_compressor_gate.weight")
    n_pool = CTX // KPOOL
    mm = M_MAX
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": MLA_LAYER, "ctx": CTX, "m_max": mm}
    try:
        rng = np.random.default_rng(int(time.time()))
        K_h = torch.from_numpy(rng.standard_normal((n_pool, IHD)).astype(np.float32)).to(torch.bfloat16)
        put = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=dev)  # noqa: E731
        RMj = ttnn.ROW_MAJOR_LAYOUT
        hn = put(torch.zeros(1, 1, 1, mm * D, dtype=torch.bfloat16), ttnn.bfloat16, RMj)
        K = put(K_h.reshape(1, 1, n_pool, IHD), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        qg = [put(torch.full((1, mm, 32, KVL), float("nan")), ttnn.bfloat16, ttnn.TILE_LAYOUT) for _ in range(2)]
        sel = put(torch.zeros(mm, NS, dtype=torch.int32), ttnn.uint32, RMj)
        cpos = put(torch.zeros(mm, dtype=torch.int32), ttnn.int32, RMj)
        lat = put(torch.zeros(1, 1, CTX, KVL), ttnn.bfloat16, RMj)
        step = put(step_words(1), ttnn.uint32, RMj)
        m = MlaQuery(dev, r, K, n_pool, hn, qg, sel, cpos, lat, m_max=mm, step=step)
        m.cpos_t = cpos
        res["cb_kb"] = round(m.cb_kb, 1)
        res["rows"] = {}
        for M in range(1, mm + 1):
            hn_rows = torch.from_numpy(rng.standard_normal((M, D)).astype(np.float32)).to(torch.bfloat16)
            full = torch.zeros(mm * D, dtype=torch.bfloat16)
            flat = blocks(pad_rows(hn_rows), mp_of(M))
            full[:flat.numel()] = flat
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(full.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                                                            layout=RMj), hn)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(step_words(M), dtype=ttnn.uint32, layout=RMj), step)
            m.M = M
            m()
            m.serve()
            ttnn.synchronize_device(dev)
            e = m.errors_m(hn_rows, K_h)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            m()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            for n, key in ((5, None), (REPS, "traced_us")):
                ttnn.synchronize_device(dev)
                t = time.perf_counter()
                for _ in range(n):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                    m.serve()
                ttnn.synchronize_device(dev)
                if key:
                    e[key] = round((time.perf_counter() - t) / n * 1e6, 1)
            ttnn.release_trace(dev, tid)
            res["rows"][M] = e
            print(f"M {M}:", json.dumps(e), flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, f"g3_fmq_m{mm}_ctx{CTX}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


M_MAX = int(os.environ.get("FMQ_M", 1))


def main() -> None:
    if M_MAX > 1:
        return main_m()
    g = Gguf(MODEL)
    p = f"blk.{MLA_LAYER}."
    r = {x: g(p + f"attn_{x}.weight") for x in ("q_a", "q_b", "kv_a_mqa", "k_b", "q_a_norm")}
    for x in ("attn_k", "attn_q_b", "proj"):
        r["i_" + x] = g(p + f"indexer.{x}.weight")
    r["i_gate"] = g(p + "indexer_compressor_gate.weight")
    n_pool = CTX // KPOOL
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": MLA_LAYER, "ctx": CTX}
    try:
        rng = np.random.default_rng(int(time.time()))
        hn_h = torch.from_numpy(rng.standard_normal(D).astype(np.float32)).to(torch.bfloat16)
        K_h = torch.from_numpy(rng.standard_normal((n_pool, IHD)).astype(np.float32)).to(torch.bfloat16)
        put = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=dev)  # noqa: E731
        hn = put(hn_h.reshape(1, 1, 1, D), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        K = put(K_h.reshape(1, 1, n_pool, IHD), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        qg = [put(torch.full((1, 1, 32, KVL), float("nan")), ttnn.bfloat16, ttnn.TILE_LAYOUT) for _ in range(2)]
        sel = put(torch.zeros(1, NS, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        cpos = put(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        lat = put(torch.zeros(1, 1, CTX, KVL), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        m = MlaQuery(dev, r, K, n_pool, hn, qg, sel, cpos, lat)
        m()
        m.serve()
        ttnn.synchronize_device(dev)
        res["check"] = m.errors(hn_h, K_h)
        # q_abs end to end against G2's float32 formulas
        h = hn_h.float()
        qr = r["q_a"] @ h
        qr = qr * torch.rsqrt(qr.pow(2).mean() + RMS_EPS) * r["q_a_norm"]
        q_abs = torch.einsum("hlk,hk->hl", r["k_b"], (r["q_b"] @ qr).reshape(NH, HK))
        got = torch.cat([ttnn.to_torch(t).float().reshape(32, KVL) for t in qg])
        res["check"]["q_abs_pcc_vs_f32"] = round(pcc(got.double(), q_abs.double()), 6)
        sel_c = ttnn.to_torch(sel).flatten().long()
        res["check"]["sel_ids_ok"] = bool(torch.equal(sel_c[:ITOP].reshape(NSEL, KPOOL)[:, 0] // KPOOL,
                                                      torch.from_numpy(m.pools.astype(np.int64))))
        res["check"]["cur_pos"] = int(ttnn.to_torch(cpos).flatten()[0])
        print("check:", json.dumps(res["check"]), f"(circular buffers {m.cb_kb:.0f} KB a core)", flush=True)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        m()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        for n, key in ((5, None), (REPS, "traced_us")):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                m.serve()
            ttnn.synchronize_device(dev)
            if key:
                res[key] = round((time.perf_counter() - t) / n * 1e6, 1)
        ttnn.release_trace(dev, tid)
        mb = (m.wa.volume() + m.wb.volume() + m.kb.volume()) * 1.0625 / 1e6
        print(f"ctx {CTX}: traced {res['traced_us']} us with the host's top 512 ({mb:.1f} MB of weights; the stock "
              f"route ~690 us)", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, f"g3_fmq_ctx{CTX}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
