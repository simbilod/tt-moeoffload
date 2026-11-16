"""One mHC site of GLM on the card (tt/kernels/glm_hc_*.cpp), the parametrization on the
host over the sockets (the user's call: the host route for now; the on-card one is a Future investigations row).

The residual streams as fp32 stream tiles (column tile t of D = 4096: stream s in row s), 2 column tiles a core on
the first NC = 64 cores. In one program: A, each core's partial of the 24 mixes (fn's bf16 k-tiles over its own
slices) and of the sum of squares; barrier 1, at which the coordinator sends the 64 partials to the host (8,448 B)
and multicasts the 24 parameters the host sends back; B, h = PRE @ X_t; C, the new streams C' @ X'_t (X' = X with
the sublayer's output in row 4). Real weights: blk.4's attn site (fn, scale, base); random streams and output.

Checked against G2's hc_ref / hc_post_ref (llama.cpp's build_hc_pre / post in float32 torch): the parameters, h and
the new streams; the site split in two launches (2d.4a: the pre half, then the post half reading the parameters
back from a DRAM tensor) against the one launch. Timed: the program with its host round trip, untraced (launch included), and the host's part alone.

  python3 tt_moeoffload/launch.py --memory-max $((16 << 30)) --engine tt/glm/g3_hc.py
"""
from __future__ import annotations

import json
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
from g2_layers import D, HC, HC_EPS, MODEL, OUT, RMS_EPS, SINKHORN, Gguf, hc_post_ref, hc_ref, pcc  # noqa: E402
assert os.path.dirname(sys.modules["g2_layers"].__file__) == HERE
from fused.grid import Grid  # noqa: E402
from glm_loop import GlmLoop  # noqa: E402
from fused.mrows import blocks, mp_of, pad_rows, step_words, unblocks  # noqa: E402
from u1_sockets import pinned  # noqa: E402
import ctypes  # noqa: E402

_LIB = ctypes.CDLL(os.path.join(TT, "cpulane", "build", "libcpulane_glm.so"))   # cl_hc_params (tt/cpulane/hc.cpp)
_LIB.cl_hc_params.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p]

NC, TPC = 64, 2                                    # working cores, column tiles a core
NT = D // 32                                       # 128 column tiles
PAGE_OUT = NC * 128 + NC * 16                      # the partials: mixes [NC][32] fp32, then a 16-byte slot a square
PAGE_IN = 128                                      # pre [4], post [4], C [dst][src] [16] fp32, padded
REPS = int(os.environ.get("HC_REPS", 50))


def host_params(page: np.ndarray, scale: np.ndarray, base: np.ndarray) -> np.ndarray:
    """The 24 parameters from the cores' partials (float64, llama.cpp's formulas), as the card's 128-byte page."""
    f = page.view(np.float32)
    mixes = f[:NC * 32].reshape(NC, 32)[:, :24].astype(np.float64).sum(0)
    ss = f[NC * 32::4][:NC].astype(np.float64).sum()
    r = 1 / np.sqrt(ss / (HC * D) + RMS_EPS)
    a = r * mixes * np.concatenate([np.repeat(scale[0], 4), np.repeat(scale[1], 4), np.repeat(scale[2], 16)]) + base
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    pre, post = sig(a[:4]) + HC_EPS, 2 * sig(a[4:8])
    m = a[8:24].reshape(HC, HC)                                       # [src][dst]
    m = np.exp(m - m.max(1, keepdims=True))
    m = m / m.sum(1, keepdims=True) + HC_EPS
    m = m / (m.sum(0, keepdims=True) + HC_EPS)
    for _ in range(SINKHORN - 1):
        m = m / (m.sum(1, keepdims=True) + HC_EPS)
        m = m / (m.sum(0, keepdims=True) + HC_EPS)
    out = np.zeros(32, np.float32)
    out[:4], out[4:8], out[8:24] = pre, post, m.T.reshape(-1)          # C [dst][src]
    return out


class HcSite:
    """One mHC site's programs on the card (glm_hc_*.cpp) and its host side. Tensors: x (the streams, fp32 tiles
    [32, D], stream s in row s), out (the sublayer's output, row-major fp32 [D]), res (h in columns 0..D of row 0,
    the new streams in columns D..2D, rows 0..3), par (the 24 parameters). prog: the whole site (phases 3); pre: A,
    the host, B (phases 1, par written); post: C from par (phases 2), once out holds the sublayer's output. Every
    launch with phase 1 needs serve() on the host meanwhile."""

    def __init__(self, dev, grid, fn, scale, base, h2d, d2h, hn=None, norm=None, xn=None, x=None, m_max=1,
                 step=None):
        """hn, norm: the sublayer's input norm in the pre launch (HC_NORM): hn = h / rms(h) * norm as bf16 into hn
        (row-major bf16 [1, 1, 1, D], e.g. the fused mixer's input tensor). xn: the post's new streams go there (fp32
        tiles [1, 1, 32, D]: the next site's x) instead of res's columns D..2D. x: the streams' tensor, when the
        previous site's post writes it (a token's chain). m_max > 1: the M-row build (glm_hc_reader.cpp):
        row m's streams in rows 4 m + s, out and hn as mp-row blocks, M from step (the step tensor)."""
        self.xn, self.m_max, self.step, self.M = xn, m_max, step, 1
        assert grid.cores[0] == (0, 0), "the coordinator holds the sockets"
        put = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=dev)  # noqa: E731
        w = torch.zeros(HC * D, 32)
        w[:, :24] = fn.T
        self.fn = put(w.reshape(1, 1, HC * D, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.x = x if x is not None else put(torch.zeros(1, 1, 32, D), ttnn.float32, ttnn.TILE_LAYOUT)
        self.out = put(torch.zeros(1, 1, 1, m_max * D), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
        self.res = put(torch.zeros(1, 1, 32, 2 * D), ttnn.float32, ttnn.TILE_LAYOUT)
        self.par = put(torch.zeros(1, 1, 1, 32 * m_max), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
        self.h2d, self.d2h, self.grid = h2d, d2h, grid
        self.sc32, self.bs32 = scale.numpy().astype(np.float32), base.numpy().astype(np.float32)
        self.page = torch.zeros(m_max * PAGE_OUT, dtype=torch.uint8)
        self.reply = torch.zeros(m_max * PAGE_IN, dtype=torch.uint8)
        self.page_np, self.reply_np = self.page.numpy(), self.reply.numpy()
        self.hn = hn
        self.nw = None if norm is None else put(norm.float().reshape(1, 1, 1, D), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
        self.ios = [self.x, self.fn, self.out, self.res, self.par] + ([hn, self.nw] if hn is not None else []) + \
            ([xn] if xn is not None else []) + ([step] if m_max > 1 else [])
        self.prog, self.pre_prog, self.post_prog = (self.program(p) for p in (3, 1, 2))

    def program(self, phases):
        grid = self.grid
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        defines = [("H2D_CONFIG", str(self.h2d.get_config_buffer_address())),
                   ("D2H_CONFIG", str(self.d2h.get_config_buffer_address()))]
        norm = self.hn is not None
        if norm:
            defines.append(("HC_NORM", "1"))
        if self.m_max > 1:
            defines.append(("M_MAX", str(self.m_max)))
        mrow = [self.step.buffer_address()] if self.m_max > 1 else []
        extra = [self.hn.buffer_address(), self.nw.buffer_address()] if norm else []
        rargs, wargs, crt = {}, {}, ttnn.RuntimeArgs()
        for idx, (cx, cy) in enumerate(grid.cores):
            t0, nt = (idx * TPC, TPC) if idx < NC else (0, 0)
            rargs[(cx, cy)] = grid.args(idx) + [self.x.buffer_address(), self.fn.buffer_address(),
                                                self.out.buffer_address(), t0, nt, phases,
                                                self.par.buffer_address()] + mrow
            wargs[(cx, cy)] = grid.args(idx) + [self.res.buffer_address(), idx, t0, nt, phases,
                                                self.xn.buffer_address() if self.xn is not None else 0] + extra + mrow
            crt[cx][cy] = [nt, phases]
        kernels = grid.dm_kernels(os.path.join(KDIR, "glm_hc_reader.cpp"), os.path.join(KDIR, "glm_hc_writer.cpp"),
                                  [NC] + acc(self.x) + acc(self.fn) + acc(self.out) + acc(self.par),
                                  [NC] + acc(self.res) + (acc(self.hn) + acc(self.nw) if norm else []), rargs, wargs,
                                  defines=defines)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        cfg.unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * 64
        kernels.append(ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "glm_hc_compute.cpp"),
                                             core_ranges=grid.ranges, compile_time_args=[], runtime_args=crt,
                                             config=cfg, defines=[("M_MAX", str(self.m_max))] if self.m_max > 1 else []))
        mr = self.m_max > 1   # M rows: x fp32, the compute's row sums (ones, the squared tiles), cb 6 a tile
        cbs = [grid.cb(0, ttnn.bfloat16, 2048, HC * TPC),
               grid.cb(1, ttnn.float32, 4096, HC * TPC) if mr else grid.cb(1, ttnn.bfloat16, 2048, HC * TPC),
               grid.cb(3, ttnn.float32, 128, TPC * self.m_max), grid.cb(4, ttnn.float32, 4096, TPC),
               grid.cb(5, ttnn.float32, 4096, 2), grid.cb(6, ttnn.float32, 4096 if mr else 64, 1),
               grid.cb(8, ttnn.float32, self.m_max * (PAGE_OUT + PAGE_IN + NC * 16) + 64, 1),
               grid.cb(9, ttnn.uint8, 1024 if self.m_max == 1 else 4096, 1),
               grid.cb(16, ttnn.float32, 4096, 2)] + ([grid.cb(10, ttnn.bfloat16, 2048, 1),
                                                         grid.cb(11, ttnn.float32, 4096, HC * TPC)] if mr else [])
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=grid.semaphores(), cbs=cbs)

    def run(self, prog):
        ttnn.generic_op(self.ios, prog)

    def params(self):
        """the host's parameters, per row at M rows (row m's page at m PAGE_OUT, its reply at m PAGE_IN)"""
        for m in range(self.M if self.m_max > 1 else 1):
            pg = ctypes.c_void_p(self.page_np.ctypes.data + m * PAGE_OUT)
            rp = ctypes.c_void_p(self.reply_np.ctypes.data + m * PAGE_IN)
            _LIB.cl_hc_params(pg, NC, D, SINKHORN, self.sc32.ctypes.data_as(ctypes.c_void_p),
                              self.bs32.ctypes.data_as(ctypes.c_void_p), rp)

    def serve(self):
        """The host's side of a phase-1 launch: the partials in, the parameters out (M rows: M pages each way)."""
        self.d2h.read_tensor(self.page)
        self.params()
        self.h2d.write_tensor(self.reply)


def sockets(dev, m_max=1):
    """The mHC sites' socket pair on core (0, 0), allocated before any trace capture (pages of m_max rows)."""
    where = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
    h2d = pinned(lambda: ttnn.H2DSocket(dev, where, ttnn.BufferType.L1, 4 * m_max * PAGE_IN, ttnn.H2DMode.DEVICE_PULL))
    h2d.set_page_size(m_max * PAGE_IN)
    d2h = pinned(lambda: ttnn.D2HSocket(dev, where, 4 * m_max * PAGE_OUT))
    d2h.set_page_size(m_max * PAGE_OUT)
    return h2d, d2h


def main_m() -> None:
    """One mHC site at M rows (HC_M = M_MAX), every row against G2's float32 hc_ref / hc_post_ref."""
    g = Gguf(MODEL)
    fn, scale, base = (g(f"blk.4.hc_attn_{n}.weight") for n in ("fn", "scale", "base"))
    nw = g("blk.4.attn_norm.weight")
    mm, RMj, TL = M_MAX, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "site": "blk.4 attn", "m_max": mm, "rows": {}}
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
    try:
        h2d, d2h = sockets(dev, mm)
        grid = Grid(dev)
        step = ttnn.from_torch(step_words(1), dtype=ttnn.uint32, layout=RMj, device=dev)
        hn_dev = ttnn.from_torch(torch.zeros(1, 1, 1, mm * D), dtype=ttnn.bfloat16, layout=RMj, device=dev)
        hs = HcSite(dev, grid, fn, scale, base, h2d, d2h, hn=hn_dev, norm=nw, m_max=mm, step=step)
        gen = torch.Generator().manual_seed(int(time.time()))
        for M in range(1, mm + 1):
            mp = mp_of(M)
            X = torch.randn(M, HC, D, generator=gen)
            out = torch.randn(M, D, generator=gen)
            Xt = torch.zeros(32, D)
            for m in range(mp):
                Xt[4 * m:4 * m + 4] = X[min(m, M - 1)]
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(Xt.reshape(1, 1, 32, D), dtype=ttnn.float32, layout=TL),
                                            hs.x)
            ob = torch.zeros(mm * D)
            fl = blocks(pad_rows(out), mp)
            ob[:fl.numel()] = fl
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(ob.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=RMj),
                                            hs.out)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(step_words(M), dtype=ttnn.uint32, layout=RMj), step)
            hs.M = M
            hs.run(hs.prog)
            hs.serve()
            ttnn.synchronize_device(dev)
            got = ttnn.to_torch(hs.res).float().reshape(32, 2 * D)
            hn_c = unblocks(ttnn.to_torch(hn_dev).double().flatten(), D // 32, mp)
            p = hs.reply.numpy().view(np.float32).reshape(mm, 32)
            e = {"pre": 0.0, "post": 0.0, "comb": 0.0, "h": 0.0, "streams": 0.0, "hn": 0.0}
            for m in range(M):
                h_ref, post_ref, comb_ref = hc_ref(X[m], fn, scale, base)
                y_ref = hc_post_ref(out[m], X[m], post_ref, comb_ref)
                e["post"] = max(e["post"], float((torch.from_numpy(p[m, 4:8].copy()) - post_ref).abs().max()))
                e["comb"] = max(e["comb"], float((torch.from_numpy(p[m, 8:24].copy()).reshape(HC, HC)
                                                  - comb_ref).abs().max()))
                e["h"] = max(e["h"], rel(got[m, :D], h_ref))
                e["streams"] = max(e["streams"], rel(got[4 * m:4 * m + 4, D:], y_ref))
                hn_ref = h_ref.double() * torch.rsqrt(h_ref.double().pow(2).mean() + RMS_EPS) * nw.double()
                e["hn"] = max(e["hn"], rel(hn_c[m], hn_ref))
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            hs.run(hs.prog)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            for n, key in ((5, None), (REPS, "traced_us")):
                ttnn.synchronize_device(dev)
                t = time.perf_counter()
                for _ in range(n):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                    hs.serve()
                ttnn.synchronize_device(dev)
                if key:
                    e[key] = round((time.perf_counter() - t) / n * 1e6, 1)
            ttnn.release_trace(dev, tid)
            # as the token serves it: the C++ loop (glm_loop.cpp's HC exchange, cl_hc_params per row); the whole site,
            # its pre half (A, the host, B and the norm) and its post half (C) apart
            loop = GlmLoop(_LIB)
            loop.hc(hs)
            assert _LIB.gl_rows(M) == 0
            for name, prog, serve in (("site_cpp_us", hs.prog, True), ("pre_cpp_us", hs.pre_prog, True),
                                      ("post_us", hs.post_prog, False)):
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                hs.run(prog)
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                for n, key in ((5, None), (REPS, name)):
                    ttnn.synchronize_device(dev)
                    t = time.perf_counter()
                    for _ in range(n):
                        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                        if serve:
                            loop.run()
                    ttnn.synchronize_device(dev)
                    if key:
                        e[key] = round((time.perf_counter() - t) / n * 1e6, 1)
                ttnn.release_trace(dev, tid)
            loop.clear()
            res["rows"][M] = e
            print(f"M {M}:", json.dumps(e), flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    json.dump(res, open(os.path.join(OUT, f"g3_hc_m{mm}.json"), "w"), indent=1)


M_MAX = int(os.environ.get("HC_M", 1))


def main() -> None:
    if M_MAX > 1:
        return main_m()
    g = Gguf(MODEL)
    fn, scale, base = (g(f"blk.4.hc_attn_{n}.weight") for n in ("fn", "scale", "base"))   # fn [24, 16384]
    gen = torch.Generator().manual_seed(8)
    X = torch.randn(HC, D, generator=gen)
    out = torch.randn(D, generator=gen)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "site": "blk.4 attn"}
    try:
        h2d, d2h = sockets(dev)
        grid = Grid(dev)
        nw = g("blk.4.attn_norm.weight")
        hn_dev = ttnn.from_torch(torch.zeros(1, 1, 1, D), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=dev)
        hs = HcSite(dev, grid, fn, scale, base, h2d, d2h, hn=hn_dev, norm=nw)
        Xt = torch.zeros(32, D)
        Xt[:HC] = X
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(Xt.reshape(1, 1, 32, D), dtype=ttnn.float32,
                                                        layout=ttnn.TILE_LAYOUT), hs.x)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(out.reshape(1, 1, 1, D), dtype=ttnn.float32,
                                                        layout=ttnn.ROW_MAJOR_LAYOUT), hs.out)
        prog, pre, post, ios, res_dev = hs.prog, hs.pre_prog, hs.post_prog, hs.ios, hs.res
        page, reply, page_np, reply_np = hs.page, hs.reply, hs.page_np, hs.reply_np
        params_c = hs.params
        sc, bs = scale.double().numpy(), base.double().numpy()

        def site():
            ttnn.generic_op(ios, prog)
            t = time.perf_counter()
            d2h.read_tensor(page)
            t_in = time.perf_counter()
            params_c()
            h2d.write_tensor(reply)
            t_out = time.perf_counter()
            ttnn.synchronize_device(dev)
            return t_in - t, t_out - t_in

        site()
        got = ttnn.to_torch(res_dev).float().reshape(32, 2 * D)
        h_card, y_card = got[0, :D], got[:HC, D:]
        h_ref, post_ref, comb_ref = hc_ref(X, fn, scale, base)
        y_ref = hc_post_ref(out, X, post_ref, comb_ref)
        p = reply.numpy().view(np.float32)
        rel = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
        flat = X.flatten()
        mixes = fn @ (flat * torch.rsqrt(flat.pow(2).mean() + RMS_EPS))
        pre_ref = torch.sigmoid(mixes[:4] * scale[0] + base[:4]) + HC_EPS
        res["check"] = {
            "pre_max_abs_err": float((torch.from_numpy(p[:4].copy()) - pre_ref).abs().max()),
            "post_max_abs_err": float((torch.from_numpy(p[4:8].copy()) - post_ref).abs().max()),
            "comb_max_abs_err": float((torch.from_numpy(p[8:24].copy()).reshape(HC, HC) - comb_ref).abs().max()),
            "h_pcc": round(pcc(h_card, h_ref), 7), "h_max_rel_err": rel(h_card, h_ref),
            "streams_pcc": round(pcc(y_card, y_ref), 7), "streams_max_rel_err": rel(y_card, y_ref),
        }
        res["check"]["host_c_vs_numpy_max_abs"] = float(np.abs(reply_np.view(np.float32)[:24] -
                                                              host_params(page_np, sc, bs)[:24]).max())
        print("check:", json.dumps(res["check"]), flush=True)
        f = page.numpy().view(np.float32)
        part = torch.from_numpy(f[:NC * 32].reshape(NC, 32).copy())
        print("diag: card mixes[:6]", part[:, :24].sum(0)[:6].tolist(), "exact", (fn @ flat)[:6].tolist(), flush=True)
        print("diag: card sum sq", float(f[NC * 32::4][:NC].astype(np.float64).sum()), "exact", float(flat.pow(2).sum()),
              flush=True)
        ex = torch.stack([fn[:, (s * NT + t) * 32:(s * NT + t + 1) * 32] @ X[s, t * 32:(t + 1) * 32]
                          for t in range(2) for s in range(HC)]).sum(0)
        print("diag: core 0 partial[:6]", part[0, :6].tolist(), "exact", ex[:6].tolist(), flush=True)
        print("diag: core 0 sq", float(f[NC * 32]), "exact", float(X[:, :64].pow(2).sum()), flush=True)
        walls, host = [], []
        for _ in range(REPS):
            t = time.perf_counter()
            wait_us, host_us = site()
            walls.append((time.perf_counter() - t) * 1e6)
            host.append(host_us * 1e6)
        res["site_us_untraced"] = round(statistics.median(walls), 1)
        # traced: the program captured once (the sockets were allocated before it, see g3_overlap.py's trap 3),
        # replayed without blocking while the host serves each round trip
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        ttnn.generic_op(ios, prog)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        for n, key in ((5, None), (REPS, "site_us_traced")):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                d2h.read_tensor(page)
                params_c()
                h2d.write_tensor(reply)
            ttnn.synchronize_device(dev)
            if key:
                res[key] = round((time.perf_counter() - t) / n * 1e6, 1)
        ttnn.release_trace(dev, tid)
        got2 = ttnn.to_torch(res_dev).float().reshape(32, 2 * D)
        res["check"]["traced_same_as_untraced"] = bool(torch.equal(got2, got))
        # 2d.4a: the site split in two launches (pre: h; post: the streams from the parameters in par) gives the same
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.zeros(1, 1, 32, 2 * D), dtype=ttnn.float32,
                                                        layout=ttnn.TILE_LAYOUT), res_dev)
        ttnn.generic_op(ios, pre)
        d2h.read_tensor(page)
        params_c()
        h2d.write_tensor(reply)
        ttnn.generic_op(ios, post)
        ttnn.synchronize_device(dev)
        got3 = ttnn.to_torch(res_dev).float().reshape(32, 2 * D)
        res["check"]["split_same_as_one"] = bool(torch.equal(got3, got))
        res["check"]["split_max_abs_diff"] = float((got3 - got).abs().max())
        # the input norm (HC_NORM, every launch here): hn against h_ref / rms(h_ref) * w, rounded to bf16
        hn_c = ttnn.to_torch(hn_dev).double().flatten()
        hn_ref = (h_ref.double() * torch.rsqrt(h_ref.double().pow(2).mean() + RMS_EPS) * nw.double())
        res["check"]["hn_max_rel_err"] = float((hn_c - hn_ref).abs().max() / hn_ref.abs().max())
        res["check"]["hn_pcc"] = round(pcc(hn_c, hn_ref), 7)
        print("hn:", res["check"]["hn_max_rel_err"], res["check"]["hn_pcc"], flush=True)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        ttnn.generic_op(ios, pre)
        ttnn.generic_op(ios, post)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        for n, key in ((5, None), (REPS, "split_us_traced")):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                d2h.read_tensor(page)
                params_c()
                h2d.write_tensor(reply)
            ttnn.synchronize_device(dev)
            if key:
                res[key] = round((time.perf_counter() - t) / n * 1e6, 1)
        ttnn.release_trace(dev, tid)
        print(f"split in two launches: identical {res['check']['split_same_as_one']} "
              f"(max diff {res['check']['split_max_abs_diff']}), {res['split_us_traced']} us traced", flush=True)
        res["host_part_us"] = round(statistics.median(host), 1)
        print(f"one site with its host round trip: untraced {res['site_us_untraced']} us, traced "
              f"{res['site_us_traced']} us (the host's params {res['host_part_us']} us); traced result identical: "
              f"{res['check']['traced_same_as_untraced']}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    json.dump(res, open(os.path.join(OUT, "g3_hc.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "g3_hc.json"))


if __name__ == "__main__":
    main()
