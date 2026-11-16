"""G3 step 9: the decode head as one program on all 110 cores (tt/kernels/fhead_*.cpp): the final hyper-connection mix
(the model's output norm: output_hc norm, down, up; no inject) and the LM head (output.weight, Q6_K in the GGUF,
bfp8_b on the card: 2560 -> 248,320, 675 MB), real weights.

Checked: the mix's hn against float64 from the card's partials (as g3_hc.py), the logits against hn (the card's, bf16)
times the GGUF's weights in float64 (so the bfp8_b rounding of the weights is in the error), the top 10 tokens;
then timed (traced, one launch).

  python3 tt_moeoffload/launch.py --memory-max $((48 << 30)) --engine tt/flashnext/g3_head.py

Measured 2026-10-23: the mix within 8.0e-4 of float64, the logits within 7.6e-3 of the GGUF's weights (bfp8_b's
rounding), the top 10 and the argmax the same; 1,779.2 us traced, the LM head ~1,717 us of it (at the 391 GB/s floor).
With rows 0, 1, 9 reading through NOC1 (fused/grid.py) and the LM head's tiles split by bandwidth: 1,462.8 us (1,787.9
on NOC0 alone), the LM head's 675 MB at ~480 GB/s.
2026-10-24: now a Head class (the token loop's, g4_model.py; x32_t / x16_t settable, get_logits()), weights through
g3_hc.stream's cache: built in 9.6 s cold, ~0 s from the cache; 1,465.0 us traced, the logits 6.7e-3 of the GGUF's.
Program(m, logits, ids): m <= 4 rows of x (row 0 hn, rows 1..m-1 hm_t's until the M-row mix), their
logits, and their top 64 on the card (fhead_writer.cpp: a heap per row while the tiles stream, two merge levels after),
read back as 2 KB (get_ids) instead of the 1 MB of logits. Checked here: the top 64 (ids and values, ties to the lowest
id) exactly those of the card's own logits at m = 1 and 4, with and without the logits, on rows hn, random, one-hot
(45 tied values in the top 64) and zero (all tied: ids 0..63); row 0's logits the same bits as the default program's.
(The float64 argmax of x times the GGUF's Q6_K weights can differ on a near-tie: bfp8_b's rounding, not the selection.)
Measured: 1,465.8 us default, 1,459.1 ids only (the sort and merges after the stream cost less than the logits' writes
they replace), 1,496.5 logits and ids, 1,513.1 ids at m = 4; the host's read-back 1,092-1,216 us for the logits, 123-127
for the ids (ttnn.to_torch).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
from fused.grid import NOC1_ROWS, Grid, profile_plumbing, split_weighted, traced_us  # noqa: E402
from g2_layers import MODEL, Gguf  # noqa: E402
from g3_gdn import EPS, nan_tensor  # noqa: E402
from g3_hc import HC, LO, LR, model_tag, stream  # noqa: E402
from g3_layer import WORKER_L1, bf, rel  # noqa: E402

KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
OUT = os.path.join(HERE, "results")
D, VOCAB = 2560, 248320
NT = VOCAB // 32
BATCH, DEPTH = 8, 4
IDLE = 0xFFFFFFFF
EXACT = (3, 6, 9, 11, 12, 13)
M_MAX, TOPK, GROUP = 4, 64, 10     # fhead_writer.cpp's rows and top k; cores per merge group
TOPK_BYTES = (M_MAX + 1 + 16 + 2) * 8 * TOPK    # its cb 17


class Head:
    """The decode head's weights and program: the final hyper-connection mix and the LM head. The token loop
    (g4_model.py) points x32_t, x16_t at the model's X; logits is [m_max x 7760 x 32] fp32 (a page per output tile,
    row r's from page r * 7760); ids_t [m_max x 128] uint32 (per row the top 64's ids, best first, then their
    logits' bits; program(ids=True)). hm_t: rows 1..m-1 of the LM head's x ([m_max - 1 x 2560] bf16; the test's)."""

    def __init__(self, dev, g: Gguf, back=True, m_max=1, x_blocks=False):
        """back: keep the float64 copies of the mix's weights and the GGUF's LM head (fp32) for the checks.
        m_max: the most rows a program() of this head takes (the logits' and the arena's size).
        x_blocks: X in the layer programs' M-row block layout (fhead_reader.cpp X_BLOCKS; m_max 2 or 4), the mix at M
        rows, M from the step tensor step_t (g3_layer.step_words); the checks then hold m_max rows' blocks."""
        assert 1 <= m_max <= M_MAX and (not x_blocks or m_max in (2, 4))
        self.dev, self.grid = dev, Grid(dev)
        tag = model_tag(g)
        self.norm = g("output_hc_norm.weight")

        def wd():
            w = torch.zeros(HC * D, LO)
            w[:, :LR] = (g("output_hc_down.weight") * self.norm[None, :]).T    # no inject: its column stays zero
            return w
        self.wd_t, self.wdq = stream(dev, wd, back, f"{tag}/output_hc_wd")
        self.wu_t, self.wuq = stream(dev, lambda: g("output_hc_up.weight").T, back, f"{tag}/output_hc_wu")
        self.norm_t = rm(dev, self.norm)
        self.wl_t, _ = stream(dev, lambda: g("output.weight").T, False, f"{tag}/output")   # [2560, 248320], 675 MB
        self.wout = g("output.weight") if back else None                        # [248320, 2560]
        self.x32_t = self.x16_t = self.step_t = None
        mc = m_max if x_blocks else 1
        self.chk = dict(p1=nan_tensor(dev, (1, 1, 1, mc * HC * LO)), hn=nan_tensor(dev, (1, 1, 1, mc * D)),
                        w=nan_tensor(dev, (1, 1, 1, mc * 16)))
        self.m_max, self.x_blocks = m_max, x_blocks
        self.logits = nan_tensor(dev, (1, 1, NT * m_max, 32))
        self.ids_t = ttnn.from_torch(torch.full((1, 1, m_max, 2 * TOPK), -1, dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.hm_t = None
        self.prof = ttnn.from_torch(torch.zeros(1, 1, 1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.gprof, self.gprof_n = None, 32  # grid.h's per-core clocks (fused/grid.py profile_tensor; g4_phases.py)
        o, a = {}, 0
        # x_blocks: the owner layout (fattn1_reader.cpp M rows): x32 / x16 an H1 core's stream, rs the slot of xown,
        # p1 then the r_s, lo then w, by M
        regions = (("x32", m_max * D * 4), ("x16", m_max * D * 2), ("p1", m_max * (HC * LO * 4 + 256)),
                   ("rs", m_max * HC * 128), ("hn", m_max * D * 2), ("lo", m_max * 704), ("w", 0)) if x_blocks else (
            ("x32", HC * D * 4), ("x16", HC * D * 2), ("p1", HC * LO * 4), ("rs", 64), ("hn", D * 2), ("lo", 640),
            ("w", 64)) + ((("hm", (m_max - 1) * D * 2),) if m_max > 1 else ())
        for name, size in regions:
            o[name] = a
            a += size
        self.off, self.arena = o, a

    def program(self, noc1_rows=NOC1_ROWS, m=1, logits=True, ids=False):
        """The head's program, the rows noc1_rows reading through NOC1 (fused/grid.py): the mix's phases on the NOC0
        rows first, the LM head's tiles split by each core's share of card memory bandwidth. m rows of x (row 0 hn,
        rows 1..m-1 hm_t's), their logits (logits) and their top 64 on the card (ids); the defaults are the token
        loop's program as it was before m and ids (fhead_writer.cpp without HEAD_M).
        With x_blocks the rows are X's M rows (the step tensor's M; m unused), always with the top 64 machinery."""
        xb = self.x_blocks
        assert 1 <= m <= self.m_max and (logits or ids) and (m == 1 or self.hm_t is not None or xb)
        assert not xb or self.step_t is not None, "x_blocks: M comes from the step tensor (step_t)"
        ext = (m, logits, ids) != (1, True, False) or xb
        grid, A = self.grid, (lambda t: t.buffer_address())                     # noqa: E731
        offs = [self.off[k] for k in ("x32", "x16", "p1", "rs", "hn", "lo", "w")]
        acc = list(ttnn.TensorAccessorArgs(self.wl_t).get_compile_time_args())
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        for cb in EXACT:
            mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        gp, gdef, gbytes = profile_plumbing(self.gprof, self.gprof_n, 63, 64)  # in cb 63, after the 16 stamps
        f32, bf16, u8 = ttnn.float32, ttnn.bfloat16, ttnn.uint8
        T4, T2 = 4096, 2048
        spec = [(0, ttnn.bfloat8_b, 1088, BATCH * DEPTH), (1, bf16, T2, 80), (2, bf16, T2, 1),
                (3, f32, T4, 3), (4, f32, T4, 3), (5, f32, T4, 1), (6, f32, T4, 1), (7, f32, T4, 2), (8, f32, T4, 1),
                (9, f32, T4, 4), (10, bf16, T2, 10), (11, f32, T4, 4), (12, f32, T4, 4), (13, f32, T4, 4),
                (14, bf16, T2, 1), (15, f32, T4, 1), (16, f32, T4, 2),
                (60, u8, T4, 1), (61, u8, self.arena, 1), (62, u8, 2048 if ext else 1024, 1),
                (63, u8, 64 + gbytes, 1)] + ([(17, u8, TOPK_BYTES, 1)] if ext else [])
        if xb:  # M rows: the ones tile + Sel + the step's words, H1's raw X chunks double-buffered
            spec = [(i, t, pg, {2: 3, 3: 6}.get(i, n)) for i, t, pg, n in spec]
        defs = gdef + ([("HEAD_M", "1")] if ext else []) + ([("X_BLOCKS", str(self.m_max))] if xb else [])
        xy = [n.x | n.y << 16 for n in grid.noc]
        ng = -(-len(grid.cores) // GROUP)
        assert len(grid.cores) - (ng - 1) * GROUP >= M_MAX and ng >= M_MAX  # every group leads every row
        order = grid.order(noc1_rows)
        ls = split_weighted(NT, grid.speeds(noc1_rows))
        h1_cores = [grid.noc[order[k]].x | grid.noc[order[k]].y << 16 for k in range((LO // 32) * HC)]
        rargs, crt, wargs = {}, ttnn.RuntimeArgs(), {}
        for idx, (cx, cy) in enumerate(grid.cores):
            k = order.index(idx)
            h_item = k if k < (LO // 32) * HC else IDLE
            h_tile = k if k <= D // 32 else IDLE
            lf, lc = ls[idx]
            rargs[(cx, cy)] = [A(self.x32_t), A(self.x16_t)] + grid.args(idx) + offs + [
                A(self.wd_t), A(self.wu_t), A(self.norm_t), h_item, h_tile, A(self.wl_t), lf, lc,
                A(self.prof)] + gp(idx)
            crt[cx][cy] = [h_item, h_tile, lc]
            wargs[(cx, cy)] = grid.args(idx) + offs + [A(self.chk["p1"]), A(self.chk["hn"]), A(self.chk["w"]),
                                                       A(self.logits), h_item, h_tile, lf, lc] + gp(idx)
            if xb:   # the step tensor in m's place; H1's 44 cores by item (NOC x | y << 16)
                rargs[(cx, cy)] += [A(self.step_t), 0, 0] + h1_cores
                wargs_m = A(self.step_t)
            elif ext:
                rargs[(cx, cy)] += [m, self.off.get("hm", 0), 0 if self.hm_t is None else A(self.hm_t)]
                wargs_m = m
            if ext:
                g, r = divmod(idx, GROUP)
                members = list(range(g * GROUP, min((g + 1) * GROUP, len(grid.cores)))) if r < M_MAX else []
                fin = idx == r * GROUP + r and r < M_MAX
                wargs[(cx, cy)] += [wargs_m, int(logits) | 2 * int(ids), A(self.ids_t)] + \
                    [xy[g * GROUP + j] for j in range(M_MAX)] + \
                    [r if members else IDLE, xy[r * GROUP + r] if members else 0, len(members)] + \
                    [xy[j] for j in members] + [r if fin else IDLE, ng if fin else 0] + \
                    ([xy[j * GROUP + r] for j in range(ng)] if fin else [])
        kernels = [ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fhead_compute.cpp"),
                                         core_ranges=grid.ranges, compile_time_args=[BATCH], runtime_args=crt,
                                         defines=defs, config=cfg)]
        kernels += grid.dm_kernels(os.path.join(KDIR, "fhead_reader.cpp"), os.path.join(KDIR, "fhead_writer.cpp"),
                                   [BATCH] + acc, acc, rargs, wargs, defs, noc1_rows=noc1_rows)
        sems = grid.semaphores() + ([ttnn.SemaphoreDescriptor(id=i, core_ranges=grid.ranges, initial_value=0)
                                     for i in (3, 4)] if ext else [])
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=[grid.cb(*c) for c in spec])
        ins = [self.x32_t, self.x16_t, self.wd_t, self.wu_t, self.norm_t, self.wl_t, self.prof] + \
            list(self.chk.values()) + [self.logits] + ([] if self.gprof is None else [self.gprof]) + \
            ([self.ids_t] + ([] if self.hm_t is None else [self.hm_t]) if ext else []) + ([self.step_t] if xb else [])
        return lambda: ttnn.generic_op(ins, prog)

    def get_logits(self) -> torch.Tensor:
        """The VOCAB logits (fp32, host) of the last launch."""
        return ttnn.to_torch(self.logits).flatten()[:VOCAB]

    def get_ids(self, m=1) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows 0..m-1's top 64 of the last program(ids=True) launch: ids [m x 64] (int64, best first: column 0 the
        argmax) and their logits [m x 64] (fp32)."""
        w = ttnn.to_torch(self.ids_t).reshape(self.m_max, 2 * TOPK)[:m]
        return w[:, :TOPK].long() & 0xFFFFFFFF, w[:, TOPK:].contiguous().view(torch.float32)


def rm(dev, t):
    return ttnn.from_torch(t.float().reshape(1, 1, 1, -1).contiguous(), dtype=ttnn.float32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)


def refill(t, value) -> None:
    """Tensor t's contents set to value in place (an output not written keeps its old data)."""
    dt = torch.float32 if t.dtype == ttnn.float32 else torch.int32
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.full(tuple(t.shape), value, dtype=dt), dtype=t.dtype,
                                                    layout=t.layout), t)


def topk_ref(lg: torch.Tensor, k: int = TOPK) -> tuple[torch.Tensor, torch.Tensor]:
    """The top k of logits lg (1-D) best first, a tie to the lowest id (the card's rule, as llama.cpp's argmax):
    ids (int64), values."""
    order = torch.from_numpy(np.lexsort((np.arange(len(lg)), -lg.double().numpy()))[:k].astype(np.int64))
    return order, lg[order]


def host_read_us(fn, n: int = 50) -> float:
    """The median host time of fn() (a read-back), us."""
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(t))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))
    g = Gguf(MODEL)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, worker_l1_size=WORKER_L1)
    res = {"date": time.strftime("%Y-%m-%d %H:%M")}
    try:
        t0 = time.time()
        H = Head(dev, g, m_max=M_MAX)
        res["build_s"] = time.time() - t0
        print(f"head built in {res['build_s']:.0f} s", flush=True)
        x = torch.from_numpy(rng.standard_normal(HC * D).astype(np.float32))
        H.x32_t = rm(dev, x)
        H.x16_t = ttnn.from_torch(x.to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        # x's rows 1..3 for the M-row runs: random, one-hot (the logits one column of W, bfp8_b's values: many exact
        # ties), zero (every logit 0: the top 64 must be ids 0..63)
        hm = torch.zeros(M_MAX - 1, D)
        hm[0] = torch.from_numpy(rng.standard_normal(D).astype(np.float32))
        hm[1, int(rng.integers(D))] = 1.0
        hm = hm.to(torch.bfloat16)
        H.hm_t = ttnn.from_torch(hm.reshape(1, 1, M_MAX - 1, D), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=dev)
        run = H.program()
        run()
        ttnn.synchronize_device(dev)
        get = lambda t: ttnn.to_torch(t).double().flatten()               # noqa: E731
        X, X16 = x.double().reshape(HC, D), bf(x.double()).reshape(HC, D)
        r = torch.rsqrt(X.pow(2).mean(-1) + EPS)
        p1 = get(H.chk["p1"]).reshape(HC, LO)
        p_ref = torch.stack([r[s] * (X16[s] @ H.wdq[s * D:(s + 1) * D]) for s in range(HC)])
        res["H1"] = rel(p1[:, :LR], p_ref[:, :LR])
        lo = p1.sum(0)
        gate = (bf(torch.nn.functional.silu(lo[:LR] / HC)) @ H.wuq).reshape(HC, D)
        hn = (torch.sigmoid(gate) * X * H.norm.double().reshape(HC, D) * r[:, None]).mean(0)
        res["H2_hn"] = rel(get(H.chk["hn"]), hn)
        hn_card = bf(get(H.chk["hn"]))
        lg0 = H.get_logits()
        lg = lg0.double()
        ref = (hn_card.float() @ H.wout.T).double()                       # the GGUF's weights (fp32 math)
        res["logits"] = rel(lg, ref)
        res["top10_agree"] = len(set(torch.topk(lg, 10).indices.tolist()) & set(torch.topk(ref, 10).indices.tolist()))
        res["argmax_same"] = int(torch.argmax(lg) == torch.argmax(ref))
        print(f"max rel err: H1 {res['H1']:.1e}, hn {res['H2_hn']:.1e}, logits {res['logits']:.1e} (against the "
              f"GGUF's weights); top 10 agree {res['top10_agree']}, argmax same {res['argmax_same']}", flush=True)

        # ---- the top 64 on the card: exactly the top 64 of the card's own logits (ties to the lowest id); the M-row
        # run's logits are the reference rows (row 0 the same bits as the program above), and float64 x W beside them
        xs = [hn_card] + [hm[j].double() for j in range(M_MAX - 1)]
        f64s = [torch.cat([H.wout[i:i + 16384].double() @ v for i in range(0, VOCAB, 16384)]) for v in xs]
        rows, names = None, ["hn", "random", "one-hot", "zero"]
        res["ids"] = {}
        for m, want_lg in ((M_MAX, True), (1, True), (1, False), (M_MAX, False)):
            refill(H.logits, float("nan"))
            refill(H.ids_t, -1)
            H.program(m=m, logits=want_lg, ids=True)()
            ttnn.synchronize_device(dev)
            ids, vals = H.get_ids(m)
            key = f"m{m}_" + ("logits+ids" if want_lg else "ids")
            out = res["ids"][key] = {}
            if want_lg:
                lgm = ttnn.to_torch(H.logits).flatten()[:m * VOCAB].reshape(m, VOCAB)
                out["row0_logits_bitident"] = bool(torch.equal(lgm[0].view(torch.int32), lg0.view(torch.int32)))
                if rows is None:
                    rows = lgm
                else:
                    out["logits_same_as_m4"] = bool(torch.equal(lgm.view(torch.int32), rows[:m].view(torch.int32)))
            for j in range(m):
                rid, rv = topk_ref(rows[j])
                f64 = f64s[j]                                             # x times the GGUF's weights
                out[names[j]] = dict(ids_exact=bool(torch.equal(ids[j], rid)), values_exact=bool((vals[j] == rv).all()),
                                     argmax=int(ids[j, 0]), argmax_f64=int(torch.argmax(f64)),
                                     f64_gap=float(f64.max() - f64[ids[j, 0]]),   # how near a tie a miss is
                                     tied_in_top64=int(TOPK - len(set(rv.tolist()))),
                                     logits_vs_f64=rel(rows[j].double(), f64) if f64.abs().max() > 0 else
                                     float(rows[j].abs().max()))   # the zero row: the largest |logit|
            ok = all(v["ids_exact"] and v["values_exact"] for v in out.values() if isinstance(v, dict))
            print(f"{key}: top 64 {'exact' if ok else 'WRONG'}; " + "; ".join(
                f"{n} argmax {v['argmax']} (f64 {v['argmax_f64']}), ties {v['tied_in_top64']}"
                for n, v in out.items() if isinstance(v, dict)) +
                "".join(f"; {k} {v}" for k, v in out.items() if not isinstance(v, dict)), flush=True)

        res["traced_us"] = traced_us(dev, run, reps=8, tokens=10)
        st = ttnn.to_torch(H.prof).flatten().long().tolist()
        names = ["X", "H1", "HL", "H2", "LM head"]
        res["phase_cycles"] = dict(zip(names, [(st[i + 1] - st[i]) & 0xFFFFFFFF for i in range(5)]))
        print(f"head traced {res['traced_us']:.1f} us; phases (cycles): {res['phase_cycles']}", flush=True)
        for m, want_lg in ((1, False), (1, True), (M_MAX, False)):
            key = f"traced_us_m{m}_" + ("logits+ids" if want_lg else "ids")
            res[key] = traced_us(dev, H.program(m=m, logits=want_lg, ids=True), reps=8, tokens=10)
            print(f"{key} {res[key]:.1f}", flush=True)
        res["host_read_us_logits"] = host_read_us(H.get_logits)
        res["host_read_us_ids"] = host_read_us(H.get_ids)
        print(f"host read: logits {res['host_read_us_logits']:.1f} us, ids {res['host_read_us_ids']:.1f} us",
              flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    with open(os.path.join(OUT, "g3_head.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
