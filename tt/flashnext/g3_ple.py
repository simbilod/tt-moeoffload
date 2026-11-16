"""G3 step 10: Flash-Next's PLE block (layer 1's per-layer embedding, llama.cpp's build_ple) for one decode token as one
program on all 110 cores (tt/kernels/fple_*.cpp), real blk.1 weights: key|value projections of the token's embedding,
the per-stream sigmoid gate from the key and X norms, the gated value's RMSNorm, the dilated causal conv over the
token history ring, X += gated + silu(conv). Up to m <= 8 tokens per program (a verify pass's rows) and a ring of
RING = 17 slots (fple_reader.cpp: rejected rows never overwrite a slot a later pass reads).

The embedding (the host gathers it from the hashed n-gram table), X and the history are random. Checked against
float64 on the same inputs (the card's bf16 embedding, its bfp8_b weights dequantized), then timed (traced, one launch).

  python3 tt_moeoffload/launch.py --memory-max $((24 << 30)) --engine tt/flashnext/g3_ple.py

Measured 2026-10-23: X within 1.6e-4 of float64, the gates 1.3e-4, the new history row 5.5e-4; 186.8 us traced
(projection ~116, gate reduction ~20, gated norm + conv ~30 us). Two bugs found on the way: 32-byte reads from card
memory misread the embedding (it now arrives through the arena), and the conv read row t - 9 of the history after the
writer had overwritten it with this token's value (the reader now reads the history before the first pass's inputs).
With rows 0, 1, 9 reading through NOC1 (fused/grid.py) and P1/P3 split by bandwidth: 163.4 us (189.4 on NOC0 alone).
Then a verify loop's passes (PASSES: m = 1..4 rows, rows rejected, the ring's unused slots NaN at the
start) against float64 with the accepted tokens' history: every pass's X within 2.9e-4, n 1.1e-3, gates 5.6e-4. The
m = 1 program gives the same bits as before (X, X16, n, gates; in place too). Time by position: the ring's three read
rows and nc / taps share DRAM banks at some positions (163-173 us, the same spread with 9 slots); over 17 positions
169.5 us against 169.0 with 9 slots; 276.5 us at m = 4.
The same passes with X in the DeltaNet program's M-row layout (Ple(x_blocks=True), M from the step
tensor, one program): every row against float64 as above, rows M..Mp-1 of X out copies of row M - 1.
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
from g3_hc import HC, model_tag, stream  # noqa: E402
from g3_layer import WORKER_L1, bf, blocks, mp_of, rel, step_words, unblocks  # noqa: E402

KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
OUT = os.path.join(HERE, "results")
D, CH = 2560, 10240
BATCH, DEPTH = 8, 4
IDLE = 0xFFFFFFFF
POS = 1000
M_RING = 8                  # the most rows a pass may have against the ring (fple_reader.cpp)
RING = 9 + M_RING           # the history ring's slots: token q's n at row q % RING
EXACT = (3, 8, 14, 15, 16, 17, 18, 19, 20, 22, 23)


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def rm(dev, t):
    return ttnn.from_torch(t.float().reshape(1, 1, -1, t.shape[-1]) if t.dim() > 1 else t.float().reshape(1, 1, 1, -1),
                           dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)


class Ple:
    """The PLE block's weights and program. The token loop (g4_model.py) points x32_t, x16_t at the model's X and
    sets inplace (X back into them), emb_t at the host's gathered embedding, step_t at the token's step tensor.
    With m rows (program(m)): X [m x 4 x 2560] (fp32, bf16), emb [m x 2560], tokens pos..pos+m-1. With x_blocks: X
    in the M-row layout of the DeltaNet program (g3_layer.Layer(m_max).x_tensors), M from the step tensor."""

    def __init__(self, dev, g: Gguf, layer=1, back=True, m_max=1, x_blocks=False):
        """m_max: the most rows a program() of this block takes (the arena's and the checks' size). x_blocks: X in the
        M-row layout (fple_reader.cpp X_BLOCKS): [4 x m_max 2560], a stream a page, each tile a block of its Mp rows;
        M and Mp from the step tensor's words 1 and 4 (g3_layer.step_words), one program for every M <= m_max."""
        assert 1 <= m_max <= M_RING and (not x_blocks or m_max in (2, 4))
        self.dev, self.grid, self.m_max, self.x_blocks = dev, Grid(dev), m_max, x_blocks
        p = f"blk.{layer}."
        self.n = {x: g(p + f"ple_norm_{x}.weight") for x in ("key", "query", "conv")}   # [10240] each
        self.taps = g(p + "ple_conv1d.weight").T.contiguous()                   # [4, 10240], row 0 = oldest
        self.w_t, self.wq = stream(dev, lambda: torch.cat([g(p + "ple_key.weight"), g(p + "ple_value.weight")]).T,
                                   back, f"{model_tag(g)}/{p}ple_kv")              # [2560, 12800]
        self.nk_t, self.nq_t, self.nc_t = (rm(dev, self.n[x]) for x in ("key", "query", "conv"))
        self.taps_t = rm(dev, self.taps)
        self.ring_t = rm(dev, torch.zeros(RING, CH))                            # the history (zero at the start)
        self.x32_t = self.x16_t = self.emb_t = self.step_t = None
        self.inplace = False
        self.chk = dict(ps=nan_tensor(dev, (1, 1, m_max, 720 * 4)), pg=nan_tensor(dev, (1, 1, m_max, 32)),
                        x=nan_tensor(dev, (1, 1, HC, m_max * D) if x_blocks else (1, 1, m_max, HC * D)))
        self.prof = ttnn.from_torch(torch.zeros(1, 1, 1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.gprof, self.gprof_n = None, 32  # grid.h's per-core clocks (fused/grid.py profile_tensor; g4_phases.py)
        o, a = {}, 0      # per token blocks (fple_reader.cpp T1, T2), so m rows broadcast as one run of the arena
        first = (("x32", HC * D * 4), ("emb", D * 2)) if x_blocks else (("x32", HC * D * 4), ("x16", HC * D * 2),
                                                                        ("emb", D * 2))
        for block in (first, (("pv", D * 4), ("ps", 720 * 16)), (("pg", 128),)):
            for name, size in block:
                o[name] = a
                a += size
            a += (m_max - 1) * sum(size for _, size in block)
        if x_blocks:      # X's blocks from X32 on, the embeddings after them (placed by Mp), no X bf16
            o["x16"] = o["emb"] = o["x32"]
        self.off, self.arena = o, a

    def program(self, noc1_rows=NOC1_ROWS, m=1):
        """PLE's program for m rows, the rows noc1_rows reading through NOC1 (fused/grid.py): P1's and P3's tiles split
        by each core's share of card memory bandwidth, P2's items (row, stream) on the NOC0 rows. With x_blocks m is
        the step tensor's (P2's items then for m_max rows)."""
        m = self.m_max if self.x_blocks else m
        assert 1 <= m <= self.m_max
        grid, A = self.grid, (lambda t: t.buffer_address())                     # noqa: E731
        offs = [self.off[k] for k in ("x32", "x16", "emb", "pv", "ps", "pg")]
        acc = list(ttnn.TensorAccessorArgs(self.w_t).get_compile_time_args())
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        for cb in EXACT:
            mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        gp, gdef, gbytes = profile_plumbing(self.gprof, self.gprof_n, 63, 64)  # in cb 63, after the 16 stamps
        xdef = [("X_BLOCKS", str(self.m_max))] if self.x_blocks else []        # (every kernel takes it)
        f32, bf16, u8 = ttnn.float32, ttnn.bfloat16, ttnn.uint8
        T4, T2 = 4096, 2048
        spec = [(0, ttnn.bfloat8_b, 1088, BATCH * DEPTH), (1, bf16, T2, 80), (2, bf16, T2, 1), (3, f32, T4, 3),
                (5, f32, T4, 2), (6, f32, T4, 2), (7, f32, T4, 1), (8, f32, T4, 3), (9, f32, T4, 3),
                (11, f32, T4, 3), (12, f32, T4, 4), (13, f32, T4, 1), (26, f32, T4, 1), (14, f32, T4, 1),
                (15, f32, T4, 2), (16, f32, T4, 1), (17, f32, T4, 4), (18, f32, T4, 3), (19, f32, T4, 1),
                (20, f32, T4, 1), (21, f32, T4, 1), (22, f32, T4, 1), (23, f32, T4, 1), (24, f32, T4, 1),
                (25, bf16, T2, 1), (27, u8, 128 * self.m_max, 2), (60, u8, T4, 1), (61, u8, self.arena, 1),
                (62, u8, max(2048, 320 * self.m_max), 1), (63, u8, 64 + gbytes, 1)]
        sp = grid.speeds(noc1_rows)
        s1, s3, order = split_weighted(400, sp), split_weighted(320, sp), grid.order(noc1_rows)
        x_chk = A(self.x32_t) if self.inplace else A(self.chk["x"])
        x16_out = A(self.x16_t) if self.inplace else 0
        rargs, crt, wargs = {}, ttnn.RuntimeArgs(), {}
        for idx, (cx, cy) in enumerate(grid.cores):
            k = order.index(idx)
            st = k if k < HC * m else IDLE
            rargs[(cx, cy)] = [A(self.x32_t), A(self.x16_t), A(self.emb_t)] + grid.args(idx) + offs + [
                A(self.w_t), *s1[idx], st, *s3[idx], A(self.nk_t), A(self.nq_t), A(self.nc_t), A(self.taps_t),
                A(self.ring_t), A(self.step_t), A(self.prof)] + gp(idx) + [m]
            crt[cx][cy] = [*s1[idx], st, s3[idx][1]]
            wargs[(cx, cy)] = grid.args(idx) + offs + [*s1[idx], st, *s3[idx], A(self.ring_t), A(self.step_t),
                                                       A(self.chk["ps"]), A(self.chk["pg"]), x_chk, x16_out] + \
                gp(idx) + [m]
        kernels = [ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fple_compute.cpp"),
                                         core_ranges=grid.ranges, compile_time_args=[BATCH], runtime_args=crt,
                                         config=cfg, defines=xdef)]
        kernels += grid.dm_kernels(os.path.join(KDIR, "fple_reader.cpp"), os.path.join(KDIR, "fple_writer.cpp"),
                                   [BATCH] + acc, acc, rargs, wargs, gdef + xdef, noc1_rows=noc1_rows)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=grid.semaphores(), cbs=[grid.cb(*c) for c in spec])
        ins = [self.x32_t, self.x16_t, self.emb_t, self.w_t, self.nk_t, self.nq_t, self.nc_t, self.taps_t,
               self.ring_t, self.step_t, self.prof] + list(self.chk.values()) + (
                   [] if self.gprof is None else [self.gprof])
        return lambda: ttnn.generic_op(ins, prog)


def step_tensor(dev, pos: int):
    """The token's step tensor (uint32 [16]; word 0 the position), shared by the token loop's programs."""
    v = torch.zeros(1, 1, 1, 16, dtype=torch.int32)
    v[..., 0] = pos
    return ttnn.from_torch(v, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)


def reference(P, X, emb, hist, t):
    """Float64 PLE for token t: X [4 x 2560], emb [2560] (bf16 values), hist {q: n_q [10240]} holding t - 9, t - 6,
    t - 3 -> X', n_t [10240], gate [4], g [4] (gate / rms(gated))."""
    nk, nq, nc, taps = (v.double() for v in (P.n["key"], P.n["query"], P.n["conv"], P.taps))
    y = emb.double() @ P.wq                                              # [12800]
    key, value = y[:CH].reshape(HC, D), y[CH:]
    X = X.double().reshape(HC, D)
    s = (rms(key) * nk.reshape(HC, D) * rms(X) * nq.reshape(HC, D)).sum(-1) / D ** 0.5
    gate = torch.sigmoid(torch.sign(s) * torch.sqrt(torch.clamp(s.abs(), min=1e-6)))
    gated = value[None, :] * gate[:, None]
    n = rms(gated) * nc.reshape(HC, D)
    h = [hist[t - 9 + 3 * k].double().reshape(HC, D) for k in range(3)]
    conv = (taps.reshape(4, HC, D) * torch.stack(h + [n])).sum(0)
    g = gate * torch.rsqrt(gate ** 2 * value.pow(2).mean() + EPS)
    return X + gated + torch.nn.functional.silu(conv), n.flatten(), gate, g


def put(t, v: torch.Tensor) -> None:
    """Host tensor v into card tensor t in place (its shape, dtype)."""
    dt = {ttnn.bfloat16: torch.bfloat16, ttnn.uint32: torch.int32}.get(t.dtype, torch.float32)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(v.reshape(tuple(t.shape)).to(dt), dtype=t.dtype,
                                                    layout=t.layout), t)


# a verify loop's passes: (first position, rows, rows accepted); the rejected rows' n stay in the ring
PASSES = ((POS, 4, 2), (POS + 2, 4, 1), (POS + 3, 4, 4), (POS + 7, 1, 1), (POS + 8, 3, 3), (POS + 11, 4, 4),
          (POS + 15, 2, 1), (POS + 16, 4, 4))


def verify_passes(dev, P, R, rand) -> list:
    """PASSES through P (token rows: a program per m; x_blocks: one, M from the step tensor) against float64 with R's
    weights: M rows each, the rejected rows' n left in the ring; the ring's other slots NaN at the start (a read of a
    slot it should not need shows), the history the accepted tokens' float64 n. With x_blocks X goes in and out in the
    M-row layout (rows M..Mp-1 copies of row M - 1, which X out must keep)."""
    get = lambda t: ttnn.to_torch(t).double()                                              # noqa: E731
    mm, xb = P.m_max, P.x_blocks
    shape = (1, 1, HC, mm * D) if xb else (1, 1, mm, HC * D)
    P.x32_t = rm(dev, torch.zeros(shape[2:]))
    P.x16_t = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    P.emb_t = ttnn.from_torch(torch.zeros(1, 1, mm, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    P.step_t = step_tensor(dev, POS)
    ring = torch.full((RING, CH), float("nan"))
    acc = {}
    for q in range(POS - 9, POS):
        acc[q] = rand(CH) * 0.5
        ring[q % RING] = acc[q]
    put(P.ring_t, ring)
    progs = {m: P.program(m=m) for m in sorted({m for _, m, _ in PASSES})}
    out = []
    for p, m, keep in PASSES:
        xs, es = rand(mm, HC * D), rand(mm, D).to(torch.bfloat16)
        if xb:
            mp = mp_of(m)
            xs[m:mp] = xs[m - 1]
            xin = torch.stack([blocks(xs[:mp, c * D:(c + 1) * D], mp) for c in range(HC)])
            xin = torch.cat([xin, torch.zeros(HC, (mm - mp) * D)], 1)
        else:
            xin = xs
        put(P.x32_t, xin)
        put(P.x16_t, xin)
        put(P.emb_t, es)
        put(P.step_t, step_words(m, pos=p) if xb else torch.tensor([p] + [0] * 15, dtype=torch.int32))
        for t in P.chk.values():
            put(t, torch.full(tuple(t.shape), float("nan")))
        progs[m]()
        ttnn.synchronize_device(dev)
        if xb:
            raw = get(P.chk["x"]).reshape(HC, mm * D)
            xc = torch.stack([unblocks(raw[c], D // 32, mp) for c in range(HC)], 1)        # [mp, HC, D]
        else:
            xc = get(P.chk["x"]).reshape(mm, HC, D)
        rc = get(P.ring_t).reshape(RING, CH)
        pgc = get(P.chk["pg"]).reshape(mm, HC, 2, 4)[..., 0]
        h, err = dict(acc), {"X": 0.0, "n": 0.0, "gate": 0.0, "g": 0.0}
        for r in range(m):
            xr, n, gate, gref = reference(R, xs[r], es[r], h, p + r)
            h[p + r] = n                                  # this pass's rows feed its later rows' conv
            if r < keep:
                acc[p + r] = n
            for k, v in (("X", rel(xc[r], xr.reshape(HC, D))), ("n", rel(rc[(p + r) % RING], n)),
                         ("gate", rel(pgc[r, :, 0], gate)), ("g", rel(pgc[r, :, 1], gref))):
                err[k] = max(err[k], v)
        if xb:
            err["rows_after_M_copies"] = bool(all(torch.equal(xc[r], xc[m - 1]) for r in range(m, mp)))
        out.append(dict(pos=p, m=m, keep=keep, **err))
        print(f"pass at {p}, M = {m}, keep {keep}{' (M-row layout)' if xb else ''}: max rel err " +
              ", ".join(f"{k} {v:.1e}" if isinstance(v, float) else f"{k} {v}" for k, v in err.items()), flush=True)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))
    g = Gguf(MODEL)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, worker_l1_size=WORKER_L1)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "pos": POS, "ring": RING}
    inplace = os.environ.get("INPLACE", "0") == "1"   # X back into x32_t / x16_t (the token loop's way)
    rand = lambda *shape: torch.from_numpy(rng.standard_normal(shape).astype(np.float32))  # noqa: E731
    get = lambda t: ttnn.to_torch(t).double()                                              # noqa: E731
    try:
        P = Ple(dev, g, m_max=1 if inplace else 4)
        x = rand(HC * D)
        emb = rand(D).to(torch.bfloat16)
        hist = rand(RING, CH) * 0.5
        P.x32_t, P.ring_t = rm(dev, x), rm(dev, hist)
        P.x16_t = ttnn.from_torch(x.to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        P.emb_t = ttnn.from_torch(emb.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=dev)
        P.step_t = step_tensor(dev, POS)
        P.inplace = inplace
        run = P.program()
        run()
        ttnn.synchronize_device(dev)
        xr, n, gate, gref = reference(P, x, emb, {q: hist[q % RING] for q in range(POS - 9, POS)}, POS)
        pg = get(P.chk["pg"]).flatten()[:32].reshape(HC, 2, 4)[:, :, 0]
        res["gate"], res["g"] = rel(pg[:, 0], gate), rel(pg[:, 1], gref)
        res["X"] = rel(get(P.x32_t if inplace else P.chk["x"]).flatten()[:HC * D].reshape(HC, D), xr)
        if inplace:
            res["X16"] = rel(get(P.x16_t).reshape(HC, D), bf(xr))
        res["ring_row"] = rel(get(P.ring_t).reshape(RING, CH)[POS % RING], n)
        print("M = 1, max rel err: " + ", ".join(f"{k} {v:.1e}" for k, v in res.items() if isinstance(v, float)),
              flush=True)
        if not inplace:
            res["traced_us"] = traced_us(dev, run)
            st = ttnn.to_torch(P.prof).flatten().long().tolist()
            names = ["X", "P1", "P2", "P3"]
            res["phase_cycles"] = dict(zip(names, [(st[i + 1] - st[i]) & 0xFFFFFFFF for i in range(4)]))
            print(f"PLE traced {res['traced_us']:.1f} us; phases (cycles): {res['phase_cycles']}", flush=True)

            # ---- a verify loop's passes, X as token rows (one program per m), then in the M-row layout (one program)
            res["passes"] = verify_passes(dev, P, P, rand)
            res["traced_us_m4"] = traced_us(dev, P.program(m=4))
            print(f"PLE traced at M = 4: {res['traced_us_m4']:.1f} us", flush=True)
            B = Ple(dev, g, m_max=4, x_blocks=True, back=False)
            res["passes_blocks"] = verify_passes(dev, B, P, rand)
            run = B.program()
            for m in (1, 4):
                put(B.step_t, step_words(m, pos=POS))
                res[f"traced_us_blocks_m{m}"] = traced_us(dev, run)
                print(f"PLE (M-row layout) traced at M = {m}: {res[f'traced_us_blocks_m{m}']:.1f} us", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    with open(os.path.join(OUT, "g3_ple.json" if not inplace else "g3_ple_inplace.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
