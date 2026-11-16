"""G3 step 2: Flash-Next's gated DeltaNet mixer as one program on all 110 cores (tt/kernels/fgdn_*.cpp), in three
phases with a grid barrier between them, real blk.0 weights.

Phase A: the input projection hn -> q | k | v | z | alpha | beta (2560 -> 16512, real blk.0 weights from
the HDD GGUF in bfp8_b) on all cores, each core's output tiles followed by their channel-local epilogue in fp32: the
causal conv (kernel 4) + silu on the conv channels with the new conv state written back, the L2 norm of each q and k
head (q also scaled by 1/sqrt(128)), sigmoid(z), the decay exp(softplus(alpha + dt) * neg_a) and sigmoid(beta). The
results are gathered into the coordinator's arena and broadcast at the barrier.
Phase B: one value head per core (48 cores): the delta rule on its 128 x 128 fp32 state (read from card memory before
the barrier, written back after) and the gated RMSNorm.
Phase C: ssm_out (6144 -> 2560) on 80 cores.

Checked against float64 on the same inputs (the card's bfp8_b weights dequantized, hn in bf16, the conv state and the
constants in fp32), with every output tensor pre-filled with NaN and fresh random inputs per run (so stale data from
an earlier run cannot pass), then timed.

  python3 tt_moeoffload/launch.py --memory-max $((24 << 30)) --engine tt/flashnext/g3_gdn.py
  (G3_PHASES=1 or 3 runs phase A, or A and B, alone)

Measured 2026-10-23: every output within 6.1e-4 of float64 (q 4.9e-4, conv state 2.8e-4, the delta rule's output
6.1e-4 and new state 4.3e-4, ssm_out 3.1e-4); 264.2 us traced for the mixer (A alone 169.7, A + B 208.6) against
616.1 us for G2's stock ops; its 61.6 MB of weights need ~157 us at the GEMV's 391 GB/s.
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
from fused.grid import Grid, traced_us  # noqa: E402
from g2_layers import MODEL, Gguf  # noqa: E402

KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
OUT = os.path.join(HERE, "results")
LAYER = 0
D, NK, NV, HK = 2560, 16, 48, 128
CONV, ZD = 2 * NK * HK + NV * HK, NV * HK      # 10240 conv channels, 6144 gate
NA = 64                                        # alpha, beta padded to 2 tiles each
NA_T = CONV // 32 + ZD // 32                   # first alpha tile (512)
NT = NA_T + 4                                  # 516 output tiles
KT = D // 32
KC = ZD // 32
IDLE = 0xFFFFFFFF
W_PAGE = 1088
T_V, T_Q, T_K, T_Z, T_A, T_B = range(6)
EPS = 1e-6
BATCH, DEPTH = 8, 8
PHASES = int(os.environ.get("G3_PHASES", 7))       # bit 0 A, 1 B, 2 C


def units() -> list[tuple[int, int, int]]:
    """(first tile, count, type) of every work unit: q and k heads are 4 tiles (their norm needs the whole head)."""
    out = [(4 * h, 4, T_Q) for h in range(NK)] + [(64 + 4 * h, 4, T_K) for h in range(NK)]
    out += [(n, 1, T_V) for n in range(128, CONV // 32)] + [(n, 1, T_Z) for n in range(CONV // 32, NA_T)]
    out += [(NA_T + j, 1, T_A) for j in range(2)] + [(NA_T + 2 + j, 1, T_B) for j in range(2)]
    return out


def assign(work, n_cores, speed=None):
    """Greedy by tiles, the 4-tile heads first; a core's head (at most one) leads its list (its CBs start aligned).
    With speed (each core's relative share of card memory bandwidth, Grid.speeds()), a unit goes to the core that
    would finish it first, the faster core on a tie."""
    speed = speed or [1.0] * n_cores
    load, lists = [0] * n_cores, [[] for _ in range(n_cores)]
    for u in sorted(work, key=lambda u: -u[1]):
        c = min(range(n_cores), key=lambda i: ((load[i] + u[1]) / speed[i], -speed[i]))
        lists[c].append(u)
        load[c] += u[1]
    assert all(sum(1 for u in lst if u[1] == 4) <= 1 for lst in lists)
    return lists


def nan_tensor(dev, shape, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(torch.full(shape, float("nan")), dtype=ttnn.float32, layout=layout, device=dev)


class Mixer:
    def __init__(self, dev, g: Gguf, rng, phases=7):
        self.dev, self.phases = dev, phases
        self.grid = Grid(dev)
        p = f"blk.{LAYER}."
        qkv, z = g(p + "attn_qkv.weight"), g(p + "attn_gate.weight")             # [10240, 2560], [6144, 2560]
        alpha, beta = g(p + "ssm_alpha.weight"), g(p + "ssm_beta.weight")        # [48, 2560]
        pad = lambda m: torch.cat([m, torch.zeros(NA - NV, D)])                  # noqa: E731
        self.w, self.wq = self.stream(torch.cat([qkv, z, pad(alpha), pad(beta)]).T)  # [2560, 16512]
        self.wo, self.woq = self.stream(g(p + "ssm_out.weight").T)                  # [6144, 2560]
        self.taps = g(p + "ssm_conv1d.weight").T.contiguous()                   # [4, 10240], row 0 = oldest
        dtna = torch.zeros(2, NA)
        dtna[0, :NV], dtna[1, :NV] = g(p + "ssm_dt.bias"), g(p + "ssm_a")        # ssm_a holds -exp(A_log)
        self.dtna, self.nw = dtna, g(p + "ssm_norm.weight")                      # [128]
        self.taps_t, self.dtna_t = self.rm(self.taps.reshape(1, 1, 4, CONV)), self.rm(dtna.reshape(1, 1, 2, NA))
        self.nw_t = self.rm(self.nw.reshape(1, 1, 1, HK))
        self.set_inputs(rng)
        self.lists = assign(units(), len(self.grid))
        self.max_c = max(sum(u[1] for u in lst if u[2] in (T_V, T_Q, T_K, T_A)) for lst in self.lists)
        # arena: hn (bf16), region A (fp32, 516 tiles x 32), B (bf16, 6144), C (fp32, 2560)
        self.ra = KT * 64
        self.rb = self.ra + NT * 128
        self.rc = self.rb + NV * HK * 2
        self.arena = self.rc + D * 4

    def rm(self, t):
        return ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.dev)

    def stream(self, w):
        """[k, n] -> bfp8_b tiles n-major (page n * kt + k), and the dequantized weights back as float64 [k, n]."""
        k, n = w.shape
        kt, nt = k // 32, n // 32
        st = w.contiguous().to(torch.bfloat16).reshape(kt, 32, nt, 32).permute(2, 0, 1, 3).reshape(1, 1, -1, 32)
        t = ttnn.from_torch(st, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.dev)
        return t, ttnn.to_torch(t).double().reshape(nt, kt, 32, 32).permute(1, 2, 0, 3).reshape(k, n)

    def set_inputs(self, rng):
        """Fresh hn, conv state, DeltaNet state; every output pre-filled with NaN."""
        hn = torch.from_numpy(rng.standard_normal(D).astype(np.float32)).to(torch.bfloat16)
        self.hn_host = hn.double()
        self.hn = ttnn.from_torch(hn.reshape(1, 1, 1, D), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=self.dev)
        self.s0 = torch.from_numpy(rng.standard_normal((3, CONV)).astype(np.float32)) * 0.5
        self.state = self.rm(self.s0.reshape(1, 1, 3, CONV))
        self.S0 = torch.from_numpy(rng.standard_normal((NV, HK, HK)).astype(np.float32)) * 0.05
        tiles = self.S0.reshape(NV, 4, 32, 4, 32).permute(0, 1, 3, 2, 4).reshape(1, 1, -1, 32)   # (h, j, c) tiles
        self.S = ttnn.from_torch(tiles, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.dev)
        self.state_out = nan_tensor(self.dev, (1, 1, 3, CONV))
        self.S_out = nan_tensor(self.dev, (1, 1, NV * 16 * 32, 32), ttnn.TILE_LAYOUT)
        self.check_a = nan_tensor(self.dev, (1, 1, 1, NT * 32))
        self.check_b = nan_tensor(self.dev, (1, 1, 1, NV * HK))
        self.check_c = nan_tensor(self.dev, (1, 1, 1, D))

    def program(self):
        g = self.grid
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        c_split = [(n, 1) for n in range(D // 32)] + [(0, 0)] * (len(g) - D // 32)
        for idx, (cx, cy) in enumerate(g.cores):
            lst = self.lists[idx]
            flat = [v for u in lst for v in u]
            head = idx if idx < NV else IDLE
            cf, cc = c_split[idx]
            rrt[cx][cy] = [self.hn.buffer_address()] + g.args(idx) + [
                self.w.buffer_address(), self.state.buffer_address(), self.taps_t.buffer_address(),
                self.dtna_t.buffer_address(), len(lst)] + flat + [
                head, self.S.buffer_address(), self.nw_t.buffer_address(), self.wo.buffer_address(), cf, cc,
                self.ra, self.rb, self.rc]
            crt[cx][cy] = [len(lst)] + flat + [head, cc]
            wrt[cx][cy] = [self.state_out.buffer_address(), self.check_a.buffer_address()] + g.args(idx) + \
                [self.ra, len(lst)] + flat + [head, self.S_out.buffer_address(), self.check_b.buffer_address(),
                                             self.rb, cf, cc, self.check_c.buffer_address(), self.rc]
        acc = list(ttnn.TensorAccessorArgs(self.w).get_compile_time_args())
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (3, 4, 6, 8, 10, 13, 14, 16, 24):
            mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fgdn_reader.cpp"), core_ranges=g.ranges,
                                  compile_time_args=[BATCH, DEPTH, self.phases] + acc, runtime_args=rrt,
                                  config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fgdn_writer.cpp"), core_ranges=g.ranges,
                                  compile_time_args=[self.phases] + acc, runtime_args=wrt,
                                  config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fgdn_compute.cpp"), core_ranges=g.ranges,
                                  compile_time_args=[BATCH, self.phases], runtime_args=crt, config=cfg),
        ]
        f32, bf16 = ttnn.float32, ttnn.bfloat16
        spec = [(0, ttnn.bfloat8_b, W_PAGE, BATCH * DEPTH), (1, bf16, 2048, KC), (2, bf16, 2048, 1),
                (3, f32, 4096, 7 * self.max_c), (4, f32, 4096, 4), (5, f32, 4096, 4), (6, f32, 4096, 1),
                (7, f32, 4096, 4), (8, f32, 4096, 16), (9, f32, 4096, 16), (10, f32, 4096, 4),
                (11, f32, 4096, 4), (12, f32, 4096, 4), (13, f32, 4096, 10), (14, f32, 4096, 4),
                (15, f32, 4096, 4), (16, f32, 4096, 8), (17, f32, 4096, 8), (18, f32, 4096, 12),
                (19, f32, 4096, 8), (20, f32, 4096, 16), (21, bf16, 2048, 4), (22, f32, 4096, 4),
                (23, f32, 4096, 4), (24, f32, 4096, 4), (29, ttnn.uint8, 4096, 1),
                (30, ttnn.uint8, self.arena, 1), (31, ttnn.uint8, 1024, 1)]
        cbs = [g.cb(*x) for x in spec]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=cbs)
        ins = [self.hn, self.w, self.wo, self.state, self.taps_t, self.dtna_t, self.nw_t, self.S, self.state_out,
               self.S_out, self.check_a, self.check_b, self.check_c]
        return lambda: ttnn.generic_op(ins, prog)

    def reference(self):
        """float64 on the card's inputs; each phase from the card's own previous outputs (as the card reads them)."""
        y = self.hn_host @ self.wq                                               # [16512]
        s, taps = self.s0.double(), self.taps.double()
        win = torch.cat([s, y[None, :CONV]])
        x = torch.nn.functional.silu((win * taps).sum(0))
        q = x[:NK * HK].reshape(NK, HK)
        k = x[NK * HK:2 * NK * HK].reshape(NK, HK)
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + EPS) * HK ** -0.5
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + EPS)
        z = torch.sigmoid(y[CONV:CONV + ZD])
        dt, na = self.dtna.double()
        a = y[NA_T * 32:NA_T * 32 + NA]
        decay = torch.exp(torch.nn.functional.softplus(a + dt) * na)
        b = torch.sigmoid(y[NA_T * 32 + NA:])
        A = torch.cat([q.flatten(), k.flatten(), x[2 * NK * HK:], z, decay, b])
        # B from the card's A
        a_card = ttnn.to_torch(self.check_a).double().flatten()
        q, k = a_card[:2048].reshape(NK, HK), a_card[2048:4096].reshape(NK, HK)
        v, z = a_card[4096:CONV].reshape(NV, HK), a_card[CONV:CONV + ZD].reshape(NV, HK)
        d, b = a_card[NA_T * 32:NA_T * 32 + NV], a_card[NA_T * 32 + NA:NA_T * 32 + NA + NV]
        q, k = q.repeat(3, 1), k.repeat(3, 1)                                    # v head h reads k head h % 16
        S = self.S0.double() * d[:, None, None]
        delta = (v - torch.einsum("hk,hkv->hv", k, S)) * b[:, None]
        S = S + k[:, :, None] * delta[:, None, :]
        o = torch.einsum("hk,hkv->hv", q, S)
        B = (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * self.nw.double() * z).flatten()
        # C from the card's B, in bf16 as the card reads it
        b_card = ttnn.to_torch(self.check_b).double().flatten().to(torch.bfloat16).double()
        C = b_card @ self.woq
        return A, torch.cat([s[1:], y[None, :CONV]]), B, S, C

    def errors(self) -> dict:
        A, sref, B, S, C = self.reference()
        rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()  # noqa: E731
        got = ttnn.to_torch(self.check_a).double().flatten()
        segs = {"q": (0, 2048), "k": (2048, 4096), "v": (4096, CONV), "z": (CONV, CONV + ZD),
                "decay": (NA_T * 32, NA_T * 32 + NV), "beta": (NA_T * 32 + NA, NA_T * 32 + NA + NV)}
        res = {k: rel(got[a:b], A[a:b]) for k, (a, b) in segs.items()}
        res["1-decay"] = rel(1 - got[NA_T * 32:NA_T * 32 + NV], 1 - A[NA_T * 32:NA_T * 32 + NV])
        res["conv_state"] = rel(ttnn.to_torch(self.state_out).double().reshape(3, CONV), sref)
        if self.phases & 2:
            res["B_out"] = rel(ttnn.to_torch(self.check_b).double().flatten(), B)
            st = ttnn.to_torch(self.S_out).double().reshape(NV, 4, 4, 32, 32).permute(0, 1, 3, 2, 4)
            res["B_state"] = rel(st.reshape(NV, HK, HK), S)
        if self.phases & 4:
            res["C_out"] = rel(ttnn.to_torch(self.check_c).double().flatten(), C)
        return res


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))
    t0 = time.time()
    g = Gguf(MODEL)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": LAYER}
    try:
        m = Mixer(dev, g, rng, PHASES)
        print(f"weights in {time.time() - t0:.0f} s; max tiles per core "
              f"{max(sum(u[1] for u in lst) for lst in m.lists)}", flush=True)
        m.program()()
        ttnn.synchronize_device(dev)
        res["max_rel_err"] = m.errors()
        print("max rel err: " + ", ".join(f"{k} {v:.1e}" for k, v in res["max_rel_err"].items()), flush=True)
        res["traced_us"] = traced_us(dev, m.program())
        mb = (m.w.volume() + m.wo.volume()) * 1.0625 / 1e6
        print(f"phases {PHASES}: traced {res['traced_us']:.1f} us ({mb:.1f} MB of weights)", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, "g3_gdn.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
