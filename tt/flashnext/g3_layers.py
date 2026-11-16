"""G3 step 8: consecutive DeltaNet layers with their own weights in one program (tt/kernels/flayer_*.cpp, a layer table):
each pass reads its layer's row of tensor addresses from card memory, and the conv and DeltaNet states are updated in
place, as in the decode loop. Layers blk.0..2 (blk.1's PLE block is not part of this program).

Checked: the program over the three layers against the three layers run one after another as separate programs (each
the single-layer program g3_layer.py checks against float64, the next fed the previous one's output), which must
agree to the rounding of the residual; then timed (traced, one launch) against the three separate launches.

  python3 tt_moeoffload/launch.py --memory-max $((32 << 30)) --engine tt/flashnext/g3_layers.py

Measured 2026-10-23: identical to the three separate programs (max rel diff 0); 1,129.7 us for the three layers in
one launch against 1,166.8 us as three launches; 1,028.3 against 1,078.9 with g3_layer.py's prefetched constants
and NOC split; 993.5 against 1,048.2 with the mixes' weight prefetch.

G3_CARD=n: the same with the card's share of the routed experts (CARD_EXPERTS, g5_layer.py) on layers blk.5, 6, 8
(their cached experts, $TT_MOEOFFLOAD_DATA/g5_cache), n routes from a tensor in card memory (the same slots for every
layer); the separate programs then take one-row layer tables (the cache's addresses come only from a table) and the
states are restored in between. The borrowed buffers must be left where the next layer's phases need them.

G3_M=M (3 or 4: every check filled): the same at M rows (the M_MAX = 4 program, g6_layer_m.py); the states are read
from one parity and written to the other, so the separate programs need no table and no restore.
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
from fused.grid import traced_us  # noqa: E402
from g2_layers import MODEL, Gguf  # noqa: E402
from g3_layer import WORKER_L1, WORKER_L1_M, Layer, rel  # noqa: E402

OUT = os.path.join(HERE, "results")
CARD_N = int(os.environ.get("G3_CARD", -1))   # >= 0: the card's experts on, this many routes
LAYERS = [5, 6, 8] if CARD_N >= 0 else [0, 1, 2]
ROWS = int(os.environ.get("G3_M", 0))       # > 0: the M-row program at this many rows
ROW = ("wd_a", "wu_a", "norm_a", "wd_f", "wu_f", "norm_f", "wa", "cstate", "taps_t", "dtna_t", "S", "nw_t", "wo",
       "wr", "wgu", "wsd", "routed_t", "gu_codes", "gu_scales", "d_vals", "d_scales")
CARD = ("gu_codes", "gu_scales", "d_vals", "d_scales")   # the layer's card expert cache (fused/dexp_dm.h), 0 without


def addr(L: Layer, name: str) -> int:
    if name in CARD:
        return L.card[name].buffer_address() if L.card else 0
    if name in ("wd_a", "wu_a", "norm_a", "wd_f", "wu_f", "norm_f"):
        site = L.hc["attn" if name.endswith("_a") else "ffn"]
        return site[{"wd": "wd", "wu": "wu", "no": "norm_t"}[name[:2]]].buffer_address()
    return getattr(L, name).buffer_address()


def table_of(dev, Ls: list[Layer]):
    rows = torch.zeros(len(Ls), 32, dtype=torch.int64)
    for i, L in enumerate(Ls):
        rows[i, :len(ROW)] = torch.tensor([addr(L, k) for k in ROW])
    return ttnn.from_torch(rows.to(torch.int32).reshape(1, 1, len(Ls), 32), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)


def restore_states(L: Layer) -> None:
    """The conv and DeltaNet states back to the layer's initial ones (a layer table updates them in place)."""
    tiles = L.S0.reshape(-1, 4, 32, 4, 32).permute(0, 1, 3, 2, 4).reshape(1, 1, -1, 32)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(L.s0.float().reshape(1, 1, 3, -1), dtype=ttnn.float32,
                                                    layout=ttnn.ROW_MAJOR_LAYOUT), L.cstate)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(tiles.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
                                    L.S)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))
    t0 = time.time()
    card = CARD_N >= 0
    from g4_model import SSD_MODEL, Cache    # g4_model imports this module
    g = Gguf(SSD_MODEL if card else MODEL)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20,
                                worker_l1_size=WORKER_L1_M if ROWS else WORKER_L1)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layers": LAYERS, "rows": ROWS or 1}
    try:
        Ls = [Layer(dev, g, rng, layer=il, m_max=4 if ROWS else 1) for il in LAYERS]
        for L in Ls if ROWS else []:
            L.set_inputs_m(rng, ROWS)
        if card:
            cache = Cache(dev, layers=LAYERS)
            n_slots = min(int(cache.slot_of[il].max()) + 1 for il in LAYERS)
            slots = rng.choice(n_slots, CARD_N, replace=False)
            w = rng.uniform(0.05, 0.3, CARD_N)
            w = (w / max(w.sum(), 1e-9) * 0.6).astype(np.float32)
            words = np.zeros(32, np.uint32)
            words[0], words[1:1 + CARD_N], words[11:11 + CARD_N] = CARD_N, slots, w.view(np.uint32)
            routes = ttnn.from_torch(torch.from_numpy(words.view(np.int32)).reshape(1, 1, 1, 32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            for L in Ls:
                L.card, L.card_on, L.part, L.card_routes = cache.tensors[L.layer], True, cache.part, routes
            res["card_routes"], res["card_slots"] = CARD_N, slots.tolist()
            print(f"the card's experts on: {CARD_N} routes, slots {slots.tolist()}", flush=True)
        print(f"weights for {len(LAYERS)} layers in {time.time() - t0:.0f} s", flush=True)
        x0 = Ls[0].x.clone()
        # one after another: layer i's X is layer i - 1's output, fp32 and the card's own bf16 copy, in place as the
        # decode loop hands it on (a bf16 rounded on the host differs where the packer's rounding of a tie does); the
        # states go to the *_out tensors, untouched (with the card, to the layer's own state tensors through a one-row
        # table, restored afterwards)
        xt = lambda: Ls[0].x_tensors(x0) if ROWS else (Ls[0].rm(x0), ttnn.from_torch(  # noqa: E731
            x0.float().to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev))
        x32, x16 = xt()
        for L in Ls:
            L.x32_t, L.x16_t, L.inplace, L.checks_inplace = x32, x16, True, True
            if card:
                L.table = table_of(dev, [L])
                L.table_addr = L.table.buffer_address()
            L.program()()
            ttnn.synchronize_device(dev)
        seq = ttnn.to_torch(x32).double().flatten()
        states = [L.states_m(1 - L.parity) for L in Ls] if ROWS else []
        last = {k: Ls[-1].get(k) for k in Ls[-1].chk}   # the last layer's check outputs (the timing overwrites them)
        last["x2"] = seq
        res["single_traced_us"] = [traced_us(dev, L.program()) for L in Ls]
        if card:
            for L in Ls:
                restore_states(L)
        # all three in one program, from layer 0's X, the states in place
        table = table_of(dev, Ls)
        A = Ls[0]
        A.inplace = False
        A.x32_t, A.x16_t = xt()
        A.table_addr = table.buffer_address()
        A.program(len(Ls))()
        ttnn.synchronize_device(dev)
        res["vs_one_after_another"] = rel(A.get("x2"), seq)
        if ROWS:   # the states each layer left in the other parity (the separate programs read the same ones)
            res["states_vs_one_after_another"] = max(rel(a, b) for L, st in zip(Ls, states)
                                                     for a, b in zip(L.states_m(1 - L.parity), st))
            print(f"  states vs one after another: {res['states_vs_one_after_another']:.1e}", flush=True)
        # the last layer's every phase (its check outputs: the combined program's land in layer 0's)
        res["phases_vs_one_after_another"] = {k: rel(A.get(k), last[k]) for k in A.chk if k not in ("wa", "wf")}
        print("  the last layer's phases: " + ", ".join(f"{k} {v:.1e}" for k, v in
                                                       res["phases_vs_one_after_another"].items()), flush=True)
        print(f"{len(Ls)} layers in one program vs one after another: max rel diff {res['vs_one_after_another']:.1e}",
              flush=True)
        res["program_traced_us"] = traced_us(dev, A.program(len(Ls)))
        print(f"{len(Ls)} layers: one program {res['program_traced_us']:.1f} us, separately "
              f"{sum(res['single_traced_us']):.1f} us ({', '.join(f'{t:.1f}' for t in res['single_traced_us'])})",
              flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    name = f"g3_layers_card{CARD_N}.json" if card else f"g3_layers_m{ROWS}.json" if ROWS else "g3_layers.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
