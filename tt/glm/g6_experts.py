"""GLM's routed experts in Flash-Next's card form (tt/flashnext/g7_experts.py, a Shape of GLM's):
gate|up IQ3_S as is (the code tiles and compact d / m pages Flash-Next's gate|up uses), down IQ4_XS by DECODE_IQ4NL with
HAS_M (d per 256 weights, an m per 32, as gate|up's). The 34 IQ4_XS-down layers 7-40, and (2026-11-14) those of
the 8 Q5_K-down trunk layers (3-6, 41-44) whose card form is built: the down re-quantized to int8 with the same pages
(g7_experts.q5k_as_q8, pf_gexp's VALUES + HAS_M), a share of at most CardExperts.q8_groups; the MTP's 45 stays on the lane. A layer's card form is 3.75 GB (gate|up codes 8 MB an expert,
their scales 688 KB, down 4 MB + 344 KB), 128 GB for the 34, in $TT_MOEOFFLOAD_DATA/glm_experts/<model tag>.

L1 at K = 4096 (the GLM process's worker L1, g3_fkda.WORKER_L1_M, 1,404 KB): gate|up's decoded W column is 128 fp32 tiles
(512 KB) and its x 128 bf16 tiles a token tile (256 KB), so MT 3 (1,386 KB of circular buffers; Flash-Next's 5 at K 2560);
the down's x (h, fp32) 64 tiles a token tile (256 KB) beside its 256 KB column: MT_DN 4 (1,364 KB). Groups: the 9
pieces of 32 experts (the split with the lane in ninths).

  python tt/glm/g6_experts.py build [--workers N]     (no card)
"""
from __future__ import annotations

import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_path = list(sys.path)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "flashnext")))
from g7_experts import CardExperts, Shape, build  # noqa: E402,F401  (CardExperts re-exported)
from expert_types import card_layers  # noqa: E402  (tt/formats: g7_experts put it on the path)
if __name__ != "__main__":
    sys.path[:] = _path        # tt/flashnext's g2_layers, g3_hc, g4_token, ... must not shadow tt/glm's

MODEL = os.environ.get("TT_MOEOFFLOAD_MODEL_DIR") or os.path.join(DATA, "models/GLM-5.3-Flash Maya-L")   # engine.py --model-dir sets it    # g2_layers.MODEL (which needs ttnn)
# from the GGUF's types (the validated Maya-L: 7-40 and 3-6, 41-44); the MTP's 45 (g5_mtp.MTP) stays on the lane
CARD_LAYERS = card_layers(MODEL, "IQ4_XS", exclude=(45,))
Q5K_LAYERS = card_layers(MODEL, "Q5_K", exclude=(45,))  # Q5_K downs, re-quantized to int8 (g7_experts.q5k_as_q8): those built
SHAPE = Shape(4096, 2048, 288, 8, CARD_LAYERS, MODEL, os.path.join(DATA, "glm_experts"), ng=9,
              dn_m=True, dn_q="IQ4_XS", mt=int(os.environ.get("GP_MT", 3)), mt_dn=int(os.environ.get("GP_MT_DN", 4)),
              swiglu_limit=10.0)          # llama.cpp's swiglu_clamp_exp for GLM's routed experts (the lane's too)


def built(il: int) -> bool:
    return all(os.path.exists(os.path.join(SHAPE.out_dir(), f"blk.{il}.p{j}.dn_s.npy")) for j in range(SHAPE.NPIECE))


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:     # build [--layers 3,4 | q5k] [--workers N]: the IQ4_XS-down layers by default
        if "--layers" in sys.argv:
            v = sys.argv[sys.argv.index("--layers") + 1]
            SHAPE.layers = Q5K_LAYERS if v == "q5k" else [int(x) for x in v.split(",")]
        build(int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16, SHAPE)
else:
    SHAPE.layers = sorted([*CARD_LAYERS, *(il for il in Q5K_LAYERS if built(il))])
