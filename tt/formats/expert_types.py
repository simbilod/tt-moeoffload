"""The routed experts' GGUF types per layer, from the shards' headers (no tensor data read), and what the card makes of
them: which layers it takes, in which form, and which types it cannot take. Any GGUF of the two architectures whose
experts use the types below works, not only the two validated quantizations.

    expert_types(model_dir)        # {layer: (gate, up, down)}, e.g. {0: ("IQ3_S", "IQ3_S", "IQ4_NL"), ...}
"""
from __future__ import annotations

import glob
import os
import threading
from functools import lru_cache

GU_CARD = ("IQ3_S", "IQ4_XS")            # gate|up in the card form (g7_experts.DEFINES; IQ4_XS through IQ4_NL's table)
DN_CARD = {"flashnext": ("IQ4_NL", "Q8_0")}   # Flash-Next's down (Shape dn_q, and Q8_0's dn_v); GLM's: g6_experts
WHICH = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


_lock = threading.Lock()                 # callers in a thread pool (g7_experts' Layer) wait for one read, not 48


def expert_types(model_dir: str) -> dict[int, tuple[str, str, str]]:
    """{layer: (gate, up, down) type names} for every layer with routed experts, in layer order ({}: no GGUF there)"""
    with _lock:
        return _read(model_dir)


@lru_cache
def _read(model_dir: str) -> dict[int, tuple[str, str, str]]:
    from gguf import GGUFReader
    seen: dict[int, dict[str, str]] = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.gguf"))):
        for t in GGUFReader(f).tensors:
            p = t.name.split(".")
            if len(p) == 4 and p[0] == "blk" and p[2] in WHICH:
                seen.setdefault(int(p[1]), {})[p[2]] = t.tensor_type.name
    return {il: tuple(d.get(w, "missing") for w in WHICH) for il, d in sorted(seen.items())}


def card_layers(model_dir: str, down: str, exclude=()) -> list[int]:
    """the layers whose down is `down` and whose gate and up are one type the card form decodes"""
    return [il for il, (g, u, d) in expert_types(model_dir).items()
            if d == down and g == u and g in GU_CARD and il not in exclude]


def card_refusals(model_dir: str) -> list[str]:
    """Flash-Next streams every layer's experts to the card: each layer whose types its kernels lack, as a problem"""
    dn = DN_CARD["flashnext"]
    return [f"layer {il}'s routed experts are gate {g} / up {u} / down {d}: the card's kernels take gate and up as one "
            f"of {', '.join(GU_CARD)} and down as {' or '.join(dn)} (choose a GGUF quantized that way)"
            for il, (g, u, d) in expert_types(model_dir).items() if not (g == u and g in GU_CARD and d in dn)]
