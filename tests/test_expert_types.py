"""C: what the card does with each layer's experts follows from the GGUF's types. The validated GGUFs (when the data root
has them) give the constants the engine had written in before."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tt" / "formats"), str(ROOT / "third_party" / "llama.cpp" / "gguf-py")]
import expert_types as et  # noqa: E402

DATA = Path(os.environ.get("TT_MOEOFFLOAD_DATA") or Path.home() / ".cache" / "tt-moeoffload")
FLASHNEXT = DATA / "models/qwen3.8-flash-next-ud-iq4xs/UD-IQ4_XS"
GLM = DATA / "models/GLM-5.3-Flash Maya-L"


@pytest.mark.skipif(not list(FLASHNEXT.glob("*.gguf")), reason=f"no Flash-Next GGUF in {FLASHNEXT}")
def test_flashnext():
    t = et.expert_types(str(FLASHNEXT))
    assert [il for il, (_, _, d) in t.items() if d == "Q8_0"] == [2, 4, 30, 46, 47]   # g7_prefill's RESIDENT
    assert {il: g for il, (g, _, _) in t.items() if g != "IQ3_S"} == {2: "IQ4_XS"}      # FLASHNEXT's gu_q
    assert len(t) == 48 and et.card_refusals(str(FLASHNEXT)) == []


@pytest.mark.skipif(not list(GLM.glob("*.gguf")), reason=f"no GLM GGUF in {GLM}")
def test_glm():
    assert et.card_layers(str(GLM), "IQ4_XS", exclude=(45,)) == list(range(7, 41))
    assert et.card_layers(str(GLM), "Q5_K", exclude=(45,)) == [3, 4, 5, 6, 41, 42, 43, 44]


def test_refusals(monkeypatch):
    monkeypatch.setattr(et, "expert_types", lambda d: {0: ("IQ3_S", "IQ3_S", "IQ4_NL"), 1: ("Q4_K", "Q4_K", "Q8_0"),
                                                        2: ("IQ3_S", "IQ4_XS", "Q8_0"), 3: ("IQ4_XS", "IQ4_XS", "Q6_K")})
    found = et.card_refusals("x")
    assert [s.split("'")[0] for s in found] == ["layer 1", "layer 2", "layer 3"]
