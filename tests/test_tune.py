import importlib.util
import json
from pathlib import Path

import pytest

TUNE = Path(__file__).resolve().parents[1] / "tt" / "tune.py"


def load(tmp_path, monkeypatch, values=None):
    monkeypatch.setenv("TT_MOEOFFLOAD_DATA", str(tmp_path))
    if values is not None:
        (tmp_path / "tune.json").write_text(json.dumps(values))
    spec = importlib.util.spec_from_file_location("tune_under_test", TUNE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_file_keeps_the_defaults(tmp_path, monkeypatch):
    t = load(tmp_path, monkeypatch)
    split = [(2304, 1, 0.38, 0.00124)]
    assert t.tuned("PREFILL_S", 4.4) == 4.4 and t.tuned("SPLIT", split) is split


def test_values_and_their_form(tmp_path, monkeypatch):
    t = load(tmp_path, monkeypatch, {"PREFILL_S": 5, "SPLIT": [[1000, 1, 0.5, 0.001]], "GLM_SPLIT": [[800, 0], [1e999, 7]],
                                     "STEP_S": "fast", "BAD_SPLIT": [[1, 2]]})
    assert t.tuned("PREFILL_S", 4.4) == 5
    assert t.tuned("SPLIT", [(2304, 1, 0.38, 0.00124)]) == [(1000, 1, 0.5, 0.001)]
    assert t.tuned("GLM_SPLIT", [(800, 0), (float("inf"), 7)])[-1] == (float("inf"), 7)
    with pytest.raises(ValueError):
        t.tuned("STEP_S", 0.023)
    with pytest.raises(ValueError):
        t.tuned("BAD_SPLIT", [(2304, 1, 0.38, 0.00124)])
