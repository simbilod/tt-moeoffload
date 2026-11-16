"""The host's cost model. The engine's modules hold their cost constants' defaults, measured on the development host
(a p150a at PCIe Gen4 x16); $TT_MOEOFFLOAD_DATA/tune.json, {name: value}, replaces any of them for another
host (calibrate writes it). No file: the defaults.

    PREFILL_S = tuned("PREFILL_S", 4.4)
"""
from __future__ import annotations

import json
import os

PATH = os.path.join(os.environ.get("TT_MOEOFFLOAD_DATA") or os.path.expanduser("~/.cache/tt-moeoffload"), "tune.json")
try:
    with open(PATH) as f:
        TUNE = json.load(f)
except FileNotFoundError:
    TUNE = {}
if not isinstance(TUNE, dict):
    raise ValueError(f"{PATH}: not a JSON object of name: value")


def _shape(v):
    """a value's form: numbers are one kind (an int default may be tuned to a float), lists element by element"""
    if isinstance(v, bool):
        return bool
    if isinstance(v, (int, float)):
        return float
    if isinstance(v, (list, tuple)):
        return [_shape(x) for x in v]
    return type(v)


def tuned(name: str, default):
    """tune.json's value for name when it has one (in the default's form: a number, or a list of rows of numbers of
    the default's row length), else the default. Lists come back as lists of tuples, as the defaults are written."""
    if name not in TUNE:
        return default
    v = TUNE[name]
    ok = _shape(v) == _shape(default) if not isinstance(default, list) else (
        isinstance(v, list) and all(_shape(r) == _shape(default[0]) for r in v))   # [] allowed: no split
    if not ok:
        raise ValueError(f"{PATH}: {name} = {v!r} is not in the form of its default {default!r}")
    return [tuple(r) for r in v] if isinstance(default, list) else v
