from tt_moeoffload.calibrate import fit

SIZES = (256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
LINES = {"cpu": lambda k: 0.17 + 0.0018 * k, "split1": lambda k: max(1.25, 0.38 + 0.00124 * k),   # today's, with the
         "split2": lambda k: max(2.6, 1.04 + 0.00086 * k), "card": lambda k: 4.4 + k / 3000}       # card shares' floors


def measured(lines, cpu_max=3072):
    return {"prefill_s": {p: {str(n): f(n) for n in SIZES if p != "cpu" or n <= cpu_max} for p, f in lines.items()},
            "step_s": 0.0095, "reset_s": 2.2}


def test_todays_lines():
    t = fit(measured(LINES), 0.0)
    assert 512 <= t["CPU_PREFILL_MAX"] <= 1024
    assert [g for _, g, _, _ in t["SPLIT"]] == [1, 2] and t["SPLIT"][0][0] < t["SPLIT"][1][0] <= 8192
    assert abs(t["PREFILL_S"] - 4.4) < 0.01 and abs(t["PREFILL_TOK_S"] - 3000) < 30
    assert abs(t["CPU_PREFILL_TOK_S"] - 0.0018) < 1e-5 and t["SpecModel.step_s"] == 0.0095


def test_a_slow_lane_hands_short_prompts_to_the_card():
    slow = dict(LINES, cpu=lambda k: 0.5 + 0.0018 * 5 * k, split1=lambda k: max(1.25, 0.38 + 0.004 * k),
                split2=lambda k: max(2.6, 1.04 + 0.003 * k))
    t = fit(measured(slow), 0.0)
    assert t["CPU_PREFILL_MAX"] < 512 and all(b <= 3072 for b, _, _, _ in t["SPLIT"])
