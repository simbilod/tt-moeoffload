"""The engine's line protocol, sampler, prefill choice, speculation and checkpoints, on its fake model (no card, no
numpy): engine.py --serve --fake as a subprocess of the host's Python, and its classes imported."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "tt" / "serve" / "engine.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eng = _module(ENGINE, "ttcard_engine")


class Fake:
    """The line protocol of engine.py --serve --fake as a subprocess (the host's Python, no container)."""

    def __init__(self, *args):
        self.p = subprocess.Popen([sys.executable, str(ENGINE), "--serve", "--fake", *args], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.info, self.ready = self.line(), self.line()

    def line(self) -> str:
        return self.p.stdout.readline().strip()

    def send(self, s: str) -> None:
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def gen(self, head: str, ids) -> tuple[list[str], list[int], list[str]]:
        """(the lines before DONE, the generated ids, DONE's fields)."""
        self.send(f"{head} {','.join(map(str, ids))}")
        lines = []
        while True:
            ln = self.line()
            if ln.startswith(("DONE", "ERR")) or not ln:
                return lines, [int(x[2:]) for x in lines if x.startswith("T ")], ln.split()
            lines.append(ln)

    def close(self) -> int:
        self.send("QUIT")
        return self.p.wait(timeout=30)


@pytest.fixture
def fake():
    f = Fake("--fake-vocab", "500")
    yield f
    if f.p.poll() is None:
        f.close()


def test_ready_and_greedy_is_deterministic(fake):
    assert fake.info.startswith("INFO engine=ttcard") and "logit_bias=1" in fake.info
    assert fake.ready == "READY 16384 stop"
    _, a, done = fake.gen("GEN 6", [1, 2, 3])
    assert len(a) == 6 and done[0] == "DONE" and done[1] == "6" and done[2] == "3" and done[5] == "length"
    assert done[14] == "3"                               # prompt_read: all of it
    _, b, _ = fake.gen("GEN 6", [1, 2, 3])               # a shorter prompt than the state: reset, the same answer
    assert a == b
    assert fake.close() == 0


def test_a_prompt_that_extends_the_state_resumes(fake):
    _, a, _ = fake.gen("GEN 4", [5, 6, 7])
    held = [5, 6, 7] + a[:-1]                            # the last generated token is not fed back
    lines, b, done = fake.gen("GEN 3", held + [a[-1], 9, 9])
    assert lines[0] == f"RESUME {len(held)}"
    assert done[8] == str(len(held)) and done[14] == "3"  # reused, prompt_read
    _, c, done = fake.gen("GEN 3", [5, 6, 7] + a[:-1] + [a[-1], 9, 9])
    assert done[8] == "0" and c == b                     # a different turn since: read again from 0, same tokens


def test_sampling_is_seeded_and_bias_forces(fake):
    _, a, _ = fake.gen("GEN 8 temperature=1.0 top_p=0.95 seed=1234", [1, 2])
    _, b, _ = fake.gen("GEN 8 temperature=1.0 top_p=0.95 seed=1234", [1, 2])
    assert a == b
    _, c, _ = fake.gen("GEN 3 logit_bias=7:100", [1, 2])
    assert c == [7, 7, 7]


def test_refusals_leave_the_engine_serving(fake):
    _, _, err = fake.gen("GEN 20000", [1])
    assert err[0] == "ERR" and "exceeds the context" in " ".join(err)
    _, _, err = fake.gen("GEN 2", [1, 900])
    assert err[0] == "ERR" and "vocabulary" in " ".join(err)
    fake.send("HELLO")
    assert fake.line().startswith("ERR unsupported")
    _, a, done = fake.gen("GEN 2", [1])
    assert len(a) == 2 and done[0] == "DONE"


def test_stop_ends_a_request():
    f = Fake("--fake-vocab", "500", "--fake-ms", "20")
    f.send("GEN 500 1,2,3")
    while not f.line().startswith("T "):
        pass
    f.send("STOP")
    while True:
        ln = f.line()
        if ln.startswith("DONE"):
            break
    done = ln.split()
    assert done[5] == "stop" and int(done[1]) < 500
    assert f.close() == 0


def test_end_of_turn_token_ends_with_stop():
    f = Fake()                                           # the real vocabulary: <|im_end|> is 248046
    _, ids, done = f.gen("GEN 5 logit_bias=248046:100", [1, 2])
    assert ids == [248046] and done[5] == "stop" and done[1] == "1"
    assert f.close() == 0


def test_sampler_rules():
    L = eng.ListLogits
    logits = [0.0] * 10
    logits[3], logits[5] = 5.0, 4.9
    assert eng.Sampler({}).pick(L(list(logits)), []) == 3
    assert eng.Sampler({"temperature": "1.0", "top_k": "1"}).pick(L(list(logits)), []) == 3
    # a repeat penalty pushes the repeated token below the next one; the logits are not changed in place
    v = L(list(logits))
    assert eng.Sampler({"penalty_repeat": "1.5", "penalty_last_n": "8"}).pick(v, [3]) == 5 and v.get(3) == 5.0
    # min_p keeps only the near-top tokens: never the 8 zeros
    s = eng.Sampler({"temperature": "1.0", "min_p": "0.5", "seed": "7"})
    assert {s.pick(L(list(logits)), []) for _ in range(50)} <= {3, 5}
    draws = lambda seed: [eng.Sampler({"temperature": "2.0", "seed": seed}).pick(L(list(logits)), [])  # noqa: E731
                          for _ in range(3)]
    assert draws("11") == draws("11")
    assert eng.Sampler({"logit_bias": "3:-100"}).pick(L(list(logits)), []) == 5


def test_dump_writes_every_position(tmp_path):
    ids = tmp_path / "ids.i32"
    import array
    ids.write_bytes(array.array("i", [4, 8, 15, 16]).tobytes())
    out = tmp_path / "o.f32"
    subprocess.run([sys.executable, str(ENGINE), "--dump-logits", "--fake", "--fake-vocab", "100", "--ids", str(ids),
                    "--out", str(out)], check=True, capture_output=True)
    assert out.stat().st_size == 4 * 100 * 4


def test_dump_through_verify_passes(tmp_path):
    """a model without step() (GLM's): passes of its rows, every row's full logits in order, all kept"""
    import array
    ids = tmp_path / "ids.i32"
    ids.write_bytes(array.array("i", [4, 8, 15, 16, 23]).tobytes())

    class Passes:
        rows, ctx, seen = 2, 64, []

        def feed(self, toks, pos, prev):
            assert prev == [4, 8, 15, 16, 23][:pos]
            self.seen.append((pos, list(toks)))
            return [eng.Top([0], [0.0], lambda t=t: eng.ListLogits([float(t)] * 3)) for t in toks]

        def keep(self, n):
            pass

    m = Passes()
    out = tmp_path / "o.f32"
    eng.dump_logits(m, str(ids), str(out))
    assert m.seen == [(0, [4, 8]), (2, [15, 16]), (4, [23])]
    assert array.array("f", out.read_bytes()).tolist() == [float(t) for t in (4, 8, 15, 16, 23) for _ in range(3)]


def _top(v: list[float]):
    ids = sorted(range(len(v)), key=lambda i: (-v[i], i))[:64]
    return eng.Top(ids, [v[i] for i in ids], lambda: eng.ListLogits(list(v)))


@pytest.mark.parametrize("keys,falls", [
    ({}, False), ({"temperature": "0.8", "seed": "3"}, False),
    ({"temperature": "1.0", "top_p": "0.9", "min_p": "0.05", "seed": "5"}, False),
    ({"temperature": "1.0", "top_k": "20", "seed": "9"}, False),
    ({"penalty_repeat": "1.3", "penalty_present": "0.5"}, False),
    ({"temperature": "1.0", "seed": "2", "penalty_freq": "4.0", "logit_bias": "1:-100,2:-100"}, None),
    ({"logit_bias": "250:30"}, None),                                 # outside the top 64: the full logits
    ({"temperature": "1.0", "seed": "4", "penalty_repeat": "0.5"}, True),   # a reward: always the full logits
])
def test_picks_from_the_top_64_equal_the_full_logits(keys, falls):
    """The card's path (Top) gives the full logits' pick at every step; the steps that may need a token outside the
    top 64 read the full logits."""
    import random
    r = random.Random(1)
    a, b = eng.Sampler(keys), eng.Sampler(keys)
    recent = []
    for i in range(60):
        v = [r.gauss(0.0, 3.0) for _ in range(300)]
        t = a.pick(_top(v), recent)
        assert t == b.pick(eng.ListLogits(v), recent), i
        recent.append(t)
    if falls is not None:                                             # None: some steps of each
        assert (a.fallbacks >= 59) if falls else (a.fallbacks < 30)


def test_engine_on_top_64_matches_the_logits_engine():
    import queue
    out = []
    w = type("W", (), {"write": lambda self, s: out.append(s), "flush": lambda self: None})()
    toks = []
    for top in (False, True):
        e = eng.Engine(eng.FakeModel(vocab=500, top=top), w, queue.Queue())
        del out[:]
        e.generate(12, {"temperature": "1.0", "seed": "7", "penalty_repeat": "1.2"}, [1, 2, 3])
        toks.append([x for x in out if x.startswith("T ")])
    assert toks[0] == toks[1] and len(toks[0]) == 12


def test_no_request_reaches_past_the_context():
    f = Fake("--fake-vocab", "500", "--fake-ctx", "20")
    assert f.ready == "READY 20 stop"
    _, _, err = f.gen("GEN 11", list(range(10)))          # 10 + 11 > 20
    assert err[0] == "ERR"
    _, ids, done = f.gen("GEN 10", list(range(10)))       # 10 + 10 = 20: the last fed position is 18
    assert done[0] == "DONE" and len(ids) == 10
    assert f.close() == 0


def test_long_prompts_go_through_the_prefill():
    """A long prompt is prefilled (all but its last token) and gives the decode loop's tokens; a short follow-up
    resumes on the decode loop; a long one is read again from 0 by the prefill; a short prompt stays on the loop."""
    import queue
    out = []
    w = type("W", (), {"write": lambda self, s: out.append(s.strip()), "flush": lambda self: None})()
    prompt = [(7 * i) % 500 for i in range(400)]

    def run(e, ids, n=4):
        del out[:]
        e.generate(n, {}, ids)
        return [int(x[2:]) for x in out if x.startswith("T ")], out[-1].split(), out[0]
    loop = eng.Engine(eng.FakeModel(vocab=500), w, queue.Queue())
    loop.prefill = None
    m = eng.FakeModel(vocab=500)
    e = eng.Engine(m, w, queue.Queue())
    want, _, _ = run(loop, prompt)
    got, done, _ = run(e, prompt)
    assert got == want and m.prefilled == [399] and done[14] == "400" and e.held == prompt + got[:-1]
    short = e.held + [got[-1], 1, 2]
    _, done, first = run(e, short)
    assert first == f"RESUME {len(prompt) + 3}" and m.prefilled == [399] and done[8] == str(len(prompt) + 3)
    long = e.held + [3] * 400
    got, done, first = run(e, long)
    assert not first.startswith("RESUME") and m.prefilled == [399, len(long) - 1] and done[8] == "0"
    assert m.resets == 0                                 # the prefill needs no reset
    assert got == run(loop, long)[0]
    run(e, [1, 2, 3])
    assert m.prefilled == [399, len(long) - 1] and m.resets == 1
    assert not e.use_prefill(100, 0) and e.use_prefill(400, 0) and not e.use_prefill(4000, 3990)
    assert e.use_prefill(150, 0, reset=True) and not e.use_prefill(150, 0)   # a reset tips a short prompt over

    def boom(ids, start=0):
        raise RuntimeError("TT_FATAL: Out of Memory")
    e.prefill = boom                                     # a failed prefill: the decode loop reads it, same answer
    got, done, _ = run(e, prompt)
    assert got == want and done[0] == "DONE" and done[14] == "400" and m.resets == 2 and not e.dirty



def test_long_follow_ups_continue_the_held_state_through_the_prefill():
    """2026-11-06: a follow-up of many tokens on a long held context: the decode loop reads up to a multiple of 128, the
    prefill continues the held state from there (not from 0), the tokens the decode loop's; short follow-ups stay on
    the loop; a failed extension leaves the loop to read the rest, same tokens. (2026-11-10: a multiple of 4.)"""
    import queue
    out = []
    w = type("W", (), {"write": lambda self, s: out.append(s.strip()), "flush": lambda self: None})()

    def run(e, ids, n=4):
        del out[:]
        e.generate(n, {}, ids)
        return [int(x[2:]) for x in out if x.startswith("T ")], out[-1].split(), out[0]
    held = [(7 * i) % 500 for i in range(30000)]                # at 8K a prefill from 0 costs about the same
    m = eng.FakeModel(vocab=500, ctx=65536)
    e = eng.Engine(m, w, queue.Queue())
    run(e, held)
    assert m.prefilled == [29999]
    k = len(e.held)
    follow = e.held + [(3 * i) % 500 for i in range(2000)]   # the held tokens (the reply's too), then new ones
    loop = eng.Engine(eng.FakeModel(vocab=500, ctx=65536), w, queue.Queue())
    loop.prefill = None
    want = run(loop, follow)[0]
    a = -(-k // eng.EXTEND_ALIGN) * eng.EXTEND_ALIGN
    assert e.extend_at(len(follow), k) == a and a % 4 == 0 and a - k < 4
    got, done, first = run(e, follow)
    assert first == f"RESUME {k}" and m.prefilled == [29999, (a, len(follow) - 1)] and m.resets == 0
    assert got == want and e.held == follow + got[:-1]
    assert e.extend_at(len(e.held) + 40, len(e.held)) is None              # short: the loop
    assert eng.Engine(eng.FakeModel(vocab=500, ctx=65536), w, queue.Queue()).extend_at(32000, 0) is None

    def boom(ids, start=0):
        raise RuntimeError("TT_FATAL: Out of Memory")
    m2 = eng.FakeModel(vocab=500, ctx=65536)
    e2 = eng.Engine(m2, w, queue.Queue())
    run(e2, held)
    assert e2.held == follow[:k]
    e2.prefill = boom
    assert run(e2, follow)[0] == want and not e2.dirty


# ------------------------------------------------------------------------------------------------ speculation
def _writer(out: list):
    return type("W", (), {"write": lambda self, s: out.append(s.strip()), "flush": lambda self: None})()


def _run(e, ids, n, out, keys=None):
    del out[:]
    e.generate(n, keys or {}, ids)
    return [int(x[2:]) for x in out if x.startswith("T ")], out[-1].split(), out[0]


def test_speculation_gives_the_one_row_tokens(monkeypatch):
    """Greedy passes of 4 rows (the fake's drafts: its greedy continuation, one in three wrong) give the one-row
    engine's tokens; the prompt is read in 4-row passes; a follow-up resumes; max_new and an end-of-turn token inside a
    pass cut it, the rows after them not held."""
    import queue
    out = []
    prompt = [(7 * i) % 500 for i in range(30)]
    base = eng.Engine(eng.FakeModel(vocab=500), _writer(out), queue.Queue())
    base.prefill = None
    m = eng.FakeModel(vocab=500, rows=4)
    e = eng.Engine(m, _writer(out), queue.Queue())
    e.prefill = None
    want, _, _ = _run(base, prompt, 40, out)
    got, done, _ = _run(e, prompt, 40, out)
    assert got == want and done[1] == "40" and e.held == prompt + got[:-1]
    assert m.passes[:8] == [4] * 7 + [1] and max(m.passes) == 4     # 29 prompt rows, then the first pass (no drafts)
    assert len(m.passes) - 8 < 30                                    # drafts were kept
    follow = e.held + [got[-1], 3, 4]
    got2, done, first = _run(e, follow, 7, out)
    assert first == f"RESUME {len(prompt) + 39}" and done[8] == str(len(prompt) + 39)
    assert got2 == _run(base, follow, 7, out)[0] and len(got2) == 7 and e.held == follow + got2[:-1]
    monkeypatch.setattr(eng, "EOS_IDS", (want[5],))
    got3, done, _ = _run(e, prompt, 40, out)
    assert got3 == want[:6] and done[5] == "stop" and e.held == prompt + want[:5]


def test_rejection_sampling_keeps_the_distribution():
    """Sampled passes: a draft kept with p(d), else the row's token drawn from p without it: the first token is
    distributed as pick()'s, whether the draft is likely, unlikely or cut by min_p."""
    L = eng.ListLogits
    v = [0.0] * 40
    v[3], v[5], v[7], v[9] = 3.0, 2.5, 1.0, -2.0
    keys = {"temperature": "0.8", "min_p": "0.05", "seed": "5"}
    toks, p = eng.Sampler(keys)._dist(sorted(L(v).topk(64), key=lambda iv: (-iv[1], iv[0])))
    want = {t: x / sum(p) for t, x in zip(toks, p)}
    assert 9 not in want
    for d in (3, 7, 9):                                  # likely, unlikely, outside min_p's cut
        s, n, freq = eng.Sampler(keys), 20000, {}
        for _ in range(n):
            t = s.accept([L(v), L(v)], [d], [])[0]
            freq[t] = freq.get(t, 0) + 1
        assert set(freq) <= set(want)
        assert all(abs(freq.get(t, 0) / n - q) < 0.015 for t, q in want.items()), (d, freq)
    g = eng.Sampler({})                                  # greedy: exact match
    assert g.accept([L(v)] * 3, [3, 3], []) == [3, 3, 3] and g.accept([L(v)] * 3, [5, 3], []) == [3]


def test_spec_engine_line_protocol():
    f = Fake("--spec-rows", "4")                         # the real vocabulary: <|im_end|> is 248046
    _, ids, done = f.gen("GEN 5 logit_bias=248046:100", [1, 2])
    assert ids == [248046] and done[5] == "stop"
    _, a, done = f.gen("GEN 6 temperature=1.0 seed=3", [1, 2, 3])
    _, b, _ = f.gen("GEN 6 temperature=1.0 seed=3", [1, 2, 3])
    assert a == b and len(a) == 6 and done[1] == "6"
    assert f.close() == 0


# ------------------------------------------------------------------------------------------------ prompt checkpoints
def test_prompt_checkpoints_restore_a_follow_up_that_diverges(capsys):
    """2026-11-07: the read stops at the prompt's last turn token and saves there; a later prompt that shares the prefix
    through it but not what is held (S3's turns, a regenerate) restores it, with the tokens an engine without
    checkpoints gives; another prompt read from 0 overwrites the cells, so the checkpoint is not used after it."""
    import queue
    out = []
    w = _writer(out)
    m = eng.FakeModel(vocab=500, ctx=65536, ckpt=True)
    e = eng.Engine(m, w, queue.Queue())
    e.turn = 499
    a = [(7 * i) % 499 for i in range(3000)] + [499, 11, 12]          # the last turn opens at 3000
    _run(e, a, 8, out)
    assert m.saves == [3000] and m.prefilled == [3000]                 # the prefill to it, the loop after
    b = a + [(5 * i) % 499 for i in range(20)]                         # diverges from the held reply
    got, _, first = _run(e, b, 8, out)
    assert first == "RESUME 3000" and m.restores == 1 and m.saves == [3000]
    plain = eng.Engine(eng.FakeModel(vocab=500, ctx=65536), w, queue.Queue())
    assert _run(plain, b, 8, out)[0] == got
    assert _run(e, a, 8, out)[2] == "RESUME 3000" and m.restores == 2  # a regenerate
    e.can_conv = False                     # (with snapshots the cells come back: test_a_conversation_whose_reply...)
    _run(e, [(3 * i) % 499 for i in range(3000)], 4, out)               # another prompt, from 0
    assert not _run(e, b, 8, out)[2].startswith("RESUME") and m.restores == 2
    log = capsys.readouterr().err
    assert "request: prompt 3023 = 3000 reused (checkpoint) + 23 read" in log and "checkpoint saved at 3000" in log


def test_conversation_snapshots_bring_a_conversation_back(capsys):
    """2026-11-08: a request that switches away from a held conversation of CONV_MIN tokens or more snapshots it; a
    later prompt continuing it restores the whole state (RESUME at its length) with the tokens an engine without
    snapshots gives; a short conversation is not kept; one restored and not extended is not snapshotted again."""
    import queue
    out = []
    w = _writer(out)
    m = eng.FakeModel(vocab=500, ctx=65536, ckpt=True)
    e = eng.Engine(m, w, queue.Queue())
    e.turn = 499                                                      # no turn tokens below: no checkpoints
    a = [(7 * i) % 499 for i in range(3000)]
    b = [(3 * i) % 499 for i in range(2500)]
    got_a = _run(e, a, 8, out)[0]
    held_a = len(e.held)
    got_b = _run(e, b, 8, out)[0]                                     # the switch: A kept
    assert m.saves == [("conv", held_a)]
    a2 = a + got_a + [(5 * i) % 499 for i in range(20)]
    got, _, first = _run(e, a2, 8, out)
    assert first == f"RESUME {held_a}" and m.restores == 1 and m.saves[-1] == ("conv", len(b) - 1 + 8)
    plain = eng.Engine(eng.FakeModel(vocab=500, ctx=65536), w, queue.Queue())
    _run(plain, a, 8, out)
    assert _run(plain, a2, 8, out)[0] == got
    _run(e, [1, 2, 3] * 100, 4, out)                                   # A kept again; then a short one, not kept
    n = len(m.saves)
    _run(e, b + got_b + [9, 9], 4, out)
    assert len(m.saves) == n and m.restores == 2                      # B back from its snapshot
    log = capsys.readouterr().err
    assert f"= {held_a} reused (conversation)" in log and f"conversation of {held_a} snapshotted" in log


def test_a_conversation_whose_reply_comes_back_rerendered_returns_to_its_checkpoint(capsys):
    """2026-11-08 (the comparison's replay: no snapshot ever restored): the prompt that continues a snapshotted
    conversation carries the reply as the client re-renders it, not the tokens generated, so the snapshot comes back
    through the checkpoint it carries (taken out of the checkpoint slots, which the other conversation fills) and the
    read resumes at that checkpoint, with the tokens an engine without snapshots gives. The slots are allocated at
    start and the first snapshot takes one."""
    import queue
    out = []
    w = _writer(out)
    m = eng.FakeModel(vocab=500, ctx=65536, ckpt=True)
    spares, save = [], m.save
    m.save = lambda seq, spare=None, full=False: (spares.append(spare) if full else None) or save(seq, spare, full)
    m.alloc = lambda: {"host": "pooled"}
    e = eng.Engine(m, w, queue.Queue())
    assert len(e.pool) == eng.CONV_SLOTS
    e.turn = 498
    a = [(7 * i) % 497 for i in range(3000)]
    a[-10] = 498                                                      # A's last turn opens there: its checkpoint
    q = len(a) - 10
    got_a = _run(e, a, 8, out)[0]
    b = [(3 * i) % 497 for i in range(2500)]
    for k in range(4):                                                # B's turns fill the checkpoint slots
        b = b + [498] + [(11 * i + k) % 497 for i in range(30)]
        _run(e, b, 4, out)
    assert len(e.ckpts) == eng.CKPT_SLOTS and all(c[0][:2500] == b[:2500] for c in e.ckpts)
    assert [c[2][0] for c in e.convs] == [a[:q + 1]] and spares[0] == {"host": "pooled"}
    reply = [499] * 20
    assert reply != got_a[:20]
    a2 = a + reply + [(5 * i) % 497 for i in range(20)]
    n = m.restores
    got, _, first = _run(e, a2, 8, out)
    assert first == f"RESUME {q}" and m.restores == n + 2            # the snapshot, then its checkpoint
    plain = eng.Engine(eng.FakeModel(vocab=500, ctx=65536), w, queue.Queue())
    _run(plain, a, 8, out)
    assert _run(plain, a2, 8, out)[0] == got
    _run(e, b + [1, 2], 4, out)                                       # A away again: one snapshot of A, not two
    assert sum(c[0][:q + 1] == a[:q + 1] for c in e.convs) == 1
    assert f"= {q} reused (conversation's checkpoint)" in capsys.readouterr().err


def test_a_request_that_fails_is_answered_and_the_engine_serves_on(capsys):
    """2026-11-07: an exception in a request (S3's: the draft head's index past its vocabulary) left serve(), and
    main's close() then hung on the card with the traceback unprinted: now logged, ERR, the state dirty, served on."""
    import queue

    class Boom(eng.FakeModel):
        def step(self, tok, pos, prev):
            if tok == 77 and not getattr(self, "boomed", False):
                self.boomed = True
                raise IndexError("index 46211 is out of bounds for axis 0 with size 46211")
            return super().step(tok, pos, prev)
    out, lines = [], queue.Queue()
    e = eng.Engine(Boom(vocab=500), _writer(out), lines)
    for ln in ("GEN 4 1,2,77", "GEN 4 1,2,3", None):
        lines.put(ln)
    assert e.serve() == 0
    errs = [x for x in out if x.startswith("ERR")]
    assert len(errs) == 1 and "IndexError" in errs[0] and out[-1].startswith("DONE 4")
    assert "request failed: IndexError" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ GLM
def test_engine_stops_at_the_models_own_end_of_turn_ids():
    class M(eng.FakeModel):
        eos_ids, turn = (5,), 6
    e = eng.Engine(M(vocab=64, ctx=256), None, None)
    assert e.eos == (5,) and e.turn == 6
    assert eng.Engine(eng.FakeModel(vocab=64, ctx=256), None, None).eos == eng.EOS_IDS


def test_cpu_lacks(tmp_path):
    info, rec = tmp_path / "cpuinfo", tmp_path / "lib.so.cpu"
    info.write_text("processor\t: 0\nflags\t\t: fpu avx512f avx512bw\n\nprocessor\t: 1\nflags\t\t: fpu avx512f avx512bw\n")
    rec.write_text("avx512f\nfpu\n")
    assert eng.cpu_lacks(str(rec), str(info)) == []
    rec.write_text("avx512_vnni\navx512f\nfpu\nvbmi\n")
    assert eng.cpu_lacks(str(rec), str(info)) == ["avx512_vnni", "vbmi"]


def test_bad_sampling_keys_are_bad_requests():
    for keys in ({"temperature": "abc"}, {"temperature": "nan"}, {"top_k": "x"}, {"logit_bias": "-1:100"},
                 {"logit_bias": "300:1"}, {"logit_bias": "5:inf"}):
        with pytest.raises(eng.BadRequest):
            eng.Sampler(keys, vocab=256)
    assert eng.Sampler({"logit_bias": "255:-1", "temperature": "0.7"}, vocab=256).bias == {255: -1.0}
