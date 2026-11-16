"""GLM's check: tt/serve/engine.py --serve --model glm driven over its line protocol (what an API server sends), inside the
tt-metal container:
  a  glm_ref's first 16 tokens, greedy 64: against g5_spec.py's greedy tokens of the same prompt (the decoder's own
     4-row passes; equal up to near-ties: the prefill's rounding is not the passes')
  b  a follow-up: (a)'s prompt + its reply + 8 more of glm_ref's tokens (RESUME: the held state continued)
  c  glm_ref_long's first 4,487 tokens, greedy 32: the first token against llama.cpp's argmax at position 4,486, the
     time to the first token and the decode rate
  d  STOP mid-reply
  e  glm_ref_long's first 2,500 tokens, then its first 4,487: the prefill continues its last read from 2,499 (the
     first token against llama.cpp's argmax, the tokens against (c)'s read from 0: equal up to near-ties)
  f  a follow-up of (e): its prompt + its reply + the rest of glm_ref_long (255 tokens; RESUME, the read's time)

  python3 tt_moeoffload/launch.py --cpus all --memory-max $((320 << 30)) --engine tt/glm/g7_check.py
"""
from __future__ import annotations

import json
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")
REF, REF_LONG = os.path.join(DATA, "glm_ref"), os.path.join(DATA, "glm_ref_long")
VOCAB = 154880
CTX = int(os.environ.get("GC_CTX", 32768))


class Proc:
    def __init__(self, args):
        self.p = subprocess.Popen([sys.executable, os.path.join(TT, "serve", "engine.py"), "--serve", *args],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lines = []
        self.cv = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            with self.cv:
                self.lines.append(line.rstrip("\n"))
                self.cv.notify_all()

    def wait(self, pred, timeout=3600):
        t0 = time.time()
        with self.cv:
            while True:
                for i, l in enumerate(self.lines):
                    if pred(l):
                        got, self.lines = self.lines[:i + 1], self.lines[i + 1:]
                        return got
                if time.time() - t0 > timeout or self.p.poll() is not None:
                    raise RuntimeError(f"no answer: {self.lines[-5:]}")
                self.cv.wait(1.0)

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def gen(self, ids, max_new, keys="", stop_after=None):
        t0 = time.perf_counter()
        self.send(f"GEN {max_new} {keys} {','.join(map(str, ids))}".replace("  ", " "))
        if stop_after:
            got = self.wait(lambda l: l.startswith("T "))
            time.sleep(stop_after)
            self.send("STOP")
            got += self.wait(lambda l: l.startswith(("DONE", "ERR")))
        else:
            got = self.wait(lambda l: l.startswith(("DONE", "ERR")))
        wall = time.perf_counter() - t0
        toks = [int(l.split()[1]) for l in got if l.startswith("T ")]
        done = next(l for l in got if l.startswith(("DONE", "ERR")))
        resume = [int(l.split()[1]) for l in got if l.startswith("RESUME")]
        f = done.split()
        r = {"tokens": toks, "done": done, "wall_s": round(wall, 2), "resume": resume[0] if resume else 0}
        if f[0] == "DONE":
            r.update(prompt_ms=float(f[3]), decode_ms=float(f[4]), finish=f[5],
                     decode_tok_s=round((len(toks) - 1) / max(float(f[4]), 1e-3) * 1e3, 1) if len(toks) > 1 else None)
        return r


def main() -> None:
    ids = [int(x) for x in np.fromfile(os.path.join(REF, "ids.i32"), np.int32)]
    long_ids = [int(x) for x in np.fromfile(os.path.join(REF_LONG, "ids.i32"), np.int32)]
    ref_long = np.memmap(os.path.join(REF_LONG, "logits.f32"), np.float32, "r").reshape(-1, VOCAB)
    spec = json.load(open(os.path.join(OUT, "g5_spec.json")))
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "ctx": CTX}
    t0 = time.time()
    e = Proc(["--model", "glm", "--spec-rows", os.environ.get("GC_ROWS", "2"), "--max-context", str(CTX)])
    ready = e.wait(lambda l: l.startswith("READY"), timeout=3600)
    res["ready_s"] = round(time.time() - t0, 1)
    print("ready:", ready[-1], f"after {res['ready_s']} s", flush=True)
    # a
    a = e.gen(ids[:16], 64)
    want = spec["greedy_tokens"]
    same = next((i for i, (x, y) in enumerate(zip(a["tokens"], want)) if x != y), min(len(a["tokens"]), len(want)))
    sm2 = spec.get("spec_m2_tokens", [])
    same2 = next((i for i, (x, y) in enumerate(zip(a["tokens"], sm2)) if x != y), min(len(a["tokens"]), len(sm2)))
    res["a"] = {**{k: v for k, v in a.items() if k != "tokens"}, "equal_to_passes_greedy": same, "of": len(want),
                "equal_to_spec_m2": same2, "tokens": a["tokens"]}
    print("a:", json.dumps(res["a"]), flush=True)
    # b
    b_ids = ids[:16] + a["tokens"] + ids[16:24]
    b = e.gen(b_ids, 16)
    res["b"] = {k: v for k, v in b.items() if k != "tokens"}
    print("b:", json.dumps(res["b"]), flush=True)
    # c
    c = e.gen(long_ids[:4487], 32)
    c_tokens = c["tokens"]
    first_ok = bool(c["tokens"] and c["tokens"][0] == int(np.argmax(ref_long[0])))
    res["c"] = {**{k: v for k, v in c.items() if k != "tokens"}, "first_token_is_llama_argmax": first_ok}
    print("c:", json.dumps(res["c"]), flush=True)
    # d
    d = e.gen(ids[:40], 400, stop_after=1.0)
    res["d"] = {k: v for k, v in d.items() if k != "tokens"} | {"generated": len(d["tokens"])}
    print("d:", json.dumps(res["d"]), flush=True)
    # e
    e.gen(long_ids[:2500], 8)
    x = e.gen(long_ids[:4487], 32)
    same = next((i for i, (u, v) in enumerate(zip(x["tokens"], c_tokens)) if u != v), min(len(x["tokens"]), len(c_tokens)))
    res["e"] = {**{k: v for k, v in x.items() if k != "tokens"}, "equal_to_c": same, "of": len(c_tokens),
                "first_token_is_llama_argmax": bool(x["tokens"] and x["tokens"][0] == int(np.argmax(ref_long[0])))}
    print("e:", json.dumps(res["e"]), flush=True)
    # f
    f = e.gen(long_ids[:4487] + x["tokens"] + long_ids[4487:], 16)
    res["f"] = {k: v for k, v in f.items() if k != "tokens"}
    print("f:", json.dumps(res["f"]), flush=True)
    res["host_peak_rss_gb"] = round(next(int(x.split()[1]) for x in open(f"/proc/{e.p.pid}/status")
                                         if x.startswith("VmHWM")) / 1e6, 1)   # the engine's peak (preflight's RAM need)
    e.send("QUIT")
    e.p.wait(timeout=120)
    os.makedirs(os.path.join(DATA, "results"), exist_ok=True)   # this run's output; g5_spec.json stays the reference
    json.dump(res, open(os.path.join(DATA, "results", "g7_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
