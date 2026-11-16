#!/usr/bin/env python3
"""Everything the engine and its checks read, made from nothing into the data root ($TT_MOEOFFLOAD_DATA, default
~/.cache/tt-moeoffload). Resumable: a step whose outputs are there is skipped; each says what it makes and why.

  python3 tt_moeoffload/prepare.py flashnext|glm [--checks] [--from DIR]

  flashnext  the GGUF (pins.json, sha256 checked), the API server's tokenizer, the MTP draft layer's runtime files (Strata's
             tools: range reads of the BF16 checkpoint), the card expert cache, the routing census
  glm        the GGUF, the API server's tokenizer, GLM's card expert cache
  --checks   also the reference logits the checks compare with (llama.cpp's, tools/llama/dump-logits)
  --from DIR the model's files are already in DIR (its layout under models/<name>/): checked, then linked, not fetched

Run tt/env/setup.sh first (the checkouts and builds). The card expert caches are built in the pinned container
(tt_moeoffload/launch.py); the rest runs on the host (python3 with numpy, jinja2, regex).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tt_moeoffload.launch import data_root  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PINS = json.loads((REPO / "pins.json").read_text())
TOOLS = REPO / "tools" / "llama" / "build"
STOOLS = REPO / "third_party" / "strata" / "tools"
HF = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
GB = 1e9
CENSUS_MODEL = "qwen3.8-flash-next"
# the reference logits the checks read: (folder, model, prompt, --dump-from, dump-logits args); the prompts are this
# repo's (tools/llama/prompts) and llama.cpp's own docs at its pin (the long ones)
LLAMA_DOCS = REPO / "third_party" / "llama.cpp"
REFS = {
    "flashnext": [("g4_ref", "short", 0, ["-c", "512", "-b", "512", "-ub", "512"]),
                  ("g4_ref_long", "long", 4775, ["-c", "8192", "-b", "2048", "-ub", "2048"])],
    "glm": [("glm_ref", "short", 0, ["-c", "512", "-b", "512", "-ub", "512"]),
            ("glm_ref_long", "long", 4486, ["-c", "8192", "-b", "2048", "-ub", "2048"])],
}


def say(step: str, what: str) -> None:
    print(f"\n== {step}: {what}", flush=True)


def need_disk(path: Path, nbytes: float, what: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < nbytes:
        sys.exit(f"{what} needs {nbytes / GB:.1f} GB in {path}, {free / GB:.1f} GB are free")


def run(argv: list, **kw) -> None:
    print("  $ " + " ".join(str(a) for a in argv), flush=True)
    r = subprocess.run([str(a) for a in argv], **kw)
    if r.returncode:
        sys.exit(f"failed ({r.returncode}): {argv[0]}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def verified(path: Path, want: str, size: int) -> bool:
    """The file's sha256 against the pin, remembered by size and mtime in models/.verified.json (88-156 GB: hashed once)."""
    if not path.is_file() or path.stat().st_size != size:
        return False
    book_path = DATA / "models" / ".verified.json"
    book = json.loads(book_path.read_text()) if book_path.exists() else {}
    key, st = str(path.relative_to(DATA / "models")), path.stat()
    if book.get(key) == [want, st.st_size, st.st_mtime_ns]:
        return True
    print(f"  sha256 {key} ...", flush=True)
    if sha256(path) != want:
        return False
    book = json.loads(book_path.read_text()) if book_path.exists() else {}   # another prepare may have added to it
    book[key] = [want, st.st_size, st.st_mtime_ns]
    tmp = book_path.with_name(f".verified.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(book, indent=1))
    tmp.replace(book_path)                    # whole or not at all: a reader never sees half a file
    return True


def download(url: str, dest: Path, size: int, want: str) -> None:
    """A resumable download into dest.part (a range request after an interruption), sha256 checked, then renamed."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:                           # not a prefix of this file: start over
        part.unlink()
        have = 0
    need_disk(dest.parent, size - have, dest.name)
    if have < size:                           # a whole .part needs no request (a range past the end is refused)
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
            if have and r.status != 206:
                sys.exit(f"{url}: the server ignored the range request (status {r.status}); delete {part} to start over")
            done = have
            while b := r.read(1 << 24):
                f.write(b)
                done += len(b)
                print(f"\r  {dest.name}: {done / GB:.1f} of {size / GB:.1f} GB", end="", flush=True)
        print()
    if part.stat().st_size != size or sha256(part) != want:
        part.unlink()
        sys.exit(f"{dest.name}: the download's size or sha256 is not the pinned one (deleted; run again)")
    part.rename(dest)


def model(name: str, from_dir: Path | None) -> Path:
    pin = PINS["models"][name]
    files = pin["files"]
    top = {f["to"].split("/")[0] for f in files.values()}.pop()
    root = DATA / "models" / top
    say("model", f"{pin['repo']} at {pin['revision'][:12]} ({len(files)} files, "
                 f"{sum(f['size'] for f in files.values()) / GB:.1f} GB) -> {root}: the weights every other step reads")
    if from_dir is not None and root.exists():
        print(f"  {root} is already there: --from {from_dir} not used")
    elif from_dir is not None:
        # DIR laid out as models/<top>/ is, or as the Hugging Face repo is (they differ for GLM: its shards sit in the
        # repo's Maya-L/ folder)
        rel = {p: f["to"].split("/", 1)[1] for p, f in files.items()}
        bases = [from_dir] + [from_dir / p[:len(p) - len(rel[p])] for p in files if p.endswith(rel[p])][:1]
        base = next((b for b in bases if all((b / r).is_file() for r in rel.values())), None)
        if base is None:
            missing = [r for r in rel.values() if not (from_dir / r).is_file()]
            sys.exit(f"--from {from_dir}: {len(missing)} pinned files are not there ({missing[0]} ...)")
        root.parent.mkdir(parents=True, exist_ok=True)
        root.symlink_to(base.resolve(), target_is_directory=True)   # the launcher mounts where it leads
        print(f"  linked {root} -> {base.resolve()}")
    for path, f in sorted(files.items()):
        dest = DATA / "models" / f["to"]
        if verified(dest, f["sha256"], f["size"]):
            continue
        if dest.exists():
            sys.exit(f"{dest} is not the pinned file (size or sha256): move it away")
        dest.parent.mkdir(parents=True, exist_ok=True)
        download(f"{HF}/{pin['repo']}/resolve/{pin['revision']}/{path}", dest, f["size"], f["sha256"])
        verified(dest, f["sha256"], f["size"])
    return DATA / "models" / sorted(f["to"] for f in files.values())[0]   # the first shard


def built(name: str) -> None:
    lane = "libcpulane_glm.so" if name == "glm" else "libcpulane.so"
    need = [TOOLS / t for t in ("routing-census", "dump-logits", "llama-tokenize")] + \
           [REPO / "tt" / "cpulane" / "build" / lane, REPO / "tt" / "spec" / "build" / "libstrata_spec.so",
            REPO / "third_party" / "strata" / "serve" / "frontend.py", STOOLS / "mtp_fetch.py",
            REPO / "third_party" / "maya" / "tools" / "strata_tokenizer.py"]
    missing = [str(p.relative_to(REPO)) for p in need if not p.exists()]
    if missing:
        sys.exit(f"not built or checked out: {', '.join(missing)}: run tt/env/setup.sh")
    try:
        import jinja2, numpy, regex  # noqa: F401,E401
    except ImportError as e:
        sys.exit(f"this host's python3 lacks {e.name}: pip install numpy jinja2 regex")


def host_env() -> dict:
    return {**os.environ, "STRATA_GGUF_PY": str(REPO / "third_party" / "llama.cpp" / "gguf-py")}


def tokenizer(name: str, gguf: Path) -> None:
    """The API server's tokenizer (--tokenizer / a config's "tokenizer"), by its own project's tool: Strata's for
    Flash-Next, project-maya's for GLM."""
    out = DATA / "tokenizers" / name
    say("tokenizer", f"{out}/tokenizer: the API server's tokenizer and chat template, read from the GGUF")
    if (out / "tokenizer" / "chat_template.jinja").exists():
        return print("  there")
    tool = (REPO / "third_party" / "maya" / "tools" if name == "glm" else STOOLS) / "strata_tokenizer.py"
    run([sys.executable, "-I", tool, "--gguf", gguf, "--out", out], env=host_env())


def mtp(_gguf: Path) -> None:
    d = DATA / "strata" / "mtp"
    rt = d / "rt"
    say("mtp", f"{rt}: the MTP draft layer (speculation's drafts), from Qwen/Qwen3.8-Flash-Next's BF16 checkpoint: "
               f"range reads of its 31 mtp.* tensors (~5 GB), packed (Q2_0 experts) and laid out for the engine")
    if all((rt / f).exists() for f in ("dense.txt", "dense.bin", "experts.bin")):
        return print("  there")
    need_disk(d, 7 * GB, "the MTP files")
    env = host_env()
    run([sys.executable, "-I", STOOLS / "mtp_fetch.py", "fetch", "--out", d], env=env)   # resumable, sha256 checked
    run([sys.executable, "-I", STOOLS / "mtp_pack.py", "--src", d, "--experts", "q2_0", "--out", d / "mtp-q2_0.gguf"],
        env=env)
    tmp = d / "rt.tmp"
    run([sys.executable, "-I", STOOLS / "mtp_rt.py", "--gguf", d / "mtp-q2_0.gguf", "--out", tmp], env=env)
    tmp.rename(rt)


def card_cache(name: str) -> None:
    script = "tt/glm/g6_experts.py" if name == "glm" else "tt/flashnext/g7_experts.py"
    say("card experts", f"the routed experts in the card's tile form ({script} build: ~"
                        f"{165 if name == 'glm' else 70} GB under {DATA}): the prefill streams them to the card; built "
                        "in the pinned container (no card used), resumable: it skips the pieces there and stops near a "
                        "full disk")
    # GLM: the IQ4_XS-down layers, then the Q5_K-down trunk layers (the engine uses those built; validated with all)
    for extra in ([[], ["--layers", "q5k"]] if name == "glm" else [[]]):
        run([sys.executable, REPO / "tt_moeoffload" / "launch.py", "--engine", REPO / script, "--name",
             f"tt-moeoffload-prepare-{name}", "--", "build", *extra])


def physical_cores() -> int:
    """This host's physical cores (SMT siblings counted once), from /sys."""
    cores = set()
    for c in os.sched_getaffinity(0):
        t = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
        cores.add(((t / "physical_package_id").read_text().strip(), (t / "core_id").read_text().strip()))
    return len(cores)


def census(gguf: Path) -> None:
    d = DATA / "census" / CENSUS_MODEL
    say("census", f"{d}: which routed experts each layer picks over a fixed corpus (tools/llama/corpus.py, then "
                  "routing-census, ~15 min with a GPU): the card cache's slots and the lane's warm follow it; the "
                  "served check reads its prompts")
    if all((d / f).exists() for f in ("census.json", "census.trace", "ids")):
        return print("  there")
    prompts = d / "prompts.jsonl"
    run([sys.executable, REPO / "tools" / "llama" / "corpus.py", gguf, prompts])
    want = PINS["census_corpus"]["sha256"]
    if sha256(prompts) != want:
        sys.exit(f"{prompts}'s sha256 is not pins.json's census_corpus ({want}): are third_party/'s checkouts at "
                 "their pins (tt/env/setup.sh)? Its jinja2 renders differently?")
    tmp = d / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "ids").mkdir(parents=True)
    n = str(physical_cores())
    run([TOOLS / "routing-census", "-m", gguf, "-ngl", "99", "-ot", "exps=CPU", "-c", "131072", "-t", n, "-tb", n,
         "-ub", "4096", "-b", "8192", "--load-mode", "mmap", "--lazy-mode", "on", "--fit", "off",
         "--census-in", prompts, "--census-out", tmp / "census.json", "--census-trace", tmp / "census.trace",
         "--census-ids", tmp / "ids", "--census-decode", "256"],
        env={**os.environ, "GGML_OP_OFFLOAD_MIN_BATCH": "2048"}, stdout=open(d / "census.log", "w"),
        stderr=subprocess.STDOUT)
    for f in ("census.trace", "ids", "census.json"):    # census.json last: the step's done mark
        (tmp / f).rename(d / f)
    tmp.rmdir()


def prompt_text(kind: str) -> str:
    if kind == "short":
        return (REPO / "tools" / "llama" / "prompts" / "lighthouse.txt").read_text()
    readme, build = (LLAMA_DOCS / "README.md").read_bytes(), (LLAMA_DOCS / "docs" / "build.md").read_bytes()
    return (readme + build[:9000]).decode("utf-8")


def refs(name: str, gguf: Path) -> None:
    for folder, kind, start, args in REFS[name]:
        d = DATA / folder
        say(f"reference {folder}", f"{d}: llama.cpp's logits of the {kind} prompt"
                                   f"{f' from position {start}' if start else ''}, which the checks compare with")
        if (d / "logits.f32").exists():
            print("  there")
            continue
        d.mkdir(parents=True, exist_ok=True)
        (d / "prompt.txt").write_text(prompt_text(kind))
        out = subprocess.run([TOOLS / "llama-tokenize", "-m", gguf, "-f", d / "prompt.txt", "--ids", "--no-bos",
                              "--log-disable"], capture_output=True, text=True)
        if out.returncode:
            sys.exit(f"llama-tokenize failed:\n{out.stderr[-2000:]}")
        import array
        array.array("i", json.loads(out.stdout.strip().splitlines()[-1])).tofile(open(d / "ids.i32", "wb"))
        run([TOOLS / "dump-logits", "-m", gguf, "-ngl", "99", "-ot", "exps=CPU", *args, "-t", str(physical_cores()), "-tb", str(physical_cores()),
             "--dump-ids", d / "ids.i32", "--dump-out", d / "logits.tmp"] + (["--dump-from", start] if start else []),
            stdout=open(d / "dump.log", "w"), stderr=subprocess.STDOUT)
        (d / "logits.tmp").rename(d / "logits.f32")


def main(argv=None) -> int:
    global DATA
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", choices=("flashnext", "glm"))
    ap.add_argument("--checks", action="store_true", help="also the reference logits the checks read")
    ap.add_argument("--from", dest="from_dir", type=Path, help="the model's files are already in this folder")
    a = ap.parse_args(argv)
    DATA = data_root()
    print(f"data root {DATA}")
    built(a.model)
    gguf = model(a.model, a.from_dir)
    tokenizer(a.model, gguf)
    if a.model == "flashnext":
        mtp(gguf)
        census(gguf)
    card_cache(a.model)
    if a.checks:
        refs(a.model, gguf)
    print(f"\nprepared {a.model} in {DATA}")
    return 0


DATA = data_root()
if __name__ == "__main__":
    sys.exit(main())
