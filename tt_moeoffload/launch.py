#!/usr/bin/env python3
"""The engine's launcher: what a Strata-lineage API server starts as its engine (`exe` in its config; the server runs
`<exe> --serve <args>` and speaks the engine line protocol over stdin and stdout). It runs the engine in the pinned
tt-metal container (pins.json) with stdin and stdout attached, stderr to the server's log.

  launch.py [launcher options] [--] <engine args>     every argument the launcher does not know goes to the engine

Mounts, each at its own path: this repo, the data root ($TT_MOEOFFLOAD_DATA, default ~/.cache/tt-moeoffload: models, caches,
references) and where the links in its first two levels lead, each --mount DIR, and the folder of every absolute path
among the engine's arguments (so --model-dir, --ids or --out may point anywhere). Nothing else of the host: the container's HOME is $TT_MOEOFFLOAD_DATA/home (tt-metal's
kernel cache lives there). It runs as the calling user. Passed into the container: --env K=V, and the launcher's
environment variables with the prefixes the engine reads (ENV_PREFIXES: a server config's "env" block arrives that way).
A container of the same --name still running refuses the start; a stopped one left behind is removed.

On SIGTERM (the server's close) the engine gets --grace s to close the card and exit, then docker kill: the container
never outlives this process. Before a start on the card, preflight.py's checks; --fake: no card in the container (no
/dev/tenstorrent, no hugepages) and no checks, for the request path off the card.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):   # run as a file (a server's exe): import the package from this repo
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tt_moeoffload import preflight  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "tt" / "serve" / "engine.py"   # --engine: another script in the same environment (tests, the gate)
TTENV = REPO / "tt" / "env" / "ttenv.sh"   # PYTHONPATH (tt-metal model code, tt-pydeps, gguf-py, tt/), cwd
# the card and its hugepages; no --privileged (checked 2026-11-15: the device, a matmul, H2D/D2H sockets and the served
# check's prefill with pinned host memory work without it)
CARD_ARGS = ["--device=/dev/tenstorrent:/dev/tenstorrent", "--volume=/dev/hugepages-1G:/dev/hugepages-1G"]
# every prefix the engine and its checks read (git grep "environ"): TT_ covers TT_MOEOFFLOAD_, TT_METAL_, TT_CTX and
# TT_VISIBLE_DEVICES; the rest are the engine's tuning and check switches
ENV_PREFIXES = ("STRATA_", "TT_", "TTC_", "G2_", "G3_", "G4_", "G6_", "W4_", "W51_", "CL_", "GP_", "GT_", "GC_", "GS_",
                "GLM_", "MTP_", "FK_", "FMO_", "FMQ_", "FSH_", "HC_", "IDX_", "KL_", "ML_", "OV_", "DEC_", "INPLACE")


def data_root() -> Path:
    return Path(os.environ.get("TT_MOEOFFLOAD_DATA") or Path.home() / ".cache" / "tt-moeoffload").resolve()


def link_targets(root: Path, depth: int = 2) -> list[Path]:
    """Where the symbolic links in the data root's first `depth` levels lead (models or a census kept on another disk),
    so they resolve inside the container too."""
    out, level = [], [root]
    for _ in range(depth):
        nxt = []
        for d in level:
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                if e.is_symlink():
                    t = Path(os.path.realpath(e.path))
                    out.append(t if t.is_dir() else t.parent)
                elif e.is_dir():
                    nxt.append(e.path)
        level = nxt
    return out


# folders an engine argument must not mount (read-write into the container): the root and the system's own
REFUSED = {Path(p) for p in ("/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64", "/opt", "/proc", "/root",
                             "/run", "/sbin", "/srv", "/sys", "/tmp", "/usr", "/var")}


def arg_dirs(args: list[str]) -> list[Path]:
    """The folder of each absolute path among the engine's arguments (the path itself when it is a folder)."""
    out = []
    for x in args:
        if x.startswith("/"):
            p = Path(x)
            d = p if p.is_dir() else p.parent              # mounted as named: the engine's argument says that path
            if d.resolve() in REFUSED:
                raise SystemExit(f"[tt-moeoffload] {x}: its folder {d.resolve()} would be mounted into the container; "
                                 "use a path in a folder of its own")
            out.append(d)
    return out


def image() -> str:
    return json.loads((REPO / "pins.json").read_text())["tt_metalium"]["image"]


def container_argv(engine: Path, engine_args: list[str], *, fake: bool, name: str, cpus: str = "all",
                   memory_max: int | None = None, mounts: list[Path] = (), env: dict[str, str] | None = None,
                   cwd: Path | None = None, data: Path | None = None, img: str | None = None) -> list[str]:
    data, cwd = (data or data_root()).resolve(), (cwd or Path.cwd()).resolve()
    linked = set(link_targets(data))                         # models or a census kept elsewhere: read, never written
    dirs: list[Path] = []
    for d in sorted({REPO, data, *linked, *(Path(m).resolve() for m in mounts),
                     *arg_dirs([str(engine), *engine_args])},
                    key=lambda d: len(d.parts)):
        if not any(d.is_relative_to(x) for x in dirs):       # a folder inside a mounted one is already there
            dirs.append(d)
    workdir = cwd if any(cwd.is_relative_to(d) for d in dirs) else REPO
    a = ["docker", "run", "--rm", "-i", "--log-driver", "none", "--security-opt", "label=disable",
         "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
         *([] if fake else CARD_ARGS), *(f"--volume={d}:{d}" + (":ro" if d in linked else "") for d in dirs),
         # getpwuid() must know the user (torch and pytest call it)
         "--volume=/etc/passwd:/etc/passwd:ro", "--volume=/etc/group:/etc/group:ro",
         f"--user={os.getuid()}:{os.getgid()}", f"--env=HOME={data / 'home'}", f"--env=TT_MOEOFFLOAD_DATA={data}",
         f"--env=USER={os.environ.get('USER', 'user')}",
         f"--workdir={workdir}", f"--name={name}",
         *([f"--cpuset-cpus={cpus}"] if cpus != "all" else []), *([f"--memory={memory_max}"] if memory_max else []),
         # names only: docker reads each value from this process's environment (main), so none is in the argv or
         # the log line below
         *(f"--env={k}" for k in (env or {}) if k not in ("HOME", "USER", "TT_MOEOFFLOAD_DATA"))]
    return a + ["--entrypoint", "/bin/bash", img or image(), "-c",
                f"source {shlex.quote(str(TTENV))} && exec python {shlex.quote(str(engine))} {shlex.join(engine_args)}"]


SECRET = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)$")   # e.g. the API server's STRATA_API_KEY: the engine needs none


def passed_env(extra: list[str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIXES) and not SECRET.search(k)}
    for kv in extra:
        k, sep, v = kv.partition("=")
        if not sep:
            raise SystemExit(f"[tt-moeoffload] --env {kv}: expected K=V")
        env[k] = v
    return env


def running(name: str) -> bool:
    r = subprocess.run(["docker", "ps", "-q", "--filter", f"name=^{name}$"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fake", action="store_true", help="no card: the engine's fake mode, no checks")
    ap.add_argument("--engine", type=Path, default=ENGINE, help="the engine script (default: %(default)s)")
    ap.add_argument("--cpus", default="all", help="the container's CPUs (docker --cpuset-cpus)")
    ap.add_argument("--memory-max", type=int, default=None, help="the container's RAM limit in bytes")
    ap.add_argument("--mount", action="append", default=[], type=Path, metavar="DIR", help="also mount DIR")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="set in the container (repeatable)")
    ap.add_argument("--name", default=f"tt-moeoffload-{os.getpid()}", help="the container's name (default: per process)")
    # Strata's server ends its engine with QUIT, then SIGTERM, then SIGKILL, 20 s apart
    ap.add_argument("--grace", type=float, default=15.0, help="seconds the engine gets to close the card on SIGTERM")
    argv = sys.argv[1:] if argv is None else list(argv)
    cut = argv.index("--") if "--" in argv else len(argv)   # after --: the engine's alone, even names we also use
    a, unknown = ap.parse_known_args(argv[:cut])
    engine_args = unknown + argv[cut + 1:] + (["--fake"] if a.fake else [])
    if not a.fake:
        model = None                                    # the engine's RAM need; other scripts size themselves
        if a.engine.resolve() == ENGINE:
            i = engine_args.index("--model") if "--model" in engine_args else -1
            model = "glm" if 0 <= i < len(engine_args) - 1 and engine_args[i + 1] == "glm" else "flashnext"
        found = preflight.problems(model)
        if found:
            print("[tt-moeoffload] not starting:\n" + "\n".join(f"  - {p}" for p in found), file=sys.stderr)
            return 2
    if running(a.name):
        print(f"[tt-moeoffload] not starting: a container named {a.name} is running (another engine?)", file=sys.stderr)
        return 2
    subprocess.run(["docker", "rm", a.name], capture_output=True)   # a stopped one left behind (rm without -f)
    (data_root() / "home").mkdir(parents=True, exist_ok=True)    # created as this user, not by docker as root
    env = passed_env(a.env)
    argv = container_argv(a.engine.resolve(), engine_args, fake=a.fake, name=a.name, cpus=a.cpus,
                          memory_max=a.memory_max, mounts=a.mount, env=env)
    print(f"[tt-moeoffload] {shlex.join(argv)}", file=sys.stderr, flush=True)
    stopping: list[float] = []
    proc = subprocess.Popen(argv, env={**os.environ, **env})   # stdin, stdout: the server's pipes, straight through

    def term(*_):                                       # on to the engine (docker's sig-proxy), then wait --grace s
        if not stopping:
            stopping.append(time.time())
            proc.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, term)
    try:
        while True:
            try:
                rc = proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if stopping and time.time() - stopping[0] > a.grace:
                    rc = 143
                    break
    finally:
        if proc.poll() is None:
            subprocess.run(["docker", "kill", a.name], capture_output=True)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0 if stopping else rc


if __name__ == "__main__":
    raise SystemExit(main())
