"""The card engine's stall watcher (2026-11-07): a separate process, so it runs while the engine's main thread holds the
GIL inside a hung native call (where a watchdog thread in the engine never ran). The engine touches BEAT on every line
it says and writes "1" into it while a request runs, "0" between; when a request has said nothing for STALL seconds
this dumps the engine's threads (py-spy dump --native) to stderr, the engine log, once a stall. Exits with the engine.

    stall_watch.py <engine pid> <beat file> <py-spy> <stall s>        (started by engine.Engine.serve)
"""
import os
import subprocess
import sys
import time


def main() -> int:
    pid, beat, spy, stall = int(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
    dumped = None
    while os.getppid() == pid:
        time.sleep(5)
        try:
            mtime = os.stat(beat).st_mtime
            with open(beat) as f:
                busy = f.read(1) == "1"
        except OSError:
            continue
        quiet = time.time() - mtime
        if busy and quiet >= stall and dumped != mtime:
            dumped = mtime
            print(f"[ttcard] a request silent for {quiet:.0f} s; py-spy dump --native:", file=sys.stderr, flush=True)
            try:
                subprocess.run([spy, "dump", "--native", "--pid", str(pid)], stdout=sys.stderr, stderr=sys.stderr,
                               timeout=180)
            except (OSError, subprocess.TimeoutExpired) as e:
                print(f"[ttcard] stall dump failed: {e}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
