"""A stand-in engine for the launcher's tests: the engine line protocol (READY, GEN -> T.. DONE, QUIT), no card.
Each generated token echoes the prompt's ids back; SIGTERM exits 0 after printing BYE to stderr."""
import signal
import sys

signal.signal(signal.SIGTERM, lambda *_: (print("BYE", file=sys.stderr, flush=True), sys.exit(0)))
print("INFO stub=1 args=" + ",".join(sys.argv[1:]), flush=True)
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    w = line.split()
    if w[:1] == ["QUIT"]:
        break
    if w[:1] == ["GEN"]:
        ids = w[-1].split(",")
        n = min(int(w[1]), len(ids))
        for t in ids[:n]:
            print(f"T {t}", flush=True)
        print(f"DONE {n} {len(ids)} 0 0 length 0 0 0 0 0 0 0 0 {len(ids)}", flush=True)
