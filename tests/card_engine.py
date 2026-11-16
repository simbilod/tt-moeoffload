"""A stand-in engine that opens the card: through the launcher's card path it checks what the real engine needs from
the container (the device, hugepages, kernel compiles, a matmul's result, H2D and D2H sockets pinned on the first try,
which the preflight's DMA-FQ check promises), then speaks the line protocol as stub_engine.py does. QUIT or SIGTERM
close the card."""
import os
import signal
import sys
import time

# the protocol owns stdout: tt-metal's logs (written to fd 1) go to stderr, as the real engine does it
proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr

import ttnn  # noqa: E402

t0 = time.time()
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))


def close(*_):
    ttnn.close_mesh_device(dev)
    print("CLOSED", file=sys.stderr, flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, close)
a = ttnn.Tensor([1.0] * 32 * 32, [32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
got = ttnn.matmul(a, a).cpu().to_list()
assert all(v == 32.0 for row in got for v in row) if isinstance(got[0], list) else all(v == 32.0 for v in got), got
core = ttnn.CoreCoord(0, 0)
where = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), core)
h2d = ttnn.H2DSocket(dev, where, ttnn.BufferType.L1, 64 << 10, ttnn.H2DMode.DEVICE_PULL)
d2h = ttnn.D2HSocket(dev, where, 64 << 10)
del h2d, d2h
print(f"INFO card=ok matmul=ok sockets=ok open_s={time.time() - t0:.1f}", file=proto)
print("READY 4096 stop", file=proto)
for line in sys.stdin:
    w = line.split()
    if w[:1] == ["QUIT"]:
        break
    if w[:1] == ["GEN"]:
        ids = w[-1].split(",")
        n = min(int(w[1]), len(ids))
        for t in ids[:n]:
            print(f"T {t}", file=proto)
        print(f"DONE {n} {len(ids)} 0 0 length 0 0 0 0 0 0 0 0 {len(ids)}", file=proto)
close()
