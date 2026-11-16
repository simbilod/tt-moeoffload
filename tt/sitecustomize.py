"""Every Python process in the card container (tt/ is on PYTHONPATH through tt/env/ttenv.sh; the image's venv Python has
no sitecustomize of its own): allow any same-user tracer, so py-spy can dump or record it although it is not the
process's parent (hosts with Yama ptrace_scope 1; tt/serve/stall_watch.py). 2026-11-07."""
try:
    import ctypes
    ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1).value, 0, 0, 0)   # PR_SET_PTRACER, PR_SET_PTRACER_ANY
except (OSError, AttributeError):
    pass
