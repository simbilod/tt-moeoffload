"""What the host must provide before the engine starts, each missing piece said in a sentence a user can act on.

  python3 -m tt_moeoffload.preflight        prints the problems (exit 1) or "ok"

The launcher runs it before every start on the card (not with --fake). `root` lets tests point the checks at a fake
/sys, /proc and /dev tree.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

# shortcut: the versions tested on the first card are the minimum; lower them when someone runs older ones
MIN_KMD = (2, 11, 0)
MIN_FW = (19, 16, 0, 0)
# Strata's AVX-512 i-quant rows (src/kernels/cpu/iq_avx512.cpp) use 8/16-bit ops (BW) and dpbusd/dpwssd (VNNI)
CPU_FLAGS = ("avx512f", "avx512bw", "avx512_vnni")
# the engine's peak host memory per model (VmHWM, GB), measured at the largest context served, +10% headroom:
# Flash-Next 157 GB at 131K with speculation (check_card.py spec serve, 2026-11-15); GLM 333.9 GB at 32K (g7_check, 2026-11-15)
RAM_GB = {"flashnext": 157 * 1.1, "glm": 333.9 * 1.1}
DISK_GB = {"flashnext": 94 + 70 + 7 + 1, "glm": 156 + 165 + 1}   # under the data root: model, card cache, MTP, references
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILT = ("tt/cpulane/build/libcpulane.so", "tt/cpulane/build/libcpulane_glm.so", "tt/spec/build/libstrata_spec.so",
         "third_party/strata/serve/server.py", "third_party/maya/serve/server.py")


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _version(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split("-")[0].split(".") if x.isdigit())


def cards(root: str = "/") -> list[str]:
    """The cards' sysfs directories (/sys/class/tenstorrent/tenstorrent!N)."""
    return sorted(glob.glob(os.path.join(root, "sys/class/tenstorrent/*")))


def check_cards(root: str = "/") -> list[str]:
    found = cards(root)
    if not found:
        return ["no Tenstorrent card found (/sys/class/tenstorrent is empty): install tt-kmd with tt-installer and "
                "check that the card shows in lspci"]
    problems = []
    if not glob.glob(os.path.join(root, "dev/tenstorrent/[0-9]*")):
        problems.append("/dev/tenstorrent has no card device: tt-kmd is loaded but did not bind the card")
    kmd = _read(os.path.join(root, "sys/module/tenstorrent/version"))
    if kmd is None or _version(kmd) < MIN_KMD:
        problems.append(f"tt-kmd {kmd or 'not loaded'}: {'.'.join(map(str, MIN_KMD))} or newer is needed "
                        "(tt-installer)")
    for c in found:
        name = os.path.basename(c)
        kind = _read(os.path.join(c, "tt_card_type")) or "unknown"
        if not kind.startswith("p150"):   # the kernels assume a p150's 11 x 10 cores and 8 DRAM banks
            problems.append(f"{name} is a {kind}: only the p150 (a, b, c) is supported (the kernels assume its 110 "
                            "cores and 8 DRAM banks; the p100a and p300 differ)")
        fw = _read(os.path.join(c, "tt_fw_bundle_ver"))
        if fw is None or _version(fw) < MIN_FW:
            problems.append(f"{name} firmware {fw or 'unknown'}: {'.'.join(map(str, MIN_FW))} or newer is needed "
                            "(tt-flash, from tt-installer)")
        group = os.path.join(c, "device", "iommu_group")
        iommu = _read(os.path.join(group, "type"))
        # the engine's host link uses tt-metal sockets: their FIFOs need the card's DMA translated, or pinning
        # fails on any buffer that is not physically contiguous
        if iommu not in ("DMA", "DMA-FQ"):
            dev = os.path.basename(os.path.realpath(os.path.join(c, "device")))
            if iommu:
                problems.append(f"{name} ({dev}): IOMMU group {os.path.basename(os.path.realpath(group))} is {iommu}, "
                                "the engine's sockets need DMA-FQ. Once: sudo tools/iommu/tt-iommu-dmafq.sh; at every "
                                "boot and card reset: install tools/iommu (its header says how)")
            else:
                problems.append(f"{name} ({dev}): no IOMMU group (the IOMMU is off). Enable it in the BIOS and on the "
                                "kernel command line (amd_iommu=on or intel_iommu=on; iommu=pt is fine), then "
                                "tools/iommu")
    pages = _read(os.path.join(root, "sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"))
    if not pages or int(pages) < len(found):
        problems.append(f"{pages or 0} 1 GiB hugepages reserved, at least one per card ({len(found)}) is needed "
                        "(tt-installer sets them up)")
    if not os.path.isdir(os.path.join(root, "dev/hugepages-1G")):
        problems.append("/dev/hugepages-1G is not mounted (tt-installer sets it up)")
    return problems


def check_ram(model: str, root: str = "/") -> list[str]:
    info = _read(os.path.join(root, "proc/meminfo")) or ""
    avail = next((int(ln.split()[1]) * 1024 for ln in info.splitlines() if ln.startswith("MemAvailable:")), 0)
    need = RAM_GB[model] * 1e9
    return [f"{avail / 1e9:.0f} GB of RAM available, {model} needs about {need / 1e9:.0f} GB (its routed experts "
            "live in host memory; no SSD tier yet)"] if avail < need else []


def check_cpu(root: str = "/") -> list[str]:
    info = _read(os.path.join(root, "proc/cpuinfo")) or ""
    line = next((ln for ln in info.splitlines() if ln.startswith("flags")), "")
    flags = set(line.split(":", 1)[-1].split())
    missing = [f for f in CPU_FLAGS if f not in flags]
    return [f"the CPU lacks {', '.join(missing)}: the CPU lane needs AVX-512 with BW and VNNI "
            "(AMD Zen 4 or newer, Intel Ice Lake or newer)"] if missing else []


def check_docker() -> list[str]:
    if shutil.which("docker") is None:
        return ["docker is not installed: the engine runs in Tenstorrent's tt-metal container"]
    r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True)
    if r.returncode:
        return [f"docker does not answer ({(r.stderr.strip().splitlines() or ['no output'])[-1]}): start it, and "
                "add this user to the docker group"]
    return []


def problems(model: str | None = None, root: str = "/") -> list[str]:
    """model (flashnext, glm): also its RAM need."""
    return check_cards(root) + check_cpu(root) + (check_ram(model, root) if model else []) + check_docker()


def check_setup() -> list[str]:
    missing = [p for p in BUILT if not os.path.exists(os.path.join(REPO, p))]
    return [f"{missing[0]} (and {len(missing) - 1} more) not there: run bash tt/env/setup.sh"] if missing else []


def fits() -> tuple[list[str], list[str]]:
    """Per model: its RAM and the free disk under the data root against its needs -> (lines, problems when none fits)"""
    data = os.environ.get("TT_MOEOFFLOAD_DATA") or os.path.expanduser("~/.cache/tt-moeoffload")
    d = data
    while not os.path.exists(d):
        d = os.path.dirname(d)
    free = shutil.disk_usage(d).free / 1e9
    pins = json.load(open(os.path.join(REPO, "pins.json")))["models"]
    lines, ok = [], []
    for m in RAM_GB:
        ram = not check_ram(m)
        top = next(iter(pins[m]["files"].values()))["to"].split("/")[0]
        disk = free >= DISK_GB[m] or os.path.exists(os.path.join(data, "models", top))   # prepared already
        ok.append(ram and disk)
        lines.append(f"{m}: RAM {'ok' if ram else 'too little'} (needs {RAM_GB[m]:.0f} GB), disk under {data} "
                     f"{'ok' if disk else 'too little'} ({free:.0f} GB free, needs {DISK_GB[m]} GB)")
    return lines, ([] if any(ok) else ["neither model fits this host's RAM and disk (above)"])


if __name__ == "__main__":     # the host and this checkout; the launcher runs problems() before each start
    lines, none = fits()
    found = problems() + check_setup() + none
    print("\n".join(f"  {ln}" for ln in lines))
    print("\n".join(f"- {p}" for p in found) if found else "ok")
    sys.exit(1 if found else 0)
