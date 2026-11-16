"""The CPU lane of Flash-Next's card decode (experts.cpp): the routed experts of one token per layer, between the card's
D2H page and its H2D reply. Build it first with tt/cpulane/build.sh (in the container: tt/env/setup.sh builds it).

    lane = Lane()                            # lane_cpus(): a thread per physical core, the first 8 left to the host
    lane.load(MODEL_DIR)                     # every layer's expert tensors into the lane's own memory
    lane.exchange(layer, page_u8, reply_f32)  # numpy arrays (or anything with a buffer address)
    lane.moe_m(layer, x, ids, w, out)        # M <= 8 tokens (a verify pass): x, out [M, D]; ids, w [M, k]
    loop = Loop(lane)                        # a token's exchanges in one call (host_loop.cpp; g4_model.Link.bind_loop)
    loop.rows(4)                             # its M-row exchanges' and selections' rows (a verify pass)
    Gather(...)                              # the token's embedding and PLE rows (gather.cpp; g4_host.Host)
"""
from __future__ import annotations

import ctypes
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ.get("CL_LIB") or os.path.join(HERE, "build", "libcpulane.so")   # CL_LIB: e.g. the traced build
WHICH = {"ffn_gate_exps": 0, "ffn_up_exps": 1, "ffn_down_exps": 2}


def _ptr(a) -> ctypes.c_void_p:
    if isinstance(a, np.ndarray):
        assert a.flags["C_CONTIGUOUS"]
        return ctypes.c_void_p(a.ctypes.data)
    return ctypes.c_void_p(a.data_ptr())  # torch


def _cpu_list(s: str) -> list[int]:
    """'8-15,20' -> [8, ..., 15, 20] (the kernel's and docker's cpuset lists)."""
    out = []
    for part in s.split(","):
        a, _, b = part.strip().partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def lane_cpus(affinity=None, sysfs: str = "/sys/devices/system/cpu", reserve: int = 8) -> list[int]:
    """The lane's CPUs, thread 0's first: TT_MOEOFFLOAD_LANE_CPUS (a cpuset list, e.g. 8-63) when set; else one CPU per
    physical core this process may run on (a core's lowest sibling in the affinity: the lane's workers spin, and two on
    one core halve each other), sorted, the first `reserve` left to the host (the engine's Python, tt-metal's threads).
    8 is one L3 cluster on many x86 server chips (AMD's CCX), so the lane's 8-thread groups (experts.cpp GRP) each stay
    on one L3."""
    mine = set(os.sched_getaffinity(0) if affinity is None else affinity)
    env = os.environ.get("TT_MOEOFFLOAD_LANE_CPUS")
    if env:
        cpus = _cpu_list(env)
        if not set(cpus) <= mine:
            raise ValueError(f"TT_MOEOFFLOAD_LANE_CPUS={env}: {sorted(set(cpus) - mine)} not among this process's CPUs")
        return cpus

    def core(c):
        try:
            with open(f"{sysfs}/cpu{c}/topology/thread_siblings_list") as f:
                return min([s for s in _cpu_list(f.read()) if s in mine] + [c])
        except OSError:                       # no topology (some VMs): every CPU its own core
            return c
    cores = sorted({core(c) for c in mine})
    if len(cores) <= reserve:
        raise ValueError(f"the lane needs more than {reserve} physical cores; this process has {len(cores)}"
                         " (set TT_MOEOFFLOAD_LANE_CPUS)")
    return cores[reserve:]


class Lane:
    def __init__(self, threads: int | None = None, cpus=None, host_torch_threads: int = 1):
        """host_torch_threads: torch's intra-op pool once the lane runs (None: left as it is). The lane's workers spin
        between phases on their CPUs, and torch's pool, woken by any host-side op, would sit on the same CPUs: a
        prefill's per-layer host LayerNorm took 0.16 s instead of ~0.01 (GLM 2026-11-13; Flash-Next's, which
        each of its scripts did by hand)."""
        self.lib = ctypes.CDLL(LIB)
        L = self.lib
        L.cl_init.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        L.cl_threads.argtypes = [ctypes.c_int]
        L.cl_stats.argtypes = [ctypes.c_void_p]
        L.cl_load.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64,
                              ctypes.c_int]
        L.cl_route.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        L.cl_moe.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p]
        L.cl_moe_m.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_void_p]
        L.cl_exchange.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.max_m = L.cl_max_m() if hasattr(L, "cl_max_m") else 8   # tokens a moe_m call takes
        self.d = L.cl_d() if hasattr(L, "cl_d") else 2560             # the build's hidden size (CL_SHAPE)
        L.cl_route_m.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                 ctypes.c_void_p]
        if threads is None:                   # the default: lane_cpus()
            cpus = lane_cpus() if cpus is None else list(cpus)
            threads = len(cpus)
        cpus = list(cpus)[:threads] if cpus is not None else None
        self.cpus = cpus                      # thread i's CPU (i = 0: the caller's, cl_init's pin)
        arr = (ctypes.c_int * threads)(*cpus) if cpus else None
        mine = os.sched_getaffinity(0)
        before = set(os.listdir("/proc/self/task"))
        self.threads = L.cl_init(threads, arr)
        self.tids = {int(t) for t in set(os.listdir("/proc/self/task")) - before}   # its workers (fence_others)
        # cl_init pins the calling thread to cpus[0]; give it back its CPUs, or every thread it starts later (torch's
        # pool among them) inherits one CPU: building the model's weights took 44 s a layer instead of 1
        os.sched_setaffinity(0, mine)
        self.allowed = set(mine)              # the process's CPUs at the start
        if host_torch_threads is not None:
            import torch
            torch.set_num_threads(host_torch_threads)
        self.types = {}                       # (layer, which) -> ggml type

    def fence_others(self, keep=()) -> list:
        """Every thread of the process but the lane's workers (and `keep`) off the lane's CPUs, pinned ones too
        (tt-metal pins some of its own, which spin): a phase waits for its slowest worker. -> [(tid, its CPUs before)]
        for unfence."""
        free = self.allowed - set(self.cpus)
        moved = []
        for t in os.listdir("/proc/self/task"):
            tid = int(t)
            if tid in self.tids or tid in keep:
                continue
            try:
                had = os.sched_getaffinity(tid)
                if had & set(self.cpus):
                    os.sched_setaffinity(tid, (had - set(self.cpus)) or free)
                    moved.append((tid, had))
            except OSError:                       # a thread that ended
                pass
        return moved

    @staticmethod
    def unfence(moved: list) -> None:
        for tid, had in moved:
            try:
                os.sched_setaffinity(tid, had)
            except OSError:
                pass

    def load(self, folder: str, layers=None, workers: int = 8) -> int:
        """Reads the expert tensors of `layers` (default all) from the GGUF files; returns the bytes read."""
        from gguf import GGUFReader
        jobs = []
        for f in sorted(glob.glob(os.path.join(folder, "*.gguf"))):
            for t in GGUFReader(f).tensors:
                parts = t.name.split(".")
                if len(parts) == 4 and parts[2] in WHICH and (layers is None or int(parts[1]) in layers):
                    jobs.append((int(parts[1]), WHICH[parts[2]], f, int(t.data_offset), int(t.n_bytes),
                                 int(t.tensor_type)))

        def one(j):
            rc = self.lib.cl_load(j[0], j[1], j[2].encode(), j[3], j[4], j[5])
            if rc:
                raise OSError(-rc, f"cl_load layer {j[0]} tensor {j[1]}")
            self.types[(j[0], j[1])] = j[5]

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, jobs))
        return sum(j[4] for j in jobs)

    def stats(self) -> np.ndarray:
        """The cumulative ns per step since the last call: quantize hn, gate|up, quantize h, down."""
        out = np.zeros(4, np.uint64)
        self.lib.cl_stats(_ptr(out))
        return out

    def set_threads(self, n: int) -> int:
        return self.lib.cl_threads(n)

    def route(self, logits: np.ndarray, k: int = 10):
        ids, w = np.zeros(k, np.int32), np.zeros(k, np.float32)
        self.lib.cl_route(_ptr(np.ascontiguousarray(logits, np.float32)), k, _ptr(ids), _ptr(w))
        return ids, w

    def route_m(self, logits: np.ndarray, k: int = 10):
        """cl_route for each row of logits [m, >= 512] (fp32, rows contiguous): ids [m, k] i32, w [m, k] f32."""
        lg = np.ascontiguousarray(logits, np.float32)
        m = lg.shape[0]
        ids, w = np.zeros((m, k), np.int32), np.zeros((m, k), np.float32)
        self.lib.cl_route_m(_ptr(lg), m, lg.shape[1], k, _ptr(ids), _ptr(w))
        return ids, w

    def moe(self, layer: int, x: np.ndarray, ids: np.ndarray, w: np.ndarray, out: np.ndarray) -> None:
        rc = self.lib.cl_moe(layer, _ptr(x), len(ids), _ptr(ids), _ptr(w), _ptr(out))
        if rc:
            raise ValueError(f"cl_moe: layer {layer} not loaded")

    def moe_m(self, layer: int, x: np.ndarray, ids: np.ndarray, w: np.ndarray, out: np.ndarray) -> None:
        """M <= max_m tokens at once: x [M, D] f32, ids [M, k] i32, w [M, k] f32 -> out [M, D] f32 (each distinct expert
        streamed once; every row of out is moe() of that token alone, to the bit)."""
        m, k = ids.shape
        assert x.shape == (m, self.d) and w.shape == (m, k) and out.shape == (m, self.d)
        rc = self.lib.cl_moe_m(layer, m, _ptr(x), k, _ptr(ids), _ptr(w), _ptr(out))
        if rc:
            raise ValueError(f"cl_moe_m: layer {layer} not loaded, or bad routes")

    def warm(self, layer: int, ids: np.ndarray) -> None:
        """Post a read of layer's experts ids (distinct, int32) into the pool's caches, not waited for (cl_warm)."""
        ids = np.ascontiguousarray(ids, np.int32)
        self.lib.cl_warm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        if self.lib.cl_warm(layer, len(ids), _ptr(ids)):
            raise ValueError(f"cl_warm: layer {layer} not loaded, or bad ids")

    def router_load(self, layer: int, w: np.ndarray, bias: np.ndarray) -> None:
        """GLM: the layer's router (fp32 [NX, D], rows = experts) and its routing bias (fp32 [NX]) for glm_route."""
        L = self.lib
        L.cl_router_load.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        if L.cl_router_load(layer, _ptr(np.ascontiguousarray(w, np.float32)), _ptr(np.ascontiguousarray(bias, np.float32))):
            raise ValueError(f"cl_router_load: layer {layer}")

    def glm_route(self, layer: int, x: np.ndarray, k: int = 8, scale: float = 2.5):
        """GLM's routing on the lane's threads: sigmoid(W_r x), the top k of it + bias, weights normalized x scale.
        -> ids [k] i32, w [k] f32, logits [NX] f32."""
        L = self.lib
        L.cl_glm_route.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_void_p]
        ids, w, lg = np.zeros(k, np.int32), np.zeros(k, np.float32), np.zeros(L.cl_nx(), np.float32)
        if L.cl_glm_route(layer, _ptr(np.ascontiguousarray(x, np.float32)), k, scale, _ptr(ids), _ptr(w), _ptr(lg)):
            raise ValueError(f"cl_glm_route: layer {layer} has no router")
        return ids, w, lg

    def glm_route_m(self, layer: int, x: np.ndarray, k: int = 8, scale: float = 2.5):
        """glm_route for m rows at once (x fp32 [m, D]; a prefill chunk's), rows split over the lane's threads, every
        row's ids and weights glm_route's to the bit -> ids [m, k] i32, w [m, k] f32"""
        L = self.lib
        L.cl_glm_route_m.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_float,
                                     ctypes.c_void_p, ctypes.c_void_p]
        x = np.ascontiguousarray(x, np.float32)
        m = x.shape[0]
        ids, w = np.zeros((m, k), np.int32), np.zeros((m, k), np.float32)
        if L.cl_glm_route_m(layer, m, _ptr(x), k, scale, _ptr(ids), _ptr(w)):
            raise ValueError(f"cl_glm_route_m: layer {layer} has no router")
        return ids, w

    def glm_exchange(self, layer: int, hn_bf16, reply, ids=None, w=None) -> None:
        """GLM's exchange: hn (bf16 [D], uint16 or uint8 bytes) -> routing -> the experts' sum into reply (fp32 [D])."""
        L = self.lib
        L.cl_glm_exchange.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        rc = L.cl_glm_exchange(layer, _ptr(hn_bf16), _ptr(reply), _ptr(ids) if ids is not None else None,
                               _ptr(w) if w is not None else None)
        if rc:
            raise ValueError(f"cl_glm_exchange: layer {layer} ({rc})")

    def exchange(self, layer: int, page, reply, ids=None, w=None) -> None:
        rc = self.lib.cl_exchange(layer, _ptr(page), _ptr(reply), _ptr(ids) if ids is not None else None,
                                  _ptr(w) if w is not None else None)
        if rc:
            raise ValueError(f"cl_exchange: layer {layer} not loaded")


def transitions(census: str) -> np.ndarray:
    """The warm's layer -> layer + 1 table (Loop.warm) from a census trace (tools/llama/census.cpp: per prompt "P", then
    records phase u8, layer u16, rows u32, k u8, the rows' k experts u16; phase 1 a decode token's): float32 [47, 512,
    512], row e of [l] how often a token at l + 1 went to each expert when it went to e at l, summing to 1."""
    import struct
    data = open(census, "rb").read()
    i, prompts = 0, []
    while i < len(data):
        if data[i:i + 1] == b"P":
            n, = struct.unpack_from("<I", data, i + 1)
            i += 9 + n
            prompts.append([[] for _ in range(48)])
            continue
        phase, layer, rows, k = struct.unpack_from("<BHIB", data, i + 1)
        if phase == 1:
            prompts[-1][layer].append(np.frombuffer(data, np.uint16, k, i + 9))
        i += 9 + 2 * rows * k
    c = np.zeros(47 * 512 * 512, np.float64)
    for p in prompts:
        n = min(len(x) for x in p)
        if not n:
            continue
        d = np.stack([np.stack(x[:n]) for x in p]).astype(np.int64)   # [48, n, k]
        k = d.shape[2]
        a = np.repeat(d[:-1], k, 2)                                   # [47, n, k k]: e at l, each with ...
        b = np.tile(d[1:], (1, 1, k))                                 # ... every e' at l + 1
        c += np.bincount(((np.arange(47)[:, None, None] * 512 + a) * 512 + b).ravel(), minlength=c.size)
    c = c.reshape(47, 512, 512)
    return (c / np.maximum(c.sum(2, keepdims=True), 1)).astype(np.float32)


class Loop:
    """The token's host loop (host_loop.cpp): every exchange of a traced token, in launch order, in one call. The
    sockets are bound by g4_model.Link.bind_loop (or bind_test: pages from an array, replies into one)."""

    KIND = {"x": 0, "s": 1, "card": 2, "xm": 3, "sm": 4, "xmc": 5}

    def __init__(self, lane: Lane):
        L = self.lib = lane.lib
        p, i = ctypes.c_void_p, ctypes.c_int
        L.hl_error.restype = ctypes.c_char_p
        L.hl_bind.argtypes = [p, p, p, p]
        L.hl_bind_test.argtypes = [p, i, p, i]
        L.hl_plan.argtypes = [i, p, p]
        L.hl_slots.argtypes = [i, p, i]
        L.hl_rows.argtypes = [i]
        L.hl_select.argtypes = [i, p, p, p]
        L.hl_token.argtypes = [i, p, p]
        self.n = 0

    def _check(self, rc: int) -> None:
        if rc:
            raise RuntimeError(self.lib.hl_error().decode())

    def route_log(self, cap: int):
        """Every M-row exchange's routes from now on: returns the buffer, int16 [cap, 2 + 4 * 10] (layer, M,
        the rows' 10 experts each, -1 past M); route_count() records are filled. cap 0: off."""
        import numpy as np
        self._routes = np.full((cap, 2 + 4 * 10), -1, np.int16) if cap else None
        self.lib.hl_route_count.restype = ctypes.c_int64
        self.lib.hl_route_log(_ptr(self._routes) if cap else None, ctypes.c_int64(cap))
        return self._routes

    def route_count(self) -> int:
        self.lib.hl_route_count.restype = ctypes.c_int64
        return int(self.lib.hl_route_count())

    def card_costs(self, balance: bool, lane_exp: float = 11.6, lane_use: float = 1.6, card_exp: float = 15.1,
                   card_fix: float = 41.0) -> None:
        """Kind 5's split of a layer's resident routes: all to the card (balance False), or as many as make the
        card's and the lane's parts closest by these costs in us (the lane per distinct expert and per extra use, the
        card per expert and its fixed part a layer)."""
        f = ctypes.c_float
        self.lib.hl_card_costs(int(balance), f(lane_exp), f(lane_use), f(card_exp), f(card_fix))

    def plan(self, kinds, layers, slot_of: dict | None = None, card_max: int = 0) -> None:
        """The exchanges (kinds: "x" lane, "s" selection, "card" card + lane, "xm" a verify pass's M-row lane exchange
        (Link.serve_m), "sm" its M-row selection (Link.serve_select_m; M from rows()), "xmc" an M-row exchange with
        the card's share (Link.serve_m_card); their layers); slot_of: per layer the card's expert -> slot map (int32
        [512]) for "card" and "xmc". stamps / n_card hold the last token's clocks and counts."""
        k = np.array([self.KIND[x] for x in kinds], np.int32)
        il = np.array(layers, np.int32)
        self._check(self.lib.hl_plan(len(k), _ptr(k), _ptr(il)))
        self.slots = {l: np.ascontiguousarray(m, np.int32) for l, m in (slot_of or {}).items()}   # kept alive
        for l in range(48):
            m = self.slots.get(l)
            self.lib.hl_slots(l, _ptr(m) if m is not None else None, card_max)
        self.n = len(k)
        self.stamps, self.n_card = np.zeros((self.n, 4), np.int64), np.zeros(self.n, np.int32)
        self.p_stamps, self.p_n_card = _ptr(self.stamps), _ptr(self.n_card)

    def warm(self, n: int, trans: np.ndarray | None = None) -> None:
        """After each M-row exchange, the lane's threads read the n experts the next one likeliest routes to while the
        card computes it (host_loop.cpp warm(), experts.cpp cl_warm): trans (transitions()) and the last 16 rows at
        that layer. n 0: off. Reads only: the replies are the same bits."""
        self._trans = None if trans is None else np.ascontiguousarray(trans, np.float32)   # kept alive
        assert self._trans is None or self._trans.shape == (47, 512, 512)
        self.lib.hl_warm.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self.lib.hl_warm(n, _ptr(self._trans) if self._trans is not None else None)

    def rows(self, m: int) -> None:
        """The "xm" and "sm" exchanges' rows from the next token() on (a verify pass's M, 1..4)."""
        self._check(self.lib.hl_rows(m))

    def bind(self, d2h: int, h2d: int, read_fn: int, write_fn: int) -> None:
        self.lib.hl_bind(d2h, h2d, read_fn, write_fn)

    def bind_test(self, pages: np.ndarray, replies: np.ndarray) -> None:
        """pages uint8 [n, 8192] in, replies uint8 [m, 10240] out (kept alive by the caller)."""
        assert pages.shape[1] == 8192 and replies.shape[1] == 10240
        self.lib.hl_bind_test(_ptr(pages), len(pages), _ptr(replies), len(replies))

    def token(self, pos: int, stamps: bool = True) -> None:
        """One token's exchanges; with stamps, self.stamps gets each one's ns (CLOCK_MONOTONIC, time.perf_counter's
        clock) before its page, after it, after the work, after the reply."""
        self._check(self.lib.hl_token(pos, self.p_stamps if stamps else None, self.p_n_card))

    def select(self, pos: int, scores: np.ndarray, words: np.ndarray) -> tuple[int, int]:
        out = np.zeros(2, np.int32)
        self.lib.hl_select(pos, _ptr(np.ascontiguousarray(scores, np.float32)), _ptr(words), _ptr(out))
        return int(out[0]), int(out[1])


class Gather:
    """The token's inputs (gather.cpp): token_embd's row and the PLE rows, dequantized by ggml's to_float, as
    g4_host.Host's embed / ple (to the bit; Host routes to it). embd, ple: (GGUF file, offset, ggml type, rows, ne0);
    the PLE hash's parameters as Host's."""

    def __init__(self, embd, ple, n_gram, per_gram, eos, mult, offsets, vocab_sizes):
        L = self.lib = ctypes.CDLL(LIB)
        p, i, i64, u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_uint64
        L.gt_open_embd.argtypes = [ctypes.c_char_p, u64, i, i64, i64]
        L.gt_open_ple.argtypes = [ctypes.c_char_p, u64, i, i64, i64, i, i, i64, p, p, p]
        L.gt_embed.argtypes = [i64, p]
        L.gt_ple_rows.argtypes = [i64, p, i, p]
        L.gt_ple.argtypes = [i64, p, i, p]
        if hasattr(L, "gt_inputs"):
            L.gt_inputs.argtypes = [p, i64, i, i, p, p, i]
        hp = [np.array(v, np.uint64) for v in (mult, offsets, vocab_sizes)]   # alive through the call
        for name, rc in (("token_embd", L.gt_open_embd(embd[0].encode(), *embd[1:])),
                         ("per_layer_token_embd", L.gt_open_ple(ple[0].encode(), *ple[1:], n_gram, per_gram, eos,
                                                                *map(_ptr, hp)))):
            if rc:
                raise OSError(-rc, f"gather: {name} ({rc})")
        self.n_prev, self.n_heads, self.d = n_gram - 1, (n_gram - 1) * per_gram, embd[4]
        self.Prev = ctypes.c_int64 * self.n_prev

    def embed(self, tok: int) -> np.ndarray:
        out = np.empty(4 * self.d, np.float32)
        if self.lib.gt_embed(tok, out.ctypes.data):
            raise IndexError(f"gather: token {tok}")
        return out

    def _prev(self, prev):
        p = prev[-self.n_prev:]
        return self.Prev(*p), len(p)

    def ple_rows(self, tok: int, prev) -> list[int]:
        idx = np.empty(self.n_heads, np.int64)
        self.lib.gt_ple_rows(tok, *self._prev(prev), idx.ctypes.data)
        return idx.tolist()

    def ple(self, tok: int, prev) -> np.ndarray:
        out = np.empty(self.d, np.float32)
        if self.lib.gt_ple(tok, *self._prev(prev), out.ctypes.data):
            raise IndexError(f"gather: token {tok}")
        return out
