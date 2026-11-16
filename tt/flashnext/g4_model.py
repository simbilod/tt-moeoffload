"""G4: Flash-Next's decode token on the card as a chain of fused programs, built once (the token loop's pieces).

Every program reads the residual streams X from the model's two X tensors (fp32 and its bf16 copy, [4 x 2560]) and
writes the new X back into them; per-token values (the position) live in a step tensor in card memory (uint32 [16],
word 0 the position), so a program is built once and launched every token. The routed experts run on the CPU
(tt/cpulane) behind the host link (fused/link.h): each layer's coordinator sends hn_f and the router's logits over a
D2H socket and pulls the experts' sum back over an H2D socket; the host services them in launch order.

  Link       the two sockets on the coordinator core, shared by every program that exchanges; a traced token's
             exchanges all in one C++ call (tt/cpulane/host_loop.cpp, Link.bind_loop; G4_PY_LOOP=1: Python's);
             serve_m: a verify pass's M-row exchange (M <= 4 rows' hn_f and logits in, M routed sums out)
  DeltaRun   consecutive DeltaNet layers in one program from a layer table (g3_layer.py's), states in place; with
             M-row layers (g3_layer.Layer(m_max > 1)) a verify pass's, M from its step tensor
  Model      the token's programs; the head returns the logits (1 MB read per token), the top 64 found on the card
             (2 KB: ids mode, the timing's and serving's) or both (the parities' checks): Model.set_head
             With m_max = 4 the verify pass's programs instead: every program at M <= 4 rows (g3_layer.Layer,
             Attn1, Attn2, Ple, Head built with M rows), X in the block layout, one traced build for M = 1..4 rows
             at positions pos .. pos + M - 1, M, n_keep and the states' parity from the step tensor (step_words);
             Model.verify runs a pass (each attention layer's selection per row and their union in C++: kind "sm"),
             accept() is the greedy acceptance, reset() a sequence's start
  Spec       speculative decoding: the verify model and the MTP draft layer (g6_mtp.MtpCard) in one process,
             a pass = verify [the bonus token + drafts] -> greedy acceptance -> the MTP's catch-up over the kept rows
             (their h_nextn read from X after layer 47) and its chained drafts for the next pass
"""
from __future__ import annotations

import ctypes
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import sys
import time

import numpy as np
import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
from fused.grid import profile_tensor  # noqa: E402
from g3_layer import Layer, mp_of, step_words  # noqa: E402
from g3_layers import ROW, addr  # noqa: E402
from u1_sockets import pinned  # noqa: E402

D, HC = 2560, 4
OUT_PAGE, IN_PAGE = 8192, D * 4        # hn_f (bf16) + 17 x 32 logits, padded; the routed sum (fp32)
FIFO = 64 << 10
M_MAX = 4                               # an M-row exchange's most rows (flayer_reader.cpp's M_MAX)
ROW_OUT = 2 * D + 17 * 128              # its pages' bytes per row: hn_f (bf16), the logits' 17 tiles (fp32)
SSD_MODEL = os.environ.get("TT_MOEOFFLOAD_MODEL_DIR") or os.path.join(DATA, "models/qwen3.8-flash-next-ud-iq4xs/UD-IQ4_XS")   # engine.py --model-dir sets it
TOPK = 64                               # g3_head.TOPK: the head's top k per row (fhead_writer.cpp)
NC_M, RQ_W, RQ_BYTES = 24, 25, 512      # fused/dexp_dm.h NC, W0, RQ_BYTES: the card's routes at M rows
SLOT_BYTES = 40 * 10 * 4096 + 40 * 4 * 4096 + 80 * 20 * 1088 + 80 * 4096   # a cache slot in card memory (g5_cache)
HEAD_MODES = {"logits": {}, "ids": dict(logits=False, ids=True), "both": dict(logits=True, ids=True)}  # program()'s


def out_pages(M: int) -> int:
    """An M-row exchange's D2H pages: the mp-row blocks of hn_f, then of the logits (at Mp = 1 the M = 1 page)."""
    return -(-mp_of(M) * ROW_OUT // OUT_PAGE)


def unblock_m(buf: np.ndarray, M: int) -> tuple[np.ndarray, np.ndarray]:
    """An M-row exchange's pages (uint8, out_pages(M) x OUT_PAGE) -> hn_f fp32 [M, D] (from bf16), the router's
    logits fp32 [M, 512] (g3_layer.unblocks' layout)."""
    mp = mp_of(M)
    hb = buf[:mp * 2 * D].view(np.uint16).reshape(D // 32, 2, mp, 16).transpose(2, 0, 1, 3).reshape(mp, D)[:M]
    lg = buf[mp * 2 * D:mp * ROW_OUT].view(np.float32).reshape(17, 2, mp, 16).transpose(2, 0, 1, 3).reshape(mp, 544)
    return (hb.astype(np.uint32) << 16).view(np.float32), lg[:M, :512].copy()   # (copies: never views of buf)


def accept(drafts, greedy) -> int:
    """Greedy acceptance of a verify pass: the rows kept, 1 + the drafts that match the greedy id of the row before
    them, up to the first that does not (drafts[m] is row m + 1's token, greedy[m] row m's argmax). The next pass
    starts with greedy[n - 1] (the bonus token) at pos + n, with n_keep = n."""
    n = 1
    for d, gid in zip(drafts, greedy):
        if int(d) != int(gid):
            break
        n += 1
    return n


def x_tensors(dev, x: torch.Tensor | None = None):
    """The model's X tensors (fp32, bf16 [4 x 2560], row-major)."""
    x = torch.zeros(HC * D) if x is None else x.float().flatten()
    x32 = ttnn.from_torch(x.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    x16 = ttnn.from_torch(x.to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                          layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    return x32, x16


def set_x(dev, x32, x16, x: torch.Tensor) -> None:
    """Write X (host, [4 x 2560]) into the model's X tensors, in order with the queued programs."""
    x = x.float().flatten()
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(x.reshape(1, 1, 1, -1), dtype=ttnn.float32,
                                                    layout=ttnn.ROW_MAJOR_LAYOUT), x32)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(x.to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                                                    layout=ttnn.ROW_MAJOR_LAYOUT), x16)


class Inputs:
    """A token's inputs as host tensors made once, filled in place (writable views of their buffers): X fp32 and bf16,
    the PLE input bf16, the step words; the same bytes as set_x's and step()'s ttnn.from_torch, without building
    four host tensors per token. lib: the lane's library, whose gt_fill (gather.cpp) fills them in one call (6 us
    against torch's copies' 19, off the card); None: torch's."""

    def __init__(self, lib=None):
        def one(n, dt, tdt):
            h = ttnn.from_torch(torch.zeros(1, 1, 1, n, dtype=tdt), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)
            return h, torch.from_dlpack(h.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))).view(tdt)
        made = [one(HC * D, ttnn.float32, torch.float32), one(HC * D, ttnn.bfloat16, torch.bfloat16),
                one(D, ttnn.bfloat16, torch.bfloat16), one(16, ttnn.uint32, torch.int32)]
        self.host = [h for h, _ in made]
        self.x32, self.x16, self.emb, self.step = [v for _, v in made]
        self.gt_fill, self.ptrs = (lib.gt_fill if lib else None), [v.data_ptr() for _, v in made]
        if lib:
            p = ctypes.c_void_p
            lib.gt_fill.argtypes = [p, p, ctypes.c_int32, p, p, p, p]

    def fill(self, x0: np.ndarray, emb: np.ndarray, pos: int) -> None:
        if self.gt_fill:
            x, e = np.ascontiguousarray(x0, np.float32), np.ascontiguousarray(emb, np.float32)
            assert x.size == HC * D and e.size == D
            self.gt_fill(x.ctypes.data, e.ctypes.data, pos, *self.ptrs)
            return
        x = torch.from_numpy(np.ascontiguousarray(x0, np.float32).reshape(-1))
        self.x32.copy_(x)
        self.x16.copy_(x)                      # round to nearest even, as .to(torch.bfloat16)
        self.emb.copy_(torch.from_numpy(np.ascontiguousarray(emb, np.float32).reshape(-1)))
        self.step[0] = pos


def bf16_bits(a: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bits, rounded to nearest even (torch's .to(torch.bfloat16) for finite values)."""
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class InputsM:
    """A verify pass's inputs as host tensors made once, filled in place by numpy (no torch between launches: its
    thread pool then sits on the lane's CPUs): X fp32 and bf16 in the block layout ([4 x m_max 2560], rows
    M..Mp-1 copies of row M - 1), the PLE inputs bf16 [m_max x 2560] (rows past M row M - 1's), the step words."""

    def __init__(self, m_max: int):
        def one(shape, dt, tdt):
            h = ttnn.from_torch(torch.zeros(shape, dtype=tdt), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)
            v = torch.from_dlpack(h.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0)))
            return h, v.view(torch.int16 if tdt == torch.bfloat16 else tdt).numpy().reshape(shape[2:])
        made = [one((1, 1, HC, m_max * D), ttnn.float32, torch.float32),
                one((1, 1, HC, m_max * D), ttnn.bfloat16, torch.bfloat16),
                one((1, 1, m_max, D), ttnn.bfloat16, torch.bfloat16), one((1, 1, 1, 16), ttnn.uint32, torch.int32)]
        self.host = [h for h, _ in made]
        self.x32, self.x16, self.emb, self.step = [v for _, v in made]

    def fill(self, x0s, embs, pos: int, M: int, n_keep: int, parity: int) -> None:
        mp = mp_of(M)
        x = np.empty((mp, HC * D), np.float32)
        x[:M] = np.asarray(x0s, np.float32).reshape(M, HC * D)
        x[M:] = x[M - 1]
        blk = x.reshape(mp, HC, D // 32, 2, 16).transpose(1, 2, 3, 0, 4).reshape(HC, mp * D)  # g3_layer.blocks
        self.x32[:, :mp * D] = blk
        self.x16[:, :mp * D] = bf16_bits(blk).view(np.int16)
        e = bf16_bits(np.asarray(embs, np.float32).reshape(M, D)).view(np.int16)
        self.emb[:M] = e
        self.emb[M:] = e[M - 1]
        self.step[0, :5] = (pos, M, n_keep, parity, mp)


class Link:
    """The host link's sockets on the coordinator core, and the host's side of one exchange."""

    def __init__(self, dev, core, sockets=None):
        """sockets: (d2h, h2d) objects with read_tensor / write_tensor instead of the card's (g4_host_loop.py)."""
        if sockets is None:
            where = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*core))
            self.h2d = pinned(lambda: ttnn.H2DSocket(dev, where, ttnn.BufferType.L1, FIFO, ttnn.H2DMode.DEVICE_PULL))
            self.h2d.set_page_size(IN_PAGE)
            self.d2h = pinned(lambda: ttnn.D2HSocket(dev, where, FIFO))
            self.d2h.set_page_size(OUT_PAGE)
            self.defines = [("EXCHANGE", "1"), ("H2D_CONFIG", str(self.h2d.get_config_buffer_address())),
                            ("D2H_CONFIG", str(self.d2h.get_config_buffer_address())),
                            ("XCHG_OUT_BYTES", str(OUT_PAGE))]
        else:
            self.d2h, self.h2d = sockets
        from g3_attn1 import NPOOL
        self.page = torch.zeros(OUT_PAGE, dtype=torch.uint8)
        # the pool scores' pages (a page per 2048 pools up to the token's: up to 8 at 64K cells), together
        self.score_pages = torch.zeros(NPOOL * 4 // OUT_PAGE, OUT_PAGE, dtype=torch.uint8)
        self.scores = self.score_pages.numpy().reshape(-1).view(np.float32)
        self.reply = torch.zeros(IN_PAGE, dtype=torch.uint8)
        self.page_np, self.reply_np = self.page.numpy(), self.reply.numpy().view(np.float32)
        # serve_card's views of the same buffers and its own, with their addresses (the host serves 36 such exchanges a
        # token: a numpy call costs ~1-3 us, a ctypes call through the lane's wrappers ~10)
        self.words, self.logits = self.reply_np.view(np.uint32), self.page_np[2 * D:2 * D + 2048].view(np.float32)
        self.hn16, self.hn_bits = self.page_np[:2 * D].view(np.uint16), np.empty(D, np.uint32)
        self.ids, self.w = np.zeros(10, np.int32), np.zeros(10, np.float32)
        self.ids_cpu, self.w_cpu, self.cpu = np.zeros(10, np.int32), np.zeros(10, np.float32), np.ones(10, bool)
        ptr = lambda a: ctypes.c_void_p(a.ctypes.data)                       # noqa: E731
        self.p_logits, self.p_hn, self.p_out = ptr(self.logits), ptr(self.hn_bits), ptr(self.reply_np)
        self.p_ids, self.p_w = ptr(self.ids), ptr(self.w)
        self.p_ids_cpu, self.p_w_cpu = ptr(self.ids_cpu), ptr(self.w_cpu)
        self.pages_m = torch.zeros(M_MAX, OUT_PAGE, dtype=torch.uint8)    # serve_m's pages, its replies (rows)
        self.replies_m = torch.zeros(M_MAX, IN_PAGE, dtype=torch.uint8)
        self.out_m = self.replies_m.numpy().view(np.float32)
        self.ids_m, self.w_m = np.zeros((M_MAX, 10), np.int32), np.zeros((M_MAX, 10), np.float32)
        self.t = np.zeros(3)                  # the host's seconds: waiting for pages, the lane, writing replies
        self.trace = None                     # a list: every exchange's (kind, layer, wait, page, done, written) clocks

    def serve(self, lane, layer: int, ids=None, w=None) -> None:
        """One exchange: the card's page in, the CPU lane's experts for `layer`, the reply out."""
        t0 = time.perf_counter()
        self.d2h.read_tensor(self.page)
        t1 = time.perf_counter()
        lane.exchange(layer, self.page_np, self.reply_np, ids, w)
        t2 = time.perf_counter()
        self.h2d.write_tensor(self.reply)
        t3 = time.perf_counter()
        self.t += (t1 - t0, t2 - t1, t3 - t2)
        if self.trace is not None:
            self.trace.append(("x", layer, t0, t1, t2, t3))

    def serve_m(self, lane, layer: int, M: int) -> None:
        """A verify pass's exchange (a DeltaNet layer of the M-row program, flayer_reader.cpp): the M rows' pages in,
        each row routed as cl_exchange routes it, the lane's experts for the M rows at once (cl_moe_m: each row's sum
        the bits of cl_moe for that row alone), row m's sum out as reply page m. The routes stay in ids_m / w_m."""
        t0 = time.perf_counter()
        for k in range(out_pages(M)):
            self.d2h.read_tensor(self.pages_m[k])
        t1 = time.perf_counter()
        hn, logits = unblock_m(self.pages_m.numpy().reshape(-1), M)
        for m in range(M):
            self.ids_m[m], self.w_m[m] = lane.route(logits[m])
        lane.moe_m(layer, hn, self.ids_m[:M], self.w_m[:M], self.out_m[:M])
        t2 = time.perf_counter()
        for m in range(M):
            self.h2d.write_tensor(self.replies_m[m])
        t3 = time.perf_counter()
        self.t += (t1 - t0, t2 - t1, t3 - t2)
        if self.trace is not None:
            self.trace.append(("xm", layer, t0, t1, t2, t3))

    def serve_m_card(self, lane, layer: int, M: int, slot_of: np.ndarray | None, card_max: int) -> int:
        """serve_m with the card's share (the M-row program with CARD_EXPERTS, fused/dexp_dm.h): the first reply = the
        card's routes (its cached experts among the rows' routes, first seen first, at most card_max: word 0 n, words
        1..n the slots, word RQ_W + NC_M m + j row m's weight for expert j), then row m's lane sum of the rest as reply
        page m: each row's lane routes padded to a common k with zero-weight uses of other lane experts (each row the
        bits of its lane routes alone). host_loop.cpp's kind 5 to the bit. Returns n."""
        t0 = time.perf_counter()
        for k in range(out_pages(M)):
            self.d2h.read_tensor(self.pages_m[k])
        t1 = time.perf_counter()
        hn, logits = unblock_m(self.pages_m.numpy().reshape(-1), M)
        for m in range(M):
            self.ids_m[m], self.w_m[m] = lane.route(logits[m])
        words = self.reply_np.view(np.uint32)
        words[:RQ_BYTES // 4] = 0
        card = {}
        for e in (self.ids_m[:M].reshape(-1) if slot_of is not None else ()):
            if len(card) == min(card_max, NC_M):
                break
            if slot_of[e] >= 0 and int(e) not in card:
                card[int(e)] = len(card)
                words[1 + card[int(e)]] = slot_of[e]
        words[0] = len(card)
        lane_rows = []
        for m in range(M):
            lane_rows.append([(int(e), w) for e, w in zip(self.ids_m[m], self.w_m[m]) if int(e) not in card])
            for e, w in zip(self.ids_m[m], self.w_m[m]):
                if int(e) in card:
                    words[RQ_W + NC_M * m + card[int(e)]] = np.float32(w).view(np.uint32)
        self.h2d.write_tensor(self.reply)
        k = max(len(r) for r in lane_rows)
        if k == 0:
            self.out_m[:M] = 0
        else:
            ids, w = np.zeros((M, k), np.int32), np.zeros((M, k), np.float32)
            for m, row in enumerate(lane_rows):
                have = [e for e, _ in row]
                for r in lane_rows:
                    for e, _ in r:
                        if len(have) < k and e not in have:
                            have.append(e)
                ids[m] = have
                w[m, :len(row)] = [v for _, v in row]
            lane.moe_m(layer, hn, ids, w, self.out_m[:M])
        t2 = time.perf_counter()
        for m in range(M):
            self.h2d.write_tensor(self.replies_m[m])
        t3 = time.perf_counter()
        self.t += (t1 - t0, t2 - t1, t3 - t2)
        if self.trace is not None:
            self.trace.append(("xmc", layer, t0, t1, t2, t3))
        return len(card)

    def serve_card(self, lane, layer: int, slot_of: np.ndarray | None, card_max: int) -> int:
        """One exchange with the card's share of the experts (a program with CARD_EXPERTS, fused/dexp_dm.h): the page
        in; the routing; the first reply = the card's routes (resident experts, at most card_max, in routing order:
        words 0, 1..10, 11..20); the CPU lane's experts for the rest; the second reply = their sum. Returns the card's
        count. The lane's functions are called directly on preallocated buffers (cpulane.Lane wraps the same)."""
        t0 = time.perf_counter()
        self.d2h.read_tensor(self.page)
        t1 = time.perf_counter()
        lib, ids, w, words = lane.lib, self.ids, self.w, self.words
        lib.cl_route(self.p_logits, 10, self.p_ids, self.p_w)
        words[:21] = 0
        k = 0
        if slot_of is not None:
            s = slot_of[ids]
            card = np.flatnonzero(s >= 0)[:card_max]
            k = len(card)
            if k:
                words[1:1 + k] = s[card]
                words[11:11 + k] = w[card].view(np.uint32)
        words[0] = k
        self.h2d.write_tensor(self.reply)
        if k < 10:
            np.left_shift(self.hn16, 16, out=self.hn_bits, dtype=np.uint32)   # hn_f, bf16 -> fp32
            if k:
                self.cpu[:] = True
                self.cpu[card] = False
                np.compress(self.cpu, ids, out=self.ids_cpu[:10 - k])
                np.compress(self.cpu, w, out=self.w_cpu[:10 - k])
                rc = lib.cl_moe(layer, self.p_hn, 10 - k, self.p_ids_cpu, self.p_w_cpu, self.p_out)
            else:
                rc = lib.cl_moe(layer, self.p_hn, 10, self.p_ids, self.p_w, self.p_out)
            if rc:
                raise ValueError(f"cl_moe: layer {layer} not loaded")
        else:
            self.reply_np[:D] = 0
        t2 = time.perf_counter()
        self.h2d.write_tensor(self.reply)
        t3 = time.perf_counter()
        self.t += (t1 - t0, t2 - t1, t3 - t2)
        if self.trace is not None:
            self.trace.append(("x", layer, t0, t1, t2, t3))
        return k

    def serve_select(self, pos: int, il: int = -1) -> tuple[int, int]:
        """An attention layer's selection exchange (attention program 1 with the link): the card's pool scores up to
        the token's block in, a page per 2048 pools (fattn1_reader.cpp's), program 2's selection words out (the card
        writes them into program 2's selection tensor)."""
        t0 = time.perf_counter()
        for k in range(pos // 4 * 4 // OUT_PAGE + 1):
            self.d2h.read_tensor(self.score_pages[k])
        t1 = time.perf_counter()
        nsel, own = select_words(pos, self.scores, self.reply_np.view(np.uint32))
        t2 = time.perf_counter()
        self.h2d.write_tensor(self.reply)
        t3 = time.perf_counter()
        self.t += (t1 - t0, t2 - t1, t3 - t2)
        if self.trace is not None:
            self.trace.append(("s", il, t0, t1, t2, t3))
        return nsel, own

    def serve_select_m(self, pos: int, M: int, il: int = -1) -> np.ndarray:
        """A verify pass's selection exchange (attention program 1 at M rows): row m's pool scores as serve_select's
        at position pos + m (its pages), select_words per row, program 2's union words out (g3_attn2.union_words).
        Returns the words (a view of the reply)."""
        from g3_attn2 import union_words
        t0 = time.perf_counter()
        sc = []
        for m in range(M):
            for k in range((pos + m) // 4 * 4 // OUT_PAGE + 1):
                self.d2h.read_tensor(self.score_pages[k])
            sc.append(self.scores.copy())
        t1 = time.perf_counter()
        words, rows = np.zeros(OUT_PAGE // 4 * 2, np.uint32), []
        for m in range(M):
            n, own = select_words(pos + m, sc[m], words)
            rows.append((words[:n].copy(), own))
        w = self.reply_np.view(np.uint32)
        u = union_words(pos, rows, M_MAX)
        w[:] = 0
        w[:len(u)] = u
        t2 = time.perf_counter()
        self.h2d.write_tensor(self.reply)
        t3 = time.perf_counter()
        self.t += (t1 - t0, t2 - t1, t3 - t2)
        if self.trace is not None:
            self.trace.append(("sm", il, t0, t1, t2, t3))
        return w

    def bind_loop(self, loop) -> None:
        """Point the C++ host loop (cpulane.Loop) at these sockets: tt-metal's D2HSocket / H2DSocket objects behind
        the Python ones and their read() / write(), called directly. Checked: the D2H FIFO's size, the H2D mode, one
        mesh device for both."""
        maps = [line.split()[-1] for line in open("/proc/self/maps") if line.rstrip().endswith(".so")]
        # the release build's libtt_metal.so; the Tracy build (TT_MOEOFFLOAD_TTNN=tracy) links tt-metal into _ttnncpp.so
        lib = next((m for m in maps if m.endswith("/libtt_metal.so")), None) or next(
            (m for m in maps if m.endswith("/_ttnncpp.so")), None)
        if lib is None:
            raise RuntimeError("bind_loop: neither libtt_metal.so nor _ttnncpp.so is loaded")
        tm = ctypes.CDLL(lib)
        ns = "2tt8tt_metal11distributed"
        fn = lambda name: ctypes.cast(tm[name], ctypes.c_void_p).value  # noqa: E731
        d2h, h2d = cpp_object(self.d2h), cpp_object(self.h2d)

        class Span(ctypes.Structure):
            _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]
        fifo, mode = tm[f"_ZNK{ns}9D2HSocket9host_fifoEv"], tm[f"_ZNK{ns}9H2DSocket12get_h2d_modeEv"]
        dev_d, dev_h = tm[f"_ZNK{ns}9D2HSocket15get_mesh_deviceEv"], tm[f"_ZNK{ns}9H2DSocket15get_mesh_deviceEv"]
        fifo.restype, mode.restype, dev_d.restype, dev_h.restype = Span, ctypes.c_uint8, ctypes.c_void_p, ctypes.c_void_p
        for f in (fifo, mode, dev_d, dev_h):
            f.argtypes = [ctypes.c_void_p]
        got = (fifo(d2h).size, mode(h2d), dev_d(d2h))
        if got != (FIFO, 1, dev_h(h2d)) or not got[2]:      # 1: H2DMode::DEVICE_PULL
            raise RuntimeError(f"bind_loop: the sockets' C++ objects do not check out ({got})")
        loop.bind(d2h, h2d, fn(f"_ZN{ns}9D2HSocket4readEPvjb"), fn(f"_ZN{ns}9H2DSocket5writeEPvj"))


def cpp_object(obj) -> int:
    """The C++ object behind a nanobind instance (ttnn's bindings). nanobind's nb_inst: PyObject_HEAD, int32 offset,
    then the bits state:2 (2 ready), direct:1; the object at self + offset, or (not direct) the pointer stored there."""
    base = id(obj)
    off, bits = ctypes.c_int32.from_address(base + 16).value, ctypes.c_uint32.from_address(base + 20).value
    if bits & 3 != 2:
        raise RuntimeError(f"cpp_object: {type(obj).__name__} is not a ready nanobind instance")
    return base + off if bits >> 2 & 1 else ctypes.c_void_p.from_address(base + off).value


def select_words(pos: int, scores: np.ndarray, words: np.ndarray, nsel_max: int = 512) -> tuple[int, int]:
    """g4_host.Host.select's rule (llama.cpp's QSA: the top min(512, complete pools), the incomplete tail by the
    own-block core) as program 2's selection words: pool ids in words[:512], nsel at 512, own_valid at 513. The
    pools come unordered (np.argpartition: the attention sums over them; ~10x faster than torch.topk + sort)."""
    complete, own = (pos + 1) // 4, pos // 4
    if complete <= nsel_max:
        top = np.arange(complete, dtype=np.uint32)
    else:
        top = np.argpartition(-scores[:complete], nsel_max - 1)[:nsel_max].astype(np.uint32)
    if pos % 4 != 3:
        own_valid = pos % 4 + 1
    else:
        hit = np.nonzero(top == own)[0]
        if len(hit):
            top = np.delete(top, hit[0])
            own_valid = 4
        else:
            own_valid = 0
    words[:nsel_max + 16] = 0
    words[:len(top)] = top
    words[nsel_max], words[nsel_max + 1] = len(top), own_valid
    return len(top), own_valid


class Cache:
    """The card's expert cache (g5_cache.py's $TT_MOEOFFLOAD_DATA/g5_cache): per layer the resident experts' card form in
    card memory and the expert -> slot map (-1: CPU only); card_max caps the card's share of a token's 10 routes
    (g5_experts.py: ~11 us per card expert; in the token loop the lane's exchange costs ~120 us plus ~3-4 us per
    expert, g4_prof.py 2026-10-24, so the card's share saves the lane little); at M rows the card's experts of
    an exchange's union (at most NC_M), part_m their partial rows. The DeltaNet layers' only (`layers` narrows them):
    the attention programs take no card experts yet. slots, gib: the cache's size in card memory."""

    def __init__(self, dev, folder: str = os.path.join(DATA, "g5_cache"), layers=None,
                 card_max: int = 4):
        t0 = time.time()
        self.card_max, self.tensors, self.slot_of = card_max, {}, {}
        tile = lambda a, dt: ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)).reshape(1, 1, -1, 32),  # noqa
                                             dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".npz"):
                continue
            il = int(f.split(".")[1])
            if is_attention(il) or (layers is not None and il not in layers):
                continue
            z = np.load(os.path.join(folder, f))
            m = np.full(512, -1, np.int32)
            m[z["experts"]] = np.arange(len(z["experts"]), dtype=np.int32)
            self.slot_of[il] = m
            self.tensors[il] = dict(gu_codes=tile(z["gu_codes"].view(np.int32), ttnn.uint32),
                                    gu_scales=tile(z["gu_scales"], ttnn.float32),
                                    d_vals=tile(z["d_vals"].astype(np.float32), ttnn.bfloat8_b),
                                    d_scales=tile(z["d_scales"], ttnn.float32))
        self.part = ttnn.from_torch(torch.zeros(1, 1, 800, 32), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                    device=dev)
        # M rows (fused/dexp_dm.h): NC_M experts' 80 partial row blocks of 4 rows each
        self.part_m = ttnn.from_torch(torch.zeros(1, 1, NC_M * 80, 128), dtype=ttnn.float32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.seconds = time.time() - t0
        self.slots = sum(int((m >= 0).sum()) for m in self.slot_of.values())
        self.gib = self.slots * SLOT_BYTES / 2**30


class DeltaRun:
    """Consecutive DeltaNet layers (their Layer objects, g3_layer.py) as one program: a layer table in card memory,
    the conv and DeltaNet states updated in place, X from and back into the model's X tensors, the host link after
    each layer's router when `link` is given. M-row layers (m_max > 1): X the M-row layout's tensors (Layer.x_tensors),
    `step` the pass's step tensor (g3_layer.step_words: M, n_keep, parity), each exchange Link.serve_m's (with the
    cache Link.serve_m_card's: the card's share at M rows)."""

    def __init__(self, dev, layers: list[Layer], x32, x16, link: Link | None = None, cache: Cache | None = None,
                 step=None):
        self.layers = [L.layer for L in layers]
        self.card = cache is not None and link is not None
        if layers[0].m_max > 1:
            assert step is not None, "M rows: a step tensor"
            layers[0].step_t = step
        for L in layers:
            L.card = cache.tensors.get(L.layer) if self.card else None
        rows = torch.zeros(len(layers), 32, dtype=torch.int64)
        for i, L in enumerate(layers):
            rows[i, :len(ROW)] = torch.tensor([addr(L, k) for k in ROW])
        self.table = ttnn.from_torch(rows.to(torch.int32).reshape(1, 1, len(layers), 32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        A = layers[0]
        A.x32_t, A.x16_t, A.inplace = x32, x16, True
        A.table_addr = self.table.buffer_address()
        A.defines = list(link.defines) if link else []
        if link and os.environ.get("G4_NO_A_PREFETCH") == "1":   # flayer_reader.cpp's A prefetch off (comparisons)
            A.defines.append(("NO_A_PREFETCH", "1"))
        if self.card:
            A.card_on, A.part = True, cache.part_m if A.m_max > 1 else cache.part
        self.run = A.program(len(layers))
        self.keep = layers                     # their tensors must outlive the program


def work(segs, m_rows: bool = False, card: bool = False) -> list[tuple[str, int]]:
    """The host's work for a token's programs (segments()), in launch order: ("x", il) layer il's expert exchange,
    ("s", il) attention layer il's selection (program 1's), then its exchange (program 2's). m_rows: a verify pass's
    ("xm" and "sm": the M-row kinds; card: the DeltaNet runs' "xmc")."""
    out = []
    x, sel = ("xm", "sm") if m_rows else ("x", "s")
    for kind, what in segs:
        if kind == "run":
            out += [("xmc" if m_rows and card else x, il) for il in what]
        elif kind == "attn":
            out += [(sel, what), (x, what)]
    return out


def is_attention(il: int) -> bool:
    return (il + 1) % 4 == 0                  # qwen4exp: full_attention_interval 4


def segments(n_layers: int = 48):
    """The token's programs in order: DeltaNet runs between the attention layers, the PLE block before layer 1."""
    out, run = [], []
    for il in range(n_layers):
        if il == 1:
            if run:
                out.append(("run", run))
            out.append(("ple", 1))
            run = []
        if is_attention(il):
            if run:
                out.append(("run", run))
            out.append(("attn", il))
            run = []
        else:
            run.append(il)
    if run:
        out.append(("run", run))
    return out


class Attention:
    """One QSA attention layer as two programs (g3_attn1.py, g3_attn2.py) sharing its caches: program 1's Q, gates
    and inject weights feed program 2; the host takes the top 512 pools from program 1's scores in between. m_max > 1:
    the M-row programs (X the block layout, the pass's rows from the step tensor), the host's reply each row's
    selection as their union (Link.serve_select_m)."""

    def __init__(self, dev, g, rng, il: int, x32, x16, step, link: Link | None, back=False, gprof_n: int = 0,
                 m_max: int = 1):
        from g3_attn1 import Attn1
        from g3_attn2 import Attn2
        from g3_attn1 import HD, IHD, KPOOL, NKV, NPOOL, RAWK
        self.il = il
        mr = dict(pos=0, m_max=m_max) if m_max > 1 else {}
        A1 = Attn1(dev, g, rng, layer=il, back=back, zero_caches=True, **mr)
        A2 = Attn2(dev, g, rng, layer=il, back=back, zero_caches=True, **mr)
        z = lambda shape, dt, lay: ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=lay, device=dev)  # noqa
        A1.rawk = z((1, 1, RAWK, IHD), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)   # the ring (row p % RAWK)
        A1.pooled = z((1, 1, NPOOL * 4, 32), ttnn.float32, ttnn.TILE_LAYOUT)
        A1.x32_t, A1.x16_t, A1.step = x32, x16, step
        A2.x32_t, A2.x16_t, A2.step_t = x32, x16, step
        A2.kc, A2.vc = A1.kc, A1.vc
        A2.q_t, A2.g_t, A2.wa_t = A1.q, A1.g_out, A1.chk["w"]
        A2.inplace = True
        A2.defines = list(link.defines) if link else []
        if link:  # the selection over the link: program 1 sends its scores, writes the host's reply into A2.sel_t
            A1.defines, A1.sel_out = list(link.defines), A2.sel_t
        self.link_select = link is not None   # the step then calls link.serve_select(pos) after program 1
        if gprof_n:                           # grid.h's per-core clocks (g4_phases.py)
            for P in (A1, A2):
                P.gprof, P.gprof_n = profile_tensor(dev, gprof_n), gprof_n
        self.A1, self.A2 = A1, A2
        self.p1, self.p2 = A1.program(), A2.program()
        self.words = np.zeros(A2.sel_t.shape[-1], np.uint32)

    def select(self, pos: int) -> tuple[int, int]:
        """The host's selection from program 1's scores (synchronizes), into program 2's selection tensor: the
        no-link path (with a link, program 1 exchanges them itself and the host calls Link.serve_select)."""
        from g3_attn1 import NPOOL
        from g4_host import Host
        sc = ttnn.to_torch(self.A1.scores).float().flatten().numpy()
        if pos % 4 == 3:   # the block the token closes: its score from the new key (program 1's word NPOOL)
            sc[pos // 4] = sc[NPOOL]
        sel, nsel, own = Host.select(pos, sc[:NPOOL])
        w = self.words
        w[:] = 0
        w[:len(sel)] = sel.astype(np.uint32)
        w[len(sel)], w[len(sel) + 1] = nsel, own
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(w.view(np.int32)).reshape(
            tuple(self.A2.sel_t.shape)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.A2.sel_t)
        return nsel, own


class Model:
    """The decode token's programs (segments()), built once; step() runs one token and returns its logits (or its
    top 64's ids: set_head)."""

    def __init__(self, dev, g, lane, n_layers: int = 48, head: bool = True, log=print, gprof_n: int = 0,
                 cache: Cache | None = None, head_mode: str = "logits", m_max: int = 1):
        """gprof_n: every program with grid.h's per-core clocks, gprof_n barriers kept (g4_phases.py; 0 none).
        cache: the card's expert cache (Cache); the DeltaNet runs then compute its share of the routed experts.
        head_mode: what a step returns (set_head). m_max = 4: the verify pass's programs (verify(); the device opened
        with g3_layer.WORKER_L1_M; with the cache WORKER_L1_MC: the DeltaNet runs' exchanges then Link.serve_m_card's,
        the host loop's kind "xmc"), no profile."""
        import time
        from g3_layer import Layer
        from g3_ple import Ple
        rng = np.random.default_rng(0)
        self.dev, self.lane, self.n_layers, self.m_max = dev, lane, n_layers, m_max
        self.cache, self.n_card = cache, {}     # n_card: per DeltaNet layer, the card's share of each exchange
        mm = m_max
        assert mm == 1 or (mm == M_MAX and not gprof_n)
        self.step_t = ttnn.from_torch(torch.zeros(1, 1, 1, 16, dtype=torch.int32) if mm == 1 else step_words(1),
                                      dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        if mm == 1:
            self.x32, self.x16 = x_tensors(dev)
        else:   # the block layout: [4 x m_max 2560], a stream a page (g3_layer.Layer.x_tensors)
            z = torch.zeros(1, 1, HC, mm * D)
            self.x32 = ttnn.from_torch(z, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            self.x16 = ttnn.from_torch(z.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                       device=dev)
        self.emb_t = ttnn.from_torch(torch.zeros(1, 1, mm, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.parity = 0                         # the next verify pass's states' parity
        self.inputs = None                      # a verify pass's host inputs (InputsM)
        self.segs = segments(n_layers)
        self.link = None
        self.progs = []
        self.profiles = []                     # with gprof_n: (kind, what, program builder) per launch, in order

        def profiled(P, kind, what):
            if gprof_n:
                P.gprof, P.gprof_n = profile_tensor(dev, gprof_n), gprof_n
                self.profiles.append((kind, what, P))
        for kind, what in self.segs:
            t0 = time.time()
            if kind == "run":
                Ls = [Layer(dev, g, rng, layer=il, back=False, m_max=mm) for il in what]
                for L in Ls:
                    L.zero_states()
                if self.link is None:
                    self.link = Link(dev, Ls[0].grid.cores[0])
                profiled(Ls[0], kind, what)
                self.progs.append(DeltaRun(dev, Ls, self.x32, self.x16, self.link, cache,
                                           step=self.step_t if mm > 1 else None))
            elif kind == "ple":
                P = Ple(dev, g, layer=what, back=False, **(dict(m_max=mm, x_blocks=True) if mm > 1 else {}))
                P.x32_t, P.x16_t, P.emb_t, P.step_t, P.inplace = self.x32, self.x16, self.emb_t, self.step_t, True
                profiled(P, kind, what)
                P.run = P.program()
                self.progs.append(P)
            else:
                self.progs.append(Attention(dev, g, rng, what, self.x32, self.x16, self.step_t, self.link,
                                            gprof_n=gprof_n, m_max=mm))
                if gprof_n:
                    self.profiles += [("attn1", what, self.progs[-1].A1), ("attn2", what, self.progs[-1].A2)]
            log(f"  {kind} {what}: {time.time() - t0:.0f} s")
        self.head, self.tid = None, None
        if head and n_layers == 48:
            from g3_head import Head
            H = Head(dev, g, back=False, **(dict(m_max=mm, x_blocks=True) if mm > 1 else {}))
            H.x32_t, H.x16_t, H.step_t = self.x32, self.x16, self.step_t
            profiled(H, "head", None)
            self.head = H
            # the top 64's host copy, made once and read into in place (ttnn.to_torch per token: ~125 us)
            self.ids_host = ttnn.from_torch(torch.zeros(tuple(H.ids_t.shape), dtype=torch.int32), dtype=ttnn.uint32,
                                            layout=ttnn.ROW_MAJOR_LAYOUT)
            self.ids_w = torch.from_dlpack(self.ids_host.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))).view(
                torch.int32).numpy().reshape(H.m_max, 2 * TOPK)
            self.set_head(head_mode)

    def set_head(self, mode: str) -> None:
        """The head's output, and so what step() and step_traced() return: "logits" the logits (torch, fp32 [VOCAB]);
        "ids" the greedy id (int), with top=True the top 64 (top64()); "both" the logits, the top 64 by top64() (the
        parities' checks). A captured token is captured again with the new head."""
        H = self.head
        H.run = H.program(**HEAD_MODES[mode])
        self.head_mode = mode
        if self.tid is not None:
            H.run()                            # its first launch loads its binaries (not allowed in a capture)
            ttnn.release_trace(self.dev, self.tid)
            self.capture_trace()

    def top64(self) -> tuple[np.ndarray, np.ndarray]:
        """The last token's top 64 from the head (modes ids, both): ids (int32, best first: [0] the argmax, a tie to
        the lowest id) and their logits (fp32), views of one host buffer read in place (valid until the next read)."""
        ttnn.copy_device_to_host_tensor(self.head.ids_t, self.ids_host, blocking=True)
        w = self.ids_w[0]
        return w[:TOPK], w[TOPK:].view(np.float32)

    def result(self, top: bool = False):
        """What a step returns (set_head): the logits, the greedy id, or with top the top 64."""
        if self.head_mode != "ids":
            return self.head.get_logits()
        ids, vals = self.top64()
        return (ids, vals) if top else int(ids[0])

    def step(self, x0: np.ndarray, emb: np.ndarray, pos: int, after=None, prof: dict | None = None,
             top: bool = False):
        """One token: X's init (the embedding x 4), the PLE input, the position; then every program, the host
        serving the link (the CPU lane's experts) and the attention selections. after(kind, what) is called after
        each segment, synchronized (the checks' hook). prof accumulates the host's wall seconds per stage.
        Returns result(top) (the logits, the greedy id or the top 64: set_head) or None without the head."""
        dev = self.dev
        clock = [time.perf_counter()]

        def tick(name):
            if prof is not None:
                now = time.perf_counter()
                prof[name] = prof.get(name, 0.0) + now - clock[0]
                clock[0] = now
        set_x(dev, self.x32, self.x16, torch.from_numpy(np.asarray(x0)))
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(np.asarray(emb, np.float32)).to(
            torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), self.emb_t)
        st = torch.zeros(1, 1, 1, 16, dtype=torch.int32)
        st[..., 0] = pos
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(st, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                        self.step_t)
        tick("inputs")
        # the token as launches, each with the host's work after it: ("x", il) an expert exchange, ("s", attention)
        # the selection over the link, ("sync", attention) the selection after a synchronize. The next launch is
        # queued before the host serves the current one (the card runs on into it), except across a synchronize:
        # a synchronize with a queued program that waits on the host would never return.
        items = []
        for (kind, what), p in zip(self.segs, self.progs):
            if kind == "run":
                items.append((p.run, [("x", il) for il in what], kind, what, "deltanet runs"))
            elif kind == "ple":
                items.append((p.run, [], kind, what, "ple"))
            elif getattr(p, "link_select", False):
                items.append((p.p1, [("s", p)], None, None, "attention p1 + selection"))
                items.append((p.p2, [("x", what)], kind, what, "attention p2"))
            else:
                items.append((p.p1, [("sync", p)], None, None, "attention p1 + selection"))
                items.append((p.p2, [("x", what)], kind, what, "attention p2"))
        if self.head is not None:
            items.append((self.head.run, [], "head", None, "head"))
        ahead = after is None
        trace = self.link.trace if self.link is not None else None
        launched = 0

        def launch_at(j):
            t0 = time.perf_counter()
            items[j][0]()
            if trace is not None:
                trace.append(("launch", items[j][4], t0, time.perf_counter()))
        for i, (launch, work, kind, what, stage) in enumerate(items):
            if launched == i:
                launch_at(i)
                launched += 1
            if ahead and launched == i + 1 and i + 1 < len(items) and not any(w[0] == "sync" for w in work):
                launch_at(i + 1)
                launched += 1
            for w in work:
                if w[0] == "x":
                    self.serve_x(w[1])
                elif w[0] == "s":
                    self.link.serve_select(pos, w[1].il)
                else:
                    w[1].select(pos)
            tick(stage)
            if after and kind in ("run", "ple", "attn"):
                ttnn.synchronize_device(dev)
                after(kind, what)
        if self.head is None:
            ttnn.synchronize_device(dev)
            return None
        out = self.result(top)
        tick("head + logits")
        return out

    def capture(self) -> None:
        """Capture the token's launches as one trace (needs the link's selection path and the head): launching the
        26 programs from Python costs the host ~7 ms per token (g4_prof.py, 2026-10-24), during which it cannot
        serve the card's exchanges. step_traced() then replays it (m_max > 1: verify(), the C++ loop only)."""
        assert self.head is not None and all(getattr(p, "link_select", True) for p in self.progs)
        if self.m_max > 1:
            from cpulane import Loop
            self.work = work(self.segs, m_rows=True, card=self.cache is not None)
            self.loop = Loop(self.lane)
            self.loop.plan([k for k, _ in self.work], [il for _, il in self.work], *self.card_plan())
            self.warm_plan()
            self.link.bind_loop(self.loop)
            self.capture_trace()
            return
        self.work = work(self.segs)            # the host's work after each program, in launch order
        self.loop = None                      # the C++ host loop (G4_PY_LOOP=1: Link's Python methods)
        if os.environ.get("G4_PY_LOOP") != "1":
            from cpulane import Loop
            self.loop = Loop(self.lane)
            card = self.cache is not None
            kinds = ["card" if k == "x" and card and not is_attention(il) else k for k, il in self.work]
            self.loop.plan(kinds, [il for _, il in self.work], self.cache.slot_of if card else None,
                           self.cache.card_max if card else 0)
            self.link.bind_loop(self.loop)
            self.inputs = Inputs(self.lane.lib)
        self.capture_trace()

    def capture_trace(self) -> None:
        """The token's launches, the head's as set_head left it, captured as one trace (self.tid)."""
        launches = []
        for (kind, what), p in zip(self.segs, self.progs):
            launches += [p.run] if kind in ("run", "ple") else [p.p1, p.p2]
        launches.append(self.head.run)
        ttnn.synchronize_device(self.dev)
        self.tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        for launch in launches:
            launch()
        ttnn.end_trace_capture(self.dev, self.tid, cq_id=0)

    def step_traced(self, x0: np.ndarray, emb: np.ndarray, pos: int, top: bool = False):
        """step() through the captured trace: the inputs, one replay, the host serving every exchange in order,
        then result(top)."""
        dev = self.dev
        if self.loop is None:
            set_x(dev, self.x32, self.x16, torch.from_numpy(np.asarray(x0)))
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(np.asarray(emb, np.float32)).to(
                torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), self.emb_t)
            st = torch.zeros(1, 1, 1, 16, dtype=torch.int32)
            st[..., 0] = pos
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(st, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                            self.step_t)
        else:
            self.inputs.fill(x0, emb, pos)
            for h, d in zip(self.inputs.host, (self.x32, self.x16, self.emb_t, self.step_t)):
                ttnn.copy_host_to_device_tensor(h, d)
        trace = self.link.trace
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, self.tid, cq_id=0, blocking=False)
        if trace is not None:
            trace.append(("launch", "trace", t0, time.perf_counter()))
        if self.loop is None:
            for kind, il in self.work:
                if kind == "x":
                    self.serve_x(il)
                else:
                    self.link.serve_select(pos, il)
            return self.result(top)
        L = self.loop
        L.token(pos, stamps=trace is not None)
        # the clocks into Link.trace and the card's counts, while the card runs the head
        if trace is not None:
            ts = L.stamps * 1e-9
            trace += [(k, il, *ts[i]) for i, (k, il) in enumerate(self.work)]
        if self.cache is not None:
            for i, (k, il) in enumerate(self.work):
                if k == "x" and not is_attention(il):
                    self.n_card.setdefault(il, []).append(int(L.n_card[i]))
        return self.result(top)

    def put_pass(self, x0s: np.ndarray, embs: np.ndarray, pos: int, n_keep: int) -> int:
        """A verify pass's inputs into the card's tensors (in queue): the rows' X inits x0s [M, 4 x 2560] and PLE
        inputs embs [M, 2560], the step words (pos, M, n_keep, the parity, Mp). Returns M."""
        from g3_attn1 import CTX
        M = len(x0s)
        assert 1 <= M <= self.m_max and 1 <= n_keep and pos + M <= CTX
        if self.inputs is None:
            self.inputs = InputsM(self.m_max)
        self.inputs.fill(x0s, embs, pos, M, n_keep, self.parity)
        for h, d in zip(self.inputs.host, (self.x32, self.x16, self.emb_t, self.step_t)):
            ttnn.copy_host_to_device_tensor(h, d)
        return M

    def warm(self, x0s: np.ndarray, embs: np.ndarray, pos: int, n_keep: int = 1, after=None):
        """A verify pass untraced (each program's first launch loads its binaries, which a capture cannot), the C++
        loop serving each program's exchanges after its launch; then capture() may run. after(kind, what) is called
        after each segment, synchronized (the checks' hook, as step()'s). Advances the parity; returns result_m(M)."""
        from cpulane import Loop
        M = self.put_pass(x0s, embs, pos, n_keep)
        loop = Loop(self.lane)
        self.link.bind_loop(loop)
        loop.rows(M)
        items = work(self.segs, m_rows=True, card=self.cache is not None)
        k = 0
        for (kind, what), p in zip(self.segs, self.progs):
            for launch in [p.run] if kind in ("run", "ple") else [p.p1, p.p2]:
                launch()
                n = len(what) if kind == "run" else 1 if kind == "attn" else 0
                if n:
                    loop.plan([w for w, _ in items[k:k + n]], [il for _, il in items[k:k + n]], *self.card_plan(loop))
                    loop.token(pos, stamps=False)
                    k += n
            if after:
                ttnn.synchronize_device(self.dev)
                after(kind, what)
        self.head.run()
        self.parity ^= 1
        return self.result_m(M)

    def verify(self, x0s: np.ndarray, embs: np.ndarray, pos: int, n_keep: int, top: bool = True):
        """One verify pass through the captured trace: rows m < M = len(x0s) at positions pos + m (row 0 the last
        accepted token, then the drafts), n_keep the rows the previous pass accepted (1 at a sequence's start).
        Returns result_m(M, top). Advances the parity."""
        M = self.put_pass(x0s, embs, pos, n_keep)
        L, trace = self.loop, self.link.trace
        L.rows(M)
        t0 = time.perf_counter()
        ttnn.execute_trace(self.dev, self.tid, cq_id=0, blocking=False)
        if trace is not None:
            trace.append(("launch", "trace", t0, time.perf_counter()))
        L.token(pos, stamps=trace is not None)
        self.parity ^= 1
        if trace is not None:
            ts = L.stamps * 1e-9
            trace += [(k, il, *ts[i]) for i, (k, il) in enumerate(self.work)]
        if self.cache is not None:
            for i, (k, il) in enumerate(self.work):
                if k == "xmc":
                    self.n_card.setdefault(il, []).append(int(L.n_card[i]))
        return self.result_m(M, top)

    def warm_plan(self) -> None:
        """G4_WARM=n (2026-11-08, default 30, 0 off): after each of a pass's M-row exchanges the lane's threads read the
        n experts the next one likeliest needs while the card computes it (Loop.warm; the census's layer -> layer + 1
        transitions with the recent rows; G4_WARM_CENSUS another census trace, "" the recent rows alone). 4-row
        passes at 16K: 40.6 -> 38.7 ms (20: 38.9, 40: 39.0); served decode 4-5% faster (check_card spec serve)."""
        n = int(os.environ.get("G4_WARM", "30"))
        if not n:
            return
        from cpulane import transitions
        path = os.environ.get("G4_WARM_CENSUS", os.path.join(DATA, "census", "qwen3.8-flash-next", "census.trace"))
        self.loop.warm(n, transitions(path) if path else None)

    def card_plan(self, loop=None) -> tuple:
        """The host loop's card arguments (Loop.plan's slot_of, card_max): the cache's, or none. G4_CARD_BALANCE=1
       : kind 5 splits each layer's resident routes so the card and the lane take about as long
        (Loop.card_costs; G4_CARD_COSTS = lane us an expert, a use, card us an expert, fixed a layer)."""
        if self.cache is not None and os.environ.get("G4_CARD_BALANCE") == "1":
            c = [float(v) for v in os.environ.get("G4_CARD_COSTS", "11.6,1.6,15.1,41").split(",")]
            (loop or self.loop).card_costs(True, *c)
        return (self.cache.slot_of, self.cache.card_max) if self.cache is not None else (None, 0)

    def top64_m(self, M: int) -> tuple[np.ndarray, np.ndarray]:
        """The last pass's rows' top 64 (head modes ids, both): ids [M, 64] (int32, best first: [:, 0] the argmax)
        and their logits (fp32), views of one host buffer read in place."""
        ttnn.copy_device_to_host_tensor(self.head.ids_t, self.ids_host, blocking=True)
        w = self.ids_w[:M]
        return w[:, :TOPK], w[:, TOPK:].view(np.float32)

    def result_m(self, M: int, top: bool = True):
        """What a verify pass returns (set_head): head mode ids, the rows' top 64 (top64_m) or their greedy ids
        (int32 [M]); logits or both, the rows' logits (torch fp32 [M, VOCAB]; both: top64_m(M) beside them)."""
        if self.head_mode != "ids":
            return ttnn.to_torch(self.head.logits).reshape(self.m_max, -1)[:M]
        ids, vals = self.top64_m(M)
        return (ids, vals) if top else ids[:, 0].copy()

    def reset(self) -> None:
        """Every state a token or a pass leaves behind back to a sequence's start, in place (the programs hold the
        addresses): the DeltaNet layers' conv and recurrent states (M rows: Layer.set_states_m's, parity 0 with a
        pending page that commits nothing), the PLE ring (not the attention caches: below); the next pass's parity 0."""
        def zero(t):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.zeros(tuple(t.shape)), dtype=t.dtype,
                                                            layout=t.layout), t)
        for p in self.progs:
            if isinstance(p, DeltaRun):
                for L in p.keep:
                    if L.m_max > 1:
                        L.zero_states()
                    else:
                        zero(L.cstate)
                        zero(L.S)
            elif isinstance(p, Attention):
            # the attention caches are not zeroed: every read is of a cell at or before the position, written in this
            # sequence first (program 1's later pools are scored but never selected: select_words takes
            # scores[:complete]; masked cells add -4096): zeroing them took 22.8 s at 256K
                continue
            else:                                                      # the PLE block
                zero(p.ring_t)
        self.parity = 0
        ttnn.synchronize_device(self.dev)

    def serve_x(self, il: int) -> None:
        """Layer il's expert exchange: with the card's share where its program computes one (a DeltaNet run with the
        cache), else all ten on the CPU lane."""
        if self.cache is not None and not is_attention(il):
            k = self.link.serve_card(self.lane, il, self.cache.slot_of.get(il), self.cache.card_max)
            self.n_card.setdefault(il, []).append(k)
        else:
            self.link.serve(self.lane, il)

    def card_share(self) -> tuple[float, dict]:
        """The card's routes per DeltaNet exchange (of 10; M rows: its experts of the rows' union) since the build or
        the last call: the mean over the exchanges, and per layer; then the counts are cleared."""
        every = [k for v in self.n_card.values() for k in v]
        share = (float(np.mean(every)) if every else 0.0,
                 {il: round(float(np.mean(v)), 3) for il, v in sorted(self.n_card.items())})
        self.n_card = {}
        return share

    def x(self) -> torch.Tensor:
        return ttnn.to_torch(self.x32).double().reshape(HC, D)

    def x_m(self, M: int) -> torch.Tensor:
        """A verify pass's X (the block layout) as rows: float64 [M, 4 x 2560]."""
        from g3_layer import unblocks
        st = ttnn.to_torch(self.x32).double().reshape(HC, -1)
        return torch.cat([unblocks(st[s], D // 32, mp_of(M)) for s in range(HC)], 1)[:M]


def rows_of(blk: np.ndarray, M: int) -> np.ndarray:
    """X's block layout (fp32 [4, m_max 2560]) -> its first M rows, fp32 [M, 4 x 2560] (InputsM.fill's undone)."""
    mp = mp_of(M)
    return blk[:, :mp * D].reshape(HC, D // 32, 2, mp, 16).transpose(3, 0, 1, 2, 4).reshape(mp, HC * D)[:M]


class Spec:
    """Speculative decoding on the card: the verify model V (Model(m_max = 4), captured, head mode ids) and the
    MTP draft layer C (g6_mtp.MtpCard, through its public interface) in one process and one queue. A pass at position
    p verifies rows [x, d_0 .. d_{T-2}] (x the last pass's bonus token), keeps n = accept() rows, then the MTP catches
    its K/V up over the kept cells p .. p + n - 1 (cell c: the target's h_nextn at c, read from X after layer 47, with
    the token at c + 1) and drafts up to rows - 1 tokens for the next pass, one traced step at a time (each step's
    draft read from its head's top 64 before the next is launched, so min_p can end the chain). The prompt runs as
    teacher-forced 4-row passes with the MTP's K/V caught up alongside (no drafts until its last pass).

    The window: rows - 1 MTP drafts, or with min_p Strata's --spec-min-p rule (the leading drafts whose probability,
    approximated by a softmax over the head's top 64, is >= min_p), or with min_q the leading drafts whose cumulative
    probability (all of them right) is >= min_q (a row costs ~6 ms against ~17 ms a token: replay with the
    card's pass costs puts min_q 0.4-0.5 ~8% above 4 fixed rows, ~4% above min_p 0.3). Past the MTP's dense
    attention's window (g6_mtp.WINDOW cells, its last) there were no drafts until 2026-11-06."""

    def __init__(self, V: Model, C, host):
        assert V.m_max == M_MAX and V.head_mode == "ids" and V.tid is not None
        self.V, self.C, self.host = V, C, host

        def host_of(t):                        # a host copy of card tensor t, read into in place, and its numpy view
            h = ttnn.from_torch(torch.zeros(tuple(t.shape), dtype=torch.int32 if t.dtype == ttnn.uint32 else
                                            torch.float32), dtype=t.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            return h, torch.from_dlpack(h.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))).view(torch.int32).numpy()
        self.x_host, x = host_of(V.x32)
        self.x_np = x.view(np.float32).reshape(HC, -1)
        self.d_host = {n: host_of(h.H.ids_t) for n, h in C.heads.items()}
        self.tids = {}
        self.stats = []                        # the last generate()'s decode passes: (p, T, n, verify s, draft s,
        self.t_set = 0.0                       # the draft's h_nextn read-back + set_pass s)

    def capture(self) -> None:
        """Each head's draft steps 0 .. S_MAX - 2 as a trace each (their first launches untraced), then the MTP's K/V
        zeroed."""
        from g6_mtp import S_MAX
        C, dev = self.C, self.V.dev
        for name in C.heads:
            C.use_head(name)
            C.set_pass(0, np.zeros((1, HC * D), np.float32), [0], S_MAX - 1)
            for k in range(S_MAX - 1):
                C.launch(k)
            ttnn.synchronize_device(dev)
            for k in range(S_MAX - 1):
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                C.launch(k)
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                self.tids[(name, k)] = tid
        tid = ttnn.begin_trace_capture(dev, cq_id=0)       # the K/V-only catch-up (its programs launched above)
        C.launch(0, kv_only=True)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        self.tids[("kv", 0)] = tid
        C.zero_kv()

    def inputs(self, toks, prev, p: int):
        """The rows' X inits and PLE inputs for toks at positions p .. (prev: the tokens before p, at least the PLE's
        predecessors)."""
        h, tail = self.host, list(prev[-16:])
        return (np.stack([h.embed(t) for t in toks]),
                np.stack([h.ple(t, tail + list(toks[:m])) for m, t in enumerate(toks)]))

    def draft(self, head: str, p: int, M: int, n: int, nxt, steps: int, min_p: float = 0.0, min_q: float = 0.0):
        """After a pass of M rows at p that kept n: the MTP's catch-up over cells p .. p + n - 1 (nxt: the tokens at
        p + 1 .. p + n), then up to `steps` chained drafts (0: the K/V only). Returns (drafts, probabilities)."""
        from g6_mtp import CTX_DENSE, VOCAB
        if p + n > CTX_DENSE:
            return [], []
        t0 = time.perf_counter()
        ttnn.copy_device_to_host_tensor(self.V.x32, self.x_host, blocking=True)
        return self.chain(head, p, rows_of(self.x_np, M)[:n], nxt, steps, min_p, min_q, t0)

    def chain(self, head: str, c0: int, R: np.ndarray, nxt, steps: int, min_p: float = 0.0, min_q: float = 0.0,
              t0: float | None = None):
        """The MTP's catch-up over cells c0 .. c0 + n - 1 (R [n <= 4, 10240]: the target's final residual streams
        there; nxt: the tokens at c0 + 1 .. c0 + n), then up to `steps` chained drafts (0: the K/V only, one trace).
        Returns (drafts, probabilities)."""
        from g6_mtp import CTX_DENSE
        C, dev, n = self.C, self.V.dev, len(nxt)
        if c0 + n > CTX_DENSE:
            return [], []
        steps = min(steps, CTX_DENSE - (c0 + n - 1))
        t0 = time.perf_counter() if t0 is None else t0
        C.set_pass(c0, R, nxt, max(1, steps))
        self.t_set = time.perf_counter() - t0
        if steps < 1:
            ttnn.execute_trace(dev, self.tids[("kv", 0)], cq_id=0, blocking=False)
            return [], []
        (h, w), vocab, out, probs, q = self.d_host[head], C.hd.vocab, [], [], 1.0
        for k in range(steps):
            ttnn.execute_trace(dev, self.tids[(head, k)], cq_id=0, blocking=False)
            ttnn.copy_device_to_host_tensor(C.hd.H.ids_t, h, blocking=True)
            j, v = int(w[0]), w[TOPK:2 * TOPK].view(np.float32).astype(np.float64)
            prob = float(1.0 / np.exp(v - v[0]).sum())   # shortcut: the top 64's softmax (>= the true p)
            if vocab is not None and j >= len(vocab):
                j = 0     # a padding row: g6_mtp.head_rows pads a subset head with copies of its first row, and the
                #           card's top 64 does not break the tie by the lower index (S3 2026-11-07: index 46,211 of
                #           the fr head's 46,211 rows at cell 54,536, the top values equal: an IndexError until then)
            if not np.isfinite(prob) or j >= (len(vocab) if vocab is not None else VOCAB):
                # a draft the head cannot give: no more drafts this pass, the verify goes on; the first one logged
                if not getattr(self, "bad_draft_logged", False):
                    self.bad_draft_logged = True
                    print(f"[ttcard] draft head: index {j} of {len(vocab) if vocab is not None else VOCAB} at cell "
                          f"{c0 + n - 1 + k} (step {k}), top values {v[:4].tolist()}: drafting stopped for the pass",
                          file=sys.stderr, flush=True)
                break
            q *= prob
            if prob < min_p or q < min_q:
                break
            out.append(j if vocab is None else int(vocab[j]))
            probs.append(prob)
        return out, probs

    def generate(self, prompt, n_new: int, head: str = "fr", rows: int = M_MAX, min_p: float = 0.0,
                 min_q: float = 0.0, eos=()):
        """Greedy decoding of n_new tokens after prompt (rows = 1: one-row passes on the verify build, no MTP); stops
        after a token in eos. Returns the tokens; self.stats gets the decode passes', self.tops each token's row's top
        64 (ids, logits)."""
        V, C = self.V, self.C
        assert 1 <= rows <= M_MAX
        V.reset()
        S = rows - 1
        if S:
            C.zero_kv()
            C.use_head(head)
        seq, p, n_keep, n_p, drafts = list(prompt), 0, 1, len(prompt), []
        while p < n_p:                         # the prompt: teacher-forced 4-row passes
            M = min(M_MAX, n_p - p)
            ids, vals = V.verify(*self.inputs(seq[p:p + M], seq[max(0, p - 16):p], p), p, n_keep)
            g, last = ids[:, 0], p + M == n_p
            tops = [(ids[M - 1].copy(), vals[M - 1].copy())]
            nxt = seq[p + 1:p + M] + [int(g[M - 1]) if last else seq[p + M]]
            if S:
                drafts = self.draft(head, p, M, M, nxt, S if last else 0, min_p, min_q)[0]
            p, n_keep = p + M, M
        out, self.stats, self.tops = [nxt[-1]], [], tops
        while len(out) < n_new and out[-1] not in eos:
            t0 = time.perf_counter()
            toks = [out[-1]] + drafts
            ids, vals = V.verify(*self.inputs(toks, (seq[-16:] + out)[:-1], p), p, n_keep)
            g = ids[:, 0]
            n = accept(toks[1:], g)
            self.tops += [(ids[m].copy(), vals[m].copy()) for m in range(n)]
            t1 = time.perf_counter()
            out += [int(t) for t in g[:n]]
            drafts = self.draft(head, p, len(toks), n, out[-n:], S, min_p, min_q)[0] if S else []
            self.stats.append((p, len(toks), n, t1 - t0, time.perf_counter() - t1, self.t_set if S else 0.0))
            p, n_keep = p + n, n
        cut = next((i + 1 for i, t in enumerate(out) if t in eos), len(out))
        return out[:min(cut, n_new)]
