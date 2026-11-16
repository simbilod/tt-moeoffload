"""G4: the host's share of a Flash-Next decode token on the card, as llama.cpp computes it (src/models/qwen4exp.cpp):

  embed(tok)          the token embedding row (token_embd, one row dequantized) and X's init: the 4 hyper-connection
                      streams as identical copies (hc_init), fp32 [4 x 2560]
  ple(tok, prev)      layer 1's PLE input: the n-gram hash of the token and its predecessors (llm_graph_input_qwen4exp_ple:
                      n = 2, 3; 8 heads per n; missing predecessors and anything at or before an EOS read as EOS) into the
                      rows of per_layer_token_embd, gathered and dequantized: [16 heads x 160] = 2560, head-major
  rope_table(ctx)     attention program 1's RoPE rows: row 4 pos + j = cos t(pos), sin t(pos), cos t(pos - 3), sin t(pos - 3)
                      (t_i = pos base^(-2i / 64), i < 32: NEOX on the first 64 dims; IMRoPE's sections all see the text
                      position), 32 floats each, [4 ctx x 32]
  select(pos, score)  the QSA selection of one decode token (build_qsa_sel): the top min(512, complete pools) pools by the
                      indexer's score, and the incomplete tail block; split for the card's attention program 2 into the
                      pool cores' list and the own-block core's valid rows

embed and ple run in C++ when the lane's library is built (tt/cpulane/gather.cpp, ggml's own to_float; G4_PY_GATHER=1:
gguf-py's dequantize), the same bits (g4_host_loop.py: the whole vocabulary, ~370K PLE rows); 3.4 + 6.9 us per token
against 42 + 95.

Checked (main): the embedding and the PLE gather against llama.cpp's own tensors for a prompt (llama.cpp's
tools/llama dump-logits --dump-tensors model.input_embed,ple_embd, in $TT_MOEOFFLOAD_DATA/g4_ref).

  (card-free) python3 tt_moeoffload/launch.py --fake --engine tt/flashnext/g4_host.py -- [g4_ref dir]
"""
from __future__ import annotations

import glob
import json
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import sys

import numpy as np
import torch

from gguf import GGUFReader
from gguf.quants import dequantize

MODEL = os.environ.get("TT_MOEOFFLOAD_MODEL_DIR") or os.path.join(DATA, "models/qwen3.8-flash-next-ud-iq4xs/UD-IQ4_XS")   # engine.py --model-dir sets it
REF = os.path.join(DATA, "g4_ref")
D, HC, KPOOL, NSEL = 2560, 4, 4, 512
U64 = (1 << 64) - 1


class Host:
    def __init__(self, model_dir: str = MODEL, cpp: bool | None = None):
        """cpp: embed and ple through the C++ gather (cpulane.Gather, the same bits; embed_py / ple_py stay Python's);
        None: when the lane's library is built and G4_PY_GATHER is not 1."""
        self.t, self.kv, self.path = {}, {}, {}
        for f in sorted(glob.glob(os.path.join(model_dir, "*.gguf"))):
            r = GGUFReader(f)
            for t in r.tensors:
                self.t[t.name], self.path[t.name] = t, f
            for k, v in r.fields.items():
                if k.startswith(("qwen4exp.", "tokenizer.ggml.")) and v.types and v.types[0].name != "STRING":
                    try:
                        self.kv[k] = [v.parts[i].tolist() for i in v.data]
                    except Exception:  # noqa: BLE001 - a field gguf-py cannot read as numbers
                        pass
        one = lambda k: self.kv[k][0][0]  # noqa: E731
        self.n_gram = one("qwen4exp.ple.ngram_size")
        self.per_gram = one("qwen4exp.ple.heads_per_ngram")
        self.eos = one("qwen4exp.ple.eos_token_id")
        self.mult = [m[0] for m in self.kv["qwen4exp.ple.layer_multipliers"]]
        self.offsets = [o[0] for o in self.kv["qwen4exp.ple.head_offsets"]]
        self.vocab_sizes = [v[0] for v in self.kv["qwen4exp.ple.head_vocab_sizes"]]
        self.n_heads = (self.n_gram - 1) * self.per_gram
        self.head_dim = one("qwen4exp.embedding_length_per_layer_input")
        self.rope_base = one("qwen4exp.rope.freq_base")
        self.n_rot = one("qwen4exp.rope.dimension_count")
        assert self.n_heads * self.head_dim == D
        self.gather = None
        if cpp is None:
            sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
            from cpulane import LIB
            cpp = os.path.exists(LIB) and os.environ.get("G4_PY_GATHER") != "1"
        if cpp:
            from cpulane import Gather
            tab = lambda k: (self.path[k], int(self.t[k].data_offset), int(self.t[k].tensor_type),  # noqa: E731
                             int(self.t[k].shape[1]), int(self.t[k].shape[0]))
            self.gather = Gather(tab("token_embd.weight"), tab("per_layer_token_embd.weight"), self.n_gram,
                                 self.per_gram, self.eos, self.mult, self.offsets, self.vocab_sizes)

    def _rows(self, name: str, ids) -> np.ndarray:
        """Rows `ids` of a 2-D GGUF tensor, dequantized to float32 [len(ids), ne0]."""
        t = self.t[name]
        rows = np.stack([np.asarray(t.data[i]) for i in ids])
        return dequantize(rows, t.tensor_type).astype(np.float32).reshape(len(ids), -1)

    def embed(self, tok: int) -> np.ndarray:
        """X's init for a token: the embedding row, 4 identical streams, fp32 [4 x 2560]."""
        return self.gather.embed(tok) if self.gather else self.embed_py(tok)

    def embed_py(self, tok: int) -> np.ndarray:
        e = self._rows("token_embd.weight", [tok])[0]
        return np.tile(e, HC).astype(np.float32)

    def ple_rows(self, tok: int, prev) -> list[int]:
        """The 16 rows of per_layer_token_embd for `tok`; `prev` its predecessors oldest first (any length)."""
        ctx = [tok]
        cut = False
        for s in range(1, self.n_gram):  # predecessor s positions back; none there reads as EOS, as does all before one
            t = prev[-s] if s <= len(prev) else -1
            cut = cut or t < 0 or t == self.eos
            ctx.append(self.eos if cut else t)
        idx = []
        for n in range(2, self.n_gram + 1):
            mixed = (ctx[0] * self.mult[0]) & U64
            for j in range(1, n):
                mixed ^= (ctx[j] * self.mult[j]) & U64
            for g in range(self.per_gram):
                h = (n - 2) * self.per_gram + g
                idx.append(mixed % self.vocab_sizes[h] + self.offsets[h])
        return idx

    def ple(self, tok: int, prev) -> np.ndarray:
        """Layer 1's PLE input emb for `tok`: fp32 [2560], head-major."""
        return self.gather.ple(tok, prev) if self.gather else self.ple_py(tok, prev)

    def ple_py(self, tok: int, prev) -> np.ndarray:
        return self._rows("per_layer_token_embd.weight", self.ple_rows(tok, prev)).reshape(-1)

    def rope_table(self, ctx: int) -> np.ndarray:
        """Attention program 1's RoPE rows, fp32 [4 ctx, 32]: row 4 pos + j as in the module docstring."""
        half = self.n_rot // 2
        inv = self.rope_base ** (-np.arange(half, dtype=np.float64) * 2 / self.n_rot)
        pos = np.arange(ctx, dtype=np.float64)[:, None]
        a, b = pos * inv, (pos - 3) * inv
        return np.stack([np.cos(a), np.sin(a), np.cos(b), np.sin(b)], 1).reshape(4 * ctx, half).astype(np.float32)

    @staticmethod
    def select(pos: int, score) -> tuple[np.ndarray, int, int]:
        """The QSA selection for the token at `pos` from the indexer's pool scores (pool p = cells 4p .. 4p + 3):
        returns (the pool cores' list, padded with pool 0 to 512; how many of it are live; the own-block core's valid
        rows). llama.cpp keeps the top min(512, complete pools) by score and the incomplete tail; the card's own-block
        core attends block pos // 4: its first pos % 4 + 1 cells while the block is incomplete, and when the token
        completes it (pos % 4 == 3) all 4 if that pool is selected (then not in the list) and none if not."""
        complete = (pos + 1) // KPOOL
        own = pos // KPOOL
        top = []
        if complete:
            s = torch.as_tensor(np.asarray(score, dtype=np.float32)[:complete])
            top = torch.topk(s, min(NSEL, complete)).indices.tolist()
        if pos % KPOOL != KPOOL - 1:
            own_valid = pos % KPOOL + 1
        elif own in top:
            top.remove(own)
            own_valid = KPOOL
        else:
            own_valid = 0
        sel = np.zeros(NSEL, np.int32)
        sel[:len(top)] = top
        return sel, len(top), own_valid


def check(ref: str = REF) -> dict:
    """The embedding and the PLE gather against llama.cpp's tensors for the prompt in `ref`."""
    h = Host()
    ids = np.fromfile(os.path.join(ref, "ids.i32"), dtype=np.int32).tolist()
    n = len(ids)
    rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())  # noqa: E731
    emb_ref = np.fromfile(os.path.join(ref, "tensors", "model.input_embed.f32"), np.float32).reshape(n, D)
    init_ref = np.fromfile(os.path.join(ref, "tensors", "hc_init.f32"), np.float32).reshape(n, HC * D)
    ple_ref = np.fromfile(os.path.join(ref, "tensors", "ple_embd.f32"), np.float32).reshape(n, D)
    emb = np.stack([h.embed(t) for t in ids])
    ple = np.stack([h.ple(t, ids[:i]) for i, t in enumerate(ids)])
    res = {"tokens": n, "embed_vs_input_embed": rel(emb[:, :D], emb_ref), "x_init_vs_hc_init": rel(emb, init_ref),
           "ple_vs_ple_embd": rel(ple, ple_ref),
           "types": {k: h.t[k].tensor_type.name for k in ("token_embd.weight", "per_layer_token_embd.weight")},
           "shapes": {k: [int(x) for x in h.t[k].shape] for k in ("token_embd.weight", "per_layer_token_embd.weight")}}
    # the selection rule on llama.cpp's own indexer scores (blk.3), every position
    sc = np.fromfile(os.path.join(ref, "tensors", "indexer_score-3.f32"), np.float32).reshape(n, -1)
    res["select_examples"] = {p: dict(zip(("nsel", "own_valid"), Host.select(p, sc[p])[1:])) for p in (0, 3, 4, 7, n - 1)}
    return res


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1] if len(sys.argv) > 1 else REF), indent=1))
