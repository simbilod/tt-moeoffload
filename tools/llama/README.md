# Producer tools (on the pinned llama.cpp)

They make the data the engine and its checks read, from the model files, with llama.cpp itself as the reference.
`bash tools/llama/build.sh` (run by `tt/env/setup.sh`) builds them on the host against `third_party/llama.cpp`
(pins.json), static, with CUDA when `nvcc` is found: the dense layers run on the GPU (`-ngl 99 -ot exps=CPU`), the
routed experts on the CPU. The references in use were made that way, with a GPU; a CPU-only build makes valid
references whose bits differ from those.

| Tool | Makes | Read by |
| --- | --- | --- |
| `corpus.py` | the census corpus `census/<model>/prompts.jsonl` (51 prompts, pinned sha256) | the census, the served check's prompts |
| `routing-census` (`census.cpp`) | `census.json` (per-layer expert route counts), `census.trace` (every routing decision in order), `ids/<id>.i32` (each prompt's tokens) | the card expert cache's slot ranking, the lane's warm, the served check |
| `dump-logits` (`dump_logits.cpp`) | every position's logits of a token sequence (`g4_ref*`, `glm_ref*`), optionally named graph tensors | the gate, the served check, GLM's check |
| `gen-greedy` (`gen_greedy.cpp`) | llama.cpp's greedy continuations of the corpus with the last layer's residual and top-8 (`w4_draft/census`) | `tt/flashnext/w4_draft.py`, `g6_mtp.py` (MTP draft studies) |
| `llama-tokenize` | llama.cpp's own tokenizer (`--ids --no-bos`) | the references' `ids.i32` |

`tt_moeoffload/prepare.py` runs them with the commands below; they are here to be read and rerun by hand.

## The corpus

Part A is Strata's `bench/profile_corpus.json` (45 short prompts). Part B is six coding-agent sessions over the pinned
checkouts in `third_party/` (project-maya, llama.cpp, Strata): a task, tool calls, and tool results that are the files
at their pinned commits. Rendering is the API server's own (`serve/frontend.py`'s `ChatTemplate`) with the chat
template read from the GGUF, so the bytes are the same on any host with the same checkouts.

    python3 tools/llama/corpus.py <first .gguf shard> $TT_MOEOFFLOAD_DATA/census/qwen3.8-flash-next/prompts.jsonl

## The census

    GGML_OP_OFFLOAD_MIN_BATCH=2048 tools/llama/build/routing-census -m <first shard> -ngl 99 -ot exps=CPU \
        -c 131072 -t <cores> -tb <cores> -ub 4096 -b 8192 --load-mode mmap --lazy-mode on --fit off \
        --census-in prompts.jsonl --census-out census.json --census-trace census.trace --census-ids ids \
        --census-decode 256

Each prompt is read from an empty cache, then 256 tokens are decoded greedily; the prompt and decode phases are
counted apart. About 15 minutes with a GPU.

## Reference logits

    tools/llama/build/llama-tokenize -m <first shard> -f prompt.txt --ids --no-bos     # -> ids.i32 (int32)
    tools/llama/build/dump-logits -m <first shard> -ngl 99 -ot exps=CPU -c 512 -b 512 -ub 512 -t 64 -tb 64 \
        --dump-ids ids.i32 --dump-out logits.f32 [--dump-from N] [--dump-tensors a,b --dump-dir DIR]

`--dump-from N` keeps the positions from N on (a long context's tail). Tensor dumps append: empty `DIR` first.
