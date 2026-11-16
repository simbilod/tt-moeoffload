# tt-moeoffload

> [!WARNING]  
> This project is currently a limited work-in-progress; it is publicized in case the custom Tenstorrent kernels are useful to others!

A MoE expert-offloading inference engine for a Tenstorrent Blackhole p150 card: the card runs the model's layers, and
the routed experts run on the host's CPU, on the card, or split between them. It runs as the engine of a
Strata-lineage API server (it speaks their `strata --serve` line protocol), so the server's APIs,
chat template, tool calls and streaming work unchanged: Strata's server for Flash-Next, project-maya's for GLM.

## Models and quantizations

> [!NOTE]  
> More models, quantizations, and features (e.g. SSD offloading) would broaden usability -- contributions welcome!

| Model | Tested GGUF (pinned in pins.json) | Other GGUFs the engine takes | Context |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next | [unsloth `UD-IQ4_XS`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | the same architecture with routed experts in gate/up IQ3_S or IQ4_XS and down IQ4_NL or Q8_0 (the data check refuses others) | 131,072 (262,144 builds) |
| GLM-5.3-Flash | [`Maya-L`](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) | the same architecture: IQ4_XS-down layers and Q5_K-down layers with IQ3_S gate/up run on the card, any other expert type on the CPU lane | 32,768 |

Only the two tested GGUFs have been run. The card's layers and their forms are read from the GGUF's tensor types,
so another quantization of either model needs no code change when its types are in the lists above.

## What is Tenstorrent-specific

- **Whole-grid fused programs.** A decode layer runs as one program on all 110 Tensix cores, with grid barriers
  between its phases (`tt/kernels/fused/grid.h`). Flash-Next's decode token is 26 such programs, captured once in a
  trace and replayed.
- **Experts in the GGUF's own codes.** IQ3_S, IQ4_XS and IQ4_NL weights stay 4-bit codes with their scales, repacked
  exactly into tiles, and the card's kernels decode them in place (`tt/kernels/iq_decode*`, `gexpq_*`,
  `tt/flashnext/g7_experts.py`). Q8_0 goes to the card as bfp8_b, also exact; GLM's Q5_K downs are re-quantized to
  int8. A Flash-Next expert layer takes 1.46 GB on the card against 1.15 GB in the GGUF.
- **A CPU expert lane beside the card.** During decode the routed experts run on the host's AVX-512 cores while the
  card runs everything else. Each layer hands off over H2D/D2H sockets from persistent kernels, not the command queue
  (`tt/flashnext/u1_sockets.py`, `tt/cpulane/`).
- **Prefill that streams experts over PCIe.** Expert layers stream to the card through two slots while it computes.
  Short prompts keep the experts on the CPU lane; middle ones split them between the card and the lane (Flash-Next: a
  quarter or a half on the card; GLM: 3 to 7 ninths). `calibrate.py` measures where Flash-Next's bounds sit on a
  given host.
- **Speculative decoding on the card.** The MTP draft layer runs as fused programs, up to 4 draft steps a pass, and
  4-row verify passes check the drafts. Strata's suffix drafter and draft policy join in through a C API shim
  (`tt/spec/`).
- **The models' own layers as card kernels:** Flash-Next's gated DeltaNet, QSA sparse attention, hyper-connection mix
  and PLE block; GLM's KDA, MLA with its DSA indexer, and mHC. The top-512 pool selection for sparse attention runs on
  the host, between two card programs.

## What it needs

| | Flash-Next (UD-IQ4_XS) | GLM-5.3-Flash (Maya-L) |
| --- | --- | --- |
| Card | one p150 (a, b or c); tt-kmd 2.11.0+, firmware 19.16+ | the same |
| Host RAM | 173 GB (157 GB measured at 131K context, +10%) | 367 GB (333.9 GB at 32K, +10%) |
| Disk under the data root | 94 GB model + 70 GB card cache + 7 GB MTP + 1 GB census and references | 156 GB model + 165 GB card cache + 1 GB references |
| Context | up to 131,072 tokens (262,144 builds) | 32,768 tokens |

The host also needs:

- An x86-64 CPU with AVX-512 F, BW and VNNI (the CPU lane's rows). VBMI speeds up GLM's IQ3_S path.
- One 1 GiB hugepage per card, mounted at `/dev/hugepages-1G` (tt-installer sets it up).
- The IOMMU on, with the card's IOMMU group in DMA-FQ mode for the host-to-card sockets.
  `cat /sys/bus/pci/devices/<card>/iommu_group/type` should print `DMA-FQ` (or `DMA`). `tools/iommu` sets DMA-FQ
  at boot (root).
- Docker. The engine runs in the pinned tt-metal image (pins.json).
- git, cmake, g++ and python3 (3.10 or newer) with venv. Setup below makes `.venv` with the packages prepare and
  the API servers need.
- Optional: an NVIDIA GPU with nvcc. `prepare` runs llama.cpp to make the routing census and the reference logits.
  With a GPU the dense layers run there (the census in about 15 minutes). Without one, everything runs on
  the CPU, slower, and the results are valid but not bit-equal to a GPU build's.

The preflight (below) says what this host lacks, each item with its fix.

## Limits

- Tested on one host only. The cost model's defaults come from it; `calibrate.py` measures another host's.
- No SSD expert tier: the host's RAM holds the routed experts and the prefill's pinned copies, and the launcher refuses
  to start when it cannot.
- One card. A second p150 and `--device N` are not supported yet.
- The CPU lane needs AVX-512 (F, BW, VNNI); there is no AVX2 build.
- `calibrate.py` covers Flash-Next; GLM's costs are the defaults.

## Setup

    git clone <this repo> tt-moeoffload && cd tt-moeoffload
    bash tt/env/setup.sh                 # pinned checkouts into third_party/, the lane, the spec shim, tools/llama
    python3 -m venv .venv && .venv/bin/pip install -r third_party/strata/requirements.txt
    .venv/bin/python -m tt_moeoffload.preflight

The data root is `$TT_MOEOFFLOAD_DATA` (default `~/.cache/tt-moeoffload`). It holds the models, the derived caches
and the references. Set it before every command below when you use another one.

## Prepare a model

    .venv/bin/python tt_moeoffload/prepare.py flashnext          # or: glm
    .venv/bin/python tt_moeoffload/prepare.py flashnext --checks # also the reference logits the checks compare with

Each step says what it makes and why, and a rerun skips what is there. The steps:

1. The model's GGUF files from Hugging Face, at the revision pins.json names, each one checked against its sha256.
   Files you already have: `--from DIR`, where DIR is a download of the model's Hugging Face repo (or the folder
   that holds its shards: Flash-Next's `UD-IQ4_XS/`, GLM's `Maya-L/`). They are checked, then linked.
2. The API server's tokenizer, read from the GGUF.
3. Flash-Next only: the MTP draft layer's runtime files. They come from range reads of the 31 `mtp.*` tensors in
   Qwen/Qwen3.8-Flash-Next's BF16 checkpoint (~5 GB), not the 360 GB checkpoint.
4. Flash-Next only: the routing census over a pinned corpus (`tools/llama`).
5. The card expert cache: the routed experts in the card's tile form, built in the container. Nothing runs on the
   card, but the launcher checks the card first, so the host must pass the preflight.
6. With `--checks`: llama.cpp's reference logits.

## Calibrate (Flash-Next)

    .venv/bin/python tt_moeoffload/calibrate.py --cpus all      # the --cpus your server config gives the engine

About 10 minutes on the card. It times each way the prefill can take the experts (the CPU lane, part of them on the
card, the card alone) at prompt sizes up to 8,192, the prompt loop and the reset, then writes
`$TT_MOEOFFLOAD_DATA/tune.json`. The engine reads that file when it starts, and its bounds decide which path a prompt
takes on this host. Without the file the engine uses built-in defaults, fit on the development host while the engine
was being written; calibrate found several of them stale (the card's prefill, the prompt loop, the reset), so run
it. Rerun it when the CPUs, the RAM or the PCIe link change. `--dry-run` prints the values without writing them. GLM's costs keep their
defaults.

## Serve

The server starts `tt_moeoffload/launch.py` as its engine. The launcher runs the engine in the container and checks
the host first. Flash-Next behind Strata's server, with this config as `flashnext.json` (paths absolute):

```json
{
  "exe": "/path/to/tt-moeoffload/tt_moeoffload/launch.py",
  "args": ["--cpus", "all", "--", "--max-context", "131072", "--spec-rows", "4", "--spec-head", "fr"],
  "cwd": "/path/to/tt-moeoffload",
  "tokenizer": "/path/to/data-root/tokenizers/flashnext/tokenizer",
  "env": {"TT_MOEOFFLOAD_DATA": "/path/to/data-root"},
  "model_name": "qwen3.8-flash-next",
  "log": "/path/to/data-root/engine-flashnext.log"
}
```

    .venv/bin/python third_party/strata/serve/server.py --engine strata --config flashnext.json --port 8095

GLM behind project-maya's server, with `glm.json`:

```json
{
  "exe": "/path/to/tt-moeoffload/tt_moeoffload/launch.py",
  "args": ["--cpus", "all", "--", "--model", "glm", "--max-context", "32768", "--spec-rows", "2"],
  "cwd": "/path/to/tt-moeoffload",
  "tokenizer": "/path/to/data-root/tokenizers/glm/tokenizer",
  "env": {"TT_MOEOFFLOAD_DATA": "/path/to/data-root"},
  "model_name": "glm-5.3-flash",
  "log": "/path/to/data-root/engine-glm.log"
}
```

    .venv/bin/python third_party/maya/serve/server.py --engine maya --config glm.json --port 8096

`"log"` is where the engine's own lines go (its start, each request, any error); without it the servers drop them.
The first start compiles the card's kernels and loads the card expert cache: allow several minutes before the server
answers.

## Check

After `prepare.py <model> --checks`, on the card (nothing else using it):

    .venv/bin/pip install pytest && .venv/bin/python -m pytest -q tests   # no card; TT_MOEOFFLOAD_CARD=1 adds two
    python3 tt_moeoffload/launch.py --engine tt/flashnext/g4_gate.py --name tt-moeoffload-gate   # the gate, ~4 min
    python3 tt_moeoffload/launch.py --engine tt/serve/check_card.py --cpus all --memory-max $((220 << 30)) \
        --env TT_CTX=131072 --env TTC_STEPS=serve -- spec                  # the served check
    python3 tt_moeoffload/launch.py --engine tt/glm/g7_check.py --cpus all --memory-max $((320 << 30))  # GLM's check

Each check writes its result under `$TT_MOEOFFLOAD_DATA/results/`; the references it compares with are the repo's
`tt/*/results/` files. The gate's KL and top-1 limits are the same on every host. Its speed is the host's: the first run that passes writes
`$TT_MOEOFFLOAD_DATA/gate_baseline.json` (PARTIAL: nothing to compare yet), and later runs must not be slower than it
beyond its noise band. `--write-baseline` replaces it, e.g. after a hardware change.

## Acknowledgements

- [Strata](https://github.com/Niko1221/Strata): the API server Flash-Next is served behind, its tokenizer and MTP
  tools, the suffix drafter and draft policy (through `tt/spec/`), and the AVX-512 i-quant rows the CPU lane links.
- [project-maya](https://github.com/mw00/project-maya): the API server GLM-5.3-Flash is served behind.
- [Xolotl](https://github.com/Indras-Mirror/Xolotl): the Strata fork the engine was built from until 2026-11-16;
  until then GLM's lane used its clamped SwiGLU rows.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): ggml under the CPU lane, gguf-py, and the reference logits and
  routing census the checks compare with (`tools/llama`).
- Tenstorrent:
  - [tt-metal](https://github.com/tenstorrent/tt-metal): the runtime, ttnn and the pinned container image. Two of
    its model demos are used directly: `models/demos/deepseek_v3_d_p` (the KDA and mHC modules in GLM's prefill) and
    `models/demos/blackhole/qwen36` (the fused chunked gated delta rule in Flash-Next's prefill), and a prefill
    study (`tt/flashnext/g2_conv.py`) imports `models/experimental/gated_attention_gated_deltanet`. Two small pieces
    are derived from tt-metal (NOTICE): a DRAM-sharded matmul recipe and a PCIe write helper. Others were studied,
    `models/demos/gpt_oss` (its fused decode and dflash) most of all.
  - Read while designing the kernels and tools:
    [tt-llk](https://github.com/tenstorrent/tt-llk),
    [tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation),
    [tt-exalens](https://github.com/tenstorrent/tt-exalens),
    [tt-umd](https://github.com/tenstorrent/tt-umd),
    [tt-zephyr-platforms](https://github.com/tenstorrent/tt-zephyr-platforms) and
    [tt-awesome](https://github.com/tenstorrent/tt-awesome), the community projects list.

## License

MIT (LICENSE). Code derived from other projects keeps its authors' copyright and license: NOTICE lists each file,
and LICENSES/ holds the texts. The models are not part of this repository and keep their own licenses: the GLM-5.3-Flash
GGUF declares MIT; the Qwen3.8-Flash-Next GGUF declares none, so see Qwen's model card.
