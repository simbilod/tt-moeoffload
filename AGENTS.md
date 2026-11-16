# AGENTS.md: working in tt-moeoffload

A MoE expert-offloading inference engine for one Tenstorrent Blackhole p150 card (Qwen3.8-Flash-Next, GLM-5.3-Flash).
The card runs the model's layers as fused whole-grid programs; the routed experts run on the host's CPU lane, on the
card, or split between them. The engine is the `exe` of a Strata-lineage API server and speaks its line protocol on
stdin/stdout. Read README.md (what users run) and PLAN.md (decisions, open work) before changing anything.

## Layout

| Path | What it is |
| --- | --- |
| `tt_moeoffload/` | Host-side tools: `launch.py` (runs any engine script in the pinned tt-metal container), `preflight.py` (host checks), `prepare.py` (builds the data root), `calibrate.py` (writes `tune.json`) |
| `tt/serve/engine.py` | The engine: line protocol, sampler, prompt reading (prefill or decode loop), speculation, checkpoints; `--fake` runs without a card |
| `tt/serve/check_card.py` | The served check (`spec serve`) and the calibration measurements (`spec fit`) |
| `tt/flashnext/` | Flash-Next on the card: `g4_model.py` decode token, `g7_prefill.py` chunked prefill, `g7_experts.py` card expert cache, `g6_mtp.py` MTP draft layer, `g4_gate.py` the gate |
| `tt/glm/` | GLM-5.3-Flash on the same machinery: `g7_engine.py` (the engine's GLM model), `g6_prefill.py`, `g6_experts.py`, `g7_check.py` |
| `tt/kernels/`, `tt/kernels/fused/` | Device kernels (reader / compute / writer per program); `fused/grid.h` the grid barrier, `fused/link.h` the H2D/D2H socket hand-off |
| `tt/cpulane/` | The CPU expert lane (C++, AVX-512) and its Python binding (`Lane`, `lane_cpus`) |
| `tt/spec/` | A C API shim over Strata's speculation sources (suffix drafter, draft policy) |
| `tt/formats/` | GGUF tensor repacking for the card, `expert_types.py` (which layers the card takes, from tensor types) |
| `tt/tune.py` | The cost model's constants, overridable by `$TT_MOEOFFLOAD_DATA/tune.json` |
| `tt/env/` | `setup.sh`, `ttenv.sh` (the container's PYTHONPATH and cwd), the hash-pinned `requirements.lock`, the tt-metal sparse-checkout list |
| `tools/llama/` | llama.cpp-based producers of the routing census and reference logits |
| `tools/iommu/` | Boot-time fix putting the card's IOMMU group in DMA-FQ (root) |
| `tests/` | No-card tests; `TT_MOEOFFLOAD_CARD=1` adds card tests |
| `pins.json` | Every external dependency and model, pinned by commit, digest, revision and sha256 |
| `*/results/` | References the checks compare with (not run outputs) |

`third_party/` (fetched by setup) and `tt/*/build/` are untracked. The modules' `g<N>_` prefixes are historical stage
numbers; renaming them is open work (PLAN.md).

## Commands

    bash tt/env/setup.sh                                   # pinned deps into third_party/, lane + spec + tools builds
    python3 -m venv .venv && .venv/bin/pip install -r third_party/strata/requirements.txt pytest
    .venv/bin/python -m pytest -q tests                    # no card, ~6 s; run after every change
    .venv/bin/python -m tt_moeoffload.preflight            # what the host lacks
    printf 'GEN 4 1,2,3\nQUIT\n' | python3 tt/serve/engine.py --serve --fake   # the protocol, no card, no container

On the card, every script goes through the launcher (paths absolute: the container's cwd is the tt-metal checkout):

    python3 tt_moeoffload/launch.py --engine tt/flashnext/g4_gate.py --name tt-moeoffload-gate          # ~4 min
    python3 tt_moeoffload/launch.py --engine tt/serve/check_card.py --cpus all --memory-max $((220 << 30)) \
        --env TT_CTX=131072 --env TTC_STEPS=serve -- spec                                               # ~6 min warm
    python3 tt_moeoffload/launch.py --engine tt/glm/g7_check.py --cpus all --memory-max $((320 << 30))  # ~20 min
    python3 tt_moeoffload/calibrate.py --cpus all --dry-run                                             # ~10 min

Launcher options before `--`, the script's after it; `--env K=V` sets a variable in the container (a variable's
value is never put on the docker command line). `--fake` runs without the card.

## Which check proves a change

| You changed | Run |
| --- | --- |
| Host tools, protocol, sampler, launcher, prepare | the tests (and fake mode for protocol behaviour) |
| Flash-Next decode, kernels, the lane | the gate: per-position KL and top-1 must equal the reference's bit for bit |
| Prompt reading, prefill, speculation, the cost model | the served check: every request's tokens equal the reference's |
| Anything GLM | GLM's check against `tt/glm/results/g7_check_ref_2026-11-15.json` |
| Launcher flags, mounts, container env | the tests with `TT_MOEOFFLOAD_CARD=1`, then the gate |

To compare the gate's parity with the repo's reference: `-- --baseline $PWD/tt/flashnext/results/gate_main_extract.json`
(its timing is the development host's: on another host compare parity only). Compare JSON results field by field,
not by eye.

A change that should keep the numbers must keep them **bit for bit**. A change that moves them on purpose (prompt
routing, the cost model, a numerics change) is judged by the **near-tie rule**: each request's first differing token
must be a near-tie (logit gap under `check_card.py`'s `TIE`, 0.75, on either side). Say which of the two applies.

## Rules

- **The card is exclusive.** The launcher takes no lock: before a card run make sure no other engine or check uses
  it (`docker ps`), and don't run CPU-heavy work (builds, `prepare.py`) beside a timing run (the gate, calibrate).
- **The data root** (`TT_MOEOFFLOAD_DATA`, default `~/.cache/tt-moeoffload`) holds models, caches, references and
  every run's results (`results/`). A new kind of data gets a `prepare.py` step. The engine checks its data before
  the card opens (`engine.data_problems`): keep that list in step with what the engine reads.
- **Results:** runs write under `$TT_MOEOFFLOAD_DATA/results/`. Copy one into `tt/*/results/` only to refresh a
  reference, in the same change that explains why.
- **Nothing tuned for one machine.** No core lists, thread counts, PCI addresses, IOMMU group numbers or hostnames
  in code: use `lane_cpus()`, `tune.json`/`calibrate.py`, the GGUF's tensor types, and sysfs lookups.
- **stdout belongs to the protocol.** tt-metal logs to fd 1, so `engine.py` dups it and points fd 1 at stderr; log
  with `log()`.
- **The container stays unprivileged:** no `--privileged`, capabilities dropped, only the repo, the data root and
  the engine's argument folders mounted. Never forward secrets into it.
- **Pins:** dependencies by commit or digest, pip packages by hash. A pin change gets the gate (and, for the lane's
  ggml, the served check).
- **Reference logits and the census** come from llama.cpp, which is not reproducible run to run on long prompts and
  greedy decodes: a regenerated one is close, not equal. Judge it by distance.
- **Licenses:** code derived from another project keeps its copyright line in the file and an entry in NOTICE, with
  the license text in LICENSES/. Check a borrowed snippet's license before adding it.

## Style

- The smallest change that does the job; no speculative options. Comments say why, in plain sentences, and carry
  measurements with their date when a constant comes from one.
- New non-trivial logic gets a small no-card test in `tests/`.
- Keep commit messages and files free of private paths, hostnames and personal details.
