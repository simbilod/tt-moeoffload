# tt-moeoffload: plan

The Tenstorrent Blackhole p150 engine for Qwen3.8-Flash-Next and GLM-5.3-Flash. It runs as the `exe` of a
Strata-lineage API server (Strata's `serve/server.py` for Flash-Next, project-maya's for GLM) and speaks their engine
line protocol (`strata --serve`: READY / GEN / T / DONE), so whatever drives those servers uses it as it uses Strata's
own engine.

## Decisions

- **Its own project, not a fork of Strata.** The engine needs no change to Strata's server, and GLM's server is
  project-maya's. Strata's API server, tokenizer tools, MTP tools, speculation sources (`src/spec`, through a C API
  shim) and AVX-512 i-quant rows are used from pinned checkouts, not copied (NOTICE lists the few derived files).
- **Any compatible host.** Nothing is tuned for one machine: the CPU lane sizes itself from the CPU topology, the
  prefill's cost model comes from `calibrate.py` (`$TT_MOEOFFLOAD_DATA/tune.json`), the card's layers from the GGUF's
  tensor types, and every path from one data root (`TT_MOEOFFLOAD_DATA`).
- **Users regenerate every derived file.** Nothing derived from model weights is shipped: `prepare.py` makes the data
  root from the pinned models.
- **Bit for bit where it can be.** A change that keeps the numbers is checked bit for bit (the gate's per-position KL
  and top-1, the served check's tokens); one that moves them on purpose is judged by the near-tie rule (`check_card.py`'s
  TIE: a first difference whose logit gap is under 0.75 on either side).
- **No SSD expert tier yet.** The host's RAM holds the routed experts and the prefill's pinned copies; the launcher
  refuses to start when it cannot.
- **No thermal guard** in the launcher.

## Done

- The launcher and preflight: the engine in the pinned tt-metal container, the host's checks before each start, no
  `--privileged`, only the repo, the data root and the engine's argument folders mounted, secrets never forwarded.
- The engine: Flash-Next's decode as 26 fused whole-grid programs, the CPU expert lane over H2D/D2H sockets, the
  chunked prefill with its experts on the lane, the card or split, speculative decoding with the MTP draft layer,
  conversation snapshots, prompt checkpoints; GLM-5.3-Flash on the same machinery.
- `prepare.py`: the data root from nothing (model download with sha256 checks or `--from`, tokenizer, MTP runtime
  files, routing census, card expert cache, reference logits).
- Host tuning: the lane's CPUs from `/sys` topology; the cost model in `tune.json`, measured by `calibrate.py`.
- Any GGUF of the two architectures whose expert types the card decodes; the data check refuses the others.
- Another host's safety nets: the lane library records its build CPU's flags and is refused elsewhere; the gate's
  speed baseline is each host's own.
- Licenses: every derived file credited (NOTICE, LICENSES/).

## Open

- **Names:** modules named for what they do (`g7_prefill.py` -> `prefill.py`, ...) and the development's stage
  labels out of the remaining comments, as one change checked bit for bit.
- **CI:** the no-card tests on every push (the container tests skip without the image).
- **Pins:** llama.cpp is pinned to a commit that only a pull-request ref holds; move it to a release tag (or a fork's
  branch) and the tt-metal image to a release image, each checked by the gate.
- **Untested paths:** prepare's real Hugging Face download; `prepare.py glm --checks` on a fresh data root; the checks
  from a fresh clone on a second host.
- **Calibrate for GLM** (its costs are the defaults), and the built-in defaults refreshed from a calibration.
- **An AVX2 build of the CPU lane** for hosts without AVX-512.
- **More hardware:** `--device N` for a host with several cards, then a model split over two networked p150s.
- **An SSD expert tier**, so the host needs less RAM.
- **py-spy stall dumps** without `--privileged` (`--cap-add SYS_PTRACE` only).
