#!/usr/bin/env bash
# The engine's dependencies at the commits in pins.json, into third_party/ (gitignored), then its two native builds.
# Host side; needs git, docker and the network. Safe to re-run: present checkouts are moved to their pins.
#   third_party/llama.cpp  ggml (the lane's ggml-cpu), gguf-py, and the producer tools' llama.cpp (tools/llama)
#   third_party/strata     src/kernels/cpu, src/spec, include, data, tools, serve (the lane's rows, the drafter, the draft
#                          vocab, the MTP and tokenizer tools, the API server for Flash-Next, the census corpus's part A)
#   third_party/maya       GLM's API server; files the census corpus reads
#   third_party/tt-metal   sparse checkout (tt/env/tt-metal-sparse.txt): the model code the engine imports
#   third_party/tt-pydeps  tt/env/requirements.lock (exact, hash-checked, --no-deps), installed by the image's uv for its
#                          Python 3.10 (its cache and the container's HOME in third_party/home)
#   tt/cpulane/build (libcpulane.so, libcpulane_glm.so), tt/spec/build   built in the container, where they run (its glibc)
#   tools/llama/build      routing-census, dump-logits, gen-greedy, llama-tokenize: on the host (CUDA if nvcc is found)
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
pin() { python3 -c "import json,sys; print(json.load(open('pins.json'))[sys.argv[1]][sys.argv[2]])" "$1" "$2"; }

checkout() {   # dir repo commit: one commit, no history
  local dir=$1 repo=$2 commit=$3
  [ -d "$dir/.git" ] || { git init -q "$dir" && git -C "$dir" remote add origin "$repo"; }
  if [ "$(git -C "$dir" rev-parse -q --verify HEAD 2>/dev/null)" != "$commit" ]; then
    git -C "$dir" fetch -q --depth 1 origin "$commit"
    git -C "$dir" checkout -q --detach "$commit"
  fi
}

mkdir -p third_party
checkout third_party/llama.cpp "$(pin llama_cpp repo)" "$(pin llama_cpp commit)"
checkout third_party/strata "$(pin strata repo)" "$(pin strata commit)"
checkout third_party/maya "$(pin maya repo)" "$(pin maya commit)"
if [ ! -d third_party/tt-metal/.git ]; then
  git clone -q --filter=blob:none --no-checkout --depth 1 --branch "$(pin tt_metal tag)" "$(pin tt_metal repo)" \
    third_party/tt-metal
fi
git -C third_party/tt-metal sparse-checkout set --no-cone --stdin < tt/env/tt-metal-sparse.txt
git -C third_party/tt-metal checkout -q "$(pin tt_metal commit)"

in_container() {   # the pinned image, as this user, the repo at its own path and nothing else of the host's
  mkdir -p third_party/home
  docker run --rm --log-driver none --user "$(id -u):$(id -g)" --env "HOME=$REPO/third_party/home" \
    --env "UV_CACHE_DIR=$REPO/third_party/home/.uv-cache" \
    --volume "$REPO:$REPO" --volume /etc/passwd:/etc/passwd:ro --volume /etc/group:/etc/group:ro \
    --workdir "$REPO" --entrypoint /bin/bash "$(pin tt_metalium image)" -c "$1"
}
rm -rf third_party/tt-pydeps && mkdir -p third_party/tt-pydeps   # exactly the lock, nothing left from an older one
in_container "uv pip install -q --python /opt/venv/bin/python --target third_party/tt-pydeps --no-deps --require-hashes --index-strategy unsafe-best-match -r tt/env/requirements.lock"
in_container "bash tt/cpulane/build.sh && CL_SHAPE=glm bash tt/cpulane/build.sh && bash tt/spec/build.sh"   # the lane: Flash-Next, GLM
bash tools/llama/build.sh
echo "setup done: $REPO/third_party"
