#!/usr/bin/env bash
# The producer tools (routing-census, dump-logits, gen-greedy) and llama.cpp's llama-tokenize, built on the host
# against the pinned third_party/llama.cpp (static; tt/env/setup.sh checks it out). CUDA when nvcc is found (the dense
# layers on the GPU, -ngl 99 -ot exps=CPU; the references in use were made that way), else CPU only.
#   bash tools/llama/build.sh            -> tools/llama/build/{routing-census,dump-logits,gen-greedy,llama-tokenize}
# LLAMA_CUDA=0 forces a CPU-only build; LLAMA_CUDA_ARCH (default: native) picks the GPU architecture.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=$HERE/../../third_party/llama.cpp
B=$L/build
OUT=$HERE/build
[ -f "$L/CMakeLists.txt" ] || { echo "no $L: run tt/env/setup.sh first" >&2; exit 1; }
cuda=${LLAMA_CUDA:-$(command -v nvcc > /dev/null && echo 1 || echo 0)}
cmake -S "$L" -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
  -DGGML_CUDA=$([ "$cuda" = 1 ] && echo ON || echo OFF) -DCMAKE_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCH:-native}" > "$B.log"
cmake --build "$B" -j "$(nproc)" --target llama-tokenize llama-common llama vendor-hash cpp-httplib >> "$B.log"
libs=("$B/common/libllama-common.a" "$B/common/libllama-common-base.a" "$B/src/libllama.a" "$B/ggml/src/libggml.a"
      "$B/ggml/src/libggml-cpu.a" "$B/ggml/src/libggml-base.a" "$B/vendor/hash/libvendor-hash.a"
      "$B/vendor/cpp-httplib/libcpp-httplib.a")
sys=(-lssl -lcrypto -lpthread -ldl -lm)
if [ "$cuda" = 1 ]; then
  libs+=("$B/ggml/src/ggml-cuda/libggml-cuda.a")
  sys=(-L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt -lcuda "${sys[@]}")
fi
mkdir -p "$OUT"
for t in census:routing-census dump_logits:dump-logits gen_greedy:gen-greedy; do
  g++ -O2 -std=c++17 -march=native -fopenmp -I"$L/include" -I"$L/common" -I"$L/ggml/include" -I"$L/vendor" \
    "$HERE/${t%%:*}.cpp" -o "$OUT/${t##*:}" -Wl,--start-group "${libs[@]}" -Wl,--end-group "${sys[@]}"
done
cp "$B/bin/llama-tokenize" "$OUT/"
echo "built $OUT ($([ "$cuda" = 1 ] && echo CUDA || echo CPU only))"
