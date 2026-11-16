#!/bin/bash
# Build the CPU lane (experts.cpp), the token's host loop (host_loop.cpp), its inputs (gather.cpp) and GLM's mHC host
# parametrization (hc.cpp) against the pinned
# llama.cpp's ggml (CPU backend only, static, -march=native), inside the card's container, which is where it runs:
#   tt/env/setup.sh runs it (and CL_SHAPE=glm for GLM's library)
# CL_TRACE=1: build/libcpulane_trace.so instead, with the per-call, per-thread stamps (cl_trace; CL_LIB selects it).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GGML=$HERE/../../third_party/llama.cpp/ggml
B=$HERE/build
cmake -S "$GGML/.." -B "$B/ggml" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build "$B/ggml" -j "$(nproc)" --target ggml-cpu ggml-base >/dev/null
# Strata's AVX-512 i-quant rows (third_party/strata, pins.json)
STRATA=$HERE/../../third_party/strata
OUT=$B/libcpulane.so FLAGS=()
if [ "${CL_TRACE:-0}" = 1 ]; then OUT=$B/libcpulane_trace.so FLAGS=(-DCL_TRACE); fi
# CL_SHAPE=glm: GLM-5.3-Flash's expert shapes (D 4096, FF 2048, 288 experts, top-8, 46 layers) and its SwiGLU
# clamp (swiglu_clamp_exp 10) in libcpulane_glm.so
if [ "${CL_SHAPE:-}" = glm ]; then
  OUT=${OUT%.so}_glm.so FLAGS+=(-DCL_D=4096 -DCL_FF=2048 -DCL_NX=288 -DCL_MAXK=8 -DCL_LAYERS=46 -DCL_XQB=8192 -DCL_HQB=4096 -DCL_SWIGLU_LIM=10.0f)
fi
g++ -O3 -march=native -std=c++17 "${FLAGS[@]}" -shared -fPIC -o "$OUT" "$HERE/experts.cpp" "$HERE/host_loop.cpp" \
    "$HERE/gather.cpp" "$HERE/hc.cpp" "$HERE/glm_loop.cpp" "$STRATA/src/kernels/cpu/iq_avx512.cpp" \
    -I"$GGML/include" -I"$GGML/src" -I"$STRATA/include" \
    -Wl,--whole-archive "$(find "$B/ggml" -name libggml-cpu.a)" -Wl,--no-whole-archive \
    "$(find "$B/ggml" -name libggml-base.a)" -lpthread
# -march=native: the build CPU's flags beside the library; the engine's data check refuses it on a CPU lacking one
grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2 | tr ' ' '\n' | grep . | sort -u > "$OUT.cpu"
echo "built $OUT"
