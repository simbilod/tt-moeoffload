# Source inside the tt-metal container (the launcher does; home and the repo are at their host paths): tt-metal's
# Python model code (third_party/tt-metal, a sparse checkout of the pinned tag), its pip dependencies
# (third_party/tt-pydeps), llama.cpp's gguf-py and this repo's tt/ package.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH=$REPO/third_party/tt-metal:$REPO/third_party/tt-pydeps:$REPO/third_party/llama.cpp/gguf-py:$REPO/tt${PYTHONPATH:+:$PYTHONPATH}
export TT_MOEOFFLOAD_DATA=${TT_MOEOFFLOAD_DATA:-$HOME/.cache/tt-moeoffload}   # the data root: caches, references, models
# TT_MOEOFFLOAD_TTNN=tracy: profiling runs only (tt-metal's device profiler, TT_METAL_DEVICE_PROFILER=1), the same tag
# built with Tracy into $TT_MOEOFFLOAD_DATA/ttnn-tracy (its tt-metal tree in $TT_MOEOFFLOAD_DATA/tt-metal-tracy) ahead of the
# image's release ttnn; serving and benchmarks keep the release build (2026-11-07)
if [ "${TT_MOEOFFLOAD_TTNN:-}" = tracy ]; then
  export PYTHONPATH=$TT_MOEOFFLOAD_DATA/ttnn-tracy:$PYTHONPATH
  # its _ttnncpp.so holds tt-metal and Tracy statically: preloaded, or its static TLS does not fit at import
  export LD_PRELOAD=$TT_MOEOFFLOAD_DATA/ttnn-tracy/ttnn/build/lib/_ttnncpp.so${LD_PRELOAD:+:$LD_PRELOAD}
  export TT_METAL_RUNTIME_ROOT=$TT_MOEOFFLOAD_DATA/tt-metal-tracy   # its kernels and SoC descriptors, the same tag
fi
export TT_METAL_HOME=$REPO/third_party/tt-metal
export HF_MODEL=${HF_MODEL:-$REPO/third_party/tt-metal/models/demos/gpt_oss/configs/gpt-oss-120b}
export ARCH_NAME=blackhole
cd "$TT_METAL_HOME" 2>/dev/null || true   # a repo without third_party: the stand-in engines' tests
