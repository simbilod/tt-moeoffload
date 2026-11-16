#!/bin/bash
# Build libstrata_spec.so (Strata's suffix drafter, draft policy and controller from third_party/strata/src/spec, behind
# tt/spec/spec_capi.cpp) and Strata's own three spec tests, into OUT (default tt/spec/build).
# Run where the Python that loads it runs (the tt-metal container: its glibc).
#   bash tt/spec/build.sh [OUT]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
X=$ROOT/third_party/strata
OUT=${1:-$ROOT/tt/spec/build}
mkdir -p "$OUT"
SRC="$X/src/spec/suffix_drafter.cpp $X/src/spec/draft_policy.cpp $X/src/spec/controller.cpp $X/src/spec/draft_source.cpp"
CXX="g++ -O2 -std=c++17 -I$X/include"
$CXX -fPIC -shared $SRC "$ROOT/tt/spec/spec_capi.cpp" -o "$OUT/libstrata_spec.so"
for t in suffix_drafter draft_policy controller; do
    $CXX $SRC "$X/src/spec/${t}_test.cpp" -o "$OUT/${t}_test"
done
echo "$OUT/libstrata_spec.so"
