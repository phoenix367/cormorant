#!/usr/bin/env bash
# make_fixture.sh <build-dir> <test-index> <out-dir>
# Copy one dumped ConvKernel test case (from <build-dir>/conv_test_data) into
# a standalone fixture directory the test stand can run on its own.
set -euo pipefail
build=${1:?build dir}; idx=${2:?test index}; out=${3:?out dir}
src="$build/conv_test_data"
[ -f "$src/manifest.txt" ] || { echo "no $src/manifest.txt — run: make -C $build gen_conv_test_data" >&2; exit 1; }
printf -v ii "%02d" "$idx"
mkdir -p "$out"
head -2 "$src/manifest.txt" > "$out/manifest.txt"
grep -E "^$idx " "$src/manifest.txt" >> "$out/manifest.txt" || { echo "index $idx not in manifest" >&2; exit 1; }
cp "$src"/test_${ii}_{x,w,b,y}.hex "$out/"
echo "fixture $out: $(tail -1 "$out/manifest.txt" | cut -c1-90)"
