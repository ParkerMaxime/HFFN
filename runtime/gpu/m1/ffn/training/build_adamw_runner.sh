#!/usr/bin/env zsh
set -euo pipefail
root=${0:A:h}
while [[ ! -d $root/runtime/common/pythia/hffn_pythia ]]; do root=${root:h}; [[ $root == / ]] && { echo "tree root not found" >&2; exit 1; }; done
leaf=$root/runtime/gpu/m1/ffn/training
out=$leaf/build
mkdir -p "$out"
xcrun clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  "$leaf/metal_hffn_adamw_runner.mm" \
  -o "$out/metal_hffn_adamw_runner"
