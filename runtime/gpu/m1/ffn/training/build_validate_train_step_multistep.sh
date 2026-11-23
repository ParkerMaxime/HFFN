#!/usr/bin/env zsh
set -euo pipefail
root=${0:A:h}
while [[ ! -d $root/runtime/common/pythia/hffn_pythia ]]; do root=${root:h}; [[ $root == / ]] && { echo "tree root not found" >&2; exit 1; }; done
leaf=$root/runtime/gpu/m1/ffn/training
out=$leaf/build
mkdir -p "$out"
clang++ -std=c++17 "$leaf/validate_train_step_multistep.cpp" -o "$out/validate_train_step_multistep"
