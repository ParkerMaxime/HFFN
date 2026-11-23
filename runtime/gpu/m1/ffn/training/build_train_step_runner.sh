#!/usr/bin/env zsh
set -euo pipefail
leaf=${0:A:h}
hffn_root=$leaf
while [[ ! -d "$hffn_root/runtime/common/pythia/hffn_pythia" && "$hffn_root" != / ]]; do hffn_root=${hffn_root:h}; done
[[ -d "$hffn_root/runtime/common/pythia/hffn_pythia" ]] || { echo 'tree root missing' >&2; exit 2; }
base=$hffn_root/runtime/gpu/m1/ffn/training
mkdir -p "$leaf/build"
${HFFN_PYTHON:-python} -B \
  "$hffn_root/runtime/gpu/m1/common/hffn_m1/activation.py" \
  --emit-header "$leaf/build/hffn_m1_activation_generated.h"
${HFFN_PYTHON:-python} -B \
  "$hffn_root/runtime/gpu/m1/common/hffn_m1/adamw.py" \
  --emit-header "$leaf/build/hffn_m1_adamw_generated.h"
${HFFN_PYTHON:-python} -B \
  "$hffn_root/runtime/gpu/m1/common/hffn_m1/fused_edges.py" \
  --emit-header "$leaf/build/hffn_m1_fused_edges_generated.h"
${HFFN_PYTHON:-python} -B \
  "$hffn_root/runtime/gpu/m1/common/hffn_m1/backward_sums.py" \
  --emit-header "$leaf/build/hffn_m1_backward_sums_generated.h"
${HFFN_PYTHON:-python} -B \
  "$hffn_root/runtime/gpu/m1/common/hffn_m1/backward_plan.py" \
  --emit-header "$leaf/build/hffn_m1_backward_plan_generated.h"
clang++ -std=c++17 -fobjc-arc -fvisibility=hidden -I "$leaf" -I "$base" -I "$leaf/build" \
  -c "$leaf/native_train_context.mm" -o "$leaf/build/native_train_context.o"
clang++ -std=c++17 -fobjc-arc -fvisibility=hidden -framework Foundation -framework Metal \
  -I "$leaf" "$leaf/metal_hffn_train_step_runner.mm" "$leaf/build/native_train_context.o" \
  -o "$leaf/build/metal_hffn_train_step_runner.native"
clang++ -dynamiclib -framework Foundation -framework Metal \
  "$leaf/build/native_train_context.o" -o "$leaf/build/libhffn_train.dylib"
cat > "$leaf/build/metal_hffn_train_step_runner" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
leaf=$(cd "$(dirname "$0")/.." && pwd)
exec ${HFFN_PYTHON:-python} -B "$leaf/native_train_launcher.py" "$@"
WRAPPER
chmod +x "$leaf/build/metal_hffn_train_step_runner"
