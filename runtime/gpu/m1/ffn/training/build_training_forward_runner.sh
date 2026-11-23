#!/usr/bin/env zsh
set -euo pipefail
leaf=${0:A:h}
mkdir -p "$leaf/build"
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal "$leaf/metal_hffn_training_forward_runner.mm" -o "$leaf/build/metal_hffn_training_forward_runner.native"
cat > "$leaf/build/metal_hffn_training_forward_runner" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
leaf=$(cd "$(dirname "$0")/.." && pwd)
exec ${HFFN_PYTHON:-python} -B "$leaf/run_native_train_step.py" --hffn-forward-prototype "$@"
WRAPPER
chmod +x "$leaf/build/metal_hffn_training_forward_runner"
