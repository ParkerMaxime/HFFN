#!/usr/bin/env bash
# Builds the bridge of native_ffn_adapter.py into build/.
# Without ARC: the installed PyTorch OperationUtils.h uses manual retain/release.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
python=${HFFN_PYTHON:-python}
torch_root=$($python -B -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent)')
torch_version=$($python -B -c 'import torch;print(torch.__version__)')
if [ "$torch_version" != "2.10.0" ]; then
  echo "bridge requires torch 2.10.0 (internal header), found $torch_version" >&2
  exit 2
fi
py_include=$($python -B -c 'import sysconfig;print(sysconfig.get_path("include"))')
mkdir -p "$here/build"
clang++ -std=c++17 -O2 -fno-objc-arc -shared -undefined dynamic_lookup \
  -DTORCH_EXTENSION_NAME=native_ffn_bridge -I "$py_include" -I "$torch_root/include" \
  -I "$torch_root/include/torch/csrc/api/include" "$here/bridge.mm" \
  -L "$torch_root/lib" -Wl,-rpath,"$torch_root/lib" \
  -ltorch -ltorch_cpu -ltorch_python -lc10 -framework Foundation -framework Metal \
  -o "$here/build/native_ffn_bridge.so"
echo "built $here/build/native_ffn_bridge.so"
