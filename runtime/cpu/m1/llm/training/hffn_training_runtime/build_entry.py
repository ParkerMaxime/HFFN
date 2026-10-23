"""Build constants of the M1 leaf, read by the shared ``compact_torch_extension`` (macOS arm64, clang++)."""
NAME = "hffn_compact_g4_dualroute_cpu_v1"
COMPILER = "clang++"
FLAGS = ["-O3", "-ffast-math", "-std=c++17", "-fPIC", "-dynamiclib", "-pthread"]
LINK_ARGUMENTS = ["-ltorch", "-ltorch_cpu", "-lc10", "-undefined", "dynamic_lookup"]
# First attempt through torch.utils.cpp_extension.load; without Ninja the command above is used. The two recipes do not
# build the same machine code (the 21 kernel bodies are equal; two cold parts of kernels and 14 binding symbols differ),
# so the build that was measured is the command branch: install nothing that provides Ninja when reproducing a result.
TORCH_LOAD = {"extra_cflags": ["-O3", "-ffast-math", "-std=c++17", "-pthread"], "extra_ldflags": ["-pthread"]}
