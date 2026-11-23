"""Build constants of the M1 leaf, read by the shared ``compact_torch_extension`` (macOS arm64, clang++)."""
NAME = "hffn_compact_g4_dualroute_cpu_v1"
COMPILER = "clang++"
FLAGS = ["-O3", "-ffast-math", "-std=c++17", "-fPIC", "-dynamiclib", "-pthread"]
LINK_ARGUMENTS = ["-ltorch", "-ltorch_cpu", "-lc10", "-undefined", "dynamic_lookup"]
# First attempt through torch.utils.cpp_extension.load (kept, README section 8 point 3); without Ninja the command above is used.
# (outcome B, kernel_identical=false): the load recipe does not build the same machine code as the
# command branch. The 21 kernel bodies are equal, but two kernel cold parts (validate .cold.2, adamw_f32 .cold.1) and 14
# pybind11/libc++ binding symbols differ; cause not isolated. Both branches are kept; Ninja is absent from ML, so the command
# branch is the one that runs.
TORCH_LOAD = {"extra_cflags": ["-O3", "-ffast-math", "-std=c++17", "-pthread"], "extra_ldflags": ["-pthread"]}
