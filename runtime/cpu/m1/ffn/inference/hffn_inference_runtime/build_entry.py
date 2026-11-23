"""Build constants of the M1 inference leaf, read by its ``compact_torch_extension`` (macOS arm64, clang++)."""
NAME = "hffn_compact_g4_inference_cpu_v1"
COMPILER = "clang++"
FLAGS = ["-O3", "-ffast-math", "-std=c++17", "-fPIC", "-dynamiclib", "-pthread"]
LINK_ARGUMENTS = ["-ltorch", "-ltorch_cpu", "-lc10", "-undefined", "dynamic_lookup"]
# First attempt through torch.utils.cpp_extension.load, as the shared recipe; without Ninja the command above is used.
TORCH_LOAD = {"extra_cflags": ["-O3", "-ffast-math", "-std=c++17", "-pthread"], "extra_ldflags": ["-pthread"]}
