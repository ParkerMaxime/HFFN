"""Build constants of the x86 leaf, read by the shared ``compact_torch_extension`` (Linux x86_64, g++)."""
NAME = "hffn_compact_g4_training_x86"
COMPILER = "g++"
# -mno-recip: with -ffast-math GCC computes a vectorised sqrtf by a reciprocal square root, which returns NaN for a
# denormal input when the process does not flush denormals (a Python process does not): the AdamW update of an entry
# whose second moment is denormal then becomes NaN.
FLAGS = ["-O3", "-ffast-math", "-mno-recip", "-std=c++20", "-fPIC", "-shared", "-pthread"]      # C++20: required by the headers of PyTorch 2.14
LINK_ARGUMENTS = ["-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_python"]
TORCH_LOAD = None  # no first attempt through torch.utils.cpp_extension.load on x86
