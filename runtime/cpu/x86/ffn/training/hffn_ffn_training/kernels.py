"""Entry point of the x86 leaf for the shared ``step.py``: the slots entry of the dual route of the shared x86 inference
kernel. Its decode path (N = 1) does not store ``z``, hence the smallest N."""
from hffn_x86_common.inference import _F, _I, _L, _P, InferenceKernel, _bind  # noqa: F401

INFERENCE_MIN_N = 2
