from __future__ import annotations

import torch


REFERENCE_BACKEND_NAME = "pytorch_cpu_auto_thread_masked"
HFFN_BACKEND_NAME = "hffn_compact_python_cpu"
HFFN_NATIVE_FORWARD_BACKEND_NAME = "hffn_compact_native_cpu_forward"


def get_torch_cpu_runtime_info() -> dict[str, object]:
    return {
        "device": "cpu",
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "thread_policy": "auto",
    }
