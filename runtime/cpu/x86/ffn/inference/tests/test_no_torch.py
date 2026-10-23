import subprocess
import sys

from hffn import build


def test_runtime_does_not_import_torch():
    """The x86 runtime (loader + kernels) runs without PyTorch."""
    code = ("import sys; from hffn.compact import CompactKernel; from hffn_ffn_layers import layers; "
            "layer, case = layers.load('homog_256_1024'); k = CompactKernel(layer); "
            "k.forward(layers.make_input(case, 1)); k.forward(layers.make_input(case, 8)); "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=build.ROOT)
