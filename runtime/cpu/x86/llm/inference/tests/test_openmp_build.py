"""The OpenMP build of the kernel, the one this leaf runs, against the private-pool build of the same source.

The kernel source is shared by the x86 leaves and has two group schedulers chosen at compile time: a private
persistent pool (the build of the FFN leaf) and the OpenMP team of the process (`-fopenmp -DHFFN_GROUPS_OPENMP`,
this leaf). Both builds are compiled here from that source and run on a synthetic layer of irregular supports,
built from a fixed seed (no asset is read). The tests state, and fail if not:

- the bits of the output depend neither on the scheduler nor on the thread count T (decode, N = 1, and the dual
  route with batches split into blocks, N up to 130);
- the output is within ATOL of a float64 oracle of the same fixed-support FFN (the parity tolerance of the FFN
  leaf: 2e-5 on outputs of magnitude a few units);
- once PyTorch is imported, loading the OpenMP build leaves a single OpenMP runtime in the process (the claim that
  the kernel runs on PyTorch's threads).

PyTorch is imported first, as in the leaf, so that the OpenMP build can share its runtime.
"""
from __future__ import annotations

import ctypes
import subprocess
from unittest import mock

import pytest
import torch

from checks.parity_layers import dense_forward
from hffn_pythia_x86 import build
from hffn_pythia_x86.runtime import HFFNLayer, HFFNRuntime

D_MODEL, D_FF, LAYERS = 96, 384, 2
BATCHES = (1, 2, 7, 8, 32, 96, 130)
THREADS = (1, 2, 3, 8)
ATOL = 2e-5


def side(generator: torch.Generator, d_model: int, d_ff: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hidden-major compact storage of one projection: offsets, feature index, values. Degrees 4 to 60, some 0."""
    degrees = torch.randint(4, 61, (d_ff,), generator=generator)
    degrees[::37] = 0
    offsets = torch.zeros(d_ff + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(degrees, 0)
    index = torch.cat([torch.randperm(d_model, generator=generator)[:int(degree)].sort().values for degree in degrees]).to(torch.int16)
    values = torch.cat([torch.randn(int(degree), generator=generator) / max(int(degree), 1) ** 0.5 for degree in degrees])
    return offsets, index, values


def synthetic_layers(seed: int = 7) -> list[dict]:
    """Export-format layers (the arrays `read_layer` returns), identical on every machine."""
    generator = torch.Generator().manual_seed(seed)
    layers = []
    for _ in range(LAYERS):
        up_offsets, up_idx, up_val = side(generator, D_MODEL, D_FF)
        down_offsets, down_idx, down_val = side(generator, D_MODEL, D_FF)
        layers.append({"d_model": D_MODEL, "d_ff": D_FF, "up_row_offsets": up_offsets, "up_idx": up_idx, "up_val": up_val,
                       "b_up": 0.1 * torch.randn(D_FF, generator=generator), "down_row_offsets": down_offsets, "down_idx": down_idx,
                       "down_val": down_val, "b_down": 0.1 * torch.randn(D_MODEL, generator=generator)})
    return layers


def inputs(seed: int = 11) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {n: torch.randn(n, D_MODEL, generator=generator) for n in BATCHES}


@pytest.fixture(scope="module")
def libraries(tmp_path_factory) -> dict[str, object]:
    """The OpenMP build of the leaf, and the private-pool build of the same source and flags."""
    pool = tmp_path_factory.mktemp("pool") / "pool_build.so"
    flags = [flag for flag in build.COMMAND if flag not in ("-fopenmp", "-DHFFN_GROUPS_OPENMP")]
    subprocess.run([*flags, "-pthread", str(build.SOURCE), "-o", str(pool)], check=True)
    return {"openmp": build.build(), "pool": pool}


@pytest.fixture(scope="module")
def outputs(libraries) -> dict[tuple[str, int], dict[tuple[int, int], torch.Tensor]]:
    """Per (scheduler, T): the output of every layer and batch size."""
    compact = [HFFNLayer.from_export(layer) for layer in synthetic_layers()]
    x = inputs()
    result = {}
    for scheduler, library in libraries.items():
        for threads in THREADS:
            with mock.patch.object(build, "build", lambda library=library: library):
                hffn = HFFNRuntime(compact, threads)
            result[scheduler, threads] = {(index, n): hffn.forward(index, x[n]).clone() for index in range(LAYERS) for n in BATCHES}
    return result


def test_bits_do_not_depend_on_scheduler_or_threads(outputs):
    reference = outputs["openmp", 1]
    for key, produced in outputs.items():
        for cell, y in produced.items():
            assert torch.equal(y, reference[cell]), f"{key} differs from openmp T=1 on layer {cell[0]}, N={cell[1]}"


def test_output_within_tolerance_of_float64(outputs):
    layers, x = synthetic_layers(), inputs()
    for (index, n), y in outputs["openmp", 8].items():
        error = float((y.double() - dense_forward(layers[index], x[n].double())).abs().max())
        assert error <= ATOL, f"layer {index}, N={n}: |y - float64| = {error:.3e} > {ATOL:.0e}"


def test_the_layers_are_irregular_and_deterministic():
    first, second = synthetic_layers(), synthetic_layers()
    assert all(torch.equal(a[name], b[name]) for a, b in zip(first, second) for name in a if name not in ("d_model", "d_ff"))
    degrees = first[0]["up_row_offsets"].diff()
    assert int(degrees.min()) == 0 and int(degrees.max()) > 40


def test_a_single_openmp_runtime_in_the_process(libraries):
    """After `import torch` and the load of the OpenMP build, one libgomp file is mapped."""
    ctypes.CDLL(str(libraries["openmp"]))
    with open("/proc/self/maps") as maps:
        mapped = {line.split()[-1] for line in maps if "libgomp" in line}
    assert len(mapped) == 1, f"several OpenMP runtimes are mapped: {sorted(mapped)}"
