# CPU shared level: native build recipes

The two build calls that the CPU leaves share, so that no leaf carries its own compile command. Neither file is a package: a
leaf loads it by path, from the first ancestor folder that holds `runtime/`.

| File | Role |
|---|---|
| `compact_extension.py` | compile and load a PyTorch extension of a compact G4 kernel (M1 and x86 CPU leaves): the leaf gives the module name, compiler, flags, link arguments and sources; the library lives in a folder named by their hash, so a change of flags always rebuilds. An optional first attempt through `torch.utils.cpp_extension.load` (M1 only) falls back to the plain command when Ninja is missing. |
| `plain_dylib.py` | compile a torch-free native library once per command and sources, named by the hash of the command, the source and the files it includes; atomic for the readers of the final name. |

Nothing here runs a kernel: it only builds one. Outputs go to the `build/` folder of the leaf that asks.
