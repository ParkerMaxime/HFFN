"""Where the Pythia assets of a machine are: the base checkpoints and the compact exports.

The folder named by the environment variable ``HFFN_ASSETS`` holds

    $HFFN_ASSETS/pythia/models/base/pythia70m, pythia160m, pythia410m     base checkpoints
    $HFFN_ASSETS/pythia/exports/<model>_sparse<density>_ffn_only/export   compact exports

There is no default location: when the variable is not set, the two folders must be given explicitly.

    python -B -c "from hffn_pythia.assets import models_dir; print(models_dir())"
"""
from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    value = os.environ.get("HFFN_ASSETS")
    if not value:
        raise SystemExit("HFFN_ASSETS is not set: point it at the folder that holds pythia/models/base and pythia/exports, "
                         "or give the base checkpoints folder and the exports folder explicitly")
    return Path(value) / "pythia"


def models_dir() -> Path:
    return root() / "models" / "base"


def exports_dir() -> Path:
    return root() / "exports"
