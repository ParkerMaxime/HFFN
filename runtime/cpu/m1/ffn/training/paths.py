import os
from pathlib import Path


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p/'runtime'/'common'/'pythia'/'hffn_pythia').is_dir())
if not os.environ.get("HFFN_ASSETS"):
    raise SystemExit("HFFN_ASSETS is not set: point it at the folder that holds homogeneous/models")
FFN_ROOT = Path(os.environ["HFFN_ASSETS"]) / "homogeneous" / "models"
FFN_SHARED = FFN_ROOT / "_shared"
TRAINING_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = TRAINING_ROOT / "runs"
