"""Small helpers shared by every script: seeds, config loading, run logging."""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Project root = the folder that contains src/. Paths are built from here so
# scripts work no matter which folder they are launched from (local or Colab).
ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "outputs" / "runs"


def set_seed(seed: int) -> None:
    """Fix every random number generator we use, so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # no-op when there is no GPU
    except ImportError:
        pass  # torch is only needed for the BERT part


def load_config(path: str | Path) -> dict:
    """Read a YAML config file into a plain Python dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_run(name: str, config: dict, metrics: dict, runs_dir: Path = RUNS_DIR) -> Path:
    """Save the config and metrics of one experiment in its own folder.

    The folder is named like 2026-10-29_14-05-12_baseline, so runs never
    overwrite each other and are sorted by date.
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(runs_dir) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return run_dir
