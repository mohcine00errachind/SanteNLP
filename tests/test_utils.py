import json
import random

import numpy as np
import yaml

from src.utils import load_config, save_run, set_seed


def test_set_seed_makes_random_numbers_repeatable():
    set_seed(42)
    first = (random.random(), np.random.rand())
    set_seed(42)
    second = (random.random(), np.random.rand())
    assert first == second


def test_set_seed_fixes_torch():
    import torch

    set_seed(0)
    a = torch.rand(3)
    set_seed(0)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\nmodel:\n  lr: 0.001\n", encoding="utf-8")
    assert load_config(path) == {"seed": 42, "model": {"lr": 0.001}}


def test_save_run_writes_config_and_metrics(tmp_path):
    config = {"seed": 42, "C": 1.0}
    metrics = {"macro_f1": 0.5}
    run_dir = save_run("test", config, metrics, runs_dir=tmp_path)

    assert run_dir.name.endswith("_test")
    assert yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) == config
    assert json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) == metrics
