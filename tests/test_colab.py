import sys

import pytest

from src import classify
from src.colab import check_output_dir, show_files


def test_unfilled_notebook_variable_is_rejected():
    # What IPython passes on when DRIVE is undefined: the text "{DRIVE}" itself
    with pytest.raises(ValueError, match="not filled in"):
        check_output_dir("{DRIVE}/outputs/runs")


def test_drive_path_without_mounted_drive_is_rejected():
    # Google Drive is never mounted at /content/drive on this machine
    with pytest.raises(RuntimeError, match="not mounted"):
        check_output_dir("/content/drive/MyDrive/SanteNLP/outputs/runs")


def test_normal_folder_is_accepted(tmp_path):
    assert check_output_dir(tmp_path / "runs") == tmp_path / "runs"


def test_classify_stops_before_training(monkeypatch):
    # Must fail at once, not after hours of training
    monkeypatch.setattr(sys, "argv", ["classify", "--runs-dir", "{DRIVE}/outputs/runs"])
    monkeypatch.setattr(classify, "load_mtsamples_splits", lambda: pytest.fail("training should not start"))
    with pytest.raises(ValueError, match="not filled in"):
        classify.main()


def test_show_files_lists_saved_files(tmp_path, capsys):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "metrics.json").write_text("{}")
    show_files(tmp_path)
    out = capsys.readouterr().out
    assert str(tmp_path.resolve()) in out and "run/metrics.json" in out and "(1 files)" in out
