"""Checks on the Colab notebook that can run locally, without Colab."""

import json
import re

import pytest
from IPython.core.inputtransformer2 import TransformerManager
from IPython.core.interactiveshell import InteractiveShell

from src.utils import ROOT

NOTEBOOK = ROOT / "notebooks" / "colab_train_bert.ipynb"
# Installed on Colab by other means (torch: Colab's GPU build) or not needed there
EXCLUDED = {"torch", "en_core_web_md", "ipykernel", "nbformat", "nbclient", "streamlit"}


def code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def requirement_names() -> list[tuple[str, str]]:
    """(line, package name) for every requirement line in requirements.txt."""
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [(line, re.match(r"[A-Za-z0-9_.-]+", line).group())
            for line in lines if line.strip() and not line.startswith("#")]


def test_no_control_characters():
    # A "\b" written without escaping becomes a backspace and silently breaks grep
    # (checked on the decoded cells: in the file, JSON stores it as the text "\b")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "".join(cell["source"]))


def test_every_code_cell_compiles():
    # Turns !shell, %magic and %%bash lines into Python first, as Colab does
    for source in code_cells():
        compile(TransformerManager().transform_cell(source), "<cell>", "exec")


def test_step4_filter_excludes_the_right_packages():
    [grep_line] = [line for cell in code_cells() for line in cell.splitlines() if line.startswith("!grep")]
    # What IPython actually runs: also checks that "$" in the pattern is not taken as a variable
    command = InteractiveShell.instance().var_expand(grep_line[1:])
    pattern = re.search(r'-E "([^"]+)"', command).group(1)  # grep -E and Python re agree on this pattern

    kept = {name for line, name in requirement_names() if not re.search(pattern, line)}
    all_names = {name for _, name in requirement_names()}
    assert EXCLUDED <= all_names  # the excluded packages really are in requirements.txt
    assert kept == all_names - EXCLUDED  # exactly those are excluded, everything else is kept


@pytest.mark.parametrize("line", ["torchvision==0.30.0", "streamlit-extras==1.0", "nbformat_x==1"])
def test_step4_filter_keeps_similar_names(line):
    [grep_line] = [line for cell in code_cells() for line in cell.splitlines() if line.startswith("!grep")]
    pattern = re.search(r'-E "([^"]+)"', grep_line).group(1)
    assert not re.search(pattern, line)
