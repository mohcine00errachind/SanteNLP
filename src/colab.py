"""Helpers for the Colab notebook: Google Drive folders and checks.

Standard library only: the notebook imports this module directly, so it must
work even right after pip has upgraded numpy/pandas in the running session
(importing those in the notebook would then fail until a restart).
"""

from pathlib import Path

DRIVE_MOUNT = Path("/content/drive")
DRIVE_PROJECT = DRIVE_MOUNT / "MyDrive" / "SanteNLP"


def check_output_dir(path: str | Path) -> Path:
    """Stop with a clear error if an output folder is not the one we meant.

    - "{" or "}" in the path: a notebook variable such as {DRIVE} was not
      filled in. IPython leaves `!cmd {NAME}` unchanged when NAME is undefined
      (e.g. after a runtime restart), so files would go to a local folder
      literally named "{DRIVE}" on the Colab machine and be lost with it.
    - a path under /content/drive while Drive is not mounted: writing there
      would create an ordinary local folder, also lost with the machine.
    """
    text = str(path)
    if "{" in text or "}" in text:
        raise ValueError(
            f"Output path {text!r} contains '{{' or '}}': a notebook variable was not filled in. "
            "Rerun the cell that defines it."
        )
    path = Path(path)
    on_drive = path.as_posix().startswith(DRIVE_MOUNT.as_posix() + "/")
    if on_drive and not (DRIVE_MOUNT / "MyDrive").is_dir():
        raise RuntimeError(f"{text} is on Google Drive, but Drive is not mounted. Run step 2 (Connect Google Drive).")
    return path


def drive_dirs() -> tuple[Path, Path]:
    """The runs and models folders on Drive (created if needed).
    Fails with a clear error if Drive is not mounted."""
    runs = check_output_dir(DRIVE_PROJECT / "outputs" / "runs")
    models = check_output_dir(DRIVE_PROJECT / "models" / "bert_specialty")
    runs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return runs, models


def show_files(folder: str | Path) -> None:
    """Print the full path of a folder and every file in it with its size,
    to check what was really saved."""
    folder = Path(folder)
    if not folder.is_dir():
        print(f"{folder}: folder does not exist")
        return
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    print(f"{folder.resolve()}  ({len(files)} files)")
    for p in files:
        print(f"  {p.relative_to(folder).as_posix():60s} {p.stat().st_size / 1e6:8.1f} MB")
