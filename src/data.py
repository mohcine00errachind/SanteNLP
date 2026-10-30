"""Download and prepare the two datasets.

MTSamples  -> specialty classification (one label per report)
    raw CSV -> clean (remove empty, duplicate, ambiguous reports)
            -> stratified 70/15/15 split -> data/processed/mtsamples/*.csv

MACCROBAT  -> NER (SYMPTOM, TREATMENT)
    brat .txt/.ann files -> keep and rename 3 labels -> spaCy Doc objects
            -> 70/15/15 split by document -> data/processed/maccrobat/*.spacy

Run: python -m src.data --config configs/data.yaml
"""

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import spacy
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc, DocBin
from spacy.util import filter_spans

from src.utils import ROOT, load_config, set_seed

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# MTSamples
# ---------------------------------------------------------------------------

# Pinned to a specific commit of the dataset repo: if the owner changes the
# file later, we still get exactly the version this project was built on.
MTSAMPLES_REPO = "harishnair04/mtsamples"
MTSAMPLES_REVISION = "139c3fb2d00cd390766b15ff905f88dd40e0faee"

# Section headers that appear in operative / procedure reports. Used only for
# analysis (does the model learn "report type" instead of specialty?).
OPERATIVE_PATTERN = re.compile(
    r"PREOPERATIVE DIAGNOSIS|POSTOPERATIVE DIAGNOSIS|"
    r"DESCRIPTION OF (THE )?(PROCEDURE|OPERATION)|PROCEDURE IN DETAIL"
)


def download_mtsamples() -> Path:
    """Download mtsamples.csv into data/raw/ (skipped if already there)."""
    path = hf_hub_download(
        repo_id=MTSAMPLES_REPO,
        filename="mtsamples.csv",
        repo_type="dataset",
        revision=MTSAMPLES_REVISION,
        local_dir=RAW_DIR / "mtsamples",
    )
    return Path(path)


def load_mtsamples_raw() -> pd.DataFrame:
    """Load the raw CSV, with surrounding spaces stripped from the labels."""
    df = pd.read_csv(download_mtsamples())
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    return df


def clean_mtsamples(
    df: pd.DataFrame, classes: list[str], non_specialty: list[str], min_words: int
) -> tuple[pd.DataFrame, dict]:
    """Keep only complete reports that belong to exactly one of our specialties.

    Returns the clean table (text, label, raw_labels) and a "funnel" dict
    with how many reports each step removed.
    """
    funnel = {"raw_rows": len(df)}

    # 1. Reports without text are useless.
    df = df.dropna(subset=["transcription"])
    funnel["removed_empty"] = funnel["raw_rows"] - len(df)

    # 2. The same report often appears in several rows, once per label.
    #    Group identical texts and collect all the labels each one received.
    texts = df["transcription"].str.strip()
    labels_per_text = df.groupby(texts)["medical_specialty"].agg(lambda s: sorted(set(s)))
    funnel["removed_duplicates"] = len(df) - len(labels_per_text)

    # 3. Very short reports are truncated scrapes (e.g. "INDICATIONS:," and nothing else).
    n_words = labels_per_text.index.str.split().str.len()
    funnel["removed_too_short"] = int((n_words < min_words).sum())
    labels_per_text = labels_per_text[n_words >= min_words]

    # 4. Remove the catch-all / note-type labels; what is left are specialties.
    specialties = labels_per_text.map(lambda labels: [l for l in labels if l not in non_specialty])
    n_specialties = specialties.map(len)

    funnel["removed_catch_all_only"] = int((n_specialties == 0).sum())
    funnel["removed_multi_specialty"] = int((n_specialties > 1).sum())

    # 5. Keep reports with exactly one specialty, if it is one of our classes.
    single = specialties[n_specialties == 1].map(lambda labels: labels[0])
    in_classes = single.isin(classes)
    funnel["removed_other_specialty"] = int((~in_classes).sum())

    kept = single[in_classes]
    clean = pd.DataFrame({
        "text": kept.index,
        "label": kept.values,
        # All original labels, kept for analysis (e.g. "also filed as Surgery").
        "raw_labels": labels_per_text[kept.index].map("; ".join).values,
    })
    funnel["kept"] = len(clean)
    return clean, funnel


def split_dataset(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int, stratify_col: str | None = None
) -> dict[str, pd.DataFrame]:
    """Split into train/val/test. If stratify_col is given, every split keeps
    the same class proportions as the full dataset."""
    strat = df[stratify_col] if stratify_col else None
    train, rest = train_test_split(
        df, test_size=val_size + test_size, random_state=seed, stratify=strat
    )
    strat = rest[stratify_col] if stratify_col else None
    val, test = train_test_split(
        rest, test_size=test_size / (val_size + test_size), random_state=seed, stratify=strat
    )
    return {"train": train, "val": val, "test": test}


def is_operative(text: str) -> bool:
    """Rough check: does this report look like an operative/procedure note?"""
    return bool(OPERATIVE_PATTERN.search(text))


def prepare_mtsamples(cfg: dict, seed: int) -> dict:
    """Clean, split and save MTSamples. Returns the cleaning funnel."""
    clean, funnel = clean_mtsamples(
        load_mtsamples_raw(), cfg["classes"], cfg["non_specialty_labels"], cfg["min_words"]
    )
    splits = split_dataset(clean, cfg["val_size"], cfg["test_size"], seed, stratify_col="label")

    out_dir = PROCESSED_DIR / "mtsamples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False)
        funnel[f"{name}_size"] = len(part)
    with open(out_dir / "funnel.json", "w", encoding="utf-8") as f:
        json.dump(funnel, f, indent=2)
    return funnel


def load_mtsamples_splits() -> dict[str, pd.DataFrame]:
    """Load the saved splits. Every model uses these exact files."""
    out_dir = PROCESSED_DIR / "mtsamples"
    return {name: pd.read_csv(out_dir / f"{name}.csv") for name in ["train", "val", "test"]}


# ---------------------------------------------------------------------------
# MACCROBAT
# ---------------------------------------------------------------------------

# Original source on figshare (CC BY 4.0), with the checksum figshare publishes.
MACCROBAT_URL = "https://ndownloader.figshare.com/files/21489405"
MACCROBAT_MD5 = "9d090f3e0719e77e7a33c38750548bb8"
MACCROBAT_DIR = RAW_DIR / "maccrobat"


def download_maccrobat() -> Path:
    """Download and unzip MACCROBAT2020 into data/raw/maccrobat/."""
    zip_path = MACCROBAT_DIR / "MACCROBAT2020.zip"
    if not zip_path.exists():
        MACCROBAT_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MACCROBAT_URL, zip_path)

    md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
    if md5 != MACCROBAT_MD5:
        raise ValueError(f"MACCROBAT checksum mismatch: got {md5}, expected {MACCROBAT_MD5}")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(MACCROBAT_DIR)
    return MACCROBAT_DIR


def parse_brat(ann_text: str) -> tuple[list[tuple[int, int, str]], list[str]]:
    """Read entity lines from a brat .ann file.

    Entity lines look like:  T7<TAB>Sign_symptom 78 90<TAB>palpitations
    Other lines (events E, relations R, ...) are ignored. Discontinuous
    entities ("78 82;85 90") are skipped because spaCy entities must be one
    continuous span.

    Returns (list of (start, end, label), labels of the skipped entities).
    """
    entities, skipped = [], []
    for line in ann_text.splitlines():
        if not line.startswith("T"):
            continue
        _, info, _surface = line.split("\t", 2)  # at most 2 splits: text may end with a tab
        label, offsets = info.split(" ", 1)
        if ";" in offsets:
            skipped.append(label)
            continue
        start, end = map(int, offsets.split())
        entities.append((start, end, label))
    return entities, skipped


def make_doc(nlp, text: str, entities: list[tuple[int, int, str]], label_map: dict) -> tuple[Doc, dict]:
    """Build a spaCy Doc with our entity labels.

    - entities whose label is not in label_map are dropped
    - character offsets are snapped to spaCy token boundaries
    - if two kept entities overlap, only the longer one is kept
    """
    doc = nlp.make_doc(text)
    spans = []
    for start, end, label in entities:
        if label not in label_map:
            continue
        # "expand": if an offset falls inside a token, include the whole token
        span = doc.char_span(start, end, label=label_map[label], alignment_mode="expand")
        if span is not None:
            spans.append(span)
    kept = filter_spans(spans)  # removes overlaps, prefers longer spans
    doc.ents = kept
    return doc, {"mapped": len(spans), "overlaps_removed": len(spans) - len(kept)}


def prepare_maccrobat(cfg: dict, seed: int) -> dict:
    """Convert MACCROBAT to spaCy's binary format, split by document."""
    raw_dir = download_maccrobat()
    nlp = spacy.blank("en")  # tokenizer only, no trained model needed

    docs, stats = {}, {"documents": 0, "discontinuous_skipped": 0, "overlaps_removed": 0}
    for ann_path in sorted(raw_dir.glob("*.ann")):  # sorted -> same order everywhere
        text = ann_path.with_suffix(".txt").read_text(encoding="utf-8")
        entities, skipped = parse_brat(ann_path.read_text(encoding="utf-8"))
        doc, doc_stats = make_doc(nlp, text, entities, cfg["label_map"])
        docs[ann_path.stem] = doc
        stats["documents"] += 1
        stats["overlaps_removed"] += doc_stats["overlaps_removed"]
        # Only skipped entities of the labels we keep matter.
        stats["discontinuous_skipped"] += sum(label in cfg["label_map"] for label in skipped)

    ids = pd.DataFrame({"doc_id": list(docs)})
    splits = split_dataset(ids, cfg["val_size"], cfg["test_size"], seed)

    out_dir = PROCESSED_DIR / "maccrobat"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        split_docs = [docs[i] for i in part["doc_id"]]
        DocBin(docs=split_docs).to_disk(out_dir / f"{name}.spacy")
        stats[f"{name}_docs"] = len(split_docs)
        for label in sorted(set(cfg["label_map"].values())):
            stats[f"{name}_{label}"] = sum(1 for d in split_docs for e in d.ents if e.label_ == label)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_maccrobat_split(name: str, nlp=None) -> list[Doc]:
    """Load one saved split ("train", "val" or "test") as a list of Docs."""
    nlp = nlp or spacy.blank("en")
    return list(DocBin().from_disk(PROCESSED_DIR / "maccrobat" / f"{name}.spacy").get_docs(nlp.vocab))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MTSamples and MACCROBAT.")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    print("MTSamples cleaning funnel:")
    for step, n in prepare_mtsamples(config["mtsamples"], config["seed"]).items():
        print(f"  {step:26s} {n}")

    print("MACCROBAT:")
    for key, n in prepare_maccrobat(config["maccrobat"], config["seed"]).items():
        print(f"  {key:26s} {n}")


if __name__ == "__main__":
    main()
