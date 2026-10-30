import pandas as pd
import spacy

from src.data import clean_mtsamples, is_operative, make_doc, parse_brat, split_dataset

CLASSES = ["Cardiovascular / Pulmonary", "Orthopedic", "Neurology"]
NON_SPECIALTY = ["Surgery", "Consult - History and Phy."]


def raw_rows():
    """A tiny fake MTSamples table that hits every cleaning step once."""
    rows = [
        ("Surgery", "Report A"),                     # A is filed twice:
        ("Cardiovascular / Pulmonary", "Report A"),  # -> kept as Cardio
        ("Consult - History and Phy.", "Report B"),  # catch-all only -> removed
        ("Neurology", "Report C"),                   # two specialties -> removed
        ("Orthopedic", "Report C"),
        ("Dermatology", "Report D"),                 # not one of our classes -> removed
        ("Orthopedic", "  Report E "),               # kept (spaces stripped)
        ("Orthopedic", None),                        # empty -> removed
        ("Orthopedic", "INDICATIONS:,"),             # 1 word < MIN_WORDS -> removed
    ]
    return pd.DataFrame(rows, columns=["medical_specialty", "transcription"])


MIN_WORDS = 2  # the fake reports are 2 words long ("Report A")


def test_clean_mtsamples_funnel_counts_every_step():
    _, funnel = clean_mtsamples(raw_rows(), CLASSES, NON_SPECIALTY, MIN_WORDS)
    assert funnel == {
        "raw_rows": 9,
        "removed_empty": 1,
        "removed_duplicates": 2,  # 8 rows with text, 6 distinct reports
        "removed_too_short": 1,
        "removed_catch_all_only": 1,
        "removed_multi_specialty": 1,
        "removed_other_specialty": 1,
        "kept": 2,
    }


def test_clean_mtsamples_keeps_the_specialty_not_the_catch_all():
    clean, _ = clean_mtsamples(raw_rows(), CLASSES, NON_SPECIALTY, MIN_WORDS)
    labels = dict(zip(clean["text"], clean["label"]))
    assert labels == {"Report A": "Cardiovascular / Pulmonary", "Report E": "Orthopedic"}
    raw = dict(zip(clean["text"], clean["raw_labels"]))
    assert raw["Report A"] == "Cardiovascular / Pulmonary; Surgery"


def test_split_dataset_is_stratified_disjoint_and_reproducible():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(200)], "label": ["a"] * 100 + ["b"] * 100})
    splits = split_dataset(df, val_size=0.15, test_size=0.15, seed=42, stratify_col="label")

    assert [len(splits[s]) for s in ["train", "val", "test"]] == [140, 30, 30]
    all_texts = pd.concat(splits.values())["text"]
    assert all_texts.is_unique and len(all_texts) == 200
    for part in splits.values():  # each split keeps the 50/50 class balance
        assert (part["label"] == "a").mean() == 0.5

    again = split_dataset(df, val_size=0.15, test_size=0.15, seed=42, stratify_col="label")
    assert list(again["test"]["text"]) == list(splits["test"]["text"])


def test_is_operative():
    assert is_operative("PREOPERATIVE DIAGNOSIS:, Cataract.")
    assert not is_operative("HISTORY OF PRESENT ILLNESS:, Chest pain.")


def test_parse_brat_reads_entities_and_skips_the_rest():
    ann = (
        "T1\tSign_symptom 0 5\tfever\n"
        "E1\tSign_symptom:T1 \n"                 # event line: ignored
        "T2\tMedication 15 22\taspirin\t\n"      # trailing tab (happens once in MACCROBAT)
        "T3\tBiological_structure 0 3;8 12\tx y\n"  # discontinuous: skipped
        "R1\tMODIFY Arg1:T2 Arg2:T1\t\n"         # relation line: ignored
    )
    entities, skipped = parse_brat(ann)
    assert entities == [(0, 5, "Sign_symptom"), (15, 22, "Medication")]
    assert skipped == ["Biological_structure"]


def test_make_doc_maps_labels_and_removes_overlaps():
    nlp = spacy.blank("en")
    text = "fever treated with aspirin therapy"
    entities = [
        (0, 5, "Sign_symptom"),                 # -> SYMPTOM
        (19, 26, "Medication"),                 # "aspirin", inside the longer span below
        (19, 34, "Therapeutic_procedure"),      # "aspirin therapy" -> kept (longer)
        (6, 13, "Clinical_event"),              # not in the map -> dropped
    ]
    label_map = {"Sign_symptom": "SYMPTOM", "Medication": "TREATMENT", "Therapeutic_procedure": "TREATMENT"}
    doc, stats = make_doc(nlp, text, entities, label_map)

    assert [(e.text, e.label_) for e in doc.ents] == [("fever", "SYMPTOM"), ("aspirin therapy", "TREATMENT")]
    assert stats == {"mapped": 3, "overlaps_removed": 1}
