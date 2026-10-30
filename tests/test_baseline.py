import pandas as pd

from src.baseline import build_pipeline, train_with_search
from src.data import resplit_train_val, stratified_subset
from src.evaluate import classification_metrics

# Small config for tiny data (min_df=1: every word counts)
CFG = {
    "tfidf": {"lowercase": True, "ngram_range": [1, 2], "min_df": 1, "max_df": 1.0, "sublinear_tf": True},
    "logreg": {"class_weight": "balanced", "max_iter": 1000},
    "search": {"C": [0.1, 1, 10]},
}


def toy_reports(n_per_class=20):
    """Two 'specialties' with clearly different vocabulary."""
    cardio = [f"chest pain echocardiogram heart murmur case {i}" for i in range(n_per_class)]
    ortho = [f"knee fracture cast bone x-ray case {i}" for i in range(n_per_class)]
    return pd.DataFrame({"text": cardio + ortho, "label": ["cardio"] * n_per_class + ["ortho"] * n_per_class})


def test_baseline_learns_an_easy_task():
    df = toy_reports()
    model, C, val_f1 = train_with_search(df, df, CFG, seed=0)
    assert C in CFG["search"]["C"]
    assert val_f1 == 1.0
    assert list(model.predict(["heart murmur", "bone fracture"])) == ["cardio", "ortho"]


def test_baseline_is_deterministic_so_seeds_alone_change_nothing():
    """Why the multi-seed runs re-split train/val: with the same data,
    logistic regression gives the same model whatever the seed."""
    df = toy_reports()
    probs = [build_pipeline(CFG, C=1, seed=s).fit(df["text"], df["label"]).predict_proba(df["text"]) for s in (0, 1)]
    assert (probs[0] == probs[1]).all()


def test_resplit_keeps_test_untouched_and_sizes():
    df = toy_reports(50)
    splits = {"train": df.iloc[:70], "val": df.iloc[70:85], "test": df.iloc[85:]}
    train, val = resplit_train_val(splits, seed=3)
    assert len(train) == 70 and len(val) == 15
    assert set(train["text"]).isdisjoint(val["text"])
    assert set(train["text"]) | set(val["text"]) == set(df.iloc[:85]["text"])  # nothing from test


def test_stratified_subset():
    df = toy_reports(50)
    sub = stratified_subset(df, 20, seed=0)
    assert len(sub) == 20 and (sub["label"] == "cardio").sum() == 10
    assert stratified_subset(df, None, seed=0) is df


def test_classification_metrics_known_values():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    m = classification_metrics(y_true, y_pred, labels=["a", "b"])
    assert m["accuracy"] == 0.75
    # class a: P=1, R=0.5, F1=0.667 ; class b: P=0.667, R=1, F1=0.8 -> macro F1 = 0.7333
    assert m["macro_f1"] == 0.7333
    assert m["per_class"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    assert m["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]
