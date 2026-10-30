"""TF-IDF + logistic regression baseline for specialty classification.

Every other classifier is compared against this model.

  TF-IDF: turns each report into a vector with one weight per word (or word
          pair). Words frequent in a report but rare overall get high weights.
  Logistic regression: learns one weight per (term, class) and predicts the
          class with the highest score.

Two kinds of runs:
  1. Main model: trained on the saved train split, C tuned on the saved val
     split, scored on test. Saved to models/baseline/ for the app.
  2. Multi-seed estimate: for each seed, a new train/val split (test fixed),
     so we can report macro-F1 as mean +/- std.

Run: python -m src.baseline --config configs/baseline.yaml
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.data import load_mtsamples_splits, resplit_train_val, stratified_subset
from src.evaluate import classification_metrics
from src.utils import ROOT, load_config, save_run, set_seed


def build_pipeline(cfg: dict, C: float, seed: int) -> Pipeline:
    """TF-IDF vectorizer followed by a logistic regression classifier."""
    tfidf_cfg = dict(cfg["tfidf"], ngram_range=tuple(cfg["tfidf"]["ngram_range"]))
    return Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_cfg)),
        ("clf", LogisticRegression(C=C, random_state=seed, **cfg["logreg"])),
    ])


def train_with_search(
    train: pd.DataFrame, val: pd.DataFrame, cfg: dict, seed: int
) -> tuple[Pipeline, float, float]:
    """Fit one model per C value on train; keep the best one on val (macro-F1).

    Returns (best model, best C, its val macro-F1).
    """
    best = None
    for C in cfg["search"]["C"]:
        model = build_pipeline(cfg, C, seed).fit(train["text"], train["label"])
        val_f1 = f1_score(val["label"], model.predict(val["text"]), average="macro")
        if best is None or val_f1 > best[2]:  # ties keep the smaller C (more regularized)
            best = (model, C, val_f1)
    return best


def run(cfg: dict) -> tuple[dict, Pipeline, pd.DataFrame]:
    splits = load_mtsamples_splits()
    test = splits["test"]
    labels = sorted(splits["train"]["label"].unique())

    # 1. Main model on the saved split
    set_seed(cfg["seed"])
    train = stratified_subset(splits["train"], cfg["subset_size"], cfg["seed"])
    model, C, val_f1 = train_with_search(train, splits["val"], cfg, cfg["seed"])
    test_pred = model.predict(test["text"])
    main = {
        "train_size": len(train),
        "best_C": C,
        "val_macro_f1": round(val_f1, 4),
        "test": classification_metrics(test["label"], test_pred, labels),
    }
    predictions = pd.DataFrame({"test_row": range(len(test)), "label": test["label"], "pred": test_pred})

    # 2. Multi-seed estimate: new train/val split per seed, same test set
    per_seed = []
    for seed in cfg["seeds"]:
        set_seed(seed)
        train_s, val_s = resplit_train_val(splits, seed)
        train_s = stratified_subset(train_s, cfg["subset_size"], seed)
        model_s, C_s, val_f1_s = train_with_search(train_s, val_s, cfg, seed)
        m = classification_metrics(test["label"], model_s.predict(test["text"]), labels)
        per_seed.append({
            "seed": seed, "best_C": C_s, "val_macro_f1": round(val_f1_s, 4),
            "test_macro_f1": m["macro_f1"], "test_accuracy": m["accuracy"],
        })

    f1s = np.array([s["test_macro_f1"] for s in per_seed])
    metrics = {
        "main": main,
        "seeds": per_seed,
        # sample standard deviation (ddof=1): 5 seeds are a sample of all possible splits
        "test_macro_f1_mean": round(float(f1s.mean()), 4),
        "test_macro_f1_std": round(float(f1s.std(ddof=1)), 4),
    }
    return metrics, model, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF + logistic regression baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--subset", type=int, default=None, help="train on N reports (quick local run)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.subset:
        cfg["subset_size"] = args.subset

    metrics, model, predictions = run(cfg)

    # A quick subset run must never replace the real model used by the app.
    is_subset = cfg["subset_size"] is not None
    model_dir = ROOT / cfg["model_dir"]
    if not is_subset:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")

    run_dir = save_run(cfg["name"] + ("_subset" if is_subset else ""), cfg, metrics)
    # Only row numbers and labels, no report text (data is never committed)
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    test = metrics["main"]["test"]
    print(f"Main model (saved split): best C = {metrics['main']['best_C']}, "
          f"val macro-F1 = {metrics['main']['val_macro_f1']:.3f}")
    print(f"  test: macro-F1 = {test['macro_f1']:.3f} | macro-P = {test['macro_precision']:.3f} | "
          f"macro-R = {test['macro_recall']:.3f} | accuracy = {test['accuracy']:.3f}")
    print("  per class:")
    for label, s in test["per_class"].items():
        print(f"    {label:28s} P={s['precision']:.2f} R={s['recall']:.2f} F1={s['f1']:.2f} (n={s['support']})")
    print("Multi-seed (new train/val split per seed, same test):")
    for s in metrics["seeds"]:
        print(f"  seed {s['seed']}: C = {s['best_C']:<5} val F1 = {s['val_macro_f1']:.3f}  test F1 = {s['test_macro_f1']:.3f}")
    print(f"  test macro-F1 = {metrics['test_macro_f1_mean']:.3f} ± {metrics['test_macro_f1_std']:.3f}")
    print(f"Saved run to {run_dir.relative_to(ROOT)}")
    print("Subset run: model not saved." if is_subset else f"Saved model to {model_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
