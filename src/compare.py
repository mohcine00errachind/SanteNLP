"""Phase 4 comparison: fine-tuned BERT vs the TF-IDF baseline.

Uses the saved test predictions of finished runs (no BERT training here):
  1. overall test scores (main model + mean +/- std over seeds);
  2. the same on non-operative and operative reports, and on reports longer
     or shorter than BERT's limit;
  3. what happens to each of the baseline's errors: fixed or repeated by BERT;
  4. why head_tail scores below head: the TF-IDF baseline is retrained on only
     the tokens each strategy keeps ("window test").

Run: python -m src.compare --config configs/compare.yaml
Saves a run folder with config.yaml, metrics.json and report_table.csv
(row numbers, labels and predictions only, no report text).
"""

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from src.baseline import train_with_search
from src.data import is_operative, load_mtsamples_splits, resplit_train_val
from src.evaluate import mcnemar_p, subset_macro_f1
from src.utils import ROOT, load_config, save_run

MODELS = ["baseline", "head", "head_tail"]


def window(ids: list[int], first: int, last: int) -> list[int]:
    """The first `first` and last `last` tokens; the whole list if it is short enough."""
    if len(ids) <= first + last:
        return ids
    return ids[:first] + (ids[-last:] if last else [])


def baseline_seed_predictions(splits: dict, cfg: dict, seeds: list[int], logged: dict) -> dict:
    """Retrain the baseline for each seed and check it reproduces the logged scores."""
    test = splits["test"]
    preds = {}
    for seed in seeds:
        train, val = resplit_train_val(splits, seed)
        model, _, _ = train_with_search(train, val, cfg, seed)
        preds[seed] = model.predict(test["text"])
        f1 = round(f1_score(test["label"], preds[seed], average="macro"), 4)
        expected = next(s["test_macro_f1"] for s in logged["seeds"] if s["seed"] == seed)
        assert f1 == expected, f"baseline seed {seed}: {f1} differs from the logged {expected}"
    return preds


def load_predictions(cfg: dict, splits: dict, tokenizer) -> pd.DataFrame:
    """One row per test report: gold label, report type, length, every model's predictions."""
    test = splits["test"]
    table = pd.DataFrame({
        "test_row": range(len(test)),
        "label": test["label"].values,
        "operative": test["text"].map(is_operative).values,
        "n_tokens": [len(ids) for ids in tokenizer(list(test["text"]), add_special_tokens=False)["input_ids"]],
    })
    for name in ["head", "head_tail"]:
        saved = pd.read_csv(ROOT / cfg["runs"][name] / "test_predictions.csv")
        assert (saved["label"].values == table["label"].values).all(), f"{name}: test rows are not in the same order"
        table[name] = saved["pred"].values
        for seed in cfg["seeds"]:
            table[f"{name}_seed{seed}"] = saved[f"pred_seed{seed}"].values

    saved = pd.read_csv(ROOT / cfg["runs"]["baseline"] / "test_predictions.csv")
    assert (saved["label"].values == table["label"].values).all(), "baseline: test rows are not in the same order"
    table["baseline"] = saved["pred"].values
    logged = json.loads((ROOT / cfg["runs"]["baseline"] / "metrics.json").read_text(encoding="utf-8"))
    base_cfg = load_config(ROOT / cfg["baseline_config"])
    for seed, pred in baseline_seed_predictions(splits, base_cfg, cfg["seeds"], logged).items():
        table[f"baseline_seed{seed}"] = pred
    return table


def scores(table: pd.DataFrame, mask, seeds: list[int]) -> dict:
    """Main model and mean +/- std over seeds, on the reports selected by mask."""
    t = table[mask]
    out = {"n_reports": int(len(t))}
    for name in MODELS:
        seed_f1 = [subset_macro_f1(t["label"], t[f"{name}_seed{s}"]) for s in seeds]
        seed_err = [int((t["label"] != t[f"{name}_seed{s}"]).sum()) for s in seeds]
        out[name] = {
            "main_macro_f1": round(subset_macro_f1(t["label"], t[name]), 4),
            "main_errors": int((t["label"] != t[name]).sum()),
            "seeds_macro_f1_mean": round(float(np.mean(seed_f1)), 4),
            "seeds_macro_f1_std": round(float(np.std(seed_f1, ddof=1)), 4),
            "seeds_errors": seed_err,
        }
    return out


def error_analysis(table: pd.DataFrame, seeds: list[int]) -> dict:
    """What BERT does with the baseline's errors, and which errors it adds."""
    right = {name: table[name] == table["label"] for name in MODELS}  # main models
    n_right = {name: sum((table[f"{name}_seed{s}"] == table["label"]).astype(int) for s in seeds) for name in MODELS}

    def rows(mask) -> list[dict]:
        return [{"test_row": int(r["test_row"]), "label": r["label"], "operative": bool(r["operative"]),
                 "n_tokens": int(r["n_tokens"]),
                 **{f"{name}_pred": r[name] for name in MODELS},
                 **{f"{name}_seeds_right": int(n_right[name][i]) for name in MODELS}}
                for i, r in table[mask].iterrows()]

    out = {"baseline_errors": rows(~right["baseline"])}
    for name in ["head", "head_tail"]:
        out[name] = {
            "fixes_baseline_errors": int((~right["baseline"] & right[name]).sum()),
            "repeats_baseline_errors": int((~right["baseline"] & ~right[name]).sum()),
            "new_errors": rows(right["baseline"] & ~right[name]),
            "mcnemar_vs_baseline": mcnemar_p(table["label"], table["baseline"], table[name]),
            # same comparison for each seed (both models trained on that seed's train/val split)
            "mcnemar_vs_baseline_per_seed": {
                seed: mcnemar_p(table["label"], table[f"baseline_seed{seed}"], table[f"{name}_seed{seed}"])
                for seed in seeds},
        }
    out["mcnemar_head_vs_head_tail"] = mcnemar_p(table["label"], table["head"], table["head_tail"])
    # Which class do wrong predictions go to (all seeds together)?
    out["wrong_predictions_by_class"] = {
        name: dict(Counter(p for s in seeds for p, y in zip(table[f"{name}_seed{s}"], table["label"]) if p != y)
                   .most_common())
        for name in MODELS
    }
    return out


def window_test(splits: dict, cfg: dict, tokenizer, long_mask) -> dict:
    """TF-IDF baseline trained and tested on only some tokens of each report."""
    base_cfg = load_config(ROOT / cfg["baseline_config"])
    ids = {name: tokenizer(list(df["text"]), add_special_tokens=False)["input_ids"] for name, df in splits.items()}

    def cut(df, name, first, last):
        return df.assign(text=[tokenizer.decode(window(x, first, last)) for x in ids[name]])

    out = {}
    for win, (first, last) in cfg["windows"].items():
        # same tokens for every report, then the usual new train/val split per seed
        cut_splits = {name: cut(df, name, first, last) for name, df in splits.items()}
        all_f1, long_f1 = [], []
        for seed in cfg["seeds"]:
            train, val = resplit_train_val(cut_splits, seed)
            model, _, _ = train_with_search(train, val, base_cfg, seed)
            pred = model.predict(cut_splits["test"]["text"])
            y = cut_splits["test"]["label"].values
            all_f1.append(f1_score(y, pred, average="macro"))
            long_f1.append(f1_score(y[long_mask], pred[long_mask], average="macro"))
        out[win] = {"first_tokens": first, "last_tokens": last,
                    "test_macro_f1_mean": round(float(np.mean(all_f1)), 4),
                    "test_macro_f1_std": round(float(np.std(all_f1, ddof=1)), 4),
                    "long_reports_macro_f1_mean": round(float(np.mean(long_f1)), 4),
                    "long_reports_macro_f1_std": round(float(np.std(long_f1, ddof=1)), 4)}
        print(f"  window {win:10s} test macro-F1 {out[win]['test_macro_f1_mean']:.3f} ± {out[win]['test_macro_f1_std']:.3f}"
              f" | long reports {out[win]['long_reports_macro_f1_mean']:.3f}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BERT with the TF-IDF baseline.")
    parser.add_argument("--config", default="configs/compare.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    splits = load_mtsamples_splits()
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / cfg["tokenizer"]))
    table = load_predictions(cfg, splits, tokenizer)
    seeds = cfg["seeds"]
    is_long = (table["n_tokens"] > 510).values  # longer than BERT's 510 text tokens

    metrics = {
        "subsets": {
            "all": scores(table, table["label"].notna(), seeds),
            "non_operative": scores(table, ~table["operative"], seeds),
            "operative": scores(table, table["operative"], seeds),
            "longer_than_510_tokens": scores(table, is_long, seeds),
            "up_to_510_tokens": scores(table, ~is_long, seeds),
        },
        "errors": error_analysis(table, seeds),
    }
    for subset, s in metrics["subsets"].items():
        print(f"{subset} ({s['n_reports']} reports)")
        for name in MODELS:
            m = s[name]
            print(f"  {name:10s} main F1 {m['main_macro_f1']:.3f} ({m['main_errors']} errors) | "
                  f"seeds {m['seeds_macro_f1_mean']:.3f} ± {m['seeds_macro_f1_std']:.3f} errors {m['seeds_errors']}")
    for name in ["head", "head_tail"]:
        e = metrics["errors"][name]
        print(f"{name}: fixes {e['fixes_baseline_errors']} / repeats {e['repeats_baseline_errors']} baseline errors, "
              f"{len(e['new_errors'])} new errors, McNemar vs baseline {e['mcnemar_vs_baseline']}")
        print("  per seed p-values:", {k: v["p_value"] for k, v in e["mcnemar_vs_baseline_per_seed"].items()})

    print("Window test (TF-IDF baseline on part of each report):")
    metrics["window_test"] = window_test(splits, cfg, tokenizer, is_long)

    run_dir = save_run(cfg["name"], cfg, metrics)
    table.to_csv(run_dir / "report_table.csv", index=False)  # labels only, no report text
    print(f"Saved {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
