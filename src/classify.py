"""Fine-tune Bio_ClinicalBERT to classify reports by medical specialty.

How it works:
  1. The tokenizer cuts each report into word pieces (tokens).
  2. BERT reads at most 512 tokens, and 63% of our reports are longer, so
     long reports are shortened:
       head       keep the first 510 tokens
       head_tail  keep the first 128 and the last 382 tokens (the end of a
                  report often holds the assessment and plan)
  3. A classification layer on top of BERT's [CLS] vector predicts one of
     the 8 specialties. The whole network is fine-tuned for a few epochs,
     and the epoch with the best val macro-F1 is kept.

Same protocol as the baseline: a main model on the saved split, plus one run
per seed where the seed re-splits train/val (test fixed) -> mean +/- std.

Run (GPU recommended, e.g. Colab): python -m src.classify --config configs/classify.yaml
Quick CPU check:                    python -m src.classify --smoke
"""

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

from src.data import load_mtsamples_splits, resplit_train_val, stratified_subset
from src.evaluate import classification_metrics
from src.utils import ROOT, RUNS_DIR, git_commit, load_config, save_run, set_seed

# Saved next to each model so the app shortens texts exactly as in training
SETTINGS_FILE = "santenlp_settings.json"


# ---------------------------------------------------------------------------
# Shortening long reports
# ---------------------------------------------------------------------------

def select_tokens(ids: list[int], strategy: str, max_tokens: int, head_tokens: int) -> list[int]:
    """Shorten a list of token ids to at most max_tokens."""
    if len(ids) <= max_tokens:
        return ids
    if strategy == "head":
        return ids[:max_tokens]
    if strategy == "head_tail":
        return ids[:head_tokens] + ids[-(max_tokens - head_tokens):]
    raise ValueError(f"Unknown strategy: {strategy!r}")


def encode(tokenizer, texts, strategy: str, max_length: int, head_tokens: int) -> list[list[int]]:
    """Token ids for each text: shortened by the strategy, then wrapped as BERT
    expects: [CLS] tokens [SEP]."""
    budget = max_length - 2  # room for [CLS] and [SEP]: 510 tokens of text
    all_ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return [
        [tokenizer.cls_token_id] + select_tokens(ids, strategy, budget, head_tokens) + [tokenizer.sep_token_id]
        for ids in all_ids
    ]


def make_dataset(tokenizer, df: pd.DataFrame, label2id: dict, cfg: dict, strategy: str) -> Dataset:
    ids = encode(tokenizer, df["text"], strategy, cfg["max_length"], cfg["head_tokens"])
    return Dataset.from_dict({
        "input_ids": ids,
        "attention_mask": [[1] * len(x) for x in ids],  # 1 = real token (padding is added per batch)
        "labels": [label2id[label] for label in df["label"]],
    })


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(train, val, test, labels: list[str], cfg: dict, strategy: str, seed: int, checkpoint_dir: str):
    """Fine-tune one model, keep its best epoch (val macro-F1), predict on test.

    Returns (trainer, best val macro-F1, test predictions, per-epoch history).
    """
    set_seed(seed)
    t = cfg["training"]
    label2id = {label: i for i, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["model_revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], revision=cfg["model_revision"], num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=label2id,
    )
    data = {name: make_dataset(tokenizer, df, label2id, cfg, strategy)
            for name, df in [("train", train), ("val", val), ("test", test)]}

    total_steps = math.ceil(len(train) / t["batch_size"]) * t["epochs"]
    args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=t["epochs"],
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["eval_batch_size"],
        weight_decay=t["weight_decay"],
        warmup_steps=math.ceil(t["warmup_ratio"] * total_steps),
        # score on val after every epoch, keep the best epoch's weights
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=t["fp16"] and torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),  # only useful (and only valid) with a GPU
        seed=seed,
        data_seed=seed,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        return {"macro_f1": f1_score(y, logits.argmax(-1), average="macro")}

    trainer = Trainer(
        model=model, args=args,
        train_dataset=data["train"], eval_dataset=data["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),  # pads each batch to its longest report
        compute_metrics=compute_metrics,
    )
    trainer.train()

    history = [{"epoch": int(h["epoch"]), "val_macro_f1": round(h["eval_macro_f1"], 4)}
               for h in trainer.state.log_history if "eval_macro_f1" in h]
    logits = trainer.predict(data["test"]).predictions
    test_pred = [labels[i] for i in logits.argmax(-1)]
    return trainer, round(trainer.state.best_metric, 4), test_pred, history


def save_classifier(trainer, model_dir: Path, cfg: dict, strategy: str) -> None:
    """Save weights, tokenizer and the shortening settings (needed at prediction time)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    trainer.processing_class.save_pretrained(str(model_dir))
    settings = {"strategy": strategy, "max_length": cfg["max_length"], "head_tokens": cfg["head_tokens"]}
    (model_dir / SETTINGS_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")


def run_strategy(cfg: dict, strategy: str, splits: dict, model_dir: Path | None) -> tuple[dict, pd.DataFrame]:
    """Main model + one run per seed for one shortening strategy."""
    labels = sorted(splits["train"]["label"].unique())
    test = stratified_subset(splits["test"], cfg["eval_subset_size"], 0)
    val = stratified_subset(splits["val"], cfg["eval_subset_size"], 0)

    # 1. Main model on the saved split
    train = stratified_subset(splits["train"], cfg["subset_size"], cfg["seed"])
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ckpt:
        trainer, val_f1, test_pred, history = train_and_evaluate(
            train, val, test, labels, cfg, strategy, cfg["seed"], ckpt)
        if model_dir is not None:
            save_classifier(trainer, model_dir / strategy, cfg, strategy)
    main = {"train_size": len(train), "val_macro_f1": val_f1, "history": history,
            "test": classification_metrics(test["label"], test_pred, labels)}
    print(f"[{strategy}] main: val macro-F1 = {val_f1:.3f}, test macro-F1 = {main['test']['macro_f1']:.3f}", flush=True)
    predictions = pd.DataFrame({"test_row": range(len(test)), "label": test["label"].values, "pred": test_pred})

    # 2. One run per seed: new train/val split, same test set
    per_seed = []
    for seed in cfg["seeds"]:
        train_s, val_s = resplit_train_val(splits, seed)
        train_s = stratified_subset(train_s, cfg["subset_size"], seed)
        val_s = stratified_subset(val_s, cfg["eval_subset_size"], seed)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ckpt:
            _, val_f1_s, pred_s, history_s = train_and_evaluate(
                train_s, val_s, test, labels, cfg, strategy, seed, ckpt)
        m = classification_metrics(test["label"], pred_s, labels)
        per_seed.append({"seed": seed, "val_macro_f1": val_f1_s, "test_macro_f1": m["macro_f1"],
                         "test_accuracy": m["accuracy"], "history": history_s})
        predictions[f"pred_seed{seed}"] = pred_s
        print(f"[{strategy}] seed {seed}: val macro-F1 = {val_f1_s:.3f}, test macro-F1 = {m['macro_f1']:.3f}", flush=True)

    f1s = np.array([s["test_macro_f1"] for s in per_seed])
    metrics = {
        "strategy": strategy,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "main": main,
        "seeds": per_seed,
        "test_macro_f1_mean": round(float(f1s.mean()), 4),
        # sample std (ddof=1); undefined with a single seed (smoke run)
        "test_macro_f1_std": round(float(f1s.std(ddof=1)), 4) if len(f1s) > 1 else None,
    }
    return metrics, predictions


# ---------------------------------------------------------------------------
# Prediction (used by the evaluation and the app)
# ---------------------------------------------------------------------------

def load_classifier(model_dir: str | Path):
    """Load a saved model, its tokenizer and its shortening settings."""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    settings = json.loads((model_dir / SETTINGS_FILE).read_text(encoding="utf-8"))
    return tokenizer, model, settings


def predict_proba(texts: list[str], tokenizer, model, settings: dict, batch_size: int = 8) -> pd.DataFrame:
    """Probability of each specialty for each text (one row per text)."""
    ids = encode(tokenizer, texts, settings["strategy"], settings["max_length"], settings["head_tokens"])
    probs = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            batch = tokenizer.pad({"input_ids": ids[i:i + batch_size]}, return_tensors="pt")
            probs.append(torch.softmax(model(**batch).logits, dim=-1).numpy())
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return pd.DataFrame(np.vstack(probs), columns=labels)


# ---------------------------------------------------------------------------

def smoke_config(cfg: dict) -> dict:
    """Tiny settings to check the whole pipeline on CPU in a few minutes.
    The scores of a smoke run mean nothing."""
    cfg = dict(cfg, subset_size=24, eval_subset_size=24, max_length=128, head_tokens=32, seeds=[0])
    cfg["training"] = dict(cfg["training"], epochs=1, batch_size=8, eval_batch_size=8)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Bio_ClinicalBERT for specialty classification.")
    parser.add_argument("--config", default="configs/classify.yaml")
    parser.add_argument("--strategies", nargs="+", default=None, help="e.g. --strategies head")
    parser.add_argument("--subset", type=int, default=None, help="train on N reports")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run to check that everything works")
    parser.add_argument("--runs-dir", default=None, help="where run folders go (Colab: a Google Drive folder)")
    parser.add_argument("--model-dir", default=None, help="where the main models go (Colab: a Google Drive folder)")
    args = parser.parse_args()

    hf_logging.set_verbosity_error()  # hide the expected "sequence longer than 512" warnings
    cfg = load_config(args.config)
    if args.smoke:
        cfg = smoke_config(cfg)
    if args.subset:
        cfg["subset_size"] = args.subset
    if args.strategies:
        cfg["strategies"] = args.strategies
    cfg["git_commit"] = git_commit()  # which version of the code produced this run

    is_quick = args.smoke or cfg["subset_size"] is not None
    runs_dir = Path(args.runs_dir) if args.runs_dir else (ROOT / "outputs" / "smoke" if args.smoke else RUNS_DIR)
    # Quick runs never overwrite the real models
    model_dir = Path(args.model_dir) if args.model_dir else ROOT / cfg["model_dir"]
    if is_quick:
        model_dir = ROOT / "models" / "smoke_bert" if args.smoke else None

    splits = load_mtsamples_splits()
    print(f"Device: {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    for strategy in cfg["strategies"]:
        metrics, predictions = run_strategy(cfg, strategy, splits, model_dir)
        suffix = "_smoke" if args.smoke else ("_subset" if is_quick else "")
        run_dir = save_run(f"{cfg['name']}_{strategy}{suffix}", dict(cfg, strategies=[strategy]), metrics, runs_dir=runs_dir)
        # Only row numbers and labels, no report text
        predictions.to_csv(run_dir / "test_predictions.csv", index=False)
        std = metrics["test_macro_f1_std"]
        print(f"[{strategy}] test macro-F1 over seeds = {metrics['test_macro_f1_mean']:.3f}"
              + (f" ± {std:.3f}" if std is not None else "") + f" | saved {run_dir}", flush=True)

    if args.smoke:  # check that a saved model loads and predicts
        tokenizer, model, settings = load_classifier(model_dir / cfg["strategies"][0])
        probs = predict_proba(splits["test"]["text"].head(2).tolist(), tokenizer, model, settings)
        print("Reload check, probabilities for 2 test reports:\n", probs.round(3).to_string())


if __name__ == "__main__":
    main()
