"""Entity extraction (SYMPTOM, TREATMENT), trained on MACCROBAT.

Two models:
  1. Dictionary baseline: a spaCy EntityRuler that tags every phrase seen as
     an entity in the training documents (with a precision filter, see below).
  2. Trained spaCy NER: a statistical model that learns from context, so it
     can find entities it never saw during training.

Both are scored with exact-match entity precision/recall/F1: a predicted
entity counts only if its start, end and label are all correct.

Run: python -m src.ner --config configs/ner.yaml
"""

import argparse
import itertools
import random
from collections import Counter, defaultdict

import numpy as np
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import minibatch

from src.data import load_maccrobat_split, load_mtsamples_splits
from src.utils import ROOT, load_config, save_run, set_seed


def split_into_sentences(docs: list[Doc]) -> list[Doc]:
    """MACCROBAT files have one sentence per line: cut each document at line
    breaks. Entities never cross a line break, so none are lost."""
    sentences = []
    for doc in docs:
        start = 0
        for token in doc:
            if "\n" in token.text:
                if token.i > start:
                    sentences.append(doc[start:token.i].as_doc())
                start = token.i + 1
        if start < len(doc):
            sentences.append(doc[start:].as_doc())
    return sentences


def to_examples(nlp, gold_docs: list[Doc]) -> list[Example]:
    """Pair each gold Doc with an unannotated copy of its text (what spaCy trains on)."""
    return [Example(nlp.make_doc(doc.text), doc) for doc in gold_docs]


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Do two (start, end) character spans share at least one character?"""
    return a[0] < b[1] and b[0] < a[1]


def _prf(tp_pred: int, n_pred: int, tp_gold: int, n_gold: int) -> dict:
    """Precision = correct predictions / predictions; recall = found gold / gold."""
    p = tp_pred / n_pred if n_pred else 0.0
    r = tp_gold / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f": round(f, 4)}


def evaluate_ner(nlp, gold_docs: list[Doc], labels: list[str]) -> dict:
    """Entity scores per label, overall (micro) and macro (average over labels).

    exact:   a prediction is correct only if start, end and label all match.
    overlap: a prediction is correct if it overlaps a gold entity of the same
             label ("murmur" vs gold "regurgitation murmur"). The gap between
             the two shows how many errors are only about span boundaries.
    """
    pred_docs = list(nlp.pipe(doc.text for doc in gold_docs))
    counts = {mode: {l: Counter() for l in labels} for mode in ["exact", "overlap"]}
    for pred, gold in zip(pred_docs, gold_docs):
        p_ents = [(e.start_char, e.end_char, e.label_) for e in pred.ents]
        g_ents = [(e.start_char, e.end_char, e.label_) for e in gold.ents]
        for label in labels:
            p = [(s, e) for s, e, l in p_ents if l == label]
            g = [(s, e) for s, e, l in g_ents if l == label]
            exact = set(p) & set(g)
            for mode, tp_pred, tp_gold in [
                ("exact", len(exact), len(exact)),
                ("overlap", sum(any(_overlaps(x, y) for y in g) for x in p),
                            sum(any(_overlaps(y, x) for x in p) for y in g)),
            ]:
                counts[mode][label].update(tp_pred=tp_pred, n_pred=len(p), tp_gold=tp_gold, n_gold=len(g))

    result = {}
    for mode, per in counts.items():
        per_label = {l: _prf(c["tp_pred"], c["n_pred"], c["tp_gold"], c["n_gold"]) for l, c in per.items()}
        total = sum(per.values(), Counter())
        result[mode] = {
            "macro_f1": round(float(np.mean([per_label[l]["f"] for l in labels])), 4),
            "micro": _prf(total["tp_pred"], total["n_pred"], total["tp_gold"], total["n_gold"]),
            "per_label": per_label,
        }
    # Exact match is the main metric
    result["macro_f1"] = result["exact"]["macro_f1"]
    result["per_label"] = result["exact"]["per_label"]
    return result


# ---------------------------------------------------------------------------
# 1. Dictionary baseline
# ---------------------------------------------------------------------------

def phrase_statistics(train_docs: list[Doc]) -> dict:
    """For each phrase annotated in train: its label counts and, over all its
    occurrences in the training text, how many are annotated with each label."""
    nlp = spacy.blank("en")
    label_counts = defaultdict(Counter)  # phrase -> Counter(label -> times annotated)
    for doc in train_docs:
        for ent in doc.ents:
            label_counts[ent.text.lower()][ent.label_] += 1

    # Find every occurrence of every phrase (case-insensitive), annotated or not
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for phrase in label_counts:
        matcher.add(phrase, [nlp.make_doc(phrase)])
    occurrences = Counter()
    for doc in train_docs:
        for match_id, start, end in matcher(doc):
            occurrences[nlp.vocab.strings[match_id]] += 1

    return {
        phrase: {"labels": counts, "occurrences": max(occurrences[phrase], sum(counts.values()))}
        for phrase, counts in label_counts.items()
    }


def build_dictionary(stats: dict, min_count: int, min_precision: float) -> list[dict]:
    """EntityRuler patterns for the phrases that pass both filters.

    A phrase gets its most frequent label. Its precision = how often it is
    annotated with that label / how often it appears at all. This drops
    ambiguous words like "symptoms" or "treatment" that are often not entities.
    """
    patterns = []
    for phrase, s in stats.items():
        label, count = s["labels"].most_common(1)[0]
        if count >= min_count and count / s["occurrences"] >= min_precision:
            patterns.append({"label": label, "pattern": phrase})
    return patterns


def make_dictionary_nlp(patterns: list[dict]):
    """A blank English pipeline with only an EntityRuler (case-insensitive).
    When two patterns overlap, the ruler keeps the longest match."""
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    ruler.add_patterns(patterns)
    return nlp


def tune_dictionary(train_docs, val_docs, cfg) -> tuple:
    """Try every (min_count, min_precision) pair; keep the best on val macro-F1."""
    stats = phrase_statistics(train_docs)
    best = None
    for min_count, min_precision in itertools.product(cfg["dictionary"]["min_count"], cfg["dictionary"]["min_precision"]):
        patterns = build_dictionary(stats, min_count, min_precision)
        nlp = make_dictionary_nlp(patterns)
        val_f1 = evaluate_ner(nlp, val_docs, cfg["labels"])["macro_f1"]
        if best is None or val_f1 > best[3]["macro_f1"]:
            best = (nlp, {"min_count": min_count, "min_precision": min_precision},
                    len(patterns), {"macro_f1": val_f1})
    return best


# ---------------------------------------------------------------------------
# 2. Trained spaCy NER
# ---------------------------------------------------------------------------

def new_pipeline(vectors: str | None):
    """A blank English pipeline, optionally with pretrained word vectors.

    Word vectors (e.g. from en_core_web_md) give the model a meaning for words
    it never saw in MACCROBAT: "lasix" is close to "furosemide".
    """
    nlp = spacy.blank("en")
    if vectors:
        nlp.vocab.vectors = spacy.load(vectors).vocab.vectors
    return nlp


def train_ner(train_docs: list[Doc], val_docs: list[Doc], cfg: dict, seed: int) -> tuple:
    """Train spaCy's NER from scratch (optionally on top of word vectors).

    Each epoch: shuffle the training sentences, update the model in small
    batches, then score on val. Keep the weights of the best epoch and stop
    early when val F1 stops improving (early stopping).
    """
    spacy.util.fix_random_seed(seed)
    random.seed(seed)
    t = cfg["training"]

    vectors = cfg.get("vectors")
    nlp = new_pipeline(vectors)
    # With vectors, the token encoder reads them in addition to its own features
    ner = nlp.add_pipe("ner", config={"model": {"tok2vec": {"pretrained_vectors": True}}} if vectors else {})
    for label in cfg["labels"]:
        ner.add_label(label)

    examples = to_examples(nlp, train_docs)
    optimizer = nlp.initialize(lambda: examples)

    best_f1, best_weights, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, t["max_epochs"] + 1):
        random.shuffle(examples)
        losses = {}
        for batch in minibatch(examples, size=t["batch_size"]):
            nlp.update(batch, sgd=optimizer, drop=t["dropout"], losses=losses)

        val_f1 = evaluate_ner(nlp, val_docs, cfg["labels"])["macro_f1"]
        history.append({"epoch": epoch, "loss": round(float(losses["ner"]), 1), "val_macro_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_weights, best_epoch = val_f1, nlp.to_bytes(), epoch
        elif epoch - best_epoch >= t["patience"]:
            break

    nlp.from_bytes(best_weights)  # back to the best epoch
    return nlp, {"best_epoch": best_epoch, "val_macro_f1": best_f1, "history": history}


# ---------------------------------------------------------------------------

def show_examples(models: dict, n_reports: int = 3, n_chars: int = 700) -> None:
    """Print what each model extracts from a few MTSamples test reports
    (no gold labels there: this is a qualitative look at the domain shift)."""
    test = load_mtsamples_splits()["test"]
    picks = [test[test["label"] == label].iloc[0] for label in
             ["Orthopedic", "Gastroenterology", "Neurology"][:n_reports]]
    for row in picks:
        text = row["text"][:n_chars]
        print(f"\n=== MTSamples test report ({row['label']}), first {n_chars} characters ===")
        print(text)
        for name, nlp in models.items():
            doc = nlp(text)
            for label in ["SYMPTOM", "TREATMENT"]:
                found = [e.text for e in doc.ents if e.label_ == label]
                print(f"  [{name}] {label}: {found}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dictionary baseline vs trained spaCy NER.")
    parser.add_argument("--config", default="configs/ner.yaml")
    parser.add_argument("--subset", type=int, default=None, help="train on N documents (quick local run)")
    parser.add_argument("--vectors", default=None, help="e.g. en_core_web_md: train on top of pretrained word vectors")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.subset:
        cfg["subset_size"] = args.subset
    if args.vectors:
        cfg["vectors"] = args.vectors
        cfg["name"] += "_vectors"
    is_subset = cfg["subset_size"] is not None

    train_docs = load_maccrobat_split("train")
    if is_subset:
        train_docs = random.Random(0).sample(train_docs, cfg["subset_size"])
    train, val, test = (split_into_sentences(d) for d in [train_docs, load_maccrobat_split("val"), load_maccrobat_split("test")])
    print(f"Sentences: train {len(train)}, val {len(val)}, test {len(test)}")

    # 1. Dictionary baseline (deterministic: one run)
    dict_nlp, dict_params, n_patterns, dict_val = tune_dictionary(train, val, cfg)
    dict_test = evaluate_ner(dict_nlp, test, cfg["labels"])
    print(f"\nDictionary: {n_patterns} phrases, {dict_params}, val macro-F1 = {dict_val['macro_f1']:.3f}")
    for mode in ["exact", "overlap"]:
        m = dict_test[mode]
        print(f"  test {mode:7s}: macro-F1 = {m['macro_f1']:.3f} | micro P/R/F1 = "
              f"{m['micro']['p']:.3f}/{m['micro']['r']:.3f}/{m['micro']['f']:.3f} | per label {m['per_label']}")

    # 2. Trained NER, one run per seed
    runs = []
    for seed in cfg["seeds"]:
        set_seed(seed)
        nlp, info = train_ner(train, val, cfg, seed)
        scores = evaluate_ner(nlp, test, cfg["labels"])
        runs.append((nlp, info, scores))
        print(f"  seed {seed}: best epoch {info['best_epoch']:2d}, val macro-F1 = {info['val_macro_f1']:.3f}, "
              f"test macro-F1 = {scores['macro_f1']:.3f} (SYMPTOM {scores['per_label']['SYMPTOM']['f']:.3f}, "
              f"TREATMENT {scores['per_label']['TREATMENT']['f']:.3f}) | overlap macro-F1 = {scores['overlap']['macro_f1']:.3f}")

    test_f1 = np.array([s["macro_f1"] for _, _, s in runs])
    overlap_f1 = np.array([s["overlap"]["macro_f1"] for _, _, s in runs])
    # The model kept for the app = the seed with the best *val* score (test is not used to choose)
    best_nlp, best_info, best_scores = max(runs, key=lambda r: r[1]["val_macro_f1"])
    metrics = {
        "sentences": {"train": len(train), "val": len(val), "test": len(test)},
        "dictionary": {"params": dict_params, "n_patterns": n_patterns, "val_macro_f1": dict_val["macro_f1"], "test": dict_test},
        "spacy_ner": {
            "seeds": [{"seed": seed, "best_epoch": i["best_epoch"], "val_macro_f1": i["val_macro_f1"], "test": s, "history": i["history"]}
                      for seed, (_, i, s) in zip(cfg["seeds"], runs)],
            "test_macro_f1_mean": round(float(test_f1.mean()), 4),
            "test_macro_f1_std": round(float(test_f1.std(ddof=1)), 4),
            "test_overlap_macro_f1_mean": round(float(overlap_f1.mean()), 4),
            "test_overlap_macro_f1_std": round(float(overlap_f1.std(ddof=1)), 4),
            "kept_model_test": best_scores,
        },
    }
    s = metrics["spacy_ner"]
    print(f"\nTrained NER over {len(runs)} seeds: test macro-F1 exact = {s['test_macro_f1_mean']:.3f} ± {s['test_macro_f1_std']:.3f}, "
          f"overlap = {s['test_overlap_macro_f1_mean']:.3f} ± {s['test_overlap_macro_f1_std']:.3f}")

    run_dir = save_run(cfg["name"] + ("_subset" if is_subset else ""), cfg, metrics)
    print(f"Saved run to {run_dir.relative_to(ROOT)}")
    if not is_subset:  # a quick subset run must never replace the real models
        model_dir = ROOT / cfg["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)  # spaCy only creates the last folder level
        best_nlp.to_disk(model_dir / ("spacy_ner_vectors" if cfg.get("vectors") else "spacy_ner"))
        dict_nlp.to_disk(model_dir / "dictionary")
        print(f"Saved models to {model_dir.relative_to(ROOT)}")

    show_examples({"dictionary": dict_nlp, "spacy NER": best_nlp})


if __name__ == "__main__":
    main()
