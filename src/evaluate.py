"""Metrics shared by every classifier, so all models are scored the same way.

(Extended in the evaluation phase with comparisons and plots.)
"""

import math

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    """Accuracy, macro precision/recall/F1, per-class scores, confusion matrix.

    Macro = the average over classes, each class counting equally. With
    imbalanced classes this is fairer than accuracy, which is dominated by
    the largest class.
    """
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        label: {
            "precision": round(report[label]["precision"], 4),
            "recall": round(report[label]["recall"], 4),
            "f1": round(report[label]["f1-score"], 4),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4),
        "macro_f1": round(float(f1), 4),
        "per_class": per_class,
        # rows = true label, columns = predicted label, in the order of "labels"
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }


def subset_macro_f1(y_true, y_pred) -> float:
    """Macro-F1 over the classes present in y_true only.

    On a subset of the test set some classes can be missing (e.g. there is no
    operative Neurology report). Plain macro-F1 would count such a class as
    F1 = 0 and pull the average down for no reason.
    """
    present = sorted(set(y_true))
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def mcnemar_p(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test: are models A and B really different on this test set?

    Only the reports where exactly one model is right count:
      a_only = A right, B wrong;  b_only = B right, A wrong.
    If both models were equally good, each such report would be a coin flip,
    so the p-value is a two-sided binomial test with probability 1/2.
    """
    a_ok = [p == y for p, y in zip(pred_a, y_true)]
    b_ok = [p == y for p, y in zip(pred_b, y_true)]
    a_only = sum(a and not b for a, b in zip(a_ok, b_ok))
    b_only = sum(b and not a for a, b in zip(a_ok, b_ok))
    n, k = a_only + b_only, min(a_only, b_only)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"a_only_right": a_only, "b_only_right": b_only, "p_value": round(p, 4)}
