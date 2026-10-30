"""Metrics shared by every classifier, so all models are scored the same way.

(Extended in the evaluation phase with comparisons and plots.)
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
