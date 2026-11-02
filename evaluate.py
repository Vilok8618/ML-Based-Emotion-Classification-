import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)


def evaluate(model, X, y, out_dir: Path, split: str = "test") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = model.predict(X)
    labels = sorted(set(y) | set(pred))
    metrics = {
        "split": split,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "report": classification_report(y, pred, labels=labels, output_dict=True,
                                        zero_division=0),
    }
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, labels=labels, ax=ax, cmap="Blues", xticks_rotation=45,
        normalize="true", values_format=".2f")
    ax.set_title(f"Confusion matrix ({split}, row-normalised)")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{split}.png", dpi=150)
    plt.close(fig)
    return metrics
