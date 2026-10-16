# AI-GENERATED SECTION START
"""Create simple evaluation plots after classifier training."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    metrics_path = Path("artifacts/metrics.json")
    if not metrics_path.exists():
        raise SystemExit("Run python -m src.classifier.train first.")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    labels = data["class_labels"]
    cm = np.array(data["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Intent Classifier Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig("artifacts/confusion_matrix.png", dpi=160)
    plt.close(fig)

    report = data["classification_report"]
    f1 = [report[label]["f1-score"] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, f1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-intent F1")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig("artifacts/f1_by_intent.png", dpi=160)
    plt.close(fig)

    print("Created artifacts/confusion_matrix.png and artifacts/f1_by_intent.png")


if __name__ == "__main__":
    main()
# AI-GENERATED SECTION END
