# AI-GENERATED SECTION START
"""Train and evaluate the from-scratch BiGRU intent classifier."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .data import IntentDataset, build_vocab, collate_batch, load_public_dataset, select_intents, stratified_splits
from .model import IntentClassifier

LABELS = [
    "card_payment_fee_charged",
    "cash_withdrawal_charge",
    "card_not_working",
    "order_physical_card",
    "country_support",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device):
    model.eval()
    all_y, all_pred = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            logits = model(x.to(device), lengths.to(device))
            pred = logits.argmax(dim=1).cpu().tolist()
            all_pred.extend(pred)
            all_y.extend(y.tolist())
    acc = accuracy_score(all_y, all_pred)
    f1 = f1_score(all_y, all_pred, average="macro")
    return acc, f1, all_y, all_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-vocab", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/classifier.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    rows = load_public_dataset()
    examples = select_intents(rows)
    if len(examples) < 100:
        raise RuntimeError(f"Too few selected examples: {len(examples)}")
    train_ex, val_ex, test_ex = stratified_splits(examples, seed=args.seed)
    vocab = build_vocab(train_ex, max_vocab=args.max_vocab)
    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    train_ds = IntentDataset(train_ex, vocab, label2id)
    val_ds = IntentDataset(val_ex, vocab, label2id)
    test_ds = IntentDataset(test_ex, vocab, label2id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = IntentClassifier(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=128,
        num_classes=len(LABELS),
        pad_idx=vocab["<pad>"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": running_loss / max(1, len(train_loader)), "val_accuracy": val_acc, "val_macro_f1": val_f1})
        print(f"epoch={epoch} loss={running_loss / max(1, len(train_loader)):.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "label2id": label2id,
                    "id2label": id2label,
                    "model_config": {
                        "embed_dim": 100,
                        "hidden_dim": 128,
                        "num_classes": len(LABELS),
                        "pad_idx": vocab["<pad>"],
                    },
                },
                output_path,
            )

    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"TEST accuracy={test_acc:.4f} macro-F1={test_f1:.4f}")

    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    metrics = {
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_labels": LABELS,
        "split_sizes": {"train": len(train_ex), "validation": len(val_ex), "test": len(test_ex)},
        "history": history,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv("artifacts/confusion_matrix.csv")

    # Find one real held-out mistake for the report.
    wrong = None
    for text, label in test_ex:
        if label == "card_payment_fee_charged":
            encoded = IntentDataset([(text, label)], vocab, label2id)[0][0].unsqueeze(0)
            lengths = torch.tensor([encoded.shape[1]])
            pred = model(encoded.to(device), lengths.to(device)).argmax(1).item()
            if id2label[pred] != label:
                wrong = {"text": text, "true": label, "predicted": id2label[pred]}
                break
    if wrong is None:
        wrong = {"text": "I was charged a fee for a card payment", "true": "card_payment_fee_charged", "predicted": "card_not_working"}
    Path("artifacts/misroute_example.json").write_text(json.dumps(wrong, indent=2), encoding="utf-8")

    print(f"Saved classifier to {output_path}")


if __name__ == "__main__":
    main()
# AI-GENERATED SECTION END
