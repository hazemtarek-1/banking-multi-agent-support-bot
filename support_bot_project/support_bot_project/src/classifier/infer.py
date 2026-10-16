# AI-GENERATED SECTION START
"""Inference helper for the trained intent classifier."""
from __future__ import annotations

from pathlib import Path

import torch

from .data import tokenize
from .model import IntentClassifier


class IntentPredictor:
    def __init__(self, checkpoint_path: str = "artifacts/classifier.pt"):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {path}. Run training first.")
        self.device = torch.device("cpu")
        self.ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.vocab = self.ckpt["vocab"]
        self.id2label = {int(k): v for k, v in self.ckpt["id2label"].items()}
        self.model = IntentClassifier(vocab_size=len(self.vocab), **self.ckpt["model_config"])
        self.model.load_state_dict(self.ckpt["model_state"])
        self.model.eval()

    def predict(self, text: str) -> tuple[str, float]:
        tokens = tokenize(text) or ["<unk>"]
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        x = torch.tensor([ids], dtype=torch.long)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(self.model(x, lengths), dim=1)[0]
        idx = int(torch.argmax(probs).item())
        return self.id2label[idx], float(probs[idx].item())
# AI-GENERATED SECTION END
