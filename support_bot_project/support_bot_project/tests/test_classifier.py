# AI-GENERATED SECTION START
import torch

from src.classifier.data import build_vocab, tokenize
from src.classifier.model import IntentClassifier


def test_tokenizer():
    assert tokenize("Payment FAILED! Please help.") == ["payment", "failed", "please", "help"]


def test_vocab_and_model_shape():
    examples = [("payment failed", "payment_issue"), ("place an order", "place_order")]
    vocab = build_vocab(examples)
    model = IntentClassifier(vocab_size=len(vocab), num_classes=2)
    x = torch.tensor([[vocab["payment"], vocab["failed"]]])
    lengths = torch.tensor([2])
    logits = model(x, lengths)
    assert logits.shape == (1, 2)
# AI-GENERATED SECTION END
