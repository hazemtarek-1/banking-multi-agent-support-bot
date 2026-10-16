# AI-GENERATED SECTION START
"""Dataset loading, filtering, tokenization, and batching."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# Five real BANKING77 intents. Two are handled by Billing, giving us
# five classifier classes but four specialist agents.
SELECTED_INTENTS = {
    "card_payment_fee_charged": "billing",
    "cash_withdrawal_charge": "billing",
    "card_not_working": "technical",
    "order_physical_card": "booking",
    "country_support": "general",
}

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def load_public_dataset():
    """Load BANKING77 from Hugging Face."""
    from datasets import load_dataset

    ds = load_dataset("mteb/banking77")
    if "train" not in ds:
        raise RuntimeError("Expected a 'train' split in BANKING77.")
    return ds["train"]


def select_intents(rows) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for row in rows:
        intent = str(row["label_text"])
        if intent in SELECTED_INTENTS:
            text = str(row["text"]).strip()
            if text:
                examples.append((text, intent))
    return examples


def stratified_splits(examples, seed: int = 42):
    from sklearn.model_selection import train_test_split

    texts = [x[0] for x in examples]
    labels = [x[1] for x in examples]
    train_x, temp_x, train_y, temp_y = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x,
        temp_y,
        test_size=0.50,
        random_state=seed,
        stratify=temp_y,
    )
    return (
        list(zip(train_x, train_y)),
        list(zip(val_x, val_y)),
        list(zip(test_x, test_y)),
    )


def build_vocab(examples: Iterable[tuple[str, str]], max_vocab: int = 10000) -> dict[str, int]:
    counter = Counter()
    for text, _ in examples:
        counter.update(tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


@dataclass
class EncodedExample:
    ids: list[int]
    label: int


class IntentDataset(Dataset):
    def __init__(self, examples, vocab, label2id):
        self.items = [
            EncodedExample(
                [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokenize(text)] or [vocab[UNK_TOKEN]],
                label2id[label],
            )
            for text, label in examples
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return torch.tensor(item.ids, dtype=torch.long), item.label


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)
# AI-GENERATED SECTION END
