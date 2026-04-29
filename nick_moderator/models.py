"""PyTorch models — 2-layer MLP and char-CNN."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .baselines import make_vectorizer
from .config import DROPOUT, HIDDEN_DIM, NICK_MAX_LEN, N_HASH_FEATURES


# ─────────────────────────────────────────────────────────────────────────
# 1. 2-layer MLP on char n-gram hashed features
# ─────────────────────────────────────────────────────────────────────────
class MLPClassifier(nn.Module):
    """Configurable MLP on hashed char-ngram features.

    hidden_dims=[256] → 2-layer (input→256→1)
    hidden_dims=[256, 128] → 3-layer (input→256→128→1)
    """

    def __init__(self, input_dim: int = N_HASH_FEATURES,
                 hidden_dim: int | list[int] = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        if isinstance(hidden_dim, int):
            dims = [hidden_dim]
        else:
            dims = list(hidden_dim)
        layers = []
        prev = input_dim
        for d in dims:
            layers.append(nn.Linear(prev, d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = d
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        # Keep these for backward-compat with old checkpoints
        self.hidden_dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────
# 2. Char-level CNN (operates directly on character ids)
# ─────────────────────────────────────────────────────────────────────────
PAD_IDX = 0
UNK_IDX = 1
RESERVED = 2  # PAD + UNK


def build_char_vocab(allowed_chars: str | None = None) -> dict[str, int]:
    if allowed_chars is None:
        allowed_chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            "ąćęłńóśźż"
            "_-.!@#$%^&*+=()[]{}|/\\:;,?<>'\" "
        )
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for c in allowed_chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def encode_nick(nick: str, vocab: dict[str, int], max_len: int = NICK_MAX_LEN) -> list[int]:
    nick = nick.lower()[:max_len]
    ids = [vocab.get(c, UNK_IDX) for c in nick]
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids


def encode_batch(nicks: list[str], vocab: dict[str, int], max_len: int = NICK_MAX_LEN) -> torch.Tensor:
    return torch.tensor([encode_nick(n, vocab, max_len) for n in nicks], dtype=torch.long)


class CharCNN(nn.Module):
    """Embedding -> parallel conv filters (k=3,4,5) -> max-pool -> 2-layer FC."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, num_filters: int = 64,
                 kernel_sizes: tuple[int, ...] = (3, 4, 5), hidden_dim: int = 128, dropout: float = DROPOUT):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) long
        e = self.embed(x).transpose(1, 2)  # (B, embed, L)
        feats = []
        for conv in self.convs:
            h = self.act(conv(e))  # (B, F, L)
            h, _ = h.max(dim=2)  # (B, F)
            feats.append(h)
        h = torch.cat(feats, dim=1)
        h = self.drop(self.act(self.fc1(h)))
        return self.fc2(h).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────
def save_mlp(model: MLPClassifier, vectorizer, path: Path, meta: dict | None = None) -> None:
    import joblib
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), path)
    joblib.dump(vectorizer, path.with_suffix(".vectorizer.joblib"))
    if meta:
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def load_mlp(path: Path) -> tuple[MLPClassifier, "HashingVectorizer", dict]:  # type: ignore[name-defined]
    import joblib
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # hidden_dim can be int or list (for multi-layer)
    hidden_dim = meta.get("hidden_dim", HIDDEN_DIM)
    model = MLPClassifier(
        input_dim=meta.get("input_dim", N_HASH_FEATURES),
        hidden_dim=hidden_dim,
        dropout=meta.get("dropout", DROPOUT),
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    vectorizer = joblib.load(path.with_suffix(".vectorizer.joblib"))
    return model, vectorizer, meta


def save_cnn(model: CharCNN, vocab: dict[str, int], path: Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), path)
    path.with_suffix(".vocab.json").write_text(json.dumps(vocab))
    if meta:
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def load_cnn(path: Path) -> tuple[CharCNN, dict[str, int], dict]:
    path = Path(path)
    vocab = json.loads(path.with_suffix(".vocab.json").read_text())
    meta = json.loads(path.with_suffix(".meta.json").read_text()) if path.with_suffix(".meta.json").exists() else {}
    model = CharCNN(
        vocab_size=len(vocab),
        embed_dim=meta.get("embed_dim", 32),
        num_filters=meta.get("num_filters", 64),
        kernel_sizes=tuple(meta.get("kernel_sizes", (3, 4, 5))),
        hidden_dim=meta.get("hidden_dim", 128),
        dropout=meta.get("dropout", DROPOUT),
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model, vocab, meta
