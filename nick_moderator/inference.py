"""Unified inference module — load all trained models from disk and predict."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .baselines import HammingBaseline, LogRegBaseline, WordListBaseline
from .config import ARTIFACTS_DIR, NICK_MAX_LEN
from .models import encode_batch, load_cnn, load_mlp


class NickModerator:
    """Production-ready predictor — loads all approaches once and exposes predict()."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR, device: str = "cpu"):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = torch.device(device)

        # Rule-based
        self.wordlist = WordListBaseline.load(self.artifacts_dir / "wordlist.txt")
        self.hamming = HammingBaseline.load(self.artifacts_dir / "hamming.json")

        # Classic ML
        self.logreg = LogRegBaseline.load(self.artifacts_dir / "logreg.joblib")

        # PyTorch models
        self.mlp, self.mlp_vec, _ = load_mlp(self.artifacts_dir / "mlp.pt")
        self.mlp.to(self.device).eval()

        self.cnn, self.cnn_vocab, _ = load_cnn(self.artifacts_dir / "cnn.pt")
        self.cnn.to(self.device).eval()

    @torch.no_grad()
    def predict_mlp_proba(self, nicks: list[str]) -> np.ndarray:
        X = self.mlp_vec.transform(nicks).toarray().astype(np.float32)
        x = torch.from_numpy(X).to(self.device)
        logits = self.mlp(x)
        return torch.sigmoid(logits).cpu().numpy()

    @torch.no_grad()
    def predict_cnn_proba(self, nicks: list[str]) -> np.ndarray:
        ids = encode_batch(nicks, self.cnn_vocab, NICK_MAX_LEN).to(self.device)
        logits = self.cnn(ids)
        return torch.sigmoid(logits).cpu().numpy()

    def predict_all(self, nicks: list[str], threshold: float = 0.5) -> dict[str, np.ndarray]:
        """Returns dict of model_name -> binary predictions."""
        return {
            "wordlist": self.wordlist.predict(nicks),
            "hamming": self.hamming.predict(nicks),
            "logreg": self.logreg.predict(nicks, threshold=threshold),
            "mlp": (self.predict_mlp_proba(nicks) >= threshold).astype(np.int8),
            "cnn": (self.predict_cnn_proba(nicks) >= threshold).astype(np.int8),
        }

    def predict_all_proba(self, nicks: list[str]) -> dict[str, np.ndarray]:
        """Returns dict of model_name -> probability scores (or 0/1 for rule-based)."""
        return {
            "wordlist": self.wordlist.predict(nicks).astype(np.float32),
            "hamming": self.hamming.predict(nicks).astype(np.float32),
            "logreg": self.logreg.predict_proba(nicks),
            "mlp": self.predict_mlp_proba(nicks),
            "cnn": self.predict_cnn_proba(nicks),
        }
