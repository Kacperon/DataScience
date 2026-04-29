"""Rule-based and classic ML baselines for nickname vulgarity detection."""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from .config import (
    HAMMING_MAX_DIST,
    HAMMING_MIN_WORD_LEN,
    NGRAM_RANGE,
    N_HASH_FEATURES,
)


# ─────────────────────────────────────────────────────────────────────────
# 1. Word-list exact / substring match
# ─────────────────────────────────────────────────────────────────────────
class WordListBaseline:
    """Simple substring-match baseline against a banlist."""

    def __init__(self, banlist: list[str], min_word_len: int = 3):
        self.banlist = sorted({w.lower() for w in banlist if len(w) >= min_word_len})
        # Compile a single regex for fast scanning
        escaped = [re.escape(w) for w in self.banlist]
        self._regex = re.compile("|".join(escaped)) if escaped else None

    def predict_one(self, nick: str) -> int:
        if self._regex is None:
            return 0
        return int(bool(self._regex.search(nick.lower())))

    def predict(self, nicks: list[str]) -> np.ndarray:
        return np.array([self.predict_one(n) for n in nicks], dtype=np.int8)

    def save(self, path: Path) -> None:
        Path(path).write_text("\n".join(self.banlist), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, min_word_len: int = 3) -> "WordListBaseline":
        banlist = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        return cls(banlist, min_word_len=min_word_len)


# ─────────────────────────────────────────────────────────────────────────
# 2. Hamming distance (sliding window)
# ─────────────────────────────────────────────────────────────────────────
def _hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


class HammingBaseline:
    """For each ban word, slide it over the nick and check Hamming distance.

    Pros: catches typos/single substitutions like 'kurva' for 'kurwa'.
    Cons: misses insertions/deletions ('k_urwa') — Levenshtein would handle those
          but is slower.
    """

    def __init__(self, banlist: list[str], max_dist: int = HAMMING_MAX_DIST,
                 min_word_len: int = HAMMING_MIN_WORD_LEN):
        self.max_dist = max_dist
        self.min_word_len = min_word_len
        self.banlist = sorted({w.lower() for w in banlist if len(w) >= min_word_len})

    def predict_one(self, nick: str) -> int:
        nick = nick.lower()
        for word in self.banlist:
            wlen = len(word)
            if wlen > len(nick):
                continue
            for i in range(len(nick) - wlen + 1):
                if _hamming(nick[i : i + wlen], word) <= self.max_dist:
                    return 1
        return 0

    def predict(self, nicks: list[str]) -> np.ndarray:
        return np.array([self.predict_one(n) for n in nicks], dtype=np.int8)

    def save(self, path: Path) -> None:
        meta = {"max_dist": self.max_dist, "min_word_len": self.min_word_len, "banlist": self.banlist}
        Path(path).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HammingBaseline":
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(meta["banlist"], max_dist=meta["max_dist"], min_word_len=meta["min_word_len"])


# ─────────────────────────────────────────────────────────────────────────
# 3. Char n-gram + Logistic Regression
# ─────────────────────────────────────────────────────────────────────────
def make_vectorizer() -> HashingVectorizer:
    """Char-level n-gram hashing vectorizer (deterministic, no fit needed)."""
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=NGRAM_RANGE,
        n_features=N_HASH_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )


class LogRegBaseline:
    def __init__(self):
        self.vectorizer = make_vectorizer()
        self.model: LogisticRegression | None = None

    def fit(self, nicks: list[str], labels: np.ndarray) -> None:
        X = self.vectorizer.transform(nicks)
        self.model = LogisticRegression(max_iter=200, n_jobs=-1, C=1.0)
        self.model.fit(X, labels)

    def predict_proba(self, nicks: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(nicks)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, nicks: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(nicks) >= threshold).astype(np.int8)

    def save(self, path: Path) -> None:
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, path)

    @classmethod
    def load(cls, path: Path) -> "LogRegBaseline":
        data = joblib.load(path)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.model = data["model"]
        return obj
