"""Train all models and save artifacts to disk."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .baselines import HammingBaseline, LogRegBaseline, WordListBaseline, make_vectorizer
from .config import (
    ARTIFACTS_DIR, BATCH_SIZE, DROPOUT, EPOCHS, HIDDEN_DIM,
    LR, NICK_MAX_LEN, N_HASH_FEATURES, SEED,
)
from .data import build_dataset, load_pl_banlist
from .models import (
    CharCNN, MLPClassifier, build_char_vocab, encode_batch,
    save_cnn, save_mlp,
)


# ── Sparse-aware MLP dataset ──────────────────────────────────────────────
class SparseFeatDataset(Dataset):
    """Custom batching: keep features as csr_matrix, convert a whole batch
    at once (much faster than per-row .toarray() in __getitem__)."""

    def __init__(self, X: csr_matrix, y: np.ndarray, batch_size: int):
        self.X = X.tocsr()
        self.y = y.astype(np.float32)
        self.batch_size = batch_size
        self.n = X.shape[0]

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_batch(self, idx: int, perm: np.ndarray | None = None):
        i0 = idx * self.batch_size
        i1 = min(i0 + self.batch_size, self.n)
        rows = perm[i0:i1] if perm is not None else slice(i0, i1)
        Xb = self.X[rows].toarray().astype(np.float32, copy=False)
        yb = self.y[rows]
        return torch.from_numpy(Xb), torch.from_numpy(yb)

    def iter_epoch(self, shuffle: bool, rng: np.random.Generator):
        perm = rng.permutation(self.n) if shuffle else None
        for idx in range(len(self)):
            yield self.get_batch(idx, perm)


class CharIdsDataset(Dataset):
    def __init__(self, ids: torch.Tensor, y: np.ndarray):
        self.ids = ids
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, i: int):
        return self.ids[i], self.y[i]


def _seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_torch(model: nn.Module, train_iter_fn, val_iter_fn, n_train_batches: int, n_val_batches: int,
                 device: torch.device, epochs: int = EPOCHS, lr: float = LR) -> dict:
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history: list[dict] = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0
        pbar = tqdm(train_iter_fn(), total=n_train_batches,
                    desc=f"Epoch {epoch}/{epochs} train", leave=True, mininterval=1.0)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * yb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += yb.size(0)
            pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.4f}")

        train_loss = loss_sum / total
        train_acc = correct / total

        model.eval()
        v_total, v_correct, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_iter_fn(), total=n_val_batches,
                               desc=f"Epoch {epoch}/{epochs} val  ", leave=True, mininterval=1.0):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss_sum += loss_fn(logits, yb).item() * yb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                v_correct += (preds == yb.long()).sum().item()
                v_total += yb.size(0)
        val_loss = v_loss_sum / v_total
        val_acc = v_correct / v_total

        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "time_s": dt})
        print(f"  Epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({dt:.1f}s)")

        best_val = min(best_val, val_loss)

    return {"history": history, "best_val_loss": best_val}


def main() -> None:
    _seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Build dataset
    train_df, val_df, test_df, info = build_dataset()
    train_df.to_parquet(ARTIFACTS_DIR / "train.parquet")
    val_df.to_parquet(ARTIFACTS_DIR / "val.parquet")
    test_df.to_parquet(ARTIFACTS_DIR / "test.parquet")
    (ARTIFACTS_DIR / "dataset_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"Dataset saved to {ARTIFACTS_DIR}")

    train_nicks = train_df["nick"].tolist()
    val_nicks = val_df["nick"].tolist()
    train_y = train_df["label"].to_numpy()
    val_y = val_df["label"].to_numpy()

    # 2. Word-list baseline (no training, just save banlist)
    print("\n[1/5] Word-list baseline")
    banlist = load_pl_banlist()
    wl = WordListBaseline(banlist)
    wl.save(ARTIFACTS_DIR / "wordlist.txt")
    print(f"  Saved banlist of {len(banlist):,} words")

    # 3. Hamming baseline
    print("\n[2/5] Hamming baseline")
    ham = HammingBaseline(banlist)
    ham.save(ARTIFACTS_DIR / "hamming.json")
    print(f"  Saved Hamming baseline")

    # 4. LogReg
    print("\n[3/5] LogReg (char n-gram)")
    t0 = time.time()
    lr = LogRegBaseline()
    lr.fit(train_nicks, train_y)
    lr.save(ARTIFACTS_DIR / "logreg.joblib")
    val_pred = lr.predict(val_nicks)
    print(f"  val_acc={(val_pred == val_y).mean():.4f}  ({time.time()-t0:.1f}s)")

    # 5. MLP
    print("\n[4/5] MLP (char n-gram + 2-layer)", flush=True)
    vec = make_vectorizer()
    X_train = vec.transform(train_nicks)
    X_val = vec.transform(val_nicks)
    train_ds_mlp = SparseFeatDataset(X_train, train_y, BATCH_SIZE)
    val_ds_mlp = SparseFeatDataset(X_val, val_y, BATCH_SIZE)
    rng = np.random.default_rng(SEED)

    mlp = MLPClassifier(input_dim=N_HASH_FEATURES, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    mlp_meta = {"input_dim": N_HASH_FEATURES, "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
                "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE}
    mlp_log = _train_torch(
        mlp,
        train_iter_fn=lambda: train_ds_mlp.iter_epoch(shuffle=True, rng=rng),
        val_iter_fn=lambda: val_ds_mlp.iter_epoch(shuffle=False, rng=rng),
        n_train_batches=len(train_ds_mlp),
        n_val_batches=len(val_ds_mlp),
        device=device, epochs=EPOCHS, lr=LR,
    )
    mlp_meta["history"] = mlp_log["history"]
    save_mlp(mlp, vec, ARTIFACTS_DIR / "mlp.pt", meta=mlp_meta)
    print(f"  Saved MLP -> {ARTIFACTS_DIR / 'mlp.pt'}", flush=True)

    # 6. Char-CNN — uses simple in-memory dataset (char ids, fast)
    print("\n[5/5] Char-CNN", flush=True)
    vocab = build_char_vocab()
    train_ids = encode_batch(train_nicks, vocab, NICK_MAX_LEN)
    val_ids = encode_batch(val_nicks, vocab, NICK_MAX_LEN)

    class _IdsBatcher:
        def __init__(self, ids: torch.Tensor, y: np.ndarray, bs: int):
            self.ids = ids
            self.y = torch.from_numpy(y.astype(np.float32))
            self.bs = bs
            self.n = ids.shape[0]
        def __len__(self): return (self.n + self.bs - 1) // self.bs
        def iter_epoch(self, shuffle: bool, rng: np.random.Generator):
            perm = torch.from_numpy(rng.permutation(self.n)) if shuffle else None
            ids = self.ids[perm] if perm is not None else self.ids
            y = self.y[perm] if perm is not None else self.y
            for i in range(0, self.n, self.bs):
                yield ids[i:i + self.bs], y[i:i + self.bs]

    train_b = _IdsBatcher(train_ids, train_y, BATCH_SIZE)
    val_b = _IdsBatcher(val_ids, val_y, BATCH_SIZE)

    cnn = CharCNN(vocab_size=len(vocab)).to(device)
    cnn_meta = {"vocab_size": len(vocab), "embed_dim": 32, "num_filters": 64,
                "kernel_sizes": [3, 4, 5], "hidden_dim": 128, "dropout": DROPOUT,
                "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE}
    cnn_log = _train_torch(
        cnn,
        train_iter_fn=lambda: train_b.iter_epoch(shuffle=True, rng=rng),
        val_iter_fn=lambda: val_b.iter_epoch(shuffle=False, rng=rng),
        n_train_batches=len(train_b),
        n_val_batches=len(val_b),
        device=device, epochs=EPOCHS, lr=LR,
    )
    cnn_meta["history"] = cnn_log["history"]
    save_cnn(cnn, vocab, ARTIFACTS_DIR / "cnn.pt", meta=cnn_meta)
    print(f"  Saved CharCNN -> {ARTIFACTS_DIR / 'cnn.pt'}", flush=True)

    print("\nAll models saved to:", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
