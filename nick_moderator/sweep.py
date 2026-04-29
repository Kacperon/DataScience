"""Hyperparameter sweep for MLP — train multiple configs and save results.

Loads pre-built train/val/test splits from artifacts/ (created by `train.py`).
Saves each model variant to artifacts/sweep/ + a results.json comparison file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from .baselines import make_vectorizer
from .config import ARTIFACTS_DIR, BATCH_SIZE, LR, N_HASH_FEATURES, NICK_MAX_LEN, SEED
from .models import MLPClassifier, save_mlp


SWEEP_DIR = ARTIFACTS_DIR / "sweep"
SWEEP_DIR.mkdir(exist_ok=True, parents=True)


def _seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_mlp(model: nn.Module, X_train, y_train, X_val, y_val,
               device: torch.device, epochs: int, batch_size: int = BATCH_SIZE) -> list[dict]:
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    n = X_train.shape[0]
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        total, correct, loss_sum = 0, 0, 0.0
        t0 = time.time()
        n_batches = (n + batch_size - 1) // batch_size
        for b in tqdm(range(n_batches), desc=f"  ep{epoch} train", leave=False, mininterval=2.0):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = torch.from_numpy(X_train[idx].toarray().astype(np.float32)).to(device)
            yb = torch.from_numpy(y_train[idx].astype(np.float32)).to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * yb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += yb.size(0)

        model.eval()
        v_total, v_correct, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            n_val = X_val.shape[0]
            for b in range((n_val + batch_size - 1) // batch_size):
                xb = torch.from_numpy(
                    X_val[b * batch_size : (b + 1) * batch_size].toarray().astype(np.float32)
                ).to(device)
                yb = torch.from_numpy(
                    y_val[b * batch_size : (b + 1) * batch_size].astype(np.float32)
                ).to(device)
                logits = model(xb)
                v_loss_sum += loss_fn(logits, yb).item() * yb.size(0)
                v_correct += ((torch.sigmoid(logits) >= 0.5).long() == yb.long()).sum().item()
                v_total += yb.size(0)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total,
                        "val_loss": v_loss_sum / v_total, "val_acc": v_correct / v_total, "time_s": dt})
        print(f"  ep{epoch}/{epochs} train_acc={correct/total:.4f}  val_acc={v_correct/v_total:.4f}  ({dt:.1f}s)", flush=True)
    return history


@torch.no_grad()
def _eval_set(model: MLPClassifier, vec, nicks: list[str], y: np.ndarray, device: torch.device) -> dict:
    model.eval()
    X = vec.transform(nicks).toarray().astype(np.float32)
    logits = model(torch.from_numpy(X).to(device))
    proba = torch.sigmoid(logits).cpu().numpy()
    pred = (proba >= 0.5).astype(np.int8)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
    }


def main(epochs: int = 3, eval_sample: int = 5000, skip_existing: bool = True) -> None:
    _seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load splits
    train_df = pd.read_parquet(ARTIFACTS_DIR / "train.parquet")
    val_df = pd.read_parquet(ARTIFACTS_DIR / "val.parquet")
    test_df = pd.read_parquet(ARTIFACTS_DIR / "test.parquet")

    # Load custom held-out set
    custom = json.loads(Path("tests/custom_test_set.json").read_text(encoding="utf-8"))
    custom_df = pd.DataFrame(custom["samples"])

    # Subsample test for speed
    test_eval = test_df.sample(n=min(eval_sample, len(test_df)), random_state=42).reset_index(drop=True)
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test (sampled): {len(test_eval):,}  Custom: {len(custom_df)}", flush=True)

    # Featurize once
    vec = make_vectorizer()
    print("Featurizing train/val...", flush=True)
    X_train = vec.transform(train_df["nick"].tolist())
    X_val = vec.transform(val_df["nick"].tolist())
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    print(f"X_train shape: {X_train.shape}", flush=True)

    # Sweep configurations
    configs = [
        # ─── Round 1: width / depth / dropout sweep ───
        {"name": "h64",       "hidden_dim": 64,           "dropout": 0.3, "epochs": epochs},
        {"name": "h128",      "hidden_dim": 128,          "dropout": 0.3, "epochs": epochs},
        {"name": "h256",      "hidden_dim": 256,          "dropout": 0.3, "epochs": epochs},
        {"name": "h512",      "hidden_dim": 512,          "dropout": 0.3, "epochs": epochs},
        {"name": "h256_128",  "hidden_dim": [256, 128],   "dropout": 0.3, "epochs": epochs},
        {"name": "h512_128",  "hidden_dim": [512, 128],   "dropout": 0.3, "epochs": epochs},
        {"name": "h256_d01",  "hidden_dim": 256,          "dropout": 0.1, "epochs": epochs},
        {"name": "h256_d05",  "hidden_dim": 256,          "dropout": 0.5, "epochs": epochs},
        # ─── Round 2: 4-layer + epoch ablation ───
        {"name": "h512_256_128",  "hidden_dim": [512, 256, 128],  "dropout": 0.3, "epochs": epochs},
        {"name": "h1024_256_128", "hidden_dim": [1024, 256, 128], "dropout": 0.3, "epochs": epochs},
        {"name": "h512_128_64",   "hidden_dim": [512, 128, 64],   "dropout": 0.3, "epochs": epochs},
        {"name": "h256_128_64",   "hidden_dim": [256, 128, 64],   "dropout": 0.3, "epochs": epochs},
        {"name": "h512_512_128",  "hidden_dim": [512, 512, 128],  "dropout": 0.3, "epochs": epochs},
        {"name": "h512_128_e5",   "hidden_dim": [512, 128],       "dropout": 0.3, "epochs": 5},
    ]

    # Load any existing results to preserve already-trained configs
    results_path = SWEEP_DIR / "results.json"
    existing = {}
    if results_path.exists():
        for r in json.loads(results_path.read_text()):
            existing[r["config"]] = r

    results = list(existing.values())

    for cfg in configs:
        cfg_epochs = cfg.pop("epochs", epochs)
        model_path = SWEEP_DIR / f"mlp_{cfg['name']}.pt"
        if skip_existing and model_path.exists() and cfg["name"] in existing:
            print(f"\n=== {cfg['name']} — already trained, skipping ===", flush=True)
            continue

        print(f"\n=== Training {cfg['name']}  hidden={cfg['hidden_dim']}  dropout={cfg['dropout']}  epochs={cfg_epochs} ===", flush=True)
        model = MLPClassifier(input_dim=N_HASH_FEATURES, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        history = _train_mlp(model, X_train, y_train, X_val, y_val, device, epochs=cfg_epochs)
        train_time = time.time() - t0

        # Eval
        test_metrics = _eval_set(model, vec, test_eval["nick"].tolist(), test_eval["label"].to_numpy(), device)
        custom_metrics = _eval_set(model, vec, custom_df["nick"].tolist(), custom_df["label"].to_numpy(), device)

        # Save model
        model_path = SWEEP_DIR / f"mlp_{cfg['name']}.pt"
        meta = {**cfg, "input_dim": N_HASH_FEATURES, "epochs": epochs,
                "n_params": n_params, "train_time_s": train_time, "history": history}
        save_mlp(model, vec, model_path, meta=meta)
        print(f"  Saved -> {model_path}", flush=True)

        new_row = {
            "config": cfg["name"],
            "hidden_dim": cfg["hidden_dim"],
            "dropout": cfg["dropout"],
            "n_params": n_params,
            "params_M": round(n_params / 1e6, 2),
            "train_time_s": round(train_time, 1),
            "epochs": cfg_epochs,
            "test_f1": test_metrics["f1"],
            "test_acc": test_metrics["accuracy"],
            "test_auc": test_metrics["roc_auc"],
            "custom_f1": custom_metrics["f1"],
            "custom_acc": custom_metrics["accuracy"],
        }
        # Replace existing row if any (when retraining), else append
        results = [r for r in results if r["config"] != cfg["name"]]
        results.append(new_row)
        # Persist after every config — robust to interruptions
        results_path.write_text(json.dumps(results, indent=2))
        print(f"  test_f1={test_metrics['f1']:.4f}  custom_f1={custom_metrics['f1']:.4f}  params={n_params/1e6:.2f}M  time={train_time:.0f}s", flush=True)

    results_df = pd.DataFrame(results)
    print(f"\nResults saved to {results_path}", flush=True)
    print("\n" + results_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
