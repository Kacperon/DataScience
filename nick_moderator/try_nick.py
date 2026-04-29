"""Production model test — reads nicknames from stdin, predicts with MLP (best model).

Usage:
    python -m nick_moderator.try_nick           # type nicks, Enter, Ctrl-D to quit
    python -m nick_moderator.try_nick -t 0.7    # custom threshold
    echo "xXkurw4Xx" | python -m nick_moderator.try_nick
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from .config import ARTIFACTS_DIR
from .models import load_mlp


def main() -> int:
    parser = argparse.ArgumentParser(description="Read nicknames from stdin and classify with the production MLP model.")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5).")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    args = parser.parse_args()

    print("Loading MLP from artifacts/...", file=sys.stderr)
    device = torch.device(args.device)
    model, vectorizer, _ = load_mlp(ARTIFACTS_DIR / "mlp.pt")
    model.to(device).eval()
    print("Ready. Type a nick and press Enter (Ctrl-D to quit).\n", file=sys.stderr)

    while True:
        try:
            nick = input().strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if not nick:
            continue
        with torch.no_grad():
            X = vectorizer.transform([nick]).toarray().astype(np.float32)
            logits = model(torch.from_numpy(X).to(device))
            score = float(torch.sigmoid(logits).cpu().numpy()[0])
        flag = "VULGAR" if score >= args.threshold else "ok"
        print(f"{nick!r}  {score:.3f}  {flag}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
