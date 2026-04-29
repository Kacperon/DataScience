# DataScience — Vulgar nickname moderator

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn torch joblib tqdm pyarrow scipy
python -m nick_moderator.try_nick
```

Trained models are committed to [artifacts/](artifacts/) — no training required. After launching, type a nick and press Enter — the tool prints the result and waits for the next one (Ctrl-D to quit). Options: `-t 0.7` (threshold), `--device cuda`.
