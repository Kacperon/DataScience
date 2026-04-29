# nick_moderator

Vulgar nickname detection — 5 approaches for comparison, all production-ready (saved to disk, loadable on demand).

## Approaches

| Module | Approach | Type |
|---|---|---|
| `baselines.WordListBaseline` | Substring match against PL banlist | rule-based |
| `baselines.HammingBaseline` | Sliding-window Hamming distance ≤ 1 | rule-based |
| `baselines.LogRegBaseline` | Char n-gram (2-5) + Logistic Regression | classic ML |
| `models.MLPClassifier` | Char n-gram (2-5) + 2-layer MLP | NN |
| `models.CharCNN` | Char embeddings + parallel Conv1D + 2-layer FC | NN |

## Usage

### Train all models (saves to `artifacts/`)
```bash
python -m nick_moderator.train
```

This produces:
- `train.parquet`, `val.parquet`, `test.parquet` — splits
- `wordlist.txt` — banlist
- `hamming.json` — Hamming baseline metadata
- `logreg.joblib` — sklearn pipeline
- `mlp.pt` + `.vectorizer.joblib` + `.meta.json` — PyTorch MLP
- `cnn.pt` + `.vocab.json` + `.meta.json` — PyTorch CharCNN

### Inference
```python
from nick_moderator import NickModerator

mod = NickModerator()  # loads all models from artifacts/
proba = mod.predict_all_proba(["xXkurw4Xx", "marek_92"])
# {'wordlist': ..., 'hamming': ..., 'logreg': ..., 'mlp': ..., 'cnn': ...}
```

## Data

- **Namespotting** — 4.5M Reddit usernames with toxic/non-toxic labels
- **PL augmented** — synthetic vulgar nicks generated from PL banlist (Steam PL filter, BAN-PL, coldner/wulgaryzmy, LDNOOBW PL) via leet/separator/affix transformations
- **PL clean** — synthetic clean nicks from Polish first names + gamer words

See [data/README.md](../data/README.md) for source attribution.

## Comparison

See [wyniki_modeli.ipynb](../wyniki_modeli.ipynb) for the full evaluation report.
