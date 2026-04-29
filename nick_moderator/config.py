from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

NICK_MIN_LEN = 3
NICK_MAX_LEN = 32

# Featurization
NGRAM_RANGE = (2, 5)
N_HASH_FEATURES = 16384

# Model
HIDDEN_DIM = 256
DROPOUT = 0.3

# Training
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 5
VAL_FRACTION = 0.1
SEED = 42

# Hamming baseline
HAMMING_MAX_DIST = 1  # max edits allowed when sliding a vulgar word over a nick
HAMMING_MIN_WORD_LEN = 4  # ignore very short ban words to avoid false positives

# Artifacts
MODEL_PATH = ARTIFACTS_DIR / "nick_classifier.pt"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"
BANLIST_PATH = ARTIFACTS_DIR / "banlist_pl.txt"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
