"""Load and prepare nickname dataset for training."""
import json
import random
from pathlib import Path

import pandas as pd

from .augment import augment_corpus
from .config import DATA_DIR, NICK_MAX_LEN, SEED, VAL_FRACTION


# ── Polish names for clean nick generation ────────────────────────────────
PL_FIRST_NAMES = [
    "marek", "anna", "tomek", "kasia", "piotr", "ewa", "krzysiek", "magda",
    "michal", "ola", "jakub", "asia", "adam", "natalia", "bartek", "agnieszka",
    "lukasz", "monika", "pawel", "joanna", "rafal", "karolina", "andrzej", "iza",
    "darek", "beata", "robert", "marta", "jarek", "patrycja", "wojtek", "dorota",
    "filip", "weronika", "kamil", "alicja", "dominik", "klaudia", "mateusz", "zuzanna",
    "szymon", "wiktoria", "dawid", "julia", "hubert", "oliwia", "igor", "amelia",
]
PL_LAST_BITS = [
    "ski", "cki", "wicz", "owski", "ek", "ak", "uk", "owicz", "kowski",
]
PL_GAMER_WORDS = [
    "dragon", "ninja", "ghost", "wolf", "shadow", "fire", "ice", "thunder",
    "king", "lord", "master", "warrior", "knight", "hunter", "killer", "pro",
    "elite", "legend", "hero", "mage", "rogue", "assassin", "samurai", "viking",
    "smok", "wilk", "orzel", "rycerz", "lowca", "krol", "wojownik", "pan",
]


# ── Loaders ───────────────────────────────────────────────────────────────
def load_namespotting() -> pd.DataFrame:
    """Returns DataFrame with columns: nick, label (1=vulgar, 0=clean)."""
    df = pd.read_csv(
        DATA_DIR / "namespotting_data" / "datasets" / "usernamesToxicOrNot.tsv",
        sep="\t", header=None, names=["nick", "toxic_label", "_c", "_t"],
        usecols=["nick", "toxic_label"],
    )
    df = df.dropna(subset=["nick"])
    df["nick"] = df["nick"].astype(str).str.strip()
    df = df[df["nick"].str.len().between(2, NICK_MAX_LEN)]
    df["label"] = (df["toxic_label"] == "toxic").astype(int)
    return df[["nick", "label"]].reset_index(drop=True)


def load_pl_banlist() -> list[str]:
    """Combine all PL profanity word lists into a deduplicated set."""
    sources = []

    # Steam PL (production filter)
    for fn in ["polish-profanity.txt", "polish-banned.txt"]:
        fp = DATA_DIR / "steam-profanity-filter" / "polish" / fn
        if fp.exists():
            sources.extend(line.strip() for line in fp.open(encoding="utf-8") if line.strip())

    # coldner/wulgaryzmy
    fp = DATA_DIR / "wulgaryzmy" / "wulgaryzmy.json"
    if fp.exists():
        sources.extend(json.loads(fp.read_text(encoding="utf-8")))

    # BAN-PL extended vulgarisms
    fp = DATA_DIR / "BAN-PL" / "resources" / "polish_vulgarisms_extended_2.0.txt."
    if fp.exists():
        sources.extend(line.strip() for line in fp.open(encoding="utf-8") if line.strip())

    # LDNOOBW PL
    fp = DATA_DIR / "LDNOOBW" / "pl"
    if fp.exists():
        sources.extend(line.strip() for line in fp.open(encoding="utf-8") if line.strip())

    deduped = sorted({w.lower() for w in sources if 3 <= len(w) <= NICK_MAX_LEN})
    return deduped


def load_pl_clean_whitelist() -> set[str]:
    """Words flagged as profanity false-positives — must NOT be classified as vulgar."""
    fp = DATA_DIR / "steam-profanity-filter" / "polish" / "polish-clean_public.txt"
    if not fp.exists():
        return set()
    return {w.strip().lower() for w in fp.open(encoding="utf-8") if w.strip()}


# ── Synthetic clean PL nicks ──────────────────────────────────────────────
def generate_clean_pl_nicks(n: int, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    out: set[str] = set()
    while len(out) < n:
        kind = rng.choice(["name_num", "name_name", "gamer", "compound"])
        if kind == "name_num":
            nick = f"{rng.choice(PL_FIRST_NAMES)}{rng.choice(['', '_', '.'])}{rng.randint(1, 9999)}"
        elif kind == "name_name":
            nick = f"{rng.choice(PL_FIRST_NAMES)}{rng.choice(['_', '.'])}{rng.choice(PL_FIRST_NAMES)}"
        elif kind == "gamer":
            nick = f"{rng.choice(PL_GAMER_WORDS)}{rng.choice(['', '_', '.'])}{rng.choice(PL_GAMER_WORDS)}"
        else:
            nick = f"{rng.choice(PL_FIRST_NAMES)}{rng.choice(PL_LAST_BITS)}{rng.randint(0, 99)}"
        if 3 <= len(nick) <= NICK_MAX_LEN:
            out.add(nick.lower())
    return sorted(out)


# ── Build the full dataset ────────────────────────────────────────────────
def build_dataset(
    n_pl_augmented_per_word: int = 8,
    en_clean_sample: int = 200_000,
    pl_clean_count: int = 80_000,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns: train_df, val_df, test_df, info_dict
    Each DataFrame has columns: nick, label, source.
    """
    rng = random.Random(seed)

    # 1. Namespotting — both classes
    print("Loading Namespotting...")
    ns = load_namespotting()
    ns_toxic = ns[ns["label"] == 1].copy()
    ns_clean = ns[ns["label"] == 0].copy()
    if en_clean_sample and len(ns_clean) > en_clean_sample:
        ns_clean = ns_clean.sample(n=en_clean_sample, random_state=seed).reset_index(drop=True)
    ns_toxic["source"] = "namespotting_toxic"
    ns_clean["source"] = "namespotting_clean"
    print(f"  Namespotting: {len(ns_toxic):,} toxic + {len(ns_clean):,} clean")

    # 2. PL banlist seed words
    print("Loading PL banlist...")
    banlist = load_pl_banlist()
    print(f"  PL banlist: {len(banlist):,} unique words")

    # 3. PL augmentation — vulgar synthetic nicks
    print("Augmenting PL vulgar nicks...")
    pl_vulgar = augment_corpus(banlist, n_per_word=n_pl_augmented_per_word, max_len=NICK_MAX_LEN, seed=seed)
    pl_vulgar_df = pd.DataFrame({"nick": pl_vulgar, "label": 1, "source": "pl_augmented"})
    print(f"  PL augmented vulgar: {len(pl_vulgar_df):,}")

    # 4. PL synthetic clean nicks
    print("Generating clean PL nicks...")
    pl_clean = generate_clean_pl_nicks(pl_clean_count, seed=seed)
    pl_clean_df = pd.DataFrame({"nick": pl_clean, "label": 0, "source": "pl_synthetic_clean"})
    print(f"  PL synthetic clean: {len(pl_clean_df):,}")

    # 5. Combine + dedupe + shuffle
    df = pd.concat([ns_toxic, ns_clean, pl_vulgar_df, pl_clean_df], ignore_index=True)
    df["nick"] = df["nick"].str.lower().str.strip()
    df = df.drop_duplicates(subset="nick", keep="first").reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 6. Train/val/test split
    n = len(df)
    n_test = int(n * VAL_FRACTION)
    n_val = int(n * VAL_FRACTION)
    test_df = df.iloc[:n_test].reset_index(drop=True)
    val_df = df.iloc[n_test : n_test + n_val].reset_index(drop=True)
    train_df = df.iloc[n_test + n_val :].reset_index(drop=True)

    info = {
        "total": n,
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "vulgar_pct": float(df["label"].mean()),
        "by_source": df["source"].value_counts().to_dict(),
        "banlist_size": len(banlist),
    }

    print(f"\nFinal dataset:")
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print(f"  Vulgar: {df['label'].mean()*100:.1f}%")
    print(f"  By source: {info['by_source']}")

    return train_df, val_df, test_df, info


if __name__ == "__main__":
    train, val, test, info = build_dataset()
    print(json.dumps(info, indent=2, default=str))
