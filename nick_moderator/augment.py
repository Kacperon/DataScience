"""PL-specific nickname augmentation: leet, separators, affixes."""
import random
from typing import Iterable

# From BAN-PL src/__init__.py — leet -> Polish letter
POPULAR_CHANGES = {
    "i": ["1", "!", "&"],
    "a": ["@", "4"],
    "e": ["3"],
    "l": ["1", "!"],
    "s": ["$", "5"],
    "ś": ["$"],
    "o": ["0"],
    "z": ["2"],
    "b": ["8"],
    "k": ["q"],
}

SEPARATORS = ["", "_", ".", "-", "x", "0"]
PREFIXES = ["", "xX", "x", "the_", "mr_", "pan_", "_", "iam"]
SUFFIXES = ["", "Xx", "_69", "_420", "_pl", "69", "420", "1", "12", "123", "_", "_xd", "_official"]

# Clean words to use as suffixes/prefixes around vulgar word — generates patterns
# like `cipa_master`, `kurwa_killer` that look like real combined nicks.
COMPOUND_WORDS = [
    "master", "killer", "lord", "king", "boss", "pro", "elite", "legend",
    "ninja", "gamer", "warrior", "hunter", "smok", "wilk", "rycerz",
    "official", "real", "the", "best", "top", "ultra", "mega", "super",
]


def leet(word: str, p: float = 1.0) -> str:
    """Replace letters with leet equivalents with probability p per letter."""
    out = []
    for ch in word.lower():
        if ch in POPULAR_CHANGES and random.random() < p:
            out.append(random.choice(POPULAR_CHANGES[ch]))
        else:
            out.append(ch)
    return "".join(out)


def with_separator(word: str, sep: str | None = None) -> str:
    if sep is None:
        sep = random.choice(SEPARATORS)
    return sep.join(word)


def with_affix(word: str) -> str:
    return random.choice(PREFIXES) + word + random.choice(SUFFIXES)


def random_case(word: str) -> str:
    return "".join(c.upper() if random.random() < 0.3 else c for c in word)


def with_compound(word: str) -> str:
    """vulgar_word + _ + clean_word  OR  clean_word + _ + vulgar_word."""
    other = random.choice(COMPOUND_WORDS)
    sep = random.choice(["_", ".", ""])
    if random.random() < 0.5:
        return f"{word}{sep}{other}"
    return f"{other}{sep}{word}"


def augment_nick(word: str, n_variants: int = 5, max_len: int = 32) -> list[str]:
    """Generate diverse vulgar nick variants from a seed vulgar word.

    Mix of: bare word, leet, separator, affix, case, and compound (vulgar+clean_word).
    """
    word = word.lower().strip()
    if not word or len(word) > max_len:
        return []

    variants: set[str] = {word}

    for i in range(n_variants):
        v = word
        # First few variants: structurally simple (bare or compound) — these match
        # real-world patterns like 'cipa_master' that don't use heavy obfuscation.
        if i == 0:
            v = with_compound(v)
        elif i == 1:
            v = with_affix(v)
        else:
            ops = random.sample(["leet", "sep", "affix", "case", "compound"], k=random.randint(1, 3))
            if "leet" in ops:
                v = leet(v, p=random.uniform(0.3, 0.9))
            if "sep" in ops:
                v = with_separator(v)
            if "affix" in ops:
                v = with_affix(v)
            if "compound" in ops:
                v = with_compound(v)
            if "case" in ops:
                v = random_case(v)
        v = v[:max_len]
        if v and v != word:
            variants.add(v)

    return list(variants)


def augment_corpus(seed_words: Iterable[str], n_per_word: int = 5, max_len: int = 32, seed: int = 42) -> list[str]:
    random.seed(seed)
    out: set[str] = set()
    for w in seed_words:
        for v in augment_nick(w, n_variants=n_per_word, max_len=max_len):
            out.add(v)
    return sorted(out)
