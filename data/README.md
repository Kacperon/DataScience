# Data Sources

Datasets and word lists for training a vulgar nickname moderation model.

## Nickname Datasets

| Directory | Source | Description | Size |
|---|---|---|---|
| `namespotting_data/` | [Namespotting (Google Drive)](https://drive.google.com/drive/folders/1Yqq8TPLR3yMPx18n9oEOpHbEzcWae9Vw) | Reddit usernames with toxicity labels. Paper: [Urbaniak et al. 2022](https://www.sciencedirect.com/science/article/abs/pii/S0747563222001935), [GitHub](https://github.com/rfl-urbaniak/namespotting) | 4.5M usernames (122k toxic) |

## Polish Word Lists

| Directory | Source | Description | Size |
|---|---|---|---|
| `steam-profanity-filter/polish/` | [kast1450/steam-profanity-filter](https://github.com/kast1450/steam-profanity-filter) | Valve's production profanity filter extracted from Steam Web API. 28 languages. | 7,101 PL words |
| `BAN-PL/resources/` | [NASK-NLP/BAN-PL](https://github.com/NASK-NLP/BAN-PL) | Polish vulgarisms extended list + leet-speak mappings. ZIP passwords in `data/README.md`. | 6,482 words |
| `wulgaryzmy/` | [coldner/wulgaryzmy](https://github.com/coldner/wulgaryzmy) | Polish curse words (JSON). License: MIT. | 711 words |
| `LDNOOBW/pl` | [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) | Bad words list by Shutterstock, 31 languages. License: CC-BY-4.0. | 54 PL words |

## English Word Lists

| Directory | Source | Description | Size |
|---|---|---|---|
| `surgeai_profanity/` | [mmathys/profanity (HuggingFace)](https://huggingface.co/datasets/mmathys/profanity) | Surge AI profanity list with severity ratings (1-6) and categories. | 1,598 words |
| `google-profanity-words/` | [coffee-and-fun/google-profanity-words](https://github.com/coffee-and-fun/google-profanity-words) | Google's profanity word list. | ~960 words |
| `LDNOOBW/en` | [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) | English bad words. License: CC-BY-4.0. | 403 words |
| `BannedWords/` | [grimsausy/BannedWords](https://github.com/grimsausy/BannedWords) | Gaming-specific banned words with bypass variants. | EN word lists |

## Polish Text Datasets (for profanity extraction)

| Directory | Source | Description | Size |
|---|---|---|---|
| `BAN-PL/data/` | [NASK-NLP/BAN-PL](https://github.com/NASK-NLP/BAN-PL) | Wykop.pl moderated posts. Password-protected ZIPs (passwords in `BAN-PL/data/README.md`). License: CC-BY-4.0. | 48k posts (24k harmful + 24k neutral) |
| `BAN-PL/data/HATE-WAR-PL_1/` | [NASK-NLP/BAN-PL](https://github.com/NASK-NLP/BAN-PL) | Hate speech subcorpus. Encrypted XLSX (password: `HATE-WAR-PL_1`). | 2,263 samples |
