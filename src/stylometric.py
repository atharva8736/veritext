"""Stylometric features: numbers that describe HOW a text is written (its style),
not WHAT it says. AI text tends to be smooth, uniform and formal; human text is
more irregular ("bursty"), with more personal/casual quirks."""
import re
import numpy as np

FUNCTION_WORDS = set("""the a an and or but if of to in on at by for with from as is are was were be been
it this that these those i you he she we they not no so than then there their his her its our your""".split())
TRANSITIONS = {"moreover", "furthermore", "additionally", "consequently", "therefore", "overall",
               "however", "thus", "notably", "importantly", "ultimately", "in conclusion", "in summary"}

FEATURE_NAMES = [
    "num_words", "avg_word_len", "avg_sentence_len", "sentence_len_std", "burstiness",
    "type_token_ratio", "hapax_ratio", "comma_ratio", "punct_ratio", "uppercase_ratio",
    "digit_ratio", "function_word_ratio", "transition_ratio", "contraction_ratio",
    "first_person_ratio", "question_exclaim_ratio", "repeated_bigram_ratio",
]

def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

def extract(text: str) -> np.ndarray:
    words = re.findall(r"[A-Za-z']+", text.lower())
    n = max(len(words), 1)
    sents = _sentences(text) or [text]
    sl = np.array([len(re.findall(r"[A-Za-z']+", s)) for s in sents], dtype=float)
    chars = max(len(text), 1)
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    bigrams = list(zip(words, words[1:]))
    rep_bi = (len(bigrams) - len(set(bigrams))) / max(len(bigrams), 1)
    mean_sl, std_sl = sl.mean(), sl.std()
    low = text.lower()
    feats = [
        len(words),
        np.mean([len(w) for w in words]) if words else 0,
        mean_sl,
        std_sl,
        (std_sl - mean_sl) / (std_sl + mean_sl + 1e-9),           # burstiness
        len(counts) / n,                                           # type-token ratio
        sum(1 for c in counts.values() if c == 1) / n,             # hapax ratio
        text.count(",") / n,
        len(re.findall(r"[^\w\s]", text)) / chars,
        sum(1 for c in text if c.isupper()) / chars,
        sum(1 for c in text if c.isdigit()) / chars,
        sum(1 for w in words if w in FUNCTION_WORDS) / n,
        sum(low.count(t) for t in TRANSITIONS) / n,
        len(re.findall(r"\b\w+'(?:t|s|m|re|ve|ll|d)\b", low)) / n,
        sum(1 for w in words if w in {"i", "me", "my", "mine", "we", "our"}) / n,
        (text.count("?") + text.count("!")) / max(len(sents), 1),
        rep_bi,
    ]
    return np.array(feats, dtype=float)

def extract_many(texts):
    return np.vstack([extract(t) for t in texts])
