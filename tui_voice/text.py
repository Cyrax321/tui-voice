"""Malayalam text cleaning. v0 rule: sentences with digits or Latin letters are dropped,
not expanded (Malayalam number words are irregular; add an expander in v1)."""
import re
import unicodedata

ZWNJ, ZWJ = "\u200c", "\u200d"            # kept: they select chillu / conjunct forms
_STRIP = re.compile("[\u200b\u200e\u200f\u2060\ufeff]")
_PUNCT_MAP = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'", "\u2026": ".",
              "\u2013": "-", "\u2014": "-", "\u0964": "."}
_ML = re.compile("[\u0d00-\u0d7f]")
_LATIN = re.compile("[A-Za-z]")
_DIGIT = re.compile("[0-9\u0d66-\u0d6f]")
PAUSE_PUNCT = ",;:.?!"                     # becomes a pause token; other punctuation is dropped


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = _STRIP.sub("", t)
    for a, b in _PUNCT_MAP.items():
        t = t.replace(a, b)
    t = re.sub(r"[\"'()\[\]{}<>*_/\\|~^`#@$%&+=-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def usable(text: str, min_ml_ratio: float = 0.8):
    """Return (ok, reason)."""
    if not text:
        return False, "empty"
    if _DIGIT.search(text):
        return False, "digits"
    if _LATIN.search(text):
        return False, "latin"
    letters = [c for c in text if c.isalpha() or _ML.match(c)]
    if not letters:
        return False, "no_letters"
    ratio = sum(1 for c in letters if _ML.match(c)) / len(letters)
    if ratio < min_ml_ratio:
        return False, "low_malayalam_ratio"
    return True, "ok"


def split_clauses(text: str):
    """'a, b. c' -> [('a', ','), ('b', '.'), ('c', '')]  (clause text, trailing punctuation)."""
    out, buf = [], []
    for ch in text:
        if ch in PAUSE_PUNCT:
            s = "".join(buf).strip()
            if s:
                out.append((s, ch))
            buf = []
        else:
            buf.append(ch)
    s = "".join(buf).strip()
    if s:
        out.append((s, ""))
    return out
