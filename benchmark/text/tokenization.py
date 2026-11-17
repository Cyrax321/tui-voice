"""Malayalam Word and Token Extraction Utilities for Benchmark Evaluation."""
import re
from .normalization import normalize_malayalam


def tokenize_words(text: str, strip_punct: bool = True) -> list[str]:
    """Tokenize Malayalam sentence into word units.
    Handles hyphenated compound words and English code-switched terms.
    """
    clean_text = normalize_malayalam(text, verbalize_nums=False, strip_punct=strip_punct)
    if not clean_text:
        return []
    # Match words composed of Malayalam script, English letters, and internal apostrophes
    words = clean_text.split()
    if strip_punct:
        words = [re.sub(r'^[^\w]+|[^\w]+$', '', w) for w in words]
        words = [w for w in words if w]
    return words
