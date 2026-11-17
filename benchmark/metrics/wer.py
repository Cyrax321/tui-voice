"""INT-02: Malayalam Word Error Rate (WER).

Formula:
  WER (%) = 100 * (Substitutions + Deletions + Insertions) / Reference Word Count
"""
from ..text.tokenization import tokenize_words
from .cer import levenshtein_distance


def calculate_wer(ref: str, hyp: str) -> dict:
    """Calculate Word Error Rate between reference and hypothesis text."""
    ref_words = tokenize_words(ref, strip_punct=True)
    hyp_words = tokenize_words(hyp, strip_punct=True)

    ref_len = len(ref_words)
    if ref_len == 0:
        return {
            "wer": 0.0 if len(hyp_words) == 0 else 100.0,
            "ref_words": 0,
            "hyp_words": len(hyp_words),
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_words),
        }

    dist, subs, dels, inss = levenshtein_distance(ref_words, hyp_words)
    wer = (dist / ref_len) * 100.0

    return {
        "wer": round(wer, 4),
        "ref_words": ref_len,
        "hyp_words": len(hyp_words),
        "substitutions": subs,
        "deletions": dels,
        "insertions": inss,
    }
