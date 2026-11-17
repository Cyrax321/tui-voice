"""INT-03 through INT-06: Sentence, Word Omission, Word Repetition, and ASR Failure Rates.

INT-03: Sentence Error Rate (SER) - Percentage of utterances where hyp != ref
INT-04: Word Omission Rate (WOR) - Percentage of reference words missing in hypothesis
INT-05: Word Repetition Rate (WRR) - Rate of spurious repeated words in hypothesis
INT-06: ASR Failure Rate (AFR) - Percentage of audio files where ASR failed or returned empty
"""
from typing import List, Dict, Any
from ..text.tokenization import tokenize_words
from .cer import levenshtein_distance


def calculate_sentence_error(ref: str, hyp: str) -> bool:
    """Return True if hypothesis does not exactly match reference after normalization."""
    ref_w = tokenize_words(ref, strip_punct=True)
    hyp_w = tokenize_words(hyp, strip_punct=True)
    return ref_w != hyp_w


def calculate_word_omission_and_repetition(ref: str, hyp: str) -> Dict[str, float]:
    """Calculate Word Omission Rate and Word Repetition Rate via token alignment."""
    ref_words = tokenize_words(ref, strip_punct=True)
    hyp_words = tokenize_words(hyp, strip_punct=True)

    if not ref_words:
        return {"word_omission_rate": 0.0, "word_repetition_rate": 0.0}

    # Levenshtein distance gives exact deletions (omissions)
    _, _, deletions, _ = levenshtein_distance(ref_words, hyp_words)
    omission_rate = (deletions / len(ref_words)) * 100.0

    # Repetition detection: count adjacent identical words in hyp that weren't repeated in ref
    def count_repetitions(words: List[str]) -> int:
        reps = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                reps += 1
        return reps

    hyp_reps = count_repetitions(hyp_words)
    ref_reps = count_repetitions(ref_words)
    spurious_reps = max(0, hyp_reps - ref_reps)

    repetition_rate = (spurious_reps / len(ref_words)) * 100.0

    return {
        "word_omission_rate": round(omission_rate, 4),
        "word_repetition_rate": round(repetition_rate, 4),
        "deleted_words": deletions,
        "spurious_repetitions": spurious_reps,
    }


def aggregate_intelligibility_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate sentence error, word omission, word repetition, and ASR failure over a dataset."""
    total = len(records)
    if total == 0:
        return {}

    sentence_errors = 0
    total_ref_words = 0
    total_deletions = 0
    total_repetitions = 0
    asr_failures = 0

    for r in records:
        ref = r.get("reference", "")
        hyp = r.get("hypothesis", "")
        failed = r.get("asr_failed", False) or not hyp.strip()

        if failed:
            asr_failures += 1
            sentence_errors += 1
            ref_w = tokenize_words(ref, strip_punct=True)
            total_ref_words += len(ref_w)
            total_deletions += len(ref_w)
            continue

        if calculate_sentence_error(ref, hyp):
            sentence_errors += 1

        stats = calculate_word_omission_and_repetition(ref, hyp)
        ref_w = tokenize_words(ref, strip_punct=True)
        total_ref_words += len(ref_w)
        total_deletions += stats["deleted_words"]
        total_repetitions += stats["spurious_repetitions"]

    return {
        "total_utterances": total,
        "sentence_error_rate": round((sentence_errors / total) * 100.0, 2),
        "word_omission_rate": round((total_deletions / max(1, total_ref_words)) * 100.0, 2),
        "word_repetition_rate": round((total_repetitions / max(1, total_ref_words)) * 100.0, 2),
        "asr_failure_rate": round((asr_failures / total) * 100.0, 2),
    }
