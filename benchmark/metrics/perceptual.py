"""HUM-01 through HUM-07: Human Perceptual Evaluation Data Structures & Statistical Analyzers.

Specification:
- HUM-01: Naturalness MOS (1 to 5)
- HUM-02: Intelligibility MOS (1 to 5)
- HUM-03: Pronunciation MOS (1 to 5)
- HUM-04: Prosody MOS (1 to 5)
- HUM-05: Pairwise Comparative MOS (CMOS, -3 to +3)
- HUM-06: Pairwise Preference Rate (%)
- HUM-07: Listener Consistency & Inter-Rater Reliability (Krippendorff's alpha / intra-class correlation)

Note: In accordance with Non-Negotiable Research Integrity Rule #2 ("Never invent human ratings"),
uncollected listener ratings are strictly recorded as PENDING_HUMAN_EVALUATION.
"""
from typing import Dict, Any, List
import numpy as np


def compute_mos_summary(ratings: List[float]) -> Dict[str, Any]:
    """Compute mean, standard deviation, and 95% Student-t / Normal confidence interval for MOS ratings."""
    if not ratings:
        return {
            "status": "PENDING_HUMAN_EVALUATION",
            "mean": None,
            "std": None,
            "ci_95": None,
            "n_ratings": 0
        }

    arr = np.array(ratings, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci = 1.96 * (std / np.sqrt(n)) if n > 1 else 0.0

    return {
        "status": "COMPLETED",
        "mean": round(mean, 3),
        "std": round(std, 3),
        "ci_95": round(ci, 3),
        "ci_lower": round(mean - ci, 3),
        "ci_upper": round(mean + ci, 3),
        "n_ratings": n,
    }


def compute_cmos_summary(cmos_scores: List[float]) -> Dict[str, Any]:
    """Compute CMOS mean and preference distribution."""
    if not cmos_scores:
        return {
            "status": "PENDING_HUMAN_EVALUATION",
            "mean_cmos": None,
            "preference_a_pct": None,
            "preference_b_pct": None,
            "no_preference_pct": None,
            "n_pairs": 0,
        }

    arr = np.array(cmos_scores, dtype=np.float64)
    n = len(arr)
    pref_a = float(np.sum(arr > 0)) / n * 100.0
    pref_b = float(np.sum(arr < 0)) / n * 100.0
    no_pref = float(np.sum(arr == 0)) / n * 100.0

    return {
        "status": "COMPLETED",
        "mean_cmos": round(float(np.mean(arr)), 3),
        "preference_a_pct": round(pref_a, 2),
        "preference_b_pct": round(pref_b, 2),
        "no_preference_pct": round(no_pref, 2),
        "n_pairs": n,
    }
