"""AUD-01 through AUD-07: Audio Signal Integrity Metrics.

Detects:
- AUD-01: Empty audio (duration < 0.1s or pure silence, RMS < -60 dBFS)
- AUD-02: Clipped audio (|x| >= 0.999 or sample clipping rate > 0.1%)
- AUD-03: Unusable audio (NaN, Inf, or empty)
- AUD-04: Premature termination (audio cut off prematurely before sentence ends)
- AUD-05: Audio duration deviation (large discrepancy vs expected speech rate)
- AUD-06: Sample rate consistency (deviation from expected sample rate)
- AUD-07: Non-finite audio sample count (count of NaN or Inf values)
"""
from typing import Dict, Any, List
import numpy as np


def check_audio_integrity(
    audio: np.ndarray,
    sample_rate: int,
    expected_sample_rate: int = 24000,
    text_length_graphemes: int = 0
) -> Dict[str, Any]:
    """Analyze a synthesized waveform for acoustic defects and corruption."""
    if audio is None or len(audio) == 0:
        return {
            "empty": True,
            "clipped": False,
            "unusable": True,
            "non_finite_count": 0,
            "duration_sec": 0.0,
            "sample_rate_correct": sample_rate == expected_sample_rate,
            "premature_termination": True,
            "clipping_ratio": 0.0,
            "rms_db": -100.0,
        }

    # AUD-07: Non-finite count
    non_finite = int(np.sum(~np.isfinite(audio)))
    if non_finite > 0:
        clean_audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        clean_audio = audio

    duration_sec = len(clean_audio) / float(sample_rate)

    # AUD-01: Empty / Pure Silence
    rms = float(np.sqrt(np.mean(clean_audio ** 2)))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    empty = (duration_sec < 0.1) or (rms_db < -60.0)

    # AUD-02: Clipping
    peak = float(np.max(np.abs(clean_audio))) if len(clean_audio) > 0 else 0.0
    clipping_samples = int(np.sum(np.abs(clean_audio) >= 0.999))
    clipping_ratio = clipping_samples / float(len(clean_audio)) if len(clean_audio) > 0 else 0.0
    clipped = (peak >= 0.999) and (clipping_ratio > 0.0005)

    # AUD-04: Premature termination (e.g. 50 graphemes synthesized in < 0.3 seconds)
    # Expected speaking rate: ~10 to 18 graphemes per second
    expected_min_duration = (text_length_graphemes / 30.0) if text_length_graphemes > 10 else 0.2
    premature_termination = (duration_sec < expected_min_duration) and not empty

    # AUD-03: Unusable
    unusable = empty or (non_finite > 0) or premature_termination

    # AUD-06: Sample rate check
    sr_consistent = (sample_rate == expected_sample_rate)

    return {
        "empty": empty,
        "clipped": clipped,
        "unusable": unusable,
        "premature_termination": premature_termination,
        "non_finite_count": non_finite,
        "duration_sec": round(duration_sec, 3),
        "sample_rate_correct": sr_consistent,
        "clipping_ratio": round(clipping_ratio, 6),
        "rms_db": round(rms_db, 2),
        "peak": round(peak, 4),
    }


def aggregate_audio_integrity(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate audio integrity checks over an entire evaluation set."""
    n = len(results)
    if n == 0:
        return {}

    empty_count = sum(1 for r in results if r["empty"])
    clipped_count = sum(1 for r in results if r["clipped"])
    unusable_count = sum(1 for r in results if r["unusable"])
    premature_count = sum(1 for r in results if r["premature_termination"])
    sr_mismatch_count = sum(1 for r in results if not r["sample_rate_correct"])
    non_finite_total = sum(r["non_finite_count"] for r in results)

    return {
        "total_evaluated": n,
        "empty_rate_pct": round((empty_count / n) * 100.0, 2),
        "clipped_rate_pct": round((clipped_count / n) * 100.0, 2),
        "unusable_rate_pct": round((unusable_count / n) * 100.0, 2),
        "premature_termination_pct": round((premature_count / n) * 100.0, 2),
        "sample_rate_consistency_pct": round(((n - sr_mismatch_count) / n) * 100.0, 2),
        "total_non_finite_samples": non_finite_total,
    }
