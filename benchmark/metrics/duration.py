"""PRO-04 and PRO-05: Duration Accuracy Metrics.

PRO-04: Duration MAE (Mean Absolute Error in Milliseconds)
PRO-05: Duration length ratio error (Percentage error in overall speaking length)
"""
from typing import Dict, Any
import numpy as np


def calculate_duration_metrics(ref_duration_sec: float, syn_duration_sec: float) -> Dict[str, Any]:
    """Calculate utterance-level duration MAE and ratio deviation."""
    ref_ms = ref_duration_sec * 1000.0
    syn_ms = syn_duration_sec * 1000.0

    mae_ms = abs(ref_ms - syn_ms)
    ratio = syn_duration_sec / max(1e-4, ref_duration_sec)
    ratio_error_pct = abs(1.0 - ratio) * 100.0

    return {
        "duration_mae_ms": round(mae_ms, 2),
        "duration_length_ratio": round(ratio, 4),
        "duration_length_error_pct": round(ratio_error_pct, 2),
        "ref_duration_sec": round(ref_duration_sec, 3),
        "syn_duration_sec": round(syn_duration_sec, 3),
    }
