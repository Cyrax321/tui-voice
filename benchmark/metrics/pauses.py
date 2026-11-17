"""PRO-06 through PRO-09: Pause Detection Precision, Recall, F1, and Duration MAE.

Detects silence regions (> 100 ms with energy < -45 dBFS) and calculates:
- PRO-06: Pause detection precision (%)
- PRO-07: Pause detection recall (%)
- PRO-08: Pause detection F1 (score 0.0 - 1.0)
- PRO-09: Pause duration MAE (ms)
"""
import numpy as np
from typing import Dict, Any, List, Tuple


def detect_silence_intervals(
    audio: np.ndarray,
    sample_rate: int = 24000,
    min_pause_ms: float = 120.0,
    threshold_db: float = -42.0,
    frame_ms: float = 20.0
) -> List[Tuple[float, float]]:
    """Detect start and end timestamps (in seconds) of silent pause intervals."""
    if len(audio) == 0:
        return []

    frame_len = int(sample_rate * (frame_ms / 1000.0))
    min_frames = int(min_pause_ms / frame_ms)

    num_frames = len(audio) // frame_len
    silent_mask = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        chunk = audio[i * frame_len:(i + 1) * frame_len]
        rms = np.sqrt(np.mean(chunk ** 2)) + 1e-12
        db = 20.0 * np.log10(rms)
        if db < threshold_db:
            silent_mask[i] = True

    # Group contiguous silent frames
    pauses = []
    in_pause = False
    start_frame = 0

    for i, is_silent in enumerate(silent_mask):
        if is_silent and not in_pause:
            in_pause = True
            start_frame = i
        elif not is_silent and in_pause:
            in_pause = False
            length_frames = i - start_frame
            if length_frames >= min_frames:
                t_start = (start_frame * frame_len) / float(sample_rate)
                t_end = (i * frame_len) / float(sample_rate)
                pauses.append((t_start, t_end))

    if in_pause:
        length_frames = num_frames - start_frame
        if length_frames >= min_frames:
            t_start = (start_frame * frame_len) / float(sample_rate)
            t_end = (num_frames * frame_len) / float(sample_rate)
            pauses.append((t_start, t_end))

    return pauses


def calculate_pause_metrics(
    ref_audio: np.ndarray,
    syn_audio: np.ndarray,
    sample_rate: int = 24000
) -> Dict[str, Any]:
    """Calculate pause detection precision, recall, F1, and duration MAE."""
    ref_pauses = detect_silence_intervals(ref_audio, sample_rate)
    syn_pauses = detect_silence_intervals(syn_audio, sample_rate)

    ref_count = len(ref_pauses)
    syn_count = len(syn_pauses)

    if ref_count == 0 and syn_count == 0:
        return {
            "pause_precision_pct": 100.0,
            "pause_recall_pct": 100.0,
            "pause_f1": 1.0,
            "pause_duration_mae_ms": 0.0,
            "ref_pause_count": 0,
            "syn_pause_count": 0,
        }

    # Match pauses by overlapping time window (with 250ms tolerance)
    matched_ref = set()
    matched_syn = set()
    duration_errors = []

    for s_idx, (s_start, s_end) in enumerate(syn_pauses):
        for r_idx, (r_start, r_end) in enumerate(ref_pauses):
            if r_idx in matched_ref:
                continue
            # Overlap check
            if max(s_start, r_start) <= min(s_end, r_end) + 0.25:
                matched_ref.add(r_idx)
                matched_syn.add(s_idx)
                err_ms = abs((s_end - s_start) - (r_end - r_start)) * 1000.0
                duration_errors.append(err_ms)
                break

    tp = len(matched_syn)
    fp = syn_count - tp
    fn = ref_count - len(matched_ref)

    precision = (tp / max(1, (tp + fp))) * 100.0
    recall = (tp / max(1, (tp + fn))) * 100.0
    f1 = (2 * precision * recall) / max(1e-6, (precision + recall)) / 100.0
    mae = float(np.mean(duration_errors)) if duration_errors else 0.0

    return {
        "pause_precision_pct": round(precision, 2),
        "pause_recall_pct": round(recall, 2),
        "pause_f1": round(f1, 4),
        "pause_duration_mae_ms": round(mae, 2),
        "ref_pause_count": ref_count,
        "syn_pause_count": syn_count,
    }
