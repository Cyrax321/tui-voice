"""PRO-01 through PRO-03: Fundamental Frequency (F0) & Voicing Metrics.

Calculates:
- PRO-01: F0 RMSE (Root Mean Square Error in Hz and semitones)
- PRO-02: F0 Pearson Correlation Coefficient
- PRO-03: Voicing Decision Error (VDE % of frames where voiced/unvoiced classification disagrees)
"""
import numpy as np
from typing import Dict, Any, Tuple


def extract_f0_autocorr(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0,
    f0_min: float = 60.0,
    f0_max: float = 400.0,
    energy_threshold_db: float = -45.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract fundamental frequency (F0) and voiced mask via normalized autocorrelation."""
    if len(audio) == 0:
        return np.array([]), np.array([])

    frame_len = int(sample_rate * (frame_size_ms / 1000.0))
    hop_len = int(sample_rate * (hop_size_ms / 1000.0))
    lag_min = int(sample_rate / f0_max)
    lag_max = int(sample_rate / f0_min)

    num_frames = max(1, (len(audio) - frame_len) // hop_len + 1)
    f0_series = np.zeros(num_frames, dtype=np.float32)
    voiced_mask = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        start = i * hop_len
        frame = audio[start:start + frame_len]
        if len(frame) < frame_len:
            break

        # Check energy
        rms = np.sqrt(np.mean(frame ** 2)) + 1e-12
        rms_db = 20.0 * np.log10(rms)
        if rms_db < energy_threshold_db:
            continue

        # Windowing
        win_frame = frame * np.hanning(len(frame))
        # Autocorrelation via FFT
        fft = np.fft.rfft(win_frame, n=2 * frame_len)
        corr = np.fft.irfft(fft * np.conj(fft))[:frame_len]
        if corr[0] <= 0:
            continue
        norm_corr = corr / corr[0]

        # Find peak in search range
        search_region = norm_corr[lag_min:lag_max]
        if len(search_region) == 0:
            continue
        peak_idx = np.argmax(search_region) + lag_min
        peak_val = norm_corr[peak_idx]

        if peak_val > 0.35:  # Voicing threshold
            f0_series[i] = sample_rate / float(peak_idx)
            voiced_mask[i] = True

    return f0_series, voiced_mask


def calculate_f0_metrics(
    ref_audio: np.ndarray,
    syn_audio: np.ndarray,
    sample_rate: int = 24000
) -> Dict[str, Any]:
    """Calculate F0 RMSE, F0 Correlation, and Voicing Decision Error."""
    ref_f0, ref_v = extract_f0_autocorr(ref_audio, sample_rate)
    syn_f0, syn_v = extract_f0_autocorr(syn_audio, sample_rate)

    min_len = min(len(ref_f0), len(syn_f0))
    if min_len == 0:
        return {"f0_rmse_hz": 0.0, "f0_rmse_semitones": 0.0, "f0_correlation": 0.0, "voicing_decision_error_pct": 0.0}

    ref_f0 = ref_f0[:min_len]
    syn_f0 = syn_f0[:min_len]
    ref_v = ref_v[:min_len]
    syn_v = syn_v[:min_len]

    # PRO-03: Voicing Decision Error (VDE)
    vde_count = np.sum(ref_v != syn_v)
    vde_pct = (vde_count / float(min_len)) * 100.0

    # Mutually voiced frames for F0 error calculation
    mutually_voiced = ref_v & syn_v
    if np.sum(mutually_voiced) < 5:
        return {
            "f0_rmse_hz": 0.0,
            "f0_rmse_semitones": 0.0,
            "f0_correlation": 0.0,
            "voicing_decision_error_pct": round(float(vde_pct), 2),
            "mutually_voiced_frames": int(np.sum(mutually_voiced)),
        }

    v_ref = ref_f0[mutually_voiced]
    v_syn = syn_f0[mutually_voiced]

    # PRO-01: F0 RMSE in Hz
    rmse_hz = np.sqrt(np.mean((v_ref - v_syn) ** 2))

    # F0 RMSE in Semitones: 12 * log2(syn / ref)
    semitone_diff = 12.0 * np.log2((v_syn + 1e-6) / (v_ref + 1e-6))
    rmse_semitones = np.sqrt(np.mean(semitone_diff ** 2))

    # PRO-02: F0 Pearson Correlation
    if np.std(v_ref) > 1e-4 and np.std(v_syn) > 1e-4:
        corr_matrix = np.corrcoef(v_ref, v_syn)
        corr = float(corr_matrix[0, 1])
    else:
        corr = 0.0

    return {
        "f0_rmse_hz": round(float(rmse_hz), 2),
        "f0_rmse_semitones": round(float(rmse_semitones), 2),
        "f0_correlation": round(corr, 4),
        "voicing_decision_error_pct": round(float(vde_pct), 2),
        "mutually_voiced_frames": int(np.sum(mutually_voiced)),
    }
