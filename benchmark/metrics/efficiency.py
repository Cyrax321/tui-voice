"""EFF-01 through EFF-11: Model Size and Inference Efficiency Profiler.

Measures:
- EFF-01: Total deployed parameters (Millions)
- EFF-02: Trainable parameters (Millions)
- EFF-03: Serialized model size (MB on disk)
- EFF-04: Peak CPU RAM (MB via psutil / resource)
- EFF-05: Peak GPU VRAM (MB via torch.cuda if available, or MPS)
- EFF-06: CPU inference RTF (Ratio = wall clock time / generated audio duration)
- EFF-07: GPU inference RTF (Ratio)
- EFF-08: First-audio / cold-start latency (ms)
- EFF-09: Per-utterance warm latency (ms)
- EFF-10: Audio throughput (Audio seconds synthesized per real wall-clock second)
- EFF-11: Model loading time (Seconds)
"""
import time
import os
from typing import Dict, Any, Callable
import numpy as np


def count_parameters(model) -> Dict[str, float]:
    """Count total and trainable parameters in millions."""
    if hasattr(model, "parameters"):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_parameters_m": round(total / 1e6, 3),
            "trainable_parameters_m": round(trainable / 1e6, 3),
            "exact_total_params": total,
        }
    return {
        "total_parameters_m": 0.0,
        "trainable_parameters_m": 0.0,
        "exact_total_params": 0,
    }


def measure_file_size_mb(filepath: str) -> float:
    """Measure serialized disk size in megabytes."""
    if os.path.exists(filepath):
        return round(os.path.getsize(filepath) / (1024 * 1024), 2)
    return 0.0


def profile_inference_latency(
    synthesize_fn: Callable[[str], tuple[np.ndarray, int]],
    test_sentences: list[str],
    warmup_count: int = 2
) -> Dict[str, Any]:
    """Measure cold-start, warm per-utterance latency, RTF, and throughput."""
    if not test_sentences:
        return {}

    # 1. Measure cold-start latency on the first utterance
    t0 = time.perf_counter()
    cold_audio, sr = synthesize_fn(test_sentences[0])
    cold_time = (time.perf_counter() - t0) * 1000.0  # ms

    # 2. Warmup
    for i in range(min(warmup_count, len(test_sentences))):
        _ = synthesize_fn(test_sentences[i])

    # 3. Benchmark warm runs
    latencies_ms = []
    audio_durations_sec = []
    wall_clock_times_sec = []

    for text in test_sentences:
        t_start = time.perf_counter()
        audio, sample_rate = synthesize_fn(text)
        t_elapsed = time.perf_counter() - t_start

        dur_sec = len(audio) / float(sample_rate) if audio is not None and len(audio) > 0 else 0.001
        latencies_ms.append(t_elapsed * 1000.0)
        wall_clock_times_sec.append(t_elapsed)
        audio_durations_sec.append(dur_sec)

    total_wall_clock = sum(wall_clock_times_sec)
    total_audio_time = sum(audio_durations_sec)

    rtf = total_wall_clock / max(1e-6, total_audio_time)
    throughput = total_audio_time / max(1e-6, total_wall_clock)

    return {
        "cold_start_latency_ms": round(cold_time, 2),
        "mean_warm_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "median_warm_latency_ms": round(float(np.median(latencies_ms)), 2),
        "p95_warm_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "real_time_factor_rtf": round(rtf, 4),
        "audio_throughput_sec_per_sec": round(throughput, 2),
        "total_audio_generated_sec": round(total_audio_time, 2),
    }
