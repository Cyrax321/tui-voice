"""Training bridge: prepares acoustic features, remaps checkpoint, and launches
tiny-voca training on Malayalam annotations.

This bridges tui-voice's annotation format into what tiny_voca.train expects:
  1. Extract acoustic features (mel, prosody, wave) from 24 kHz wavs -> .pt files
  2. Patch annotations.json with the right fingerprints
  3. Remap upstream checkpoint embeddings using vocab_plan.json
  4. Launch tiny_voca.train with --init-from and --new-data

Usage (Colab, cwd = /content/tiny-voca):
  python /content/tui-voice/scripts/06_train_bridge.py \\
    --annotations /content/data/annotations \\
    --wavs /content/data/wavs \\
    --upstream-ckpt weights/tiny-voca.pt \\
    --vocab-plan /content/data/vocab_plan.json \\
    --output /content/drive/MyDrive/tui_voice/run1 \\
    --steps 40000 --gan-start 8000
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Ensure tiny-voca package is importable
sys.path.insert(0, "/content/tiny-voca/src")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CACHED_FILTERS = None


def get_mel_filters(sr=24000, n_fft=1024, n_mels=80):
    global _CACHED_FILTERS
    if _CACHED_FILTERS is None:
        import librosa
        _CACHED_FILTERS = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return _CACHED_FILTERS


def extract_features(wav_path, config, speaker_idx=0):
    """Robust extraction matching tiny-voca using PyTorch pitch tracking (no Numba crash)."""
    import librosa
    import torchaudio.functional as AF

    sr_target = getattr(config, "sample_rate", 24000)
    hop = getattr(config, "hop", 256)
    n_fft = getattr(config, "n_fft", 1024)
    n_mels = getattr(config, "n_mels", 80)
    f0_min = getattr(config, "f0_min", 50.0)
    f0_max = getattr(config, "f0_max", 600.0)
    pitch_ref = getattr(config, "pitch_reference_hz", 200.0)

    y, sr = sf.read(wav_path, dtype="float32")
    if sr != sr_target:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)

    frames = math.ceil(len(y) / hop)
    if frames < 1:
        return None

    y_pad = np.pad(y, (0, frames * hop - len(y))).astype(np.float32)

    # Mel spectrogram using cached filters
    spectrum = librosa.stft(y_pad, n_fft=n_fft, hop_length=hop,
                            center=True, pad_mode="constant")[:, :frames]
    filters = get_mel_filters(sr=sr_target, n_fft=n_fft, n_mels=n_mels)
    mel = np.log(np.maximum(filters @ np.abs(spectrum) ** 2, 1e-5)).T

    # Pitch tracking via PyTorch (GPU accelerated if available, zero Numba crash)
    w_tensor = torch.from_numpy(y_pad).unsqueeze(0)
    try:
        if torch.cuda.is_available():
            f0 = AF.detect_pitch_frequency(
                w_tensor.cuda(), sample_rate=sr_target,
                frame_time=hop / sr_target,
                freq_low=f0_min, freq_high=f0_max
            )[0].cpu().numpy()
        else:
            f0 = AF.detect_pitch_frequency(
                w_tensor, sample_rate=sr_target,
                frame_time=hop / sr_target,
                freq_low=f0_min, freq_high=f0_max
            )[0].numpy()
    except Exception:
        f0 = AF.detect_pitch_frequency(
            w_tensor, sample_rate=sr_target,
            frame_time=hop / sr_target,
            freq_low=f0_min, freq_high=f0_max
        )[0].numpy()

    if len(f0) < frames:
        f0 = np.pad(f0, (0, frames - len(f0)))
    else:
        f0 = f0[:frames]

    voiced = (f0 > f0_min)
    pitch = np.where(voiced, np.log(np.maximum(f0, 1.0) / pitch_ref), 0.0)

    # Energy (RMS)
    rms = librosa.feature.rms(y=y_pad, frame_length=n_fft, hop_length=hop, center=True, pad_mode="constant")[0, :frames]
    prosody = np.stack((pitch, voiced.astype(np.float32), np.log(np.maximum(rms, 1e-5))), axis=-1)

    return {
        "wave": torch.from_numpy(y_pad),
        "mel": torch.from_numpy(mel.astype(np.float32)),
        "prosody": torch.from_numpy(prosody.astype(np.float32)),
        "speaker": speaker_idx,
    }


def _extract_worker(item):
    wav_path_str, out_path_str, config = item
    out_path = Path(out_path_str)
    if out_path.exists():
        return True
    try:
        feats = extract_features(wav_path_str, config)
        if feats is None:
            return False
        torch.save(feats, out_path)
        return True
    except Exception as e:
        print(f"Error {wav_path_str}: {e}", flush=True)
        return False


def compute_acoustic_fingerprint(config):
    """Match tiny_voca.config.Config.acoustic_fingerprint."""
    fields = ("sample_rate", "hop", "n_fft", "n_mels", "f0_min", "f0_max")
    ac = {**{key: config[key] for key in fields}, "feature_version": 2,
          "pitch_reference_hz": 200, "log_floor": 1e-5}
    return hashlib.sha256(json.dumps(ac, sort_keys=True).encode()).hexdigest()


def compute_frontend_fingerprint():
    """Match tiny_voca.text.frontend_fingerprint for the installed front end."""
    try:
        from tiny_voca.text import frontend_fingerprint
        return frontend_fingerprint()
    except ImportError:
        return None


def patch_annotations(ann_dir, acoustic_root, acoustic_fp, frontend_fp, annotation_fp):
    """Update annotations.json and each row in train/validation/test.jsonl with
    the fingerprints tiny_voca.data.AnnotatedDataset expects."""
    ann_dir = Path(ann_dir)

    # Write annotations.json
    meta = {
        "complete": True,
        "acoustic_root": str(acoustic_root),
        "fingerprint": annotation_fp,
    }
    with open(ann_dir / "annotations.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Patch each row in the split manifests
    for split in ("train", "validation", "test"):
        path = ann_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        with open(path, "w") as f:
            for row in rows:
                row["frontend"] = frontend_fp
                row["acoustic"] = acoustic_fp
                row["annotation"] = annotation_fp
                # Rewrite path to point to the .pt feature file
                stem = Path(row["path"]).stem
                row["path"] = f"{stem}.pt"
                # Add text_hash for leakage check
                if "text_hash" not in row:
                    row["text_hash"] = hashlib.sha256(row["text"].encode()).hexdigest()[:16]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return meta


def remap_checkpoint(upstream_ckpt, vocab_plan, output_path):
    """Load upstream checkpoint, remap symbol embeddings, save."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tui_voice.init_ckpt import remap_embeddings

    state = torch.load(upstream_ckpt, map_location="cpu", weights_only=False)
    plan = json.load(open(vocab_plan))
    init_from = plan["init_from"]

    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model_sd = state["model"]
    else:
        model_sd = state

    new_sd, changed = remap_embeddings(model_sd, init_from)
    print(f"Remapped {len(changed)} embedding tables: {changed}")

    if isinstance(state, dict) and "model" in state:
        state["model"] = new_sd
    else:
        state = new_sd

    torch.save(state, output_path)
    print(f"Saved remapped checkpoint to {output_path}")
    return output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True, help="Dir with train/validation/test.jsonl from 03_align.py")
    ap.add_argument("--wavs", required=True, help="Dir with .wav files (24 kHz)")
    ap.add_argument("--upstream-ckpt", default="weights/tiny-voca.pt", help="Upstream tiny-voca checkpoint")
    ap.add_argument("--vocab-plan", required=True, help="vocab_plan.json from 02_phonemize.py")
    ap.add_argument("--output", required=True, help="Training output directory")
    ap.add_argument("--features-dir", default="", help="Where to write .pt features (default: alongside wavs)")
    ap.add_argument("--config", default="weights/config.json", help="Upstream config.json")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--gan-start", type=int, default=8000)
    ap.add_argument("--batch-frames", type=int, default=9000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep-every", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-features", action="store_true", help="Skip feature extraction (already done)")
    ap.add_argument("--skip-remap", action="store_true", help="Skip checkpoint remapping (already done)")
    ap.add_argument("--dry-run", action="store_true", help="Prepare everything but don't launch training")
    args = ap.parse_args()

    ann_dir = Path(args.annotations)
    wav_dir = Path(args.wavs)
    feat_dir = Path(args.features_dir) if args.features_dir else wav_dir.parent / "features"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tiny_voca.config import Config
        model_cfg = Config()
        acoustic_fp = model_cfg.acoustic_fingerprint
    except Exception:
        cfg = json.load(open(args.config))
        model_cfg = cfg.get("model", cfg)
        acoustic_fp = compute_acoustic_fingerprint(model_cfg)

    # Only extract features for clips that are in the annotations
    valid_stems = set()
    for s in ("train", "validation", "test"):
        p = ann_dir / f"{s}.jsonl"
        if p.exists():
            for line in open(p):
                if line.strip():
                    try:
                        valid_stems.add(Path(json.loads(line)["path"]).stem)
                    except Exception:
                        pass

    # ── Step 1: Extract acoustic features ──
    if not args.skip_features:
        from concurrent.futures import ThreadPoolExecutor
        feat_dir.mkdir(parents=True, exist_ok=True)
        wavs = [wp for wp in sorted(wav_dir.glob("*.wav")) if wp.stem in valid_stems]
        n_workers = 16

        items = [(str(wp), str(feat_dir / f"{wp.stem}.pt"), model_cfg) for wp in wavs]
        pending = [it for it in items if not Path(it[1]).exists()]
        print(f"Extracting features for {len(wavs)} accepted clips -> {feat_dir} ({n_workers} threads, GPU accelerated)...", flush=True)
        print(f"  {len(items) - len(pending)} already cached, {len(pending)} to process.", flush=True)
        
        if pending:
            done_count = 0
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for res in ex.map(_extract_worker, pending):
                    done_count += 1
                    if done_count % 1000 == 0 or done_count == len(pending):
                        pct = done_count * 100 // len(pending)
                        print(f"  [{done_count}/{len(pending)}] features extracted ({pct}%)", flush=True)
        print(f"Feature extraction complete: {len(list(feat_dir.glob('*.pt')))} files", flush=True)

    # ── Step 2: Patch annotations ──
    frontend_fp = compute_frontend_fingerprint()
    if frontend_fp is None:
        print("WARNING: tiny_voca not importable; using placeholder frontend fingerprint", flush=True)
        frontend_fp = "placeholder_requires_new_data_flag"

    # Compute annotation fingerprint from the train manifest content
    train_path = ann_dir / "train.jsonl"
    ann_content = train_path.read_bytes() if train_path.exists() else b""
    annotation_fp = hashlib.sha256(ann_content).hexdigest()

    patch_annotations(ann_dir, str(feat_dir), acoustic_fp, frontend_fp, annotation_fp)
    print(f"Patched annotations in {ann_dir}", flush=True)

    # ── Step 3: Remap checkpoint ──
    remapped_ckpt = output_dir / "remapped.pt"
    if not args.skip_remap:
        remap_checkpoint(args.upstream_ckpt, args.vocab_plan, str(remapped_ckpt))
    elif not remapped_ckpt.exists():
        print(f"ERROR: --skip-remap but {remapped_ckpt} does not exist")
        sys.exit(1)

    # ── Step 4: Launch training ──
    train_cmd = [
        sys.executable, "-m", "tiny_voca.train",
        "--train", str(ann_dir / "train.jsonl"),
        "--validation", str(ann_dir / "validation.jsonl"),
        "--output", str(output_dir),
        "--init-from", str(remapped_ckpt),
        "--new-data",
        "--steps", str(args.steps),
        "--gan-start", str(args.gan_start),
        "--batch-frames", str(args.batch_frames),
        "--learning-rate", str(args.learning_rate),
        "--save-every", str(args.save_every),
        "--keep-every", str(args.keep_every),
        "--device", args.device,
    ]
    print("\n" + "=" * 60)
    print("TRAINING COMMAND:")
    print(" \\\n  ".join(train_cmd))
    print("=" * 60)

    if args.dry_run:
        print("\nDry run complete. Run the command above manually or remove --dry-run.")
        return

    import subprocess
    print("\nLaunching training...")
    result = subprocess.run(train_cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
