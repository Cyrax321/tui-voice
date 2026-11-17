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


def extract_features(wav_path, config):
    """Replicate tiny_voca.prepare.extract: wav -> (mel, prosody, wave) tensors."""
    import librosa
    import torch

    y, sr = sf.read(wav_path, dtype="float32")
    if sr != config["sample_rate"]:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=config["sample_rate"])

    hop = config["hop"]
    frames = math.ceil(len(y) / hop)
    if frames < 1:
        return None

    # Pad to exact frame count
    y = np.pad(y, (0, frames * hop - len(y))).astype(np.float32)

    # Mel spectrogram
    spectrum = librosa.stft(y, n_fft=config["n_fft"], hop_length=hop,
                            center=True, pad_mode="constant")[:, :frames]
    filters = librosa.filters.mel(sr=config["sample_rate"], n_fft=config["n_fft"],
                                  n_mels=config["n_mels"])
    mel = np.log(np.maximum(filters @ np.abs(spectrum) ** 2, 1e-5)).T

    # Pitch and voicing via pyin
    f0, voiced, _ = librosa.pyin(
        y, fmin=config["f0_min"], fmax=config["f0_max"],
        sr=config["sample_rate"], frame_length=config["n_fft"],
        hop_length=hop, center=True, pad_mode="constant"
    )
    f0 = f0[:frames]
    voiced = voiced[:frames]
    f0 = np.nan_to_num(f0, nan=config.get("pitch_reference_hz", 200.0))
    log_pitch = np.log(np.maximum(f0, 1e-5) / config.get("pitch_reference_hz", 200.0))
    voiced_float = voiced.astype(np.float32)

    # Energy (log RMS per frame)
    energy = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        c = i * hop
        seg = y[max(0, c - hop):c + hop]
        energy[i] = np.log(np.sqrt(np.mean(seg ** 2)) + 1e-9)

    prosody = np.stack([log_pitch, voiced_float, energy], axis=-1).astype(np.float32)
    wave = torch.from_numpy(y)
    mel_t = torch.from_numpy(mel.astype(np.float32))
    prosody_t = torch.from_numpy(prosody)

    return {"mel": mel_t, "prosody": prosody_t, "wave": wave}


def _extract_worker(item):
    wav_path_str, out_path_str, config = item
    out_path = Path(out_path_str)
    if out_path.exists():
        return True
    feats = extract_features(wav_path_str, config)
    if feats is None:
        return False
    import torch
    torch.save(feats, out_path)
    return True


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

    cfg = json.load(open(args.config))
    model_cfg = cfg.get("model", cfg)
    acoustic_fp = compute_acoustic_fingerprint(model_cfg)

    # ── Step 1: Extract acoustic features ──
    if not args.skip_features:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        feat_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(wav_dir.glob("*.wav"))
        n_workers = min(multiprocessing.cpu_count() or 4, 8)
        print(f"Extracting features for {len(wavs)} wav files -> {feat_dir} ({n_workers} CPU workers)...")

        items = [(str(wp), str(feat_dir / f"{wp.stem}.pt"), model_cfg) for wp in wavs]
        # Filter already existing
        pending = [it for it in items if not Path(it[1]).exists()]
        print(f"  {len(items) - len(pending)} already cached, {len(pending)} to process.")
        
        if pending:
            done_count = 0
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                for res in ex.map(_extract_worker, pending, chunksize=16):
                    done_count += 1
                    if done_count % 1000 == 0 or done_count == len(pending):
                        print(f"  [{done_count}/{len(pending)}] features extracted ({done_count * 100 // len(pending)}%)")
        print(f"Feature extraction complete: {len(list(feat_dir.glob('*.pt')))} files")

    # ── Step 2: Patch annotations ──
    frontend_fp = compute_frontend_fingerprint()
    if frontend_fp is None:
        print("WARNING: tiny_voca not importable; using placeholder frontend fingerprint")
        frontend_fp = "placeholder_requires_new_data_flag"

    # Compute annotation fingerprint from the train manifest content
    train_path = ann_dir / "train.jsonl"
    ann_content = train_path.read_bytes() if train_path.exists() else b""
    annotation_fp = hashlib.sha256(ann_content).hexdigest()

    patch_annotations(ann_dir, str(feat_dir), acoustic_fp, frontend_fp, annotation_fp)
    print(f"Patched annotations in {ann_dir}")

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
