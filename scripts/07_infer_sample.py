"""Synthesize a sample Malayalam audio clip from a trained checkpoint or inference.pt.

Usage:
  python scripts/07_infer_sample.py \\
    --checkpoint /content/drive/MyDrive/tui_voice/run1/last.pt \\
    --vocab-plan /content/data/vocab_plan.json \\
    --text "നമസ്കാരം, ഞാൻ തുയി വോയ്സ് ആണ്." \\
    --output sample.wav
"""
import argparse
import json
import os
import sys
from pathlib import Path
import soundfile as sf
import numpy as np
import torch

# Ensure paths
sys.path.insert(0, "/content/tiny-voca/src")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tui_voice import phonemes as P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/tui_voice/run1/last.pt")
    parser.add_argument("--vocab-plan", default="/content/data/vocab_plan.json")
    parser.add_argument("--config", default="/content/tiny-voca/weights/config.json")
    parser.add_argument("--text", default="നമസ്കാരം, ഞാൻ തുയി വോയ്സ് ആണ്.")
    parser.add_argument("--output", default="sample.wav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--voice", type=int, default=0, help="Speaker index (0 for Female F2, 1 for Male M2)")
    parser.add_argument("--pitch-shift", type=float, default=0.0, help="Raise/lower pitch (e.g. 0.15 to keep voice cleanly female)")
    parser.add_argument("--pitch-range", type=float, default=1.0, help="Scale pitch movement (0.85-0.9 makes voice more steady)")
    args = parser.parse_args()

    from tiny_voca.config import Config
    from tiny_voca.model import TinyVoca

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        alt = ckpt_path.parent / "inference.pt"
        if alt.exists():
            ckpt_path = alt
        else:
            alt_last = ckpt_path.parent / "last.pt"
            if alt_last.exists():
                ckpt_path = alt_last

    print(f"Loading checkpoint from {ckpt_path}...", flush=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Config
    cfg_dict = state.get("config")
    if not cfg_dict and Path(args.config).exists():
        cfg_dict = json.load(open(args.config))
    cfg = Config.from_dict(cfg_dict) if cfg_dict else Config()
    model = TinyVoca(cfg)

    # Extract state dict
    if "model" in state and isinstance(state["model"], dict):
        sd = state["model"]
    elif "system" in state and isinstance(state["system"], dict):
        sd = {k[6:]: v for k, v in state["system"].items() if k.startswith("model.")}
        if not sd:
            sd = state["system"]
    else:
        sd = state

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Notice: {len(missing)} missing keys in model (e.g. vocoder cache)")
    model.eval().to(args.device)

    # Load vocab plan
    plan = json.load(open(args.vocab_plan))
    
    # Try resolving upstream symbols from weights or tiny_voca
    try:
        from tiny_voca.text import SYMBOLS
        upstream_symbols = list(SYMBOLS)
    except Exception:
        upstream_symbols = plan.get("new_symbols", [])

    # Split text into sentences for natural long-form reading
    import re
    chunks = re.split(r'([.?!।\n]+)', args.text)
    sentences = []
    curr = ""
    for c in chunks:
        curr += c
        if re.search(r'[.?!।\n]', c) and any(ch.isalnum() for ch in curr):
            sentences.append(curr.strip())
            curr = ""
    if curr.strip() and any(ch.isalnum() for ch in curr):
        sentences.append(curr.strip())
    if not sentences:
        sentences = [args.text.strip()]

    print(f"\nText ({len(sentences)} sentence(s)): {args.text}", flush=True)

    all_waves = []
    pause_samples = int(cfg.sample_rate * 0.18)  # 180ms natural pause between sentences
    silence = torch.zeros(pause_samples, device=args.device)

    for idx, sent in enumerate(sentences):
        tokens_str = P.phonemize(sent)
        token_ids = P.encode(tokens_str, plan, upstream_symbols)
        if len(token_ids) > cfg.max_tokens - 10:
            token_ids = token_ids[:cfg.max_tokens - 10]

        tok_tensor = torch.tensor([token_ids], dtype=torch.long, device=args.device)
        stress = torch.zeros_like(tok_tensor)
        lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=args.device)
        speakers = torch.tensor([args.voice], dtype=torch.long, device=args.device)

        with torch.inference_mode():
            wave, _, _ = model.synthesize(
                tok_tensor, stress, lengths, speakers,
                speed=args.speed,
                pitch_shift=args.pitch_shift,
                pitch_range=args.pitch_range,
                voiced_vowels=True
            )
        w = wave.squeeze(0)
        all_waves.append(w)
        if idx < len(sentences) - 1:
            all_waves.append(silence)

    full_wave = torch.cat(all_waves)
    audio = full_wave.cpu().numpy()

    # Normalize peak if loud
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 0.99:
        audio = audio * (0.95 / peak)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, cfg.sample_rate)
    print(f"SUCCESS: Saved {len(audio)/cfg.sample_rate:.2f}s audio ({len(sentences)} sentences) to {out_path} (rate: {cfg.sample_rate} Hz, peak: {peak:.2f})", flush=True)


if __name__ == "__main__":
    main()
