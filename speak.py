#!/usr/bin/env python3
"""Local Text-to-Speech CLI for Tui Voice Mini Malayalam.

Synthesizes Malayalam text and plays it directly through your Mac speakers!

Usage:
  python speak.py "നമസ്കാരം, സുഖമാണോ?"
  python speak.py "Endhokke und vishesham, sugam thanneyalle?"
  python speak.py "ഇന്ന് നല്ല മഴയുണ്ട്." --speed 0.9 --pitch-shift 0.20
"""
import argparse
import os
import sys
import json
import re
import subprocess
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import torch

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "tiny-voca" / "src"))
sys.path.insert(0, str(ROOT_DIR))

from tiny_voca.config import Config
from tiny_voca.model import TinyVoca
from tui_voice import phonemes as P


def load_model(model_dir: str = "model", device: str = "cpu"):
    m_dir = Path(model_dir)
    cfg_path = m_dir / "config.json"
    ckpt_path = m_dir / "tui-voice-mini-ml.pt"
    plan_path = m_dir / "vocab_plan.json"

    raw_cfg = json.load(open(cfg_path))
    model_cfg_dict = raw_cfg.get("model", raw_cfg)
    cfg = Config.from_dict(model_cfg_dict)
    
    model = TinyVoca(cfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("model", state)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    plan = json.load(open(plan_path))
    try:
        from tiny_voca.text import SYMBOLS
        upstream_symbols = list(SYMBOLS)
    except Exception:
        upstream_symbols = plan.get("new_symbols", [])

    return model, cfg, plan, upstream_symbols


def synthesize_speech(
    text: str,
    model,
    cfg,
    plan,
    upstream_symbols,
    speed: float = 1.0,
    pitch_shift: float = 0.15,
    pitch_range: float = 0.85,
    voice: int = 0,
    device: str = "cpu"
) -> np.ndarray:
    chunks = re.split(r'([.?!।\n]+)', text)
    sentences = [c.strip() for c in chunks if c.strip() and any(ch.isalnum() for ch in c)]
    if not sentences:
        sentences = [text.strip()]

    all_waves = []
    pause_samples = int(cfg.sample_rate * 0.18)
    silence = torch.zeros(pause_samples, device=device)

    for idx, sent in enumerate(sentences):
        tokens = P.phonemize(sent)
        token_ids = P.encode(tokens, plan, upstream_symbols)
        if len(token_ids) > cfg.max_tokens - 10:
            token_ids = token_ids[:cfg.max_tokens - 10]

        tok_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        stress = torch.zeros_like(tok_tensor)
        lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
        speakers = torch.tensor([voice], dtype=torch.long, device=device)

        with torch.inference_mode():
            wave, _, _ = model.synthesize(
                tok_tensor, stress, lengths, speakers,
                speed=speed,
                pitch_shift=pitch_shift,
                pitch_range=pitch_range,
                voiced_vowels=True
            )
        all_waves.append(wave.squeeze(0))
        if idx < len(sentences) - 1:
            all_waves.append(silence)

    full_wave = torch.cat(all_waves).cpu().numpy()
    peak = np.abs(full_wave).max()
    if peak > 0:
        full_wave = full_wave / peak * 0.95

    return full_wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="നമസ്കാരം! ഇത് നിങ്ങളുടെ കമ്പ്യൂട്ടറിൽ പ്രവർത്തിക്കുന്ന തുയി വോയ്സ് ആണ്.")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed (0.8 = slower, 1.2 = faster)")
    parser.add_argument("--pitch-shift", type=float, default=0.15, help="Pitch shift (e.g. 0.15 to keep voice female)")
    parser.add_argument("--pitch-range", type=float, default=0.85, help="Pitch variance scale")
    parser.add_argument("--no-play", action="store_true", help="Don't auto-play through speakers")
    parser.add_argument("--interactive", action="store_true", help="Start interactive typing mode")
    args = parser.parse_args()

    print("Loading Tui Voice Mini Malayalam (4.5M)...", flush=True)
    model, cfg, plan, upstream_symbols = load_model("model")
    print("Model ready!\n" + "=" * 50)

    if args.interactive:
        print("🎙️ Interactive Mode: Type any Malayalam sentence and press Enter (or 'exit' to quit):")
        while True:
            try:
                user_text = input("\nMalayalam > ").strip()
                if not user_text or user_text.lower() in ("exit", "quit"):
                    break
                audio = synthesize_speech(
                    user_text, model, cfg, plan, upstream_symbols,
                    speed=args.speed, pitch_shift=args.pitch_shift, pitch_range=args.pitch_range
                )
                wav_int16 = (audio * 32767).astype(np.int16)
                wavfile.write("output.wav", 24000, wav_int16)
                print(f"Synthesized ({len(audio)/24000:.2f}s) -> Playing...")
                subprocess.run(["afplay", "output.wav"])
            except KeyboardInterrupt:
                break
        print("\nGoodbye!")
        return

    print(f"Synthesizing: {args.text}")
    audio = synthesize_speech(
        args.text, model, cfg, plan, upstream_symbols,
        speed=args.speed, pitch_shift=args.pitch_shift, pitch_range=args.pitch_range
    )

    wav_int16 = (audio * 32767).astype(np.int16)
    out_path = Path(args.output)
    wavfile.write(str(out_path), 24000, wav_int16)
    print(f"Saved {len(audio)/24000:.2f}s to {out_path}!")

    if not args.no_play and sys.platform == "darwin":
        print("🔊 Playing through your Mac speakers...")
        subprocess.run(["afplay", str(out_path)])


if __name__ == "__main__":
    main()
