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
    parser.add_argument("--voice", type=int, default=0, help="Speaker index (0 or 1)")
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

    print(f"\nText: {args.text}", flush=True)
    tokens_str = P.phonemize(args.text)
    print(f"Phonemes ({len(tokens_str)}): {' '.join(tokens_str)}", flush=True)
    token_ids = P.encode(tokens_str, plan, upstream_symbols)

    tok_tensor = torch.tensor([token_ids], dtype=torch.long, device=args.device)
    stress = torch.zeros_like(tok_tensor)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=args.device)
    speakers = torch.tensor([args.voice], dtype=torch.long, device=args.device)

    print("Synthesizing audio...", flush=True)
    with torch.inference_mode():
        wave, _, _ = model.synthesize(
            tok_tensor, stress, lengths, speakers,
            speed=args.speed,
            voiced_vowels=True
        )

    audio = wave.squeeze(0).cpu().numpy()
    # Normalize peak if loud
    peak = float(torch.from_numpy(audio).abs().max())
    if peak > 0.99:
        audio = audio * (0.95 / peak)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, cfg.sample_rate)
    print(f"SUCCESS: Saved {len(audio)/cfg.sample_rate:.2f}s audio to {out_path} (sample rate: {cfg.sample_rate} Hz, peak: {peak:.2f})", flush=True)


if __name__ == "__main__":
    main()
