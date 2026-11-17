"""Batch-generate standard showcase audio clips using examples/showcase_prompts.json.

Usage:
  python scripts/generate_showcase.py \
    --checkpoint /content/drive/MyDrive/tui_voice/run1/inference.pt \
    --vocab-plan /content/data/vocab_plan.json \
    --out-dir /content/showcase_samples \
    --voice 0 --pitch-shift 0.15 --pitch-range 0.85
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


def synthesize_text(model, text, plan, upstream_symbols, cfg, device="cuda", speed=1.0, voice=0, pitch_shift=0.15, pitch_range=0.85):
    import re
    chunks = re.split(r'([.?!।\n]+)', text)
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
        sentences = [text.strip()]

    all_waves = []
    pause_samples = int(cfg.sample_rate * 0.18)
    silence = torch.zeros(pause_samples, device=device)

    for idx, sent in enumerate(sentences):
        tokens_str = P.phonemize(sent)
        token_ids = P.encode(tokens_str, plan, upstream_symbols)
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
        w = wave.squeeze(0)
        all_waves.append(w)
        if idx < len(sentences) - 1:
            all_waves.append(silence)

    full_wave = torch.cat(all_waves)
    audio = full_wave.cpu().numpy()

    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 0.99:
        audio = audio * (0.95 / peak)

    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/tui_voice/run1/last.pt")
    parser.add_argument("--vocab-plan", default="/content/data/vocab_plan.json")
    parser.add_argument("--config", default="/content/tiny-voca/weights/config.json")
    parser.add_argument("--prompts", default="examples/showcase_prompts.json")
    parser.add_argument("--out-dir", default="/content/showcase_samples")
    parser.add_argument("--voice", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pitch-shift", type=float, default=0.15)
    parser.add_argument("--pitch-range", type=float, default=0.85)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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

    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg_dict = state.get("config")
    if not cfg_dict and Path(args.config).exists():
        cfg_dict = json.load(open(args.config))
    cfg = Config.from_dict(cfg_dict) if cfg_dict else Config()
    model = TinyVoca(cfg)

    if "model" in state and isinstance(state["model"], dict):
        sd = state["model"]
    elif "system" in state and isinstance(state["system"], dict):
        sd = {k[6:]: v for k, v in state["system"].items() if k.startswith("model.")}
        if not sd:
            sd = state["system"]
    else:
        sd = state

    model.load_state_dict(sd, strict=False)
    model.eval().to(args.device)

    plan = json.load(open(args.vocab_plan))
    try:
        from tiny_voca.text import SYMBOLS
        upstream_symbols = list(SYMBOLS)
    except Exception:
        upstream_symbols = plan.get("new_symbols", [])

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        prompts_path = Path(__file__).resolve().parent.parent / "examples" / "showcase_prompts.json"
    
    categories = json.load(open(prompts_path))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSynthesizing showcase audio clips into: {out_dir.resolve()}\n" + "="*60)
    count = 0
    for cat_name, items in categories.items():
        print(f"\n[{cat_name.upper()}]")
        for item in items:
            p_id = item["id"]
            title = item["title"]
            text_ml = item["malayalam"]
            
            wav_path = out_dir / f"{cat_name}_{p_id}.wav"
            print(f"  -> Generating: {title} ({p_id})... ", end="", flush=True)
            wav_data = synthesize_text(
                model=model,
                text=text_ml,
                plan=plan,
                upstream_symbols=upstream_symbols,
                cfg=cfg,
                device=args.device,
                speed=args.speed,
                voice=args.voice,
                pitch_shift=args.pitch_shift,
                pitch_range=args.pitch_range,
            )
            sf.write(str(wav_path), wav_data, cfg.sample_rate)
            print(f"Saved! ({len(wav_data)/cfg.sample_rate:.2f}s)")
            count += 1

    print("\n" + "="*60)
    print(f"SUCCESS: Generated {count} showcase audio files in {out_dir.resolve()}!")


if __name__ == "__main__":
    main()
