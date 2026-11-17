"""Package trained checkpoint, ONNX export, updated config, and model card,
then upload to Hugging Face Hub (e.g. Cyrax321/tui-voice-mini-ml).

Usage:
  python scripts/05_push_to_hf.py --repo-id Cyrax321/tui-voice-mini-ml \\
      --checkpoint WORK/checkpoints/best.pt \\
      --config weights/config.json \\
      --vocab-plan data/vocab_plan.json \\
      --out export
"""
import argparse
import json
import os
import shutil
from pathlib import Path


MODEL_CARD_TEMPLATE = """---
language:
- ml
tags:
- text-to-speech
- tts
- audio
- tiny-voca
license: apache-2.0
datasets:
- ai4bharat/Rasa
pipeline_tag: text-to-speech
---

# {model_name}

**{model_name}** is an ultra-compact (~10 MB) Malayalam Text-to-Speech model fine-tuned from [tiny-voca](https://github.com/n4ze3m/tiny-voca) using the [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) Malayalam dataset.

## Specifications
- **Base Architecture:** tiny-voca (FastSpeech 2 style encoder + duration predictor + source-filter waveform synthesizer)
- **Parameters:** ~4.5 million
- **Audio Output:** 24,000 Hz Mono
- **Size:** ~18 MB PyTorch / ~9 MB fp16 / ~20 MB ONNX
- **Front-end:** eSpeak NG Malayalam (`-v ml --ipa`)

## Usage
Synthesize Malayalam speech with `tiny-voca` or ONNX Runtime:
```python
# See https://github.com/Cyrax321/tui-voice for inference and training code
```

## Attribution & Licenses
- **Model Code & Weights:** Apache-2.0 (fine-tuned from [n4ze3m/tiny-voca](https://github.com/n4ze3m/tiny-voca))
- **Training Data:** [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) (CC-BY-4.0)
- **Aligner:** `facebook/wav2vec2-lv-60-espeak-cv-ft` (Apache-2.0)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="HF repo e.g. Cyrax321/tui-voice-mini-ml")
    ap.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    ap.add_argument("--config", default="weights/config.json", help="Base tiny-voca config.json")
    ap.add_argument("--vocab-plan", default="data/vocab_plan.json", help="Path to vocab_plan.json")
    ap.add_argument("--onnx", default="", help="Optional exported ONNX file")
    ap.add_argument("--out", default="export", help="Local staging directory")
    ap.add_argument("--private", action="store_true", help="Make HF repo private")
    ap.add_argument("--dry-run", action="store_true", help="Prepare export files without uploading")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Update config.json with Malayalam vocabulary mapping
    if Path(args.config).exists() and Path(args.vocab_plan).exists():
        cfg = json.load(open(args.config))
        plan = json.load(open(args.vocab_plan))
        new_symbols = plan.get("new_symbols")
        if new_symbols:
            cfg["vocab"] = {s: i for i, s in enumerate(new_symbols)}
            cfg["model"]["vocab_size"] = len(new_symbols)
        cfg["language"] = "ml"
        cfg["speaker_info"] = {"F2": "AI4Bharat Rasa Malayalam Female"}
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"Wrote updated config.json with {len(cfg.get('vocab', {}))} symbols")

    # 2. Copy checkpoint
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        shutil.copy2(ckpt_path, out_dir / "model.pt")
        print(f"Copied checkpoint to {out_dir / 'model.pt'}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found")

    # 3. Copy ONNX if present
    if args.onnx and Path(args.onnx).exists():
        (out_dir / "onnx").mkdir(exist_ok=True)
        shutil.copy2(args.onnx, out_dir / "onnx" / "model.onnx")
        print(f"Copied ONNX model to {out_dir / 'onnx' / 'model.onnx'}")

    # 4. Generate README.md (Model Card)
    model_name = args.repo_id.split("/")[-1]
    readme_content = MODEL_CARD_TEMPLATE.format(model_name=model_name)
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Generated model card at {out_dir / 'README.md'}")

    # 5. Upload to Hugging Face
    if args.dry_run:
        print("Dry run complete: export staged at", out_dir)
        return

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    print(f"Creating / verifying Hugging Face repository {args.repo_id}...")
    create_repo(args.repo_id, exist_ok=True, repo_type="model", private=args.private)
    print(f"Uploading {out_dir} to https://huggingface.co/{args.repo_id}...")
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload {model_name} weights and artifacts"
    )
    print(f"Successfully uploaded model to https://huggingface.co/{args.repo_id}!")


if __name__ == "__main__":
    main()
