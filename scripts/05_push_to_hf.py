"""Package trained checkpoint, ONNX export, updated config, and model card,
then upload to Hugging Face Hub (e.g. Cyrax321/tui-voice-mini-ml).

Usage:
  python scripts/05_push_to_hf.py --repo-id Cyrax321/tui-voice-mini-ml \\
      --checkpoint WORK/run1/last.pt \\
      --config weights/config.json \\
      --vocab-plan data/vocab_plan.json \\
      --out export
"""
import argparse
import hashlib
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
- onnx
- tiny-voca
- malayalam
license: apache-2.0
datasets:
- ai4bharat/Rasa
pipeline_tag: text-to-speech
---

# {model_name}

**{model_name}** is an ultra-compact Malayalam text-to-speech model fine-tuned from [tiny-voca](https://github.com/n4ze3m/tiny-voca) using studio recordings from the [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) dataset. The complete model, including the source-filter waveform generator, has **4.5M parameters** (~18 MB PyTorch float32, ~9 MB fp16, ~21 MB ONNX) and generates 24 kHz mono audio.

- **Repository:** https://github.com/Cyrax321/tui-voice
- **Base Architecture:** [n4ze3m/tiny-voca](https://github.com/n4ze3m/tiny-voca)
- **Dataset:** [ai4bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa)
- **License:** Apache-2.0

---

## Specifications

| Specification | Value |
| :--- | :--- |
| **Parameters** | 4,537,291 |
| **Voices** | Malayalam Female (`F2` slot) |
| **Sample Rate** | 24 kHz mono |
| **Input** | Text, converted to phonemes via eSpeak NG (`-v ml --ipa`) |
| **Waveform Head** | Source-filter synthesizer (25,862 params; no heavy neural vocoder) |
| **Pretrained Base** | `n4ze3m/tiny-voca` |
| **License** | Apache-2.0 |

---

## Files

| File | Size | Contents |
| :--- | :--- | :--- |
| `tiny-voca.pt` (or `model.pt`) | ~18.2 MB | PyTorch weights, float32 |
| `onnx/model.onnx` | ~21.4 MB | ONNX export, float32, opset 17 |
| `config.json` | ~2 KB | Model settings, Malayalam phoneme vocabulary |
| `vocab_plan.json` | ~4 KB | Upstream symbol slot remapping records |

---

## Samples

| Text (Malayalam) | Voice |
| :--- | :--- |
| നമസ്കാരം! ഇത് ടൂയി വോയ്സ് മിനി മലയാളം ആണ്. | F2 |
| ഇന്ന് നല്ല മഴ പെയ്യുന്നുണ്ട്, കാലാവസ്ഥ വളരെ മനോഹരമാണ്. | F2 |
| നിങ്ങൾക്കെല്ലാവർക്കും ഈ പ്രൊജക്റ്റിലേക്ക് സ്വാഗതം. | F2 |
| കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ തീരത്ത് സ്ഥിതി ചെയ്യുന്ന ഒരു സംസ്ഥാനമാണ്. | F2 |

---

## Usage

### PyTorch

```bash
# Clone the tui-voice repository
git clone https://github.com/Cyrax321/tui-voice.git
cd tui-voice
pip install -r requirements.txt

# Download model from Hugging Face
hf download {repo_id} --local-dir model/
```

### ONNX Runtime (Python / WebAssembly)

The ONNX graph takes phoneme token IDs and stress values and returns the waveform. Text is normalized and converted to phoneme IDs using the `tui_voice` front-end (mappings in `config.json`).

```python
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tui_voice import phonemes as P, text as T

# 1. Load config and vocabulary
cfg = json.load(open("model/config.json"))
vocab = cfg["vocab"]

# 2. Text normalization and phonemization
text = "നമസ്കാരം! ഇത് ടൂയി വോയ്സ് മിനി മലയാളം ആണ്."
cleaned = T.normalize(text)
tokens = P.phonemize(cleaned)
token_ids = [vocab[t] for t in tokens if t in vocab]

# 3. Run ONNX session
session = ort.InferenceSession("model/onnx/model.onnx")
one = lambda value: np.array([value], dtype=np.float32)

audio, durations = session.run(None, {{
    "tokens": np.array([token_ids], dtype=np.int64),
    "stress": np.zeros((1, len(token_ids)), dtype=np.int64),
    "voice": np.array([0], dtype=np.int64),  # 0 = F2 (Malayalam female)
    "speed": one(1.0),
    "pitch_shift": one(0.0),
    "pitch_range": one(1.0),
}})

# 4. Save 24 kHz wav
audio = audio[0]
audio = audio * min(1.0, 0.99 / (np.abs(audio).max() + 1e-9))
sf.write("output.wav", audio, 24000)
```

### ONNX Graph Inputs & Outputs

| Input | Type | Shape | Meaning |
| :--- | :--- | :--- | :--- |
| `tokens` | int64 | `[1, N]` | Phoneme IDs, including `<bos>` and `<eos>` |
| `stress` | int64 | `[1, N]` | 0 none, 1 primary, 2 secondary |
| `voice` | int64 | `[1]` | 0 = F2 |
| `speed` | float32 | `[1]` | 1.0 as predicted; larger is faster |
| `pitch_shift` | float32 | `[1]` | Added to log F0 |
| `pitch_range` | float32 | `[1]` | Scales pitch movement |

| Output | Type | Shape | Meaning |
| :--- | :--- | :--- | :--- |
| `audio` | float32 | `[1, samples]` | Waveform at 24 kHz |
| `durations` | int64 | `[1, N]` | Length of each token in frames of 256 samples (10.7 ms) |

---

## Model Architecture

Text is converted to phonemes with a stress value for each. A transformer encoder (4 blocks) reads the phonemes, a duration predictor gives each one a length in frames, and two frame-level transformers (2 and 5 blocks) predict pitch, voicing, energy, and parameters for the waveform generator. All blocks use 192 channels, 4 heads, and rotary position embeddings.

The waveform is produced by a source-filter synthesizer instead of a neural vocoder. A band-limited pulse train at the predicted pitch and a noise signal are each shaped by a time-varying filter, whose complex cepstrum the network predicts once per frame (256 samples, 10.7 ms).

| Component | Parameters |
| :--- | :--- |
| Embeddings (phoneme, stress, voice) | 16,704 |
| Text encoder | 1,588,864 |
| Duration and pause predictor | 123,906 |
| Frame transformer A, prosody head | 795,971 |
| Frame transformer B | 1,985,984 |
| Source-filter waveform head | 25,862 |
| **Total** | **4,537,291** |

---

## Training Details

- **Base Model:** [n4ze3m/tiny-voca](https://github.com/n4ze3m/tiny-voca) (pretrained on 104,737 steps)
- **Fine-tuning Dataset:** [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) Malayalam female studio recordings (24 kHz, ~16,000 utterances)
- **Phoneme Alignment:** Forced alignment with `facebook/wav2vec2-lv-60-espeak-cv-ft`
- **Losses:** Multi-resolution STFT & mel spectral losses; adversarial loss with multi-period and multi-resolution discriminators after step 8,000

---

## Limitations

- **Language:** Malayalam only.
- **Numbers and Latin Text:** Numbers and English words should be verbalized prior to input in v0.
- **Single Speaker:** Single studio female voice (`F2`).
- **Acoustic Characteristics:** The source-filter synthesizer provides extreme CPU/browser speed and a ~10 MB footprint, but sounds slightly synthetic compared to large diffusion models.

---

## Citation & Attribution

```bibtex
@software{{tuivoice2026malayalam,
  author = {{Cyrax321}},
  title = {{{model_name}: Ultra-compact 4.5M Malayalam Text-to-Speech}},
  year = {{2026}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

- Base architecture and code by Muhammed Nazeem: [tiny-voca](https://github.com/n4ze3m/tiny-voca) (Apache-2.0).
- Speech dataset provided by AI4Bharat: [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) (CC-BY-4.0).
"""


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        # Also copy vocab_plan.json into export
        shutil.copy2(args.vocab_plan, out_dir / "vocab_plan.json")

    # 2. Copy checkpoint
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        shutil.copy2(ckpt_path, out_dir / "tiny-voca.pt")
        print(f"Copied checkpoint to {out_dir / 'tiny-voca.pt'}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found")

    # 3. Copy ONNX if present
    if args.onnx and Path(args.onnx).exists():
        (out_dir / "onnx").mkdir(exist_ok=True)
        shutil.copy2(args.onnx, out_dir / "onnx" / "model.onnx")
        print(f"Copied ONNX model to {out_dir / 'onnx' / 'model.onnx'}")

    # 4. Generate README.md (Detailed Model Card)
    model_name = args.repo_id.split("/")[-1]
    readme_content = MODEL_CARD_TEMPLATE.format(model_name=model_name, repo_id=args.repo_id)
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Generated comprehensive model card at {out_dir / 'README.md'}")

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
        commit_message=f"Upload {model_name} weights, config, and documentation"
    )
    print(f"Successfully uploaded model to https://huggingface.co/{args.repo_id}!")


if __name__ == "__main__":
    main()
