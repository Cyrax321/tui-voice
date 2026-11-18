# Tui Voice Mini (Malayalam) 🎙️

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Cyrax321%2Ftui--voice--mini--ml-blue)](https://huggingface.co/Cyrax321/tui-voice-mini-ml)
[![Status](https://img.shields.io/badge/Status-Research%20Preview%20(v0.1)-yellow.svg)](#current-status--known-limitations)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Parameters](https://img.shields.io/badge/Parameters-4.5M-orange.svg)](#architecture-specifications)
[![Model Size](https://img.shields.io/badge/Size-~10MB%20(FP16)-purple.svg)](#released-models--checkpoints)
[![Sample Rate](https://img.shields.io/badge/Audio-24kHz%20Mono-red.svg)](#specifications)

> [!WARNING]
> **Experimental Research Prototype (`v0.1-preview`):** This model is an early research proof-of-concept for ultra-compact, on-device Malayalam speech synthesis. It is **not a finished commercial voice**. While intelligible and extremely lightweight (~10 MB), it exhibits synthetic timbre artifacts inherent to low-parameter source-filter vocoding. See [Current Status & Known Limitations](#current-status--known-limitations) below.

**Tui Voice Mini Malayalam (`tui-voice-mini-ml`)** is an experimental, ultra-compact Malayalam Text-to-Speech (TTS) neural research model. Fine-tuned from [tiny-voca](https://github.com/n4ze3m/tiny-voca) on the [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) Malayalam studio corpus, it explores the lower limits of neural speech footprint with just **4.5 Million parameters (~10 MB footprint)**.

Unlike heavy TTS architectures (like VITS, FastPitch, or XTTS) requiring hundreds of megabytes and dedicated GPU servers, Tui Voice Mini runs **100% offline directly in web browsers (WebAssembly), mobile devices, and embedded systems like Raspberry Pi and robotics controllers**.

---

## 🚀 Released Models & Checkpoints

All model weights and checkpoints are hosted on the [Hugging Face Hub (`Cyrax321/tui-voice-mini-ml`)](https://huggingface.co/Cyrax321/tui-voice-mini-ml):

| Artifact | File Name | Size | Target Environment |
| :--- | :--- | :--- | :--- |
| **PyTorch FP32 (Main)** | `tui-voice-mini-ml.pt` | 18.2 MB | Python / PyTorch / Desktop AI stacks |
| **PyTorch FP16** | `tui-voice-mini-ml-fp16.pt` | 9.1 MB | High-speed GPU inference |
| **ONNX Runtime (Standard)** | `onnx/tui-voice-mini-ml.onnx` | 21.4 MB | Web browsers (WASM), Android, iOS, C++ |
| **Quantized INT8 ONNX** | `onnx/tui-voice-mini-ml-int8.onnx` | ~5.4 MB | **Raspberry Pi, ROS Robots, Edge microcontrollers** |
| **Vocabulary & Symbols** | `config.json` / `vocab_plan.json` | ~6 KB | 76-slot Malayalam phoneme budget mapping |
| **Training Checkpoints** | `checkpoints/tui-voice-mini-ml-step*.pt` | ~87 MB | Lineage snapshots from Step 5,000 to Step 40,000 |

---

## ⚡ Quickstart

### 1. Installation

```bash
git clone https://github.com/Cyrax321/tui-voice.git
cd tui-voice
pip install -r requirements.txt
```

*(Ensure `espeak-ng` is installed for Malayalam phonemization: `sudo apt-get install espeak-ng` or `brew install espeak-ng`)*

### 2. Synthesize Malayalam Speech

```bash
python scripts/07_infer_sample.py \
  --checkpoint model/tui-voice-mini-ml.pt \
  --vocab-plan data/vocab_plan.json \
  --text "നമസ്കാരം, ഇത് തുയി വോയ്സ് മിനി മലയാളം മോഡൽ ആണ്." \
  --voice 0 --pitch-shift 0.15 --pitch-range 0.85 \
  --output sample.wav
```

### 3. Batch Generate Showcase Clips

Generate the entire 20+ benchmark showcase audio suite (railway, metro, news, everyday conversations):

```bash
python scripts/generate_showcase.py \
  --checkpoint model/tui-voice-mini-ml.pt \
  --vocab-plan data/vocab_plan.json \
  --out-dir showcase_samples/ \
  --voice 0 --pitch-shift 0.15 --pitch-range 0.85
```

---

## 🧠 Architecture Specifications

| Specification | Details |
| :--- | :--- |
| **Total Deployed Parameters** | **4,537,291** (Includes complete waveform generator) |
| **Text Encoder** | 4-block Transformer, 192 channels, 4 heads, Rotary Position Embeddings (RoPE) |
| **Duration & Pause Predictor** | Multi-layer 1D Conv with log-duration & punctuation pause classification |
| **Prosody & F0 Predictor** | 2-block Frame Transformer predicting continuous pitch & voicing probability |
| **Waveform Synthesizer** | Harmonic pulse-train oscillator + filtered noise generator (Source-Filter) |
| **Neural Vocoder Needed?** | **None!** (Waveform is synthesized directly without external HiFi-GAN) |
| **Audio Sample Rate** | 24,000 Hz Mono (Studio standard) |
| **Phoneme Coverage** | Full 59 Malayalam phonemes (Vowels, consonants, conjuncts, chillus ൻ, ർ, ൽ, ൾ, ൺ, ൿ) |

---

## 📊 Research-Grade Benchmarking Suite

This repository includes a reproducible Malayalam TTS benchmark framework designed to evaluate intelligibility, prosody, and hardware efficiency:

* **Architecture Report:** [`benchmark/reports/architecture_inspection.md`](benchmark/reports/architecture_inspection.md)
* **Dataset Audit:** [`benchmark/reports/dataset_audit.md`](benchmark/reports/dataset_audit.md)
* **Evaluation Datasets (`benchmark/data/splits/`):**
  * `dev_set.jsonl`: 100 sentences for pipeline sanity checking
  * `main_test_set.jsonl`: 500 sentences balanced across 6 linguistic categories:
    1. Everyday Conversational (100)
    2. Formal / News / Informational (100)
    3. Long Sentences & Compound Words (75)
    4. Questions & Exclamations (75)
    5. Numbers, Dates, Times & Currency (75)
    6. Malayalam-English Code-Switching (75)
  * `stress_test_set.jsonl`: 200 sentences testing extreme conjuncts and tongue twisters
* **Evaluation Metrics:**
  * **Intelligibility:** Grapheme-cluster CER (`INT-01`), Word Error Rate (`INT-02`), Sentence Error Rate (`INT-03`), Word Omission/Repetition (`INT-04/05`)
  * **Audio Integrity:** Empty audio, clipping, non-finite checks (`AUD-01` to `AUD-07`)
  * **Prosody & Timing:** F0 RMSE, F0 correlation, voicing error, duration MAE, pause F1 (`PRO-01` to `PRO-09`)
  * **Inference Efficiency:** Parameters, memory footprint, cold/warm latency, Real-Time Factor (RTF) (`EFF-01` to `EFF-11`)
  * **Human Perceptual:** Naturalness MOS, Intelligibility MOS, Pairwise CMOS, and Preference Rates (`HUM-01` to `HUM-07`)

To generate or audit the benchmark splits:
```bash
python3 benchmark/scripts/prepare_dataset.py
```

---

## ⚠️ Current Status & Known Limitations

Tui Voice Mini is released as an **experimental research prototype (v0.1-preview)**, not a commercial production system:

1. **Synthetic Timbre & Vocoding:** To achieve a ~10 MB footprint and run without an external neural vocoder, Tui Voice Mini employs a lightweight source-filter harmonic oscillator + noise generator. While speech is intelligible, it produces an audible buzz/robotic timbre compared to 100M+ parameter vocoders (e.g. HiFi-GAN, BigVGAN).
2. **Single Speaker Profile:** The model is currently trained on a single Malayalam female speaker (`female_01` from AI4Bharat Rasa). Multi-speaker or male voices are not yet supported.
3. **Number & Symbol Verbalization:** Raw digits (e.g. `123`, `2026`) and English symbols (`%`, `$`) are not automatically expanded. All inputs must be converted into full Malayalam words (e.g. `നൂറ്റിയിരുപത്തിമൂന്ന്`) before synthesis.
4. **Prosody on Long Paragraphs:** Synthesizing lengthy compound texts without punctuation pauses can result in flat or monotonous intonation contours.
5. **Pending Benchmarking Verification:** Objective intelligibility metrics (CER/WER via IndicConformer) and human subjective listening evaluations (MOS/CMOS) are currently ongoing across the 500-sentence test split.

---

## 🤝 Citation & Attribution

If you use Tui Voice Mini Malayalam in your research or application, please cite:

```bibtex
@software{tuivoice2026malayalam,
  author = {Cyrax321},
  title = {Tui Voice Mini: Ultra-Compact 4.5M Parameter Malayalam Text-to-Speech},
  year = {2026},
  publisher = {Hugging Face / GitHub},
  url = {https://huggingface.co/Cyrax321/tui-voice-mini-ml}
}
```

* **Base Architecture:** Muhammed Nazeem ([tiny-voca](https://github.com/n4ze3m/tiny-voca))
* **Dataset:** AI4Bharat ([Rasa Malayalam Corpus](https://huggingface.co/datasets/ai4bharat/Rasa))
* **License:** [Apache-2.0](LICENSE)
