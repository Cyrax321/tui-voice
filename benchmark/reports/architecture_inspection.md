# Architecture Inspection Report: Tui Voice Mini (Malayalam)

**Document ID:** TUI-ARCH-2026-01  
**Date:** 2026-11-18  
**Model Name:** Tui Voice Mini Malayalam (`tui-voice-mini-ml`)  
**Base Architecture:** `n4ze3m/tiny-voca`  
**Fine-Tuning Dataset:** AI4Bharat Rasa Malayalam Studio Corpus (Female `F2` voice)  
**Evaluator:** Senior Speech Research & Systems Engineering Team  

---

## 1. Executive Summary

Tui Voice Mini is an ultra-compact, source-filter neural Text-to-Speech (TTS) architecture designed for zero-server on-device deployment (WebAssembly, Android/iOS, and embedded ARM systems). While conventional Malayalam TTS systems (e.g., AI4Bharat Indic-TTS FastPitch+HiFi-GAN at ~110M parameters or Meta MMS-TTS VITS at ~36M parameters) require large storage and high memory footprints, Tui Voice Mini achieves complete waveform generation with only **4,537,291 parameters** (~18.2 MB PyTorch float32, ~9.1 MB float16, and ~21.4 MB ONNX).

Crucially, **the reported 4.5M parameters include the complete waveform synthesis engine**. Unlike two-stage acoustic models that require an auxiliary 20M–60M parameter neural vocoder (such as HiFi-GAN or Vocos), Tui Voice Mini integrates a direct source-filter harmonic oscillator directly inside its frame decoder, requiring zero external vocoder at inference time.

---

## 2. Model Architecture Breakdown

### 2.1 Complete Component Hierarchy

| Module | Component Name | Architecture Details | Parameters | % of Model |
| :--- | :--- | :--- | :--- | :--- |
| **Embeddings** | `phoneme_embed`, `stress_embed`, `voice_embed` | 76 symbol slots (192-dim), stress levels, voice identity | 16,704 | 0.37% |
| **Text Encoder** | `encoder` | 4 Transformer blocks, 192 channels, 4 attention heads, RoPE (Rotary Position Embeddings), SwiGLU FFN | 1,588,864 | 35.02% |
| **Duration Predictor** | `duration_head`, `pause_head` | 2-layer 1D Conv / Dense with logarithmic duration & pause probability outputs | 123,906 | 2.73% |
| **Frame Transformer A** | `frame_transformer_a` + `f0_head` | 2 Transformer blocks (192-dim, 4 heads) + pitch (F0), voicing, and energy heads | 795,971 | 17.54% |
| **Frame Transformer B** | `frame_transformer_b` | 5 Transformer blocks (192-dim, 4 heads) predicting time-varying cepstral filter coefficients | 1,985,984 | 43.77% |
| **Source-Filter Waveform Head** | `synth_head` | Linear filter synthesis (band-limited impulse train + filtered noise generator) | 25,862 | 0.57% |
| **Total Deployed Parameters** | **All Inference Modules** | **Complete end-to-end speech synthesis pipeline** | **4,537,291** | **100.0%** |

### 2.2 Trainable vs. Non-Trainable Parameters

* **Total Model Parameters:** `4,537,291`
* **Trainable Parameters during Fine-Tuning:** `4,537,291` (all weights unfrozen across Stage A and Stage B)
* **Auxiliary Training Components (Not deployed):**
  * Multi-Period Discriminator (MPD): ~3.2M parameters
  * Multi-Resolution Spectrogram Discriminator (MRD): ~4.1M parameters
  * Total training-time parameters (Generator + Discriminators): ~11.8M parameters.
  * *Note:* All discriminator networks are discarded upon export to `inference.pt` and `onnx/model.onnx`.

---

## 3. Acoustic & Waveform Generation Pipeline

Unlike conventional neural TTS pipelines that predict intermediate 80-channel mel-spectrograms and delegate time-domain inversion to a heavy convolutional neural vocoder, Tui Voice Mini employs a classic acoustic source-filter model re-imagined with modern deep learning:

1. **Source Generation:**
   * A voiced impulse train is generated at the fundamental frequency ($F_0$) predicted by `frame_transformer_a`.
   * An unvoiced noise generator produces shaped white noise based on the predicted `voicing` probability.
   * `pitch_shift` and `pitch_range` directly manipulate the fundamental frequency ($F_0$) trajectory before impulse train synthesis, providing algebraic pitch control with zero latency.
2. **Filter Application:**
   * `frame_transformer_b` predicts complex cepstral coefficients per 256-sample audio frame (10.67 ms at 24 kHz).
   * A differentiable time-varying filter filters the combined harmonic-plus-noise excitation into the final speech waveform.
3. **Sample Rate:** Strictly 24,000 Hz mono (native studio standard).

---

## 4. Text Preprocessing & Vocabulary Architecture

* **G2P Engine:** `espeak-ng` with Malayalam (`-v ml --ipa`) backend.
* **Symbol Budget:** 76 unique phoneme symbol slots allocated in `vocab_plan.json`.
* **Phonemic Mapping:** 59 native Malayalam phoneme units, including:
  * 15 vowel qualities (short and long pairs: /a, aː, i, iː, u, uː, e, eː, o, oː/, diphthongs /ai, au/, and schwa).
  * 36 consonants (full velar, palatal, retroflex, dental, labial stops with voiceless/voiced/aspirated distinctions).
  * Explicit Malayalam Chillus: ൻ (/n/), ർ (/r/), ൽ (/l/), ൾ (/ɭ/), ൺ (/ɳ/), ൿ (/k/).
  * Anusvara /m/ and Visarga /h/.
  * 5 silence / punctuation control tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`, ` `).

---

## 5. Deployment Artifacts & Integrity Verification

| Artifact Name | Format | Size | Components Included | Checksum Verified |
| :--- | :--- | :--- | :--- | :--- |
| `tui-voice-mini-ml.pt` | PyTorch State Dict | 18.2 MB | Full inference model weights | Verified |
| `tui-voice-mini-ml-fp16.pt` | PyTorch FP16 | 9.1 MB | Half-precision inference weights | Verified |
| `onnx/tui-voice-mini-ml.onnx` | ONNX (opset 17) | 21.4 MB | Complete graph (tokens $\rightarrow$ waveform) | Verified |
| `config.json` | JSON | 2.1 KB | Architecture hyperparameters + 76 vocab mapping | Verified |
| `vocab_plan.json` | JSON | 4.2 KB | Upstream slot substitution map | Verified |
| `checkpoints/step-40k.pt` | PyTorch Full State | 87.2 MB | Generator, optimizer states, training iteration | Verified |

---

## 6. Known Architectural Uncertainties & Engineering Notes

1. **Stochastic Sampling vs. Deterministic Decoding:**
   * Tui Voice Mini's duration and pitch predictors are deterministic feedforward heads (unlike flow-matching or diffusion TTS). This guarantees exact reproducibility and low jitter across consecutive inferences.
2. **Dynamic Sequence Boundaries:**
   * The text encoder accepts up to 512 input tokens. Long texts exceeding 512 tokens must be segmented along sentence boundaries (`.`, `?`, `!`, `।`, `\n`) and concatenated with silence buffers (recommended: 180 ms) to avoid buffer overflow.
3. **Absence of Pre-Trained Numerals Normalizer:**
   * Input text containing raw Arabic numerals (e.g., `2026`, `500`) or English abbreviations (e.g., `ATM`, `AI`) requires a frontend verbalizer prior to G2P conversion.
