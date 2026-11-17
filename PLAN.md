# Tui Voice Mini Malayalam: plan (v0.1-preview)

Goal: a Malayalam text-to-speech model of about 10 MB, made by fine-tuning an existing tiny model, not training from scratch.

## Base model
tiny-voca (n4ze3m/tiny-voca): 4,537,291 parameters, Apache-2.0, 18 MB fp32 (about 9 MB fp16, my arithmetic), 24 kHz.
FastSpeech 2-style text encoder, duration/pitch/energy predictors, and a source-filter waveform head (25,862 parameters; no neural vocoder).
Upstream recipe: 104,737 steps, 7.7 h on one RTX 4090 (about 0.26 s/step). Spectral losses for 8,000 steps, then GAN losses.
Why this base: only tiny model found with a public training recipe. Alternatives (Klein, SupraTTS, Inflect, Paradee) are for later tiers; training code not confirmed.

## Data
Rasa Malayalam female (ai4bharat/Rasa): 26.42 h, 16,974 utterances, CC-BY-4.0, expressive studio speech, 48 kHz -> 24 kHz. Accept the terms on Hugging Face first.
IMaSC (thennal/IMaSC): CC BY-SA 4.0, 16 kHz, 8 speakers. Too band-limited for training; use its sentences as an extra held-out text set.
Teacher option for later: Piper ml_IN arjun/meera (rhasspy/piper-checkpoints), about 64 MB; check each MODEL_CARD licence.
v0 filters: drop sentences with digits or Latin letters (no number expander yet), clips outside 2-12 s.

## Pipeline (what the code does)
1. 01_prepare_rasa.py: resample to 24 kHz, trim, normalise peak, clean Malayalam text, text-disjoint train/val/test.
2. 02_phonemize.py: eSpeak NG `-v ml --ipa --sep=_` (input newline-terminated), one token per phone, tokens for BOS/EOS/word boundary/pause punctuation.
3. Vocab plan (updated after review): the upstream symbol table is fixed. Malayalam sounds English lacks (ഴ ɻ, ള ɭ, ണ ɳ, retroflex ʈ ɖ, aspirates...) take over slots of English-only phones (θ ð ʒ æ ...). Each repurposed slot starts from the nearest English sound's embedding row. Same shapes as upstream, so the checkpoint loads unchanged. Slots are class-matched (vowel slots only take vowels, consonant slots only consonants) because upstream keeps fixed vowel/phone masks per slot; when no free slot is left, a rare phone is MERGED into its nearest English phone and listed in the report (e.g. ɭ into l). Check the MERGED list at Gate 1: merged contrasts are a quality loss.
4. 03_align.py (frame count = ceil(samples/256) to match upstream; one bad clip is rejected, never fatal): CTC forced alignment with facebook/wav2vec2-lv-60-espeak-cv-ft (alignment only), onsets shifted 19 ms earlier, pauses of 75 ms or more given to pause tokens, durations on the 256-sample grid, exact sum, at least 1 frame per token.
5. 04_check_annotations.py: compare with upstream stats (mean about 7.5 frames per phone, about 0.2% one-frame phones).
6. Train (bridge, see below): stage A about 5k steps, embeddings + text encoder only; stage B about 35k steps full model with GAN from step 3k. Starting guesses, not verified. About 3 h on a 4090-class GPU at upstream speed.
7. Evaluate: Malayalam ASR round-trip CER on held-out sentences (kit's scorer), plus listening to 20 fixed sentences at every checkpoint. Do not trust loss alone.
8. Export: ONNX, fp16, check file size under about 10 MB.

## Gates (stop if they fail)
- Gate 0: upstream_report.txt shows how train.py builds the model, loads/resumes checkpoints, and checks the annotation fingerprint.
- Gate 1: eSpeak gives sensible phones for ഴ ള ണ ന്റ ചില്ലു words; every phone has a source (UNRESOLVED list is empty).
- Gate 2: alignment on 200 clips: low reject rate, durations plausible, spot-listen to 10.
- Gate 3 (after 5k steps): speech is at least recognisable; if not, stop and inspect alignment.

## Training bridge (still to write, needs Gate 0)
Per a review of upstream source (not verified by me): training reads .pt feature files {mel (T,80), prosody (T,3), wave} made by tiny_voca.prepare; speakers must be in the upstream config (F2/M2); the annotations folder needs annotations.json with complete, fingerprint and acoustic_root.

## Known risks
- wav2vec2-lv-60-espeak-cv-ft may not cover every Malayalam phone; unknown phones fall back to a near label, which blurs boundaries (qc.unknown counts them).
- Upstream annotations carry a fingerprint that its checkpoints record and check; the bridge must handle this.
- Rasa is expressive; a 4.5M model may average prosody. If output is mushy, filter to the calmest styles.
- Source-filter synthesis has an 8-12 kHz ceiling in quality; expect a clearly synthetic voice.
- eSpeak NG is GPL-3.0: call it as an external program, do not ship it inside the model.
- No numbers, English words or abbreviations in v0.

## Licences
tiny-voca code and weights Apache-2.0 (credit it in the model card). Rasa CC-BY-4.0 (attribute AI4Bharat). wav2vec2 aligner Apache-2.0, alignment only.

## Later tiers
Super Nano / Nano / Micro: distill from this Mini model (and optionally Piper Malayalam) into smaller Klein/Supra-style students. Edge: wider model. Names finalised later.
