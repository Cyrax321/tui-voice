# tui-voice

Tiny Malayalam TTS, fine-tuned from [tiny-voca](https://github.com/n4ze3m/tiny-voca). First model: **Tui Voice Mini Malayalam** (v0.1-preview).
See PLAN.md. Run order is in notebooks/tui_voice_mini_ml.ipynb.

Tests (numpy only): `python tests/test_all.py`

Status: text, phoneme, vocab-plan, duration and checkpoint-remap code are tested. `tui_voice/align.py` and the scripts have not been run end to end (needs torch, eSpeak, the dataset). The training bridge to tiny-voca is written after Gate 0.
