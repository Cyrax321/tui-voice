"""Rasa Malayalam -> 24 kHz clean wavs + manifest.jsonl (id, text, path, dur, split).
Usage: python scripts/01_prepare_rasa.py --config <name> --out data [--limit N]
Run with --list first to see configs / columns."""
import argparse, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tui_voice import text as T

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="ai4bharat/Rasa")
ap.add_argument("--config"); ap.add_argument("--split", default="train")
ap.add_argument("--out", default="data"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--list", action="store_true")
ap.add_argument("--cfg", default="configs/mini_ml.json")
a = ap.parse_args()
cfg = json.load(open(a.cfg))

from datasets import get_dataset_config_names, load_dataset, Audio
hf_tok = os.environ.get("HF_TOKEN", True)
if a.list or not a.config:
    names = get_dataset_config_names(a.repo, token=hf_tok); print(names); sys.exit(0)
ds = load_dataset(a.repo, a.config, split=a.split, token=hf_tok)
print(ds); print(ds.features)
acol = next(k for k, v in ds.features.items() if isinstance(v, Audio))
tcol = next((c for c in ("text", "sentence", "transcription", "normalized_text", "transcript") if c in ds.features), None)
assert tcol, f"no text column in {list(ds.features)}"
print("audio col:", acol, "| text col:", tcol, "| other cols:", [c for c in ds.features if c not in (acol, tcol)])
ds = ds.cast_column(acol, Audio(sampling_rate=cfg["sample_rate"]))     # datasets resamples to 24 kHz

import soundfile as sf, librosa
os.makedirs(f"{a.out}/wavs", exist_ok=True)
lo, hi = cfg["clip_seconds"]; rows, rej = [], {}
for i, ex in enumerate(ds):
    if a.limit and i >= a.limit: break
    t = T.normalize(ex[tcol]); ok, why = T.usable(t)
    if not ok: rej[why] = rej.get(why, 0) + 1; continue
    y = np.asarray(ex[acol]["array"], dtype=np.float32)
    y, _ = librosa.effects.trim(y, top_db=cfg["trim_top_db"])
    d = len(y) / cfg["sample_rate"]
    if not (lo <= d <= hi): rej["duration"] = rej.get("duration", 0) + 1; continue
    y = y / (np.abs(y).max() + 1e-9) * cfg["peak"]
    p = f"{a.out}/wavs/{i:06d}.wav"; sf.write(p, y, cfg["sample_rate"])
    rows.append({"id": f"{i:06d}", "text": t, "path": p, "dur": round(d, 3)})
# text-disjoint split
random.Random(cfg["seed"]).shuffle(rows)
by_text = {}
for r in rows: by_text.setdefault(r["text"], []).append(r)
texts = list(by_text); random.Random(cfg["seed"]).shuffle(texts)
split_of, cnt = {}, 0
for tx in texts:
    s = "validation" if cnt < cfg["val_clips"] else "test" if cnt < cfg["val_clips"] + cfg["test_clips"] else "train"
    for r in by_text[tx]: r["split"] = s
    cnt += len(by_text[tx])
with open(f"{a.out}/manifest.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "kept; rejected:", rej, "| hours:", round(sum(r["dur"] for r in rows) / 3600, 2))
