"""GATE 2. Forced-align every clip -> annotations/{train,validation,test}.jsonl + rejects.
Usage: python scripts/03_align.py --data data --out annotations [--limit N]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tui_voice import durations as D, phonemes as P
from tui_voice.align import Aligner
import soundfile as sf, librosa

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
ap.add_argument("--out", default="annotations"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--speaker", default="F2", help="must be one of upstream config.speakers (F2/M2)")
ap.add_argument("--acoustic-root", default="", help="dir of the .pt features made by tiny_voca.prepare")
ap.add_argument("--min-conf", type=float, default=0.30); ap.add_argument("--max-unknown", type=float, default=0.10)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rows = [json.loads(l) for l in open(f"{a.data}/phonemes.jsonl")]
if a.limit: rows = rows[: a.limit]
al = Aligner()
outs = {s: open(f"{a.out}/{s}.jsonl", "w") for s in ("train", "validation", "test")}
rej = open(f"{a.out}/rejected.jsonl", "w"); stats = {"ok": 0, "rej": 0}
total_rows = len(rows)
print(f"Aligning {total_rows} clips on {al.device}...")

for idx, r in enumerate(rows, 1):
    y, _ = sf.read(r["path"], dtype="float32"); total = D.num_frames(len(y))
    toks = r["tokens_str"]; is_ph = [t not in P.SPECIALS and not (len(t) == 1 and t in ",;:.?!") for t in toks]
    phones = [t for t, p in zip(toks, is_ph) if p]
    y16 = librosa.resample(y, orig_sr=24000, target_sr=16000)
    ons, conf, nunk = al.phone_onsets(y16, phones)
    why = None
    if ons is None: why = "align_failed"
    elif conf < a.min_conf: why = f"low_conf_{conf:.2f}"
    elif nunk / max(1, len(phones)) > a.max_unknown: why = f"unknown_phones_{nunk}"
    if why:
        rej.write(json.dumps({"id": r["id"], "why": why}, ensure_ascii=False) + "\n"); stats["rej"] += 1
    else:
        d = D.build_durations(is_ph, ons, total, sil=D.silence_mask(y, total))
        rec = {"id": r["id"], "speaker": a.speaker, "text": r["text"], "path": r["path"], "symbols": toks,
               "tokens": r.get("tokens"), "stress": [0] * len(toks), "durations": d.tolist(), "frames": total,
               "qc": {"conf": round(conf, 3), "unknown": nunk},
               "provenance": {"source": "ai4bharat/Rasa", "license": "CC-BY-4.0", "aligner": "facebook/wav2vec2-lv-60-espeak-cv-ft"}}
        outs[r["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n"); stats["ok"] += 1

    if idx % 1000 == 0 or idx == total_rows:
        print(f"[{idx}/{total_rows}] ok={stats['ok']} rej={stats['rej']} ({idx*100//total_rows}%)")
json.dump({"complete": True, "acoustic_root": a.acoustic_root, "fingerprint": None, "stats": stats,
           "note": "fingerprint must be filled the way tiny_voca.annotate does (see Gate 0 report)"},
          open(f"{a.out}/annotations.json", "w"), indent=1)
print(stats)
