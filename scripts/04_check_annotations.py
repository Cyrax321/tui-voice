"""Sanity stats on annotations (compare with tiny-voca: mean phone ~7.5 frames, ~0.2% one-frame phones)."""
import json, sys, numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else "annotations"
for s in ("train", "validation", "test"):
    try: rows = [json.loads(l) for l in open(f"{d}/{s}.jsonl")]
    except FileNotFoundError: continue
    if not rows: print(s, "empty"); continue
    allp = np.concatenate([np.array(r["durations"]) for r in rows])
    bad = sum(sum(r["durations"]) != r["frames"] for r in rows)
    print(s, len(rows), "clips | mean frames/token", allp.mean().round(2), "| 1-frame share", (allp == 1).mean().round(4),
          "| sum mismatches", bad, "| mean conf", np.mean([r["qc"]["conf"] for r in rows]).round(3))
