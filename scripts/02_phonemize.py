"""manifest.jsonl -> phonemes.jsonl + vocab_plan.json (needs espeak-ng; upstream symbols from Gate 0).
Usage: python scripts/02_phonemize.py --data data --upstream-symbols upstream_symbols.json
upstream_symbols.json = JSON list of the checkpoint's symbols in embedding-index order."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tui_voice import phonemes as P

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
ap.add_argument("--upstream-symbols"); a = ap.parse_args()
rows = [json.loads(l) for l in open(f"{a.data}/manifest.jsonl")]
for r in rows: r["tokens_str"] = P.phonemize(r["text"])
cnt = P.inventory([r["tokens_str"] for r in rows])
print(len(cnt), "distinct Malayalam phones. Top:", cnt.most_common(70))
if a.upstream_symbols:
    up = json.load(open(a.upstream_symbols))
    plan = P.build_vocab_plan(up, cnt)
    print("\n".join(plan["report"]))
    print("UNRESOLVED:", plan["unresolved"])
    for r in rows: r["tokens"] = P.encode(r["tokens_str"], plan, up)
    json.dump(plan, open(f"{a.data}/vocab_plan.json", "w"), ensure_ascii=False, indent=1)
with open(f"{a.data}/phonemes.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", f"{a.data}/phonemes.jsonl")
