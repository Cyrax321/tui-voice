"""GATE 0. Run inside the cloned tiny-voca repo (cwd = /content/tiny-voca).
Writes upstream_report.txt: CLI help, vocab/embedding hits, checkpoint keys and shapes,
resume / fingerprint hooks. Paste the report back so the train bridge can be written exactly."""
import re, subprocess, sys, json, pathlib

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "upstream_report.txt")
root = pathlib.Path("src/tiny_voca")
lines = []
def add(*a): lines.append(" ".join(str(x) for x in a))

add("== files"); [add(f"{p} {len(p.read_text().splitlines())} lines") for p in sorted(root.glob("*.py"))]
for mod in ("prepare", "annotate", "train", "evaluate", "export", "infer"):
    r = subprocess.run([sys.executable, "-m", f"tiny_voca.{mod}", "--help"], capture_output=True, text=True)
    add(f"\n== {mod} --help\n{(r.stdout or r.stderr)[-3500:]}")

pat = re.compile(r"(SYMBOLS|symbols|vocab|nn\.Embedding|fingerprint|resume|init_from|load_state_dict|"
                 r"torch\.load|argparse|add_argument|phoneme|PAD|BOS|EOS|n_symbols|num_symbols)")
for p in sorted(root.glob("*.py")):
    src = p.read_text().splitlines()
    hits = [i for i, l in enumerate(src) if pat.search(l)]
    if hits:
        add(f"\n== grep {p.name}")
        shown = set()
        for i in hits:
            for j in range(max(0, i - 1), min(len(src), i + 2)):
                if j not in shown:
                    shown.add(j); add(f"{j+1:5d}: {src[j][:160]}")
            if len(shown) > 260: add("... truncated"); break

try:
    import torch
    for ck in ("weights/tiny-voca.pt", "weights/inference.pt"):
        if pathlib.Path(ck).exists():
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            add(f"\n== checkpoint {ck}: top-level type {type(sd).__name__}")
            if isinstance(sd, dict):
                add("top keys:", list(sd.keys())[:30])
                for k in ("model", "state_dict", "ema", "config", "cfg", "symbols", "vocab"):
                    if k in sd:
                        v = sd[k]
                        if isinstance(v, dict):
                            for kk, vv in list(v.items())[:400]:
                                add(f"  {k}.{kk}", tuple(vv.shape) if hasattr(vv, "shape") else str(vv)[:200])
                        else:
                            add(f"  {k}:", str(v)[:1500])
except Exception as e:
    add("checkpoint inspect failed:", e)

p = pathlib.Path("weights/phonemizer")
if p.exists():
    add("\n== weights/phonemizer:", [x.name for x in p.iterdir()][:30])
cfgp = pathlib.Path("weights/config.json")
if cfgp.exists():
    cfg = json.load(open(cfgp)); add("\n== weights/config.json keys:", list(cfg.keys()))
    for k in ("vocab", "symbols"):
        v = cfg.get(k)
        if isinstance(v, dict):
            syms = [t for t, _ in sorted(v.items(), key=lambda x: x[1])]
        elif isinstance(v, list):
            syms = list(v)
        else:
            continue
        json.dump(syms, open("upstream_symbols.json", "w"), ensure_ascii=False)
        add(f"wrote upstream_symbols.json from config[{k!r}]: {len(syms)} symbols:", syms)
        break
    if "speakers" in cfg: add("speakers:", cfg["speakers"])
OUT.write_text("\n".join(lines))
print(f"wrote {OUT} ({len(lines)} lines). Paste it back.")
