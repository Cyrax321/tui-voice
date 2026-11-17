"""eSpeak NG Malayalam front end + mapping into tiny-voca's fixed symbol table.

Why remap instead of resizing the embedding: the upstream checkpoint has a fixed symbol
table (76 symbols: ~59 eSpeak phones plus BOS/EOS/word-boundary/punctuation). Malayalam
needs sounds English lacks (retroflexes, aspirates, ഴ). We REPURPOSE slots of English-only
phones so the checkpoint loads unchanged; each repurposed slot starts from the nearest
English sound's embedding row (see init_from).
"""
import subprocess
from collections import Counter
from .text import split_clauses

BOS, EOS, WB = "<bos>", "<eos>", "<sp>"      # canonical; upstream spelling resolved via ALIASES
SPECIALS = {BOS, EOS, WB}
ALIASES = {BOS: ["<bos>", "<s>", "<BOS>"], EOS: ["<eos>", "</s>", "<EOS>"],
           WB: ["<sp>", "<wb>", "<space>", " "]}

NEAREST = {
    "ɻ": "ɹ", "ɭ": "l", "ɳ": "n", "ɖ": "d", "ʈ": "t", "ɽ": "ɾ", "ɾ": "r", "r": "ɹ",
    "ɲ": "n", "ɟ": "dʒ", "c": "tʃ", "g": "ɡ", "ʋ": "v", "ʂ": "ʃ", "ɕ": "ʃ",
    "ɨ": "ɪ", "ɐ": "ʌ", "o": "oʊ", "e": "eɪ", "ɯ": "ʊ",
    "tʰ": "t", "kʰ": "k", "pʰ": "p", "dʱ": "d", "bʱ": "b", "ɡʱ": "ɡ", "ʈʰ": "t",
    "ɖʱ": "d", "cʰ": "tʃ", "ɟʱ": "dʒ", "t̪": "t", "d̪": "d", "n̪": "n",
    # tiny-voca has no raw a / aː / u (it has ɐ ɑː ʊ uː); found by checking against its real symbols
    "a": "ɐ", "aː": "ɑː", "u": "ʊ", "uː": "uː", "eː": "eɪ", "oː": "oː", "i": "ɪ", "iː": "iː",
}
VOWEL_POOL = ["æ", "ɚ", "ɝ", "ɒ", "ɜː", "ɔɪ", "aʊ", "aɪ", "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl"]
CONS_POOL = ["θ", "ð", "ʒ", "n̩", "ʔ", "x", "ɬ"]
VOWEL_CHARS = set("aeiouɐɑɒɔəɛɜɪʊʌæɚɝɨɯøœyɤɵɘɞɶ")


def is_vowel(sym: str) -> bool:
    # upstream's model.py keeps non-persistent vowel/phone masks per slot (duration/pitch losses),
    # so a repurposed slot must keep its class: vowels stay vowels, consonants stay consonants
    return any(c in VOWEL_CHARS for c in sym)
_STRIP_MARKS = "ˈˌ'ˑ"


def run_espeak(clause: str, voice: str = "ml") -> str:
    # The trailing newline is REQUIRED: eSpeak NG 1.51 drops the last character otherwise
    # (documented in tiny-voca's annotations-v3 notes).
    r = subprocess.run(["espeak-ng", "-v", voice, "--ipa", "-q", "--sep=_", "--stdin"],
                       input=clause + "\n", capture_output=True, text=True, check=True)
    return r.stdout


def parse_espeak(raw: str):
    """'m_a_l_a\\nk_a_ɻ_i' -> list of words, each a list of phones."""
    words = []
    for line in raw.splitlines():
        for w in line.split():
            ph = [p.strip(_STRIP_MARKS) for p in w.split("_")]
            ph = [p for p in ph if p]
            if ph:
                words.append(ph)
    return words


def phonemize(text: str, espeak=run_espeak):
    """Normalised text -> tokens: <bos> phones <wb> phones , ... <eos>. Pause punctuation kept."""
    toks = [BOS]
    for clause, punct in split_clauses(text):
        words = parse_espeak(espeak(clause))
        for i, w in enumerate(words):
            if i:
                toks.append(WB)
            toks.extend(w)
        if punct:
            toks.append(punct)
        elif words:
            toks.append(WB)
    while toks and toks[-1] == WB:
        toks.pop()
    toks.append(EOS)
    return toks


def inventory(token_lists):
    c = Counter()
    for t in token_lists:
        c.update(x for x in t if x not in SPECIALS and len(x) > 0 and not (len(x) == 1 and x in ",;:.?!"))
    return c


def _fallback_chain(ph, upstream):
    if ph in upstream:
        return ph, "exact"
    if ph in NEAREST and NEAREST[ph] in upstream:
        return NEAREST[ph], "table"
    base = ph.replace("ː", "").replace("ʰ", "").replace("ʱ", "").replace("\u032a", "")
    if base in upstream:
        return base, "stripped"
    if base in NEAREST and NEAREST[base] in upstream:
        return NEAREST[base], "table_stripped"
    if base[:1] in upstream:
        return base[:1], "first_char"
    return None, "none"


def resolve_specials(upstream_symbols):
    """canonical special -> the upstream spelling (None if upstream lacks it)."""
    out = {}
    for canon, names in ALIASES.items():
        out[canon] = next((n for n in names if n in upstream_symbols), None)
    return out


def build_vocab_plan(upstream_symbols, ml_counts):
    """upstream_symbols: list[str] in embedding-index order; ml_counts: Counter of Malayalam phones.
    Same vocab size as upstream. A Malayalam phone missing from upstream either
      (a) takes over an unused English slot of the SAME class (vowel/consonant), initialised
          from its nearest English phone, or
      (b) if no such slot is left, is MERGED into its nearest English phone (no new slot).
    Returns dict(new_symbols, init_from, mapping, report, unresolved, merged)."""
    n = len(upstream_symbols)
    idx = {s: i for i, s in enumerate(upstream_symbols)}
    new_symbols, init_from = list(upstream_symbols), list(range(n))
    mapping, report, unresolved, merged = {}, [], [], []
    for p in ml_counts:
        if p in idx:
            mapping[p] = idx[p]
    missing = [(p, c) for p, c in ml_counts.most_common() if p not in idx]
    sources = {}
    for p, _ in missing:
        sources[p] = _fallback_chain(p, idx)
    protected = {v for v in resolve_specials(upstream_symbols).values() if v}
    protected |= set(mapping) | {src for src, _ in sources.values() if src}
    protected |= {s for s in upstream_symbols if len(s) == 1 and not s.isalpha()}
    pools = {True: [s for s in VOWEL_POOL if s in idx and s not in protected],
             False: [s for s in CONS_POOL if s in idx and s not in protected]}
    for cls in (True, False):                      # generic extras, still class-matched, never protected
        for s in upstream_symbols:
            if (s not in protected and s not in pools[cls] and s.strip() and not s.startswith("<")
                    and s.isalpha() and is_vowel(s) == cls):
                pools[cls].append(s)
    for p, c in missing:
        src, how = sources[p]
        if src is None:
            unresolved.append(p); report.append(f"UNRESOLVED {p} (n={c}): no source phone"); continue
        pool = pools[is_vowel(p)]
        if not pool:
            mapping[p] = idx[src]; merged.append(p)
            report.append(f"{p} (n={c}) MERGED into {src} (no free {'vowel' if is_vowel(p) else 'consonant'} slot)")
            continue
        slot_sym = pool.pop(0)
        slot = idx[slot_sym]
        new_symbols[slot] = p
        init_from[slot] = idx[src]
        mapping[p] = slot
        report.append(f"{p} (n={c}) -> slot {slot} (was {slot_sym}), init from {src} [{how}]")
    return {"new_symbols": new_symbols, "init_from": init_from, "mapping": mapping,
            "report": report, "unresolved": unresolved, "merged": merged}


def encode(tokens, plan, upstream_symbols):
    idx = {s: i for i, s in enumerate(upstream_symbols)}
    spec = resolve_specials(upstream_symbols)
    out = []
    for t in tokens:
        if t in SPECIALS:
            if spec[t] is None:
                raise KeyError(f"upstream has no symbol for {t!r}; aliases {ALIASES[t]}")
            out.append(idx[spec[t]])
        elif t in plan["mapping"]:
            out.append(plan["mapping"][t])
        elif t in idx:
            out.append(idx[t])
        else:
            raise KeyError(f"unmapped token {t!r}")
    return out
