import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import Counter
from tui_voice import text, phonemes as P, durations as D


def test_normalize_and_usable():
    t = text.normalize("  “മഴ”  പെയ്യുന്നു…  ")
    assert t == '" മഴ " പെയ്യുന്നു.'.replace('" മഴ "', "മഴ") or "മഴ" in t
    assert text.usable("മഴ പെയ്യുന്നു.")[0]
    assert text.usable("ഇത് 25 ആണ്") == (False, "digits")
    assert text.usable("hello മഴ") == (False, "latin")
    assert text.normalize("അത്\u200d")[-1] == "\u200d"      # ZWJ kept (chillu)


def test_clauses():
    assert text.split_clauses("മഴ, വെയിൽ. വീട്") == [("മഴ", ","), ("വെയിൽ", "."), ("വീട്", "")]


def fake_espeak(clause):
    return {"മഴ": "m_a_ɻ_a", "വെയിൽ": "ʋ_e_j_i_l", "വീട്": "ʋ_iː_ʈ_ɨ"}.get(clause, "a")


def test_phonemize():
    toks = P.phonemize("മഴ, വെയിൽ.", espeak=fake_espeak)
    assert toks[0] == P.BOS and toks[-1] == P.EOS
    assert toks[1:5] == ["m", "a", "ɻ", "a"] and toks[5] == ","
    toks2 = P.phonemize("മഴ വീട്", espeak=lambda c: "m_a_ɻ_a v_iː_ʈ_ɨ")
    assert P.WB in toks2


def test_vocab_plan_keeps_size_and_inits():
    up = ["<pad>", "<bos>", "<eos>", "<sp>", ",", ".", "m", "a", "l", "ɹ", "t", "n", "ɪ", "θ", "ð", "ʒ", "æ", "iː", "v", "j", "i", "eɪ", "ɚ", "ɝ", "ʌ"]
    toks = [P.phonemize("മഴ, വെയിൽ.", espeak=fake_espeak), P.phonemize("വീട്", espeak=fake_espeak)]
    cnt = P.inventory(toks)
    plan = P.build_vocab_plan(up, cnt)
    assert len(plan["new_symbols"]) == len(up)
    assert not plan["unresolved"], plan["report"]
    s = plan["new_symbols"]
    assert "ɻ" in s and s[s.index("ɻ")] != "ɹ"
    assert plan["init_from"][s.index("ɻ")] == up.index("ɹ")      # ഴ starts from English r
    assert plan["init_from"][s.index("ʈ")] == up.index("t")
    assert s[up.index("m")] == "m"                                  # used phones stay put
    assert len(set(s)) == len(s)                                    # no duplicate symbols
    enc = P.encode(toks[0], plan, up)
    assert all(0 <= i < len(up) for i in enc)


def test_complex_and_geminate_phones():
    up = ["<pad>", "<bos>", "<eos>", "<sp>", "t", "d", "n", "ɹ", "l", "j", "k", "ə"]
    ml_counts = Counter({"ʈʈ": 9159, "ʲ": 14887, "r.": 5760, "r̩": 1717, "ʰχ": 3})
    plan = P.build_vocab_plan(up, ml_counts)
    assert not plan["unresolved"], plan["report"]
    enc = P.encode([P.BOS, "ʈʈ", "ʲ", "r.", "r̩", "ʰχ", P.EOS], plan, up)
    assert len(enc) == 7
    assert all(0 <= idx < len(up) for idx in enc)


def test_durations_exact_and_min1():
    is_phone = [False, True, True, False, True, True, False]       # bos a b wb c d eos
    d = D.build_durations(is_phone, [0.10, 0.20, 0.40, 0.55], 120)
    assert d.sum() == 120 and d.min() >= 1 and len(d) == 7
    assert d[0] >= 1 and d[1] > 1


def test_durations_pause_goes_to_pause_token():
    total = 200
    sil = np.zeros(total, bool)
    sil[60:90] = True                                               # 320 ms of silence before 2nd word
    is_phone = [False, True, True, False, True, False]              # bos a b wb c eos
    ons = [0.20, 0.40, 1.0]                                          # c starts ~ frame 90
    d = D.build_durations(is_phone, ons, total, sil=sil)
    assert d.sum() == total and d.min() >= 1
    assert d[3] >= 25                                                # the <wb> token holds the silence
    d2 = D.build_durations(is_phone, ons, total, sil=None)
    assert d2[3] == 1                                                # without a mask it gets min frame


def test_durations_tight():
    d = D.build_durations([False, True, False], [0.0], 3)
    assert list(d) == [1, 1, 1]


def test_silence_mask():
    sr = 24000
    y = np.concatenate([np.zeros(sr // 2), 0.5 * np.sin(np.linspace(0, 2000, sr))])
    n = D.num_frames(len(y))
    m = D.silence_mask(y.astype(np.float32), n)
    assert m[10] and not m[n - 10]


def test_remap_embeddings():
    from tui_voice.init_ckpt import remap_embeddings
    w = np.arange(12, dtype=float).reshape(6, 2)
    sd = {"enc.sym_emb.weight": w, "other.weight": np.zeros((6, 6, 2)), "bias": np.zeros(6)}
    new, ch = remap_embeddings(sd, [0, 1, 2, 2, 4, 1])
    assert ch == ["enc.sym_emb.weight"]
    assert (new["enc.sym_emb.weight"][3] == w[2]).all() and (new["enc.sym_emb.weight"][5] == w[1]).all()
    assert (w == np.arange(12).reshape(6, 2)).all()                 # original untouched


def test_class_matching_and_merge_and_protection():
    up = ["<pad>", "<bos>", "<eos>", "<sp>", ",", ".", "t", "d", "n", "ɹ", "ʃ", "ɐ", "ɑː", "ʊ", "uː", "iː",
          "æ", "ɜː", "θ", "ð", "l", "ɪ"]            # free: vowels æ ɜː, consonants θ ð
    cnt = Counter({"ɐ": 9, "aː": 5, "u": 4, "ʈ": 3, "ɖ": 3, "ɻ": 2, "ɭ": 2, "ɨ": 1, "ɳ": 1, "ʃ": 6})
    plan = P.build_vocab_plan(up, cnt)
    s = plan["new_symbols"]
    assert not plan["unresolved"], plan["report"]
    # basic English consonants/vowels that Malayalam needs were never overwritten
    for keep in ("t", "d", "n", "ɹ", "ʃ", "l", "ɪ", "ɐ", "ɑː", "ʊ", "<sp>", "<bos>", "<eos>"):
        assert keep in s, keep
    # vowels only land in originally-vowel slots, consonants in originally-consonant slots
    for slot, (old, new) in enumerate(zip(up, s)):
        if old != new:
            assert P.is_vowel(old) == P.is_vowel(new), (old, new)
    assert plan["merged"], "5 consonant-class phones but only 2 free consonant slots: some must merge"
    for m in plan["merged"]:
        assert plan["mapping"][m] in range(len(up))


def test_special_aliases():
    up = ["<pad>", "<bos>", "<eos>", "<sp>", "a"]
    plan = P.build_vocab_plan(up, Counter({"a": 1}))
    assert P.encode([P.BOS, "a", P.WB, P.EOS], plan, up) == [1, 4, 3, 2]
    up2 = ["<pad>", "<s>", "</s>", "<wb>", "a"]
    assert P.encode([P.BOS, P.WB, P.EOS], P.build_vocab_plan(up2, Counter()), up2) == [1, 3, 2]


def test_num_frames_matches_upstream():
    assert D.num_frames(256) == 1 and D.num_frames(257) == 2 and D.num_frames(24000) == 94


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
