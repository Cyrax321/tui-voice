"""Malayalam Grapheme Cluster Segmentation and Analysis.

In Malayalam script (Unicode range U+0D00 - U+0D7F), a visible typographic letter (akshara)
often spans multiple Unicode code points:
  Base Consonant + Virama (Chandrakkala) + Consonant (Conjunct, e.g. പ്ര = പ + ്ര)
  Base Consonant + Dependent Vowel Sign (Matra, e.g. കി = ക + ി)
  Chillu Characters (Pure consonants without inherent vowel, e.g. ൻ, ർ, ൽ, ൾ, ൺ)
  Consonant + Virama + ZWJ / ZWNJ sequences.

Blind code-point or byte-level counting produces false CER distortions.
This module implements Unicode standard Indic Syllabic Category segmentation.
"""
import unicodedata
import re

# Malayalam Unicode Block
MALAYALAM_START = 0x0D00
MALAYALAM_END = 0x0D7F

# Unicode Category Sets for Malayalam
VIRAMA = '\u0D4D'  # ് Chandrakkala
ANUSVARA = '\u0D02'  # ം
VISARGA = '\u0D03'   # ഃ
ZWJ = '\u200D'
ZWNJ = '\u200C'

# Dependent Vowel Signs (Matras)
VOWEL_SIGNS = set(chr(cp) for cp in range(0x0D3E, 0x0D4D))  # ാ through ൗ
VOWEL_SIGNS.update([chr(0x0D57), chr(0x0D62), chr(0x0D63)])

# Standalone Vowels
INDEPENDENT_VOWELS = set(chr(cp) for cp in range(0x0D05, 0x0D15))

# Consonants
CONSONANTS = set(chr(cp) for cp in range(0x0D15, 0x0D3B))
CONSONANTS.update([chr(0x0D7A), chr(0x0D7B), chr(0x0D7C), chr(0x0D7D), chr(0x0D7E), chr(0x0D7F)]) # Chillus

# Atomic Chillus
CHILLUS = {
    '\u0D7A': 'NN', # ൺ
    '\u0D7B': 'N',  # ൻ
    '\u0D7C': 'R',  # ർ
    '\u0D7D': 'L',  # ൽ
    '\u0D7E': 'LL', # ൾ
    '\u0D7F': 'K',  # ൿ
}


def get_grapheme_clusters(text: str) -> list[str]:
    """Segment a Malayalam string into linguistic orthographic syllables (grapheme clusters).
    
    Examples:
      "നമസ്കാരം" -> ["ന", "മ", "സ്കാ", "രം"]
      "കേരളം"    -> ["കേ", "ര", "ളം"]
      "പച്ചച്ചക്ക" -> ["പ", "ച്ച", "ച്ച", "ക്ക"]
    """
    if not text:
        return []

    # Ensure NFC normalized form
    norm_text = unicodedata.normalize('NFC', text)
    clusters = []
    current = ""
    i = 0
    n = len(norm_text)

    while i < n:
        char = norm_text[i]
        current += char
        i += 1

        # Check if the next sequence attaches to current base
        while i < n:
            next_char = norm_text[i]

            # 1. Virama binds to subsequent consonant or ZWJ/ZWNJ (Conjunct formation)
            if char == VIRAMA:
                if next_char in CONSONANTS or next_char in (ZWJ, ZWNJ):
                    current += next_char
                    char = next_char
                    i += 1
                    continue
                else:
                    break

            # 2. Next character is virama
            if next_char == VIRAMA:
                current += next_char
                char = next_char
                i += 1
                continue

            # 3. Next character is a dependent vowel sign (matra)
            if next_char in VOWEL_SIGNS:
                current += next_char
                char = next_char
                i += 1
                continue

            # 4. Next character is modifier (Anusvara or Visarga)
            if next_char in (ANUSVARA, VISARGA):
                current += next_char
                char = next_char
                i += 1
                break  # Modifiers terminate the akshara

            # 5. Zero-width joiner continuation
            if next_char in (ZWJ, ZWNJ):
                current += next_char
                char = next_char
                i += 1
                continue

            # Otherwise, new grapheme cluster begins
            break

        clusters.append(current)
        current = ""

    return clusters


def count_graphemes(text: str) -> int:
    """Return the exact count of linguistic grapheme clusters in a text."""
    # Filter whitespace/punctuation when doing pure character error measurement
    clusters = [c for c in get_grapheme_clusters(text) if not c.isspace()]
    return len(clusters)
