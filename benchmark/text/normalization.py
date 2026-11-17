"""Malayalam-aware Text Normalization Pipeline for TTS Evaluation.

Provides:
- Unicode NFC normalization
- Legacy chillu to atomic chillu conversion (e.g., U+0D23 + Virama + ZWJ -> U+0D7A)
- Malayalam number to words expansion (0 - 99,999,999)
- Punctuation and whitespace normalization
- Code-switching preservation
"""
import re
import unicodedata

# Legacy to Atomic Chillu mappings
LEGACY_CHILLU_MAP = {
    '\u0D23\u0D4D\u200D': '\u0D7A',  # ൺ
    '\u0D28\u0D4D\u200D': '\u0D7B',  # ൻ
    '\u0D30\u0D4D\u200D': '\u0D7C',  # ർ
    '\u0D32\u0D4D\u200D': '\u0D7D',  # ൽ
    '\u0D33\u0D4D\u200D': '\u0D7E',  # ൾ
    '\u0D15\u0D4D\u200D': '\u0D7F',  # ൿ
}

# Number verbalization tables
MALAYALAM_DIGITS = {
    0: "പൂജ്യം",
    1: "ഒന്ന്",
    2: "രണ്ട്",
    3: "മൂന്ന്",
    4: "നാല്",
    5: "അഞ്ച്",
    6: "ആറ്",
    7: "ഏഴ്",
    8: "എട്ട്",
    9: "ഒൻപത്",
    10: "പത്ത്",
    11: "പതിനൊന്ന്",
    12: "പന്ത്രണ്ട്",
    13: "പതിമൂന്ന്",
    14: "പതിനാല്",
    15: "പതിനഞ്ച്",
    16: "പതിനാറ്",
    17: "പതിനേഴ്",
    18: "പതിനെട്ട്",
    19: "പത്തൊൻപത്",
    20: "ഇരുപത്",
    30: "മുപ്പത്",
    40: "നാൽപ്പത്",
    50: "അൻപത്",
    60: "അറുപത്",
    70: "എഴുപത്",
    80: "എൺപത്",
    90: "തൊണ്ണൂറ്",
}

MALAYALAM_TENS_PREFIX = {
    2: "ഇരുപത്തി",
    3: "മുപ്പത്തി",
    4: "നാൽപ്പത്തി",
    5: "അൻപത്തി",
    6: "അറുപത്തി",
    7: "എഴുപത്തി",
    8: "എൺപത്തി",
    9: "തൊണ്ണൂറ്റി",
}


def number_to_malayalam(num: int) -> str:
    """Verbalize integer into natural Malayalam text."""
    if num < 0:
        return "നെഗറ്റീവ് " + number_to_malayalam(-num)
    if num in MALAYALAM_DIGITS:
        return MALAYALAM_DIGITS[num]

    if num < 100:
        tens = num // 10
        units = num % 10
        return MALAYALAM_TENS_PREFIX[tens] + MALAYALAM_DIGITS[units]

    if num < 1000:
        hundreds = num // 100
        rem = num % 100
        h_words = {
            1: "നൂറ്", 2: "ഇരുനൂറ്", 3: "മുന്നൂറ്", 4: "നാനൂറ്",
            5: "അഞ്ഞൂറ്", 6: "അറുന്നൂറ്", 7: "എഴുന്നൂറ്", 8: "എണ്ണൂറ്", 9: "തൊള്ളായിരം"
        }
        h_prefix = {
            1: "നൂറ്റി", 2: "ഇരുനൂറ്റി", 3: "മുന്നൂറ്റി", 4: "നാനൂറ്റി",
            5: "അഞ്ഞൂറ്റി", 6: "അറുന്നൂറ്റി", 7: "എഴുന്നൂറ്റി", 8: "എണ്ണൂറ്റി", 9: "തൊള്ളായിരത്തി"
        }
        if rem == 0:
            return h_words[hundreds]
        return h_prefix[hundreds] + " " + number_to_malayalam(rem)

    if num < 100000:  # Thousands
        thousands = num // 1000
        rem = num % 1000
        t_str = "ആയിരം" if thousands == 1 else number_to_malayalam(thousands) + " ആയിരം"
        t_prefix = "ആയിരത്തി" if thousands == 1 else number_to_malayalam(thousands) + " ആയിരത്തി"
        if rem == 0:
            return t_str
        return t_prefix + " " + number_to_malayalam(rem)

    if num < 10000000:  # Lakhs (100,000)
        lakhs = num // 100000
        rem = num % 100000
        l_str = "ഒരു ലക്ഷം" if lakhs == 1 else number_to_malayalam(lakhs) + " ലക്ഷം"
        if rem == 0:
            return l_str
        return l_str + " " + number_to_malayalam(rem)

    # Crores (10,000,000)
    crores = num // 10000000
    rem = num % 10000000
    c_str = "ഒരു കോടി" if crores == 1 else number_to_malayalam(crores) + " കോടി"
    if rem == 0:
        return c_str
    return c_str + " " + number_to_malayalam(rem)


def verbalize_numbers(text: str) -> str:
    """Replace all Arabic numerals in text with Malayalam spelled-out words."""
    def _rep(match):
        val = int(match.group(0))
        return number_to_malayalam(val)
    return re.sub(r'\b\d+\b', _rep, text)


def normalize_chillus(text: str) -> str:
    """Ensure all chillus are in standard atomic Unicode (U+0D7A - U+0D7F)."""
    for legacy, atomic in LEGACY_CHILLU_MAP.items():
        text = text.replace(legacy, atomic)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse tabs, multiple spaces, and trailing whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_punctuation(text: str, preserve_sentence_bounds: bool = True) -> str:
    """Normalize Malayalam punctuation.
    Converts full stop variations (including Danda ।) to standard punctuation.
    """
    text = text.replace('।', '.')
    text = text.replace('॥', '.')
    # Standardize quotes
    text = re.sub(r'[""«»]', '"', text)
    text = re.sub(r"[''`]", "'", text)
    # Standardize hyphens / dashes
    text = re.sub(r'[—–−]', '-', text)
    if not preserve_sentence_bounds:
        # Strip all punctuation for strict ASR WER comparison
        text = re.sub(r'[^\w\s]', '', text)
    return text


def normalize_malayalam(text: str, verbalize_nums: bool = True, strip_punct: bool = False) -> str:
    """Complete end-to-end normalization pipeline for Malayalam TTS benchmark evaluation."""
    if not text:
        return ""
    # 1. Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    # 2. Canonical chillu normalization
    text = normalize_chillus(text)
    # 3. Verbalize numerals if requested
    if verbalize_nums:
        text = verbalize_numbers(text)
    # 4. Punctuation normalization
    text = normalize_punctuation(text, preserve_sentence_bounds=not strip_punct)
    # 5. Whitespace normalization
    text = normalize_whitespace(text)
    return text
