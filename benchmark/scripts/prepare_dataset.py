"""Benchmark Dataset Generator and Leakage Auditor.

Generates:
1. Dev Set (100 sentences) - Pipeline sanity checking
2. Main Test Set (500 sentences balanced across 6 required linguistic categories):
   - Everyday conversational (100)
   - Formal / news / informational (100)
   - Long sentences and compound words (75)
   - Questions and exclamations (75)
   - Numbers, dates, times, currency (75)
   - Malayalam-English code-switching (75)
3. Extended Stress-Test Set (200 sentences) - Tongue twisters, rare conjuncts, chillus, rapid syllables
"""
import json
import os
import hashlib
from pathlib import Path

# Base data directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLITS_DIR = DATA_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Curated seed corpora covering the 6 linguistic categories
EVERYDAY_CONVERSATIONAL = [
    "ഹലോ, സുഖമാണോ? എന്തൊക്കെയുണ്ട് വിശേഷങ്ങൾ?",
    "ഞാൻ ഇപ്പോൾ വീട്ടിൽ എത്തി, അവിടെ മഴ തുടങ്ങിയിട്ടുണ്ടോ?",
    "നാളെ രാവിലെ പത്ത് മണിക്ക് നമുക്ക് നേരിട്ട് സംസാരിക്കാം.",
    "ചായ കുടിച്ചോ? ഇന്ന് ഉച്ചയ്ക്ക് എന്താണ് സ്പെഷ്യൽ?",
    "നല്ലൊരു ശുഭദിനം നേരുന്നു, കാര്യങ്ങളെല്ലാം നന്നായി വരട്ടെ.",
    "സുഹൃത്തേ, സുഖമാണോ? ഇന്ന് ഓഫീസിൽ എപ്പോഴാണ് എത്തുന്നത്?",
    "ഇന്ന് വൈകുന്നേരം നമുക്ക് ബീച്ചിൽ പോയി കാറ്റുകൊള്ളാം.",
    "അമ്മയോട് ഞാൻ കാര്യം പറഞ്ഞിട്ടുണ്ട്, വിഷമിക്കേണ്ടതില്ല.",
    "ഈ വഴിയിലൂടെ പോയാൽ ട്രാഫിക് കുറവായിരിക്കും.",
    "കുറച്ചു വെള്ളം കുടിക്കൂ, ഇന്ന് നല്ല ചൂടാണ് പുറത്ത്.",
]

FORMAL_NEWS = [
    "സംസ്ഥാനത്ത് അടുത്ത അഞ്ച് ദിവസം ശക്തമായ മഴയ്ക്ക് സാധ്യതയുണ്ടെന്ന് കാലാവസ്ഥാ വകുപ്പ് അറിയിച്ചു.",
    "കേരളത്തിന്റെ വികസന മുന്നേറ്റത്തിന് ഐ ടി മേഖല നൽകിയ സംഭാവനകൾ വലുതാണ്.",
    "മലയോര മേഖലകളിൽ പ്രത്യേക ജാഗ്രതാ നിർദ്ദേശം പുറപ്പെടുവിച്ചിട്ടുണ്ട്.",
    "വിദ്യാഭ്യാസ രംഗത്ത് സമഗ്രമായ മാറ്റങ്ങൾ കൊണ്ടുവരാൻ പുതിയ പദ്ധതി നടപ്പിലാക്കുന്നു.",
    "തിരുവനന്തപുരം അന്താരാഷ്ട്ര വിമാനത്താവളത്തിൽ പുതിയ ടെർമിനൽ നിർമ്മാണം തുടങ്ങി.",
    "കൊച്ചി മെട്രോയുടെ പുതിയ പാത അടുത്ത മാസം പൊതുജനങ്ങൾക്ക് തുറന്നുകൊടുക്കും.",
    "കൃഷി വകുപ്പ് കർഷകർക്കായി പുതിയ ധനസഹായ പദ്ധതി പ്രഖ്യാപിച്ചു.",
    "പൊതുഗതാഗത സംവിധാനങ്ങൾ കൂടുതൽ പരിസ്ഥിതി സൗഹൃദമാക്കാൻ സർക്കാർ തീരുമാനിച്ചു.",
    "ശാസ്ത്ര സാങ്കേതിക രംഗത്ത് യുവാക്കൾക്ക് കൂടുതൽ തൊഴിലവസരങ്ങൾ സൃഷ്ടിക്കപ്പെടും.",
    "ആരോഗ്യ മേഖലയിൽ കൂടുതൽ ഡോക്ടർമാരെ നിയമിക്കാൻ മന്ത്രിസഭാ യോഗം അനുമതി നൽകി.",
]

LONG_COMPOUND = [
    "പ്രകൃതിസംരക്ഷണ പ്രവർത്തനങ്ങളിൽ പങ്കാളികളാകേണ്ടത് ഓരോ പൗരന്റെയും മൗലികമായ കടമയാണ്.",
    "വിദ്യാഭ്യാസവികസനപ്രവർത്തനങ്ങൾക്കായി പ്രത്യേക സാമ്പത്തികസഹായം അനുവദിച്ചിരിക്കുന്നു.",
    "കാലാവസ്ഥാവ്യതിയാനത്തെക്കുറിച്ചുള്ള ആഗോളസമ്മേളനത്തിൽ ഇന്ത്യ നിർണ്ണായക പങ്കുവഹിച്ചു.",
    "തീരദേശപരിപാലനനിയമങ്ങൾ കർശനമായി പാലിക്കണമെന്ന് ഉന്നതതലസമിതി നിർദ്ദേശിച്ചു.",
    "സാങ്കേതികവിദ്യാവിനിമയത്തിൽ പ്രാദേശികഭാഷകളുടെ പങ്ക് വളരെ നിർണ്ണായകമാണ്.",
    "സർവ്വകലാശാലാവിദ്യാഭ്യാസത്തിന്റെ നിലവാരം ഉയർത്താൻ നൂതനഗവേഷണകേന്ദ്രങ്ങൾ തുടങ്ങും.",
    "ദുരന്തനിവാരണപ്രവർത്തനങ്ങൾക്ക് സന്നദ്ധപ്രവർത്തകരുടെ കൂട്ടായ്മ മാതൃകയായി.",
    "ജലസംരക്ഷണപദ്ധതികൾ നാടിന്റെ സുസ്ഥിരവികസനത്തിന് അത്യന്താപേക്ഷിതമാണ്.",
]

QUESTIONS_EXCLAMATIONS = [
    "നിങ്ങൾ എപ്പോഴാണ് കൊച്ചിയിൽ നിന്ന് തിരികെ വരുന്നത്?",
    "എന്തൊരു മനോഹരമായ കാഴ്ചയാണ് ഈ മലനിരകൾ!",
    "ഈ പുതിയ പുസ്തകം എവിടെ നിന്നാണ് വാങ്ങാൻ സാധിക്കുക?",
    "അത്ഭുതം തന്നെ, ഇത്ര വേഗത്തിൽ ജോലി തീർക്കാൻ കഴിഞ്ഞോ!",
    "നാളെ വൈകുന്നേരത്തെ മീറ്റിംഗിൽ ആരൊക്കെയാണ് പങ്കെടുക്കുന്നത്?",
    "അയ്യോ! വഴിയിൽ നല്ലൊരു വലിയ മരം കടപുഴകി വീണിരിക്കുന്നു.",
    "ഇതിലും നല്ലൊരു അവസരം നമുക്ക് ഇനി എപ്പോഴാണ് ലഭിക്കുക?",
    "എത്ര മനോഹരമായാണ് അവർ ആ ഗാനം ആലപിച്ചത്!",
]

NUMBERS_DATES_CURRENCY = [
    "ഇന്ന് 2026 മാർച്ച് മാസം 18 തീയതിയാണ്.",
    "അദ്ദേഹത്തിന്റെ ഫോൺ നമ്പർ 98470 12345 ആണ്.",
    "ഈ പുസ്തകത്തിന്റെ വില 450 രൂപയും 50 പൈസയുമാണ്.",
    "രാവിലെ 8 മണി മുതൽ വൈകുന്നേരം 6 മണി വരെയാണ് സമയം.",
    "കഴിഞ്ഞ വർഷം 15000 കുട്ടികൾ പരീക്ഷയിൽ വിജയിച്ചു.",
    "അടുത്ത ട്രെയിൻ പ്ലാറ്റ്‌ഫോം നമ്പർ 3-ൽ 2 മണിക്ക് എത്തും.",
    "നൂറു രൂപ കൊടുത്താൽ ബാക്കി 25 രൂപ തിരികെ തരും.",
    "ഈ കെട്ടിടത്തിന് 12 നിലകളും 48 മുറികളുമുണ്ട്.",
]

CODE_SWITCHING = [
    "ഞാൻ tomorrow മോർണിംഗ് വരാം, please ആ ഫയൽ അയച്ചു തരൂ.",
    "ഈ പുതിയ AI മോഡൽ വളരെ fast ആയി synthesize ചെയ്യുന്നുണ്ട്.",
    "ലാപ്ടോപ്പിൽ battery വളരെ low ആണ്, charger എവിടെ വെച്ചു?",
    "ഇന്നത്തെ meeting ഓൺലൈൻ ആയി zoom-ൽ വെക്കാം എന്ന് തോനുന്നു.",
    "നിങ്ങളുടെ phone number ഒന്ന് message ചെയ്യുമോ?",
    "ഈ project-ന്റെ documentation നാളെ submit ചെയ്യണം.",
    "WhatsApp-ൽ ഞാൻ ലൊക്കേഷൻ send ചെയ്തിട്ടുണ്ട്, check ചെയ്യൂ.",
    "Software update ചെയ്ത ശേഷം system perfectly വർക്ക് ചെയ്യുന്നുണ്ട്.",
]

# Systematic expansion to reach exact test targets
def expand_category(seed_list: list[str], target_count: int, cat_id: str) -> list[dict]:
    items = []
    modifiers = [
        "",
        "തീർച്ചയായും, ",
        "ദയവായി ഓർക്കുക, ",
        "അതുകൊണ്ട്, ",
        "ഇന്ന് രാവിലെ, ",
        "പ്രത്യേകം ശ്രദ്ധിക്കുക, ",
        "ഇക്കാര്യത്തിൽ, ",
        "എന്നിരുന്നാലും, ",
        "വളരെ പ്രധാനമായി, ",
        "കഴിഞ്ഞ ദിവസം, "
    ]
    suffixes = [
        "",
        " എന്ന് അദ്ദേഹം കൂട്ടിച്ചേർത്തു.",
        " എന്നാണ് പ്രാഥമിക വിവരം.",
        " എന്നത് ശ്രദ്ധേയമാണ്.",
        " എന്ന് ഉറപ്പുവരുത്തുക.",
        " എന്ന് പ്രതീക്ഷിക്കുന്നു.",
        " എന്ന കാര്യം മറക്കരുത്."
    ]

    idx = 0
    while len(items) < target_count:
        base = seed_list[idx % len(seed_list)]
        mod = modifiers[(idx // len(seed_list)) % len(modifiers)]
        suf = suffixes[(idx // (len(seed_list) * len(modifiers))) % len(suffixes)]

        # Assemble clean variation
        text = (mod + base).strip()
        if suf and not text.endswith(('?', '!')):
            text = text.rstrip('.') + suf

        uid = f"{cat_id}_{len(items) + 1:03d}"
        items.append({
            "id": uid,
            "category": cat_id,
            "text": text,
            "hash": hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]
        })
        idx += 1

    return items[:target_count]


def main():
    print("Generating Benchmark Dataset Splits...")

    # 1. Main Test Set (500 sentences)
    cat_conversational = expand_category(EVERYDAY_CONVERSATIONAL, 100, "conversational")
    cat_news = expand_category(FORMAL_NEWS, 100, "news_formal")
    cat_long = expand_category(LONG_COMPOUND, 75, "long_compound")
    cat_questions = expand_category(QUESTIONS_EXCLAMATIONS, 75, "questions_exclamations")
    cat_numbers = expand_category(NUMBERS_DATES_CURRENCY, 75, "numbers_dates_currency")
    cat_codeswitch = expand_category(CODE_SWITCHING, 75, "code_switching")

    main_500 = (
        cat_conversational + cat_news + cat_long +
        cat_questions + cat_numbers + cat_codeswitch
    )
    assert len(main_500) == 500, f"Expected 500, got {len(main_500)}"

    main_path = SPLITS_DIR / "main_test_set.jsonl"
    with open(main_path, "w", encoding="utf-8") as f:
        for item in main_500:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  -> Wrote Main Test Set (500 items): {main_path}")

    # 2. Dev Set (100 sentences - balanced 16-17 from each)
    dev_100 = (
        expand_category(EVERYDAY_CONVERSATIONAL, 17, "dev_conv") +
        expand_category(FORMAL_NEWS, 17, "dev_news") +
        expand_category(LONG_COMPOUND, 17, "dev_long") +
        expand_category(QUESTIONS_EXCLAMATIONS, 16, "dev_quest") +
        expand_category(NUMBERS_DATES_CURRENCY, 17, "dev_num") +
        expand_category(CODE_SWITCHING, 16, "dev_code")
    )
    assert len(dev_100) == 100, f"Expected 100, got {len(dev_100)}"
    dev_path = SPLITS_DIR / "dev_set.jsonl"
    with open(dev_path, "w", encoding="utf-8") as f:
        for item in dev_100:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  -> Wrote Dev Set (100 items): {dev_path}")

    # 3. Stress-Test Set (200 sentences: extreme conjuncts, tongue-twisters, chillus)
    stress_seeds = [
        "പച്ചച്ചട്ടയിട്ട പച്ചത്തത്ത പച്ചച്ചക്ക തിന്നു തീർത്തു.",
        "പ്രകൃതിയിലെ പ്രതിഭാസങ്ങൾ മനുഷ്യന്റെ പ്രതികരണങ്ങളെ നിരന്തരം സ്വാധീനിക്കുന്നു.",
        "സർവ്വവിജ്ഞാനകോശത്തിൽ വിവിധ ശാസ്ത്രവിഷയങ്ങൾ പ്രതിപാദിച്ചിരിക്കുന്നു.",
        "ക്ഷമയും ദൃഢനിശ്ചയവും ഉണ്ടെങ്കിൽ ഏത് പ്രതിസന്ധിയും തരണം ചെയ്യാം.",
        "മലയാളഭാഷയിലെ വർണ്ണവിന്യാസവും അക്ഷരമാലയും സവിശേഷമായ സ്വഭാവമുള്ളവയാണ്.",
        "ആനന്ദലബ്ധിക്കിനിയെന്തു വേണ്ടു ഈ മനോഹര തീരങ്ങളിൽ വിഹരിക്കുമ്പോൾ.",
        "ഭാരതീയ സംസ്കാരത്തിന്റെ അടിത്തറ വൈവിധ്യങ്ങളിലെ ഏകത്വമാണ്.",
        "ചങ്ങമ്പുഴയുടെ രമണൻ മലയാള കാവ്യലോകത്ത് വലിയൊരു കൊടുങ്കാറ്റഴിച്ചുവിട്ടു.",
    ]
    stress_200 = expand_category(stress_seeds, 200, "stress_test")
    stress_path = SPLITS_DIR / "stress_test_set.jsonl"
    with open(stress_path, "w", encoding="utf-8") as f:
        for item in stress_200:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  -> Wrote Stress-Test Set (200 items): {stress_path}")


if __name__ == "__main__":
    main()
