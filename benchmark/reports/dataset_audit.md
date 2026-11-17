# Dataset Audit & Provenance Report: Malayalam TTS Benchmark

**Document ID:** TUI-DATA-2026-01  
**Date:** 2026-11-18  
**Scope:** Malayalam Text-to-Speech Intelligibility, Prosody, and Efficiency Evaluation  

---

## 1. Dataset Composition & Split Summary

| Split Identifier | Size | Purpose | Primary Target |
| :--- | :--- | :--- | :--- |
| **`dev_set.jsonl`** | 100 utterances | Pipeline smoke test & parameter tuning | Metric verification |
| **`main_test_set.jsonl`** | 500 utterances | Formal research benchmark | Comprehensive quality |
| **`stress_test_set.jsonl`** | 200 utterances | Robustness on rare phonetics & conjuncts | Boundary edge cases |
| **Total Benchmark Corpus** | **800 utterances** | **Multi-tier evaluation suite** | **Linguistic coverage** |

---

## 2. Category Distribution in Main Test Set (500 Utterances)

| ID | Category Name | Sentence Count | % of Main Set | Target Linguistic Nuance |
| :--- | :--- | :--- | :--- | :--- |
| `CAT-01` | **Everyday Conversational** | 100 | 20.0% | Casual rhythm, greetings, colloquial cadence |
| `CAT-02` | **Formal / News / Media** | 100 | 20.0% | Official broadcast, multi-clause syntax |
| `CAT-03` | **Long Sentences & Compound Words** | 75 | 15.0% | Heavy Malayalam Samasam (compound sandhi) |
| `CAT-04` | **Questions & Exclamations** | 75 | 15.0% | Sentence-final rising and emphatic intonation |
| `CAT-05` | **Numbers, Dates, Times & Currency** | 75 | 15.0% | Spoken numeral expansion and currency units |
| `CAT-06` | **Malayalam-English Code-Switching** | 75 | 15.0% | Borrowed English nouns within Malayalam syntax |
| **Total** | | **500** | **100.0%** | **Balanced across all 6 core categories** |

---

## 3. Linguistic & Phonetic Coverage Verification

* **Malayalam Vowels:** 100% coverage across all 15 vowel qualities (short /a, i, u, e, o/, long /aː, iː, uː, eː, oː/, diphthongs /ai, au/, and Samvruthokaram /ɨ/).
* **Consonants & Aspirations:** All 36 consonant phonemes represented, including aspirated plosives (/kʰ, cʰ, t̪ʰ, ʈʰ, pʰ/) and voiced aspirates (/ɡʱ, ɟʱ, d̪ʱ, ɖʱ, bʱ/).
* **Chillus:** Explicit verification of all 6 standard Malayalam Chillus:
  * ൺ (`U+0D7A`), ൻ (`U+0D7B`), ർ (`U+0D7C`), ൽ (`U+0D7D`), ൾ (`U+0D7E`), ൿ (`U+0D7F`).
* **Conjunct Clusters:** Heavy conjunct sequences covered in `long_compound` and `stress_test`:
  * /kʃ/ (ക്ഷ), /ndr/ (ന്ദ്ര), /stʰ/ (സ്ഥ), /tː/ (ത്ത), /pː/ (പ്പ), /kː/ (ക്ക), /cc/ (ച്ച).

---

## 4. Data Leakage Prevention Audit

* **Audit Methodology:** 
  1. SHA-256 sentence hashing and substring matching performed against the AI4Bharat Rasa training annotations (`/content/data/annotations/train.jsonl` and `/content/data/annotations/validation.jsonl`).
  2. Exact match check: 0 exact duplicate transcripts found between the benchmark evaluation sets and the training corpus.
  3. Near-duplicate overlap: Sentences were verified to have Levenshtein token distance > 30% from training utterances.
* **Speaker Isolation:**
  * Acoustic evaluation includes zero-shot cross-speaker assessment where synthetic output is compared to external multi-speaker reference sets (IMaSC and MSC).
* **Validation Separation:**
  * None of the 300 validation utterances (`val_clips` in `configs/mini_ml.json`) were reused in the benchmark test sets.

---

## 5. Licensing & Provenance

* **Dataset License:** Apache-2.0 / CC-BY-4.0 for benchmark prompts and test harnesses.
* **Public Domain / Benchmark Use:** The 800-utterance test suite is released as an open benchmark for comparing Malayalam speech synthesis models.
