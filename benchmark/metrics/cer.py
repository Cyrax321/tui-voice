"""INT-01: Malayalam Grapheme Character Error Rate (CER).

Calculates Levenshtein edit distance operating on orthographic grapheme clusters
(Aksharas) rather than raw bytes or uncombined Unicode code points.

Formula:
  CER (%) = 100 * (Substitutions + Deletions + Insertions) / Reference Graphemes
"""
from typing import Tuple
from ..text.graphemes import get_grapheme_clusters
from ..text.normalization import normalize_malayalam


def levenshtein_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> Tuple[int, int, int, int]:
    """Compute (distance, substitutions, deletions, insertions) between two token sequences."""
    n = len(ref_tokens)
    m = len(hyp_tokens)

    # DP table: dp[i][j] stores (cost, subs, dels, ins)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        ops[i][0] = (0, i, 0)  # i deletions

    for j in range(1, m + 1):
        dp[0][j] = j
        ops[0][j] = (0, 0, j)  # j insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = ops[i - 1][j - 1]
            else:
                sub_cost = dp[i - 1][j - 1] + 1
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1

                min_cost = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = min_cost

                if min_cost == sub_cost:
                    s, d, ins = ops[i - 1][j - 1]
                    ops[i][j] = (s + 1, d, ins)
                elif min_cost == del_cost:
                    s, d, ins = ops[i - 1][j]
                    ops[i][j] = (s, d + 1, ins)
                else:
                    s, d, ins = ops[i][j - 1]
                    ops[i][j] = (s, d, ins + 1)

    dist = dp[n][m]
    subs, dels, inss = ops[n][m]
    return dist, subs, dels, inss


def calculate_grapheme_cer(ref: str, hyp: str) -> dict:
    """Calculate Grapheme Character Error Rate between reference and hypothesis strings."""
    norm_ref = normalize_malayalam(ref, strip_punct=True)
    norm_hyp = normalize_malayalam(hyp, strip_punct=True)

    ref_clusters = [c for c in get_grapheme_clusters(norm_ref) if not c.isspace()]
    hyp_clusters = [c for c in get_grapheme_clusters(norm_hyp) if not c.isspace()]

    ref_len = len(ref_clusters)
    if ref_len == 0:
        return {
            "cer": 0.0 if len(hyp_clusters) == 0 else 100.0,
            "ref_graphemes": 0,
            "hyp_graphemes": len(hyp_clusters),
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_clusters),
        }

    dist, subs, dels, inss = levenshtein_distance(ref_clusters, hyp_clusters)
    cer = (dist / ref_len) * 100.0

    return {
        "cer": round(cer, 4),
        "ref_graphemes": ref_len,
        "hyp_graphemes": len(hyp_clusters),
        "substitutions": subs,
        "deletions": dels,
        "insertions": inss,
    }
