#!/usr/bin/env python3
"""Agreement between two graders' verdicts on the same run (score_run.py grades*.jsonl).

Prints, and writes to out.json: facts compared, raw agreement, Cohen's kappa, how often
each grader said conveyed, and the disagreements by fact rank (k1 = most important).

Usage: python3 compare_graders.py grades_a.jsonl grades_b.jsonl out.json
"""
import json
import sys
from collections import Counter


def load(path):
    return {g["unit_id"]: g["verdicts"] for g in map(json.loads, open(path)) if g["verdicts"] and not g["call"].get("error")}


def main(a_path, b_path, out_path):
    a, b = load(a_path), load(b_path)
    pairs = [(a[u][k], b[u][k], k) for u in a.keys() & b.keys() for k in a[u].keys() & b[u].keys()]
    n = len(pairs)
    agree = sum(x == y for x, y, _ in pairs)
    pa, pb = sum(x for x, _, _ in pairs) / n, sum(y for _, y, _ in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    result = {
        "a": a_path, "b": b_path, "rows": len(a.keys() & b.keys()), "facts": n,
        "agreement_pct": round(100 * agree / n, 1), "cohen_kappa": round((agree / n - expected) / (1 - expected), 3),
        "a_conveyed_pct": round(100 * pa, 1), "b_conveyed_pct": round(100 * pb, 1),
        "only_a_conveyed": sum(x and not y for x, y, _ in pairs), "only_b_conveyed": sum(y and not x for x, y, _ in pairs),
        "disagreements_by_rank": dict(sorted(Counter(k for x, y, k in pairs if x != y).items())),
    }
    open(out_path, "w").write(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
