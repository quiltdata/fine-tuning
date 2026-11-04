#!/usr/bin/env python3
"""Split a summary set into train and test by article, and exclude refused articles.

An article is one prefix (one PMC article version) with all its rows, so an article's
supplements never straddle train and test. Articles with any call that a fallback model
served (the reference run's calls.csv) are excluded from both: other models would be
scored on inputs known to trigger refusals. The rest are shuffled with a fixed seed and
the first round(fraction x n) go to test.

Writes split.csv: prefix, pmcid, rows, split (train | test | excluded), reason.

Usage: python3 make_split.py inputs.csv calls.csv split.csv [--test-fraction 0.2] [--seed 20261024]
"""
import argparse
import csv
import random
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("calls")
    ap.add_argument("out")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    rows = Counter(r["prefix"] for r in csv.DictReader(open(args.inputs, newline="")))
    refused = {c["prefix"] for c in csv.DictReader(open(args.calls, newline="")) if c["fallback"]}
    eligible = sorted(p for p in rows if p not in refused)
    random.Random(args.seed).shuffle(eligible)
    test = set(eligible[: round(args.test_fraction * len(eligible))])

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prefix", "pmcid", "rows", "split", "reason"])
        w.writeheader()
        for p in sorted(rows):
            split, reason = ("excluded", "reference run fell back on a refusal") if p in refused else \
                            ("test", f"seed {args.seed}") if p in test else ("train", "")
            w.writerow({"prefix": p, "pmcid": p.rstrip("/").rsplit("/", 1)[-1], "rows": rows[p], "split": split, "reason": reason})

    by = Counter()
    for p in rows:
        by["excluded" if p in refused else "test" if p in test else "train"] += 1
    print({k: f"{by[k]} articles, {sum(rows[p] for p in rows if (('excluded' if p in refused else 'test' if p in test else 'train') == k))} rows"
           for k in ("train", "test", "excluded")})


if __name__ == "__main__":
    main()
