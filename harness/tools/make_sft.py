#!/usr/bin/env python3
"""Filter teacher pairs (make_pairs.py) into a training set, by code only.

Keeps a pair only when every one of its rows passed check_outputs.py: schema, lengths,
one output per input, and every evidence quote verbatim in the input. No model judges
which examples to keep. Refuses pairs whose teacher is a Claude model: Anthropic's Usage
Policy bars training on Claude outputs without its authorization (METHODOLOGY.md §3).

Writes train.jsonl ({"system", "user", "assistant"} per line, the format the Lightning
LoRA job reads) and prints how many pairs and rows were kept.

Usage: python3 make_sft.py pairs.jsonl train.jsonl
"""
import json
import sys


def main(pairs_path, out_path):
    pairs = [json.loads(l) for l in open(pairs_path)]
    claude = [p for p in pairs if "claude" in (p.get("teacher_model") or "").lower()]
    if claude:
        sys.exit(f"{len(claude)} pairs have a Claude teacher; refusing to build training data from them")
    kept = [p for p in pairs if p["rows"] == p["rows_passing_check"]]
    with open(out_path, "w") as fh:
        for p in kept:
            fh.write(json.dumps({k: p[k] for k in ("system", "user", "assistant")}, ensure_ascii=False) + "\n")
    print(json.dumps({"pairs": len(pairs), "kept": len(kept), "rows_kept": sum(p["rows"] for p in kept),
                      "rows_dropped": sum(p["rows"] for p in pairs) - sum(p["rows"] for p in kept)}))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
