#!/usr/bin/env python3
"""Filter teacher pairs (make_pairs.py) into a training set, by code only.

Keeps a pair only when every one of its rows passed check_outputs.py: schema, lengths,
one output per input, and every evidence quote verbatim in the input. No model judges
which examples to keep.

Refuses, rather than filters:
- pairs whose teacher is a Claude model or a closed OpenAI model: their terms bar training on
  their outputs without authorization (METHODOLOGY.md §3). Open-weight gpt-oss is allowed.
- pairs of a task with published prompts (abstract-facts) that weren't made with the published
  prompt and examples, or whose system prompt differs from the published one now (tasks.py).

Writes train.jsonl ({"system", "user", "assistant"} per line, the format the Lightning
LoRA job reads) and prints how many pairs and rows were kept.

Usage: python3 make_sft.py pairs.jsonl train.jsonl
"""
import json
import sys

import tasks


def closed_teacher(model):
    m = (model or "").lower()
    return "claude" in m or ("openai." in m and "gpt-oss" not in m)


def unpublished(pairs, data=None):
    """Pairs of a published-prompt task that don't match the published prompt and examples."""
    published, bad = {}, []
    for p in pairs:
        name = p.get("task", "file-summary")
        if name == "file-summary":
            continue
        if name not in published:
            task = tasks.get(name, data=data)
            fp = task.fingerprint()
            published[name] = (fp, task.system() if fp["published"] else None)
        fp, system = published[name]
        if not (p.get("published") and fp["published"] and p.get("prompt_sha256") == fp["prompt_sha256"]
                and p.get("examples_sha256") == fp["examples_sha256"] and p.get("system") == system):
            bad.append(p)
    return bad


def main(pairs_path, out_path, data=None):
    pairs = [json.loads(l) for l in open(pairs_path)]
    closed = [p for p in pairs if any(closed_teacher(t) for t in (p.get("teacher_model") or "").split(","))]
    if closed:
        sys.exit(f"{len(closed)} pairs have a Claude or closed OpenAI teacher; refusing to build training data from them")
    bad = unpublished(pairs, data)
    if bad:
        sys.exit(f"{len(bad)} pairs weren't made with the published prompt and examples; refusing to build training data")
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
