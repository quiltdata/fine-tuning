#!/usr/bin/env python3
"""Turn a summarizer run into prompt/response pairs for training or evaluating another model.

One JSON line per batch (one call), holding exactly the system prompt and user message
run_claude.py sent, the output rows as the assistant response, the model that served
the call, and how many of its rows passed check_outputs.py.

Usage: python3 make_pairs.py inputs.csv run_dir pairs.jsonl
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_model import LIST_COLUMNS, SYSTEM, batches, user_message  # noqa: E402  (same as run_claude's, without the Anthropic SDK)


def main(inputs_path, run_dir, out_path):
    csv.field_size_limit(sys.maxsize)
    run = Path(run_dir)
    rows = list(csv.DictReader(open(inputs_path, newline="")))
    outputs = {}
    for o in csv.DictReader(open(run / "outputs.csv", newline="")):
        outputs[o["unit_id"]] = {k: json.loads(v) if k in LIST_COLUMNS else v for k, v in o.items()}
    calls = {int(c["batch"]): c for c in csv.DictReader(open(run / "calls.csv", newline=""))}
    failed = set(re.findall(r"^FAIL  (\S+)", (run / "check.txt").read_text(), re.M))
    written = 0
    with open(out_path, "w") as fh:
        for i, (key, batch) in enumerate(batches(rows)):
            ids = [r["unit_id"] for r in batch]
            if not all(u in outputs for u in ids):
                continue
            fh.write(json.dumps({
                "batch": i, "prefix": key[0], "file_type": key[1],
                "teacher_model": calls[i]["model_served"],
                "rows": len(ids), "rows_passing_check": sum(u not in failed for u in ids),
                "system": SYSTEM, "user": user_message(batch),
                "assistant": json.dumps({"rows": [outputs[u] for u in ids]}, ensure_ascii=False),
            }, ensure_ascii=False) + "\n")
            written += 1
    print(f"{written} pairs written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
