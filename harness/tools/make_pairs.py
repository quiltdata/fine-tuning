#!/usr/bin/env python3
"""Turn a teacher run into prompt/response pairs for training or evaluating another model.

One JSON line per batch, holding exactly the system prompt and user message the run sent (without
any required facts, so the student never sees them), the output rows as the assistant response,
the model that served the call, and how many of its rows passed check_outputs.py.

The task, prompt and batch size come from the run's run.json, so batches are rebuilt exactly as
the run made them; it stops if the prompt, examples or schema have changed since the run.

--mixed-batches SEED (abstract-facts) regroups the run's rows into batches of 1 to --max-batch
rows, sizes drawn with that seed, so the student learns any batch size. A regrouped batch's
response is the teacher's rows for exactly those inputs.

Usage: python3 make_pairs.py inputs.csv run_dir pairs.jsonl [--mixed-batches SEED [--max-batch 10]]
"""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model as rm  # noqa: E402  (same task definitions as run_claude.py, without the Anthropic SDK)


def set_task_from_run(run, root=rm.tasks.ROOT):
    """Makes the run's task active and checks its files still match what the run used."""
    task = rm.set_task(run.get("task", "file-summary"), run.get("prompt") if run.get("task") else None,
                       allow_no_examples=bool(run.get("examples_allowed_missing")), root=root)
    now = task.fingerprint(run.get("prompt"))
    changed = [k for k in ("prompt_sha256", "examples_sha256", "schema_sha256") if k in run and run[k] != now[k]]
    if changed:
        raise SystemExit(f"{', '.join(changed)} differ from the run's; pairs would not match what the teacher saw")
    return task


def mixed(rows, seed, max_batch):
    """Consecutive groups of 1..max_batch rows, sizes drawn with `seed`."""
    rng, out, i = random.Random(seed), [], 0
    while i < len(rows):
        n = rng.randint(1, max_batch)
        out.append((f"mixed-{len(out)}", rows[i:i + n]))
        i += n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--mixed-batches", type=int, metavar="SEED", help="regroup rows into batches of 1..--max-batch")
    ap.add_argument("--max-batch", type=int, default=10)
    a = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    run = Path(a.run_dir)
    info = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    set_task_from_run(info)
    rows = list(csv.DictReader(open(a.inputs, newline="")))
    outputs = {}
    for o in csv.DictReader(open(run / "outputs.csv", newline="")):
        outputs[o["unit_id"]] = {k: json.loads(v) if k in rm.LIST_COLUMNS else v for k, v in o.items()}
    calls = list(csv.DictReader(open(run / "calls.csv", newline="")))
    served = {}
    for c in calls:
        for u in json.loads(c["unit_ids"]) if c.get("unit_ids") else []:
            served[u] = c["model_served"]
    by_batch = {int(c["batch"]): c["model_served"] for c in calls}
    failed = set(re.findall(r"^FAIL  (\S+)", (run / "check.txt").read_text(), re.M))

    if a.mixed_batches is not None:
        groups = mixed([r for r in rows if r["unit_id"] in outputs], a.mixed_batches, a.max_batch)
    else:
        groups = [(i, b) for i, (_, b) in enumerate(rm.batches(rows, size=info.get("batch_size")))]
    written = 0
    with open(a.out, "w") as fh:
        for key, batch in groups:
            ids = [r["unit_id"] for r in batch]
            if not all(u in outputs for u in ids):
                continue
            teachers = sorted({served.get(u) or by_batch.get(key, "") for u in ids})
            fh.write(json.dumps({
                "batch": key, "teacher_model": ",".join(t for t in teachers if t),
                "rows": len(ids), "rows_passing_check": sum(u not in failed for u in ids),
                "system": rm.SYSTEM, "user": rm.user_message(batch),
                "assistant": json.dumps({"rows": [outputs[u] for u in ids]}, ensure_ascii=False),
            }, ensure_ascii=False) + "\n")
            written += 1
    print(f"{written} pairs written to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
