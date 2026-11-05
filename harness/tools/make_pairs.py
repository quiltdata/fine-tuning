#!/usr/bin/env python3
"""Turn a teacher run into prompt/response pairs for training or evaluating another model.

One JSON line per batch, holding exactly the system prompt and user message the run sent (without
any required facts, so the student never sees them), the output rows as the assistant response,
the model that served the call, and how many of its rows passed check_outputs.py.

The task, prompt and batch size come from the run's run.json, so batches are rebuilt exactly as
the run made them; it stops if the prompt, examples or schema have changed since the run.

--mixed-batches SEED regroups the run's rows into batches of 1 to --max-batch
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


def set_task_from_run(run, **where):
    """Makes the run's task active and checks the run may become training data.

    A task with published prompts (abstract-facts) only takes runs made with the published prompt
    and examples, unchanged since: a draft run, or one whose files have changed, is refused. The
    file-summary task keeps its earlier rule: its files must match the run's hashes where recorded.
    `where` (root=, data=) is for tests."""
    name = run.get("task", "file-summary")
    if name != "file-summary" and not run.get("published"):
        raise SystemExit("the run used a draft prompt or examples (published: false); "
                         "only runs on the published prompt and examples can become training data")
    prompt = run.get("prompt") if name == "file-summary" and run.get("task") else None
    task = rm.set_task(name, prompt, **where)
    now = rm.fingerprint()
    if name != "file-summary" and not now["published"]:
        raise SystemExit("the published prompt or examples are missing here")
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

    if a.mixed_batches is not None:  # only rows that passed the check, so one failure doesn't discard a group
        groups = mixed([r for r in rows if r["unit_id"] in outputs and r["unit_id"] not in failed], a.mixed_batches, a.max_batch)
    else:  # exactly as the run batched them
        groups = [(i, b) for i, (_, b) in enumerate(rm.batches(rows, info.get("per_file", False), info.get("batch_size")))]
    written = 0
    fp = rm.fingerprint()
    with open(a.out, "w") as fh:
        for key, batch in groups:
            batch = [r for r in batch if r["unit_id"] in outputs]  # an abstract without an answer drops out alone
            if not batch:
                continue
            ids = [r["unit_id"] for r in batch]
            teachers = sorted({served.get(u) or by_batch.get(key, "") for u in ids})
            fh.write(json.dumps({
                "task": fp["task"], "prompt_sha256": fp["prompt_sha256"], "examples_sha256": fp["examples_sha256"],
                "published": fp["published"],
                "batch": key, "teacher_model": ",".join(t for t in teachers if t),
                "rows": len(ids), "rows_passing_check": sum(u not in failed for u in ids),
                "system": rm.SYSTEM, "user": rm.user_message(batch),
                "assistant": json.dumps(rm.TASK.assistant_json([outputs[u] for u in ids]), ensure_ascii=False),
            }, ensure_ascii=False) + "\n")
            written += 1
    print(f"{written} pairs written to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
