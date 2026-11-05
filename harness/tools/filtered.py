#!/usr/bin/env python3
"""Add the papers a run's content filter blocked to a set's excluded.csv.

A paper counts when the filter blocked it sent on its own (a one-row call with filtered > 0; the
harness retries a blocked batch one row at a time), even if a retry then passed, because the filter
isn't deterministic. Papers only in a blocked batch, which passed alone, don't count. Every model's
scores drop the excluded papers (score_abstracts.py), so the models are compared on the same papers.

Writes excluded.csv (unit_id, reason, run), keeping what is already there.

Usage: python3 filtered.py set_dir run_dir [run_dir ...]
"""
import csv
import json
import sys
from pathlib import Path

COLUMNS = ["unit_id", "reason", "run"]


def blocked(run_dir):
    """Unit ids the filter blocked when sent alone in this run."""
    out = []
    for c in csv.DictReader(open(Path(run_dir) / "calls.csv", newline="")):
        ids = json.loads(c["unit_ids"]) if c.get("unit_ids") else []
        if len(ids) == 1 and int(c.get("filtered") or 0) > 0:
            out.append(ids[0])
    return out


def main(set_dir, *run_dirs):
    path = Path(set_dir) / "excluded.csv"
    rows = list(csv.DictReader(open(path, newline=""))) if path.exists() else []
    have = {r["unit_id"] for r in rows}
    added = 0
    for rd in run_dirs:
        model = json.loads((Path(rd) / "run.json").read_text()).get("model", "")
        for u in blocked(rd):
            if u not in have:
                rows.append({"unit_id": u, "reason": f"content filter blocked it on {model}", "run": Path(rd).name})
                have.add(u)
                added += 1
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"excluded": len(rows), "added": added, "file": str(path)}))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
