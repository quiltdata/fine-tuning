#!/usr/bin/env python3
"""Minimal-pair round, step 1: Sonnet faithfulness over the samples the reward-round selector never judged. Same rubric, same call
(score_abstracts.grade_faithfulness), same record shape as grades.two_judge.jsonl. Resumable: records
already in --grades or in the selector's file are skipped. Stops at --budget dollars for this file.
usage: grade_rest.py set_dir samples.jsonl selector_grades.jsonl out_grades.jsonl [--budget 32] [--workers 8]"""
import argparse, csv, json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "harness" / "tools")); sys.path.insert(0, str(REPO / "training"))
import run_model as rm  # noqa: E402
import score_abstracts as sa  # noqa: E402
from select_rft_abstracts import candidates  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("set_dir"); ap.add_argument("samples"); ap.add_argument("selector_grades"); ap.add_argument("out")
ap.add_argument("--grader", default="us.anthropic.claude-sonnet-5-5"); ap.add_argument("--region", default="us-east-2")
ap.add_argument("--budget", type=float, default=32.0); ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
args = SimpleNamespace(grader=a.grader, region=a.region)
rm.set_task("abstract-facts"); csv.field_size_limit(sys.maxsize)
inputs = {r["unit_id"]: r for r in csv.DictReader(open(Path(a.set_dir) / "inputs.csv", newline=""))}
done = set()
for f in (a.selector_grades, a.out):
    if Path(f).exists():
        for g in map(json.loads, open(f)):
            if g["kind"] == "faith": done.add((g["unit_id"], g["answer"]))
spent = [sum(g.get("cost_usd", 0) for g in map(json.loads, open(a.out))) if Path(a.out).exists() else 0.0]
todo = []
for rec in map(json.loads, open(a.samples)):
    uid = rec["unit_ids"][0] if len(rec.get("unit_ids") or []) == 1 else None
    if uid not in inputs: continue
    for ans in candidates(uid, rec.get("samples") or []):
        akey = json.dumps(ans, sort_keys=True, ensure_ascii=False)
        if (uid, akey) not in done: todo.append((uid, ans, akey)); done.add((uid, akey))
print(f"to grade: {len(todo)} samples; already spent ${spent[0]:.2f} of ${a.budget}", flush=True)
lock, fh, n, stopped = threading.Lock(), open(a.out, "a"), [0], [False]

def one(item):
    uid, ans, akey = item
    with lock:
        if spent[0] >= a.budget: stopped[0] = True; return
    v, _, rec = sa.grade_faithfulness(inputs[uid]["abstract"], sa.model_facts(ans), args)
    g = {"unit_id": uid, "answer": akey, "kind": "faith", "verdicts": v, "cost_usd": round((rec or {}).get("cost_usd") or 0, 6),
         "grader": (rec or {}).get("model"), "error": (rec or {}).get("error", "")}
    with lock:
        spent[0] += g["cost_usd"]; n[0] += 1
        fh.write(json.dumps(g, ensure_ascii=False) + "\n"); fh.flush()
        if n[0] % 200 == 0: print(f"{n[0]} graded, ${spent[0]:.2f}", flush=True)

with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(one, todo))
print(json.dumps({"graded": n[0], "spent": round(spent[0], 2), "budget_hit": stopped[0]}), flush=True)
