#!/usr/bin/env python3
"""Minimal-pair round, step 2: a second judge over every field Sonnet flagged. Same rubric and call (grade_faithfulness),
but only the flagged facts of each sample are sent, so one call per flagged sample. Resumable through
--out; stops at --budget. A field is a negative only if both judges flag it (decided downstream).
usage: tiebreak.py set_dir out.jsonl grades.jsonl... [--grader us.moonshotai.kimi-k3] [--budget 12] [--seed 42]"""
import argparse, csv, json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "harness" / "tools"))
import run_model as rm  # noqa: E402
import score_abstracts as sa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("set_dir"); ap.add_argument("out"); ap.add_argument("grades", nargs="+")
ap.add_argument("--grader", default="us.moonshotai.kimi-k3"); ap.add_argument("--region", default="us-east-2")
ap.add_argument("--budget", type=float, default=12.0); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
args = SimpleNamespace(grader=a.grader, region=a.region)
rm.set_task("abstract-facts"); csv.field_size_limit(sys.maxsize)
inputs = {r["unit_id"]: r for r in csv.DictReader(open(Path(a.set_dir) / "inputs.csv", newline=""))}
done = {(g["unit_id"], g["answer"]) for g in map(json.loads, open(a.out))} if Path(a.out).exists() else set()
spent = [sum(g.get("cost_usd", 0) for g in map(json.loads, open(a.out))) if Path(a.out).exists() else 0.0]
todo = []
for f in a.grades:
    for g in map(json.loads, open(f)):
        if g["kind"] != "faith" or not g["verdicts"]: continue
        bad = [k for k, v in g["verdicts"].items() if v != "supported"]
        if bad and (g["unit_id"], g["answer"]) not in done and g["unit_id"] in inputs:
            todo.append((g["unit_id"], g["answer"], bad)); done.add((g["unit_id"], g["answer"]))
random.Random(a.seed).shuffle(todo)
print(f"to tiebreak: {len(todo)} flagged samples; spent ${spent[0]:.2f} of ${a.budget}", flush=True)
lock, fh, n, stopped = threading.Lock(), open(a.out, "a"), [0], [False]

def one(item):
    uid, akey, bad = item
    with lock:
        if spent[0] >= a.budget: stopped[0] = True; return
    facts = [(i, t) for i, t in sa.model_facts(json.loads(akey)) if i in bad]
    v, _, rec = sa.grade_faithfulness(inputs[uid]["abstract"], facts, args)
    g = {"unit_id": uid, "answer": akey, "kind": "faith", "verdicts": v, "cost_usd": round((rec or {}).get("cost_usd") or 0, 6),
         "grader": (rec or {}).get("model"), "error": (rec or {}).get("error", ""), "fields": bad}
    with lock:
        spent[0] += g["cost_usd"]; n[0] += 1
        fh.write(json.dumps(g, ensure_ascii=False) + "\n"); fh.flush()
        if n[0] % 200 == 0: print(f"{n[0]} judged, ${spent[0]:.2f}", flush=True)

with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(one, todo))
print(json.dumps({"judged": n[0], "spent": round(spent[0], 2), "budget_hit": stopped[0]}), flush=True)
