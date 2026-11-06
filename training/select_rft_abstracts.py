#!/usr/bin/env python3
"""Choose reward-round training targets for the abstract task (one abstract per call).

Per training abstract, the model's samples (sample_rft.py) are tried in random order, at most
--max-tries distinct ones that pass check_outputs.py's structural check. Each is graded by an
open-weight model (Kimi K3 by default) with the scorer's two questions: recall against the
training key first, and, only if every key fact is conveyed, faithfulness against the abstract.
The first sample that conveys every key fact and states nothing overstated or unsupported is the
target. An abstract with none keeps the teacher's answer (its line in the teacher's train.jsonl).
So the round teaches the model its own answers where they are perfect by the grader, and the
teacher's elsewhere.

Grading stops once --budget dollars are spent; the remaining abstracts keep the teacher's answer.
Verdicts are cached in --grades, so a rerun costs nothing.

Usage: python3 select_rft_abstracts.py set_dir samples.jsonl teacher_train.jsonl out_train.jsonl
                                       --grades grades.jsonl [--grader us.moonshotai.kimi-k3]
                                       [--max-tries 3] [--budget 40] [--workers 16] [--seed 0]
"""
import argparse
import csv
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent / "harness"
sys.path.insert(0, str(HARNESS / "tools"))
import run_model as rm  # noqa: E402
import score_abstracts as sa  # noqa: E402
from check_outputs import abstract_problems  # noqa: E402


def candidates(uid, samples):
    """Distinct samples (normalized as the scorer reads them, without the id) that pass the structural check."""
    seen, out = set(), []
    for raw in samples:
        try:
            rows = rm.TASK.rows_from_reply(json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            continue
        if len(rows) != 1 or not isinstance(rows[0], dict):
            continue
        row = rm.TASK.normalize({"unit_id": uid, "about": rows[0].get("about"), "findings": rows[0].get("findings")})
        try:
            if abstract_problems(row):
                continue
        except (TypeError, KeyError, AttributeError):
            continue
        ans = rm.TASK.assistant_json([row])
        key = json.dumps(ans, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(ans)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_dir")
    ap.add_argument("samples")
    ap.add_argument("teacher_train")
    ap.add_argument("out")
    ap.add_argument("--grades", required=True)
    ap.add_argument("--grader", default="us.moonshotai.kimi-k3")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--max-tries", type=int, default=3)
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rm.set_task("abstract-facts")
    csv.field_size_limit(sys.maxsize)
    sd = Path(a.set_dir)
    inputs = {r["unit_id"]: r for r in csv.DictReader(open(sd / "inputs.csv", newline=""))}
    keys = {k["unit_id"]: k for k in map(json.loads, open(sd / "reference" / "facts.jsonl"))}
    by_text = {rm.TASK.user_message([r]): u for u, r in inputs.items()}  # the student's user turn
    teacher = {}
    for line in open(a.teacher_train):
        t = json.loads(line)
        if t["user"] in by_text:
            teacher[by_text[t["user"]]] = t
    cache = {}
    if Path(a.grades).exists():
        for g in map(json.loads, open(a.grades)):
            cache[(g["unit_id"], g["answer"])] = g
    lock, spent = threading.Lock(), [sum(g.get("cost_usd", 0) for g in cache.values())]
    gfh = open(a.grades, "a")

    def grade(uid, ans):
        akey = json.dumps(ans, sort_keys=True, ensure_ascii=False)
        if (uid, akey) in cache:
            return cache[(uid, akey)]
        with lock:
            if spent[0] >= a.budget:
                return None
        facts = sa.model_facts(ans)
        recall, r1 = sa.grade_recall(keys[uid]["key_facts"], facts, a)
        faith, r2 = None, None
        if recall is not None and all(recall.values()):
            faith, _, r2 = sa.grade_faithfulness(inputs[uid]["abstract"], facts, a)
        cost = sum((r or {}).get("cost_usd") or 0 for r in (r1, r2))
        g = {"unit_id": uid, "answer": akey, "recall": recall, "faithfulness": faith, "cost_usd": round(cost, 6),
             "perfect": bool(recall is not None and all(recall.values()) and faith
                             and all(v == "supported" for v in faith.values()))}
        with lock:
            spent[0] += cost
            cache[(uid, akey)] = g
            gfh.write(json.dumps(g, ensure_ascii=False) + "\n")
            gfh.flush()
        return g

    def choose(rec):
        uid = rec["unit_ids"][0] if len(rec.get("unit_ids") or []) == 1 else None
        if uid not in teacher or uid not in keys:
            return uid, None, "no teacher or key"
        cands = candidates(uid, rec.get("samples") or [])
        random.Random(f"{a.seed}-{uid}").shuffle(cands)
        for ans in cands[: a.max_tries]:
            g = grade(uid, ans)
            if g is None:
                return uid, None, "budget"
            if g["perfect"]:
                return uid, ans, "own"
        return uid, None, "teacher" if cands else "teacher (no valid sample)"

    recs = [json.loads(l) for l in open(a.samples)]
    with ThreadPoolExecutor(a.workers) as pool:
        chosen = list(pool.map(choose, recs))
    counts = {}
    with open(a.out, "w") as fh:
        for uid, ans, why in chosen:
            counts[why] = counts.get(why, 0) + 1
            if uid not in teacher:
                continue
            t = dict(teacher[uid])
            if ans is not None:  # serialized as make_pairs.py serializes the teacher's answer
                t["assistant"] = json.dumps(ans, ensure_ascii=False)
            fh.write(json.dumps({k: t[k] for k in ("system", "user", "assistant")}, ensure_ascii=False) + "\n")
    print(json.dumps({"abstracts": len(recs), "chosen": counts, "grader_cost_usd": round(spent[0], 2)}, indent=1))


if __name__ == "__main__":
    main()
