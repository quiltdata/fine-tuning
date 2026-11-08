#!/usr/bin/env python3
"""Minimal-pair round, step 3: lost-condition pairs written by Kimi K3. For a chosen (Sonnet-passed) finding, Kimi rewrites it
dropping one qualifier the abstract attaches (condition, population restriction, hedge, non-significance,
direction/comparison scope) and names the phrase it removed. Kept only if the removed phrase is in the
original fact, the edit is shorter and still close to the original. One pair per abstract, finding f1 or f2
chosen at random (seed). Output: train.py pairs with provenance. Resumable through --raw; stops at --budget.
usage: lostcond.py set_dir chosen_train.jsonl out_pairs.jsonl --raw raw.jsonl [--budget 6] [--max 700] [--seed 42]"""
import argparse, csv, difflib, json, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "harness" / "tools"))
from bedrock_tool import call_tool  # noqa: E402

SYSTEM = """You edit one sentence from a summary of a scientific abstract to make it subtly LESS faithful, for use as a
training example of what not to write. The sentence states a finding. Remove exactly ONE qualifier that the abstract
attaches to the finding and that the sentence currently carries: a condition (dose, timepoint, setting, model system),
a population or sample restriction, a hedge ("may", "suggest", "was associated with"), a non-significance or
magnitude qualifier, or the scope of a comparison. The edited sentence must read as a confident, natural claim that
the abstract does NOT fully support, must keep the same subject and the same numbers, and must change nothing else.
If the sentence carries no such qualifier, return kind "none" and copy the sentence unchanged."""
SCHEMA = {"type": "object", "required": ["kind", "removed", "edited"], "properties": {
    "kind": {"enum": ["condition", "population", "hedge", "nonsignificance", "scope", "none"]},
    "removed": {"type": "string", "description": "the words removed or weakened, quoted from the original sentence"},
    "edited": {"type": "string"}}}

ap = argparse.ArgumentParser()
ap.add_argument("set_dir"); ap.add_argument("chosen"); ap.add_argument("out"); ap.add_argument("--raw", required=True)
ap.add_argument("--model", default="us.moonshotai.kimi-k3"); ap.add_argument("--region", default="us-east-2")
ap.add_argument("--budget", type=float, default=6.0); ap.add_argument("--max", type=int, default=700)
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
csv.field_size_limit(sys.maxsize)
inputs = {r["unit_id"]: r for r in csv.DictReader(open(Path(a.set_dir) / "inputs.csv", newline=""))}
by_text = {}
import run_model as rm  # noqa: E402
rm.set_task("abstract-facts")
for u, r in inputs.items(): by_text[rm.TASK.user_message([r])] = u
rows = []
for t in map(json.loads, open(a.chosen)):
    uid = by_text.get(t["user"])
    if not uid: continue
    ans = json.loads(t["assistant"])
    fs = [i for i, f in enumerate(ans.get("findings") or []) if isinstance(f, dict) and f.get("fact")]
    if fs: rows.append((uid, t, ans, fs))
rnd = random.Random(a.seed); rnd.shuffle(rows); rows = rows[: a.max]
done = {g["unit_id"] for g in map(json.loads, open(a.raw))} if Path(a.raw).exists() else set()
spent = [sum(g.get("cost_usd", 0) for g in map(json.loads, open(a.raw))) if Path(a.raw).exists() else 0.0]
todo = [r for r in rows if r[0] not in done]
print(f"to write: {len(todo)} of {len(rows)}; spent ${spent[0]:.2f} of ${a.budget}", flush=True)
lock, fh, stopped = threading.Lock(), open(a.raw, "a"), [False]

def one(item):
    uid, t, ans, fs = item
    with lock:
        if spent[0] >= a.budget: stopped[0] = True; return
    i = random.Random(f"{a.seed}-{uid}").choice(fs); fact = ans["findings"][i]["fact"]
    msg = f"Abstract:\n{inputs[uid]['abstract']}\n\nSentence to edit (finding {i + 1}):\n{fact}"
    res, rec = call_tool(a.model, a.region, SYSTEM, [{"role": "user", "content": [{"text": msg}]}], "record_edit", SCHEMA, max_tokens=600)
    g = {"unit_id": uid, "field": f"f{i + 1}", "original": fact, "result": res, "cost_usd": round((rec or {}).get("cost_usd") or 0, 6),
         "error": (rec or {}).get("error", "")}
    with lock:
        spent[0] += g["cost_usd"]; fh.write(json.dumps(g, ensure_ascii=False) + "\n"); fh.flush()

with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(one, todo))
fh.close()

def norm(s): return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
kept, c = [], {"none": 0, "removed not in fact": 0, "not shorter": 0, "truncated": 0, "too different": 0, "error": 0, "kept": 0}
chosen_by_uid = {r[0]: r for r in rows}
for g in map(json.loads, open(a.raw)):
    r = g.get("result") or {}
    if g.get("error") or not r.get("edited"): c["error"] += 1; continue
    if r.get("kind") == "none": c["none"] += 1; continue
    fact, ed, rem = g["original"], r["edited"].strip(), norm(r.get("removed") or "")
    if not rem or rem not in norm(fact): c["removed not in fact"] += 1; continue
    if len(ed) >= len(fact): c["not shorter"] += 1; continue
    if ed[-1] != fact[-1] or len(ed) < 0.75 * len(fact): c["truncated"] += 1; continue
    if difflib.SequenceMatcher(None, fact, ed).ratio() < 0.6: c["too different"] += 1; continue
    uid, t, ans, fs = chosen_by_uid[g["unit_id"]]
    i = int(g["field"][1]) - 1
    m = json.loads(json.dumps(ans)); m["findings"][i]["fact"] = ed
    kept.append({"system": t["system"], "user": t["user"], "chosen": json.dumps(ans, ensure_ascii=False),
                 "rejected": json.dumps(m, ensure_ascii=False), "unit_id": uid, "field": g["field"], "label": "lostcond:" + r["kind"],
                 "round": "kimi-lostcond"}); c["kept"] += 1
with open(a.out, "w") as o:
    for k in kept: o.write(json.dumps(k, ensure_ascii=False) + "\n")
print(json.dumps({**c, "spent": round(spent[0], 2), "budget_hit": stopped[0]}), flush=True)
