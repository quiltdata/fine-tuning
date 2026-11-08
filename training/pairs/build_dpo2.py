#!/usr/bin/env python3
"""Minimal-pair round, step 4: the training file. Minimal pairs from (a) the selector's pairs file and (b) the rest-graded RFT v4
samples against the abstract's chosen answer, each flagged field kept only if the replication judge flagged it
too; plus (c) Kimi's lost-condition pairs, capped at --lostcond-share of the total. Dedupe, shuffle, write
train.py pairs with provenance; print counts.
usage: build_dpo2.py set_dir samples_dir out.jsonl --replication rep.jsonl --lostcond pairs.jsonl [--lostcond-share 0.33] [--seed 42]"""
import argparse, collections, json, random, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "harness" / "tools"))
import run_model as rm  # noqa: E402
import csv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("set_dir"); ap.add_argument("out")
ap.add_argument("--round", action="append", required=True, help="samples_dir:chosen_train.jsonl:grades.jsonl[,grades2.jsonl]")
ap.add_argument("--replication", required=True); ap.add_argument("--lostcond", required=True)
ap.add_argument("--lostcond-share", type=float, default=0.33); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
rm.set_task("abstract-facts"); csv.field_size_limit(sys.maxsize)
inputs = {r["unit_id"]: r for r in csv.DictReader(open(Path(a.set_dir) / "inputs.csv", newline=""))}
user_of = {u: rm.TASK.user_message([r]) for u, r in inputs.items()}
by_text = {v: k for k, v in user_of.items()}
# per round: chosen answers (the round's own/teacher targets) and Sonnet verdicts
V, chosen, round_of = {}, {}, {}
for spec in a.round:
    sdir, cfile, gfiles = spec.split(":")
    S = Path(sdir); name = S.name
    for t in map(json.loads, open(S / cfile)):
        u = by_text.get(t["user"])
        if u: chosen[(name, u)] = (t["system"], t["user"], json.loads(t["assistant"]))
    for gf in gfiles.split(","):
        for g in map(json.loads, open(S / gf)):
            if g["kind"] == "faith" and g["verdicts"]:
                V[(g["unit_id"], g["answer"])] = g["verdicts"]; round_of[(g["unit_id"], g["answer"])] = name
# replication: same key -> verdicts over the flagged fields
R = {}
for g in map(json.loads, open(a.replication)):
    if g.get("verdicts"): R[(g["unit_id"], g["answer"])] = g["verdicts"]
c = collections.Counter(); seen = set(); rows = []

def splice(ch, rj, k):
    m = json.loads(json.dumps(ch))
    if k == "about":
        if "about" not in rj: return None
        m["about"] = rj["about"]
    else:
        i = int(k[1]) - 1; rf = rj.get("findings") or []
        if i >= len(rf): return None
        mf = list(m.get("findings") or [])
        while len(mf) <= i: mf.append(rf[len(mf)])
        mf[i] = rf[i]; m["findings"] = mf
    return m

for (uid, akey), v in V.items():
    name = round_of[(uid, akey)]
    bad = [k for k, x in v.items() if x != "supported"]
    if not bad: continue
    if (name, uid) not in chosen: c["no chosen answer"] += len(bad); continue
    rep = R.get((uid, akey))
    if rep is None: c["flag not replicated (no record)"] += len(bad); continue
    sysm, user, ch = chosen[(name, uid)]; rj = json.loads(akey)
    for k in bad:
        if rep.get(k, "supported") == "supported": c["flag not replicated"] += 1; continue
        m = splice(ch, rj, k)
        if m is None: c["missing field"] += 1; continue
        cs, ms = json.dumps(ch, ensure_ascii=False), json.dumps(m, ensure_ascii=False)
        if cs == ms: c["identical"] += 1; continue
        if (uid, cs, ms) in seen: c["duplicate"] += 1; continue
        seen.add((uid, cs, ms)); c[f"sample:{k}:{v[k]}"] += 1; c[f"round:{name}"] += 1
        rows.append({"system": sysm, "user": user, "chosen": cs, "rejected": ms, "unit_id": uid, "field": k, "label": v[k], "round": name})
n_sample = len(rows)
L = [json.loads(l) for l in open(a.lostcond)]
random.Random(a.seed).shuffle(L)
cap = int(a.lostcond_share / (1 - a.lostcond_share) * n_sample)
L = L[:cap]
for r in L: c[f"lostcond:{r['label']}"] += 1
rows += L
random.Random(a.seed).shuffle(rows)
with open(a.out, "w") as o:
    for r in rows: o.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({"pairs": len(rows), "from samples": n_sample, "lostcond": len(L), "abstracts": len({r["unit_id"] for r in rows}),
                  **dict(sorted(c.items()))}, indent=1))
