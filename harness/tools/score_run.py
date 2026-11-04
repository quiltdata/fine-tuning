#!/usr/bin/env python3
"""Score one summarizer run against a set's answer key (make_facts.py).

A row's score is the fraction of its required facts the output conveys (partial credit), and
a row is strict when it conveys all of them. Code facts are checked by code. Key facts (ranked,
most important first; the top two are also reported on their own) are judged by a grader model
(prompts/grader.md), which sees the facts and the output row (what, summary, about, entities,
refers_to; not the evidence quotes). A missing or unparseable output row scores 0.
The headline metric is top-2 key-fact recall, with score_top2 the same fraction over the code
facts and the top two key facts only; score (all 3-5 key facts) is kept for continuity.

Also reported, by code: the check_outputs.py rules (schema, lengths, quotes), and how many
entities and refers_to values appear in the row's own excerpt or mentions, as written or
with their words in any order (grounding).
Spend and latency come from the run's calls.csv; a call's cost is shared equally by its rows.

Writes to run_dir (suffix added when --tag is given, for a second grader):
  scores.csv   one line per scored row
  grades.jsonl the grader's verdict per key fact, with its call record
  score.json   totals for all, train and test, by file type, plus grader cost

Usage: python3 score_run.py set_dir run_dir [--grader ID] [--region R] [--workers N] [--tag NAME] [--limit N]
                             [--reuse-grades] [--split test]
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from bedrock_tool import RATES, call_tool
from check_outputs import quote_found

ROOT = Path(__file__).resolve().parent.parent
SYSTEM = (ROOT / "prompts" / "grader.md").read_text()
OUT_SCHEMA = json.loads((ROOT / "schemas" / "output.schema.json").read_text())
LIST_COLUMNS = ["about", "entities", "refers_to", "evidence"]
GRADE_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "conveyed": {"type": "boolean"}},
        "required": ["id", "conveyed"]}}},
    "required": ["verdicts"],
}
SCORE_COLUMNS = ["unit_id", "split", "file_type", "facts", "facts_conveyed", "score", "score_top2", "strict", "code_facts", "code_conveyed",
                 "key_facts", "key_conveyed", "top2_facts", "top2_conveyed", "check_pass", "check_problems", "entities", "entities_grounded",
                 "refers_to", "refers_to_grounded", "cost_usd", "latency_s", "missed", "error"]


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def loose(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_output(row):
    try:
        for col in LIST_COLUMNS:
            row[col] = json.loads(row[col])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return None, f"unparseable: {e}"
    return row, ""


def check(out, source):
    """The check_outputs.py rules for one row."""
    problems = []
    try:
        jsonschema.validate(out, OUT_SCHEMA)
    except jsonschema.ValidationError as e:
        problems.append(f"schema: {e.message}"[:120])
    for q in out.get("evidence") or []:
        if isinstance(q, str) and not quote_found(q, source):
            problems.append(f"quote not found: {q[:60]!r}")
    return problems


def code_fact(f, out):
    values = [e.get("value", "") for e in out.get("entities") or [] if isinstance(e, dict)]
    text = loose(" ".join([out.get("summary", ""), *map(str, out.get("about") or []), *values]))
    if f["check"] == "what":
        return out.get("what") in f["expect"]
    if f["check"] == "entity_contains":
        return any(loose(f["expect"]) in loose(v) for v in values)
    if f["check"] == "license":
        return any(isinstance(e, dict) and e.get("type") == "license" for e in out.get("entities") or []) or \
            any(k in v.lower() for v in values for k in ("creativecommons", "cc by", "cc-by", "cc0"))
    if f["check"] == "text_contains_any":
        return any(loose(j) and loose(j) in text for j in f["expect"])
    raise ValueError(f["check"])


def grounded(values, source):
    """Values found in the source as written, or with all their words present in any order
    ("Diethart Matthies" for "Matthies Diethart"). The first tolerates PDF text that glues
    affiliation marks onto names ("Hoerder1")."""
    src = loose(source)
    words = set(src.split())
    return sum(1 for v in values if loose(v) and (loose(v) in src or set(loose(v).split()) <= words))


def grade(facts, out, args):
    if not facts["key_facts"]:
        return {}, None
    if facts["unit_id"] in args.previous:  # --reuse-grades: same verdicts, no grader call
        return args.previous[facts["unit_id"]]
    shown = {k: out.get(k) for k in ("what", "summary", "about", "entities", "refers_to")}
    msg = ("Reference facts:\n" + "\n".join(f'{f["id"]}. {f["fact"]}' for f in facts["key_facts"]) +
           "\n\nOutput row:\n" + json.dumps(shown, ensure_ascii=False) +
           "\n\nReturn a verdict for each fact id: " + ", ".join(f["id"] for f in facts["key_facts"]))
    wanted = [f["id"] for f in facts["key_facts"]]
    spent = 0.0
    for attempt in (1, 2):
        res, rec = call_tool(args.grader, args.region, SYSTEM, [{"role": "user", "content": [{"text": msg}]}],
                             "record_verdicts", GRADE_SCHEMA, max_tokens=2000)
        spent += rec.get("cost_usd") or 0
        rec.update(attempts=attempt, cost_usd=round(spent, 6))  # the record carries every attempt's cost
        verdicts = {str(v.get("id")).rstrip("."): bool(v.get("conveyed")) for v in (res or {}).get("verdicts", [])}
        if set(wanted) <= set(verdicts):
            return {k: verdicts[k] for k in wanted}, rec
        rec["error"] = rec["error"] or f"verdicts for {sorted(verdicts)}, wanted {wanted}"
    return None, rec


def score_row(facts, inp, out, cost_share, latency, args):
    source = norm(inp["excerpt"] + " " + " ".join(json.loads(inp["mentions"])) + " " + inp.get("parent", ""))
    row = {"unit_id": facts["unit_id"], "split": facts["split"], "file_type": facts["file_type"],
           "code_facts": len(facts["code_facts"]), "key_facts": len(facts["key_facts"]),
           "cost_usd": cost_share, "latency_s": latency, "error": ""}
    row["facts"] = row["code_facts"] + row["key_facts"]
    out, err = parse_output(dict(out)) if out else (None, "no output row")
    if out is None:
        return {**row, "facts_conveyed": 0, "score": 0.0, "score_top2": 0.0, "strict": 0, "code_conveyed": 0, "key_conveyed": 0,
                "top2_facts": min(2, row["key_facts"]), "top2_conveyed": 0,
                "check_pass": 0, "check_problems": "", "entities": 0, "entities_grounded": 0, "refers_to": 0,
                "refers_to_grounded": 0, "missed": "", "error": err}, None
    code = {f["id"]: code_fact(f, out) for f in facts["code_facts"]}
    key, rec = grade(facts, out, args)
    if key is None:
        row["error"] = "grader failed"
        key = {f["id"]: False for f in facts["key_facts"]}
    conveyed = sum(code.values()) + sum(key.values())
    problems = check(out, source)
    ents = [e.get("value", "") for e in out.get("entities") or [] if isinstance(e, dict)]
    refs = [r for r in out.get("refers_to") or [] if isinstance(r, str)]
    top2 = sum(list(key.values())[:2])
    n_top2 = row["code_facts"] + min(2, len(key))
    row.update(facts_conveyed=conveyed, score=round(conveyed / row["facts"], 4) if row["facts"] else 1.0,
               score_top2=round((sum(code.values()) + top2) / n_top2, 4) if n_top2 else 1.0,
               strict=int(conveyed == row["facts"]), code_conveyed=sum(code.values()), key_conveyed=sum(key.values()),
               top2_facts=min(2, len(key)), top2_conveyed=top2,
               check_pass=int(not problems), check_problems="; ".join(problems),
               entities=len(ents), entities_grounded=grounded(ents, source),
               refers_to=len(refs), refers_to_grounded=grounded(refs, source),
               missed=" ".join(k for k, v in {**code, **key}.items() if not v))
    return row, {"unit_id": facts["unit_id"], "verdicts": key, "call": rec}


def totals(rows):
    if not rows:
        return None
    s = lambda f: sum(r[f] for r in rows)
    spend = s("cost_usd")
    score = 100 * statistics.mean(r["score"] for r in rows)
    arts = [r for r in rows if r["code_facts"]]
    return {
        "rows": len(rows), "score": round(score, 1),
        "score_top2": round(100 * statistics.mean(float(r.get("score_top2") or 0) for r in rows), 1), "strict": s("strict"),
        "key_fact_recall": round(100 * s("key_conveyed") / s("key_facts"), 1) if s("key_facts") else None,
        "top2_key_fact_recall": round(100 * s("top2_conveyed") / s("top2_facts"), 1) if s("top2_facts") else None,
        "code_fact_recall": round(100 * sum(r["code_conveyed"] for r in arts) / sum(r["code_facts"] for r in arts), 1) if arts else None,
        "check_pass": s("check_pass"),
        "entities_grounded_pct": round(100 * s("entities_grounded") / s("entities"), 1) if s("entities") else None,
        "refers_to_grounded_pct": round(100 * s("refers_to_grounded") / s("refers_to"), 1) if s("refers_to") else None,
        "spend_usd": round(spend, 4), "usd_per_100_rows": round(100 * spend / len(rows), 3),
        "points_per_usd": round(score / spend, 1) if spend else None,
        "errors": sum(1 for r in rows if r["error"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_dir")
    ap.add_argument("run_dir")
    ap.add_argument("--grader", default="us.openai.gpt-6.1-sol", choices=sorted(RATES))
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. sonnet for a second grader")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--split", choices=["train", "test"], help="score only that split's rows (for runs made with run_model.py --only)")
    ap.add_argument("--reuse-grades", action="store_true", help="take verdicts from an earlier grades file instead of calling the grader")
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    sd, rd = Path(args.set_dir), Path(args.run_dir)
    inputs = {r["unit_id"]: r for r in csv.DictReader(open(sd / "inputs.csv", newline=""))}
    facts = [f for f in map(json.loads, open(sd / "reference" / "facts.jsonl")) if not args.split or f["split"] == args.split][: args.limit]
    outputs = {r["unit_id"]: r for r in csv.DictReader(open(rd / "outputs.csv", newline=""))}
    batch_rows = defaultdict(list)
    for uid, r in inputs.items():
        batch_rows[(r["prefix"], r["file_type"])].append(uid)
    share, latency = {}, {}
    for c in csv.DictReader(open(rd / "calls.csv", newline="")):
        uids = batch_rows[(c["prefix"], c["file_type"])]
        for uid in uids:
            share[uid] = float(c["cost_usd"] or 0) / len(uids)
            latency[uid] = float(c["latency_s"] or 0)

    sfx = f".{args.tag}" if args.tag else ""
    args.previous = {}
    if args.reuse_grades:
        for g in map(json.loads, open(rd / f"grades{sfx}.jsonl")):
            if not (g.get("call") or {}).get("error"):  # rows with no key facts have no call
                args.previous[g["unit_id"]] = (g["verdicts"], g["call"])
    started, wall = datetime.now(timezone.utc), time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda f: score_row(f, inputs[f["unit_id"]], outputs.get(f["unit_id"]),
                                                    round(share.get(f["unit_id"], 0), 6), latency.get(f["unit_id"]), args), facts))
    wall = time.perf_counter() - wall

    rows = [r for r, _ in results]
    with open(rd / f"scores{sfx}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCORE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(rd / f"grades{sfx}.jsonl", "w") as fh:
        for _, g in results:
            if g:
                fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    calls = [g["call"] for _, g in results if g and g["call"]]
    run = json.loads((rd / "run.json").read_text()) if (rd / "run.json").exists() else {}
    summary = {
        "scored": started.isoformat(timespec="seconds"), "run_dir": rd.name, "model": run.get("model"),
        "grader": args.grader, "grader_rate": RATES[args.grader], "answer_key": "reference/facts.jsonl",
        "all": totals(rows), "train": totals([r for r in rows if r["split"] == "train"]),
        "test": totals([r for r in rows if r["split"] == "test"]),
        "by_file_type": {t: totals([r for r in rows if r["file_type"] == t]) for t in sorted({r["file_type"] for r in rows})},
        "median_call_latency_s": round(statistics.median(v for v in latency.values()), 2) if latency else None,
        "grader_calls": len(calls), "grader_errors": sum(1 for c in calls if c.get("error")),
        "grader_cost_usd": round(sum(c.get("cost_usd") or 0 for c in calls), 4), "wall_clock_s": round(wall, 1),
    }
    (rd / f"score{sfx}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
