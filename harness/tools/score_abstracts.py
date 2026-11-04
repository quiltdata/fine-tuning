#!/usr/bin/env python3
"""Score one abstract-task run (run_model.py / run_claude.py --task abstract-facts) against its key.

Per abstract, two grader calls, each seeing only what its question needs:

  recall        the key's facts (make_facts.py --task abstract-facts) and the model's facts, not the
                abstract (prompts/abstract_grader_recall.md): is each key fact conveyed?
  faithfulness  the abstract and the model's facts, not the key (prompts/abstract_grader_faithfulness.md):
                is each of the model's facts supported, overstated or unsupported, and are its two
                findings redundant?

And by code: validity (check_outputs.abstract_problems) and the quote side metric. A missing,
unparseable or invalid row scores as conveying nothing; an invalid row's facts are still graded
for faithfulness when they parse.

Measures, for all rows and split by the protocol flag:

  about_recall       key about facts conveyed / rows whose key has one
  findings_recall    key findings conveyed / key findings (rows whose key has none don't count)
  faithfulness       the model's facts graded supported / facts stated; also overstated, unsupported
  redundancy         rows with redundant findings / rows with two findings
  invented_findings  rows whose key has no findings but whose output states one
  validity           rows passing the code check / rows
  quotes_found       side metric, not scored
  usd_per_100_rows   the run's spend, each call's cost shared by its rows (calls.csv unit_ids)

Writes to run_dir (suffix added when --tag is given, for a second grader):
  scores.csv    one line per abstract
  grades.jsonl  both graders' verdicts per abstract, with their call records
  score.json    the measures and grader cost

Usage: python3 score_abstracts.py set_dir run_dir [--grader ID] [--region R] [--workers N] [--tag NAME]
                                  [--limit N] [--reuse-grades]
"""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from bedrock_tool import RATES, call_tool
from check_outputs import abstract_problems, abstract_quotes, parse_abstract_row

ROOT = Path(__file__).resolve().parent.parent
RECALL_SYSTEM = (ROOT / "prompts" / "abstract_grader_recall.md").read_text()
FAITH_SYSTEM = (ROOT / "prompts" / "abstract_grader_faithfulness.md").read_text()
VERDICTS = ["supported", "overstated", "unsupported"]
RECALL_SCHEMA = {"type": "object", "required": ["verdicts"], "properties": {"verdicts": {"type": "array", "items": {
    "type": "object", "required": ["id", "conveyed"], "properties": {"id": {"type": "string"}, "conveyed": {"type": "boolean"}}}}}}
FAITH_SCHEMA = {"type": "object", "required": ["verdicts", "redundant"], "properties": {
    "verdicts": {"type": "array", "items": {"type": "object", "required": ["id", "verdict"],
                                            "properties": {"id": {"type": "string"}, "verdict": {"enum": VERDICTS}}}},
    "redundant": {"type": "boolean"}}}
COLUMNS = ["unit_id", "protocol", "valid", "problems", "key_about", "about_conveyed", "key_findings", "findings_conveyed",
           "stated", "out_findings", "supported", "overstated", "unsupported", "two_findings", "redundant",
           "quotes_given", "quotes_found", "cost_usd", "missed", "error"]


def model_facts(out):
    """[(id, fact text)] for the model's about fact and findings, in order."""
    facts = []
    if isinstance(out.get("about"), dict) and out["about"].get("fact"):
        facts.append(("about", out["about"]["fact"]))
    for i, f in enumerate(out.get("findings") or []):
        if isinstance(f, dict) and f.get("fact"):
            facts.append((f"f{i + 1}", f["fact"]))
    return facts


def graded(system, msg, schema, name, wanted, key, args):
    """Calls the grader (at most twice) until it returns a verdict for every wanted id."""
    spent, rec = 0.0, None
    for attempt in (1, 2):
        res, rec = call_tool(args.grader, args.region, system, [{"role": "user", "content": [{"text": msg}]}], name, schema,
                             max_tokens=2000)
        spent += rec.get("cost_usd") or 0
        rec.update(attempts=attempt, cost_usd=round(spent, 6))
        got = {str(v.get("id")).rstrip("."): v.get(key) for v in (res or {}).get("verdicts", [])}
        if set(wanted) <= set(got):
            return {k: got[k] for k in wanted}, (res or {}), rec
        rec["error"] = rec.get("error") or f"verdicts for {sorted(got)}, wanted {wanted}"
    return None, {}, rec


def grade_recall(key_facts, facts, args):
    if not key_facts:
        return {}, None
    if not facts:
        return {k["id"]: False for k in key_facts}, None
    msg = ("Reference facts:\n" + "\n".join(f'{k["id"]} ({k["role"]}). {k["fact"]}' for k in key_facts) +
           "\n\nThe model's facts:\n" + "\n".join(f"- {t}" for _, t in facts) +
           "\n\nReturn a verdict for each reference fact id: " + ", ".join(k["id"] for k in key_facts))
    verdicts, _, rec = graded(RECALL_SYSTEM, msg, RECALL_SCHEMA, "record_verdicts", [k["id"] for k in key_facts], "conveyed", args)
    return verdicts, rec


def grade_faithfulness(abstract, facts, args):
    if not facts:
        return {}, False, None
    msg = ("Abstract:\n" + abstract + "\n\nThe model's facts:\n" + "\n".join(f"{i}. {t}" for i, t in facts) +
           "\n\nReturn a verdict for each fact id: " + ", ".join(i for i, _ in facts) + "; and redundant.")
    verdicts, res, rec = graded(FAITH_SYSTEM, msg, FAITH_SCHEMA, "record_verdicts", [i for i, _ in facts], "verdict", args)
    return verdicts, bool(res.get("redundant")), rec


def score_row(key, inp, raw, cost_share, args):
    row = {"unit_id": key["unit_id"], "protocol": int(inp.get("protocol") or 0), "cost_usd": cost_share, "error": "",
           "key_about": int(any(k["role"] == "about" for k in key["key_facts"])),
           "key_findings": sum(k["role"] == "finding" for k in key["key_facts"])}
    out, problems = None, []
    if raw is None:
        problems = ["no output row"]
    else:
        try:
            out = parse_abstract_row(raw)
            problems = abstract_problems(out)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            problems = [f"unparseable: {e}"[:160]]
    facts = model_facts(out) if out else []
    valid = not problems
    previous = args.previous.get(key["unit_id"])
    if previous:
        recall, faith, redundant = previous["recall"], previous["faithfulness"], previous["redundant"]
        recs = previous["calls"]
    else:
        recall, r1 = grade_recall(key["key_facts"], facts if valid else [], args)
        faith, redundant, r2 = grade_faithfulness(inp["abstract"], facts, args)
        recs = [r for r in (r1, r2) if r]
        if recall is None or faith is None:
            row["error"] = "grader failed"
            recall = recall if recall is not None else {k["id"]: False for k in key["key_facts"]}
            faith = faith if faith is not None else {}
    given, missing = abstract_quotes(out, inp["abstract"]) if out else ([], [])
    out_findings = sum(1 for i, _ in facts if i != "about")
    row.update(
        valid=int(valid), problems="; ".join(problems),
        about_conveyed=int(bool(recall.get("about"))),
        findings_conveyed=sum(bool(v) for k, v in recall.items() if k != "about"),
        stated=len(facts), out_findings=out_findings,
        **{v: sum(1 for x in faith.values() if x == v) for v in VERDICTS},
        two_findings=int(out_findings == 2), redundant=int(out_findings == 2 and redundant),
        quotes_given=len(given), quotes_found=len(given) - len(missing),
        missed=" ".join(k for k, v in recall.items() if not v))
    grades = {"unit_id": key["unit_id"], "recall": recall, "faithfulness": faith, "redundant": redundant, "calls": recs}
    return row, grades


def ratio(n, d, scale=100):
    return round(scale * n / d, 1) if d else None


def totals(rows):
    if not rows:
        return None
    s = lambda f: sum(r[f] for r in rows)
    spend = s("cost_usd")
    return {
        "rows": len(rows), "valid": s("valid"), "validity": ratio(s("valid"), len(rows)),
        "about_recall": ratio(s("about_conveyed"), s("key_about")),
        "findings_recall": ratio(s("findings_conveyed"), s("key_findings")),
        "facts_stated": s("stated"), "faithfulness": ratio(s("supported"), s("stated")),
        "overstated": ratio(s("overstated"), s("stated")), "unsupported": ratio(s("unsupported"), s("stated")),
        "rows_with_two_findings": s("two_findings"), "redundancy": ratio(s("redundant"), s("two_findings")),
        "invented_findings": sum(1 for r in rows if r["key_findings"] == 0 and r["out_findings"] > 0),
        "rows_key_without_findings": sum(1 for r in rows if r["key_findings"] == 0),
        "quotes_found": ratio(s("quotes_found"), s("quotes_given")),
        "spend_usd": round(spend, 4), "usd_per_100_rows": round(100 * spend / len(rows), 4),
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
    ap.add_argument("--reuse-grades", action="store_true", help="take verdicts from an earlier grades file instead of calling the grader")
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    sd, rd = Path(args.set_dir), Path(args.run_dir)
    inputs = {r["unit_id"]: r for r in csv.DictReader(open(sd / "inputs.csv", newline=""))}
    keys = [json.loads(l) for l in open(sd / "reference" / "facts.jsonl")][: args.limit]
    outputs = {r["unit_id"]: r for r in csv.DictReader(open(rd / "outputs.csv", newline=""))}
    share = {}
    for c in csv.DictReader(open(rd / "calls.csv", newline="")):
        ids = json.loads(c["unit_ids"]) if c.get("unit_ids") else []
        for u in ids:
            share[u] = float(c["cost_usd"] or 0) / len(ids)
    sfx = f".{args.tag}" if args.tag else ""
    args.previous = {g["unit_id"]: g for g in map(json.loads, open(rd / f"grades{sfx}.jsonl"))} if args.reuse_grades else {}

    started, wall = datetime.now(timezone.utc), time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda k: score_row(k, inputs[k["unit_id"]], outputs.get(k["unit_id"]),
                                                    round(share.get(k["unit_id"], 0), 6), args), keys))
    wall = time.perf_counter() - wall

    rows = [r for r, _ in results]
    with open(rd / f"scores{sfx}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(rd / f"grades{sfx}.jsonl", "w") as fh:
        for _, g in results:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    calls = [c for _, g in results for c in g["calls"] if not args.previous]
    run = json.loads((rd / "run.json").read_text()) if (rd / "run.json").exists() else {}
    summary = {
        "scored": started.isoformat(timespec="seconds"), "run_dir": rd.name, "model": run.get("model"),
        "task": run.get("task"), "prompt_sha256": run.get("prompt_sha256"), "examples_sha256": run.get("examples_sha256"),
        "grader": args.grader, "grader_rate": RATES[args.grader], "answer_key": "reference/facts.jsonl",
        "all": totals(rows), "not_protocol": totals([r for r in rows if not r["protocol"]]),
        "protocol": totals([r for r in rows if r["protocol"]]),
        "grader_calls": len(calls), "grader_errors": sum(1 for c in calls if c.get("error")),
        "grader_cost_usd": round(sum(c.get("cost_usd") or 0 for c in calls), 4), "wall_clock_s": round(wall, 1),
    }
    (rd / f"score{sfx}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
