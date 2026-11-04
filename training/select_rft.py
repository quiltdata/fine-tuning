#!/usr/bin/env python3
"""Choose rejection-sampling training targets (the reward rounds; --facts adds the fact grader, METHODOLOGY.md §5).

Per training batch, a candidate qualifies only if it passes ingest 03's check on every row: schema
and length limits, exactly the batch's rows, every evidence quote verbatim in that row's excerpt or
mentions. Then one of two selection rules:

  default   (v2) among qualifying samples keep the one closest to the teacher's answer (token F1
            over summary, subject terms and entity values); a batch with no qualifying sample falls
            back to the teacher's answer if it passed the check (it is in train.jsonl).
  --facts   (v4-v6) candidates are the qualifying samples plus the teacher's answer if it passed. Each
            candidate's rows are graded against the batch's two required facts (facts.jsonl from
            make_facts.py, grader prompt ingest prompts/grader.md) by an open model on Bedrock's
            Chat Completions API, and the candidate with the highest top-2 recall wins. At equal
            recall an own sample beats the teacher (the model's own check-passing answers are what
            the round is for), and among own samples the one closest to the teacher's wording wins.
            Identical candidates are graded once, and verdicts are cached in --grades so a rerun
            costs nothing.

Usage: python3 select_rft.py inputs.csv samples.jsonl teacher_outputs.csv teacher_train.jsonl out_train.jsonl
                             [--facts facts.jsonl --grades grades.jsonl [--grader us.moonshotai.kimi-k3]
                              [--region us-east-2] [--workers 32] [--reasoning-effort low]]
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent.parent / "ingest"
sys.path.insert(0, str(ROOT / "tools"))
from run_model import SYSTEM, batches, parse_rows, user_message  # noqa: E402
from check_outputs import quote_found  # noqa: E402  (the clarified comparison; rounds up to v6 used norm() only)

SCHEMA = json.loads((ROOT / "schemas" / "output.schema.json").read_text())
GRADER_SYSTEM = (ROOT / "prompts" / "grader.md").read_text()
GRADE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdicts"],
                "properties": {"verdicts": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["id", "conveyed"],
                    "properties": {"id": {"type": "string"}, "conveyed": {"type": "boolean"}}}}}}


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def tokens(row):
    ents = " ".join(e.get("value", "") for e in row.get("entities") or [] if isinstance(e, dict))
    text = " ".join([row.get("summary", ""), " ".join(map(str, row.get("about") or [])), ents])
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def f1(a, b):
    common = sum((a & b).values())
    return 0.0 if not common else 2 * common / (sum(a.values()) + sum(b.values()))


def passes(rows, batch_inputs):
    wanted = {r["unit_id"]: r for r in batch_inputs}
    if sorted(r.get("unit_id") for r in rows if isinstance(r, dict)) != sorted(wanted):
        return False
    for row in rows:
        try:
            jsonschema.validate(row, SCHEMA)
        except jsonschema.ValidationError:
            return False
        inp = wanted[row["unit_id"]]
        source = norm(inp["excerpt"] + " " + " ".join(json.loads(inp["mentions"])) + " " + inp.get("parent", ""))
        if any(not quote_found(q, source) for q in row["evidence"]):
            return False
    return True


def row_key(row):
    """One row as the grader sees it (what, summary, about, entities, refers_to), canonical JSON."""
    return json.dumps({k: row.get(k) for k in ("what", "summary", "about", "entities", "refers_to")}, sort_keys=True, ensure_ascii=False)


class Grader:
    """Top-2 recall of a row against its facts, by an open model; verdicts cached by (unit_id, row)."""

    def __init__(self, a):
        from bedrock_chat import call_json
        self.call_json, self.a = call_json, a
        self.facts = {f["unit_id"]: f["key_facts"][:2] for f in map(json.loads, open(a.facts))}
        self.cache, self.spent, self.errors = {}, 0.0, 0
        self.path = Path(a.grades)
        if self.path.exists():
            for g in map(json.loads, open(self.path)):
                self.cache[(g["unit_id"], g["row"])] = g["verdicts"]
        self.fh = open(self.path, "a")

    def grade(self, uid, row):
        key = (uid, row_key(row))
        if key in self.cache:
            return self.cache[key]
        facts = self.facts.get(uid) or []
        if not facts:
            return {}
        msg = ("Reference facts:\n" + "\n".join(f'{f["id"]}. {f["fact"]}' for f in facts) +
               "\n\nOutput row:\n" + key[1] + "\n\nReturn a verdict for each fact id: " + ", ".join(f["id"] for f in facts))
        wanted = [f["id"] for f in facts]
        verdicts = None
        for _ in (1, 2):
            res, rec = self.call_json(self.a.grader, self.a.region, GRADER_SYSTEM, msg, "record_verdicts", GRADE_SCHEMA,
                                      max_tokens=2000, service_tier="flex", reasoning_effort=self.a.reasoning_effort)
            self.spent += rec.get("cost_usd") or 0
            got = {str(v.get("id")).rstrip("."): bool(v.get("conveyed")) for v in (res or {}).get("verdicts", [])}
            if set(wanted) <= set(got):
                verdicts = {k: got[k] for k in wanted}
                break
        if verdicts is None:
            self.errors += 1
            return None
        self.cache[key] = verdicts
        self.fh.write(json.dumps({"unit_id": uid, "row": key[1], "verdicts": verdicts}, ensure_ascii=False) + "\n")
        self.fh.flush()
        return verdicts

    def recall(self, rows):
        """Mean over the candidate's rows of the share of facts conveyed; None if any grade failed."""
        shares = []
        for r in rows:
            v = self.grade(r["unit_id"], r)
            if v is None:
                return None
            if v:
                shares.append(sum(v.values()) / len(v))
        return sum(shares) / len(shares) if shares else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("samples")
    ap.add_argument("teacher_outputs")
    ap.add_argument("teacher_train")
    ap.add_argument("out")
    ap.add_argument("--facts", help="two-fact key for the training rows: select by graded top-2 recall")
    ap.add_argument("--grades", help="verdict cache (appended)")
    ap.add_argument("--grader", default="us.moonshotai.kimi-k3")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--reasoning-effort", default="low")
    a = ap.parse_args()
    if bool(a.facts) != bool(a.grades):
        sys.exit("--facts and --grades go together")

    csv.field_size_limit(sys.maxsize)
    inputs = list(csv.DictReader(open(a.inputs, newline="")))
    by_key = {key: rows for key, rows in batches(inputs)}
    teacher = {}
    for o in csv.DictReader(open(a.teacher_outputs, newline="")):
        teacher[o["unit_id"]] = {k: json.loads(v) if k in ("about", "entities", "refers_to", "evidence") else v for k, v in o.items()}
    teacher_ok = {json.loads(l)["user"] for l in open(a.teacher_train)}  # teacher batches that passed the code filter
    grader = Grader(a) if a.facts else None
    stats = Counter()

    def candidates(rec):
        """(key, rows_in, user, [(rows, is_teacher)]) for one sampled batch."""
        key = (rec["prefix"], rec["file_type"])
        rows_in = by_key[key]
        user = user_message(rows_in)
        cands, seen = [], set()
        for text in rec["samples"]:
            try:
                rows = parse_rows(text)
            except ValueError:
                stats["sample unparseable"] += 1
                continue
            if not passes(rows, rows_in):
                stats["sample failed check"] += 1
                continue
            stats["sample passed check"] += 1
            sig = json.dumps(rows, sort_keys=True, ensure_ascii=False)
            if sig in seen:
                stats["sample duplicate"] += 1
                continue
            seen.add(sig)
            cands.append((rows, False))
        if grader and user in teacher_ok:
            cands.append(([teacher[r["unit_id"]] for r in rows_in], True))
        return key, rows_in, user, cands

    def choose(item):
        key, rows_in, user, cands = item
        order = [r["unit_id"] for r in rows_in]
        closeness = lambda rows: sum(f1(tokens(r), tokens(teacher.get(r["unit_id"], {}))) for r in rows) / len(rows)
        best = None
        if grader:
            scored = []
            for rows, is_teacher in cands:
                rec = grader.recall(rows)
                if rec is None:
                    stats["candidate grade failed"] += 1
                    continue
                scored.append((rec, closeness(rows), is_teacher, rows))
            if scored:
                rec, _, is_teacher, rows = max(scored, key=lambda t: (t[0], not t[2], t[1]))
                best = rows
                stats["batch: teacher won" if is_teacher else "batch: own sample"] += 1
                stats["chosen recall sum"] += rec
                if rec == 1.0:
                    stats["chosen recall 100%"] += 1
        else:
            own = [(closeness(rows), rows) for rows, _ in cands]
            if own:
                best = max(own, key=lambda t: t[0])[1]
                stats["batch: own sample"] += 1
            elif user in teacher_ok:
                best = [teacher[r["unit_id"]] for r in rows_in]
                stats["batch: teacher fallback"] += 1
        if best is None:
            stats["batch: dropped"] += 1
            return None
        best = sorted(best, key=lambda r: order.index(r["unit_id"]))
        return json.dumps({"system": SYSTEM, "user": user, "assistant": json.dumps({"rows": best}, ensure_ascii=False)}, ensure_ascii=False)

    items = [candidates(rec) for rec in map(json.loads, open(a.samples))]
    stats["candidates"] = sum(len(c[3]) for c in items)
    with ThreadPoolExecutor(a.workers if grader else 1) as pool, open(a.out, "w") as fh:
        for line in pool.map(choose, items):
            if line:
                fh.write(line + "\n")
    if grader:
        stats["grader_cost_usd"] = round(grader.spent, 2)
        stats["grader_errors"] = grader.errors
        stats["grader_cached_rows"] = len(grader.cache)
        chosen = stats["batch: teacher won"] + stats["batch: own sample"]
        stats["chosen recall mean"] = round(stats.pop("chosen recall sum") / chosen, 3) if chosen else None
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
