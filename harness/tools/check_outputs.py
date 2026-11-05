#!/usr/bin/env python3
"""Check a summarizer output table against its input table.

For every output row: it parses and validates against schemas/output.schema.json,
its unit_id is one of the inputs, and every evidence quote appears in that input
row's excerpt or mentions. Every input row must have exactly one output row.
Prints one line per row and exits 1 if any row fails.

The quote comparison (quote_found) asks whether the quote's letters and digits appear, in
order and contiguous, among the source's letters and digits: NFKC form, case folded, and
whitespace, punctuation and hyphenation ignored. A quote with no letters or digits fails.
History (METHODOLOGY.md, corrections): at first only whitespace runs were collapsed; then
NFKC and typographic quotes and dashes were unified, keeping case and punctuation; finally case
and punctuation were dropped too, because PDF extraction breaks words at line ends
(meth-\nods), splits running headers into sentences and drops or alters punctuation, and a
quote whose letters and digits match in order points at exactly the right text.
quote_norm (the middle form) is kept for comparison.

--task abstract-facts checks validity only, by code: the row parses and fits
schemas/abstract_output.schema.json (one about fact, at most two findings), its unit_id is an
input, and every input has one row. Lengths are reported, not judged. Quotes there are optional
and not scored: one not found in the abstract is printed as a note and counted, but doesn't fail
the row.

Usage: python3 check_outputs.py inputs.csv outputs.csv [--task file-summary|abstract-facts]
"""
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import jsonschema

SCHEMA = json.loads((Path(__file__).resolve().parent.parent / "schemas" / "output.schema.json").read_text())
LIST_COLUMNS = ["about", "entities", "refers_to", "evidence"]
_TYPOGRAPHIC = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
                              "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
                              "​": "", "‌": "", "‍": "", "﻿": "", "­": ""})


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def quote_norm(s):
    """Comparison form for an evidence quote or its source: whitespace ignored, typographic variants unified."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s).translate(_TYPOGRAPHIC))


def quote_skeleton(s):
    """Letters and digits only, NFKC-normalized and case-folded."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", s).casefold() if ch.isalnum())


def quote_found(q, source):
    k = quote_skeleton(q)
    return bool(k) and k in quote_skeleton(source)


ABSTRACT_SCHEMA = json.loads((Path(__file__).resolve().parent.parent / "schemas" / "abstract_output.schema.json").read_text())
ABSTRACT_LIST_COLUMNS = ["about", "findings"]


def parse_abstract_row(row):
    """An abstract-task output row from CSV, with its JSON cells decoded; raises on bad JSON."""
    out = dict(row)
    for col in ABSTRACT_LIST_COLUMNS:
        out[col] = json.loads(out[col])
    return out


def abstract_problems(out):
    """Validity of one parsed abstract-task row, by code: the schema (one about fact, at most two
    findings). Lengths are not judged: cost per abstract already counts verbosity."""
    try:
        jsonschema.validate(out, ABSTRACT_SCHEMA)
        return []
    except jsonschema.ValidationError as e:
        return [f"schema: {e.message}"[:160]]


def abstract_quotes(out, abstract):
    """(quotes given, quotes not found in the abstract). A side metric: quotes are not scored."""
    facts = [out.get("about")] + list(out.get("findings") or [])
    given = [f["quote"] for f in facts if isinstance(f, dict) and isinstance(f.get("quote"), str)]
    return given, [q for q in given if not quote_found(q, abstract)]


def main_abstract(inputs_path, outputs_path):
    csv.field_size_limit(sys.maxsize)
    inputs = {r["unit_id"]: r for r in csv.DictReader(open(inputs_path, newline=""))}
    seen, failed, quotes, misses, longest = set(), 0, 0, 0, 0
    for row in csv.DictReader(open(outputs_path, newline="")):
        uid, problems, notes = row.get("unit_id", ""), [], []
        try:
            out = parse_abstract_row(row)
            problems += abstract_problems(out)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            out, problems = None, [f"unparseable: {e}"[:160]]
        if uid not in inputs:
            problems.append("unit_id not in inputs")
        elif uid in seen:
            problems.append("duplicate unit_id")
        elif out is not None:
            given, missing = abstract_quotes(out, inputs[uid]["abstract"])
            longest = max([longest] + [len(f["fact"]) for f in [out.get("about")] + list(out.get("findings") or [])
                                       if isinstance(f, dict) and isinstance(f.get("fact"), str)])
            quotes, misses = quotes + len(given), misses + len(missing)
            notes = [f"note: quote not found (not scored): {q[:60]!r}" for q in missing]
        seen.add(uid)
        failed += bool(problems)
        print(("FAIL  " if problems else "PASS  ") + uid + "".join(f"\n      {p}" for p in problems + notes))
    for uid in inputs.keys() - seen:
        failed += 1
        print(f"FAIL  {uid}\n      no output row")
    print(f"\n{len(inputs) - failed if failed <= len(inputs) else 0}/{len(inputs)} input rows pass"
          f"\n{quotes - misses}/{quotes} quotes found in their abstract (side metric, not scored)"
          f"\nlongest fact: {longest} characters (reported, not judged)")
    sys.exit(1 if failed else 0)


def main(inputs_path, outputs_path):
    csv.field_size_limit(sys.maxsize)
    inputs = {r["unit_id"]: r for r in csv.DictReader(open(inputs_path, newline=""))}
    seen, failed = set(), 0
    for row in csv.DictReader(open(outputs_path, newline="")):
        uid, problems = row.get("unit_id", ""), []
        try:
            for col in LIST_COLUMNS:
                row[col] = json.loads(row[col])
            jsonschema.validate(row, SCHEMA)
        except (json.JSONDecodeError, jsonschema.ValidationError, KeyError) as e:
            problems.append(f"schema: {getattr(e, 'message', e)}")
        if uid not in inputs:
            problems.append("unit_id not in inputs")
        elif uid in seen:
            problems.append("duplicate unit_id")
        else:
            source = norm(inputs[uid]["excerpt"] + " " + " ".join(json.loads(inputs[uid]["mentions"])) + " " + inputs[uid].get("parent", ""))
            for q in row["evidence"] if isinstance(row.get("evidence"), list) else []:
                if not quote_found(q, source):
                    problems.append(f"quote not found: {q[:60]!r}")
        seen.add(uid)
        failed += bool(problems)
        print(("FAIL  " if problems else "PASS  ") + uid + "".join(f"\n      {p}" for p in problems))
    for uid in inputs.keys() - seen:
        failed += 1
        print(f"FAIL  {uid}\n      no output row")
    print(f"\n{len(inputs) - failed if failed <= len(inputs) else 0}/{len(inputs)} input rows pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    argv = sys.argv[1:]
    task = "file-summary"
    if "--task" in argv:
        i = argv.index("--task")
        task = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) != 2 or task not in ("file-summary", "abstract-facts"):
        sys.exit(__doc__)
    if task == "abstract-facts":
        main_abstract(*argv)
    else:
        main(*argv)
