#!/usr/bin/env python3
"""Build the answer key for a summary set: required facts per input row, model-independent.

Every model's output is scored against the same facts (score_run.py). Two kinds:

  code facts  checked by code, from fields the excerpt states. Article rows (.txt) only:
              what = article, and the PMCID, DOI, PMID, license and journal it names.
  key facts   3-5 content facts written once by a reference model (prompts/facts.md) from
              the row's excerpt and mentions alone. Each carries a verbatim quote; a fact
              whose quote isn't in the input is retried once, then dropped.
              --prompt facts_top2.md --max-facts 2 writes the two-fact key instead (top-2 metric).

Rows whose article is excluded in split.csv are skipped; pass "none" for a set without a split
(every row then gets --split-label, default "train"). --chat uses Bedrock's Chat Completions API
with strict json_schema output, for a reference model without Converse tool use (Kimi K3), at the
Flex tier.

--task abstract-facts writes the abstract task's typed key (prompts/abstract_key.md) from the
abstract alone: one `about` fact and zero to two `findings` (zero only when the abstract reports no
results), each with a verbatim quote checked by check_outputs.quote_found. A fact whose quote isn't
found is retried once, then dropped; extra findings past two are dropped and counted. Key facts get
ids "about", "f1", "f2" and a role. No code facts.

Writes to out_dir:

  facts.jsonl  one line per row: unit_id, prefix, file_type, split, code_facts, key_facts, dropped
  calls.csv    one line per reference-model call
  run.json     settings and totals

Usage: python3 make_facts.py inputs.csv split.csv|none out_dir [--model ID] [--region R] [--workers N] [--limit N]
                             [--prompt facts.md] [--max-facts 5] [--chat [--service-tier flex] [--reasoning-effort E]]
                             [--task file-summary|abstract-facts] [--split-label train|test]
"""
import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from bedrock_chat import call_json
from bedrock_tool import call_tool
from check_outputs import quote_found

ROOT = Path(__file__).resolve().parent.parent
SYSTEM = None  # read from --prompt in main()
SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {
        "type": "object",
        "properties": {"fact": {"type": "string"}, "quote": {"type": "string"}},
        "required": ["fact", "quote"]}}},
    "required": ["facts"],
}
_KEY_FACT = {"type": "object", "properties": {"fact": {"type": "string"}, "quote": {"type": "string"}},
             "required": ["fact", "quote"], "additionalProperties": False}
ABSTRACT_SCHEMA = {"type": "object", "properties": {"about": _KEY_FACT, "findings": {"type": "array", "items": _KEY_FACT}},
                   "required": ["about", "findings"], "additionalProperties": False}
MAX_FINDINGS = 2
CALL_COLUMNS = ["unit_id", "attempt", "model", "stop_reason", "input_tokens", "cache_write_tokens", "cache_read_tokens",
                "output_tokens", "latency_s", "cost_usd", "error"]


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def header(excerpt, key):
    m = re.search(rf"^{key}: *(.+)$", excerpt, re.M)
    return m.group(1).strip() if m else None


def code_facts(row):
    """Facts code can check, from the article header that make_inputs.py extracts from PMC .txt files."""
    if row["file_type"] != "txt":
        return []
    x = row["excerpt"]
    facts = [{"id": "what", "check": "what", "expect": ["article"]}]
    for key, label in (("PMCID", "pmcid"), ("DOI", "doi"), ("PMID", "pmid")):
        v = header(x, key)
        if v:
            facts.append({"id": label, "check": "entity_contains", "expect": v})
    if header(x, "License") or header(x, "License URL"):
        facts.append({"id": "license", "check": "license"})
    journals = {v for v in (header(x, "NLM Title Abbreviation"), header(x, "Journal ID")) if v}
    m = re.search(r"^NLM Journal ID:.*\n((?:.*\n){1,4}?)(?:E?ISSN|Publisher):", x, re.M)  # full title sits below the IDs
    if m:
        journals |= {ln.strip() for ln in m.group(1).splitlines() if ln.strip()}
    if journals:
        facts.append({"id": "journal", "check": "text_contains_any", "expect": sorted(journals)})
    return facts


def user_message(row, failed=None):
    payload = {k: row[k] for k in ("unit_id", "unit", "file_type", "excerpt_method", "excerpt")}
    payload["mentions"] = json.loads(row["mentions"])
    if row.get("parent"):
        payload["parent"] = row["parent"]
    msg = "Input row:\n" + json.dumps(payload, ensure_ascii=False)
    if failed:
        msg += ("\n\nIn your previous answer these quotes were not found word for word in the excerpt or mentions. "
                "Return all facts again, copying each quote exactly:\n" + json.dumps(failed, ensure_ascii=False))
    return msg


def build(row, args):
    source = norm(row["excerpt"] + " " + " ".join(json.loads(row["mentions"])) + " " + row.get("parent", ""))
    records, failed, kept = [], None, []
    for attempt in (1, 2):
        if args.chat:
            out, rec = call_json(args.model, args.region, SYSTEM, user_message(row, failed), "record_facts", SCHEMA,
                                 service_tier=args.service_tier, reasoning_effort=args.reasoning_effort)
        else:
            out, rec = call_tool(args.model, args.region, SYSTEM,
                                 [{"role": "user", "content": [{"text": user_message(row, failed)}]}], "record_facts", SCHEMA)
        records.append({"unit_id": row["unit_id"], "attempt": attempt, **rec})
        if out is None:
            continue
        facts = [f for f in out.get("facts", []) if isinstance(f, dict) and f.get("fact") and f.get("quote")][: args.max_facts]
        kept = [f for f in facts if norm(f["quote"]) in source]
        failed = [f["quote"] for f in facts if norm(f["quote"]) not in source]
        if not failed:
            break
    dropped = failed or []
    key = [{"id": f"k{i + 1}", "fact": f["fact"], "quote": f["quote"]} for i, f in enumerate(kept)]
    return {"unit_id": row["unit_id"], "prefix": row["prefix"], "file_type": row["file_type"],
            "code_facts": code_facts(row), "key_facts": key, "dropped": dropped}, records


def abstract_message(row, failed=None):
    msg = "Abstract:\n" + json.dumps({"unit_id": row["unit_id"], "abstract": row["abstract"]}, ensure_ascii=False)
    if failed:
        msg += ("\n\nIn your previous answer these quotes were not found word for word in the abstract. "
                "Return the whole key again, copying each quote exactly:\n" + json.dumps(failed, ensure_ascii=False))
    return msg


def abstract_key(out, abstract):
    """(key facts whose quotes are found, quotes not found, findings past two) from one key-writer reply."""
    def ok(f):
        return isinstance(f, dict) and (f.get("fact") or "").strip() and (f.get("quote") or "").strip()
    about = out.get("about") if ok(out.get("about")) else None
    findings = [f for f in out.get("findings") or [] if ok(f)]
    extra = max(0, len(findings) - MAX_FINDINGS)
    candidates = ([("about", "about", about)] if about else []) + \
                 [(f"f{i + 1}", "finding", f) for i, f in enumerate(findings[:MAX_FINDINGS])]
    kept = [{"id": i, "role": role, "fact": f["fact"].strip(), "quote": f["quote"].strip()}
            for i, role, f in candidates if quote_found(f["quote"], abstract)]
    failed = [f["quote"] for _, _, f in candidates if not quote_found(f["quote"], abstract)]
    return kept, failed, extra


def build_abstract(row, args):
    records, failed, kept, extra = [], None, [], 0
    for attempt in (1, 2):
        if args.chat:
            out, rec = call_json(args.model, args.region, SYSTEM, abstract_message(row, failed), "record_key", ABSTRACT_SCHEMA,
                                 service_tier=args.service_tier, reasoning_effort=args.reasoning_effort)
        else:
            out, rec = call_tool(args.model, args.region, SYSTEM,
                                 [{"role": "user", "content": [{"text": abstract_message(row, failed)}]}], "record_key", ABSTRACT_SCHEMA)
        records.append({"unit_id": row["unit_id"], "attempt": attempt, **rec})
        if out is None:
            continue
        kept, failed, extra = abstract_key(out, row["abstract"])
        if not failed:
            break
    return {"unit_id": row["unit_id"], "prefix": "", "file_type": "abstract", "protocol": int(row.get("protocol") or 0),
            "code_facts": [], "key_facts": kept, "dropped": failed or [], "extra_findings_dropped": extra}, records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("split")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="us.openai.gpt-6.1-sol")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only the first N rows")
    ap.add_argument("--prompt", help="file in prompts/ (default facts.md, or abstract_key.md for abstract-facts)")
    ap.add_argument("--max-facts", type=int, default=5)
    ap.add_argument("--chat", action="store_true", help="Bedrock Chat Completions with json_schema instead of Converse tools")
    ap.add_argument("--service-tier", default="flex", help="--chat only")
    ap.add_argument("--reasoning-effort", help="--chat only")
    ap.add_argument("--task", default="file-summary", choices=["file-summary", "abstract-facts"])
    ap.add_argument("--split-label", default="train", choices=["train", "test"], help="every row's split when split is none")
    args = ap.parse_args()
    abstract = args.task == "abstract-facts"
    args.prompt = args.prompt or ("abstract_key.md" if abstract else "facts.md")
    global SYSTEM
    SYSTEM = (ROOT / "prompts" / args.prompt).read_text()

    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(open(args.inputs, newline="")))
    key = (lambda r: r["unit_id"]) if abstract else (lambda r: r["prefix"])
    if args.split == "none":
        split = {key(r): args.split_label for r in rows}
    elif abstract:
        sys.exit("abstract-facts sets are separate files: pass split none and --split-label")
    else:
        split = {r["prefix"]: r["split"] for r in csv.DictReader(open(args.split, newline=""))}
    rows = [r for r in rows if split.get(key(r)) in ("train", "test")][: args.limit]
    started, wall = datetime.now(timezone.utc), time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda r: (build_abstract if abstract else build)(r, args), rows))
    wall = time.perf_counter() - wall

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "facts.jsonl", "w") as fh:
        for f, _ in results:
            fh.write(json.dumps({**f, "split": split[f["unit_id"] if abstract else f["prefix"]]}, ensure_ascii=False) + "\n")
    calls = [rec for _, recs in results for rec in recs]
    with open(out / "calls.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CALL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(calls)
    facts = [f for f, _ in results]
    summary = {
        "started": started.isoformat(timespec="seconds"), "model": args.model, "region": args.region, "workers": args.workers,
        "task": args.task, "prompt": args.prompt, "max_facts": None if abstract else args.max_facts,
        "api": ("chat " + args.service_tier + (f" reasoning {args.reasoning_effort}" if args.reasoning_effort else "")) if args.chat else "converse",
        "rows": len(facts), "rows_without_key_facts": sum(1 for f in facts if not f["key_facts"]),
        "key_facts": sum(len(f["key_facts"]) for f in facts), "code_facts": sum(len(f["code_facts"]) for f in facts),
        "quotes_dropped": sum(len(f["dropped"]) for f in facts),
        **({"rows_without_about": sum(1 for f in facts if not any(k["role"] == "about" for k in f["key_facts"])),
            "rows_without_findings": sum(1 for f in facts if not any(k["role"] == "finding" for k in f["key_facts"])),
            "extra_findings_dropped": sum(f["extra_findings_dropped"] for f in facts)} if abstract else {}),
        "calls": len(calls), "calls_failed": sum(1 for c in calls if c["error"]),
        **{f: sum(c.get(f) or 0 for c in calls) for f in ("input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens")},
        "cost_usd": round(sum(c.get("cost_usd") or 0 for c in calls), 4), "wall_clock_s": round(wall, 1),
    }
    (out / "run.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
