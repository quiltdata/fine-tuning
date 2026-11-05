#!/usr/bin/env python3
"""Run a task on Claude in Amazon Bedrock, recording tokens, latency and cost.

Uses the same task definition as run_model.py (--task; see tasks.py): the same system prompt,
examples, structured-output schema, batching and user message, so Claude sees exactly what any
other model sees. Writes to out_dir:

  outputs.csv  one row per input row, in the task's output schema
  calls.csv    one row per call: batch, rows, unit ids, model served, tokens, latency, cost, stop reason
  run.json     settings and totals, with the task's prompt, examples and schema hashes

Cost uses tools/bedrock_prices.json (AWS Pricing API). By default refusals fall back to Claude
Opus 4.8 through the SDK middleware, and calls.csv marks any call a fallback served. A baseline
should be the model itself: pass --no-fallback.
Check the result with: python3 tools/check_outputs.py inputs.csv out_dir/outputs.csv

Usage: python3 run_claude.py inputs.csv out_dir [--model ID] [--effort LEVEL] [--task NAME] [--prompt FILE] [--examples FILE]
                             [--batch-size N] [--no-fallback] [--workers N] [--limit N] [--region R]
                             [--price-basis regional|global]
"""
import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from anthropic import AnthropicBedrockMantle, BetaFallbackState, BetaRefusalFallbackMiddleware

import run_model as rm
import tasks

ROOT = Path(__file__).resolve().parent.parent
PRICES = json.loads((ROOT / "tools" / "bedrock_prices.json").read_text())["models"]
FALLBACK_MODEL = "anthropic.claude-opus-4-8"
CALL_COLUMNS = ["batch", "prefix", "file_type", "rows", "unit_ids", "model_requested", "model_served", "fallback",
                "stop_reason", "input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens",
                "latency_s", "cost_usd", "error"]


def cost(usage, model, basis):
    p = PRICES.get("anthropic." + bare(model), {}).get(basis)  # Bedrock reports e.g. claude-opus-5-5
    if not p:
        return None
    return (usage.input_tokens * p["input"] + usage.output_tokens * p["output"]
            + (usage.cache_creation_input_tokens or 0) * p["cache_write_5m"]
            + (usage.cache_read_input_tokens or 0) * p["cache_read"]) / 1e6


def bare(model):
    return model.removeprefix("us.").removeprefix("global.").removeprefix("anthropic.")


def call(client, args, index, key, rows):
    prefix, file_type = key
    record = {"batch": index, "prefix": prefix, "file_type": file_type, "rows": len(rows),
              "unit_ids": json.dumps([r["unit_id"] for r in rows]), "model_requested": args.model, "error": ""}
    start = time.perf_counter()
    try:
        with BetaFallbackState():  # one batch is one conversation
            resp = client.beta.messages.create(
                model=args.model,
                max_tokens=16000,
                system=[{"type": "text", "text": rm.SYSTEM, "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": args.effort, "format": {"type": "json_schema", "schema": rm.API_SCHEMA}},
                messages=[{"role": "user", "content": rm.user_message(rows)}],
            )
    except anthropic.APIStatusError as e:
        return {**record, "latency_s": round(time.perf_counter() - start, 3), "error": f"{e.status_code} {e.message}"[:300]}, []
    except anthropic.APIConnectionError as e:
        return {**record, "latency_s": round(time.perf_counter() - start, 3), "error": f"connection: {e}"[:300]}, []
    latency = time.perf_counter() - start
    u = resp.usage
    fallback = bare(resp.model or args.model) != bare(args.model)  # the middleware retried a refusal on FALLBACK_MODEL
    record.update(model_served=resp.model, fallback="yes" if fallback else "", stop_reason=resp.stop_reason,
                  input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                  cache_write_tokens=u.cache_creation_input_tokens or 0, cache_read_tokens=u.cache_read_input_tokens or 0,
                  latency_s=round(latency, 3))
    c = cost(u, resp.model or args.model, args.price_basis)
    record["cost_usd"] = round(c, 6) if c is not None else ""
    if resp.stop_reason != "end_turn":
        record["error"] = f"stop_reason {resp.stop_reason}"
        return record, []
    try:
        out = [rm.TASK.normalize(o) for o in rm.TASK.attach_ids(rm.parse_rows(next(b.text for b in resp.content if b.type == "text")), rows)]
    except (StopIteration, ValueError) as e:
        record["error"] = f"unparseable output: {e}"[:300]
        return record, []
    wanted = {r["unit_id"] for r in rows}
    stray = [o.get("unit_id") for o in out if o.get("unit_id") not in wanted]
    if stray:
        record["error"] = f"unknown unit_id in output: {stray[:3]}"
    return record, [o for o in out if o.get("unit_id") in wanted]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="anthropic.claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--task", default="file-summary", choices=sorted(tasks.TASKS))
    ap.add_argument("--examples", help="a draft examples file (path) instead of the published one")
    ap.add_argument("--prompt", help="a draft prompt file (path) instead of the published one "
                                     "(file-summary's fair baseline: summarizer_main_finding.md)")
    ap.add_argument("--batch-size", type=int, help="abstract-facts: rows per call (default 10)")
    ap.add_argument("--allow-no-examples", action="store_true", help="run without examples (drafts and tests; the run is marked unpublished)")
    ap.add_argument("--no-fallback", action="store_true", help="no refusal fallback: every row is answered by --model or not at all")
    ap.add_argument("--per-file", action="store_true", help="one row per call instead of one batch per prefix and file type")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N batches")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--price-basis", default="regional", choices=["regional", "global"])
    args = ap.parse_args()
    rm.set_task(args.task, args.prompt, args.allow_no_examples, examples=args.examples)

    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(open(args.inputs, newline="")))
    work = rm.batches(rows, args.per_file, args.batch_size)[: args.limit]
    client = AnthropicBedrockMantle(aws_region=args.region, max_retries=5,
                                    middleware=[] if args.no_fallback else [BetaRefusalFallbackMiddleware([{"model": FALLBACK_MODEL}])])
    started = datetime.now(timezone.utc)
    wall = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda w: call(client, args, *w), [(i, k, r) for i, (k, r) in enumerate(work)]))
    wall = time.perf_counter() - wall

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "outputs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rm.OUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for _, outs in results:
            for o in outs:
                w.writerow({c: json.dumps(o.get(c, []), ensure_ascii=False) if c in rm.LIST_COLUMNS else o.get(c, "") for c in rm.OUT_COLUMNS})
    with open(out / "calls.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CALL_COLUMNS)
        w.writeheader()
        w.writerows(rec for rec, _ in results)

    calls = [rec for rec, _ in results]
    ok = [c for c in calls if not c["error"]]
    lat = [c["latency_s"] for c in calls if c.get("latency_s") != ""]
    total = lambda f: sum(c.get(f) or 0 for c in calls if c.get(f) != "")
    summary = {
        "started": started.isoformat(timespec="seconds"), "model": args.model, "effort": args.effort,
        **rm.fingerprint(), "batch_size": args.batch_size, "per_file": args.per_file,
        "examples_allowed_missing": args.allow_no_examples,
        "region": args.region, "price_basis": args.price_basis, "fallback_model": None if args.no_fallback else FALLBACK_MODEL,
        "workers": args.workers,
        "input_rows": sum(len(r) for _, r in work), "output_rows": sum(len(o) for _, o in results),
        "calls": len(calls), "calls_ok": len(ok), "calls_fallback": sum(1 for c in calls if c.get("fallback")),
        "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
        "cache_write_tokens": total("cache_write_tokens"), "cache_read_tokens": total("cache_read_tokens"),
        "cost_usd": round(total("cost_usd"), 4),
        "latency_s_per_call": {"median": round(statistics.median(lat), 2), "max": round(max(lat), 2)} if lat else None,
        "wall_clock_s": round(wall, 1),
    }
    summary["cost_usd_per_row"] = round(summary["cost_usd"] / summary["output_rows"], 5) if summary["output_rows"] else None
    (out / "run.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
