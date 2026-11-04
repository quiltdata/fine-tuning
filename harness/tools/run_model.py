#!/usr/bin/env python3
"""Run the summarizer on any Bedrock model or on a SageMaker endpoint (Lightning).

Same task as run_claude.py: batches of one prefix and one file type, with
prompts/summarizer.md as the system prompt and rows in schemas/output.schema.json.
Each model gets the strongest structured output its server offers:

  bedrock    Converse with a forced tool whose input is the output table. If the model
             rejects a forced tool, the tool is offered instead; if it has no tools,
             it is asked for the JSON in its reply. calls.csv records which (mode).
  chat       Bedrock's OpenAI-compatible Chat Completions API (SigV4-signed), with the
             output table as a strict json_schema response_format. Use it where AWS
             recommends it over Converse (Kimi K3), or to set --service-tier and
             --reasoning-effort, which Converse can't. Flex is costed at half rate.
  openai     any OpenAI-compatible server (--openai-url), same request as sagemaker; used for
             quality runs inside an evaluation job, so no cost is recorded.
  sagemaker  OpenAI-style chat completions on a vLLM endpoint, with the output table as
             a json_schema response_format (guided decoding). --no-thinking turns off the
             model's reasoning through the chat template (Nemotron 3.5 Lightning honours it).

Bedrock cost is tokens times tools/model_rates.json, with cache tokens at the input
rate (an upper bound). A SageMaker endpoint is costed by instance time. By default that's
its hourly rate over its sustained throughput with every request slot busy, which is what
Bedrock's per-token prices assume of their own hardware. --cost-basis wall prices the run's
own wall clock instead. run.json's "instance" block reports both.

--required-facts facts.jsonl (make_facts.py) is for a teacher run: each batch's user message
ends with the facts every output row must convey. The rows it returns are scored and trained
on against the plain user message (make_pairs.py rebuilds that one), so the student never
sees the facts.

Writes outputs.csv, calls.csv and run.json to out_dir, like run_claude.py.

Usage: python3 run_model.py inputs.csv out_dir --model ID [--sagemaker [--no-thinking] | --chat [--service-tier T]
                            [--reasoning-effort E] | --openai-url URL [--no-thinking]] [--split split.csv --only test]
                            [--workers N] [--region R] [--max-tokens N] [--limit N] [--required-facts facts.jsonl]
"""
import argparse
import copy
import csv
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.config import Config

ROOT = Path(__file__).resolve().parent.parent
SYSTEM = (ROOT / "prompts" / "summarizer.md").read_text()
ROW_SCHEMA = json.loads((ROOT / "schemas" / "output.schema.json").read_text())
RATES = json.loads((ROOT / "tools" / "model_rates.json").read_text())
LIST_COLUMNS = ["about", "entities", "refers_to", "evidence"]
OUT_COLUMNS = ["unit_id", "what", "summary", "about", "entities", "refers_to", "evidence"]
CALL_COLUMNS = ["batch", "prefix", "file_type", "rows", "model_requested", "model_served", "mode", "attempts", "length_retry",
                "stop_reason", "input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens",
                "rows_out", "start_s", "latency_s", "cost_usd", "dropped_rows", "error"]
UNSUPPORTED = {"$schema", "$id", "title", "description", "minLength", "maxLength", "minItems", "maxItems"}
TEXT_INSTRUCTION = ('\n\nReply with only a JSON object of the form {"rows": [...]}, one object per input row with '
                    "the output columns above, and nothing else.")


def strip(schema):
    if isinstance(schema, dict):
        return {k: strip(v) for k, v in schema.items() if k not in UNSUPPORTED}
    if isinstance(schema, list):
        return [strip(v) for v in schema]
    return schema


API_SCHEMA = {"type": "object", "properties": {"rows": {"type": "array", "items": strip(copy.deepcopy(ROW_SCHEMA))}},
              "required": ["rows"], "additionalProperties": False}


def batches(rows, per_file=False):
    """Batches of one prefix and one file type; per_file=True sends each row alone (whole documents)."""
    if per_file:
        return [((r["prefix"], r["file_type"]), [r]) for r in rows]
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r["prefix"], r["file_type"]), []).append(r)
    return list(groups.items())


REQUIRED_FACTS = {}  # unit_id -> [fact, ...], from --required-facts; empty for every ordinary run


def user_message(rows, required=None):
    """The user turn for one batch, exactly as run_claude.py sends it. With `required` (a teacher
    run), the facts each row must convey follow the table; make_pairs.py never passes them."""
    table = [{**r, "members": int(r["members"]), "size_bytes": int(r["size_bytes"]),
              "example_keys": json.loads(r["example_keys"]), "mentions": json.loads(r["mentions"])} for r in rows]
    msg = "Input table, one JSON object per row:\n" + json.dumps(table, ensure_ascii=False)
    if required:
        facts = {r["unit_id"]: required[r["unit_id"]] for r in rows if required.get(r["unit_id"])}
        if facts:
            msg += ("\n\nRequired content: a reader of each output row must learn these facts about its unit. State the "
                    "gist in the summary and carry the specifics in about, entities and refers_to; a reader of the whole row "
                    "must learn each fact, but the summary alone need not spell them out. The summary limit of 300 characters "
                    "is enforced by code: write it, count its characters, and shorten it until it fits. Quotes stay at most "
                    "200 characters and verbatim.\n" + json.dumps(facts, ensure_ascii=False))
    return msg


def parse_rows(text):
    """Rows from a JSON reply, tolerating code fences, reasoning before the JSON, or a bare list."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1], text[text.find("["): text.rfind("]") + 1]):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return data["rows"]
        if isinstance(data, list):
            return data
    raise ValueError("no JSON rows in reply")


class Bedrock:
    def __init__(self, args):
        self.args = args
        self.client = boto3.client("bedrock-runtime", region_name=args.region, config=Config(
            read_timeout=600, connect_timeout=10, retries={"max_attempts": 10, "mode": "adaptive"}))
        self.mode = "tool"  # downgraded once if the model refuses it: tool -> tool-auto -> text
        self.max_tokens = args.max_tokens

    def request(self, rows, extra=""):
        req = dict(modelId=self.args.model, messages=[{"role": "user", "content": [{"text": user_message(rows, REQUIRED_FACTS) + extra}]}],
                   inferenceConfig={"maxTokens": self.max_tokens})
        tool = {"tools": [{"toolSpec": {"name": "record_rows", "inputSchema": {"json": API_SCHEMA}}}]}
        if self.mode == "tool":
            req.update(system=[{"text": SYSTEM}], toolConfig={**tool, "toolChoice": {"tool": {"name": "record_rows"}}})
        elif self.mode == "tool-auto":
            req.update(system=[{"text": SYSTEM + "\n\nAnswer only by calling the `record_rows` tool."}],
                       toolConfig={**tool, "toolChoice": {"auto": {}}})
        else:
            req.update(system=[{"text": SYSTEM + TEXT_INSTRUCTION}])
        return req

    def call(self, rows, extra=""):
        """Returns (rows or None, partial call record)."""
        while True:
            try:
                resp = self.client.converse(**self.request(rows, extra))
                break
            except self.client.exceptions.ValidationException as e:
                msg = str(e).lower()
                if self.mode == "tool" and "tool" in msg and ("choice" in msg or "support" in msg):
                    self.mode = "tool-auto"
                elif self.mode != "text" and "tool" in msg:
                    self.mode = "text"
                elif "token" in msg and self.max_tokens > 4096:
                    self.max_tokens //= 2
                else:
                    raise
        u = resp.get("usage", {})
        rec = {"model_served": self.args.model, "mode": self.mode, "stop_reason": resp.get("stopReason"),
               "input_tokens": u.get("inputTokens", 0), "output_tokens": u.get("outputTokens", 0),
               "cache_write_tokens": u.get("cacheWriteInputTokens", 0), "cache_read_tokens": u.get("cacheReadInputTokens", 0)}
        r = RATES["bedrock"].get(self.args.model)
        prompt = rec["input_tokens"] + rec["cache_write_tokens"] + rec["cache_read_tokens"]
        rec["cost_usd"] = round((prompt * r["input"] + rec["output_tokens"] * r["output"]) / 1e6, 6) if r else ""
        blocks = resp["output"]["message"]["content"]
        tool_input = next((b["toolUse"]["input"] for b in blocks if "toolUse" in b), None)
        if tool_input is not None:
            out = tool_input.get("rows") if isinstance(tool_input, dict) else tool_input
            if isinstance(out, str):  # some models put the table in the tool call as a JSON string
                out = parse_rows(out)
            return out, rec
        return parse_rows("".join(b.get("text", "") for b in blocks)), rec


TIER_MULTIPLIER = {"default": 1.0, "flex": 0.5, "priority": 1.75}  # Bedrock service tiers, applied to standard rates


class BedrockChat:
    """Bedrock's OpenAI-compatible Chat Completions endpoint, signed with the default AWS credentials."""

    def __init__(self, args):
        self.args = args
        self.url = f"https://bedrock-runtime.{args.region}.amazonaws.com/openai/v1/chat/completions"
        self.credentials = boto3.Session().get_credentials()

    def call(self, rows, extra=""):
        body = {"model": self.args.model, "max_tokens": self.args.max_tokens,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(rows, REQUIRED_FACTS) + extra}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "rows", "schema": API_SCHEMA, "strict": True}},
                "service_tier": self.args.service_tier}
        if self.args.reasoning_effort:
            body["reasoning_effort"] = self.args.reasoning_effort
        data = json.dumps(body)
        for wait in (5, 15, 45, 120, 300, None):  # throttling and transient server errors back off; others raise
            req = AWSRequest(method="POST", url=self.url, data=data, headers={"Content-Type": "application/json"})
            SigV4Auth(self.credentials.get_frozen_credentials(), "bedrock", self.args.region).add_auth(req)
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, data=req.body, headers=dict(req.headers),
                                                                   method="POST"), timeout=900) as resp:
                    reply = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or wait is None:
                    raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}")
                time.sleep(wait)
        choice = (reply.get("choices") or [{}])[0]
        usage = reply.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        rec = {"model_served": reply.get("model", self.args.model), "mode": f"chat-json_schema-{reply.get('service_tier', self.args.service_tier)}",
               "stop_reason": choice.get("finish_reason"), "input_tokens": usage.get("prompt_tokens", 0) - cached,
               "output_tokens": usage.get("completion_tokens", 0), "cache_write_tokens": 0, "cache_read_tokens": cached}
        r = RATES["bedrock"].get(self.args.model)
        m = TIER_MULTIPLIER.get(reply.get("service_tier", self.args.service_tier), 1.0)
        rec["cost_usd"] = round(m * (usage.get("prompt_tokens", 0) * r["input"] + rec["output_tokens"] * r["output"]) / 1e6, 6) if r else ""
        return parse_rows((choice.get("message") or {}).get("content") or ""), rec


class OpenAICompatible:
    """A vLLM (or other OpenAI-compatible) server at --openai-url, e.g. one started inside an evaluation
    job. Same request as the SageMaker backend. No cost: it's for quality runs, priced elsewhere."""

    def __init__(self, args):
        self.args = args
        self.url = args.openai_url.rstrip("/") + "/chat/completions"

    def call(self, rows, extra=""):
        body = {"model": self.args.model, "max_tokens": self.args.max_tokens,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(rows, REQUIRED_FACTS) + extra}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "rows", "schema": API_SCHEMA}}}
        if self.args.no_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as resp:
            reply = json.loads(resp.read())
        choice = (reply.get("choices") or [{}])[0]
        usage = reply.get("usage") or {}
        rec = {"model_served": reply.get("model", self.args.model), "mode": "json_schema" + ("-no-thinking" if self.args.no_thinking else ""),
               "stop_reason": choice.get("finish_reason"), "input_tokens": usage.get("prompt_tokens", 0),
               "output_tokens": usage.get("completion_tokens", 0), "cache_write_tokens": 0, "cache_read_tokens": 0, "cost_usd": ""}
        return parse_rows((choice.get("message") or {}).get("content") or ""), rec


class SageMaker:
    def __init__(self, args):
        self.args = args
        self.client = boto3.client("sagemaker-runtime", region_name=args.region, config=Config(
            read_timeout=900, connect_timeout=10, retries={"max_attempts": 2}))

    def call(self, rows, extra=""):
        body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(rows, REQUIRED_FACTS) + extra}],
                "max_tokens": self.args.max_tokens,
                "response_format": {"type": "json_schema", "json_schema": {"name": "rows", "schema": API_SCHEMA}}}
        if self.args.no_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        # Streamed: InvokeEndpoint gives up after 60 s, which a reasoning model under load exceeds.
        body.update(stream=True, stream_options={"include_usage": True})
        resp = self.client.invoke_endpoint_with_response_stream(EndpointName=self.args.model, ContentType="application/json",
                                                                Body=json.dumps(body).encode())
        buf, content, model, finish, usage = "", [], self.args.model, None, {}
        for event in resp["Body"]:
            buf += event.get("PayloadPart", {}).get("Bytes", b"").decode()
            *lines, buf = buf.split("\n")  # server-sent events; a line can span payload parts
            for line in lines:
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                model, usage = chunk.get("model", model), chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    content.append((choice.get("delta") or {}).get("content") or "")  # reasoning arrives separately
                    finish = choice.get("finish_reason") or finish
        rec = {"model_served": model, "mode": "json_schema" + ("-no-thinking" if self.args.no_thinking else ""),
               "stop_reason": finish, "input_tokens": usage.get("prompt_tokens", 0),
               "output_tokens": usage.get("completion_tokens", 0), "cache_write_tokens": 0, "cache_read_tokens": 0}
        return parse_rows("".join(content)), rec


T0 = time.perf_counter()  # reset at the start of the run; calls record their start relative to it


def sustained(calls, workers):
    """Rows returned per hour with every request slot busy: rows over slot-time, counting only calls
    that finished before the last call started (after that the queue drains and slots sit idle)."""
    last_start = max(c["start_s"] for c in calls)
    full = [c for c in calls if c["start_s"] + c["latency_s"] <= last_start]
    full = full if len(full) >= 20 else calls  # too few to tell: use them all
    rows_out = sum(c["rows_out"] for c in full)
    return (3600 * rows_out / (sum(c["latency_s"] for c in full) / workers) if rows_out else 0.0), len(full)


def too_long(out):
    """Teacher runs only: rows over the summary or quote limits, as feedback for one more attempt."""
    problems = []
    for o in out:
        if isinstance(o, dict):
            if len(o.get("summary") or "") > 300:
                problems.append(f'{o.get("unit_id")}: summary is {len(o["summary"])} characters, the limit is 300')
            for q in o.get("evidence") or []:
                if isinstance(q, str) and len(q) > 200:
                    problems.append(f'{o.get("unit_id")}: a quote is {len(q)} characters, the limit is 200')
    return problems


def call(backend, args, index, key, rows):
    prefix, file_type = key
    record = {"batch": index, "prefix": prefix, "file_type": file_type, "rows": len(rows),
              "model_requested": args.model, "error": ""}
    wanted = {r["unit_id"] for r in rows}
    start = time.perf_counter()
    record["start_s"] = round(start - T0, 3)
    extra, fed_back = "", False
    spent = {k: 0 for k in ("input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens", "cost_usd")}
    for attempt in (1, 2, 3):  # one retry on an error or an unusable reply; teacher runs get one more on a length problem
        record["attempts"] = attempt
        try:
            out, rec = backend.call(rows, extra)
            for k in spent:  # every attempt's tokens count
                spent[k] += rec.get(k) or 0
            record.update(rec, **{k: (round(v, 6) if k == "cost_usd" else v) for k, v in spent.items()})
            if rec.get("cost_usd") == "":
                record["cost_usd"] = ""
            if REQUIRED_FACTS and not fed_back and too_long(out):
                fed_back = True
                extra = ("\n\nYour previous answer broke these limits. Answer again, shorter, keeping the required facts:\n" +
                         "\n".join(too_long(out)))
                record["length_retry"] = 1
                continue
            first = {}
            for o in out:  # keep a model's first row per unit; repeats (a looping reply) are counted, not kept
                if isinstance(o, dict) and o.get("unit_id") in wanted:
                    first.setdefault(o["unit_id"], o)
            record["dropped_rows"] = sum(1 for o in out if isinstance(o, dict)) - len(first)
            out = list(first.values())
            got = set(first)
            record["error"] = "" if wanted <= got else f"missing unit_id in output: {len(wanted - got)} of {len(wanted)}"
            if not record["error"] or attempt >= 2 + int(fed_back):
                record.update(latency_s=round(time.perf_counter() - start, 3), rows_out=len(out))
                return record, out
        except Exception as e:  # recorded; throttling is retried inside boto3
            record["error"] = f"{type(e).__name__}: {e}"[:300]
            if attempt >= 2 + int(fed_back):
                break
    record.update(latency_s=round(time.perf_counter() - start, 3), rows_out=0)
    return record, []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True, help="Bedrock model or inference profile id, or SageMaker endpoint name")
    ap.add_argument("--sagemaker", action="store_true", help="--model is a SageMaker endpoint (vLLM, OpenAI chat format)")
    ap.add_argument("--chat", action="store_true", help="Bedrock Chat Completions API instead of Converse")
    ap.add_argument("--openai-url", help="an OpenAI-compatible server, e.g. http://localhost:8000/v1 (no cost recorded)")
    ap.add_argument("--service-tier", default="default", choices=sorted(TIER_MULTIPLIER), help="--chat only")
    ap.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh", "max"], help="--chat only")
    ap.add_argument("--no-thinking", action="store_true", help="SageMaker only: chat_template_kwargs enable_thinking=false")
    ap.add_argument("--split", help="split.csv; with --only, run just those articles")
    ap.add_argument("--only", choices=["train", "test"])
    ap.add_argument("--per-file", action="store_true", help="one row per call instead of one batch per prefix and file type")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--limit", type=int, help="only the first N batches")
    ap.add_argument("--cost-basis", choices=["sustained", "wall"], default="sustained",
                    help="SageMaker only: price rows at sustained throughput, or by this run's wall clock")
    ap.add_argument("--required-facts", help="teacher run: facts.jsonl whose key facts each output row must convey")
    args = ap.parse_args()
    if args.required_facts:
        for f in map(json.loads, open(args.required_facts)):
            REQUIRED_FACTS[f["unit_id"]] = [k["fact"] for k in f["key_facts"]]

    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(open(args.inputs, newline="")))
    if args.only:
        keep = {r["prefix"] for r in csv.DictReader(open(args.split, newline="")) if r["split"] == args.only}
        rows = [r for r in rows if r["prefix"] in keep]
    work = batches(rows, args.per_file)[: args.limit]
    backend = (OpenAICompatible(args) if args.openai_url else SageMaker(args) if args.sagemaker
               else BedrockChat(args) if args.chat else Bedrock(args))
    global T0
    started = datetime.now(timezone.utc)
    wall = T0 = time.perf_counter()
    if not (args.sagemaker or args.chat or args.openai_url):  # settle the output mode on one batch before going parallel
        first = [call(backend, args, 0, *work[0])] if work else []
        work_rest = [(i, k, r) for i, (k, r) in enumerate(work)][1:]
    else:
        first, work_rest = [], [(i, k, r) for i, (k, r) in enumerate(work)]
    with ThreadPoolExecutor(args.workers) as pool:
        results = first + list(pool.map(lambda w: call(backend, args, *w), work_rest))
    wall = time.perf_counter() - wall

    instance = None
    if args.sagemaker:  # instance time per row, at sustained throughput (default) or as this run used it
        rate = RATES["sagemaker"][args.model]["usd_per_hour"]
        n = sum(rec["rows"] for rec, _ in results) or 1
        rows_per_hour, basis_calls = sustained([rec for rec, _ in results], args.workers)
        n = sum(rec["rows_out"] for rec, _ in results) or 1  # rows returned, not requested
        per_row = {"sustained": rate / rows_per_hour if rows_per_hour else 0.0, "wall": wall / 3600 * rate / n}
        for rec, _ in results:
            rec["cost_usd"] = round(per_row[args.cost_basis] * rec["rows"], 6)
        instance = {"usd_per_hour": rate, "cost_basis": args.cost_basis, "concurrency": args.workers,
                    "sustained_rows_per_hour": round(rows_per_hour), "sustained_from_calls": basis_calls,
                    "as_run_rows_per_hour": round(3600 * n / wall), "usd_per_100_rows_sustained": round(100 * per_row["sustained"], 4),
                    "usd_per_100_rows_as_run": round(100 * per_row["wall"], 4)}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "outputs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for _, outs in results:
            for o in outs:
                w.writerow({c: json.dumps(o.get(c, []), ensure_ascii=False) if c in LIST_COLUMNS else o.get(c, "") for c in OUT_COLUMNS})
    with open(out / "calls.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CALL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rec for rec, _ in results)

    calls = [rec for rec, _ in results]
    lat = [c["latency_s"] for c in calls]
    total = lambda f: sum(c.get(f) or 0 for c in calls if c.get(f) != "")
    summary = {
        "started": started.isoformat(timespec="seconds"), "model": args.model,
        "backend": ("openai-compatible " + args.openai_url if args.openai_url else "sagemaker" if args.sagemaker
                    else "bedrock-chat" if args.chat else "bedrock"), "region": args.region,
        "workers": args.workers, "service_tier": args.service_tier if args.chat else None,
        "reasoning_effort": args.reasoning_effort if args.chat else None,
        "required_facts": args.required_facts,
        "rows_from": f"{args.split} ({args.only})" if args.only else args.inputs,
        "rate": None if args.openai_url else RATES["sagemaker" if args.sagemaker else "bedrock"].get(args.model),
        "modes": sorted({c.get("mode") for c in calls if c.get("mode")}),
        "input_rows": sum(len(r) for _, r in work), "output_rows": sum(len(o) for _, o in results),
        "calls": len(calls), "calls_ok": sum(1 for c in calls if not c["error"]),
        "dropped_rows": sum(c.get("dropped_rows") or 0 for c in calls),
        "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
        "cache_write_tokens": total("cache_write_tokens"), "cache_read_tokens": total("cache_read_tokens"),
        "cost_usd": round(total("cost_usd"), 4),
        "latency_s_per_call": {"median": round(statistics.median(lat), 2), "max": round(max(lat), 2)} if lat else None,
        "wall_clock_s": round(wall, 1),
        "instance": instance,
    }
    summary["cost_usd_per_row"] = round(summary["cost_usd"] / summary["output_rows"], 5) if summary["output_rows"] else None
    (out / "run.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
