"""One tool call on Bedrock Converse, for the reference and grader models.

Shared by make_facts.py and score_run.py. Any Converse model that supports tools works
(OpenAI, Claude, ...). RATES comes from tools/model_rates.json, in USD per 1M tokens.

Bedrock reports OpenAI prompts mostly as cache-write tokens (inputTokens can be 2 for a
4,000-token prompt), Claude prompts as inputTokens. Cost counts all three at the input
rate: no cache rate is published for these models, so it's an upper bound.
"""
import json
import time
from pathlib import Path

import boto3
from botocore.config import Config

RATES = json.loads((Path(__file__).resolve().parent / "model_rates.json").read_text())["bedrock"]
_clients = {}


def client(region):
    if region not in _clients:
        _clients[region] = boto3.client("bedrock-runtime", region_name=region, config=Config(
            read_timeout=180, connect_timeout=10, retries={"max_attempts": 8, "mode": "adaptive"}))
    return _clients[region]


def usd(model, prompt_tokens, output_tokens):
    r = RATES.get(model)
    return round((prompt_tokens * r["input"] + output_tokens * r["output"]) / 1e6, 6) if r else None


def call_tool(model, region, system, messages, name, schema, max_tokens=4000):
    """Returns (tool input dict or None, record) where record has tokens, latency, cost, stop reason, error."""
    start = time.perf_counter()
    record = {"model": model, "error": ""}
    tools = [{"toolSpec": {"name": name, "inputSchema": {"json": schema}}}]
    request = dict(modelId=model, system=[{"text": system}], messages=messages, inferenceConfig={"maxTokens": max_tokens},
                   toolConfig={"tools": tools, "toolChoice": {"tool": {"name": name}}})
    try:
        try:
            resp = client(region).converse(**request)
        except client(region).exceptions.ValidationException as e:
            if "tool_choice" not in str(e):
                raise
            # Some models (Claude Sonnet 5.5 on Bedrock) reject a forced tool; ask for it instead.
            request["toolConfig"]["toolChoice"] = {"auto": {}}
            request["system"] = [{"text": system + f"\n\nAnswer only by calling the `{name}` tool."}]
            resp = client(region).converse(**request)
    except Exception as e:  # throttling is retried by botocore; anything left is recorded
        record.update(latency_s=round(time.perf_counter() - start, 3), error=f"{type(e).__name__}: {e}"[:300])
        return None, record
    u = resp.get("usage", {})
    record.update(latency_s=round(time.perf_counter() - start, 3), stop_reason=resp.get("stopReason"),
                  input_tokens=u.get("inputTokens", 0), output_tokens=u.get("outputTokens", 0),
                  cache_write_tokens=u.get("cacheWriteInputTokens", 0), cache_read_tokens=u.get("cacheReadInputTokens", 0))
    prompt = record["input_tokens"] + record["cache_write_tokens"] + record["cache_read_tokens"]
    record["cost_usd"] = usd(model, prompt, record["output_tokens"])
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]["input"], record
    record["error"] = f"no tool call (stop_reason {resp.get('stopReason')})"
    return None, record
