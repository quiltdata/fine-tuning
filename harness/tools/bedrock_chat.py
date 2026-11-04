"""One structured call on Bedrock's Chat Completions API, for models without Converse tool use.

Kimi K3 answers through this endpoint with strict json_schema output (run_model.py --chat uses the
same request for the summarizer). This is the small-call version for the reference and grader
roles: system prompt, one user message, a schema, and the Flex tier at half price. The record has
the same fields as bedrock_tool.call_tool so make_facts.py and the reward stage can use either.
"""
import json
import time
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from bedrock_tool import RATES

TIER_MULTIPLIER = {"default": 1.0, "flex": 0.5, "priority": 1.75}
_credentials = None


def call_json(model, region, system, user, name, schema, max_tokens=4000, service_tier="flex", reasoning_effort=None):
    """Returns (parsed object or None, record)."""
    global _credentials
    if _credentials is None:
        _credentials = boto3.Session().get_credentials()
    url = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/chat/completions"
    body = {"model": model, "max_tokens": max_tokens, "service_tier": service_tier,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    data = json.dumps(body)
    start = time.perf_counter()
    record = {"model": model, "error": ""}
    reply = None
    for wait in (5, 15, 45, 120, 300, None):
        req = AWSRequest(method="POST", url=url, data=data, headers={"Content-Type": "application/json"})
        SigV4Auth(_credentials.get_frozen_credentials(), "bedrock", region).add_auth(req)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=req.body, headers=dict(req.headers), method="POST"),
                                        timeout=600) as resp:
                reply = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or wait is None:
                record.update(latency_s=round(time.perf_counter() - start, 3), error=f"HTTP {e.code}: {e.read()[:200]!r}")
                return None, record
            time.sleep(wait)
        except Exception as e:
            if wait is None:
                record.update(latency_s=round(time.perf_counter() - start, 3), error=f"{type(e).__name__}: {e}"[:300])
                return None, record
            time.sleep(wait)
    choice = (reply.get("choices") or [{}])[0]
    usage = reply.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    record.update(latency_s=round(time.perf_counter() - start, 3), stop_reason=choice.get("finish_reason"),
                  input_tokens=usage.get("prompt_tokens", 0) - cached, output_tokens=usage.get("completion_tokens", 0),
                  cache_write_tokens=0, cache_read_tokens=cached)
    r = RATES.get(model)
    m = TIER_MULTIPLIER.get(reply.get("service_tier", service_tier), 1.0)
    record["cost_usd"] = round(m * (usage.get("prompt_tokens", 0) * r["input"] + record["output_tokens"] * r["output"]) / 1e6, 6) if r else None
    text = (choice.get("message") or {}).get("content") or ""
    try:
        return json.loads(text), record
    except json.JSONDecodeError:
        record["error"] = f"no JSON object (stop_reason {choice.get('finish_reason')})"
        return None, record
