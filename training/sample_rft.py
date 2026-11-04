#!/usr/bin/env python3
"""Sample several answers per training batch from a Lightning checkpoint (rejection-sampling fine-tuning).

Runs inside an evaluation job against the local vLLM server. Sends each batch of the training
inputs with the summarizer prompt, thinking off, json_schema output, n samples at a temperature,
and writes one JSON line per batch: its prefix and file type, unit ids, and every sample's raw text.
Selection happens afterwards, by code (select_rft.py).

--task and --batch-size select the task as in run_model.py (tasks.py); the job's code channel
carries the task's prompt, examples and schema.

Usage: python3 sample_rft.py inputs.csv samples.jsonl [--n 4] [--temperature 0.8] [--workers 64]
                             [--task NAME] [--batch-size N]
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import run_model as rm  # noqa: E402

URL = "http://localhost:8000/v1/chat/completions"


def sample(key, rows, n, temperature):
    body = {"model": "lightning", "max_tokens": 4096, "n": n, "temperature": temperature,
            "messages": [{"role": "system", "content": rm.SYSTEM}, {"role": "user", "content": rm.user_message(rows)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "rows", "schema": rm.API_SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    rec = {"prefix": key[0], "file_type": key[1], "unit_ids": [r["unit_id"] for r in rows], "samples": [], "error": ""}
    try:
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as resp:
            reply = json.loads(resp.read())
        rec["samples"] = [(c.get("message") or {}).get("content") or "" for c in reply.get("choices", [])]
        rec["output_tokens"] = (reply.get("usage") or {}).get("completion_tokens")
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--task", default="file-summary")
    ap.add_argument("--batch-size", type=int)
    a = ap.parse_args()
    rm.set_task(a.task)
    csv.field_size_limit(sys.maxsize)
    work = rm.batches(list(csv.DictReader(open(a.inputs, newline=""))), size=a.batch_size)
    started = time.time()
    done = 0
    with ThreadPoolExecutor(a.workers) as pool, open(a.out, "w") as fh:
        for rec in pool.map(lambda kr: sample(kr[0], kr[1], a.n, a.temperature), work):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if done % 200 == 0:
                print(json.dumps({"sampled_batches": done, "of": len(work), "seconds": round(time.time() - started)}), flush=True)
    print(json.dumps({"sampled_batches": done, "of": len(work), "seconds": round(time.time() - started), "done": True}), flush=True)


if __name__ == "__main__":
    main()
