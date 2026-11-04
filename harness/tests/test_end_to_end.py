"""The abstract task through the real command-line tools, offline: run, check, pairs, score.

The model is a local OpenAI-compatible server stub; the prompt is a throwaway test prompt in a
temporary harness root (the real one is written by a person); the grader is mocked.
"""
import csv
import functools
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import check_outputs
import make_pairs
import run_model as rm
import score_abstracts
import tasks
from test_score_abstracts import fake_grader

ABSTRACTS = [{"unit_id": f"PMC{i}.1", "abstract": f"In {i} mice, drug Z lowered tumour growth by {i}0%.", "protocol": "0"}
             for i in range(1, 13)] + [{"unit_id": "PMC99.1", "abstract": "We will randomize 200 adults.", "protocol": "1"}]


class Stub(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.requests.append(body)
        table = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        rows = [{"unit_id": r["unit_id"], "about": {"fact": "A mouse study of drug Z.", "quote": ""},
                 "findings": [] if "will" in r["abstract"] else [{"fact": "Drug Z slowed tumour growth.", "quote": "lowered tumour growth"}]}
                for r in table]
        reply = {"model": "stub", "choices": [{"message": {"content": json.dumps({"rows": rows})}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_abstract_task_end_to_end(tmp_path, monkeypatch, capsys, server):
    root = tmp_path / "harness"
    (root / "prompts").mkdir(parents=True)
    (root / "schemas").mkdir()
    shutil.copy(tasks.ROOT / "schemas" / "abstract_output.schema.json", root / "schemas")
    (root / "prompts" / "abstract_facts.md").write_text("TEST PROMPT\n")
    monkeypatch.setattr(rm, "set_task", functools.partial(rm.set_task, root=root))
    sd, rd = tmp_path / "set", tmp_path / "run"
    (sd / "reference").mkdir(parents=True)
    with open(sd / "inputs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit_id", "abstract", "protocol"])
        w.writeheader()
        w.writerows(ABSTRACTS)

    # run: 13 abstracts in batches of 10 and 3, thinking off, through the stub server
    monkeypatch.setattr(sys, "argv", ["run_model.py", str(sd / "inputs.csv"), str(rd), "--model", "stub", "--openai-url", server,
                                      "--no-thinking", "--task", "abstract-facts", "--allow-no-examples", "--workers", "2"])
    rm.main()
    run = json.loads((rd / "run.json").read_text())
    assert run["task"] == "abstract-facts" and run["prompt_sha256"] and run["examples_sha256"] is None
    assert run["examples_allowed_missing"] is True and run["output_rows"] == 13 and run["calls_ok"] == 2
    assert sorted(len(json.loads(r["messages"][1]["content"].split("\n", 1)[1])) for r in Stub.requests) == [3, 10]
    assert all(r["messages"][0]["content"] == "TEST PROMPT\n" for r in Stub.requests)
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in Stub.requests)
    out = {r["unit_id"]: r for r in csv.DictReader(open(rd / "outputs.csv", newline=""))}
    assert json.loads(out["PMC1.1"]["about"]) == {"fact": "A mouse study of drug Z."}  # "" quote dropped
    assert json.loads(out["PMC99.1"]["findings"]) == []

    # check
    with pytest.raises(SystemExit) as e:
        check_outputs.main_abstract(str(sd / "inputs.csv"), str(rd / "outputs.csv"))
    check = capsys.readouterr().out
    assert e.value.code == 0 and "13/13 input rows pass" in check and "12/12 quotes found" in check
    (rd / "check.txt").write_text(check)

    # pairs, regrouped into batches of 1-10; the student's user turn carries no required facts
    pairs_path = tmp_path / "pairs.jsonl"
    monkeypatch.setattr(sys, "argv", ["make_pairs.py", str(sd / "inputs.csv"), str(rd), str(pairs_path), "--mixed-batches", "7"])
    make_pairs.main()
    pairs = [json.loads(l) for l in open(pairs_path)]
    assert sum(p["rows"] for p in pairs) == 13 and all(p["rows"] == p["rows_passing_check"] for p in pairs)
    assert all(p["system"] == "TEST PROMPT\n" and "Required content" not in p["user"] for p in pairs)
    assert all(p["teacher_model"] == "stub" for p in pairs)

    # score, with a mocked grader and a key
    keys = [{"unit_id": a["unit_id"], "key_facts": [{"id": "about", "role": "about", "fact": "An RCT of a diet.", "quote": "q"}]
             + ([] if a["protocol"] == "1" else [{"id": "f1", "role": "finding", "fact": "HbA1c fell.", "quote": "q"}])}
            for a in ABSTRACTS]
    (sd / "reference" / "facts.jsonl").write_text("".join(json.dumps(k) + "\n" for k in keys))
    monkeypatch.setattr(score_abstracts, "call_tool", fake_grader)
    monkeypatch.setattr(sys, "argv", ["score_abstracts.py", str(sd), str(rd)])
    score_abstracts.main()
    summary = json.loads((rd / "score.json").read_text())
    assert summary["task"] == "abstract-facts" and summary["all"]["rows"] == 13 and summary["all"]["valid"] == 13
    assert summary["protocol"]["invented_findings"] == 0
    rm.set_task.func("file-summary")
