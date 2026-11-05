"""The abstract task through the real command-line tools, offline: run, check, pairs, score.

The model is a local OpenAI-compatible server stub; the prompt and examples are throwaway test
files in a temporary data package (the real ones are published, written by a person); the grader
is mocked. A draft run of the same task is then refused by the training path.
"""
import csv
import functools
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import check_outputs
import make_pairs
import make_sft
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
        blocks = re.findall(r"Abstract (\d+):\n(.*?)(?=\n\nAbstract \d+:|\Z)", body["messages"][1]["content"], re.S)  # numbered text (35e)
        answers = [{"abstract": int(n), "about": {"fact": "A mouse study of drug Z.", "quote": ""},
                    "findings": [] if "will" in text else [{"fact": "Drug Z slowed tumour growth.", "quote": "lowered tumour growth"}]}
                   for n, text in reversed(blocks)]  # out of order on purpose: matched by number
        reply = {"model": "stub", "choices": [{"message": {"content": json.dumps({"answers": answers})}, "finish_reason": "stop"}],
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
    data = tmp_path / "data"
    (data / "task").mkdir(parents=True)
    (data / "task" / "abstract_facts.md").write_text("TEST PROMPT\n")
    (data / "task" / "abstract_examples.md").write_text("TEST EXAMPLES\n")
    monkeypatch.setattr(rm, "set_task", functools.partial(rm.set_task, data=data))
    sd, rd = tmp_path / "set", tmp_path / "run"
    (sd / "reference").mkdir(parents=True)
    with open(sd / "inputs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit_id", "abstract", "protocol"])
        w.writeheader()
        w.writerows(ABSTRACTS)

    # run: 13 abstracts, five per call, thinking off, through the stub server
    monkeypatch.setattr(sys, "argv", ["run_model.py", str(sd / "inputs.csv"), str(rd), "--model", "stub", "--openai-url", server,
                                      "--no-thinking", "--task", "abstract-facts", "--workers", "2"])
    rm.main()
    run = json.loads((rd / "run.json").read_text())
    assert run["task"] == "abstract-facts" and run["published"] is True and run["examples_sha256"]
    assert run["prompt"] == "abstract_facts.md" and str(tmp_path) not in json.dumps(run["prompt"])
    assert run["output_rows"] == 13 and run["calls_ok"] == 3
    sent = [re.findall(r"Abstract \d+:\n(.*?)(?=\n\nAbstract \d+:|\Z)", r["messages"][1]["content"], re.S) for r in Stub.requests]
    assert sorted(len(b) for b in sent) == [3, 5, 5] and sorted(t for b in sent for t in b) == sorted(a["abstract"] for a in ABSTRACTS)
    assert all("PMC" not in r["messages"][1]["content"] for r in Stub.requests)  # no ids reach the model
    assert all(r["messages"][0]["content"] == "TEST PROMPT\n\nTEST EXAMPLES\n" for r in Stub.requests)
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

    # pairs, as the run batched them: numbered abstracts in, numbered answers out, no ids
    pairs_path = tmp_path / "pairs.jsonl"
    monkeypatch.setattr(sys, "argv", ["make_pairs.py", str(sd / "inputs.csv"), str(rd), str(pairs_path)])
    make_pairs.main()
    pairs = [json.loads(l) for l in open(pairs_path)]
    assert [p["rows"] for p in pairs] == [5, 5, 3] and all(p["rows"] == p["rows_passing_check"] for p in pairs)
    for p in pairs:
        answers = json.loads(p["assistant"])["answers"]
        assert [x["abstract"] for x in answers] == list(range(1, p["rows"] + 1)) and "unit_id" not in p["assistant"]
        assert p["user"].startswith("Abstract 1:\n") and "PMC" not in p["user"]
    assert all(p["system"] == "TEST PROMPT\n\nTEST EXAMPLES\n" and "Required content" not in p["user"] for p in pairs)
    assert all(p["teacher_model"] == "stub" and p["published"] for p in pairs)
    make_sft.main(str(pairs_path), str(tmp_path / "train.jsonl"), data=data)
    assert sum(1 for _ in open(tmp_path / "train.jsonl")) == len(pairs)

    # a draft run of the same task can be run and checked, but not turned into training data
    draft = tmp_path / "drafts" / "draft1.md"
    draft.parent.mkdir()
    draft.write_text("DRAFT PROMPT\n")
    rd2 = tmp_path / "draft-run"
    monkeypatch.setattr(sys, "argv", ["run_model.py", str(sd / "inputs.csv"), str(rd2), "--model", "stub", "--openai-url", server,
                                      "--task", "abstract-facts", "--prompt", str(draft), "--workers", "2"])
    rm.main()
    assert json.loads((rd2 / "run.json").read_text())["published"] is False
    (rd2 / "check.txt").write_text(check)
    monkeypatch.setattr(sys, "argv", ["make_pairs.py", str(sd / "inputs.csv"), str(rd2), str(tmp_path / "draft-pairs.jsonl")])
    with pytest.raises(SystemExit, match="draft"):
        make_pairs.main()

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
