"""Regression tests for the review of the abstract-task harness."""
import csv
import json
import sys
from types import SimpleNamespace

import pytest

import check_outputs
import make_pairs
import run_model as rm
import score_abstracts


def test_length_never_invalidates_a_row():  # 35c: cost counts verbosity, validity doesn't
    row = {"unit_id": "A.1", "about": {"fact": "A trial.", "quote": "q" * 250}, "findings": []}
    assert check_outputs.abstract_problems(row) == []
    assert check_outputs.abstract_problems({**row, "about": {"fact": "f" * 600}}) == []
    three = {**row, "findings": [{"fact": "a"}, {"fact": "b"}, {"fact": "c"}]}
    assert check_outputs.abstract_problems(three)  # structure still counts


def test_verdicts_are_case_insensitive_and_dropped_keys_are_not_no_results(monkeypatch):
    def grader(model, region, system, messages, name, schema, max_tokens=4000):
        if system == score_abstracts.RECALL_SYSTEM:
            return {"verdicts": [{"id": "about", "conveyed": True}]}, {"cost_usd": 0.0, "error": ""}
        return {"verdicts": [{"id": "about", "verdict": "Supported"}, {"id": "f1", "verdict": " SUPPORTED "}],
                "redundant": False}, {"cost_usd": 0.0, "error": ""}

    monkeypatch.setattr(score_abstracts, "call_tool", grader)
    args = SimpleNamespace(grader="g", region="r", previous={})
    out = {"unit_id": "A.1", "about": json.dumps({"fact": "A trial."}), "findings": json.dumps([{"fact": "X fell."}])}
    dropped_key = {"unit_id": "A.1", "key_facts": [{"id": "about", "role": "about", "fact": "A trial.", "quote": "q"}],
                   "dropped": ["a finding quote the key-writer got wrong"]}
    row, grades = score_abstracts.score_row(dropped_key, {"abstract": "A trial. X fell.", "protocol": "0"}, out, 0.0, args)
    assert row["supported"] == 2 and row["key_dropped"] == 1 and not grades["reused"]
    totals = score_abstracts.totals([row])
    assert totals["faithfulness"] == 100.0
    assert totals["invented_findings"] == 0 and totals["rows_key_without_findings"] == 0


def test_only_refuses_inputs_without_article_folders(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs.csv"
    inputs.write_text("unit_id,abstract\nA.1,Text.\n")
    monkeypatch.setattr(sys, "argv", ["run_model.py", str(inputs), str(tmp_path / "out"), "--model", "m",
                                      "--split", "x.csv", "--only", "test"])
    with pytest.raises(SystemExit):
        rm.main()


def test_make_pairs_keeps_per_file_batches_and_skips_failed_rows_in_mixed(tmp_path, monkeypatch):
    rows = [{"unit_id": f"s3://b/p/{i}.txt", "prefix": "s3://b/p/", "unit": f"{i}.txt", "file_type": "txt", "members": "1",
             "example_keys": "[]", "size_bytes": "1", "excerpt_method": "x", "excerpt": "x", "mentions": "[]"} for i in range(4)]
    run = tmp_path / "run"
    run.mkdir()
    inputs = tmp_path / "inputs.csv"
    with open(inputs, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(run / "outputs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rm.OUT_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({"unit_id": r["unit_id"], "what": "other", "summary": "s", "about": "[]", "entities": "[]",
                        "refers_to": "[]", "evidence": '["x"]'})
    with open(run / "calls.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["batch", "unit_ids", "model_served"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"batch": i, "unit_ids": json.dumps([r["unit_id"]]), "model_served": "teacher"})
    (run / "check.txt").write_text(f"PASS  {rows[0]['unit_id']}\nFAIL  {rows[1]['unit_id']}\n")
    (run / "run.json").write_text(json.dumps({"per_file": True}))
    monkeypatch.setattr(sys, "argv", ["make_pairs.py", str(inputs), str(run), str(tmp_path / "p.jsonl")])
    make_pairs.main()
    pairs = [json.loads(l) for l in open(tmp_path / "p.jsonl")]
    assert [p["rows"] for p in pairs] == [1, 1, 1, 1]  # one per call, as the run made them
    monkeypatch.setattr(sys, "argv", ["make_pairs.py", str(inputs), str(run), str(tmp_path / "m.jsonl"), "--mixed-batches", "1"])
    make_pairs.main()
    mixed = [json.loads(l) for l in open(tmp_path / "m.jsonl")]
    assert sum(p["rows"] for p in mixed) == 3 and all(p["rows"] == p["rows_passing_check"] for p in mixed)
