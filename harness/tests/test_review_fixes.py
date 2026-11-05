"""Regression tests for the review of the abstract-task harness."""
import csv
import functools
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


def test_content_filtered_reply_is_a_failed_call_not_an_answer(tmp_path):
    from test_tasks import abstract_data
    rm.set_task("abstract-facts", data=abstract_data(tmp_path), allow_no_examples=True)
    calls = []

    class Filtered:
        def call(self, batch, extra=""):
            calls.append(1)
            fragment = [{"about": '{"fact": "This is a lab st', "findings": []}]
            return fragment, {"model_served": "m", "stop_reason": "content_filtered", "input_tokens": 900, "output_tokens": 40}

    record, out = rm.call(Filtered(), SimpleNamespace(model="m"), 0, ("abstract-0", "abstract"), [{"unit_id": "A.1", "abstract": "x"}])
    assert out == [] and record["rows_out"] == 0 and "content filter" in record["error"]
    assert len(calls) == 2 and record["filtered"] == 2 and record["input_tokens"] == 1800  # retried once; tokens still billed
    rm.set_task("file-summary")


def test_string_encoded_facts_are_decoded(tmp_path):
    from test_tasks import abstract_data
    import tasks
    task = tasks.get("abstract-facts", data=abstract_data(tmp_path))
    row = {"unit_id": "A.1", "about": '{"fact": "A trial.", "quote": ""}', "findings": '[{"fact": "X fell.", "quote": "x"}]'}
    assert task.normalize(row) == {"unit_id": "A.1", "about": {"fact": "A trial."}, "findings": [{"fact": "X fell.", "quote": "x"}]}
    broken = {"unit_id": "A.1", "about": '{"fact": "cut off', "findings": []}
    assert task.normalize(broken)["about"] == '{"fact": "cut off'  # left as is, so the check fails it


def test_filtered_batch_is_split_so_only_the_flagged_abstract_is_lost(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from test_tasks import abstract_data
    rm.set_task("abstract-facts", data=abstract_data(tmp_path), allow_no_examples=True)
    rows = [{"unit_id": u, "abstract": f"text {u}"} for u in ("A.1", "BAD.1", "C.1")]

    class Filter:  # blocks any call that includes the BAD abstract, answers the rest
        def call(self, batch, extra=""):
            rec = {"model_served": "m", "input_tokens": 100, "output_tokens": 0 if any("BAD" in r["unit_id"] for r in batch) else 50}
            if any("BAD" in r["unit_id"] for r in batch):
                return [], {**rec, "stop_reason": "content_filtered"}
            answers = [{"abstract": i, "about": {"fact": "ok"}, "findings": []} for i in range(1, len(batch) + 1)]
            return answers, {**rec, "stop_reason": "tool_use"}

    args = SimpleNamespace(model="m")
    work = rm.batches(rows, size=3)
    first = [rm.call(Filter(), args, i, *w) for i, w in enumerate(work)]
    assert first[0][1] == [] and first[0][0]["filtered"] == 2  # blocked, retried once, blocked again
    with ThreadPoolExecutor(2) as pool:
        extra = rm.split_filtered(Filter(), args, first, work, pool)
    got = sorted(o["unit_id"] for _, outs in extra for o in outs)
    assert got == ["A.1", "C.1"] and all(rec["split_from"] == 0 for rec, _ in extra)
    lost = [rec for rec, outs in extra if not outs]
    assert len(lost) == 1 and lost[0]["filtered"] == 2 and json.loads(lost[0]["unit_ids"]) == ["BAD.1"]
    rm.set_task("file-summary")


def test_make_pairs_keeps_the_answered_abstracts_of_a_batch(tmp_path, monkeypatch):
    from test_tasks import abstract_data
    data = abstract_data(tmp_path / "data", examples=True)
    import tasks
    fp = tasks.get("abstract-facts", data=data).fingerprint()
    run = tmp_path / "run"
    run.mkdir()
    rows = [{"unit_id": f"P{i}.1", "abstract": f"text {i}"} for i in range(5)]
    inputs = tmp_path / "inputs.csv"
    with open(inputs, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit_id", "abstract"])
        w.writeheader()
        w.writerows(rows)
    with open(run / "outputs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit_id", "about", "findings"])
        w.writeheader()
        for r in rows[:2] + rows[3:]:  # P2 got no answer (say, the filter blocked it)
            w.writerow({"unit_id": r["unit_id"], "about": json.dumps({"fact": "f"}), "findings": "[]"})
    with open(run / "calls.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["batch", "unit_ids", "model_served"])
        w.writeheader()
        w.writerow({"batch": 0, "unit_ids": json.dumps([r["unit_id"] for r in rows]), "model_served": "teacher"})
    (run / "check.txt").write_text("")
    (run / "run.json").write_text(json.dumps({**fp, "task": "abstract-facts"}))
    monkeypatch.setattr(rm, "set_task", functools.partial(rm.set_task, data=data))
    monkeypatch.setattr(sys, "argv", ["make_pairs.py", str(inputs), str(run), str(tmp_path / "p.jsonl")])
    make_pairs.main()
    (pair,) = [json.loads(l) for l in open(tmp_path / "p.jsonl")]
    assert pair["rows"] == 4 and "text 2" not in pair["user"] and pair["user"].count("Abstract ") == 4
    assert [a["abstract"] for a in json.loads(pair["assistant"])["answers"]] == [1, 2, 3, 4]
    rm.set_task.func("file-summary")
