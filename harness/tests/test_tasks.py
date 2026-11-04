import copy
import json
import shutil
from types import SimpleNamespace

import pytest

import make_pairs
import run_model as rm
import tasks

FILE_ROW = {"unit_id": "s3://b/p/a.txt", "prefix": "s3://b/p/", "unit": "a.txt", "file_type": "txt", "members": "1",
            "example_keys": '["a.txt"]', "size_bytes": "12", "excerpt_method": "full text", "excerpt": "Hello.", "mentions": "[]"}


def abstract_root(tmp_path, examples=False):
    """A throwaway harness root with a test prompt (never the real, human-written one)."""
    (tmp_path / "prompts").mkdir()
    (tmp_path / "schemas").mkdir()
    shutil.copy(tasks.ROOT / "schemas" / "abstract_output.schema.json", tmp_path / "schemas")
    (tmp_path / "prompts" / "abstract_facts.md").write_text("TEST PROMPT\n")
    if examples:
        (tmp_path / "prompts" / "abstract_examples.md").write_text("TEST EXAMPLES\n")
    return tmp_path


def test_file_summary_matches_the_earlier_code():
    rm.set_task("file-summary")
    old_table = [{**FILE_ROW, "members": 1, "size_bytes": 12, "example_keys": ["a.txt"], "mentions": []}]
    assert rm.user_message([FILE_ROW]) == "Input table, one JSON object per row:\n" + json.dumps(old_table, ensure_ascii=False)
    old_schema = json.loads((tasks.ROOT / "schemas" / "output.schema.json").read_text())
    assert rm.API_SCHEMA["properties"]["rows"]["items"] == tasks.strip(copy.deepcopy(old_schema))
    assert rm.SYSTEM == (tasks.ROOT / "prompts" / "summarizer.md").read_text()
    rows = [FILE_ROW, {**FILE_ROW, "unit_id": "s3://b/p/b.txt"}, {**FILE_ROW, "unit_id": "s3://b/q/c.txt", "prefix": "s3://b/q/"}]
    assert [len(b) for _, b in rm.batches(rows)] == [2, 1]


def test_abstract_task_needs_its_human_written_files(tmp_path):
    root = abstract_root(tmp_path)
    with pytest.raises(SystemExit, match="abstract_examples.md"):
        rm.set_task("abstract-facts", root=root)
    rm.set_task("abstract-facts", root=root, allow_no_examples=True)
    (root / "prompts" / "abstract_facts.md").unlink()
    with pytest.raises(SystemExit, match="abstract_facts.md"):
        rm.set_task("abstract-facts", root=root, allow_no_examples=True)
    rm.set_task("file-summary")


def test_abstract_task_prompt_examples_and_fingerprint(tmp_path):
    root = abstract_root(tmp_path, examples=True)
    task = rm.set_task("abstract-facts", root=root)
    assert rm.SYSTEM == "TEST PROMPT\n\nTEST EXAMPLES\n"
    before = task.fingerprint()
    (root / "prompts" / "abstract_examples.md").write_text("CHANGED\n")
    assert task.fingerprint()["examples_sha256"] != before["examples_sha256"]
    rm.set_task("file-summary")


def test_abstract_batches_view_and_schema(tmp_path):
    task = tasks.get("abstract-facts", abstract_root(tmp_path))
    rows = [{"unit_id": f"PMC{i}.1", "abstract": f"Text {i}.", "title": "hidden", "protocol": "0"} for i in range(23)]
    assert [len(b) for _, b in task.batches(rows)] == [10, 10, 3]
    assert [len(b) for _, b in task.batches(rows, size=7)] == [7, 7, 7, 2]
    assert len(task.batches(rows, per_file=True)) == 23
    assert task.batches(rows)[1][0] == ("batch-1", "abstract")
    assert task.model_view(rows[0]) == {"unit_id": "PMC0.1", "abstract": "Text 0."}
    api = task.api_schema()
    fact = api["properties"]["rows"]["items"]["properties"]["about"]
    assert fact["required"] == ["fact", "quote"] and fact["additionalProperties"] is False
    assert "$ref" not in json.dumps(api)


def test_abstract_normalize_and_limits(tmp_path):
    task = tasks.get("abstract-facts", abstract_root(tmp_path))
    row = {"unit_id": "PMC1.1", "about": {"fact": "A trial.", "quote": ""}, "findings": [{"fact": "X fell.", "quote": "x fell"}]}
    assert task.normalize(row) == {"unit_id": "PMC1.1", "about": {"fact": "A trial."}, "findings": [{"fact": "X fell.", "quote": "x fell"}]}
    long = {"unit_id": "PMC1.1", "about": {"fact": "a" * 251}, "findings": [{"fact": "ok", "quote": "q" * 201}]}
    problems = task.too_long([long])
    assert len(problems) == 2 and "251 characters" in problems[0] and "201 characters" in problems[1]


def test_run_model_call_offline(tmp_path):
    rm.set_task("abstract-facts", root=abstract_root(tmp_path), allow_no_examples=True)
    rows = [{"unit_id": "PMC1.1", "abstract": "A trial found X."}, {"unit_id": "PMC2.1", "abstract": "A protocol."}]
    reply = [{"unit_id": "PMC1.1", "about": {"fact": "A trial of X.", "quote": ""}, "findings": [{"fact": "X fell 10%.", "quote": "found X"}]},
             {"unit_id": "PMC2.1", "about": {"fact": "A protocol.", "quote": ""}, "findings": []},
             {"unit_id": "PMC1.1", "about": {"fact": "repeat"}, "findings": []}]
    seen = {}

    class Fake:
        def call(self, batch, extra=""):
            seen["user"] = rm.user_message(batch)
            return copy.deepcopy(reply), {"model_served": "fake", "mode": "json_schema", "input_tokens": 5, "output_tokens": 7}

    record, out = rm.call(Fake(), SimpleNamespace(model="fake"), 0, ("batch-0", "abstract"), rows)
    assert record["error"] == "" and record["rows_out"] == 2 and record["dropped_rows"] == 1
    assert json.loads(record["unit_ids"]) == ["PMC1.1", "PMC2.1"]
    assert out[0]["about"] == {"fact": "A trial of X."} and out[1]["findings"] == []
    assert json.loads(seen["user"].split("\n", 1)[1]) == rows
    rm.set_task("file-summary")


def test_make_pairs_mixed_batches_and_run_check(tmp_path):
    rows = [{"unit_id": str(i)} for i in range(57)]
    groups = make_pairs.mixed(rows, 3, 10)
    assert sum(len(b) for _, b in groups) == 57 and all(1 <= len(b) <= 10 for _, b in groups)
    assert make_pairs.mixed(rows, 3, 10) == groups and make_pairs.mixed(rows, 4, 10) != groups
    root = abstract_root(tmp_path, examples=True)
    good = tasks.get("abstract-facts", root).fingerprint()
    make_pairs.set_task_from_run({**good, "task": "abstract-facts"}, root=root)
    with pytest.raises(SystemExit, match="prompt_sha256"):
        make_pairs.set_task_from_run({**good, "task": "abstract-facts", "prompt_sha256": "0" * 64}, root=root)
    rm.set_task("file-summary")
