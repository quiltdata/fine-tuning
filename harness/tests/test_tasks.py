import copy
import json
from types import SimpleNamespace

import pytest

import make_pairs
import make_sft
import run_model as rm
import tasks

FILE_ROW = {"unit_id": "s3://b/p/a.txt", "prefix": "s3://b/p/", "unit": "a.txt", "file_type": "txt", "members": "1",
            "example_keys": '["a.txt"]', "size_bytes": "12", "excerpt_method": "full text", "excerpt": "Hello.", "mentions": "[]"}


def abstract_data(tmp_path, examples=False):
    """A throwaway data package with a test prompt (never the real, published one)."""
    (tmp_path / "task").mkdir(parents=True, exist_ok=True)
    (tmp_path / "task" / "abstract_facts.md").write_text("TEST PROMPT\n")
    if examples:
        (tmp_path / "task" / "abstract_examples.md").write_text("TEST EXAMPLES\n")
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


def test_abstract_task_needs_its_published_files(tmp_path, monkeypatch):
    data = abstract_data(tmp_path)
    with pytest.raises(SystemExit, match="abstract_examples.md"):
        rm.set_task("abstract-facts", data=data)
    rm.set_task("abstract-facts", data=data, allow_no_examples=True)
    (data / "task" / "abstract_facts.md").unlink()
    with pytest.raises(SystemExit, match="abstract_facts.md"):
        rm.set_task("abstract-facts", data=data, allow_no_examples=True)
    monkeypatch.delenv(tasks.DATA_ENV, raising=False)
    monkeypatch.setattr(tasks, "ROOT", tmp_path / "no-repo")  # no .env to read either
    with pytest.raises(SystemExit, match="FINE_TUNING_DATA is not set"):
        rm.set_task("abstract-facts", root=tmp_path / "no-repo", allow_no_examples=True)
    rm.set_task("file-summary")


def test_published_and_draft_fingerprints(tmp_path):
    data = abstract_data(tmp_path / "data", examples=True)
    task = rm.set_task("abstract-facts", data=data)
    assert rm.SYSTEM == "TEST PROMPT\n\nTEST EXAMPLES\n"
    fp = rm.fingerprint()
    assert fp["published"] and fp["prompt"] == "abstract_facts.md" and fp["examples"] == "abstract_examples.md"
    draft = tmp_path / "drafts" / "draft1.md"
    draft.parent.mkdir()
    draft.write_text("DRAFT PROMPT\n")
    rm.set_task("abstract-facts", prompt=str(draft), data=data)
    fp = rm.fingerprint()
    assert rm.SYSTEM.startswith("DRAFT PROMPT") and not fp["published"]
    assert fp["prompt"] == "draft1.md" and str(tmp_path) not in json.dumps(fp)  # names, never local paths
    rm.set_task("abstract-facts", data=data, allow_no_examples=True, examples=None)
    assert rm.fingerprint()["published"]  # examples exist, so they're used and it's still published
    (data / "task" / "abstract_examples.md").unlink()
    rm.set_task("abstract-facts", data=data, allow_no_examples=True)
    assert not rm.fingerprint()["published"]  # no examples: never the published task
    rm.set_task("file-summary")
    assert rm.fingerprint()["published"]


def test_abstract_one_per_call(tmp_path):
    task = tasks.get("abstract-facts", data=abstract_data(tmp_path))
    rows = [{"unit_id": f"PMC{i}.1", "abstract": f"Text {i}.", "title": "hidden", "protocol": "0"} for i in range(23)]
    assert [len(b) for _, b in task.batches(rows)] == [1] * 23 and task.batches(rows, size=1) == task.batches(rows)
    assert task.batches(rows)[1][0] == ("abstract-1", "abstract")
    with pytest.raises(ValueError):
        task.batches(rows, size=5)  # one abstract per call: no batching to fall back to
    assert task.user_message(rows[:1]) == "Text 0."  # the abstract alone: no id, no title, no number, no JSON
    teacher = task.user_message(rows[1:2], {"PMC1.1": ["A trial of X."]})
    assert teacher.startswith("Text 1.\n\nRequired content") and teacher.endswith('["A trial of X."]')
    api = task.api_schema()
    assert api["required"] == ["about", "findings"] and "unit_id" not in json.dumps(api) and "answers" not in json.dumps(api)
    assert api["properties"]["about"]["required"] == ["fact", "quote"] and "$ref" not in json.dumps(api)

def test_abstract_one_answer(tmp_path):
    task = tasks.get("abstract-facts", data=abstract_data(tmp_path))
    batch = [{"unit_id": "A.1", "abstract": "a"}]
    a1, a2 = {"about": {"fact": "one"}, "findings": []}, {"about": {"fact": "two"}, "findings": []}
    for reply in (a1, {"answers": [a1]}, [a1], {"answers": json.dumps([a1])}):
        assert task.attach_ids(task.rows_from_reply(reply), batch) == [{**a1, "unit_id": "A.1"}]
    assert task.attach_ids([{"abstract": 1, **a1}], batch) == [{**a1, "unit_id": "A.1"}]  # a stray number is dropped
    for bad in ([a1, a2], []):  # two answers for one abstract, or none: rejected, never guessed
        with pytest.raises(ValueError):
            task.attach_ids(bad, batch)
    assert task.assistant_json([{**a1, "unit_id": "A.1"}]) == a1

def test_missing_findings_stay_missing(tmp_path):
    import check_outputs
    task = tasks.get("abstract-facts", data=abstract_data(tmp_path))
    row = task.normalize({"unit_id": "A.1", "about": {"fact": "A trial.", "quote": ""}})
    assert "findings" not in row and check_outputs.abstract_problems(row)


def test_bedrock_tool_path_and_cache_point(tmp_path):
    rm.set_task("abstract-facts", data=abstract_data(tmp_path), allow_no_examples=True)
    rows = [{"unit_id": "A.1", "abstract": "a"}]
    answer = {"about": {"fact": "one", "quote": ""}, "findings": []}
    seen = []

    class Client:
        class exceptions:
            ValidationException = type("VE", (Exception,), {})
        def __init__(self, tool_input, stop="tool_use", cache_ok=True):
            self.tool_input, self.stop, self.cache_ok = tool_input, stop, cache_ok
        def converse(self, **req):
            seen.append(req["system"])
            assert req["toolConfig"]["toolChoice"] == {"tool": {"name": "record_rows"}}
            if not self.cache_ok and any("cachePoint" in b for b in req["system"]):
                raise self.exceptions.ValidationException("This model doesn't support caching")
            return {"stopReason": self.stop, "usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 1000},
                    "output": {"message": {"content": [{"toolUse": {"input": self.tool_input}}]}}}

    args = SimpleNamespace(model="us.anthropic.claude-haiku-5-5", region="r", max_tokens=100)
    for tool_input in (answer, {"answers": [answer]}, json.dumps(answer)):
        be = rm.Bedrock.__new__(rm.Bedrock)
        be.args, be.client, be.mode, be.max_tokens, be.cache = args, Client(tool_input), "tool", 100, True
        record, out = rm.call(be, args, 0, ("abstract-0", "abstract"), rows)
        assert record["error"] == "" and [o["unit_id"] for o in out] == ["A.1"] and out[0]["about"] == {"fact": "one"}
    assert seen[-1][-1] == {"cachePoint": {"type": "default"}}  # the shared system prompt is cached
    r = rm.RATES["bedrock"]["us.anthropic.claude-haiku-5-5"]
    assert record["cost_usd"] == round((10 * r["input"] + 5 * r["output"] + 1000 * r["cache_read"]) / 1e6, 6)
    be.client = Client(answer, cache_ok=False)  # a model without caching: the cache point is dropped once
    record, out = rm.call(be, args, 0, ("abstract-0", "abstract"), rows)
    assert record["error"] == "" and be.cache is False and seen[-1] == [seen[-1][0]]
    be.client = Client('{"about": {"fact": "on', stop="content_filtered")
    record, out = rm.call(be, args, 0, ("abstract-0", "abstract"), rows)
    assert out == [] and record["filtered"] == 2 and "content filter" in record["error"]  # both attempts filtered
    rm.set_task("file-summary")

def test_abstract_normalize_and_limits(tmp_path):
    task = tasks.get("abstract-facts", data=abstract_data(tmp_path))
    row = {"unit_id": "PMC1.1", "about": {"fact": "A trial.", "quote": ""}, "findings": [{"fact": "X fell.", "quote": "x fell"}]}
    assert task.normalize(row) == {"unit_id": "PMC1.1", "about": {"fact": "A trial."}, "findings": [{"fact": "X fell.", "quote": "x fell"}]}
    long = {"unit_id": "PMC1.1", "about": {"fact": "a" * 251}, "findings": [{"fact": "ok", "quote": "q" * 201}]}
    assert task.too_long([long]) == []  # 35c: no length retries


def test_run_model_call_offline(tmp_path):
    rm.set_task("abstract-facts", data=abstract_data(tmp_path), allow_no_examples=True)
    rows = [{"unit_id": "PMC1.1", "abstract": "A trial found X."}]
    reply = '```json\n{"about": {"fact": "A trial of X.", "quote": ""}, "findings": [{"fact": "X fell 10%.", "quote": "found X"}]}\n```'
    seen = {}

    class Fake:
        def call(self, batch, extra=""):
            seen["user"] = rm.user_message(batch)
            return rm.parse_rows(reply), {"model_served": "fake", "mode": "json_schema", "input_tokens": 5, "output_tokens": 7}

    record, out = rm.call(Fake(), SimpleNamespace(model="fake"), 0, ("abstract-0", "abstract"), rows)
    assert record["error"] == "" and record["rows_out"] == 1 and json.loads(record["unit_ids"]) == ["PMC1.1"]
    assert seen["user"] == "A trial found X."
    assert out[0]["findings"] == [{"fact": "X fell 10%.", "quote": "found X"}] and out[0]["about"] == {"fact": "A trial of X."}
    rm.set_task("file-summary")

def test_make_pairs_mixed_batches_and_run_check(tmp_path):
    rows = [{"unit_id": str(i)} for i in range(57)]
    groups = make_pairs.mixed(rows, 3, 10)
    assert sum(len(b) for _, b in groups) == 57 and all(1 <= len(b) <= 10 for _, b in groups)
    assert make_pairs.mixed(rows, 3, 10) == groups and make_pairs.mixed(rows, 4, 10) != groups
    data = abstract_data(tmp_path, examples=True)
    good = tasks.get("abstract-facts", data=data).fingerprint()
    assert good["published"]
    make_pairs.set_task_from_run(good, data=data)
    with pytest.raises(SystemExit, match="prompt_sha256"):
        make_pairs.set_task_from_run({**good, "prompt_sha256": "0" * 64}, data=data)
    with pytest.raises(SystemExit, match="draft"):
        make_pairs.set_task_from_run({**good, "published": False}, data=data)
    rm.set_task("file-summary")


def test_make_sft_refuses_closed_teachers_and_unpublished_pairs(tmp_path):
    data = abstract_data(tmp_path / "data", examples=True)
    task = tasks.get("abstract-facts", data=data)
    fp = task.fingerprint()
    good = {"task": "abstract-facts", "prompt_sha256": fp["prompt_sha256"], "examples_sha256": fp["examples_sha256"],
            "published": True, "teacher_model": "us.moonshotai.kimi-k3", "rows": 1, "rows_passing_check": 1,
            "system": task.system(), "user": "u", "assistant": "a"}
    def run(pairs):
        path = tmp_path / "pairs.jsonl"
        path.write_text("".join(json.dumps(p) + "\n" for p in pairs))
        make_sft.main(str(path), str(tmp_path / "train.jsonl"), data=data)
        return [json.loads(l) for l in open(tmp_path / "train.jsonl")]
    assert len(run([good, {**good, "teacher_model": "openai.gpt-oss-120b-1:0"}])) == 2
    for bad, why in (({**good, "teacher_model": "us.anthropic.claude-opus-5-5"}, "Claude"),
                     ({**good, "teacher_model": "us.openai.gpt-6.1-sol"}, "closed OpenAI"),
                     ({**good, "published": False}, "published prompt"),
                     ({**good, "system": "DRAFT PROMPT"}, "published prompt"),
                     ({**good, "prompt_sha256": "0" * 64}, "published prompt")):
        with pytest.raises(SystemExit, match=why):
            run([good, bad])
