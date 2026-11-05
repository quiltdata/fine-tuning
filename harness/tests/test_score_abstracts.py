import csv
import json
import sys

import pytest

import check_outputs
import score_abstracts

KEYS = [
    {"unit_id": "A.1", "key_facts": [{"id": "about", "role": "about", "fact": "An RCT of a diet.", "quote": "q"},
                                     {"id": "f1", "role": "finding", "fact": "HbA1c fell.", "quote": "q"},
                                     {"id": "f2", "role": "finding", "fact": "Weight did not differ.", "quote": "q"}]},
    {"unit_id": "P.1", "key_facts": [{"id": "about", "role": "about", "fact": "A trial protocol.", "quote": "q"}]},
    {"unit_id": "M.1", "key_facts": [{"id": "about", "role": "about", "fact": "A cohort.", "quote": "q"},
                                     {"id": "f1", "role": "finding", "fact": "X rose.", "quote": "q"}]},
    {"unit_id": "I.1", "key_facts": [{"id": "about", "role": "about", "fact": "A survey.", "quote": "q"},
                                     {"id": "f1", "role": "finding", "fact": "Y fell.", "quote": "q"}]},
]
INPUTS = [{"unit_id": "A.1", "abstract": "We randomized adults. HbA1c fell.", "protocol": "0"},
          {"unit_id": "P.1", "abstract": "We will randomize adults.", "protocol": "1"},
          {"unit_id": "M.1", "abstract": "In a cohort, X rose.", "protocol": "0"},
          {"unit_id": "I.1", "abstract": "A survey found Y fell.", "protocol": "0"}]
OUTPUTS = [{"unit_id": "A.1", "about": {"fact": "A diet RCT."}, "findings": [{"fact": "HbA1c fell", "quote": "HbA1c fell"}, {"fact": "Weight fell a lot"}]},
           {"unit_id": "P.1", "about": {"fact": "A planned trial."}, "findings": [{"fact": "The diet worked."}]},
           {"unit_id": "I.1", "about": {"fact": "S" * 300}, "findings": [{"fact": "Y fell.", "quote": "not in it"}]}]


def write_set(tmp_path):
    sd, rd = tmp_path / "set", tmp_path / "run"
    (sd / "reference").mkdir(parents=True)
    rd.mkdir()
    with open(sd / "inputs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit_id", "abstract", "protocol"])
        w.writeheader()
        w.writerows(INPUTS)
    (sd / "reference" / "facts.jsonl").write_text("".join(json.dumps(k) + "\n" for k in KEYS))
    with open(rd / "outputs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit_id", "about", "findings"])
        w.writeheader()
        for o in OUTPUTS:
            w.writerow({"unit_id": o["unit_id"], "about": json.dumps(o["about"]), "findings": json.dumps(o["findings"])})
    with open(rd / "calls.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["batch", "unit_ids", "cost_usd"])
        w.writeheader()
        w.writerow({"batch": 0, "unit_ids": json.dumps(["A.1", "P.1", "M.1", "I.1"]), "cost_usd": "0.04"})
    (rd / "run.json").write_text(json.dumps({"model": "m", "task": "abstract-facts"}))
    return sd, rd


def fake_grader(model, region, system, messages, name, schema, max_tokens=4000):
    text = messages[0]["content"][0]["text"]
    if system == score_abstracts.RECALL_SYSTEM:
        conveyed = {"An RCT": {"about": True, "f1": True, "f2": False}, "A trial protocol": {"about": True},
                    "A survey": {"about": False, "f1": True}}
        for marker, v in conveyed.items():
            if marker in text:
                return {"verdicts": [{"id": k, "conveyed": c} for k, c in v.items()]}, {"cost_usd": 0.001, "error": ""}
        raise AssertionError("recall grader called for an unexpected row")
    if "HbA1c" in text:
        res = {"verdicts": [{"id": "about", "verdict": "supported"}, {"id": "f1", "verdict": "supported"},
                            {"id": "f2", "verdict": "overstated"}], "redundant": False}
    elif "will randomize" in text:
        res = {"verdicts": [{"id": "about", "verdict": "supported"}, {"id": "f1", "verdict": "unsupported"}], "redundant": False}
    else:
        res = {"verdicts": [{"id": "about", "verdict": "supported"}, {"id": "f1", "verdict": "supported"}], "redundant": False}
    return res, {"cost_usd": 0.002, "error": ""}


def test_score_abstracts_measures(tmp_path, monkeypatch, capsys):
    sd, rd = write_set(tmp_path)
    monkeypatch.setattr(score_abstracts, "call_tool", fake_grader)
    monkeypatch.setattr(sys, "argv", ["score_abstracts.py", str(sd), str(rd)])
    score_abstracts.main()
    summary = json.loads((rd / "score.json").read_text())
    a = summary["all"]
    assert a["rows"] == 4 and a["valid"] == 3 and a["validity"] == 75.0   # a 300-character fact is still valid (35c)
    assert a["about_recall"] == 50.0            # A and P conveyed; M missing; I graded, not conveyed
    assert a["findings_recall"] == 50.0         # 2 of 4 key findings (A: 1 of 2, M: 0 of 1, I: 1 of 1; P has none)
    assert a["max_fact_chars"] == 300
    assert a["facts_stated"] == 7 and a["faithfulness"] == round(100 * 5 / 7, 1)
    assert a["overstated"] == round(100 / 7, 1) and a["unsupported"] == round(100 / 7, 1)
    assert a["invented_findings"] == 1 and a["rows_key_without_findings"] == 1
    assert a["rows_with_two_findings"] == 1 and a["redundancy"] == 0.0
    assert a["quotes_found"] == 50.0 and a["usd_per_100_rows"] == 1.0
    assert summary["protocol"]["rows"] == 1 and summary["not_protocol"]["rows"] == 3
    rows = {r["unit_id"]: r for r in csv.DictReader(open(rd / "scores.csv", newline=""))}
    assert rows["M.1"]["problems"] == "no output row" and rows["I.1"]["valid"] == "1" and rows["A.1"]["missed"] == "f2"
    assert summary["grader_calls"] == 6        # recall and faithfulness for A, P and I; none for the missing M


def test_reuse_grades_makes_no_calls(tmp_path, monkeypatch):
    sd, rd = write_set(tmp_path)
    monkeypatch.setattr(score_abstracts, "call_tool", fake_grader)
    monkeypatch.setattr(sys, "argv", ["score_abstracts.py", str(sd), str(rd)])
    score_abstracts.main()
    first = json.loads((rd / "score.json").read_text())["all"]

    def no_calls(*a, **k):
        raise AssertionError("grader called despite --reuse-grades")

    monkeypatch.setattr(score_abstracts, "call_tool", no_calls)
    monkeypatch.setattr(sys, "argv", ["score_abstracts.py", str(sd), str(rd), "--reuse-grades"])
    score_abstracts.main()
    assert json.loads((rd / "score.json").read_text())["all"] == first


def test_check_outputs_abstract(tmp_path, capsys):
    sd, rd = write_set(tmp_path)
    with pytest.raises(SystemExit) as e:
        check_outputs.main_abstract(str(sd / "inputs.csv"), str(rd / "outputs.csv"))
    out = capsys.readouterr().out
    assert e.value.code == 1
    assert "PASS  A.1" in out and "PASS  P.1" in out and "PASS  I.1" in out and "FAIL  M.1" in out
    assert "quote not found (not scored)" in out and "3/4 input rows pass" in out and "1/2 quotes found" in out
    assert "longest fact: 300 characters" in out
