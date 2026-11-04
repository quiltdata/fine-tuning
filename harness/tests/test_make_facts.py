from types import SimpleNamespace

import make_facts

ABSTRACT = ("Background. Diets matter. Methods. We randomized 412 adults with type 2 diabetes. "
            "Results. The low-carbohydrate diet lowered HbA1c by 0.6 percentage points more than usual care. "
            "Weight loss did not differ between groups.")


def test_abstract_key_keeps_found_quotes_and_drops_extra_findings():
    out = {"about": {"fact": "An RCT in 412 adults with T2D.", "quote": "We randomized 412 adults with type 2 diabetes."},
           "findings": [{"fact": "HbA1c fell 0.6 points more.", "quote": "lowered HbA1c by 0.6 percentage points"},
                        {"fact": "Weight loss was similar.", "quote": "Weight loss did not differ"},
                        {"fact": "A third.", "quote": "Diets matter."}]}
    kept, failed, extra = make_facts.abstract_key(out, ABSTRACT)
    assert [k["id"] for k in kept] == ["about", "f1", "f2"] and [k["role"] for k in kept] == ["about", "finding", "finding"]
    assert failed == [] and extra == 1


def test_abstract_key_reports_quotes_not_in_the_abstract():
    out = {"about": {"fact": "An RCT.", "quote": "a trial of 500 children"}, "findings": []}
    kept, failed, extra = make_facts.abstract_key(out, ABSTRACT)
    assert kept == [] and failed == ["a trial of 500 children"] and extra == 0


def test_build_abstract_retries_once_with_failed_quotes(monkeypatch):
    replies = [{"about": {"fact": "An RCT.", "quote": "made-up quote"}, "findings": []},
               {"about": {"fact": "An RCT.", "quote": "We randomized 412 adults"}, "findings": []}]
    messages = []

    def fake_tool(model, region, system, msgs, name, schema, max_tokens=4000):
        messages.append(msgs[0]["content"][0]["text"])
        return replies[len(messages) - 1], {"model": model, "error": ""}

    monkeypatch.setattr(make_facts, "call_tool", fake_tool)
    row = {"unit_id": "PMC1.1", "abstract": ABSTRACT, "protocol": "0"}
    result, records = make_facts.build_abstract(row, SimpleNamespace(chat=False, model="m", region="r"))
    assert len(records) == 2 and "made-up quote" in messages[1]
    assert [k["id"] for k in result["key_facts"]] == ["about"] and result["dropped"] == []
    assert result["file_type"] == "abstract" and result["code_facts"] == []
