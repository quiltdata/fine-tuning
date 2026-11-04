import json

import make_abstracts

XML = """<article article-type="research-article"><front><journal-meta><journal-title>Trials</journal-title></journal-meta>
<article-meta><title-group><article-title>JATS title</article-title></title-group>
<abstract><sec><title>Introduction</title><p>Why.</p></sec><sec><title>Methods and analysis</title><p>How it will run.</p></sec>
</abstract></article-meta></front></article>"""


def test_build_row(monkeypatch):
    monkeypatch.setattr(make_abstracts, "fetch_xml", lambda article_id: XML)
    article = {"pmcid": "PMC1", "version": "2", "title": "A trial: protocol for a randomised study", "pmid": "9"}
    row, skipped = make_abstracts.build(article, {"9": ["Journal Article", "Clinical Trial Protocol"]})
    assert skipped is None
    assert row["unit_id"] == "PMC1.2" and row["journal"] == "Trials" and row["title"] == article["title"]
    assert row["abstract"] == "Introduction. Why. Methods and analysis. How it will run."
    assert row["truncated"] == 0 and row["abstract_bytes"] == len(row["abstract"].encode())
    assert row["protocol"] == 1 and json.loads(row["protocol_signals"]) == {"pubmed": True, "title": True, "headings": True}
    assert json.loads(row["headings"]) == ["Introduction", "Methods and analysis"]


def test_build_skips_missing_abstract(monkeypatch):
    monkeypatch.setattr(make_abstracts, "fetch_xml", lambda article_id: "<article-meta></article-meta>")
    row, skipped = make_abstracts.build({"pmcid": "PMC2", "version": "1"}, {})
    assert row is None and skipped == {"unit_id": "PMC2.1", "reason": "no main abstract"}


def test_build_truncates_long_abstract(monkeypatch):
    long_xml = "<article-meta><abstract><p>" + ("Sentence of words. " * 300) + "</p></abstract></article-meta>"
    monkeypatch.setattr(make_abstracts, "fetch_xml", lambda article_id: long_xml)
    row, _ = make_abstracts.build({"pmcid": "PMC3", "version": "1"}, {})
    assert row["truncated"] == 1 and row["abstract_bytes"] > 4096
    assert len(row["abstract"].encode()) <= 4096 and row["abstract"].endswith(".")
