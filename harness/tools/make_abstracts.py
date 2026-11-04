#!/usr/bin/env python3
"""Build the abstract task's input table from an article list (sample_pmc.py --articles-only).

For each article, reads its JATS XML anonymously from s3://pmc-oa-opendata and takes the main
abstract by the rules in abstracts.py: the counted text, its size in UTF-8 bytes, and, past
4,096 bytes, a truncated copy cut at a whole character and marked. Adds the PubMed publication
types and the study-protocol flag. Articles with no main abstract are left out and listed in
skipped.csv beside the output.

Columns follow schemas/abstract_input.schema.json; list and object cells hold JSON. The model
sees only unit_id and abstract; the other columns are for selection, analysis and reports.

Usage: python3 make_abstracts.py articles.csv inputs.csv [--workers 32] [--no-pubmed]
"""
import argparse
import csv
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from abstracts import article_fields, main_abstract, protocol_flags, truncate_utf8
from pubmed import pmids_for, publication_types

BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com"
COLUMNS = ["unit_id", "pmcid", "pmid", "title", "journal", "jats_article_type", "pubmed_types", "structured", "headings",
           "abstract_bytes", "truncated", "abstract", "protocol", "protocol_signals"]


def fetch_xml(article_id):
    url = f"{BUCKET_URL}/{article_id}/{article_id}.xml"
    head = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-262143"}), timeout=60).read()
    text = head.decode("utf-8", "replace")
    if "</article-meta>" in text:
        return text
    return urllib.request.urlopen(url, timeout=120).read().decode("utf-8", "replace")


def build(article, pub_types):
    unit_id = f'{article["pmcid"]}.{article["version"]}'
    try:
        xml = fetch_xml(unit_id)
    except Exception as e:
        return None, {"unit_id": unit_id, "reason": f"{type(e).__name__}: {e}"[:200]}
    a = main_abstract(xml)
    if a is None:
        return None, {"unit_id": unit_id, "reason": "no main abstract"}
    fields = article_fields(xml)
    title = article.get("title") or fields["title"]
    types = pub_types.get(article.get("pmid", ""), [])
    flags = protocol_flags(types, title, a["headings"])
    text, truncated = truncate_utf8(a["text"])
    return {
        "unit_id": unit_id, "pmcid": article["pmcid"], "pmid": article.get("pmid", ""), "title": title,
        "journal": fields["journal"], "jats_article_type": fields["jats_article_type"], "pubmed_types": json.dumps(types),
        "structured": int(a["structured"]), "headings": json.dumps(a["headings"], ensure_ascii=False),
        "abstract_bytes": a["bytes"], "truncated": int(truncated), "abstract": text,
        "protocol": int(flags["protocol"]), "protocol_signals": json.dumps({k: flags[k] for k in ("pubmed", "title", "headings")}),
    }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("articles")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-pubmed", action="store_true", help="skip publication types (the protocol flag then uses title and headings only)")
    args = ap.parse_args()

    articles = list(csv.DictReader(open(args.articles, newline="")))
    pub_types = {}
    if not args.no_pubmed:
        missing = [a["pmcid"] for a in articles if not a.get("pmid")]
        found = pmids_for(missing) if missing else {}
        for a in articles:
            a["pmid"] = a.get("pmid") or found.get(a["pmcid"], "")
        pub_types = publication_types(a["pmid"] for a in articles)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda a: build(a, pub_types), articles))

    rows = [r for r, _ in results if r]
    skipped = [s for _, s in results if s]
    out = Path(args.out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(out.with_name("skipped.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit_id", "reason"])
        w.writeheader()
        w.writerows(skipped)
    print(json.dumps({"articles": len(articles), "inputs": len(rows), "skipped": len(skipped),
                      "truncated": sum(r["truncated"] for r in rows), "protocols": sum(r["protocol"] for r in rows)}), file=sys.stderr)


if __name__ == "__main__":
    main()
