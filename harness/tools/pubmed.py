"""PMCID to PMID, and PubMed publication types, through NCBI's public APIs (200 ids per request).

Publication types are assigned by NLM, so they are consistent across publishers (unlike the JATS
article-type); most articles carry only "Journal Article".
"""
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

TOOL = "quiltdata-fine-tuning"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH = 200


def _get(url, data=None):
    for wait in (1, 3, 10, 30, None):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=120).read()
        except Exception:
            if wait is None:
                raise
            time.sleep(wait)


def pmids_for(pmcids):
    """{pmcid: pmid} for the PMCIDs PubMed knows."""
    out = {}
    pmcids = list(pmcids)
    for i in range(0, len(pmcids), BATCH):
        q = urllib.parse.urlencode({"ids": ",".join(pmcids[i:i + BATCH]), "format": "json", "tool": TOOL})
        for rec in json.loads(_get(f"{IDCONV}?{q}")).get("records", []):
            if rec.get("pmid"):
                out[rec["pmcid"]] = str(rec["pmid"])
        time.sleep(0.4)  # NCBI asks for at most 3 requests a second without a key
    return out


def publication_types(pmids):
    """{pmid: [publication type, ...]}."""
    out = {}
    pmids = [str(p) for p in pmids if p]
    for i in range(0, len(pmids), BATCH):
        body = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids[i:i + BATCH]), "retmode": "xml", "tool": TOOL}).encode()
        root = ET.fromstring(_get(EFETCH, body))
        for art in root.iter("PubmedArticle"):
            out[art.findtext(".//MedlineCitation/PMID")] = [t.text for t in art.iter("PublicationType")]
        time.sleep(0.4)
    return out
