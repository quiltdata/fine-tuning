"""The abstract task's input rules: which abstract, how it is counted, how it is cut, and protocols.

One place for the rules the other tools share (METHODOLOGY.md, abstract task):

  main_abstract   the article's main abstract from its JATS <article-meta>: the first <abstract>
                  whose abstract-type is absent or not a secondary kind (graphical, toc, teaser,
                  highlights, lay summaries...). Publishers tag main abstracts "structured",
                  "primary" or "unstructured" as well as leaving them untagged.
  abstract text   section titles and labels kept as text followed by a period ("Background. ..."),
                  all other tags removed, HTML entities decoded, whitespace collapsed to single spaces.
  size            UTF-8 bytes of that text. The limit is 4,096 bytes.
  truncate_utf8   cuts a longer abstract at a whole character, preferring the last sentence end in
                  the final quarter of the limit, then the last space, and marks it truncated.
  protocol_flags  a study protocol (a trial or review not yet run, so no findings) by any of: PubMed
                  type "Clinical Trial Protocol"; a title naming a study, trial or review protocol;
                  protocol-style abstract headings with no Results heading.
"""
import html
import re

LIMIT_BYTES = 4096

SECONDARY = {"graphical", "toc", "toc-graphic", "teaser", "highlights", "author-highlights", "web-summary", "summary",
             "author-summary", "editor-summary", "significance", "key-points", "lay-summary", "video"}
NOT_SECTIONS = re.compile(r"supplementary|graphical|key points", re.I)
PROTOCOL_TITLE = re.compile(r"(study|trial|review) protocol|protocol (for|of) (a|an|the)\b|:\s*(a |the )?protocol\b|\bprotocol$", re.I)
PROTOCOL_HEADING = re.compile(r"methods? and analysis|ethics and dissemination|methods?\s*/\s*design", re.I)
RESULTS_HEADING = re.compile(r"\bresults?\b", re.I)
SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")


def flat(fragment):
    """Text of an XML fragment: tags removed, entities decoded, whitespace collapsed."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def abstract_text(body):
    """The counted text of one <abstract> body: titles and labels kept as "Title. ", then flattened."""
    return flat(re.sub(r"<(title|label)[^>]*>(.*?)</\1>", r" \2. ", body, flags=re.S))


def main_abstract(xml):
    """The article's main abstract, or None: {"text", "headings", "structured", "abstract_type", "bytes"}."""
    meta = xml.split("</article-meta>")[0]
    for attrs, body in re.findall(r"<abstract(\s[^>]*)?>(.*?)</abstract>", meta, re.S):
        kind = re.search(r'abstract-type="([^"]+)"', attrs or "")
        kind = kind.group(1) if kind else ""
        if kind.lower() in SECONDARY:
            continue
        text = abstract_text(body)
        headings = [h for h in (flat(t) for t in re.findall(r"<sec[^>]*>\s*<title[^>]*>(.*?)</title>", body, re.S))
                    if h and not NOT_SECTIONS.search(h)]
        return {"text": text, "headings": headings, "structured": "<sec" in body, "abstract_type": kind or "main",
                "bytes": utf8_bytes(text)}
    return None


def article_fields(xml):
    """Title, journal and the publisher's JATS article-type, from the article's front matter."""
    meta = xml.split("</article-meta>")[0]
    title = re.search(r"<article-title[^>]*>(.*?)</article-title>", meta, re.S)
    journal = re.search(r"<journal-title>(.*?)</journal-title>", xml, re.S)
    kind = re.search(r'<article[^>]*\sarticle-type="([^"]+)"', xml)
    return {"title": flat(title.group(1)) if title else "", "journal": flat(journal.group(1)) if journal else "",
            "jats_article_type": kind.group(1) if kind else ""}


def utf8_bytes(text):
    return len(text.encode("utf-8"))


def truncate_utf8(text, limit=LIMIT_BYTES):
    """(text, truncated): text within `limit` UTF-8 bytes, cut at a whole character.

    Prefers the last sentence end that keeps at least three quarters of the limit, then the last
    space, then the last whole character."""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text, False
    cut = data[:limit].decode("utf-8", errors="ignore")  # drops only a trailing partial character
    floor = len(data[: limit * 3 // 4].decode("utf-8", errors="ignore"))
    ends = [m.end() for m in SENTENCE_END.finditer(cut + " ") if m.end() >= floor]
    if ends:
        return cut[: ends[-1]].rstrip(), True
    space = cut.rfind(" ")
    if space >= floor:
        return cut[:space].rstrip(), True
    return cut, True


def protocol_flags(pub_types=(), title="", headings=()):
    """Which of the three protocol signals fire, and the verdict ("protocol": any of them)."""
    pubmed = "Clinical Trial Protocol" in pub_types
    by_title = bool(PROTOCOL_TITLE.search(title or ""))
    heads = [h for h in headings if h and not NOT_SECTIONS.search(h)]
    by_headings = any(PROTOCOL_HEADING.search(h) for h in heads) and not any(RESULTS_HEADING.search(h) for h in heads)
    return {"pubmed": pubmed, "title": by_title, "headings": by_headings, "protocol": pubmed or by_title or by_headings}
