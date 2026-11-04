#!/usr/bin/env python3
"""Build the summarizer's input table from a list of S3 objects.

Each line of the objects file is the s3:// URI of one file, or of a file kind:
a prefix plus a glob on the file name (s3://bucket/prefix/*.json). Blank lines and
lines starting with '#' are ignored. Objects are read anonymously.

Columns follow schemas/input.schema.json; list-valued cells hold JSON arrays.

Usage: python3 make_inputs.py objects.txt inputs.csv [--excerpt-chars N] [--pdf-pages N] [--parent-abstract]

--parent-abstract adds a `parent` column: for a file that sits in an article's folder (a supplement), the title and
abstract of that article, read from a JATS XML file in the same folder; empty for the article itself and where none
is found.

--excerpt-chars 0 and --pdf-pages 0 read documents whole (uncapped PDFs). The defaults,
8,000 characters and 20 pages, are what every earlier set was built with.
"""
import csv
import fnmatch
import html
import io
import json
import logging
import re
import sys
import zipfile

import boto3
from botocore import UNSIGNED
from botocore.config import Config

EXCERPT_CHARS = 8000      # cap on every excerpt (--excerpt-chars; 0 = no cap)
PDF_PAGES = 20            # pages read from a PDF (--pdf-pages; 0 = all)
PARENT = False            # --parent-abstract: add the `parent` column
TABLE_ROWS = 5            # data rows shown for tables
MENTION_CONTEXT = 200     # characters either side of a mention
MAX_MENTIONS = 5
MIN_TEXT = 20             # units with fewer non-space characters than this are skipped
ANCESTOR_LEVELS = 2       # signposts are read from the prefix and this many levels up (never the bucket root)
COLUMNS = ["unit_id", "prefix", "unit", "file_type", "members", "example_keys",
           "size_bytes", "excerpt_method", "excerpt", "mentions"]

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def split_uri(uri):
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def list_children(bucket, prefix):
    """Objects directly under prefix: [(key, size)]."""
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        out += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return out


def get(bucket, key, max_bytes=None):
    kwargs = {"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {}
    return s3.get_object(Bucket=bucket, Key=key, **kwargs)["Body"].read()


def file_type(key):
    name = key.rsplit("/", 1)[-1].lower()
    return name.rsplit(".", 1)[-1] if "." in name else ""


def cap(text, method):
    if EXCERPT_CHARS and len(text) > EXCERPT_CHARS:
        return f"{method}; truncated to the first {EXCERPT_CHARS:,} of {len(text):,} characters", text[:EXCERPT_CHARS]
    return method, text


def table_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def excerpt(bucket, key, size):
    """(method, text) for one object, or None if this file type isn't handled."""
    ft = file_type(key)
    if ft in {"md", "txt", "yaml", "yml", "json", "xml", "log"}:
        if size <= EXCERPT_CHARS:
            return "full text", get(bucket, key).decode("utf-8", "replace")
        text = get(bucket, key, 4 * EXCERPT_CHARS).decode("utf-8", "replace")
        return f"first {EXCERPT_CHARS:,} characters of a {size:,}-byte file", text[:EXCERPT_CHARS]
    if ft in {"csv", "tsv"}:
        limit = 2_000_000
        data = get(bucket, key, limit if size > limit else None).decode("utf-8", "replace")
        rows = list(csv.reader(io.StringIO(data), delimiter="\t" if ft == "tsv" else ","))
        if size > limit:
            rows = rows[:-1]  # last row may be cut off
        n = f"{len(rows) - 1:,} data rows" if size <= limit else f"at least {len(rows) - 1:,} data rows (partial read)"
        return cap(table_text(rows[0], rows[1:1 + TABLE_ROWS]),
                   f"header and first {TABLE_ROWS} rows; {n}; {len(rows[0])} columns")
    if ft == "parquet" and size <= 50_000_000:
        import pyarrow.parquet as pq
        t = pq.read_table(io.BytesIO(get(bucket, key)))
        schema = "\n".join(f"{f.name}: {f.type}" for f in t.schema)
        head = t.slice(0, TABLE_ROWS).to_pylist()
        rows = [[r[c] for c in t.column_names] for r in head]
        return cap(f"schema:\n{schema}\n\nfirst rows:\n" + table_text(t.column_names, rows),
                   f"schema and first {TABLE_ROWS} rows; {t.num_rows:,} rows; {t.num_columns} columns")
    if ft == "xlsx" and size <= 20_000_000:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(get(bucket, key)), read_only=True, data_only=True)
        parts = []
        for ws in wb.worksheets[:5]:
            rows = []
            for _, r in zip(range(8), ws.iter_rows(values_only=True)):
                r = ["" if v is None else v for v in r]
                while r and r[-1] == "":
                    r.pop()
                rows.append(r)
            size = f" ({ws.max_row} rows x {ws.max_column} columns)" if ws.max_row else ""
            parts.append(f"sheet {ws.title!r}{size}:\n" + table_text(rows[0] if rows else [], rows[1:]))
        return cap("\n".join(parts), f"first 8 rows of each of {min(len(wb.worksheets), 5)} of {len(wb.worksheets)} sheets")
    if ft == "docx" and size <= 30_000_000:
        z = zipfile.ZipFile(io.BytesIO(get(bucket, key)))
        x = z.read("word/document.xml").decode("utf-8", "replace")
        paras = [t for t in (html.unescape("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))).strip()
                             for p in re.findall(r"<w:p[ >].*?</w:p>", x, re.S)) if t]
        media = sum(1 for n in z.namelist() if n.startswith("word/media/"))
        return cap("\n".join(paras), f"document text, {len(paras)} paragraphs, {media} embedded images (not shown)")
    if ft == "pdf" and size <= 30_000_000:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(get(bucket, key)))
        pages = reader.pages[:PDF_PAGES] if PDF_PAGES else reader.pages
        text = "\n".join((p.extract_text() or "") for p in pages)
        return cap(text, f"text of the first {len(pages)} of {len(reader.pages)} pages" if len(pages) < len(reader.pages)
                   else f"text of all {len(reader.pages)} pages")
    if ft == "vcf":
        data = get(bucket, key, 262_144).decode("utf-8", "replace").splitlines()[:-1]
        short = lambda l: l if len(l) <= 400 else f"{l[:400]} [{len(l) - 400:,} more characters]"
        meta, seen = [], {}
        for l in (l for l in data if l.startswith("##")):
            kind = re.split(r"[=<0-9-]", l[2:], maxsplit=1)[0]    # ##contig, ##GVCFBlock, ##FORMAT ...
            seen[kind] = seen.get(kind, 0) + 1
            if seen[kind] <= 3:
                meta.append(short(l))
        meta += [f"[{n - 3} more ##{k} lines omitted]" for k, n in seen.items() if n > 3]
        tail = "\n".join([l for l in data if l.startswith("#CHROM")] + [short(l) for l in data if not l.startswith("#")][:3])
        body = "\n".join(meta)[:EXCERPT_CHARS - len(tail) - 1]
        return "VCF header (at most 3 lines of each type, long lines shortened), column line and first 3 records", f"{body}\n{tail}"
    return None


_signposts = {}


def signposts(bucket, prefix):
    """Text of README/Markdown and XML files directly under prefix: {name: text}."""
    if (bucket, prefix) not in _signposts:
        found = {}
        for key, size in list_children(bucket, prefix):
            name = key.rsplit("/", 1)[-1]
            if size <= 2_000_000 and (name.lower().startswith("readme") or name.lower().endswith((".md", ".xml"))):
                found[key] = get(bucket, key).decode("utf-8", "replace")
        _signposts[(bucket, prefix)] = found
    return _signposts[(bucket, prefix)]


def mentions(bucket, prefix, term, self_key=None):
    """Passages in nearby signposts that name term, each prefixed with the signpost's key."""
    # The bucket root is skipped: listing it can mean millions of prefixes (pmc-oa-opendata).
    parts = prefix.rstrip("/").split("/") if prefix else []
    ancestors = ["/".join(parts[:n]) + "/" for n in range(len(parts), 0, -1)][:ANCESTOR_LEVELS + 1]
    out = []
    for p in ancestors:
        for key, text in signposts(bucket, p).items():
            if key == self_key:
                continue
            for m in re.finditer(re.escape(term), text):
                window = text[max(0, m.start() - MENTION_CONTEXT):m.end() + MENTION_CONTEXT]
                if key.lower().endswith(".xml"):                      # strip markup, including tags cut by the window
                    window = re.sub(r"^[^<]*?>", "", window) if window.find(">") < window.find("<") else window
                    window = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<[^>]*$", "", window)))
                window = re.sub(r"\s+", " ", window).strip()
                out.append(f"{key}: {window}")
                if len(out) >= MAX_MENTIONS:
                    return out
    return out


def parent_text(bucket, prefix, name):
    """Title and abstract of the JATS article in this folder, unless `name` is that article's own file."""
    import xml.etree.ElementTree as ET
    for key, text in signposts(bucket, prefix).items():
        if not key.lower().endswith(".xml") or "<abstract" not in text:
            continue
        stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name.rsplit(".", 1)[0] == stem:
            return ""
        try:
            root = ET.fromstring(text.encode())
        except ET.ParseError:
            continue
        for el in root.iter():
            if isinstance(el.tag, str) and "}" in el.tag:
                el.tag = el.tag.split("}", 1)[1]
        flat = lambda el: re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""
        title = flat(root.find(".//article-meta/title-group/article-title"))
        abstract = root.find(".//article-meta/abstract")
        if abstract is None:
            continue
        parts = []
        secs = abstract.findall("./sec")
        for s in secs or [abstract]:
            label = flat(s.find("title")) if secs else ""
            body = " ".join(flat(p) for p in s.findall(".//p")) or flat(s)
            parts.append(f"{label}: {body}" if label else body)
        return f"Article title: {title}\nAbstract: " + " ".join(parts)
    return ""


def row_for(uri):
    bucket, key = split_uri(uri)
    prefix, _, name = key.rpartition("/")
    prefix = prefix + "/" if prefix else ""
    children = list_children(bucket, prefix)
    if any(c in name for c in "*?["):
        members = sorted((k, s) for k, s in children if fnmatch.fnmatch(k.rsplit("/", 1)[-1], name))
        term = prefix.rstrip("/").rsplit("/", 1)[-1]
    else:
        members = [(k, s) for k, s in children if k == key]
        term = name
    if not members:
        raise ValueError(f"no objects for {uri}")
    sample_key, sample_size = members[0]
    ex = excerpt(bucket, sample_key, sample_size)
    if ex is None:
        raise ValueError(f"file type not handled: {uri}")
    method, text = ex
    if len("".join(text.split())) < MIN_TEXT:
        raise ValueError(f"no readable text: {uri}")
    if len(members) > 1:
        method = f"one sample of {len(members):,} files; {method}"
    return {
        "unit_id": uri,
        "prefix": f"s3://{bucket}/{prefix}",
        "unit": name,
        "file_type": file_type(sample_key),
        "members": len(members),
        "example_keys": json.dumps([k.rsplit("/", 1)[-1] for k, _ in members[:3]]),
        "size_bytes": sample_size,
        "excerpt_method": method,
        "excerpt": text,
        "mentions": json.dumps(mentions(bucket, prefix, term, sample_key), ensure_ascii=False),
        **({"parent": parent_text(bucket, prefix, name)} if PARENT else {}),
    }


def main(objects_path, out_path):
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    uris = [l.strip() for l in open(objects_path) if l.strip() and not l.startswith("#")]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS + (["parent"] if PARENT else []))
        w.writeheader()
        for uri in uris:
            try:
                w.writerow(row_for(uri))
                print(f"ok    {uri}", file=sys.stderr)
            except ValueError as e:
                print(f"skip  {e}", file=sys.stderr)
            except Exception as e:  # a malformed file must not stop the run
                print(f"skip  {type(e).__name__} reading {uri}: {str(e)[:120]}", file=sys.stderr)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--parent-abstract" in args:
        PARENT = True
        args.remove("--parent-abstract")
    for flag in ("--excerpt-chars", "--pdf-pages"):
        if flag in args:
            i = args.index(flag)
            globals()["EXCERPT_CHARS" if flag == "--excerpt-chars" else "PDF_PAGES"] = int(args[i + 1])
            del args[i:i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    main(*args)
