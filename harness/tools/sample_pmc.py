#!/usr/bin/env python3
"""Sample PMC articles for the summarizer set and write their units as an objects list.

Picks articles at random (fixed seed) from one part of the daily metadata inventory
of s3://pmc-oa-opendata, keeps those that are open access, not retracted, licensed
CC BY or CC0 (redistributable with attribution), and have at least one
supplement make_inputs.py can read. Each kept article gives one unit for its .txt
and up to MAX_SUPPLEMENTS units for supplements. Candidates are read in chunks of
CANDIDATES, from further inventory parts if needed, until the target row count is met.
--exclude takes articles.csv files of other sets; their PMCIDs (any version) are skipped,
so a training set never shares an article with a test set.

Writes objects.txt (for make_inputs.py) and articles.csv (one row per article:
PMCID, license, title, units chosen).

Usage: python3 sample_pmc.py <target_rows> <out_dir> [seed] [--exclude articles.csv ...]
"""
import argparse
import csv
import gzip
import io
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "pmc-oa-opendata"
INVENTORY = "inventory-reports/pmc-oa-opendata/metadata/"
TRAIN_LICENSES = {"CC BY", "CC0"}
SUPPLEMENT_TYPES = {"docx", "xlsx", "csv", "tsv", "txt", "pdf"}   # types make_inputs.py can read
MAX_SUPPLEMENTS = 3
CANDIDATES = 4000

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=32))


def inventory_parts(rng):
    """The latest daily inventory and its parts, in a seeded random order."""
    days = [p["Prefix"] for p in s3.list_objects_v2(Bucket=BUCKET, Prefix=INVENTORY, Delimiter="/")["CommonPrefixes"]
            if p["Prefix"][len(INVENTORY):][:2] == "20"]
    manifest = json.loads(s3.get_object(Bucket=BUCKET, Key=max(days) + "manifest.json")["Body"].read())
    parts = sorted(f["key"] for f in manifest["files"])
    rng.shuffle(parts)
    return max(days), parts


def inventory_keys(part):
    data = gzip.decompress(s3.get_object(Bucket=BUCKET, Key=part)["Body"].read()).decode()
    return [r[1] for r in csv.reader(io.StringIO(data)) if r[1].endswith(".json")]


def metadata(key):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except Exception:
        return None


def units(meta):
    """S3 URIs to summarize for one article: its .txt and readable supplements."""
    txt = meta["text_url"].split("?")[0]
    supplements = [u.split("?")[0] for u in meta.get("media_urls") or []]
    supplements = [u for u in supplements if u.rsplit(".", 1)[-1].lower() in SUPPLEMENT_TYPES]
    return [txt] + supplements[:MAX_SUPPLEMENTS] if supplements else []


def main(target_rows, out_dir, seed=20261024, exclude=()):
    rng = random.Random(seed)
    skip = {r["pmcid"] for path in exclude for r in csv.DictReader(open(path, newline=""))}
    day, parts = inventory_parts(rng)
    kept, rows, read, sampled, used = [], 0, 0, 0, []
    for part in parts:
        keys = inventory_keys(part)
        rng.shuffle(keys)
        used.append(part.rsplit("/", 1)[-1])
        for start in range(0, len(keys), CANDIDATES):
            chunk = keys[start:start + CANDIDATES]
            sampled += len(chunk)
            with ThreadPoolExecutor(32) as pool:
                metas = list(pool.map(metadata, chunk))
            read += sum(1 for m in metas if m)
            for meta in metas:
                if not meta or meta.get("license_code") not in TRAIN_LICENSES or not meta.get("is_pmc_openaccess") \
                        or meta.get("is_retracted") or not meta.get("text_url") or meta.get("pmcid") in skip:
                    continue
                u = units(meta)
                if not u:
                    continue
                kept.append((meta, u))
                rows += len(u)
                if rows >= target_rows:
                    break
            if rows >= target_rows:
                break
        if rows >= target_rows:
            break
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "objects.txt", "w") as fh:
        fh.write(f"# PMC summarizer set: {len(kept)} articles, {rows} units. Inventory {day}, parts {', '.join(used)}, seed {seed}.\n")
        if skip:
            fh.write(f"# Excluded {len(skip)} PMCIDs from: {', '.join(str(p) for p in exclude)}.\n")
        fh.write("# Licenses CC BY or CC0, open access, not retracted; article .txt plus up to "
                 f"{MAX_SUPPLEMENTS} readable supplements ({', '.join(sorted(SUPPLEMENT_TYPES))}).\n")
        for _, u in kept:
            fh.write("\n".join(u) + "\n")
    with open(out / "articles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pmcid", "version", "license_code", "title", "units"])
        for meta, u in kept:
            w.writerow([meta["pmcid"], meta.get("version"), meta["license_code"], meta.get("title"), len(u)])
    print(f"{sampled} sampled, {read} read, {len(kept)} articles kept, {rows} units", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target_rows", type=int)
    ap.add_argument("out_dir")
    ap.add_argument("seed", type=int, nargs="?", default=20261024)
    ap.add_argument("--exclude", nargs="+", default=[], help="articles.csv files whose PMCIDs to skip")
    a = ap.parse_args()
    main(a.target_rows, a.out_dir, a.seed, a.exclude)
