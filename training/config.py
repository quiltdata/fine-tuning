#!/usr/bin/env python3
"""Shared AWS/SageMaker configuration, loaded from the environment.

The account-specific values (account id, execution role, job prefix, resource
tags) are not baked into the code. They come from environment variables, usually
via a git-ignored `.env` at the repo root (copy `.env.example`, or generate it
with `make-env.sh`). See `.env.example` for the list.

Real process environment variables always win over the `.env` file.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    # repo-root .env first, then a training/.env fallback; real env vars still win
    for base in (_REPO_ROOT, Path(__file__).resolve().parent):
        f = base / ".env"
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _require(key):
    v = os.environ.get(key)
    if not v:
        raise RuntimeError(
            f"{key} is not set. Copy .env.example to .env and fill it in, "
            f"or run ./make-env.sh <source.env>."
        )
    return v


ACCOUNT = _require("AWS_ACCOUNT_ID")
ROLE_NAME = _require("SAGEMAKER_ROLE_NAME")
PREFIX = _require("SAGEMAKER_JOB_PREFIX")
REGION = os.environ.get("SAGEMAKER_REGION", "us-east-2")
INSTANCE = os.environ.get("SAGEMAKER_INSTANCE", "ml.g6e.12xlarge")
ROLE = f"arn:aws:iam::{ACCOUNT}:role/{ROLE_NAME}"

# tags on every job/model/endpoint; each script appends its own "purpose" tag
BASE_TAGS = [{"Key": "project", "Value": os.environ.get("SAGEMAKER_TAG_PROJECT", "fine-tuning")}]
if os.environ.get("SAGEMAKER_TAG_OWNER"):
    BASE_TAGS.append({"Key": "owner", "Value": os.environ["SAGEMAKER_TAG_OWNER"]})
