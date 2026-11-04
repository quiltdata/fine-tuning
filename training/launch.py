#!/usr/bin/env python3
"""Launch a Lightning LoRA training job on SageMaker (us-east-2, ml.g6e.12xlarge).

Uploads source/ as source.tar.gz and the training JSONL under the prefix the execution role
(SAGEMAKER_ROLE_NAME) may read, then creates the job. Job names must match
<SAGEMAKER_JOB_PREFIX>-* (the role's policy). The base BF16 checkpoint is read straight from
the JumpStart cache. --merge writes the merged model to the job's checkpoint prefix.

Usage: python3 launch.py <name-suffix> <train.jsonl> [--max-examples N] [--max-steps N] [--epochs E]
                         [--lr LR] [--rank R] [--grad-accum G] [--merge] [--max-hours H]
                         [--region us-east-1|us-east-2] [--spot [--max-wait-hours H]] [--diagnose]
"""
import argparse
import io
import os
import tarfile
from pathlib import Path

import boto3

from config import ACCOUNT, PREFIX, ROLE, INSTANCE, BASE_TAGS


def region_paths(region):
    return (f"sagemaker-{region}-{ACCOUNT}",
            f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:2.9.0-transformers5.3.0-gpu-py312-cu130-ubuntu22.04",
            f"s3://jumpstart-cache-prod-{region}/huggingface-reasoning/huggingface-reasoning-nemotron-3-5-lightning-30b-a3b-bf16/artifacts/inference/v1.0.0/")
TAGS = [*BASE_TAGS, {"Key": "purpose", "Value": "lightning-summarizer-sft"}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suffix")
    ap.add_argument("train_jsonl")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--diagnose", action="store_true", help="check weights, kernel paths and generation, then exit")
    ap.add_argument("--merge-adapter", help="S3 prefix of a LoRA adapter: merge it into the base and exit (no training)")
    ap.add_argument("--init-adapter", help="S3 prefix of a LoRA adapter to continue training from")
    ap.add_argument("--max-hours", type=float, default=2)
    ap.add_argument("--region", default="us-east-2", choices=["us-east-1", "us-east-2"], help="the role allows both")
    ap.add_argument("--spot", action="store_true", help="managed Spot: a separate capacity pool and quota, about 70%% cheaper")
    ap.add_argument("--max-wait-hours", type=float, default=12, help="--spot only: give up if no capacity by then")
    a = ap.parse_args()
    REGION = a.region
    BUCKET, IMAGE, BASE = region_paths(REGION)

    name = f"{PREFIX}-ingest-{a.suffix}"
    s3 = boto3.client("s3", region_name=REGION)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted(Path(__file__).resolve().parent.joinpath("source").iterdir()):
            tar.add(f, arcname=f.name)
    src_key = f"{PREFIX}/ingest/{name}/source/source.tar.gz"
    data_key = f"{PREFIX}/ingest/{name}/data/train.jsonl"
    s3.put_object(Bucket=BUCKET, Key=src_key, Body=buf.getvalue())
    s3.upload_file(a.train_jsonl, BUCKET, data_key)

    hp = {"sagemaker_program": "train.py", "sagemaker_submit_directory": f"s3://{BUCKET}/{src_key}",
          "epochs": a.epochs, "lr": a.lr, "rank": a.rank, "alpha": 2 * a.rank, "grad_accum": a.grad_accum, "batch": a.batch,
          "max_examples": a.max_examples, "max_steps": a.max_steps, "merge": int(a.merge),
          "diagnose": int(a.diagnose), "merge_only": int(bool(a.merge_adapter)),
          "init_adapter": int(bool(a.init_adapter))}
    sm = boto3.client("sagemaker", region_name=REGION)
    sm.create_training_job(
        TrainingJobName=name,
        RoleArn=ROLE,
        AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File"},
        HyperParameters={k: str(v) for k, v in hp.items()},
        InputDataConfig=[
            {"ChannelName": "model", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": BASE,
                                                                      "S3DataDistributionType": "FullyReplicated"}}},
            {"ChannelName": "train", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                                                                      "S3Uri": f"s3://{BUCKET}/{PREFIX}/ingest/{name}/data/",
                                                                      "S3DataDistributionType": "FullyReplicated"}}},
        ] + ([{"ChannelName": "adapter", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": a.merge_adapter or a.init_adapter, "S3DataDistributionType": "FullyReplicated"}}}]
             if (a.merge_adapter or a.init_adapter) else []),
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{PREFIX}/output/"},
        CheckpointConfig={"S3Uri": f"s3://{BUCKET}/{PREFIX}/checkpoints/{name}/", "LocalPath": "/opt/ml/checkpoints"},
        ResourceConfig={"InstanceType": INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 400},
        StoppingCondition={"MaxRuntimeInSeconds": int(a.max_hours * 3600),
                           **({"MaxWaitTimeInSeconds": int(max(a.max_wait_hours, a.max_hours) * 3600)} if a.spot else {})},
        EnableManagedSpotTraining=a.spot,
        Tags=TAGS,
    )
    print(f"created {name}: {hp}")
    print(f"logs: aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {name} --region {REGION} --follow")


if __name__ == "__main__":
    main()
