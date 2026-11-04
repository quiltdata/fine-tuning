#!/usr/bin/env python3
"""Export a merged BF16 Lightning fine-tune to NVFP4 in a SageMaker training job (METHODOLOGY.md §6).

Same image, role, bucket and source bundle as launch.py; the entry point is source/export_nvfp4.py.
The export lands uncompressed under the job's checkpoint prefix
(s3://sagemaker-<region>-<account>/<job-prefix>/checkpoints/<job>/nvfp4/), vLLM-loadable.

Usage: python3 export_job.py <name-suffix> <s3-merged-model-prefix> <calib.jsonl>
                             [--calib-size 64] [--calib-seq 4096] [--spot] [--region us-east-2] [--max-hours 4]
"""
import argparse
import io
import tarfile
from pathlib import Path

import boto3

from config import BASE_TAGS
from launch import ACCOUNT, PREFIX, ROLE, INSTANCE, region_paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suffix")
    ap.add_argument("model_prefix")
    ap.add_argument("calib_jsonl")
    ap.add_argument("--calib-size", type=int, default=64)
    ap.add_argument("--calib-seq", type=int, default=4096)
    ap.add_argument("--modelopt", default="nvidia-modelopt==0.47.0")
    ap.add_argument("--max-hours", type=float, default=4)
    ap.add_argument("--region", default="us-east-2", choices=["us-east-1", "us-east-2"])
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--max-wait-hours", type=float, default=12)
    a = ap.parse_args()
    bucket, image, _ = region_paths(a.region)

    name = f"{PREFIX}-ingest-{a.suffix}"
    s3 = boto3.client("s3", region_name=a.region)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted(Path(__file__).resolve().parent.joinpath("source").iterdir()):
            tar.add(f, arcname=f.name)
    src_key = f"{PREFIX}/ingest/{name}/source/source.tar.gz"
    calib_key = f"{PREFIX}/ingest/{name}/calib/train.jsonl"
    s3.put_object(Bucket=bucket, Key=src_key, Body=buf.getvalue())
    s3.upload_file(a.calib_jsonl, bucket, calib_key)

    hp = {"sagemaker_program": "export_nvfp4.py", "sagemaker_submit_directory": f"s3://{bucket}/{src_key}",
          "calib_size": a.calib_size, "calib_seq": a.calib_seq, "modelopt": a.modelopt}
    sm = boto3.client("sagemaker", region_name=a.region)
    sm.create_training_job(
        TrainingJobName=name,
        RoleArn=ROLE,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File"},
        HyperParameters={k: str(v) for k, v in hp.items()},
        InputDataConfig=[
            {"ChannelName": "model", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": a.model_prefix,
                                                                      "S3DataDistributionType": "FullyReplicated"}}},
            {"ChannelName": "calib", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                                                                      "S3Uri": f"s3://{bucket}/{PREFIX}/ingest/{name}/calib/",
                                                                      "S3DataDistributionType": "FullyReplicated"}}},
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/{PREFIX}/output/"},
        CheckpointConfig={"S3Uri": f"s3://{bucket}/{PREFIX}/checkpoints/{name}/", "LocalPath": "/opt/ml/checkpoints"},
        ResourceConfig={"InstanceType": INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 400},
        StoppingCondition={"MaxRuntimeInSeconds": int(a.max_hours * 3600),
                           **({"MaxWaitTimeInSeconds": int(max(a.max_wait_hours, a.max_hours) * 3600)} if a.spot else {})},
        EnableManagedSpotTraining=a.spot,
        Tags=[*BASE_TAGS, {"Key": "purpose", "Value": "lightning-nvfp4-export"}],
    )
    print(f"created {name}: {hp}")
    print(f"export: s3://{bucket}/{PREFIX}/checkpoints/{name}/nvfp4/")


if __name__ == "__main__":
    main()
