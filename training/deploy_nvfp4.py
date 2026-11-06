#!/usr/bin/env python3
"""Create, or delete, a temporary production-shaped endpoint for an NVFP4 Lightning export (METHODOLOGY.md §6).

The same shape as the throughput sweep that measured the base model's cost (sets/pmc-v1/
lightning-throughput, endpoints.json): qurator-nemotron's container settings (lightning_container.json,
vLLM image, one L40S, tensor-parallel 1, 64K context) with max_num_seqs raised to 64, on ml.g6e.xlarge
(2xlarge as the fallback pool). Serving the fine-tune this way measures its real cost per row and
whether it keeps its score through NVFP4. Delete the endpoint after the run.

Usage: python3 deploy_nvfp4.py create <name> <s3-nvfp4-prefix>
       python3 deploy_nvfp4.py delete <name>
"""
import json
import sys
from pathlib import Path

import boto3

from config import REGION, ROLE, BASE_TAGS
CONTAINER = json.loads((Path(__file__).resolve().parent / "lightning_container.json").read_text())
POOL = ["ml.g6e.xlarge", "ml.g6e.2xlarge"]
TAGS = [*BASE_TAGS, {"Key": "purpose", "Value": "lightning-nvfp4-eval-temporary"}]
ENV = {**CONTAINER["Environment"], "SM_VLLM_MAX_NUM_SEQS": "64"}
# The AWS deep-learning container exists in every region under the same account and tag, but ECR
# pulls can't cross regions: point the image at the endpoint's own region.
IMAGE = CONTAINER["Image"].replace(".dkr.ecr.us-east-2.", f".dkr.ecr.{REGION}.")


def create(name, prefix):
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        sm.create_model(ModelName=name, ExecutionRoleArn=ROLE, Tags=TAGS, PrimaryContainer={
            "Image": IMAGE, "Mode": "SingleModel", "Environment": ENV,
            "ModelDataSource": {"S3DataSource": {"S3Uri": prefix, "S3DataType": "S3Prefix", "CompressionType": "None"}}})
    except sm.exceptions.ClientError as e:
        if "already existing" not in str(e) and "already exists" not in str(e):
            raise
        print(f"model {name} exists; reusing it")
    # One instance type: the local botocore has no InstancePools. ml.g6e.xlarge is what the base sweep measured;
    # pass a second argument "2xlarge" to fall back by hand if capacity is short.
    instance = POOL[1] if len(sys.argv) > 4 and sys.argv[4] == "2xlarge" else POOL[0]
    sm.create_endpoint_config(EndpointConfigName=name, Tags=TAGS, ProductionVariants=[{
        "VariantName": "main", "ModelName": name, "InitialInstanceCount": 1, "InstanceType": instance,
        "ModelDataDownloadTimeoutInSeconds": 1800, "ContainerStartupHealthCheckTimeoutInSeconds": 1800}])
    sm.create_endpoint(EndpointName=name, EndpointConfigName=name, Tags=TAGS)
    print(f"creating endpoint {name} from {prefix} on {instance}")


def delete(name):
    sm = boto3.client("sagemaker", region_name=REGION)
    for call, kw in ((sm.delete_endpoint, {"EndpointName": name}), (sm.delete_endpoint_config, {"EndpointConfigName": name}),
                     (sm.delete_model, {"ModelName": name})):
        try:
            call(**kw)
        except Exception as e:
            print(f"{call.__name__}: {e}")
    print(f"deleted {name}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("create", "delete"):
        sys.exit(__doc__)
    create(*sys.argv[2:4]) if sys.argv[1] == "create" else delete(sys.argv[2])
