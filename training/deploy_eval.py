#!/usr/bin/env python3
"""Create, or delete, a temporary vLLM evaluation endpoint for a BF16 Lightning checkpoint.

Base and fine-tuned models are served the same way, so a before/after measures training,
not serving: vLLM (the image qurator-nemotron uses), BF16 weights unquantized, tensor-parallel
over the 4 L40S GPUs of one ml.g6e.12xlarge (24xlarge as fallback), in us-east-2. FP8-at-load on
a single L40S doesn't fit: vLLM reads the BF16 weights first and filled all 44 GB (2026-10-24).
These endpoints measure quality; cost per row comes from the NVFP4 throughput sweep. The execution role
can read the BF16 JumpStart weights and s3://sagemaker-<region>-<account>/<job-prefix>/,
where training writes the merged model. Everything is tagged temporary; delete it after evaluation.

Usage: python3 deploy_eval.py create <name> <s3-model-prefix>
       python3 deploy_eval.py delete <name>
"""
import json
import sys
from pathlib import Path

import boto3

from config import REGION, ROLE, BASE_TAGS  # us-east-1 had no L40S capacity on 2026-10-24
IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/vllm:server-sagemaker-cuda-v2.2"
# One L40S GPU in each; xlarge first. qurator-nemotron uses the same kind of pool, because single
# types often lack capacity (ingest-lightning-base-fp8 failed on InsufficientInstanceCapacity).
POOL = ["ml.g6e.12xlarge", "ml.g6e.24xlarge"]
TAGS = [*BASE_TAGS, {"Key": "purpose", "Value": "lightning-eval-temporary"}]
# qurator-nemotron's environment (production Lightning), changed only where noted
ENV = json.loads((Path(__file__).resolve().parent / "lightning_container.json").read_text())["Environment"]
ENV.update({"SM_VLLM_TENSOR_PARALLEL_SIZE": "4",  # BF16 across the 4 GPUs
            "SM_VLLM_MAX_MODEL_LEN": "16384",       # longest test input plus a 4K output fits
            "SM_VLLM_MAX_NUM_SEQS": "64"})


def create(name, prefix):
    sm = boto3.client("sagemaker", region_name=REGION)
    sm.create_model(ModelName=name, ExecutionRoleArn=ROLE, Tags=TAGS, PrimaryContainer={
        "Image": IMAGE, "Mode": "SingleModel", "Environment": ENV,
        "ModelDataSource": {"S3DataSource": {"S3Uri": prefix, "S3DataType": "S3Prefix", "CompressionType": "None"}}})
    sm.create_endpoint_config(EndpointConfigName=name, Tags=TAGS, ProductionVariants=[{
        "VariantName": "main", "ModelName": name, "InitialInstanceCount": 1,
        "InstancePools": [{"InstanceType": t, "Priority": i + 1} for i, t in enumerate(POOL)],
        "ModelDataDownloadTimeoutInSeconds": 1800, "ContainerStartupHealthCheckTimeoutInSeconds": 1800}])
    sm.create_endpoint(EndpointName=name, EndpointConfigName=name, Tags=TAGS)
    print(f"creating endpoint {name} from {prefix}")


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
