#!/usr/bin/env python3
"""Evaluate a BF16 Lightning checkpoint on the 262 test rows inside a SageMaker training job.

On-demand 4-GPU endpoints had no capacity on 2026-10-24, while Spot training capacity did. So
the job runs the same vLLM image as qurator-nemotron with its own entrypoint: start vLLM on the
model channel (BF16, tensor-parallel 4, 32K context, 64 sequences, the model's reasoning parser),
wait for it, then run ingest's run_model.py against it with thinking off. outputs.csv, calls.csv,
run.json and the vLLM log end up in the job's model.tar.gz. Base and fine-tuned run identically.

With --sample it instead draws --n answers per training batch (pmc-v2) at --temperature, for
rejection-sampling fine-tuning (4 at 0.8 for v2, 6 at 0.9 for v4-v6).

With --inputs it evaluates another input table (e.g. whole PDFs), and --tp, --max-model-len and
--per-file serve an NVFP4 export the way production does (one GPU, 64K context) and send one document per call.

Usage: python3 eval_job.py <name-suffix> <s3-model-prefix> [--spot] [--region us-east-2] [--max-hours H]
                           [--sample [--n 4] [--temperature 0.8]]
                           [--inputs inputs.csv] [--tp 4] [--max-model-len 32768] [--per-file]
"""
import argparse
from pathlib import Path

import boto3

from config import ACCOUNT, PREFIX, ROLE, BASE_TAGS
INGEST = Path(__file__).resolve().parent.parent / "ingest"
CODE = ["tools/run_model.py", "tools/model_rates.json", "prompts/summarizer.md", "schemas/output.schema.json",
        "eligible_inputs.csv"]
SAMPLE_CODE = {"set2/inputs.csv": "train_inputs.csv"}  # --sample: ingest path -> name in the code channel
EVAL_CMD = """cd $C && python3 tools/run_model.py eligible_inputs.csv $O/run --model lightning --openai-url http://localhost:8000/v1 \\
  --no-thinking --workers 64 --max-tokens 4096__PER_FILE__"""
SAMPLE_CMD = """cd $C && python3 sample_rft.py train_inputs.csv $O/samples.jsonl --n __N__ --temperature __T__ --workers 64"""
SCRIPT = r"""
set -uo pipefail
M=/opt/ml/input/data/model; C=/opt/ml/input/data/code; O=/opt/ml/model
mkdir -p $O
python3 -c "import boto3" 2>/dev/null || pip install -q boto3
python3 -m vllm.entrypoints.openai.api_server --model $M --served-model-name lightning --port 8000 \
  --tensor-parallel-size __TP__ --max-model-len __MAXLEN__ --max-num-seqs 64 --gpu-memory-utilization 0.85 \
  --trust-remote-code --reasoning-parser-plugin $M/ultra_v3_reasoning_parser.py --reasoning-parser ultra_v3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder > $O/vllm.log 2>&1 &
python3 - <<'PY'
import time, urllib.request, sys
for i in range(180):
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=5); print("vllm ready after", i * 10, "s", flush=True); sys.exit(0)
    except Exception:
        time.sleep(10)
print("vllm did not start", flush=True); sys.exit(1)
PY
if [ $? -ne 0 ]; then tail -60 $O/vllm.log; exit 1; fi
__COMMAND__
status=$?
tail -5 $O/vllm.log
exit $status
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suffix")
    ap.add_argument("model_prefix")
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--max-hours", type=float, default=2)
    ap.add_argument("--sample", action="store_true", help="sample --n answers per training batch (sample_rft.py) instead of evaluating")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--inputs", help="input table to evaluate instead of the 262-row eligible_inputs.csv")
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel size (1 for an NVFP4 export, as production)")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--per-file", action="store_true", help="one row per call (whole documents)")
    a = ap.parse_args()

    name = f"{PREFIX}-ingest-eval-{a.suffix}"
    bucket = f"sagemaker-{a.region}-{ACCOUNT}"
    code_prefix = f"{PREFIX}/ingest/{name}/code/"
    s3 = boto3.client("s3", region_name=a.region)
    for rel in CODE:
        src = a.inputs if (rel == "eligible_inputs.csv" and a.inputs) else str(INGEST / rel)
        s3.upload_file(src, bucket, code_prefix + rel)
    if a.sample:
        for rel, as_name in SAMPLE_CODE.items():
            s3.upload_file(str(INGEST / rel), bucket, code_prefix + as_name)
        s3.upload_file(str(Path(__file__).resolve().parent / "sample_rft.py"), bucket, code_prefix + "sample_rft.py")
    command = SAMPLE_CMD.replace("__N__", str(a.n)).replace("__T__", str(a.temperature)) if a.sample else \
        EVAL_CMD.replace("__PER_FILE__", " --per-file" if a.per_file else "")
    script = SCRIPT.replace("__COMMAND__", command).replace("__TP__", str(a.tp)).replace("__MAXLEN__", str(a.max_model_len))
    s3.put_object(Bucket=bucket, Key=code_prefix + "run_eval.sh", Body=script.encode())  # arguments are capped at 256 chars
    image = f"763104351884.dkr.ecr.{a.region}.amazonaws.com/vllm:server-sagemaker-cuda-v2.2"
    sm = boto3.client("sagemaker", region_name=a.region)
    sm.create_training_job(
        TrainingJobName=name,
        RoleArn=ROLE,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File",
                                "ContainerEntrypoint": ["/bin/bash", "/opt/ml/input/data/code/run_eval.sh"]},
        InputDataConfig=[
            {"ChannelName": "model", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": a.model_prefix,
                                                                      "S3DataDistributionType": "FullyReplicated"}}},
            {"ChannelName": "code", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/{code_prefix}",
                                                                     "S3DataDistributionType": "FullyReplicated"}}},
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/{PREFIX}/output/"},
        ResourceConfig={"InstanceType": "ml.g6e.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 300},
        StoppingCondition={"MaxRuntimeInSeconds": int(a.max_hours * 3600),
                           **({"MaxWaitTimeInSeconds": int(max(a.max_hours, 12) * 3600)} if a.spot else {})},
        EnableManagedSpotTraining=a.spot,
        Tags=[*BASE_TAGS, {"Key": "purpose", "Value": "lightning-eval"}],
    )
    print(f"created {name} on {a.model_prefix}")


if __name__ == "__main__":
    main()
