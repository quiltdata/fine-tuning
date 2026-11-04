#!/usr/bin/env python3
"""Evaluate a BF16 Lightning checkpoint on the 262 test rows inside a SageMaker training job.

On-demand 4-GPU endpoints had no capacity on 2026-10-24, while Spot training capacity did. So
the job runs the same vLLM image as qurator-nemotron with its own entrypoint: start vLLM on the
model channel (BF16, tensor-parallel 4, 32K context, 64 sequences, the model's reasoning parser),
wait for it, then run harness/tools/run_model.py against it with thinking off. outputs.csv, calls.csv,
run.json and the vLLM log end up in the job's model.tar.gz. Base and fine-tuned run identically.

--inputs is the input table to evaluate (the test set's inputs.csv). With --sample it instead draws
--n answers per training batch of --train-inputs (the training set's inputs.csv) at --temperature,
for rejection-sampling fine-tuning (4 at 0.8 for v2, 6 at 0.9 for v4-v6).

--tp, --max-model-len and --per-file serve an NVFP4 export the way production does (one GPU,
64K context) and send one document per call (whole PDFs).

--task and --batch-size select the task (harness/tools/tasks.py); the job's code channel carries
the harness tools plus that task's prompt, examples and schema.

Usage: python3 eval_job.py <name-suffix> <s3-model-prefix> --inputs inputs.csv [--spot] [--region us-east-2]
                           [--max-hours H] [--tp 4] [--max-model-len 32768] [--per-file] [--task NAME] [--batch-size N]
       python3 eval_job.py <name-suffix> <s3-model-prefix> --sample --train-inputs inputs.csv [--n 4] [--temperature 0.8]
"""
import argparse
import sys
from pathlib import Path

import boto3

from config import ACCOUNT, PREFIX, ROLE, BASE_TAGS
HARNESS = Path(__file__).resolve().parent.parent / "harness"
sys.path.insert(0, str(HARNESS / "tools"))
import tasks  # noqa: E402

TOOLS = ["tools/run_model.py", "tools/tasks.py", "tools/model_rates.json"]
EVAL_CMD = """cd $C && python3 tools/run_model.py eligible_inputs.csv $O/run --model lightning --openai-url http://localhost:8000/v1 \\
  --no-thinking --workers 64 --max-tokens 4096 --task __TASK____BATCH____PER_FILE__"""
SAMPLE_CMD = """cd $C && python3 sample_rft.py train_inputs.csv $O/samples.jsonl --n __N__ --temperature __T__ --workers 64 \\
  --task __TASK____BATCH__"""


def code_files(task_name):
    """Harness files the job needs: the tools, and the task's prompt, examples (if written) and schema."""
    task = tasks.get(task_name, HARNESS)
    missing = task.missing()
    if missing:
        raise SystemExit(f"task {task_name}: missing {', '.join(missing)}")
    files = [f"prompts/{task.prompt}", f"schemas/{task.output_schema}"]
    if task.examples and (HARNESS / "prompts" / task.examples).exists():
        files.append(f"prompts/{task.examples}")
    return TOOLS + files


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
    ap.add_argument("--inputs", help="input table to evaluate (required unless --sample)")
    ap.add_argument("--train-inputs", help="--sample: the training set's input table")
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel size (1 for an NVFP4 export, as production)")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--per-file", action="store_true", help="one row per call (whole documents)")
    ap.add_argument("--task", default="file-summary", choices=sorted(tasks.TASKS))
    ap.add_argument("--batch-size", type=int, help="abstract-facts: rows per call (default 10)")
    a = ap.parse_args()
    if a.sample and not a.train_inputs:
        ap.error("--sample needs --train-inputs")
    if not a.sample and not a.inputs:
        ap.error("--inputs is required")

    name = f"{PREFIX}-ingest-eval-{a.suffix}"
    bucket = f"sagemaker-{a.region}-{ACCOUNT}"
    code_prefix = f"{PREFIX}/ingest/{name}/code/"
    s3 = boto3.client("s3", region_name=a.region)
    for rel in code_files(a.task):
        s3.upload_file(str(HARNESS / rel), bucket, code_prefix + rel)
    if a.sample:
        s3.upload_file(a.train_inputs, bucket, code_prefix + "train_inputs.csv")
        s3.upload_file(str(Path(__file__).resolve().parent / "sample_rft.py"), bucket, code_prefix + "sample_rft.py")
    else:
        s3.upload_file(a.inputs, bucket, code_prefix + "eligible_inputs.csv")
    batch = f" --batch-size {a.batch_size}" if a.batch_size else ""
    command = (SAMPLE_CMD.replace("__N__", str(a.n)).replace("__T__", str(a.temperature)) if a.sample else
               EVAL_CMD.replace("__PER_FILE__", " --per-file" if a.per_file else "")).replace("__TASK__", a.task).replace("__BATCH__", batch)
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
