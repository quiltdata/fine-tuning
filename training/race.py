#!/usr/bin/env python3
"""Watch competing SageMaker training jobs: the first to get capacity wins and the rest are stopped.

Then follow the winner: print status changes and the training script's key log lines
(kernel install, data stats, load, probe, loss, errors) until it ends. One line per event,
for a Monitor. Usage: python3 race.py job:region [job:region ...]
"""
import re
import sys
import time

import boto3

KEY = re.compile(r'"kernel|kernels_import|"examples"|loaded_seconds|probe_loss|restored_tensors|"renamed"|sampled_batches|merged_probe_loss|merge_changed|merged_files|vllm ready|vllm did not start|output_rows|calls_ok|wall_clock|weights_max_abs_diff|lm_head_is_embeddings|greedy|trainable params|optimizer_steps|'
                 r"'loss'|\"loss\"|train_loss|done_seconds|Traceback|Error|error:|OOM|Killed|CUDA out|Spot")


def status(sm, job):
    d = sm.describe_training_job(TrainingJobName=job)
    return d["TrainingJobStatus"], d["SecondaryStatus"], (d.get("FailureReason") or "")[:300]


def main(specs):
    jobs = [(s.split(":")[0], s.split(":")[1]) for s in specs]
    sms = {r: boto3.client("sagemaker", region_name=r) for _, r in jobs}
    prev, winner = {}, None
    while not winner:
        for job, region in jobs:
            try:
                st = status(sms[region], job)
            except Exception as e:  # transient API errors: try again next round
                st = ("?", str(e)[:80], "")
            if prev.get(job) != st:
                print(f"STATUS {job} {st[0]} {st[1]} {st[2]}", flush=True)
                prev[job] = st
            if st[0] == "InProgress" and st[1] not in ("Starting", "Pending"):
                winner = (job, region)
                break
        if not winner:
            if all(prev[j][0] in ("Failed", "Stopped", "Completed") for j, _ in jobs):
                print("ALL ENDED without capacity", flush=True)
                return
            time.sleep(60)
    print(f"WINNER {winner[0]}", flush=True)
    for job, region in jobs:
        if (job, region) != winner and prev.get(job, ("InProgress",))[0] == "InProgress":  # unpolled jobs are stopped too
            sms[region].stop_training_job(TrainingJobName=job)
            print(f"STOPPED {job}", flush=True)
    job, region = winner
    logs = boto3.client("logs", region_name=region)
    seen, last = set(), None
    while True:
        st = status(sms[region], job)
        if st != last:
            print(f"STATUS {job} {st[0]} {st[1]} {st[2]}", flush=True)
            last = st
        try:
            for ev in logs.filter_log_events(logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=job)["events"]:
                if ev["eventId"] not in seen:
                    seen.add(ev["eventId"])
                    if KEY.search(ev["message"]):
                        print(ev["message"][:300], flush=True)
        except Exception:
            pass
        if st[0] in ("Completed", "Failed", "Stopped"):
            print(f"FINAL {job} {st[0]} {st[2]}", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main(sys.argv[1:])
