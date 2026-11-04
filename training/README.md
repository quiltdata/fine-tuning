# training: post-training Nemotron 3.5 Lightning as a key-fact summarizer

The code for every round in [`../METHODOLOGY.md`](../METHODOLOGY.md). Data preparation, model runs and scoring use [`../harness/`](../harness/); the datasets themselves are not in this repo (`$DATA_DIR` below).

Everything runs as SageMaker training jobs under the account, execution role and job
prefix set in [`config.py`](config.py) from the environment (see Configuration below).
Job names must match `<SAGEMAKER_JOB_PREFIX>-*` (the role's policy), and inputs sit under
`s3://sagemaker-<region>-<account>/<job-prefix>/`. The instance is ml.g6e.12xlarge
(4× L40S); quota is 1 on-demand and 1 Spot per region.

## Configuration

The account-specific values (account id, execution role name, job prefix, region,
instance) are **not** hardcoded; they load from the environment via
[`config.py`](config.py). Set them in a git-ignored `.env` at the repo root:

```bash
cp .env.example .env              # then fill it in, or:
./make-env.sh /path/to/source.env # copy one kept outside the repo
```

`.env` is never committed. [`.env.example`](../.env.example) documents the variables.
Real shell environment variables override `.env`.

| File | What it does |
| --- | --- |
| `source/train.py` | The job's entry point. It installs the Mamba kernels (prebuilt CUDA 13 wheels), loads the BF16 base, restores the 46 tensors transformers 5 re-initializes, and checks a probe loss. Then LoRA SFT on answer tokens only, with Spot-resumable checkpoints; `--merge` merges and saves a vLLM-loadable folder. Also `--merge_only` (merge an adapter), `--init_adapter` (continue from one) and `--diagnose` (check weights, kernel paths and generation, then exit). |
| `source/requirements.txt` | `peft` (the kernels are installed by `train.py`) |
| `launch.py` | Uploads `source/` and a train.jsonl, and creates a job. Options: `--spot`, `--region`, `--merge`, `--merge-adapter`, `--init-adapter`, `--diagnose`. |
| `race.py` | Watches competing jobs in different capacity pools, stops the losers once one starts, and streams the winner's key log lines |
| `eval_job.py` | Evaluates a BF16 checkpoint on the 262 test rows inside a training job: vLLM's image, BF16 tensor-parallel 4, thinking off, `harness/tools/run_model.py`. `--inputs` is the table to evaluate; with `--sample`, `--train-inputs`. `--sample [--n N --temperature T]` draws answers per training batch instead. |
| `sample_rft.py` | The sampling client `eval_job.py --sample` runs |
| `select_rft.py` | Picks reward-stage targets. Default (v2): the quote, schema and length check, then closest to the teacher. `--facts --grades` (v4–v6): among candidates passing the check (samples plus the teacher's answer), the highest top-2 fact recall as graded by Kimi K3, verdicts cached |
| `source/export_nvfp4.py` | Exports a merged BF16 checkpoint to NVFP4 with ModelOpt 0.47 and NVIDIA's Lightning recipe (`w4a16_nvfp4_4o6`), calibrating on summarizer prompts, then checks the result against production's NVFP4 artifact. Two config fixes are needed after export; see below. |
| `export_job.py` | Launches `export_nvfp4.py` as a SageMaker training job (same image and role as `launch.py`). The export lands in the job's checkpoint prefix under `nvfp4/`. |
| `deploy_nvfp4.py` | A temporary single-L40S endpoint in production's shape (`lightning_container.json`, max_num_seqs 64) for measuring an NVFP4 export's cost and score. Pass `2xlarge` as a fourth argument when xlarge has no capacity. |
| `verdict-v5-nvfp4-25c.json` | `compare_ft.py --bar 25c` for v5 served in NVFP4 |
| `deploy_eval.py` | The same evaluation as a SageMaker endpoint. Not used in the end: on-demand capacity never came, so evaluation ran inside training jobs (`eval_job.py`). |
| `lightning_container.json` | `qurator-nemotron`'s container settings, which `deploy_eval.py` copies |
| `compare_ft.py` | Judges a fine-tuned run against the 25a bar, or with `--bar 25c` against the top-2 bar (top-2 recall within 3 of Opus, both graders), with paired bootstrap gaps |
| `verdict-v1.json`, `verdict-v2.json` | Its output for SFT v1 and v2 (25a bar) |
| `verdict-v3-25c.json` … `verdict-v6-25c.json` | Its output for v3 to v6 (25c bar); v5 is the second fact-graded reward round, v6 the third. v4 to v6 regenerated under the clarified quote check, under which v5 passes |

## Reproduce

```bash
export DATA_DIR=/path/to/data                   # holds sets/pmc-v1 (test) and sets/pmc-v2 (train)
TEST=/path/to/test_inputs.csv                   # the 262 eligible test rows of sets/pmc-v1
H=../harness/tools
B=s3://jumpstart-cache-prod-us-east-2/huggingface-reasoning/huggingface-reasoning-nemotron-3-5-lightning-30b-a3b-bf16/artifacts/inference/v1.0.0/

# SFT v1 on the Kimi K3 training set
python3 launch.py v1 $DATA_DIR/sets/pmc-v2/runs/kimi-k3/train.jsonl --epochs 2 --batch 2 --grad-accum 8 --merge --spot --max-hours 10
# (merge on its own if needed:  python3 launch.py merge-v1 <any.jsonl> --max-examples 2 --merge-adapter <s3 adapter prefix> --spot)

# evaluate base and fine-tune the same way, then score with the harness
python3 eval_job.py base $B --inputs $TEST --spot
python3 eval_job.py ft-v1 <s3 merged prefix> --inputs $TEST --spot

# reward stage: sample, select by code, continue training
python3 eval_job.py rft-sample <s3 merged v1 prefix> --sample --train-inputs $DATA_DIR/sets/pmc-v2/inputs.csv --spot --max-hours 4
python3 select_rft.py $DATA_DIR/sets/pmc-v2/inputs.csv samples.jsonl \
    $DATA_DIR/sets/pmc-v2/runs/kimi-k3/outputs.csv $DATA_DIR/sets/pmc-v2/runs/kimi-k3/train.jsonl train_v2.jsonl
python3 launch.py v2 train_v2.jsonl --epochs 1 --lr 5e-5 --batch 2 --grad-accum 8 --merge --init-adapter <s3 v1 adapter prefix> --spot

# judge against the 25a bar (after scoring the runs with the harness)
python3 compare_ft.py $DATA_DIR/sets/pmc-v1 lightning-ft-v2

# top-2 rounds (25c bar): two-fact key, fact-aware teacher targets, SFT v3 from the v2 adapter
python3 $H/make_facts.py $DATA_DIR/sets/pmc-v2/inputs.csv none $DATA_DIR/sets/pmc-v2/reference \
    --model us.moonshotai.kimi-k3 --chat --reasoning-effort low --prompt facts_top2.md --max-facts 2 --workers 24
python3 $H/run_model.py $DATA_DIR/sets/pmc-v2/inputs.csv $DATA_DIR/sets/pmc-v2/runs/kimi-k3-facts \
    --model us.moonshotai.kimi-k3 --chat --service-tier flex --reasoning-effort high --workers 24 \
    --required-facts $DATA_DIR/sets/pmc-v2/reference/facts.jsonl
# then check_outputs / make_pairs / make_sft as for pmc-v2 -> train_v3.jsonl
python3 launch.py v3 train_v3.jsonl --epochs 1 --lr 5e-5 --batch 2 --grad-accum 8 --merge --init-adapter <s3 v2 adapter prefix> --spot
python3 eval_job.py ft-v3 <s3 merged v3 prefix> --inputs $TEST --spot
python3 eval_job.py rft-sample-v3 <s3 merged v3 prefix> --sample --train-inputs $DATA_DIR/sets/pmc-v2/inputs.csv --spot --n 6 --temperature 0.9 --max-hours 5
python3 select_rft.py $DATA_DIR/sets/pmc-v2/inputs.csv samples.jsonl \
    $DATA_DIR/sets/pmc-v2/runs/kimi-k3-facts/outputs.csv $DATA_DIR/sets/pmc-v2/runs/kimi-k3-facts/train.jsonl train_v4.jsonl \
    --facts $DATA_DIR/sets/pmc-v2/reference/facts.jsonl --grades grades.jsonl
python3 launch.py v4 train_v4.jsonl --epochs 1 --lr 5e-5 --batch 2 --grad-accum 8 --merge --init-adapter <s3 v3 adapter prefix> --spot
python3 eval_job.py ft-v4 <s3 merged v4 prefix> --inputs $TEST --spot
python3 compare_ft.py $DATA_DIR/sets/pmc-v1 lightning-ft-v4 --bar 25c
```

`race.py job:region …` follows a set of jobs launched in several pools (Spot and on-demand, us-east-1 and us-east-2), so whichever gets capacity first runs.

## After export: two config fixes

ModelOpt 0.47's export needs two corrections before vLLM serves it like NVIDIA's own
NVFP4 Lightning. Keep the files as exported alongside (e.g. `modelopt-raw/`).

1. **Leaked placeholder.** ModelOpt writes an internal name-mapper placeholder
   (`\x00backbone.pt_name_sentinel`) into every module name in `config.json` and
   `hf_quant_config.json`. Remove it. The weights' own names are unaffected.
2. **Unquantized-module list.** The export lists only `mtp*` as unquantized, where
   NVIDIA's artifact lists 72 modules. Both leave exactly the same module families
   unquantized (check tensor by tensor; no listed module carries a quantization
   scale), so use the reference artifact's list.

After both, the quantization sections match the reference artifact except for the
producer version.
