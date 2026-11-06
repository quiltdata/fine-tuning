# Methodology

How we post-trained NVIDIA Nemotron 3.5 Lightning (30B-A3B, open weights) to state
the key facts of a scientific file, how we judged it, and what the result does and
doesn't support. The code is in [`training/`](training/).

## The result, honestly

- **Level with Claude Opus 5.5 on excerpts, at about 1% of the cost.** Served the way
  it would ship (4-bit NVFP4, one L40S GPU), the fine-tune conveys a file's two
  top-ranked key facts 82.4% / 81.5% of the time (two graders) against Opus's
  80.9% / 78.2%. The paired gaps, +1.5 [−1.9, 4.9] and +3.2 [−0.4, 6.9], straddle
  zero: **level, not ahead**. Measured serving cost is **$0.030 per 100 rows**, about
  **1.2%** of Opus's $2.39. This holds for the first 8,000 characters of a file.
- **The served model misses our own quoting bar.** Each fact must carry a verbatim
  quote from the input; the bar was 85% of rows. The served model passes on 219 of
  262 (84%), four rows short. In BF16 (four GPUs, not a deployment) it passes.
- **On whole papers, Opus is ahead.** Given a fair prompt, Opus states a whole paper's
  two key facts about 10 points more often. The fine-tune never trained on whole papers.
- **One headline was withdrawn.** An earlier "Lightning beats Opus on papers" came from
  comparing a model trained on the target against one never told it. See
  [Corrections](#corrections-that-changed-the-conclusion).
- **Total spend:** about $353 of a $500 budget, across six training rounds, the export,
  and every evaluation.

## 1. Task

Given a file's text, write a short summary plus the facts a reader needs to decide
whether to open it, each with a verbatim evidence quote from the input. Output is
strict JSON in a fixed schema, many rows per request. The model runs with
**thinking off**, so the cost is the answer alone.

What counts as a key fact was drawn at **two**: what the file is and what it's about,
then its main point or finding. A reference key gave 3–5 ranked facts per row, but
facts 3–5 were one model's opinion of what came next; a summarizer that picks a
different fourth fact isn't worse but would score as if it were. So the headline
metric is **top-2 key-fact recall**.

## 2. Data

- **Source:** PubMed Central open-access articles, **CC BY or CC0 only**, not retracted.
  Text, PDF, Word and Excel files, cut to 8,000-character excerpts.
- **Test set:** 262 rows from 97 articles (each paper plus its supplements), fixed
  before any training. A single score's 95% interval is about ±3 points.
- **Training set:** 2,928 rows from 1,166 *different* articles. Every test article's
  PMCID is excluded at sampling time.
- **Scope narrowed later:** supplements were dropped (a supplement is covered by its
  paper's summary). Because every round reported results by file type and resampled
  by article, paper-only numbers needed no re-runs.

## 3. Teacher, and what may touch training

- **Kimi K3 is the teacher** (via Amazon Bedrock), level with Opus within noise on a
  48-row comparison. Its licence restricts the software, not its outputs.
- **No Claude or OpenAI output touches training**, including selecting training data.
  Anthropic's terms prohibit using outputs to train models without authorization;
  OpenAI's were assumed similar. Claude serves **only** as an evaluation baseline and
  a grader.
- **The teacher thinks; the student doesn't.** Kimi ran at high reasoning effort; the
  student learns the answer alone.
- **Training data is filtered by code only:** valid schema, every row returned, every
  evidence quote found in the input, summary within length. No LLM picks SFT data.
  1,955 of the teacher's 2,928 rows survived.

## 4. Evaluation, fixed before training

The bar for each round was written down **before** that round trained, so the result
couldn't be argued after the fact.

- **Answer key:** a fixed set of ranked facts per test row, written by a third model
  (GPT-6.1 sol) from the same input, with a verbatim quote per fact.
- **Two graders:** GPT-6.1 sol and Claude Sonnet 5.5 each judge whether a row conveys
  each key fact. A claim must hold under **both**.
- **Paired comparisons:** every gap between two models is a paired bootstrap,
  **resampled by article**, reported with its 95% interval. "Level" means the interval
  includes zero.
- **Mechanical checks, separate from meaning:** schema valid, ≥ 99% of rows returned,
  ≤ 800 output tokens per row, and the quote check (see below). These are code, not
  graders.
- **Same serving for every model compared:** base and fine-tunes run identically in
  vLLM, thinking off, 64 concurrent requests.

Two bars were used, named for the plans that fixed them; `compare_ft.py --bar` takes
either.

| Bar | Pass criteria (all 262 test rows, both graders) |
| --- | --- |
| **25a** (round 1) | ≥ 7 points above base; within 3 of the better of Haiku 5.5 and GPT-6 luna; ≥ 85% quote check; ≥ 99% rows returned; ≤ 800 tokens/row; ≤ $0.05 per 100 rows. Fail = not reached after one SFT and one reward round, then stop. |
| **25c** (top-2 rounds) | Top-2 recall within 3 of Opus 5.5 (≥ 77.9 GPT grader, ≥ 75.2 Sonnet); ≥ 85% quote check; ≥ 99% rows; ≤ 800 tokens/row. |

## 5. Training recipe

All rounds are LoRA on the BF16 base, run as SageMaker training jobs on one
ml.g6e.12xlarge (4× L40S), on Spot.

- **LoRA:** rank 32, alpha 64, dropout 0.05, on attention (q/k/v/o), the Mamba
  in/out projections and the shared-expert MLP. Routed experts frozen. 28.7M
  trainable parameters.
- **Format:** chat template, thinking off, **loss on the assistant's JSON only**.
- **Continue, don't restart:** each round starts from the previous adapter, one epoch
  at learning rate 5e-5.

| Round | What it trained on | Outcome |
| --- | --- | --- |
| **v1** — SFT | The teacher's code-filtered answers. 2 epochs, effective batch 16, lr 1e-4. | Large gain over base; missed the 25a quote line. |
| **v2** — reward round | v1 sampled 4 answers per training batch; code kept those passing the quote and schema check, closest to the teacher's wording. | **Passed 25a.** Bought quoting at the cost of fact coverage. |
| **v3** — fact-aware SFT | Kimi wrote two facts per training row, then re-answered each batch *with those facts in view*. The student saw only the plain prompt. | Closed most of the gap to Opus on top-2 recall. |
| **v4** — fact-graded reward | v3 sampled 6 per batch at temperature 0.9; candidates passing the code check (plus the teacher's) were graded by **Kimi** against the two facts; highest top-2 recall won. | Level with Opus on facts; missed the quote line (74%). |
| **v5** — same, from v4 | Same selection. | **Ahead of Opus in BF16** on facts; passes 25c under the clarified quote rule. **The checkpoint.** |
| **v6** — same, from v5 | Same selection. | Within noise of v5; quoting plateaued. **Stopped here.** |

Two lessons from the rounds:

- **The gap to Opus was the target, not capacity.** Nothing trained fact coverage
  until v3; one epoch on fact-aware targets closed most of it.
- **Quoting and fact coverage traded off** when selection favored the teacher's
  wording (v2). Selecting on the metric (v4–v6) moved both together, until quoting
  stopped responding to more rounds. We stopped rather than keep training against
  noise (about ±5 rows between equally trained checkpoints).

## 6. Quantize and serve the way production does

A model's cost only counts if it's measured in the form that ships.

- **Export** (`source/export_nvfp4.py`): NVIDIA ModelOpt 0.47 with NVIDIA's own recipe
  for this model (`w4a16_nvfp4_4o6`), calibrated on 64 of the summarizer's own
  training prompts. Experts and lm_head become weight-only NVFP4; Mamba projections
  FP8; attention, routers and embeddings stay BF16; FP8 KV cache.
- **Checked against production's artifact:** module families, dtypes and shapes match
  NVIDIA's published NVFP4 Lightning exactly. Two config fixes were needed after
  export (a leaked name placeholder, and the unquantized-module list); see
  [`training/README.md`](training/README.md#after-export-two-config-fixes).
- **Served** (`deploy_nvfp4.py`) on a temporary single-L40S endpoint with production's
  container settings, 64 concurrent requests, then scored like every other run.
- **Quantizing cost** about 3 points of top-2 recall under one grader
  (−2.9 [−5.9, +0.2]) and nothing measurable under the other (−0.4). That's why the
  served model is level with Opus rather than ahead.

## 7. Results

All 262 test rows, 8,000-character excerpts, thinking off. Top-2 recall under
GPT-6.1 sol / Sonnet 5.5 graders. Quote check under the 25h rule.

| Model | Top-2 recall | Quote check | Tokens / row | $ / 100 rows |
| --- | --- | --- | --- | --- |
| **Lightning v5, NVFP4, 1 GPU (served)** | **82.4 / 81.5** | 213 (81%) | 465 | **$0.030** measured |
| Lightning v5, BF16, 4 GPUs | 85.3 / 81.9 | 224 (85.5%) | 462 | — (not a deployment) |
| Claude Opus 5.5 | 80.9 / 78.2 | 257 (98%) | 559 | $2.39 |
| GPT-6 luna | 73.7 / 69.8 | 220 (84%) | 377 | $0.044 |
| Claude Haiku 5.5 | 66.6 / 64.1 | 251 (96%) | 415 | $0.064 |
| Lightning base, NVFP4 (production) | 62.4 / 59.2 | 67 (26%) | 404 | $0.027 measured |

Paired gaps for the served v5 (95% intervals, by article): against Opus **+1.5
[−1.9, 4.9] / +3.2 [−0.4, 6.9]**, level; against GPT-6 luna +8.8 / +11.6; against
Haiku +15.8 / +17.4; against its own production base +20.0 / +22.3. All three of the
latter exclude zero.

Under the final quote rule (§8), the served model passes on 219 of 262 (84%), still
below 85%, and BF16 v5 on 232. Every quote the model emits can be re-verified by
code before use.

**Cost caveat:** the per-row price assumes a workload big enough to keep a dedicated
endpoint busy. Idle, that endpoint still bills about $63–67 a day.

## Corrections that changed the conclusion

Each was recorded when it happened; the earlier numbers were kept, not overwritten.

1. **A fair prompt for the baseline.** On whole papers, the fine-tune first appeared to
   beat Opus. But it had been trained to state each paper's main finding, and Opus was
   never asked to. Adding one sentence to Opus's prompt ("for an article, say what kind
   of study it is and what it is about, and state its main finding") changed the
   result:

   | Top-2 recall, 97 papers | Excerpts | Whole papers |
   | --- | --- | --- |
   | Opus 5.5, original prompt | 73.7 / 69.6 | 72.7 / 67.5 |
   | **Opus 5.5, plus the main-finding sentence** | **87.6 / 80.4** | **92.8 / 89.7** |
   | Lightning v5, served | 83.5 / 82.5 | 82.5 / 80.4 |
   | Lightning minus prompted Opus | **level** (−4.1 [−9.8, 1.5] / +2.1 [−4.1, 8.2]) | **Opus ahead** (−10.3 [−14.9, −6.2] / −9.3 [−14.4, −4.1]) |

   "Beats Opus on papers" was withdrawn. **Use a fair, task-specific baseline prompt
   from the start.**

2. **The quote check was clarified, twice.** The first implementation compared quotes
   byte for byte; PDF extraction mangles typography. It was relaxed first to ignore
   whitespace and Unicode/typographic variants (25h), then to match a quote's letters
   and digits in order, ignoring case and punctuation. The intent never changed: a
   quote must point at text that's in the file. v5 BF16's pass on the 25c bar hinges
   on this by one row, which is within noise; we say so.

3. **Supplements out of scope.** A supplement is described by its paper's summary, not
   on its own. 60% of training rows had been supplements; the test claim was re-stated
   on the 97 papers.

## Carrying it forward

The next target is narrower: read a scientific **abstract** and extract one fact about
what the study is and one or two about its findings. What carries over from the work
above:

- Fix the target, the test split and the pass/fail bar **before** training.
- Compare stock, tuned, and a capable reference model **given a fair task-specific
  prompt**.
- Keep semantic grading (two graders, paired, by article) separate from mechanical
  checks (schema, quotes, length).
- Use an open-weight teacher and code-only filtering; select reward-round candidates
  on the metric with an open-weight grader.
- Measure quality and cost on the **deployed** form before claiming either.
- Record failures and corrections next to the results, not in place of them.

The abstract model's method and results are in [ABSTRACTS.md](ABSTRACTS.md).

## What's not in this repo

- **The datasets and answer keys.** [`harness/`](harness/) rebuilds them from PubMed
  Central's public S3 bucket, but re-sampling and re-keying won't reproduce our exact rows
  or facts.
- **Model weights.** The LoRA adapters and the NVFP4 export (~20 GB) are not in git.
