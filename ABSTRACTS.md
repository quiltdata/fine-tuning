# Abstract key facts

How we post-trained NVIDIA Nemotron 3.5 Lightning (30B-A3B, open weights) to read a
scientific abstract and state what the study is and its one or two main findings, how
we judged it, and what the result does and doesn't support. The data, every run and
every grader verdict are in the public package
`quilt+s3://quilt-ai#package=models/nemotron-lightning`. The method follows
[METHODOLOGY.md](METHODOLOGY.md); this page covers what is specific to abstracts.

## The result, honestly

- **It misses the bar.** The bar, fixed before training, was Claude Haiku 5.5 on every
  measure under both graders. The best fine-tune (RFT v2, BF16) conveys
  **87.4% / 86.4%** of the key's findings against Haiku's 90.9% / 89.6%, with
  **90.9% / 86.3%** faithfulness against Haiku's 94.9% / 94.6%. The recall gap is
  borderline (each 95% interval just reaches zero); the faithfulness gap is not.
- **Training moved it a long way.** Stock Lightning states 83% of findings but only
  76% / 67% faithfully, and invents a finding for 13 of 19 abstracts that report none.
  Supervised training and two reward rounds raised faithfulness by 15–20 points and
  recall by 3–4, and every answer is now valid.
- **No prompt we could write did the same.** Tuning the prompt for Lightning on a dev
  set made it more cautious and cost recall. Turning thinking on helped faithfulness,
  at about 45 times the output tokens; the fine-tune without thinking now matches or beats
  stock Lightning with thinking on, at about 2% of its output tokens.
- **Invented findings are not fixed.** The fine-tune still states a finding for 9 to 13
  of the 19 abstracts with none, depending on the run (Haiku: 4). A preference round
  took that to 0 but made everything else worse (§8).
- **Most of the gap is the student's, not the teacher's.** Run as it made the training
  targets, Kimi K3 beats Haiku on recall and ties it on faithfulness under GPT; only
  under Sonnet is it 2.6 points less faithful, which caps that one column (§8).
- **Served the way it would ship, it costs less than half of Haiku.** Exported to 4-bit
  NVFP4 and served on one L40S GPU, RFT v2 costs a measured **$0.0146 per 100
  abstracts** against Haiku's $0.034, and keeps its quality within noise of BF16:
  **85.3% / 85.7%** recall and **89.8% / 88.4%** faithfulness. Served, it is behind
  Haiku on both measures under both graders (recall −5.7 [−9.1, −1.9] / −4.0
  [−7.6, −0.2], faithfulness −5.1 [−7.8, −2.7] / −6.1 [−8.7, −3.6]).

## 1. Task

- **Input:** one paper's own main abstract, as plain text of at most 4,096 UTF-8 bytes.
  One abstract per call. The instructions and examples are the same on every call, so
  they are cached: a cache point on Bedrock, prefix caching on vLLM.
- **Answer:** one JSON object: one `about` fact (the kind of study, its system, its
  aim) and the one or two most important `findings`, or none when the abstract reports
  no results (a study protocol, say). Each fact in the model's own words, with an
  optional verbatim quote.
- **Thinking off** for every model compared, unless a row says otherwise.

An earlier form sent five numbered abstracts per call to save the repeated
instructions. It made the task harder in a way production doesn't need: the first
fine-tune lost 62 of 312 abstracts to answer numbering, not to summarizing. Caching
removed the reason to batch, so everything was re-measured one abstract per call.

## 2. Data

- **Source:** PubMed Central open-access articles, CC BY or CC0 only.
- **Test set:** 312 abstracts, fixed before any training. Six that Claude Opus 5.5's
  content filter blocked are excluded for every model, leaving **306 scored**. Opus
  blocked 16 more in the one-per-call form; its numbers are on the 290 it answered,
  and every claim below also holds on those 290.
- **Training set:** 1,154 abstracts from different articles.
- **Dev sets** for prompt work: 39 abstracts and 6 study protocols, separate from both.

## 3. Teacher, and what may touch training

As in [METHODOLOGY.md §3](METHODOLOGY.md#3-teacher-and-what-may-touch-training):
**Kimi K3 is the teacher and the reward-round grader; no Claude or OpenAI output touches
training.** The prompt and examples the student sees were drafted with Claude; the
student learns to write only from Kimi's answers and its own.

- **Teacher targets:** Kimi K3 answered every training abstract with the same prompt.
  1,145 of 1,154 passed the code check (valid structure, at most two findings, quotes
  found in the abstract).
- **Teacher quality:** on a 20-abstract probe, Kimi graded its own targets at 100%
  recall and 98.3% faithfulness against the training key, so the student had a
  near-perfect teacher to fall short of.

## 4. Evaluation, fixed before training

- **Answer key:** per test abstract, an about fact and up to two findings with
  verbatim quotes, written by GPT-6.1 sol. A second model checked a sample of it.
- **Two graders,** GPT-6.1 sol and Claude Sonnet 5.5, each judge **recall** (does the
  answer convey each key fact?) and **faithfulness** (is each stated fact supported
  by the abstract, or overstated, or unsupported?). A claim must hold under both.
- **Invented findings:** of the 19 abstracts whose key has no findings, how many got
  one anyway.
- **The bar:** at or above Haiku 5.5 on every measure, under both graders; Opus 5.5
  is the stretch. Differences under about two points count as ties.
- **Gaps** between models are paired bootstraps over abstracts, with 95% intervals.

## 5. A fair prompt for Lightning

The published prompt was tuned on Claude. Before reading a fine-tune against stock
Lightning, we tuned the prompt for Lightning too, on the dev sets only: nine variants
(reordered steps, a faithfulness section, a checklist built from its own errors, more
worked examples, quote-before-fact).

- **Every change cost recall,** 6 to 24 points. More rules made Lightning say less.
- **Quoting first** (the model copies its support before restating it) hurt most.
- **Thinking was the one lever that worked:** 5–8 points of faithfulness and no
  invented findings on dev, recall unchanged, at about 40 times the tokens.

So the stock baseline uses the published prompt, and stock-with-thinking is reported
as the best-effort stock arm at its real token cost.

## 6. Training recipe

LoRA on the BF16 base, as in [METHODOLOGY.md §5](METHODOLOGY.md#5-training-recipe):
rank 32, alpha 64, loss on the answer only, SageMaker Spot on one ml.g6e.12xlarge.

| Round | What it trained on | Findings recall | Faithfulness | Invented |
| --- | --- | --- | --- | --- |
| **SFT v2** | Kimi's 1,145 code-filtered answers, from the base model | 82.8 / 82.8 | 85.5 / 81.5 | 9 |
| **RFT v1** | Six samples per training abstract from SFT v2 at temperature 0.9. Kimi graded up to three code-valid samples per abstract: recall against the training key, then faithfulness against the abstract. The first sample perfect on both became the target; otherwise the teacher's answer stayed. Grading was capped at $20, so only 569 abstracts were graded (524 own answers). One epoch from SFT v2, lr 5e-5. | 85.1 / 84.7 | 87.6 / 83.7 | 7 |
| **RFT v2** | The same selection with every abstract graded: 1,058 own answers, 87 teacher's. One epoch from RFT v1. | **87.4 / 86.4** | **90.9 / 86.3** | 9 |

(SFT v1 trained on the five-per-call form and is superseded; see §1.)

RFT v2 against SFT v2: recall +4.5 [+0.4, +8.5] / +3.6 [−0.2, +7.3], faithfulness
+5.4 [+2.6, +8.2] / +4.8 [+1.6, +7.9]. Against RFT v1 each step was about +2, within
noise on its own. Output length didn't change (about 225 tokens per abstract).

## 7. Results

306 test abstracts, one per call, GPT-6.1 sol / Sonnet 5.5 graders.

| Model | Valid | Findings recall | Faithfulness | Invented (of 19) | Output tokens | $ per 100 abstracts |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 (stretch) | 290 | 91.3 / 91.1 | 96.5 / 95.5 | 4 | 394 | $1.22 |
| **Claude Haiku 5.5 (bar)** | 306 | **90.9 / 89.6** | **94.9 / 94.6** | 4 | 425 | $0.034 |
| **Lightning RFT v2, BF16** | 306 | **87.4 / 86.4** | **90.9 / 86.3** | 9 | 229 | — (not a deployment) |
| **Lightning RFT v2, NVFP4, 1 GPU (served)** | 306 | **85.3 / 85.7** | **89.8 / 88.4** | 11 | 223 | **$0.0146** measured |
| Lightning RFT v2, NVFP4, 1 GPU (Spot evaluation job, same model) | 306 | 86.6 / 84.2 | 89.7 / 86.4 | 13 | 223 | — |
| Lightning stock, thinking on | 306 | 83.2 / 81.7 | 88.0 / 82.4 | 1 | 9,397 | — |
| Lightning stock, NVFP4 | 293 | 83.0 / 83.4 | 75.9 / 66.7 | 13 | 200 | — |
| Lightning v5 (the file-summary fine-tune), NVFP4 | 288 | 82.3 / 82.5 | 79.5 / 74.0 | 10 | 265 | — |

Claude costs are on-demand Bedrock prices with the instructions cached. The served
cost is the endpoint's hourly price over its sustained throughput (17,800 abstracts an
hour at 64 concurrent requests); idle, the endpoint still bills about $63 a day.
Earlier served measurements in the five-per-call form: stock $0.032, v5 $0.013.

The two NVFP4 runs of the same model agree within noise; invented findings, on only
19 abstracts, move between 9 and 13 across the three runs of RFT v2. Quantizing to NVFP4 cost −2.1 [−5.6, +1.4] / −0.8 [−4.0, +2.6] recall and
−1.1 [−3.7, +1.4] / +2.2 [−0.5, +5.0] faithfulness: within noise. Served against
stock Lightning, faithfulness is up 13.9 / 21.8 points (intervals well clear of zero)
and recall +2.3 / +2.3 (within noise). RFT v2 (BF16) against
Haiku: recall −3.6 [−7.3, 0.0] / −3.2 [−6.6, +0.4], faithfulness −4.1 [−6.4, −1.8] /
−8.3 [−11.1, −5.6].

## 8. The teacher's ceiling, and a preference round

**The teacher on the test set.** Kimi K3 wrote its own key for each test abstract and
answered with it in view, exactly as it made the training targets (run
`kimi-k3-teacher-single`): **95.8 / 93.6** recall, **95.5 / 92.0** faithfulness, 6
invented, 304 valid. Against Haiku: recall +4.9 [+1.9, +8.0] / +4.0 [+0.7, +7.3],
faithfulness +0.5 [−1.2, +2.4] / −2.6 [−4.8, −0.4]. So the student's recall gap and its
GPT faithfulness gap are its own. About 2.6 of its −8.3 Sonnet faithfulness gap, and
about 2 of its 5 extra invented findings, come from the teacher; imitating Kimi more
closely can't close that column.

**A preference (DPO) round from RFT v2.** 580 pairs, one epoch, β 0.1, plus the
answer-only loss on the chosen answer (`train.py --dpo_beta`):

- 453 pairs from the reward rounds' grading: chosen, the RFT v2 target; rejected, a
  sample Kimi judged as missing a finding or unfaithful.
- 127 pairs on 71 new PMC abstracts whose Kimi key has no findings: chosen, Kimi's empty
  answer; rejected, an RFT v2 sample that states a finding.

| Model | Findings recall | Faithfulness | Invented (of 19) | Empty answers where the key has findings |
| --- | --- | --- | --- | --- |
| RFT v2, BF16 | 87.4 / 86.4 | 90.9 / 86.3 | 9 | 5 |
| DPO from RFT v2, BF16 | 81.9 / 81.3 | 84.0 / 77.6 | **0** | 38 |

It trained as intended (preference accuracy from about 0.2 to 0.8, the chosen answers'
loss steady) and fixed invented findings, but lost recall −5.5 [−9.9, −1.1] /
−5.1 [−9.3, −0.9] and faithfulness −6.9 [−9.7, −3.8] / −8.6 [−11.9, −5.5] against RFT v2:

- **It overcorrected to "no findings":** empty answers for 38 abstracts that have
  findings. The no-findings pairs were 22% of the round; such abstracts are 6% of the
  test set.
- **Its findings got less faithful** (supported findings 89.6% → 80.4% / 83.8% → 71.6%).
  It packs both findings into the first fact and fills the second with an
  interpretation ("consistent with …", "suggesting …") that the graders call
  overstated. The likely cause is the "missed a finding" pairs, which reward covering
  more, and rejected samples from an earlier model, which are easy to push down without
  learning what made them wrong.

RFT v2 stays the model. A second round would draw rejected samples from RFT v2 itself,
keep only unfaithfulness pairs, and hold no-findings pairs near 6%.

## What we learned

- **Batching was a false economy.** Caching the shared instructions makes one abstract
  per call cheap, and the batched form cost the model answers it could otherwise give.
- **A reward round is only as big as its grading.** The capped first round graded half
  the training set and gained about 2 points; grading all of it and training once more
  doubled the gain over SFT.
- **Faithfulness is where the gap is.** The graders judge 5.2% / 12.2% of the
  fine-tune's stated facts overstated (Haiku: 3.1% / 5.3%) and 3.9% / 1.5% unsupported
  (Haiku: 1.9% / 0.1%), and it still states findings for abstracts that report none.
- **Thinking buys faithfulness at a price.** It's the one prompt-side lever that
  worked for stock Lightning, and the fine-tune reaches the same level without it.

## Spend

About $40 for the baselines and the test and training sets, and about $195 for
training: teacher targets, three training rounds, sampling, open-weight grading,
evaluations with both graders, the NVFP4 export and the served measurement. The
teacher's ceiling and the preference round (§8) cost about $30 more.
