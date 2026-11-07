# Abstract key facts

How we post-trained NVIDIA Nemotron 3.5 Lightning (30B-A3B, open weights) to read a
scientific abstract and state what the study is and its one or two main findings, how
we judged it, and what the result does and doesn't support. The data, every run and
every grader verdict are in the public package
`quilt+s3://quilt-ai#package=models/nemotron-lightning`. The method follows
[METHODOLOGY.md](METHODOLOGY.md); this page covers what is specific to abstracts.

## The result, honestly

- **It misses the bar, by a measured amount.** The bar, fixed before training, was
  Claude Haiku 5.5. On 935 held-out abstracts (two test sets, §7), the best fine-tune
  (RFT v4, BF16) conveys **88.0%** of the key's findings against Haiku's 89.8%, a gap of
  −1.8 [−3.7, +0.1]: within the two-point rule, a tie. Its faithfulness is **91.8%**
  against Haiku's 95.0%, a gap of −3.2 [−4.4, −2.0]: the whole interval is outside the
  rule. Those are GPT-6.1's numbers, the primary grader (§4). Sonnet 5.5 grades the
  recall gap the same (−1.3) and the faithfulness gap wider (−6.4), and from RFT v4 on
  Sonnet is no longer an independent grader (§3).
- **Training moved it a long way, then stopped.** Stock Lightning states 83% of
  findings but only 76% / 67% faithfully. Supervised training and four reward rounds
  took faithfulness to 92% / 88% and recall to 88–89%, and every answer is valid. The
  last three rounds moved faithfulness by about two points in all and recall not at
  all; a fifth round moved neither. The second test set, built after the fact to check
  the first, reproduced every run within a point.
- **No prompt we could write did the same.** Tuning the prompt for Lightning on a dev
  set made it more cautious and cost recall. Turning thinking on helped faithfulness,
  at about 45 times the output tokens; the fine-tune without thinking now beats stock
  Lightning with thinking on, at about 2% of its output tokens.
- **Invented findings can be fixed, at a price.** The fine-tune states a finding for
  about half the abstracts whose key has none (Haiku: a fifth). Ninety no-findings
  training rows (RFT v5) brought that down to Haiku's rate, and cost about a point of
  recall; a preference round (§8) took it to zero and cost five.
- **Most of the gap is the student's, not the teacher's.** Run as it made the training
  targets, Kimi K3 beats Haiku on recall and ties it on faithfulness under GPT (§8).
- **Served the way it would ship, RFT v2 costs less than half of Haiku.** Exported to
  4-bit NVFP4 and served on one L40S GPU, RFT v2 costs a measured **$0.0146 per 100
  abstracts** against Haiku's $0.034, within noise of its BF16 quality. RFT v4 was not
  exported; its served numbers are not measured.

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
- **Test set 1:** 312 abstracts, fixed before any training. Six that Claude Opus 5.5's
  content filter blocked are excluded for every model, leaving **306 scored**.
- **Test set 2:** 682 abstracts from 750 further articles, none in any earlier set,
  built the same way after RFT v4 to check the first set on a larger sample. Opus's
  content filter blocked **52 of 682** (7.6%, against 1.9% on the first set; we don't
  know why), and Haiku's one more, leaving **629 scored**. Its answer key was written
  by the same model with the same instructions as the first set's.
- **Training set:** 1,154 abstracts from different articles, plus 90 abstracts whose
  key has no findings (used only in RFT v5).
- **Dev sets** for prompt work: 39 abstracts and 6 study protocols, separate from all.

## 3. Teacher, and what may touch training

As in [METHODOLOGY.md §3](METHODOLOGY.md#3-teacher-and-what-may-touch-training):
**Kimi K3 is the teacher and the reward-round grader for recall.** The prompt and
examples the student sees were drafted with Claude; the student learns to write only
from Kimi's answers and its own.

- **Teacher targets:** Kimi K3 answered every training abstract with the same prompt.
  1,145 of 1,154 passed the code check (valid structure, at most two findings, quotes
  found in the abstract).
- **Teacher quality:** on a 20-abstract probe, Kimi graded its own targets at 100%
  recall and 98.3% faithfulness against the training key, so the student had a
  near-perfect teacher to fall short of.
- **Through RFT v3, no Claude or OpenAI output touched training**, including the
  selection of training data. RFT v3 is the last model for which that holds.
- **From RFT v4, Claude Sonnet 5.5 judges faithfulness in target selection.** Kimi
  passes three quarters of the facts GPT-6.1 calls overstated (§6), so three rounds of
  Kimi-only selection could not move faithfulness. Sonnet catches most of them. It
  writes no targets: every target is still the student's own sample or Kimi's answer,
  and Sonnet only chooses among them. Two consequences: the claim "trained with
  open-weight models only" no longer holds past RFT v3, and Sonnet's own grades of RFT
  v4 and later are no longer an independent check, since Sonnet shaped what they
  learned. GPT-6.1 stayed outside training throughout and is the primary grader. Our
  reading of Anthropic's and AWS's terms is that selection is not using outputs as
  training targets; it is a reading, not legal advice.

## 4. Evaluation, fixed before training

- **Answer key:** per test abstract, an about fact and up to two findings with
  verbatim quotes, written by GPT-6.1 sol. A second model checked a sample of it.
- **Two graders,** GPT-6.1 sol and Claude Sonnet 5.5, each judge **recall** (does the
  answer convey each key fact?) and **faithfulness** (is each stated fact supported
  by the abstract, or overstated, or unsupported?).
- **Primary grader:** GPT-6.1. The bar was first set as "under both graders"; after a
  blind adjudication of the facts the graders disputed (six judges; Sonnet's solo
  flags were overturned most of the time, equally for every model), GPT-6.1 was made
  the primary grader and Sonnet is reported beside it. The decision and the
  adjudication are public in `quilt+s3://quilt-ai#package=evals/abstract-faithfulness-check`.
  Numbers below are **GPT / Sonnet**.
- **Invented findings:** of the abstracts whose key has no findings (19 in test set 1,
  55 in test set 2), how many got one anyway.
- **The bar:** at or above Haiku 5.5 on recall and faithfulness; Opus 5.5 is the
  stretch. Differences under about two points count as ties.
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
Every reward round: six samples per training abstract from the previous model at
temperature 0.9; a grader checks up to three or four code-valid samples per abstract
for recall against the training key and faithfulness against the abstract; the first
sample that passes both becomes the target, otherwise the teacher's answer stays, or
the abstract is dropped if the teacher's answer fails too. One epoch from the previous
model's adapter, lr 5e-5.

| Round | What it trained on | Findings recall | Faithfulness | Invented (of 19) |
| --- | --- | --- | --- | --- |
| **SFT v2** | Kimi's 1,145 code-filtered answers, from the base model | 82.8 / 82.8 | 85.5 / 81.5 | 9 |
| **RFT v1** | Samples from SFT v2, Kimi grading capped at $20 (569 abstracts graded, 524 own answers) | 85.1 / 84.7 | 87.6 / 83.7 | 7 |
| **RFT v2** | The same selection with every abstract graded: 1,058 own, 87 teacher's | 87.4 / 86.4 | 90.9 / 86.3 | 9 |
| **RFT v3** | Samples from RFT v2, Kimi selection: the last open-weight-only round | 89.8 / 88.7 | 90.2 / 84.9 | 9 |
| **RFT v4** | Samples from RFT v3, Kimi on recall then **Sonnet 5.5 on faithfulness**: 1,031 own, 72 teacher's, 42 dropped | **89.2 / 88.3** | **92.0 / 88.2** | 10 |
| **RFT v5** | The same from RFT v4 (1,030 own, 69 teacher's, 46 dropped) plus 90 no-findings abstracts with Kimi's empty answer, Sonnet-checked | 88.1 / 87.9 | 90.3 / 88.4 | 4 |

(SFT v1 trained on the five-per-call form and is superseded; see §1. Numbers on
test set 1, GPT / Sonnet.)

**Why the second judge.** After RFT v3 we asked why three reward rounds had moved
recall and not faithfulness. Kimi K3 passes 76% of the facts GPT-6.1 calls overstated;
GLM is as lenient; Sonnet catches 72%. The reward signal was blind to the thing the
graders were measuring. RFT v4 added Sonnet as a second gate (§3), and faithfulness
moved for the first time: +1.8 [−0.4, +3.9] / +3.3 [+0.7, +6.0] over RFT v3 on test
set 1, and +1.7 [+0.1, +3.2] under GPT on test set 2. The overstated rate fell from
6.7% to 5.7% (GPT). RFT v5 repeated the recipe and gained nothing: −1.8 [−4.0, +0.5] /
+0.2 faithfulness and −1.1 / −0.4 recall against RFT v4 on test set 1, +0.0 / −0.3 and
−0.9 / −0.2 on test set 2. Its no-findings rows did what they were for: it invented
findings on 4 of 19 and 16 of 55 no-result abstracts, against 10 and 35 for RFT v4
and 4 and 5 for Haiku.

**Where the remaining gap is.** Reading RFT v4's overstated findings beside Haiku's
answers to the same abstracts: a dropped condition ("in H2O2-stimulated fibroblasts"),
a certainty upgrade ("might improve" becomes "were associated with remarkable
improvements"), a wrong subject, an author's inference reported as a result. Haiku
keeps the hedge and the condition and splits the claim across two facts. A lexical
test of this (hedge word in the source sentence, none in the fact) does not separate
the models: both drop hedges on about 20% of facts. The difference is what Sonnet was
already judging at selection, and Sonnet passed these styles; one epoch of positives
on about 1,100 examples shifts the style only so far and gives no signal about what
not to do. That is where we stopped (§9).

## 7. Results

GPT-6.1 sol / Sonnet 5.5 graders. Output tokens and cost are per abstract on test
set 1; Claude costs are on-demand Bedrock prices with the instructions cached.

### Test set 1 (306 abstracts)

| Model | Valid | Findings recall | Faithfulness | Invented (of 19) | Output tokens | $ per 100 abstracts |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 (stretch) | 290 | 91.3 / 91.1 | 96.5 / 95.5 | 4 | 394 | $1.22 |
| **Claude Haiku 5.5 (bar)** | 306 | **90.9 / 89.6** | **94.9 / 94.6** | 4 | 425 | $0.034 |
| **Lightning RFT v4, BF16** | 306 | **89.2 / 88.3** | **92.0 / 88.2** | 10 | 233 | — (not a deployment) |
| Lightning RFT v5, BF16 | 306 | 88.1 / 87.9 | 90.3 / 88.4 | 4 | 230 | — |
| Lightning RFT v3, BF16 | 306 | 89.8 / 88.7 | 90.2 / 84.9 | 9 | 232 | — |
| Lightning RFT v2, BF16 | 306 | 87.4 / 86.4 | 90.9 / 86.3 | 9 | 229 | — |
| Lightning RFT v2, NVFP4, 1 GPU (served) | 306 | 85.3 / 85.7 | 89.8 / 88.4 | 11 | 223 | **$0.0146** measured |
| Lightning stock, thinking on | 306 | 83.2 / 81.7 | 88.0 / 82.4 | 1 | 9,397 | — |
| Lightning stock, NVFP4 | 293 | 83.0 / 83.4 | 75.9 / 66.7 | 13 | 200 | — |

(Opus answered 290: its content filter blocked 16 more in the one-per-call form, and
every claim here also holds on those 290. The RFT v2 NVFP4 evaluation-job run, the
file-summary fine-tune and the earlier served measurements are in the package.)

### Test set 2 (629 abstracts)

| Model | Findings recall | Faithfulness | Invented (of 55) |
| --- | --- | --- | --- |
| Claude Opus 5.5 (stretch) | 95.2 / 94.8 | 97.7 / 96.1 | — |
| **Claude Haiku 5.5 (bar)** | **89.3 / 88.5** | **95.0 / 94.7** | 5 |
| **Lightning RFT v4, BF16** | **87.4 / 87.3** | **91.7 / 88.2** | 35 |
| Lightning RFT v5, BF16 | 86.5 / 87.1 | 91.7 / 87.9 | 16 |
| Lightning RFT v3, BF16 | 87.0 / 88.6 | 90.0 / 86.9 | 32 |

Every run lands within a point of its test-set-1 number. Haiku's output is 417 tokens
per abstract here, RFT v4's 229.

### Both sets pooled (935 abstracts)

| Model | Findings recall | Faithfulness |
| --- | --- | --- |
| **Claude Haiku 5.5 (bar)** | **89.8 / 88.9** | **95.0 / 94.6** |
| **Lightning RFT v4, BF16** | **88.0 / 87.6** | **91.8 / 88.2** |
| Lightning RFT v5, BF16 | 87.0 / 87.4 | 91.2 / 88.0 |
| Lightning RFT v3, BF16 | 87.9 / 88.6 | 90.0 / 86.2 |

### Gaps (paired bootstrap, 2,000 resamples, 95% interval)

| Comparison | Set | Findings recall | Faithfulness |
| --- | --- | --- | --- |
| **RFT v4 − Haiku** | **pooled, GPT** | **−1.8 [−3.7, +0.1]** | **−3.2 [−4.4, −2.0]** |
| RFT v4 − Haiku | pooled, Sonnet | −1.3 [−3.2, +0.7] | −6.4 [−7.9, −5.0] |
| RFT v4 − Haiku | set 1, GPT | −1.7 [−4.8, +1.5] | −2.9 [−5.1, −0.7] |
| RFT v4 − Haiku | set 2, GPT | −1.9 [−4.2, +0.6] | −3.3 [−4.8, −1.9] |
| RFT v5 − Haiku | pooled, GPT | −2.8 [−4.7, −0.8] | −3.8 [−5.0, −2.5] |
| RFT v3 − Haiku | pooled, GPT | −1.9 [−3.8, +0.1] | −4.9 [−6.2, −3.7] |
| RFT v4 − RFT v3 | pooled, GPT | +0.1 [−1.7, +1.8] | +1.7 [+0.5, +3.0] |
| RFT v5 − RFT v4 | pooled, GPT | −0.9 [−2.7, +0.9] | −0.6 [−1.9, +0.6] |
| Opus − Haiku | set 2, GPT | +6.0 [+4.1, +7.9] | +2.7 [+1.5, +3.9] |

On test set 1 alone the faithfulness interval was ±2.2 and the gap read as a near
miss. Pooling brought it to ±1.2 and the top of the interval to −2.0: the gap is
real and it is three points. The Sonnet faithfulness gap is twice the GPT one on
every set; the adjudication in §4 is why GPT is the number we stand behind, and the
Sonnet column stays so that the reader can see the disagreement.

Quantizing RFT v2 to NVFP4 cost −2.1 [−5.6, +1.4] / −0.8 [−4.0, +2.6] recall and
−1.1 [−3.7, +1.4] / +2.2 [−0.5, +5.0] faithfulness: within noise. Served against
stock Lightning, faithfulness is up 13.9 / 21.8 points and recall +2.3 / +2.3. The
served cost is the endpoint's hourly price over its sustained throughput (17,800
abstracts an hour at 64 concurrent requests); idle, the endpoint still bills about
$63 a day. RFT v4 was not exported or served; on BF16 it is 1.8 / 1.9 points more
recall and 1.1 / 1.9 more faithfulness than RFT v2 (test set 1).

## 8. The teacher's ceiling, and a preference round

**The teacher on the test set.** Kimi K3 wrote its own key for each test abstract and
answered with it in view, exactly as it made the training targets (run
`kimi-k3-teacher-single`): **95.8 / 93.6** recall, **95.5 / 92.0** faithfulness, 6
invented, 304 valid. Against Haiku: recall +4.9 [+1.9, +8.0] / +4.0 [+0.7, +7.3],
faithfulness +0.5 [−1.2, +2.4] / −2.6 [−4.8, −0.4]. So the student's recall gap and its
GPT faithfulness gap are its own. About 2.6 of its Sonnet faithfulness gap comes from
the teacher; imitating Kimi more closely can't close that column.

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

A second preference round would draw rejected samples from the model itself, keep
only unfaithfulness pairs, and hold no-findings pairs near 6%. The RFT v4 and v5
rounds produced exactly those pairs (472, in the package as `pairs.jsonl`); it was
not run (§9).

## 9. Where we stopped, and why

RFT v4 is the model. After RFT v5 we considered what else might close the three-point
faithfulness gap, with our own odds of reaching within two points of Haiku under GPT:

- A preference round on the 472 unfaithfulness pairs, as §8 describes: about one in
  three, given what the first preference round did.
- A one-claim-per-fact constraint on selection, since any one slip condemns a packed
  fact: about one in four.
- A prompt instruction at inference to keep the authors' hedging: about one in five,
  and it changes the pre-registered prompt.
- GPT-6.1 as the faithfulness signal in selection: the strongest signal, and barred,
  since it is the primary grader.

None above even odds, so none was run. A third test set would not change the
answer either: the pooled faithfulness interval is already ±1.2 around −3.2, and
doubling the sample would narrow it to about ±0.9 without moving it inside the rule.
What remains uncertain is how strict the grader is: of 14 of RFT v4's overstated
verdicts we read, about 3 looked supported to us. That strictness sits under Haiku's
number as much as ours, and we have not measured it beyond that reading.

## What we learned

- **Batching was a false economy.** Caching the shared instructions makes one abstract
  per call cheap, and the batched form cost the model answers it could otherwise give.
- **A reward round is only as big as its grading.** The capped first round graded half
  the training set and gained about 2 points; grading all of it and training once more
  doubled the gain over SFT.
- **A reward signal has to see what the grader sees.** Three rounds of open-weight
  selection moved recall and not faithfulness, because the open-weight judge passed
  most of what the graders flagged. One round with a judge that sees it moved
  faithfulness; a second round with the same judge did not. Positives-only selection
  teaches what to write, not what to leave out.
- **A second test set is cheap insurance.** About $30 reproduced every number within a
  point and turned a near miss into a measured gap.
- **Thinking buys faithfulness at a price.** It's the one prompt-side lever that
  worked for stock Lightning, and the fine-tune reaches the same level without it.

## Spend

About $40 for the baselines and the test and training sets, and about $195 for
training through RFT v2: teacher targets, three training rounds, sampling, open-weight
grading, evaluations with both graders, the NVFP4 export and the served measurement.
The teacher's ceiling and the preference round (§8) cost about $30 more. RFT v3 and the
study of which judges see overstatement, about $70; RFT v4, $59; RFT v5 and the second
test set with its five runs, $122. About $515 in all.
