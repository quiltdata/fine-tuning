# harness: building, running and scoring the summarizer task

The data pipeline and evaluation behind [`../METHODOLOGY.md`](../METHODOLOGY.md): sample
articles, extract inputs, write answer keys, run any model, check and score its outputs,
and turn teacher runs into training data. [`../training/`](../training/) uses these for
its SageMaker jobs.

```bash
pip install -r requirements.txt
```

Models are called through Amazon Bedrock (Converse or its Chat Completions API), a
SageMaker endpoint, or any OpenAI-compatible server. Input files are read anonymously
from public S3 (PubMed Central's `pmc-oa-opendata`).

## Pipeline

| Step | Tool | Writes |
| --- | --- | --- |
| Sample articles (CC BY / CC0, open access, not retracted) | `tools/sample_pmc.py` | `objects.txt`, `articles.csv` |
| Extract inputs (8,000-character excerpts, mentions) | `tools/make_inputs.py` | `inputs.csv` |
| Split train/test by article | `tools/make_split.py` | `split.csv` |
| Write the answer key | `tools/make_facts.py` | `reference/facts.jsonl` |
| Run a model | `tools/run_model.py` (any backend), `tools/run_claude.py` (Anthropic SDK) | `runs/<name>/outputs.csv`, `calls.csv`, `run.json` |
| Check outputs (schema, lengths, verbatim quotes) | `tools/check_outputs.py` | pass/fail per row |
| Score against the key, with a grader model | `tools/score_run.py` | `scores.csv`, `grades.jsonl`, `score.json` |
| Compare graders | `tools/compare_graders.py` | agreement, Cohen's kappa |
| Compare runs (score vs. cost chart) | `tools/make_report.py` | `index.html`, `comparison.csv` |
| Teacher run to training pairs | `tools/make_pairs.py`, then `tools/make_sft.py` | `pairs.jsonl`, `train.jsonl` |

A **set** is a folder with `inputs.csv`, `split.csv` (test sets), `reference/facts.jsonl`
and `runs/<name>/`. A row is one file excerpt; a run's batches group rows by article
folder and file type.

## Prompts and schemas

- `prompts/summarizer.md`: the task. `summarizer_main_finding.md` adds one sentence asking for
  an article's main finding: the fair baseline prompt (see METHODOLOGY.md, corrections).
- `prompts/facts.md`, `facts_top2.md`: writing the answer key (3–5 facts, or exactly 2).
  `facts_top2_supplement.md` was for supplements, later dropped from scope.
- `prompts/grader.md`: judging whether an output row conveys each key fact.
- `schemas/input.schema.json`, `output.schema.json`: one input row and one output row.

## Rules the tools enforce

- **Quotes are checked by code.** `check_outputs.quote_found` matches a quote's letters and
  digits, in order, against the input (case, punctuation and whitespace ignored).
- **Training data is filtered by code only.** `make_sft.py` keeps a teacher's batch only if
  every row passes the check, and refuses Claude-teacher pairs.
- **Costs are recorded per call.** Bedrock calls are priced from `tools/model_rates.json`
  (list rates as of 2026-10-24); SageMaker endpoints by instance time at sustained throughput.
