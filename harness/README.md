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

## Tasks

`tools/tasks.py` defines each task once: its prompt and examples, output schema, what the model
sees of an input row, batching and length limits. `run_model.py`, `run_claude.py`, `make_pairs.py`
and the training tools take `--task`, and every run records the task with SHA-256 of its prompt,
examples and schema.

| Task | Input | Output | Batches |
| --- | --- | --- | --- |
| `file-summary` (default) | a file excerpt and mentions | `what`, `summary`, `about`, `entities`, `refers_to`, `evidence` | one article folder and file type |
| `abstract-facts` | a paper's main abstract, at most 4,096 UTF-8 bytes | one `about` fact and zero to two `findings`, each at most 250 characters with an optional quote | 10 abstracts (`--batch-size`) |

### The abstract task

| Step | Tool | Writes |
| --- | --- | --- |
| Sample articles (no supplement requirement) | `tools/sample_pmc.py --articles-only` | `articles.csv` (with PMIDs) |
| Build inputs: main abstract, size, truncation, publication types, protocol flag | `tools/make_abstracts.py` | `inputs.csv`, `skipped.csv` |
| Write the typed key (about, findings) | `tools/make_facts.py --task abstract-facts --split-label test` | `reference/facts.jsonl` |
| Run a model | `tools/run_model.py --task abstract-facts` (Bedrock Converse, SageMaker or OpenAI-compatible) | `runs/<name>/…` |
| Check validity | `tools/check_outputs.py --task abstract-facts` | pass/fail per row; quotes as a note |
| Score | `tools/score_abstracts.py` (twice: GPT-6.1 sol, then `--grader … --tag sonnet`) | `scores.csv`, `grades.jsonl`, `score.json` |
| Teacher run to pairs, any batch size | `tools/make_pairs.py --mixed-batches SEED`, then `tools/make_sft.py` | `pairs.jsonl`, `train.jsonl` |

The input rules live in `tools/abstracts.py`: which abstract is the main one, how it is counted
(UTF-8 bytes of the flattened text, section titles kept), truncation at 4,096 bytes on a whole
character, and the three-signal study-protocol rule (PubMed type, title, protocol headings).

**The prompt and examples are data, not code.** The task's student-facing prompt and examples
are published in the public data package
([`quilt+s3://quilt-ai#package=models/nemotron-lightning`](https://nightly.quilttest.com/b/quilt-ai/packages/models/nemotron-lightning)),
as `task/abstract_facts.md` and `task/abstract_examples.md`, written by a person, not a model. The
tools read them from a local copy of that package, `FINE_TUNING_DATA` in `.env`. Runs stop until
they exist.

**Drafts stay out of training.** `--prompt PATH` and `--examples PATH` run any other files, for
iterating on a prompt elsewhere. Such a run records `"published": false` (with file names and
hashes, never local paths), and the training path refuses it: `make_pairs.py` won't build pairs
from it, and `make_sft.py` refuses any pair whose prompt and examples aren't the published ones.
Iterate on a dev set of your own, never the test set. `make_sft.py` also refuses Claude and closed
OpenAI teachers.

The key-writer and grader prompts (`prompts/abstract_key.md`, `prompts/abstract_grader_*.md`) stay
here: they are part of the evaluation and never reach the student.

**Baselines on Claude** run through `run_model.py`'s Bedrock Converse backend
(`--model us.anthropic.claude-haiku-5-5`, `us.anthropic.claude-opus-5-5`), both the same way: Haiku 5.5
isn't served on the Anthropic-SDK Bedrock endpoint `run_claude.py` uses (it returns 404). Converse
has no refusal fallback, so every row is the model's own. If `run_claude.py` is used (Opus only),
pass `--no-fallback`.

### Tests

```bash
python3 -m pytest harness/tests
```

Offline: the model, grader and NCBI are stubbed. `test_end_to_end.py` runs the abstract task
through the command-line tools against a local OpenAI-compatible stub.

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
