# fine-tuning

Example fine-tuned Nemotron models for scientific key-fact extraction, and the method
behind them.

The aim is to show **why context matters** and **how to build your own**: a concrete
path from a desired behavior to examples, post-training and measured performance,
with the evaluation fixed before training and the failures recorded next to the
results.

## Layout

| Path | What's there |
| --- | --- |
| [`METHODOLOGY.md`](METHODOLOGY.md) | How the model was trained and judged, the results, and the corrections that changed them. **Start here.** |
| [`ABSTRACTS.md`](ABSTRACTS.md) | The second model: key facts from a scientific abstract. Method, results, and where it falls short. |
| [`training/`](training/) | The post-training code: LoRA SFT, reward-stage selection, evaluation, and NVFP4 export, run as SageMaker jobs. See [`training/README.md`](training/README.md). |
| [`harness/`](harness/) | Building the datasets, running any model, and checking and scoring its outputs. See [`harness/README.md`](harness/README.md). |

## Status

**Done: a key-fact summarizer.** NVIDIA Nemotron 3.5 Lightning (30B-A3B), post-trained
with an open-weight teacher and code checks and served in 4-bit on one GPU, is
**level with Claude Opus 5.5** on stating a file's two key facts, at a **measured
$0.030 per 100 rows** (about 1.2% of Opus). That holds on 8,000-character excerpts.
The served model falls four rows short of our own 85% verbatim-quote bar, and on
whole papers a fairly prompted Opus is about 10 points ahead. Details in
[`METHODOLOGY.md`](METHODOLOGY.md).

**Done, short of its bar: an abstract key-fact extractor.** A narrower model that reads
a scientific **abstract** and states what the study is and its one or two main findings.
Four reward rounds took Lightning from 83% / 76% (findings recall / faithfulness, GPT-6.1
grading) to 88% / 92% on 935 held-out abstracts, past stock Lightning with thinking on.
The bar set before training was Claude Haiku 5.5, at 90% / 95%: the recall gap is within
two points, a tie; the faithfulness gap is 3.2 points with its whole interval outside two.
A fifth round did not move it, and we stopped. Served in 4-bit on one GPU the earlier RFT v2
costs a measured $0.0146 per 100 abstracts, under half of Haiku. Details in
[`ABSTRACTS.md`](ABSTRACTS.md).

## Configuration

The training scripts read their AWS account, execution role, job prefix and tags from
the environment, not the code. Copy [`.env.example`](.env.example) to `.env` (git-ignored)
and fill it in, or run `./make-env.sh /path/to/source.env`. See
[`training/README.md`](training/README.md#configuration).

## Not in this repo

The datasets (PubMed Central open-access articles, CC BY or CC0; `harness/` rebuilds
them from public S3) and the model weights.

## License

Copyright 2026 Quilt Data, Inc. Licensed under the [Apache License, Version 2.0](LICENSE).

This license covers the code and documentation in this repo. Models you train with it
are derivatives of NVIDIA Nemotron 3.5 Lightning and remain subject to NVIDIA's model
license; the teacher and grader models are subject to their providers' terms.
