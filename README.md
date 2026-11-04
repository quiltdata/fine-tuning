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

**Next: an abstract key-fact extractor.** A narrower model that reads a scientific
**abstract** and extracts one fact about what the study is and one or two about its
findings. Not yet run; its bar will be set before training, against a reference model
given a fair task-specific prompt.

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
