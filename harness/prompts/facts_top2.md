# Reference facts instructions (two facts)

You get one row of a summarizer's input table: one file from a data bucket, with text that code extracted from it (`excerpt`) and passages from nearby files that name it (`mentions`).

A summarizer will later describe this file in a short output row: `what` it is, a one or two sentence `summary` (at most 300 characters), subject terms (`about`), named `entities` and other files it names (`refers_to`). Your job is to write the answer key: the two facts a good output row must convey.

## What to return

Exactly 2 key facts, most important first. Each fact has:

- `fact`: one short, self-contained statement, specific enough to check (name the organism, assay, compound, measurement or finding, not "the topic of the study").
- `quote`: a short passage, at most 200 characters, copied exactly from the excerpt or mentions, that states the fact.

## Choosing facts

Pick the two things a reader deciding whether this file is relevant needs most. Usually:

1. What kind of document or data the file is, and what it is about: organism, disease, cell type, compound, method or study design (for example, a supplementary table of per-animal heart-rate measurements in rats; a research article on a microCT pipeline for leaf airspace in Arabidopsis).
2. Its main content: the central finding, what the table measures and across which groups, or what the supplement adds to the article.

Rules:

- Use only the excerpt and mentions. No outside knowledge.
- Every fact must be stated in the excerpt or mentions, and its quote must appear there word for word.
- Leave out bibliographic identifiers (PMCID, PMID, DOI, license, journal name, author names); they are checked separately.
- Don't write facts about truncation, file size or format details a summary wouldn't mention.
- Each fact should fit comfortably in a short summary. Prefer one claim per fact.
