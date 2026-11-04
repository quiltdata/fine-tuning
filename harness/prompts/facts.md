# Reference facts instructions

You get one row of a summarizer's input table: one file from a data bucket, with text that code extracted from it (`excerpt`) and passages from nearby files that name it (`mentions`).

A summarizer will later describe this file in a short output row: `what` it is, a one or two sentence `summary` (at most 300 characters), subject terms (`about`), named `entities` and other files it names (`refers_to`). Your job is to write the answer key: the facts a good output row must convey.

## What to return

Between 3 and 5 key facts, most important first. Each fact has:

- `fact`: one short, self-contained statement, specific enough to check (name the organism, assay, compound, measurement or finding, not "the topic of the study").
- `quote`: a short passage, at most 200 characters, copied exactly from the excerpt or mentions, that states the fact.

## Choosing facts

Pick what a reader deciding whether this file is relevant needs most:

1. What kind of document or data the file is (for example, a supplementary table of per-animal heart-rate measurements, a reviewer response letter, a methods appendix, a research article).
2. What it is about: organism, disease, cell type, compound, method or study design.
3. Its main content: the central finding, what the table measures and across which groups, or what the supplement adds to the article.
4. A relationship stated in the mentions, such as which figure or table of the article it supplies.

Rules:

- Use only the excerpt and mentions. No outside knowledge.
- Every fact must be stated in the excerpt or mentions, and its quote must appear there word for word.
- Leave out bibliographic identifiers (PMCID, PMID, DOI, license, journal name, author names); they are checked separately.
- Don't write facts about truncation, file size or format details a summary wouldn't mention.
- If the excerpt is thin (a few table rows, a short letter), 3 facts is fine. Don't pad.
- Each fact should fit comfortably in a short summary. Prefer one claim per fact.
