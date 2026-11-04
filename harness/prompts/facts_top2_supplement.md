# Reference facts instructions (two facts, supplementary files)

You get one row of a summarizer's input table: one supplementary file from a scientific article's folder, with text that code extracted from it (`excerpt`), passages from nearby files that name it (`mentions`), and the title and abstract of the article it belongs to (`parent`).

A summarizer will later describe this file in a short output row: `what` it is, a one or two sentence `summary` (at most 300 characters), subject terms (`about`), named `entities` and other files it names (`refers_to`). Your job is to write the answer key: the two facts a good output row must convey.

A reader browsing a catalog who finds this file needs to know which study it belongs to and what that study found, before the details of the file itself. Write the key for that reader.

## What to return

Exactly 2 key facts, most important first. Each fact has:

- `fact`: one short, self-contained statement, specific enough to check (name the organism, assay, compound, measurement or finding, not "the topic of the study").
- `quote`: a short passage, at most 200 characters, copied exactly from the excerpt, mentions or parent, that states the fact.

## Choosing facts

1. What the file is and the study it supports: the kind of supplement (supplementary tables, figures, methods, questionnaire, data sheet) and the study's subject and design, for example "supplementary methods and figures for a study measuring X in Y". Quote the parent's statement of the study's aim or design where it says so best.
2. The study's main finding, as the parent abstract's results or conclusion states it. If the parent gives no finding (a protocol, a methods paper), use what the study sets out to do or deliver.

Rules:

- Use only the excerpt, mentions and parent. No outside knowledge.
- Every fact must be stated in the excerpt, mentions or parent, and its quote must appear there word for word.
- Leave out bibliographic identifiers (PMCID, PMID, DOI, license, journal name, author names); they are checked separately.
- Don't write facts about truncation, file size or format details a summary wouldn't mention.
- Each fact should fit comfortably in a short summary. Prefer one claim per fact.
- If `parent` is empty, fall back to the file itself: what kind of file it is and what it is about, then its main content.
