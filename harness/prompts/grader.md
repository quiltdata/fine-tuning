# Grader instructions

You get a numbered list of reference facts about one file, and one output row that a summarizer wrote about that file. The row has `what`, `summary`, `about`, `entities` and `refers_to`.

For each fact, decide whether the output row conveys it.

- **Conveyed** means a reader of the row would learn the fact: it is stated in the summary, or clearly expressed by the `what`, `about`, `entities` or `refers_to` values. Paraphrase counts. A list of subject terms counts when the terms carry the fact (for example, a fact that the study is in mice is conveyed by "mouse" in `about`).
- **Not conveyed** means the fact is missing, only vaguely gestured at ("various measurements" for a specific measurement), or contradicted.
- When a fact has several parts, it is conveyed if its central claim is there, even without every detail.

Judge only against the reference facts. Don't reward or penalize anything else in the row.

Return one verdict per fact, in the order given.
