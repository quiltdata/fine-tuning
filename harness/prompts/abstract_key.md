# Answer key instructions: abstract facts

You get one scientific abstract (`abstract`). Write the answer key for it: the facts a good
reading of this abstract must convey. Other models will later be scored on whether their own
facts convey yours, so the key must be correct, specific and complete, and nothing more.

## What to return

- `about`: exactly one fact. What kind of study this is (its design: randomized trial, cohort
  study, systematic review, laboratory experiment, computational analysis, case report...), on
  what system (organism, population, cells, material, dataset), and to what end (the question it
  asks). For example: "A randomized trial in 412 adults with type 2 diabetes testing whether a
  low-carbohydrate diet improves glucose control."
- `findings`: the one or two most important results the abstract reports, most important first.
  - Give two only when the abstract reports two distinct results that both matter. Don't split one
    result into two, and don't repeat a result in other words.
  - State each result with its direction, the key quantities, and the conditions or population it
    holds for, as the abstract states them. For example: "The diet lowered HbA1c by 0.6 percentage
    points more than usual care at 12 weeks."
  - Keep the abstract's own strength of claim: "was associated with" is not "caused"; "may" stays
    "may". Don't generalize beyond the population or conditions studied.
  - Don't spend a finding on the methods, a limitation, or a recommendation for future work.
  - If the abstract reports no results (a study protocol, a planned review, a description of a
    method or resource without an evaluation), return no findings.

Each fact (the about fact and each finding) has:

- `fact`: one claim, in your own words, at most 250 characters, specific enough to check.
- `quote`: a passage of at most 200 characters, copied exactly from the abstract, that states the
  fact. Code checks it; a fact whose quote isn't in the abstract is dropped.

## Rules

- Use only the abstract. No outside knowledge.
- Every fact must be stated in the abstract.
- Leave out identifiers (trial registration numbers, DOIs) and author or journal names.
