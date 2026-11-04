# Grader instructions: does the model convey the reference facts?

You get the reference facts for one scientific abstract (an `about` fact describing the study,
and its findings, each with an id) and the facts a model wrote about the same abstract. You do
not see the abstract.

For each reference fact, decide whether the model's facts convey it.

- **Conveyed** means a reader of the model's facts would learn the reference fact. Paraphrase
  counts, and it may be carried by any of the model's facts, not only the one in the same position.
  - For the `about` fact: the kind of study, and the system it studies, are both there.
  - For a finding: the same result, in the same direction, with its central quantity or condition
    when the reference states one. "Lowered HbA1c by 0.6 points" is conveyed by "reduced HbA1c by
    about 0.6 percentage points", not by "improved glucose control".
- **Not conveyed** means the fact is missing, only gestured at ("several outcomes improved"),
  stated in the wrong direction, or contradicted.
- When a fact has several parts, it is conveyed if its central claim is there, even without every
  detail.

Judge only against the reference facts. Don't reward or penalize anything else the model wrote.

Return one verdict per reference fact id.
