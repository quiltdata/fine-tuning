# Grader instructions: is each of the model's facts supported by the abstract?

You get one scientific abstract and the facts a model wrote about it: an `about` fact and up to
two findings, each with an id. Judge every fact against the abstract alone, without outside
knowledge.

For each fact, give one verdict:

- **supported**: the abstract states it, including its direction, its quantities and the
  population or conditions it holds for, at the abstract's own strength of claim. Paraphrase is
  fine.
- **overstated**: the abstract supports only a weaker or narrower version. For example: a cause
  where the abstract reports an association; "does" where it says "may"; a result generalized
  beyond the population or conditions studied; a subgroup result stated as a general one; a
  number made more precise or larger than the abstract's.
- **unsupported**: the abstract doesn't state it, contradicts it, or reports no results at all
  (a study protocol or planned review) while the fact states one.

Then, if there are two findings, say whether they state the same result (`redundant`: true),
for example the same effect in other words, or one result split into two. Two different results
that are related are not redundant. With fewer than two findings, `redundant` is false.

Return one verdict per fact id, and `redundant`.
