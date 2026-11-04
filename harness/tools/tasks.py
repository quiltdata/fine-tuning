"""The tasks a model can be run on: prompt, examples, schemas, what the model sees, batching, limits.

  file-summary    the earlier summarizer: one row per file, batches of one article folder and file
                  type (prompts/summarizer.md, schemas/output.schema.json).
  abstract-facts  the abstract task: per abstract, one `about` fact and zero to two `findings`, each
                  in the model's own words, at most 250 characters, with an optional quote (at most
                  200). The model sees only unit_id and abstract; batches of ten in input order.

The abstract task's student-facing prompt (prompts/abstract_facts.md) and examples
(prompts/abstract_examples.md) are written by a person, not drafted by a model: every run uses
them, including the training data, which takes no Claude or OpenAI output. Until they exist, runs
of that task stop, unless a caller explicitly allows running without examples (tests).

Every run records task_fingerprint(): the task name and SHA-256 of its prompt, examples and schema.
"""
import copy
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNSUPPORTED = {"$schema", "$id", "title", "description", "minLength", "maxLength", "minItems", "maxItems", "pattern"}
FACT_CHARS = 250
QUOTE_CHARS = 200


def strip(schema):
    """A JSON schema without the keywords structured-output APIs reject; code checks them afterwards."""
    if isinstance(schema, dict):
        return {k: strip(v) for k, v in schema.items() if k not in UNSUPPORTED}
    if isinstance(schema, list):
        return [strip(v) for v in schema]
    return schema


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path and path.exists() else None


class Task:
    name = ""
    prompt = None             # file in prompts/
    examples = None           # optional file in prompts/, appended to the prompt
    output_schema = None      # file in schemas/, one output row
    list_columns = ()         # output CSV cells that hold JSON
    out_columns = ()
    requires_examples = False

    def __init__(self, root=ROOT):
        self.root = Path(root)

    def path(self, kind, name):
        return self.root / kind / name if name else None

    def missing(self):
        """Files this task needs that don't exist yet."""
        need = [self.path("prompts", self.prompt), self.path("schemas", self.output_schema)]
        if self.requires_examples:
            need.append(self.path("prompts", self.examples))
        return [str(p.relative_to(self.root)) for p in need if p and not p.exists()]

    def system(self, prompt=None):
        """The system prompt: the task prompt (or `prompt`, a file in prompts/), then the examples if any."""
        text = self.path("prompts", prompt or self.prompt).read_text()
        ex = self.path("prompts", self.examples)
        if ex and ex.exists() and ex.read_text().strip():
            text = text.rstrip() + "\n\n" + ex.read_text().strip() + "\n"
        return text

    def row_schema(self):
        return json.loads(self.path("schemas", self.output_schema).read_text())

    def api_schema(self):
        return {"type": "object", "properties": {"rows": {"type": "array", "items": strip(copy.deepcopy(self.row_schema()))}},
                "required": ["rows"], "additionalProperties": False}

    def fingerprint(self, prompt=None):
        return {"task": self.name, "prompt": prompt or self.prompt,
                "prompt_sha256": _sha(self.path("prompts", prompt or self.prompt)),
                "examples_sha256": _sha(self.path("prompts", self.examples)),
                "schema_sha256": _sha(self.path("schemas", self.output_schema))}

    def batches(self, rows, per_file=False, size=None):
        raise NotImplementedError

    def model_view(self, row):
        """What the model is shown of one input row."""
        raise NotImplementedError

    def user_message(self, rows, required=None):
        """The user turn for one batch. With `required` (a teacher run), the facts each row must convey
        follow the table; make_pairs.py never passes them, so the student never sees them."""
        msg = "Input table, one JSON object per row:\n" + json.dumps([self.model_view(r) for r in rows], ensure_ascii=False)
        facts = {r["unit_id"]: required[r["unit_id"]] for r in rows if required and required.get(r["unit_id"])}
        if facts:
            msg += "\n\n" + self.required_note + "\n" + json.dumps(facts, ensure_ascii=False)
        return msg

    def too_long(self, out):
        """Teacher runs only: rows over the task's limits, as feedback for one more attempt."""
        return []

    def normalize(self, row):
        """An output row as returned, made ready for the row schema."""
        return row


class FileSummary(Task):
    name = "file-summary"
    prompt = "summarizer.md"
    output_schema = "output.schema.json"
    list_columns = ("about", "entities", "refers_to", "evidence")
    out_columns = ("unit_id", "what", "summary", "about", "entities", "refers_to", "evidence")
    required_note = ("Required content: a reader of each output row must learn these facts about its unit. State the "
                     "gist in the summary and carry the specifics in about, entities and refers_to; a reader of the whole row "
                     "must learn each fact, but the summary alone need not spell them out. The summary limit of 300 characters "
                     "is enforced by code: write it, count its characters, and shorten it until it fits. Quotes stay at most "
                     "200 characters and verbatim.")

    def batches(self, rows, per_file=False, size=None):
        """Batches of one prefix and one file type; per_file=True sends each row alone (whole documents)."""
        if per_file:
            return [((r["prefix"], r["file_type"]), [r]) for r in rows]
        groups = OrderedDict()
        for r in rows:
            groups.setdefault((r["prefix"], r["file_type"]), []).append(r)
        return list(groups.items())

    def model_view(self, r):
        return {**r, "members": int(r["members"]), "size_bytes": int(r["size_bytes"]),
                "example_keys": json.loads(r["example_keys"]), "mentions": json.loads(r["mentions"])}

    def too_long(self, out):
        problems = []
        for o in out:
            if isinstance(o, dict):
                if len(o.get("summary") or "") > 300:
                    problems.append(f'{o.get("unit_id")}: summary is {len(o["summary"])} characters, the limit is 300')
                for q in o.get("evidence") or []:
                    if isinstance(q, str) and len(q) > QUOTE_CHARS:
                        problems.append(f'{o.get("unit_id")}: a quote is {len(q)} characters, the limit is {QUOTE_CHARS}')
        return problems


class AbstractFacts(Task):
    name = "abstract-facts"
    prompt = "abstract_facts.md"
    examples = "abstract_examples.md"
    output_schema = "abstract_output.schema.json"
    list_columns = ("about", "findings")
    out_columns = ("unit_id", "about", "findings")
    requires_examples = True
    batch_size = 10
    required_note = ("Required content: each output row must convey these facts about its abstract, the about fact and the "
                     "findings, stated in your own words. Each fact stays one claim of at most 250 characters, enforced by "
                     "code; quotes stay optional, verbatim and at most 200 characters.")

    def batches(self, rows, per_file=False, size=None):
        """Fixed-size batches in input order (one row each with per_file); keys are ("batch-<n>", "abstract")."""
        n = 1 if per_file else (size or self.batch_size)
        return [((f"batch-{i // n}", "abstract"), rows[i:i + n]) for i in range(0, len(rows), n)]

    def model_view(self, r):
        return {"unit_id": r["unit_id"], "abstract": r["abstract"]}

    def api_schema(self):
        """Written out rather than derived: strict structured output (Bedrock Chat Completions) rejects
        optional properties and $ref, so `quote` is always present and "" means no quote."""
        fact = {"type": "object", "properties": {"fact": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["fact", "quote"], "additionalProperties": False}
        row = {"type": "object", "properties": {"unit_id": {"type": "string"}, "about": fact,
                                                "findings": {"type": "array", "items": fact}},
               "required": ["unit_id", "about", "findings"], "additionalProperties": False}
        return {"type": "object", "properties": {"rows": {"type": "array", "items": row}},
                "required": ["rows"], "additionalProperties": False}

    def normalize(self, row):
        """Drops empty or missing quotes, so "" from strict structured output reads as no quote."""
        if not isinstance(row, dict):
            return row
        def fact(f):
            if not isinstance(f, dict):
                return f
            f = {k: v for k, v in f.items() if not (k == "quote" and not (isinstance(v, str) and v.strip()))}
            return f
        out = dict(row)
        if "about" in out:
            out["about"] = fact(out["about"])
        if isinstance(out.get("findings"), list):
            out["findings"] = [fact(f) for f in out["findings"]]
        return out

    def too_long(self, out):
        problems = []
        for o in out:
            if not isinstance(o, dict):
                continue
            facts = [("about", o.get("about"))] + [(f"finding {i + 1}", f) for i, f in enumerate(o.get("findings") or [])]
            for label, f in facts:
                if not isinstance(f, dict):
                    continue
                if len(f.get("fact") or "") > FACT_CHARS:
                    problems.append(f'{o.get("unit_id")}: {label} is {len(f["fact"])} characters, the limit is {FACT_CHARS}')
                if len(f.get("quote") or "") > QUOTE_CHARS:
                    problems.append(f'{o.get("unit_id")}: the {label} quote is {len(f["quote"])} characters, the limit is {QUOTE_CHARS}')
        return problems


TASKS = {t.name: t for t in (FileSummary, AbstractFacts)}


def get(name="file-summary", root=ROOT):
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; one of {sorted(TASKS)}")
    return TASKS[name](root)
