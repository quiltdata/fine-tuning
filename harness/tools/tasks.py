"""The tasks a model can be run on: prompt, examples, schemas, what the model sees, batching, limits.

  file-summary    the earlier summarizer: one row per file, batches of one article folder and file
                  type (prompts/summarizer.md, schemas/output.schema.json).
  abstract-facts  the abstract task: one abstract per call, as plain text; the model returns one JSON
                  object, one `about` fact and zero to two `findings`, each in its own words, with an
                  optional quote (35d). The harness attaches the abstract's id. Length is reported, not
                  limited or retried (35c).

Where prompts live. The file-summary prompts are in harness/prompts/. The abstract task's
student-facing prompt and examples are data, not code: they are published in the public data
package (task/abstract_facts.md, task/abstract_examples.md) and read from a local copy of it, whose
path is FINE_TUNING_DATA (environment or the repo's .env). They are written by a person, not drafted
by a model, because the training data takes no Claude or OpenAI output.

Drafts. A run can use another prompt or examples file by path (--prompt, --examples), for
iterating outside the published package. Its fingerprint then says published: false, and the
training path (make_pairs.py, make_sft.py) refuses it.

Every run records fingerprint(): the task, the prompt and examples file names, SHA-256 of prompt,
examples and schema, and whether prompt and examples are the published ones. Names only, never
local paths: runs are published.
"""
import copy
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ENV = "FINE_TUNING_DATA"
UNSUPPORTED = {"$schema", "$id", "title", "description", "minLength", "maxLength", "minItems", "maxItems", "pattern"}
QUOTE_CHARS = 200


def data_dir():
    """The local copy of the public data package: FINE_TUNING_DATA from the environment, else the
    repo's .env (real environment variables win, as in training/config.py)."""
    if DATA_ENV not in os.environ:
        env = ROOT.parent / ".env"
        for line in env.read_text().splitlines() if env.is_file() else []:
            k, _, v = line.strip().partition("=")
            if k.strip() == DATA_ENV and v.strip():
                os.environ.setdefault(DATA_ENV, v.strip().strip('"').strip("'"))
    v = os.environ.get(DATA_ENV)
    return Path(v).expanduser() if v else None


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
    prompt = None             # the published prompt's file name
    examples = None           # optional: the published examples' file name, appended to the prompt
    prompt_home = "repo"      # "repo": harness/prompts/; "data": <FINE_TUNING_DATA>/task/
    output_schema = None      # file in harness/schemas/, one output row
    list_columns = ()         # output CSV cells that hold JSON
    out_columns = ()
    requires_examples = False

    def __init__(self, root=ROOT, data=None):
        self.root = Path(root)
        self.data = Path(data) if data else data_dir()

    def home(self):
        """Where the published prompt and examples live (None if FINE_TUNING_DATA isn't set)."""
        if self.prompt_home == "data":
            return self.data / "task" if self.data else None
        return self.root / "prompts"

    def published(self, name):
        return self.home() / name if name and self.home() else None

    def resolve(self, override):
        """A --prompt or --examples value: a path to a file, or a name in harness/prompts/
        (file-summary's alternates, such as summarizer_main_finding.md)."""
        p = Path(override).expanduser()
        if p.exists():
            return p
        alt = self.root / "prompts" / override
        return alt if alt.exists() else p

    def files(self, prompt=None, examples=None):
        """(prompt path, examples path or None): the overrides if given, else the published ones."""
        return (self.resolve(prompt) if prompt else self.published(self.prompt),
                self.resolve(examples) if examples else self.published(self.examples))

    def schema_path(self):
        return self.root / "schemas" / self.output_schema

    def missing(self, prompt=None, examples=None, allow_no_examples=False):
        """What this run needs that doesn't exist yet."""
        pp, ep = self.files(prompt, examples)
        need = [("prompt", pp), ("schema", self.schema_path())]
        if self.requires_examples and not allow_no_examples:
            need.append(("examples", ep))
        return [f"{kind} ({DATA_ENV} is not set)" if p is None else f"{kind} {p.name}" for kind, p in need
                if p is None or not p.exists()]

    def system(self, prompt=None, examples=None):
        """The system prompt: the prompt, then the examples if there are any."""
        pp, ep = self.files(prompt, examples)
        text = pp.read_text()
        if ep and ep.exists() and ep.read_text().strip():
            text = text.rstrip() + "\n\n" + ep.read_text().strip() + "\n"
        return text

    def row_schema(self):
        return json.loads(self.schema_path().read_text())

    def api_schema(self):
        return {"type": "object", "properties": {"rows": {"type": "array", "items": strip(copy.deepcopy(self.row_schema()))}},
                "required": ["rows"], "additionalProperties": False}

    def fingerprint(self, prompt=None, examples=None):
        """What a run used: file names (never local paths) and hashes, and whether the prompt and
        examples are the published ones (published: true is the only kind the training path takes)."""
        pp, ep = self.files(prompt, examples)
        ep = ep if ep and ep.exists() else None
        out = {"task": self.name, "prompt": pp.name if pp else None, "prompt_sha256": _sha(pp),
               "examples": ep.name if ep else None, "examples_sha256": _sha(ep),
               "schema_sha256": _sha(self.schema_path())}
        pub_p, pub_e = self.published(self.prompt), self.published(self.examples)
        out["published"] = bool(pub_p and pub_p.exists() and out["prompt_sha256"] == _sha(pub_p)
                                and out["examples_sha256"] == _sha(pub_e)
                                and (out["examples_sha256"] or not self.requires_examples))
        return out

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

    text_instruction = ('\n\nReply with only a JSON object of the form {"rows": [...]}, one object per input row with '
                        "the output columns above, and nothing else.")

    def rows_from_reply(self, data):
        """Output rows from a parsed reply: a {"rows": [...]} table, or a bare list of rows."""
        if isinstance(data, dict) and "rows" in data:
            data = data["rows"]
            if isinstance(data, str):  # some models put the table in a tool call as a JSON string
                data = json.loads(data)
        if isinstance(data, list):
            return data
        raise ValueError("no JSON rows in reply")

    def attach_ids(self, out, batch):
        """Output rows with their unit_id: the model copies it in the table form."""
        return out

    def assistant_json(self, rows):
        """The assistant turn for training on these rows, as the model is asked to answer."""
        return {"rows": rows}


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
    prompt_home = "data"
    output_schema = "abstract_output.schema.json"
    list_columns = ("about", "findings")
    out_columns = ("unit_id", "about", "findings")
    requires_examples = True
    required_note = ("Required content: your answer must convey these facts about the abstract, the about fact and the "
                     "findings, stated in your own words. Each fact stays one claim; quotes stay optional and verbatim.")
    text_instruction = ('\n\nReply with only a JSON object of the form {"about": {...}, "findings": [...]}, '
                        "and nothing else.")

    def batches(self, rows, per_file=False, size=None):
        """One abstract per call (35d); keys are ("abstract-<n>", "abstract")."""
        if size not in (None, 1):
            raise SystemExit("abstract-facts sends one abstract per call (35d); --batch-size must be 1 or unset")
        return [((f"abstract-{i}", "abstract"), [r]) for i, r in enumerate(rows)]

    def user_message(self, rows, required=None):
        """The abstract's text, nothing else. A teacher run adds the facts the answer must convey;
        make_pairs.py never passes them, so the student never sees them."""
        (row,) = rows
        msg = row["abstract"]
        facts = required.get(row["unit_id"]) if required else None
        if facts:
            msg += "\n\n" + self.required_note + "\n" + json.dumps(facts, ensure_ascii=False)
        return msg

    def api_schema(self):
        """Written out rather than derived: strict structured output (Bedrock Chat Completions) rejects
        optional properties and $ref, so `quote` is always present and "" means no quote. One object,
        no id: the harness knows which abstract it sent."""
        fact = {"type": "object", "properties": {"fact": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["fact", "quote"], "additionalProperties": False}
        return {"type": "object", "properties": {"about": fact, "findings": {"type": "array", "items": fact}},
                "required": ["about", "findings"], "additionalProperties": False}

    def rows_from_reply(self, data):
        """The one answer object (a table with one row is accepted too)."""
        if isinstance(data, dict) and "about" in data:
            return [data]
        return super().rows_from_reply(data)[:1]

    def attach_ids(self, out, batch):
        """The answer gets the id of the abstract that was sent; a stray id from the model is overwritten."""
        (row,) = batch
        return [dict(o, unit_id=row["unit_id"]) for o in out[:1] if isinstance(o, dict)]

    def assistant_json(self, rows):
        """The answer object without the id, as the model gives it."""
        (row,) = rows
        return {k: v for k, v in row.items() if k != "unit_id"}

    def normalize(self, row):
        """Drops empty or missing quotes, so "" from strict structured output reads as no quote, and
        decodes a fact or findings list a model returned as a JSON string (seen from Opus 5.5 when
        Bedrock won't force the tool). The same for every model."""
        if not isinstance(row, dict):
            return row
        def decoded(v):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return v
            return v
        def fact(f):
            f = decoded(f)
            if not isinstance(f, dict):
                return f
            f = {k: v for k, v in f.items() if not (k == "quote" and not (isinstance(v, str) and v.strip()))}
            return f
        out = dict(row)
        if "about" in out:
            out["about"] = fact(out["about"])
        out["findings"] = decoded(out.get("findings", []))
        if isinstance(out.get("findings"), list):
            out["findings"] = [fact(f) for f in out["findings"]]
        return out

    def too_long(self, out):
        """No length retries: a long answer already costs more (35c)."""
        return []


TASKS = {t.name: t for t in (FileSummary, AbstractFacts)}


def get(name="file-summary", root=ROOT, data=None):
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; one of {sorted(TASKS)}")
    return TASKS[name](root, data)
