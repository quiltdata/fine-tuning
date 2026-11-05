"""The tasks a model can be run on: prompt, examples, schemas, what the model sees, batching, limits.

  file-summary    the earlier summarizer: one row per file, batches of one article folder and file
                  type (prompts/summarizer.md, schemas/output.schema.json).
  abstract-facts  the abstract task: five abstracts per call by default (35e), as plain numbered text;
                  the model returns one JSON object of answers, each with its abstract's number, one
                  `about` fact and zero to two `findings` in its own words, with an optional quote. The
                  harness matches answers to abstracts by number. Length is reported, not limited or
                  retried (35c).

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
    required_note = ("Required content: each answer must convey these facts about its abstract, the about fact and the "
                     "findings, stated in your own words. Each fact stays one claim; quotes stay optional and verbatim.")
    text_instruction = ('\n\nReply with only a JSON object of the form {"answers": [{"abstract": 1, "about": {...}, '
                        '"findings": [...]}, ...]}, one answer per abstract, and nothing else.')
    batch_size = 5

    def batches(self, rows, per_file=False, size=None):
        """Batches of `size` abstracts in input order (default 5, 35e; 1 with per_file); keys ("batch-<n>", "abstract")."""
        n = 1 if per_file else (size or self.batch_size)
        return [((f"batch-{i // n}", "abstract"), rows[i:i + n]) for i in range(0, len(rows), n)]

    def user_message(self, rows, required=None):
        """The abstracts as plain text, numbered from 1: no ids, no JSON. A teacher run adds, by number,
        the facts each answer must convey; make_pairs.py never passes them, so the student never sees them."""
        msg = "\n\n".join(f"Abstract {i}:\n{r['abstract']}" for i, r in enumerate(rows, 1))
        facts = {str(i): required[r["unit_id"]] for i, r in enumerate(rows, 1) if required and required.get(r["unit_id"])}
        if facts:
            msg += "\n\n" + self.required_note + "\n" + json.dumps(facts, ensure_ascii=False)
        return msg

    def api_schema(self):
        """Written out rather than derived: strict structured output (Bedrock Chat Completions) rejects
        optional properties and $ref, so `quote` is always present and "" means no quote. Answers carry
        the abstract's number, not an id: the harness knows which abstract each number is."""
        fact = {"type": "object", "properties": {"fact": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["fact", "quote"], "additionalProperties": False}
        answer = {"type": "object", "properties": {"abstract": {"type": "integer"}, "about": fact,
                                                   "findings": {"type": "array", "items": fact}},
                  "required": ["abstract", "about", "findings"], "additionalProperties": False}
        return {"type": "object", "properties": {"answers": {"type": "array", "items": answer}},
                "required": ["answers"], "additionalProperties": False}

    def rows_from_reply(self, data):
        """The answers: {"answers": [...]}, a bare list, or one answer object (one abstract)."""
        if isinstance(data, dict) and "answers" in data:
            data = data["answers"]
            if isinstance(data, str):
                data = json.loads(data)
        if isinstance(data, dict) and "about" in data:
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError("no answers in reply")

    def attach_ids(self, out, batch):
        """Each answer gets the id of the abstract its number names. An answer with a number outside the
        batch, or a repeat, is dropped; with one abstract, an unnumbered answer is taken as its answer."""
        rows, seen = [], set()
        for o in out:
            if not isinstance(o, dict):
                continue
            n = o.get("abstract", 1 if len(batch) == 1 else None)
            try:
                n = int(n)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= len(batch) and n not in seen:
                seen.add(n)
                rows.append({**{k: v for k, v in o.items() if k != "abstract"}, "unit_id": batch[n - 1]["unit_id"]})
        return rows

    def assistant_json(self, rows):
        """The answers numbered in batch order, without ids, as the model gives them."""
        return {"answers": [{"abstract": i, **{k: v for k, v in r.items() if k != "unit_id"}} for i, r in enumerate(rows, 1)]}

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
