#!/usr/bin/env python3
"""Judge a fine-tuned Lightning run against the pass/fail bar fixed before training (25a; METHODOLOGY.md §4).

Reads scored runs on all 262 rows of ingest sets/pmc-v1 (score.json and score.sonnet.json, or
score.all.json and score.all-sonnet.json for the NVFP4 sweep run) and prints a JSON verdict:
each run's numbers, paired bootstrap gaps (resampled by article) between the fine-tune and the
references, and each criterion of the bar with pass or fail.

--bar 25c judges against 25c's bar instead: top-2 key-fact recall within 3 of Opus under both
graders, plus the quote, rows-returned and token limits. Gaps are then in top-2 recall points.

Usage: python3 compare_ft.py <ingest set dir> <ft run name> [--bar 25a|25c]
"""
import csv
import json
import random
import statistics
import sys
from pathlib import Path

REFS = {"base-nvfp4": ("lightning-l40s", ".all"), "base-bf16": ("lightning-base-bf16", ""),
        "haiku": ("claude-haiku-5-5-full", ""), "gpt-6-luna": ("gpt-6-luna-full", ""), "opus": ("claude-opus-5-5", "")}


def load(rd, tag, grader):
    sfx = {"gpt": tag, "sonnet": (tag + "-sonnet") if tag else ".sonnet"}[grader]
    rows = {r["unit_id"]: r for r in csv.DictReader(open(rd / f"scores{sfx}.csv"))}
    summary = json.loads((rd / f"score{sfx}.json").read_text())["all"]
    return rows, summary


def paired(a, b, units, rng, metric="score"):
    """Point gap and 95% interval, resampling articles. metric: score (mean of row scores) or
    top2 (top-2 facts conveyed over top-2 facts, pooled)."""
    arts = sorted({u.split("/")[3] for u in units})
    by = {x: [u for u in units if u.split("/")[3] == x] for x in arts}

    def gap(us):
        if metric == "score":
            return 100 * sum(float(a[u]["score"]) - float(b[u]["score"]) for u in us) / len(us)
        n = sum(int(a[u]["top2_facts"]) for u in us) or 1
        return 100 * (sum(int(a[u]["top2_conveyed"]) for u in us) - sum(int(b[u]["top2_conveyed"]) for u in us)) / n

    diffs = sorted(gap([u for x in (rng.choice(arts) for _ in arts) for u in by[x]]) for _ in range(4000))
    return round(gap(units), 1), round(diffs[100], 1), round(diffs[3899], 1)


def main(set_dir, ft, bar_version="25a"):
    sd = Path(set_dir) / "runs"
    runs = {"ft": (ft, ""), **REFS}
    out = {"runs": {}, "paired_vs_ft": {}, "bar": {}}
    data = {}
    for key, (name, tag) in runs.items():
        for grader in ("gpt", "sonnet"):
            data[(key, grader)] = load(sd / name, tag, grader)
        run = json.loads((sd / name / "run.json").read_text())
        g, s = data[(key, "gpt")][1], data[(key, "sonnet")][1]
        out["runs"][key] = {"run": name, "rows": g["rows"], "score_gpt": g["score"], "score_sonnet": s["score"],
                            "top2": g["top2_key_fact_recall"], "top2_sonnet": s["top2_key_fact_recall"],
                            "score_top2_gpt": g.get("score_top2"), "score_top2_sonnet": s.get("score_top2"),
                            "key_facts": g["key_fact_recall"], "check_pass": g["check_pass"],
                            "rows_returned": run.get("output_rows"), "output_tokens_per_row":
                                round(run["output_tokens"] / max(1, run["output_rows"])) if run.get("output_tokens") else None}
    units = sorted(data[("ft", "gpt")][0])
    rng = random.Random(0)
    metric = "top2" if bar_version == "25c" else "score"
    out["paired_metric"] = metric
    for key in REFS:
        out["paired_vs_ft"][key] = {g: paired(data[("ft", g)][0], data[(key, g)][0], units, rng, metric) for g in ("gpt", "sonnet")}
    r = out["runs"]
    bar = out["bar"]
    if bar_version == "25c":
        for g, field in (("gpt", "top2"), ("sonnet", "top2_sonnet")):
            need = round(r["opus"][field] - 3, 1)
            bar[f"top-2 recall within 3 of Opus ({g})"] = {"need": need, "got": r["ft"][field], "pass": r["ft"][field] >= need}
    for g in ("gpt", "sonnet") if bar_version == "25a" else ():
        base = r["base-nvfp4"][f"score_{g}"]
        best_fast = max(r["haiku"][f"score_{g}"], r["gpt-6-luna"][f"score_{g}"])
        bar[f"+7 over base ({g})"] = {"need": round(base + 7, 1), "got": r["ft"][f"score_{g}"], "pass": r["ft"][f"score_{g}"] >= base + 7}
        bar[f"within 3 of best fast ({g})"] = {"need": round(best_fast - 3, 1), "got": r["ft"][f"score_{g}"],
                                               "pass": r["ft"][f"score_{g}"] >= best_fast - 3}
        bar[f"dream: within 3 of Opus ({g})"] = {"need": round(r["opus"][f"score_{g}"] - 3, 1), "got": r["ft"][f"score_{g}"],
                                                 "pass": r["ft"][f"score_{g}"] >= r["opus"][f"score_{g}"] - 3}
    bar["quote check >= 85%"] = {"need": round(0.85 * r["ft"]["rows"]), "got": r["ft"]["check_pass"],
                                 "pass": r["ft"]["check_pass"] >= 0.85 * r["ft"]["rows"]}
    bar["rows returned >= 99%"] = {"need": round(0.99 * r["ft"]["rows"]), "got": r["ft"]["rows_returned"],
                                   "pass": (r["ft"]["rows_returned"] or 0) >= 0.99 * r["ft"]["rows"]}
    bar["<= 800 output tokens per row"] = {"need": 800, "got": r["ft"]["output_tokens_per_row"],
                                           "pass": (r["ft"]["output_tokens_per_row"] or 1e9) <= 800}
    out["verdict"] = "PASS" if all(v["pass"] for k, v in bar.items() if not k.startswith("dream")) else "FAIL"
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    argv = sys.argv[1:]
    bar = "25a"
    if "--bar" in argv:
        i = argv.index("--bar")
        bar = argv[i + 1] if i + 1 < len(argv) else ""
        argv = argv[:i] + argv[i + 2:]
    if len(argv) != 2 or bar not in ("25a", "25c"):
        sys.exit(__doc__)
    main(*argv, bar)
