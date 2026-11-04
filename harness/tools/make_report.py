#!/usr/bin/env python3
"""Compare scored runs on one split: comparison.csv, comparison.json and a self-contained index.html.

Reads each run's score.json (graded by GPT-6.1-sol), score.sonnet.json (Claude Sonnet 5.5),
run.json and calls.csv. Score and spend are per split, so only runs scored on that split
are comparable. The chart plots score against $ per 100 rows (log scale) with the Pareto
frontier: the runs no other run beats on both. Hover a marker for its numbers; the table
below the chart has everything.

Usage: python3 make_report.py set_dir out_dir [--split test]
"""
import argparse
import csv
import html
import json
import math
import statistics
import sys
from pathlib import Path

# run dir -> (label, group). Groups follow the questions asked: does any high-end open-weight
# model compete with Claude, and how do fast models compare with Lightning.
RUNS = {
    "claude-opus-5-5": ("Claude Opus 5.5 (baseline: Anthropic SDK, effort medium)", "closed"),
    "claude-opus-5-5-converse": ("Claude Opus 5.5", "closed"),
    "claude-sonnet-5-5": ("Claude Sonnet 5.5", "closed"),
    "gpt-6-astra": ("GPT-6 astra", "closed"),
    "gpt-6.1-sol": ("GPT-6.1 sol", "closed"),
    "kimi-k3": ("Kimi K3", "open"),
    "glm-5.3": ("GLM-5.3", "open"),
    "deepseek-v3.2": ("DeepSeek V3.2", "open"),
    "qwen3-235b": ("Qwen3 235B A22B", "open"),
    "mistral-large-3": ("Mistral Large 3", "open"),
    "gpt-oss-120b": ("gpt-oss-120b", "open"),
    "llama4-maverick": ("Llama 4 Maverick", "open"),
    "nemotron-super-3-120b": ("Nemotron Super 3 120B", "open"),
    "claude-haiku-5-5": ("Claude Haiku 5.5", "fast"),
    "gpt-6-luna": ("GPT-6 luna", "fast"),
    "gpt-5.6-luna": ("GPT-5.6 luna", "fast"),
    "nova-2-lite": ("Nova 2 Lite", "fast"),
    "nova-micro": ("Nova Micro", "fast"),
    "llama4-scout": ("Llama 4 Scout", "fast"),
    "gpt-oss-20b": ("gpt-oss-20b", "fast"),
    "nemotron-nano-3-30b": ("Nemotron Nano 3 30B", "fast"),
    "lightning-l40s": ("Nemotron 3.5 Lightning, thinking off (SageMaker ml.g6e.xlarge, L40S)", "fast"),
    "lightning-l40s-thinking": ("Nemotron 3.5 Lightning, thinking on (SageMaker ml.g6e.xlarge, L40S)", "fast"),
    "lightning-l4": ("Nemotron 3.5 Lightning, thinking off (SageMaker ml.g6.2xlarge, L4)", "fast"),
    "lightning-l4-thinking": ("Nemotron 3.5 Lightning, thinking on (SageMaker ml.g6.2xlarge, L4)", "fast"),
    # Superseded: early 48-row runs on an endpoint configured with 8 slots, priced by their own wall clock.
    "lightning": None,
    "lightning-no-thinking": None,
}
SHORT = {"lightning-l40s": "Lightning, L40S", "lightning-l40s-thinking": "Lightning thinking, L40S", "lightning-l4": "Lightning, L4"}
GROUPS = {"closed": "High-end closed", "open": "High-end open-weight", "fast": "Fast and small"}
COLUMNS = ["run", "label", "group", "model", "backend", "mode", "rows", "rows_answered", "score", "score_sonnet",
           "strict", "key_fact_recall", "top2_key_fact_recall", "code_fact_recall", "check_pass",
           "entities_grounded_pct", "usd_per_100_rows", "points_per_usd", "median_latency_s",
           "output_tokens_per_row", "pareto"]


def load_run(rd, split, prefixes):
    s, s2 = json.loads((rd / "score.json").read_text()), json.loads((rd / "score.sonnet.json").read_text())
    block = "test" if s.get("test") and s["test"]["rows"] and s["all"]["rows"] != s["test"]["rows"] else "all"
    a, b = s[block], s2[block]
    run = json.loads((rd / "run.json").read_text())
    calls = [c for c in csv.DictReader(open(rd / "calls.csv", newline="")) if c["prefix"] in prefixes]
    scored = list(csv.DictReader(open(rd / ("scores.csv"), newline="")))
    scored = [r for r in scored if r["split"] == split]
    label, group = RUNS.get(rd.name, (rd.name, "fast"))
    return {
        "run": rd.name, "label": label, "group": group, "model": run.get("model"),
        "backend": run.get("backend", "anthropic-sdk"), "mode": ",".join(run.get("modes", ["structured output"])),
        "rows": a["rows"], "rows_answered": sum(1 for r in scored if r["error"] != "no output row"),
        "score": a["score"], "score_sonnet": b["score"], "strict": a["strict"],
        "key_fact_recall": a["key_fact_recall"], "top2_key_fact_recall": a["top2_key_fact_recall"],
        "code_fact_recall": a["code_fact_recall"], "check_pass": a["check_pass"],
        "entities_grounded_pct": a["entities_grounded_pct"], "usd_per_100_rows": a["usd_per_100_rows"],
        "points_per_usd": a["points_per_usd"],
        "median_latency_s": round(statistics.median(float(c["latency_s"]) for c in calls), 1) if calls else None,
        "output_tokens_per_row": round(sum(int(c["output_tokens"] or 0) for c in calls) / max(1, sum(int(c["rows"]) for c in calls))),
    }


def pareto(rows):
    best, front = -1, set()
    for r in sorted(rows, key=lambda r: (r["usd_per_100_rows"], -r["score"])):
        if r["score"] > best:
            best = r["score"]
            front.add(r["run"])
    return front


def chart(rows):
    W, H, L, R, T, B = 960, 560, 64, 220, 24, 56
    xs = [r["usd_per_100_rows"] for r in rows]
    lo, hi = 10 ** math.floor(math.log10(min(xs))), 10 ** math.ceil(math.log10(max(xs)))
    ymin = max(0, 10 * math.floor((min(r["score"] for r in rows) - 3) / 10))
    ymax = min(100, 10 * math.ceil((max(r["score"] for r in rows) + 3) / 10))
    px = lambda v: L + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - L - R)
    py = lambda v: T + (ymax - v) / (ymax - ymin) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="fig-title" class="chart">']
    for y in range(ymin, ymax + 1, 10):
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{py(y):.1f}" y2="{py(y):.1f}"/>'
                   f'<text class="tick" x="{L - 8}" y="{py(y) + 4:.1f}" text-anchor="end">{y}</text>')
    d = lo
    while d <= hi * 1.0001:
        for m in (1, 3):
            v = d * m
            if lo <= v <= hi * 1.0001:
                lab = f"${v:g}" if v >= 0.01 else f"${v:.3f}"
                out.append(f'<line class="grid" x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{T}" y2="{H - B}"/>'
                           f'<text class="tick" x="{px(v):.1f}" y="{H - B + 18}" text-anchor="middle">{lab}</text>')
        d *= 10
    out.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>')
    out.append(f'<text class="axis-label" x="{(L + W - R) / 2}" y="{H - 12}" text-anchor="middle">$ per 100 rows (log scale)</text>')
    out.append(f'<text class="axis-label" transform="translate(16 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle">Score (GPT-6.1-sol grader)</text>')
    front = sorted((r for r in rows if r["pareto"]), key=lambda r: r["usd_per_100_rows"])
    if front:
        pts = []
        for i, r in enumerate(front):
            if i:
                pts.append(f"{px(r['usd_per_100_rows']):.1f},{py(front[i - 1]['score']):.1f}")
            pts.append(f"{px(r['usd_per_100_rows']):.1f},{py(r['score']):.1f}")
        pts.append(f"{W - R:.1f},{py(front[-1]['score']):.1f}")
        out.append(f'<polyline class="frontier" points="{" ".join(pts)}"/>')
    labelled = []
    for r in sorted(rows, key=lambda r: r["score"]):
        x, y = px(r["usd_per_100_rows"]), py(r["score"])
        tip = html.escape(f"{r['label']}\nscore {r['score']} (Sonnet grader {r['score_sonnet']})\n"
                          f"top-2 facts {r['top2_key_fact_recall']}%, key facts {r['key_fact_recall']}%\n"
                          f"${r['usd_per_100_rows']:.3f} per 100 rows, median call {r['median_latency_s']} s\n"
                          f"{r['rows_answered']}/{r['rows']} rows answered, {r['check_pass']} pass the check")
        out.append(f'<g class="pt {r["group"]}"><title>{tip}</title><circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12"/>')
        if r["run"].startswith("lightning"):
            out.append(f'<path class="mark" d="M{x:.1f} {y - 7:.1f} L{x + 7:.1f} {y:.1f} L{x:.1f} {y + 7:.1f} L{x - 7:.1f} {y:.1f} Z"/>')
        elif r["run"] == "claude-opus-5-5":
            out.append(f'<circle class="mark hollow" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        else:
            out.append(f'<circle class="mark" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        out.append("</g>")
        if r["pareto"] or r["run"].startswith("lightning") or r["run"] in ("claude-haiku-5-5", "nemotron-nano-3-30b"):
            labelled.append((x, y, SHORT.get(r["run"], r["label"].split(" (")[0])))
    marks = [(px(r["usd_per_100_rows"]), py(r["score"])) for r in rows]
    placed, labels = [], []
    for x, y, text in sorted(labelled, key=lambda t: t[1]):
        width = 6.6 * len(text)  # 12px system sans, roughly
        # put the label on the left when another marker sits in its path on the right
        blocked = any(x + 6 < mx < x + 14 + width and abs(my - y) < 10 for mx, my in marks)
        left = blocked and x - 12 - width > L
        ly = y + 18 if blocked and not left else y + 4  # no room on the left: drop below the marker
        while any(abs(ly - p) < 14 and abs(x - px_) < 170 for px_, p in placed):
            ly += 14
        placed.append((x, ly))
        sx, anchor = (x - 12, "end") if left else (x + 12, "start")
        if abs(ly - (y + 4)) > 1:
            labels.append(f'<line class="leader" x1="{x + (-6 if left else 6):.1f}" y1="{y:.1f}" x2="{x + (-10 if left else 10):.1f}" y2="{ly - 4:.1f}"/>')
        labels.append(f'<text class="direct" x="{sx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">{html.escape(text)}</text>')
    # labels sit under the markers, so text never hides a point
    first_mark = next(i for i, s in enumerate(out) if s.startswith('<g class="pt'))
    out[first_mark:first_mark] = labels
    out.append("</svg>")
    return "\n".join(out)


def page(rows, split, set_dir):
    by = lambda g: [r for r in rows if r["group"] == g]
    best = max(rows, key=lambda r: r["score"])
    legend = "".join(f'<span class="key"><svg width="14" height="14"><circle class="mark {g}" cx="7" cy="7" r="5"/></svg>{n}</span>'
                     for g, n in GROUPS.items())
    legend += ('<span class="key"><svg width="16" height="16"><path class="mark fast" d="M8 1 L15 8 L8 15 L1 8 Z"/></svg>Lightning</span>'
               '<span class="key"><svg width="14" height="14"><circle class="mark hollow" cx="7" cy="7" r="5"/></svg>Baseline run (Anthropic SDK)</span>'
               '<span class="key"><svg width="22" height="10"><line class="frontier" x1="0" x2="22" y1="5" y2="5"/></svg>Pareto frontier</span>')
    head = "".join(f"<th>{h}</th>" for h in ["Model", "Group", "Pareto", "Score", "Score (Sonnet grader)", "Top-2 facts", "Key facts",
                                               "Strict", "Answered", "Check pass", "$ / 100 rows", "Points / $", "Median call (s)",
                                               "Output tokens / row", "Output mode"])
    body = "".join(
        f'<tr class="{"front" if r["pareto"] else ""}"><td>{html.escape(r["label"])}</td><td>{GROUPS[r["group"]]}</td><td>{"yes" if r["pareto"] else ""}</td>'
        f'<td>{r["score"]}</td><td>{r["score_sonnet"]}</td><td>{r["top2_key_fact_recall"]}%</td><td>{r["key_fact_recall"]}%</td>'
        f'<td>{r["strict"]}</td><td>{r["rows_answered"]}/{r["rows"]}</td><td>{r["check_pass"]}</td>'
        f'<td>${r["usd_per_100_rows"]:.3f}</td><td>{r["points_per_usd"]:,}</td><td>{r["median_latency_s"]}</td>'
        f'<td>{r["output_tokens_per_row"]:,}</td><td>{html.escape(r["mode"])}</td></tr>'
        for r in sorted(rows, key=lambda r: -r["score"]))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Summarizer model comparison</title>
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --ring:#fcfcfb; --border:rgba(11,11,11,.10); }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19;
  --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --ring:#1a1a19;
  --border:rgba(255,255,255,.10); }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a;
  --axis:#383835; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --ring:#1a1a19; --border:rgba(255,255,255,.10); }}
body {{ margin:0; background:var(--page); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:32px 0 8px; }}
.sub, .note {{ color:var(--ink2); }} .note {{ font-size:13px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; margin-top:16px; }}
.chart {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--axis); stroke-width:1; }}
.tick, .axis-label {{ fill:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
.axis-label {{ fill:var(--ink2); }}
.direct {{ fill:var(--ink2); font-size:12px; paint-order:stroke; stroke:var(--surface); stroke-width:4px; stroke-linejoin:round; }} .leader {{ stroke:var(--muted); stroke-width:1; }}
.frontier {{ fill:none; stroke:var(--muted); stroke-width:2; stroke-linejoin:round; }}
.mark {{ stroke:var(--ring); stroke-width:2; }}
.pt.closed .mark, .mark.closed {{ fill:var(--s1); }}
.pt.open .mark, .mark.open {{ fill:var(--s2); }} .pt.fast .mark, .mark.fast {{ fill:var(--s3); }}
.mark.hollow {{ fill:var(--surface) !important; stroke:var(--s1); stroke-width:2; }}
.hit {{ fill:transparent; }} .pt:hover .mark {{ stroke:var(--ink); }}
.legend {{ display:flex; flex-wrap:wrap; gap:16px; margin:8px 0; color:var(--ink2); font-size:13px; }}
.key {{ display:inline-flex; align-items:center; gap:6px; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:right; padding:6px 8px; border-bottom:1px solid var(--grid); white-space:nowrap; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3), th:last-child, td:last-child {{ text-align:left; }}
th {{ color:var(--ink2); font-weight:600; }} tr.front td:first-child {{ font-weight:600; }}
code {{ font-size:13px; }}
</style></head>
<body><main>
<h1>Summarizer model comparison</h1>
<p class="sub">{len(rows)} runs on the {split} split of <code>sets/pmc-v1</code>: {rows[0]["rows"]} rows of PMC articles and supplements (txt, pdf, docx, xlsx).
Same prompt, batches and output schema for every model; scored against one answer key.</p>

<div class="card">
<div id="fig-title"><strong>Score against cost</strong> <span class="note">Best score: {html.escape(best["label"])}, {best["score"]}. The grey step line is the Pareto frontier: no other run is both cheaper and better.</span></div>
<div class="legend">{legend}</div>
{chart(rows)}
<p class="note">Hover a marker for its numbers. Lightning runs have thinking off unless labelled "thinking"; L40S is ml.g6e.xlarge, L4 is ml.g6.2xlarge. Small gaps can be noise: compare runs with paired intervals resampled by article before calling one better.</p>
</div>

<h2>All runs</h2>
<div class="card tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>

<h2>How to read it</h2>
<ul class="note">
<li><strong>Score</strong>: mean over rows of the fraction of required facts the row conveys (code facts plus 3–5 ranked key facts), graded by GPT-6.1-sol. The second column is the same run graded by Claude Sonnet 5.5. A row a model didn't return scores 0.</li>
<li><strong>Top-2 facts</strong>: what the file is and what it's about. <strong>Strict</strong>: rows with every fact. <strong>Check pass</strong>: the schema, length and quote check (<code>check_outputs.py</code>).</li>
<li><strong>Spend</strong>: Bedrock tokens times on-demand list rates (<code>tools/model_rates.json</code>), with cache tokens at the input rate. Lightning is costed by instance time at sustained throughput: the instance's hourly rate over the rows per hour it sustains with every request slot busy, measured over all 262 rows. That's what Bedrock's per-token prices assume of their own hardware. An endpoint left idle still bills by the hour.</li>
<li><strong>Output mode</strong>: <code>tool</code> is a forced tool call; <code>tool-auto</code> means the model rejected forcing and was asked to call the tool; <code>json_schema</code> is vLLM guided decoding.</li>
<li>Data: <code>comparison.csv</code> and <code>comparison.json</code> beside this page; per-run files under <code>{html.escape(set_dir)}/runs/</code>.</li>
</ul>
</main></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    sd = Path(args.set_dir)
    prefixes = {r["prefix"] for r in csv.DictReader(open(sd / "split.csv", newline="")) if r["split"] == args.split}
    rows = [load_run(rd, args.split, prefixes) for rd in sorted((sd / "runs").iterdir())
            if RUNS.get(rd.name, ()) is not None and (rd / "score.json").exists() and (rd / "score.sonnet.json").exists()]
    front = pareto(rows)
    for r in rows:
        r["pareto"] = r["run"] in front
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["score"]))
    (out / "comparison.json").write_text(json.dumps({"split": args.split, "runs": sorted(rows, key=lambda r: -r["score"])}, indent=1) + "\n")
    (out / "index.html").write_text(page(rows, args.split, "sets/pmc-v1"))
    for r in sorted(rows, key=lambda r: -r["score"]):
        print(f'{r["score"]:5.1f} {r["score_sonnet"]:5.1f} top2 {r["top2_key_fact_recall"]:5.1f} ${r["usd_per_100_rows"]:8.3f} '
              f'{r["median_latency_s"]!s:>6}s {r["rows_answered"]:>2}/{r["rows"]} {"P" if r["pareto"] else " "} {r["label"]}')


if __name__ == "__main__":
    main()
