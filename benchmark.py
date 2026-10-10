"""Run Jev and OpenAI Decisions side by side on the sample and compare them.

Run:  uv run benchmark.py --limit 100      (omit --limit for all 1,000 rows)
Prints a short comparison in the terminal and writes the full report to RESULTS.md.
"""

import argparse
import asyncio
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

import jev
import openai_client
from dataset import FLAG_NAMES, QUESTIONS, load_sample

PROVIDERS = {"Jev": jev, "OpenAI": openai_client}
RESULTS_PATH = Path("RESULTS.md")
WIN = " ◀"
MAX_LEVEL = len(QUESTIONS["tone"]["levels"]) - 1

PLAIN_QUESTIONS = {
    "intent": f"What does the customer want? ({len(QUESTIONS['intent']['options'])} options)",
    "category": f"Which topic is it? ({len(QUESTIONS['category']['options'])} options)",
    "tone": "How polite is it? (rude / neutral / polite)",
    "is_offensive": "Is it offensive?",
    "has_typos": "Does it have typos?",
    "is_keyword_only": "Is it just keywords?",
    "has_negation": "Does it contain a negation?",
}
TRICKY_MESSAGES = {
    "Z": 'typos ("how cna i…")',
    "Q": 'slang ("can u…")',
    "W": "swearing",
    "K": 'only keywords ("track order")',
}
STYLE_FLAGS = ["Z", "Q", "W", "K", "N", "P", "I", "C", "E"]
CONFIDENCE_BUCKETS = [("≥ 0.9", 0.9, 1.01), ("0.5 – 0.9", 0.5, 0.9), ("< 0.5", 0.0, 0.5)]

TESTED = (
    "Each model read a customer support message and answered 7 questions about it in a single request. "
    "Both models got exactly the same message and the same questions; their answers were checked against "
    "the dataset's labels. The models never see the labels."
)
CORRECT_NOTE = (
    "\"Correct\" for yes/no questions is the average of how often the model was right on yes-messages and on "
    "no-messages, so a model that always says \"no\" scores 50%, not 98%. For politeness the model's score is "
    "rounded to the nearest level. Failed requests count as wrong."
)
NOTES = [
    "Flag letters are Bitext's own Language Generation Tags (legend below). Models only see the message text.",
    "Choice: exact match. Noul: probability ≥ 0.5 counts as yes; balanced accuracy = mean of yes-row and no-row accuracy.",
    "tone is approximate: expected level comes from flags (W → 0, P → 2, otherwise 1); rows with both W and P are skipped. "
    "MAE is averaged per level so the many neutral rows do not dominate.",
    "has_typos is approximate: Bitext tags slang like \"ur\" as colloquial (Q), not as a typo (Z).",
    "Errors and refusals count as wrong answers. Latency includes SDK retries; total time depends on the account's rate-limit tier.",
]

console = Console()


# ---------- running ----------

async def run_provider(module, rows, concurrency, progress, task):
    semaphore = asyncio.Semaphore(concurrency)
    cost = 0.0

    async def one(row):
        nonlocal cost
        async with semaphore:
            result = await module.ask(client, row["text"])
        cost += result.get("cost_usd", 0)
        progress.update(task, advance=1, cost=f"${cost:.4f}")
        return result

    async with module.make_client() as client:
        await module.ask(client, rows[0]["text"])  # warm-up, not measured
        start = time.perf_counter()
        results = await asyncio.gather(*(one(row) for row in rows))
        return {"results": results, "wall_s": time.perf_counter() - start}


async def run_all(rows, concurrency):
    progress = Progress(
        TextColumn("  {task.description:<8}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[cost]}"),
        console=console,
    )
    with progress:
        jobs = {
            name: run_provider(module, rows, concurrency, progress, progress.add_task(name, total=len(rows), cost="$0"))
            for name, module in PROVIDERS.items()
        }
        runs = await asyncio.gather(*jobs.values())
    return dict(zip(jobs, runs))


# ---------- metrics ----------

def value(result, question):
    return result.get("answers", {}).get(question, {}).get("value")


def mean(values):
    return sum(values) / len(values) if values else None


def choice_accuracy(rows, results, question):
    return mean([value(res, question) == row[question] for row, res in zip(rows, results)])


def noul_accuracy(rows, results, question):
    yes, no = [], []
    for row, res in zip(rows, results):
        v = value(res, question)
        if row[question] == "1":
            yes.append(v is not None and v >= 0.5)
        else:
            no.append(v is not None and v < 0.5)
    yes_acc, no_acc = mean(yes), mean(no)
    balanced = (yes_acc + no_acc) / 2 if yes_acc is not None and no_acc is not None else None
    return balanced, yes_acc, no_acc


def score_by_level(rows, results, question):
    """Per level: (MAE, share of answers that round to the right level)."""
    per_level = {}
    for level in range(MAX_LEVEL + 1):
        errors, hits = [], []
        for row, res in zip(rows, results):
            if row[question] != str(level):
                continue
            v = value(res, question)
            errors.append(abs(v - level) if v is not None else max(level, MAX_LEVEL - level))
            hits.append(v is not None and int(v + 0.5) == level)
        per_level[level] = (mean(errors), mean(hits))
    return per_level


def correct(rows, results, question):
    """The single "Correct %" shown in the main table."""
    kind = QUESTIONS[question]["type"]
    if kind == "choice":
        return choice_accuracy(rows, results, question)
    if kind == "noul":
        return noul_accuracy(rows, results, question)[0]
    return mean([hit for _, hit in score_by_level(rows, results, question).values() if hit is not None])


def latency_percentiles(results):
    latencies = sorted(r["latency_ms"] for r in results if "error" not in r)
    if len(latencies) < 2:
        return None, None, None
    q = statistics.quantiles(latencies, n=100)
    return q[49], q[94], q[98]


def stats(run):
    results = run["results"]
    ok = [r for r in results if "error" not in r]
    p50, p95, p99 = latency_percentiles(results)
    return {
        "model": next((r["model"] for r in ok), "–"),
        "cost": sum(r.get("cost_usd", 0) for r in results),
        "wall_s": run["wall_s"],
        "errors": len(results) - len(ok),
        "refusals": sum(any(a.get("refusal") for a in r["answers"].values()) for r in ok),
        "p50": p50, "p95": p95, "p99": p99,
        "rps": len(ok) / run["wall_s"],
        "tokens": mean([r["input_tokens"] for r in ok]),
    }


# ---------- formatting ----------

def compare(values, fmt, higher_better=True):
    """Format one value per provider and mark the single best one (no mark if it looks tied once formatted)."""
    valid = [v for v in values if v is not None]
    best = None
    if len(valid) == len(values):
        top = max(valid) if higher_better else min(valid)
        if [fmt(v) for v in valid].count(fmt(top)) == 1:
            best = top
    return ["–" if v is None else fmt(v) + (WIN if v == best else "") for v in values]


def pct(v):
    return f"{v * 100:.1f}%"


def ms(v):
    return f"{v:.0f} ms"


def seconds(v):
    return f"{v / 1000:.2f} s"


def duration(s):
    return f"{int(s // 60)} min {int(s % 60):02d} s"


def ratio(names, values, word):
    """('Jev', '2.8× cheaper') for lower-is-better values."""
    if None in values or min(values) <= 0:
        return None
    return names[values.index(min(values))], f"{max(values) / min(values):.1f}× {word}"


def headline(names, overall, s, n_rows):
    scores = [f"{name} {pct(v)}" for name, v in zip(names, overall) if v is not None]
    text = f"On {n_rows:,} customer support messages, **{scores[0]} correct**" + "".join(f" vs **{x}**" for x in scores[1:]) + "."
    wins = [w for w in (ratio(names, [x["cost"] for x in s], "cheaper"),
                        ratio(names, [x["p50"] for x in s], "faster per request")) if w]
    if len(wins) == 2 and wins[0][0] == wins[1][0]:
        text += f" {wins[0][0]} was {wins[0][1]} and {wins[1][1]}."
    else:
        text += "".join(f" {name} was {phrase}." for name, phrase in wins)
    return text


def example(rows, names, results):
    """A real message with every question's expected and actual answers."""
    index = next((i for i, row in enumerate(rows) if "Q" in row["flags"]), 0)
    row = rows[index]

    def show(question, v):
        if v is None or v == "":
            return "–"
        kind = QUESTIONS[question]["type"]
        if kind == "noul":
            return "yes" if float(v) >= 0.5 else "no"
        if kind == "score":
            return QUESTIONS[question]["levels"][int(float(v) + 0.5)][0].lower()
        return v

    table = [[PLAIN_QUESTIONS[q], show(q, row[q]), *[show(q, value(r[index], q)) for r in results]] for q in QUESTIONS]
    return row["text"], table


def build_report(rows, runs):
    names = list(runs)
    results = [runs[n]["results"] for n in names]
    s = [stats(runs[n]) for n in names]

    def per(fn):
        return [fn(r) for r in results]

    # ----- main -----
    question_scores = {q: per(lambda r: correct(rows, r, q)) for q in QUESTIONS}
    overall = [mean([scores[i] for scores in question_scores.values() if scores[i] is not None])
               for i in range(len(names))]
    main = [
        {
            "title": "Results",
            "headers": ["Question", *names],
            "rows": [[PLAIN_QUESTIONS[q], *compare(v, pct)] for q, v in question_scores.items()]
                    + [["**Overall (average)**", *compare(overall, pct)]],
            "note": CORRECT_NOTE,
        },
        {
            "title": "Speed & cost",
            "headers": ["", *names],
            "rows": [
                [f"Cost for {len(rows):,} messages", *compare([x["cost"] for x in s], lambda v: f"${v:.2f}", False)],
                ["Typical response time", *compare([x["p50"] for x in s], seconds, False)],
                [f"Time to process all {len(rows):,}", *compare([x["wall_s"] for x in s], duration, False)],
            ],
        },
    ]
    tricky_rows = []
    for flag, label in TRICKY_MESSAGES.items():
        subset = [i for i, row in enumerate(rows) if flag in row["flags"]]
        if subset:
            accs = [mean([value(r[i], "intent") == rows[i]["intent"] for i in subset]) for r in results]
            tricky_rows.append([label, f"{len(subset):,}", *compare(accs, pct)])
    main.append({
        "title": "Tricky messages: did they still understand what the customer wants?",
        "headers": ["Messages with…", "Count", *names],
        "rows": tricky_rows,
    })

    # ----- details -----
    details = []
    technical = [["Errors", "", "count", *compare([x["errors"] for x in s], str, False)],
                 ["Refusals", "", "count", *[str(x["refusals"]) for x in s]]]
    for question, spec in QUESTIONS.items():
        if spec["type"] == "choice":
            technical.append([question, "choice", "accuracy", *compare(question_scores[question], pct)])
        elif spec["type"] == "noul":
            nouls = per(lambda r: noul_accuracy(rows, r, question))
            yes_rows = sum(row[question] == "1" for row in rows)
            technical.append([question, "noul", "balanced acc.", *compare([n[0] for n in nouls], pct)])
            technical.append([f"↳ yes rows ({yes_rows})", "", "accuracy", *compare([n[1] for n in nouls], pct)])
            technical.append([f"↳ no rows ({len(rows) - yes_rows})", "", "accuracy", *compare([n[2] for n in nouls], pct)])
        else:
            levels = per(lambda r: score_by_level(rows, r, question))
            maes = [mean([m for m, _ in lv.values() if m is not None]) for lv in levels]
            technical.append([question, "score", "MAE ↓", *compare(maes, "{:.2f}".format, False)])
            for level, (label, _) in enumerate(spec["levels"]):
                count = sum(row[question] == str(level) for row in rows)
                technical.append([f"↳ {level} {label} ({count})", "", "MAE ↓",
                                  *compare([lv[level][0] for lv in levels], "{:.2f}".format, False)])
    details.append({"title": "Per-question metrics", "headers": ["Question", "Type", "Metric", *names], "rows": technical})

    style_rows = []
    for flag in STYLE_FLAGS:
        subset = [i for i, row in enumerate(rows) if flag in row["flags"]]
        if subset:
            accs = [mean([value(r[i], "intent") == rows[i]["intent"] for i in subset]) for r in results]
            style_rows.append([f"{FLAG_NAMES[flag]} ({flag})", str(len(subset)), *compare(accs, pct)])
    details.append({"title": "Intent accuracy by message style", "headers": ["Style", "Rows", *names], "rows": style_rows})

    confusions = [Counter((row["intent"], value(res, "intent")) for row, res in zip(rows, r)
                          if value(res, "intent") not in (None, row["intent"])) for r in results]
    top = sum(confusions, Counter()).most_common(5)
    details.append({
        "title": "Most confused intents",
        "headers": ["Expected → Predicted", *names],
        "rows": [[f"{exp} → {pred}", *[str(c[(exp, pred)]) for c in confusions]] for (exp, pred), _ in top],
    })

    confidence_rows = []
    for label, low, high in CONFIDENCE_BUCKETS:
        cells = []
        for r in results:
            hits = [value(res, "intent") == row["intent"] for row, res in zip(rows, r)
                    if low <= res.get("answers", {}).get("intent", {}).get("confidence", -1) < high]
            cells += [str(len(hits)), pct(mean(hits)) if hits else "–"]
        confidence_rows.append([label, *cells])
    details.append({
        "title": "Intent confidence: does \"sure\" mean \"right\"?",
        "headers": ["Confidence", *[f"{n} {col}" for n in names for col in ("rows", "accuracy")]],
        "rows": confidence_rows,
    })

    details.append({
        "title": "Latency & throughput",
        "headers": ["", *names],
        "rows": [
            ["Latency p50", *compare([x["p50"] for x in s], ms, False)],
            ["Latency p95", *compare([x["p95"] for x in s], ms, False)],
            ["Latency p99", *compare([x["p99"] for x in s], ms, False)],
            ["Requests / sec", *compare([x["rps"] for x in s], "{:.1f}".format)],
            ["Avg input tokens", *compare([x["tokens"] for x in s], "{:,.0f}".format, False)],
            ["Cost per 1K messages", *compare([x["cost"] / len(rows) * 1000 for x in s], lambda v: f"${v:.3f}", False)],
        ],
    })

    return {
        "headline": headline(names, overall, s, len(rows)),
        "example": example(rows, names, results),
        "main": main,
        "details": details,
        "models": [x["model"] for x in s],
        "names": names,
    }


# ---------- output ----------

def print_report(report):
    console.print()
    console.print(report["headline"].replace("**", ""), style="bold")
    console.print()
    for section in report["main"]:
        table = Table(title=section["title"], title_justify="left", title_style="bold", box=box.ROUNDED,
                      header_style="dim")
        for i, h in enumerate(section["headers"]):
            table.add_column(h, justify="left" if i == 0 else "right")
        for row in section["rows"]:
            table.add_row(*[f"[green]{c}[/]" if c.endswith(WIN) else c.replace("**", "") for c in row])
        console.print(table)
    console.print(f"[dim]Full report with details written to {RESULTS_PATH}[/]\n")


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def write_results(report, run_info):
    text, example_rows = report["example"]
    parts = [
        "# Jev vs OpenAI Decisions — Results", "",
        report["headline"], "",
        "## What was tested?", "",
        TESTED, "",
        f"Example message: *\"{text}\"*", "",
        markdown_table(["Question", "Expected", *report["names"]], example_rows), "",
    ]
    for section in report["main"]:
        parts += [f"## {section['title']}", "", markdown_table(section["headers"], section["rows"]), ""]
        if section.get("note"):
            parts += [f"<sub>{section['note']}</sub>", ""]
    parts += ["<details>", "<summary><b>Details</b> (technical metrics, notes, run info)</summary>", ""]
    for section in report["details"]:
        parts += [f"### {section['title']}", "", markdown_table(section["headers"], section["rows"]), ""]
    parts += ["### Notes", "", *[f"- {note}" for note in NOTES], ""]
    parts += [markdown_table(["Flag", "Meaning"], [[k, v] for k, v in FLAG_NAMES.items()]), ""]
    parts += ["### Run info", "", run_info, "", "</details>", ""]
    RESULTS_PATH.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="number of sample rows to run (default: all)")
    parser.add_argument("--concurrency", type=int, default=16, help="max parallel requests per provider")
    args = parser.parse_args()

    rows = load_sample(args.limit)
    console.print(f"\n[bold]Jev vs OpenAI Decisions[/]  ·  {len(rows):,} messages  ·  concurrency {args.concurrency}\n")
    runs = asyncio.run(run_all(rows, args.concurrency))

    for name, run in runs.items():
        errors = [r["error"] for r in run["results"] if "error" in r]
        if errors:
            console.print(f"[yellow]{name}: {len(errors)} failed requests, first: {errors[0][:200]}[/]")

    report = build_report(rows, runs)
    print_report(report)
    run_info = (f"{datetime.now():%Y-%m-%d %H:%M} · {len(rows):,} messages · {' vs '.join(report['models'])} · "
                f"concurrency {args.concurrency} · dataset: Bitext Customer Support (data/sample.csv)")
    write_results(report, run_info)


if __name__ == "__main__":
    main()
