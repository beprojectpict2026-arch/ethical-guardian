"""End-to-end demo: from biased hiring history to agent evaluation, in one command.

    uv run python -m examples.hiring_demo            # full run (about a minute)
    uv run python -m examples.hiring_demo --quick    # smaller, faster run
    uv run python -m examples.hiring_demo --llm      # also evaluate the LLM screener (Ollama)

Steps:
  1. Generate a company's hiring history with biased past decisions.
  2. Data phase: check the history before any agent is built.
  3. Train ML screeners on that history.
  4. Testing phase: evaluate the reference agent, the ML screener (protected attributes
     removed), the ML screener retrained without postcode (the fix suggested by the data
     check), and an ML screener given the protected attributes.
  5. Save every report as HTML and JSON, plus an index page with a summary and a chart of
     historical parity against bias strength.
"""

from __future__ import annotations

import argparse
import html
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import eguard
from eguard import Report
from eguard.metrics import demographic_parity_ratio
from eguard.scenarios.hiring import SkillBasedScreener, generate_candidates, historical_records
from examples.agents.ml_screener import MLScreener

MANIFEST = Path(__file__).resolve().parent / "manifests" / "hiring_agent.yaml"
BIAS_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


@dataclass
class Entry:
    label: str
    description: str
    report: Report
    stem: str


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    out = Path(args.out)
    profile = eguard.AgentProfile.from_yaml(MANIFEST)
    history_size = 2_000 if args.quick else 10_000
    phase = "development" if args.quick else "testing"

    print(f"1. Generating {history_size:,} past applicants (bias strength {args.bias}) ...")
    pool = generate_candidates(history_size, seed=args.seed, bias_strength=args.bias)
    history = historical_records(pool)

    print("2. Data phase: checking the history before any agent is built ...")
    data_report = eguard.check(profile, "data", data=history, target="hired")
    entries = [
        Entry(
            "Historical data",
            "Data-phase check of past hiring decisions",
            data_report,
            "01_data",
        )
    ]
    print(f"   {'Historical data':26} {_verdict(data_report)}")

    print("3. Training ML screeners on the history ...")
    agents = [
        (
            "Reference agent",
            "Scores only skills, experience and education",
            SkillBasedScreener(),
        ),
        (
            "ML screener",
            "Logistic regression, protected attributes removed",
            MLScreener().fit(history),
        ),
        (
            "ML screener, no postcode",
            "Retrained without the proxy flagged by the data check",
            MLScreener(drop=["postcode"]).fit(history),
        ),
        (
            "ML screener, aware",
            "Logistic regression given gender and region",
            MLScreener(use_protected=True).fit(history),
        ),
    ]
    if args.llm:
        from examples.agents.llm_screener import LLMScreener

        agents.append(
            (
                "LLM screener",
                f"{args.llm_model} scoring applicant descriptions",
                LLMScreener(args.llm_model),
            )
        )

    print(f"4. {phase.capitalize()} phase: evaluating {len(agents)} agents ...")
    for number, (label, text, agent) in enumerate(agents, start=2):
        options = {}
        if label == "LLM screener":
            options = {"seeds": (1000, 1001), "n_applicants": args.llm_applicants}
        report = eguard.evaluate(agent, profile, phase, **options)
        entries.append(Entry(label, text, report, f"{number:02d}_{_slug(label)}"))
        print(f"   {label:26} {_verdict(report)}")

    print(f"5. Saving reports to {out}/ ...")
    out.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        entry.report.to_html(out / f"{entry.stem}.html")
        entry.report.to_json(out / f"{entry.stem}.json")
    curve = _parity_curve(history_size, args.seed)
    (out / "index.html").write_text(_index_page(entries, curve, args.bias), encoding="utf-8")

    print(f"\nDone. Open {out / 'index.html'} in your browser.")
    return 0


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ethical Guardian Suite hiring demo.")
    parser.add_argument("--bias", type=float, default=1.0, help="bias in past hiring (0-3)")
    parser.add_argument("--seed", type=int, default=1, help="seed for the hiring history")
    parser.add_argument("--out", default="reports/demo", help="output folder")
    parser.add_argument("--quick", action="store_true", help="smaller, faster run")
    parser.add_argument("--llm", action="store_true", help="also evaluate the LLM screener")
    parser.add_argument("--llm-model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument(
        "--llm-applicants", type=int, default=200, help="applicants per seed for the LLM"
    )
    return parser.parse_args(argv)


def _verdict(report: Report) -> str:
    risk = report.composite_risk
    score = "not evaluated" if risk is None else f"risk {risk:.2f} ({report.risk_tier.value})"
    return f"{score} | {'passed' if report.passed else 'BLOCKED'}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _parity_curve(n: int, seed: int) -> list[tuple[float, float, float]]:
    """Historical hiring DPR for gender and region at each bias level (same applicants)."""
    curve = []
    for bias in BIAS_LEVELS:
        df = generate_candidates(n, seed=seed, bias_strength=bias)
        curve.append(
            (
                bias,
                demographic_parity_ratio(df["hired"], df["gender"]),
                demographic_parity_ratio(df["hired"], df["region"]),
            )
        )
    return curve


# ---------------------------------------------------------------------------- index page


def _chart(curve: list[tuple[float, float, float]], chosen: float) -> str:
    """Inline SVG line chart of DPR against bias strength."""
    width, height, left, right, top, bottom = 600, 300, 52, 20, 16, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    max_bias = max(b for b, _, _ in curve)

    def x(v: float) -> float:
        return left + v / max_bias * plot_w

    def y(v: float) -> float:
        return top + (1 - v) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Historical demographic parity ratio against bias strength">'
    ]
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" '
            f'y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
            f'<text class="tick" x="{left - 8}" y="{y(tick) + 4:.1f}" '
            f'text-anchor="end">{tick:.1f}</text>'
        )
    for bias, _, _ in curve:
        parts.append(
            f'<text class="tick" x="{x(bias):.1f}" y="{height - bottom + 18}" '
            f'text-anchor="middle">{bias:g}</text>'
        )
    parts.append(
        f'<text class="tick" x="{left + plot_w / 2:.1f}" y="{height - 6}" '
        'text-anchor="middle">bias strength in past hiring</text>'
    )
    parts.append(
        f'<line class="rule" x1="{left}" x2="{width - right}" '
        f'y1="{y(0.8):.1f}" y2="{y(0.8):.1f}"/>'
        f'<text class="rule-label" x="{width - right}" y="{y(0.8) - 6:.1f}" '
        'text-anchor="end">four-fifths rule (0.8)</text>'
    )
    if 0 <= chosen <= max_bias:
        parts.append(
            f'<line class="chosen" x1="{x(chosen):.1f}" x2="{x(chosen):.1f}" '
            f'y1="{top}" y2="{height - bottom}"/>'
        )
    for index, cls in ((1, "gender"), (2, "region")):
        points = " ".join(f"{x(row[0]):.1f},{y(row[index]):.1f}" for row in curve)
        parts.append(f'<polyline class="{cls}" points="{points}"/>')
        parts.extend(
            f'<circle class="{cls}" cx="{x(row[0]):.1f}" cy="{y(row[index]):.1f}" r="3.5"/>'
            for row in curve
        )
    parts.append("</svg>")
    return "".join(parts)


def _index_page(entries: list[Entry], curve: list[tuple[float, float, float]], bias: float) -> str:
    rows = []
    for entry in entries:
        risk = entry.report.composite_risk
        tier = entry.report.risk_tier.value if entry.report.risk_tier else "none"
        outcome = "Passed" if entry.report.passed else "Blocked"
        rows.append(
            "<tr>"
            f"<td>{html.escape(entry.report.phase.value)}</td>"
            f"<td><strong>{html.escape(entry.label)}</strong>"
            f'<div class="muted">{html.escape(entry.description)}</div></td>'
            f'<td class="num">{"n/a" if risk is None else f"{risk:.2f}"}</td>'
            f'<td><span class="tier {tier}">{tier}</span></td>'
            f'<td class="{outcome.lower()}">{outcome}</td>'
            f'<td><a href="{entry.stem}.html">report</a> · '
            f'<a href="{entry.stem}.json">json</a></td>'
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ethical Guardian Suite: hiring demo</title>
<style>
  :root {{ --bg:#f6f6f3; --panel:#fff; --text:#1c1c1a; --muted:#6a6a64; --border:#e2e1da;
    --pass:#1f7a4d; --fail:#b3261e; --low:#1f7a4d; --moderate:#a86400; --high:#c2410c;
    --critical:#b3261e; --gender:#2563eb; --region:#c2410c; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#151513; --panel:#1f1f1c;
    --text:#ecebe5; --muted:#a2a198; --border:#33322d; --pass:#4cc38a; --fail:#ff7a70;
    --low:#4cc38a; --moderate:#f0a53a; --high:#fb8c4a; --critical:#ff7a70;
    --gender:#6ea8ff; --region:#fb8c4a; }} }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font:15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width:1000px; margin:0 auto; padding:32px 20px 48px; }}
  h1 {{ margin:0 0 6px; font-size:26px; }} h2 {{ margin:0 0 12px; font-size:17px; }}
  .muted {{ color:var(--muted); font-size:13px; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:12px;
    padding:20px; margin-top:20px; }}
  .wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; }}
  th, td {{ text-align:left; padding:10px 8px; border-bottom:1px solid var(--border);
    vertical-align:top; }}
  th {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
  .num {{ font-variant-numeric:tabular-nums; }}
  .tier {{ padding:2px 10px; border-radius:999px; font-size:12px; font-weight:700;
    text-transform:uppercase; color:var(--panel); background:var(--muted); }}
  .tier.low {{ background:var(--low); }} .tier.moderate {{ background:var(--moderate); }}
  .tier.high {{ background:var(--high); }} .tier.critical {{ background:var(--critical); }}
  .passed {{ color:var(--pass); font-weight:700; }}
  .blocked {{ color:var(--fail); font-weight:700; }}
  a {{ color:inherit; }}
  svg {{ width:100%; max-width:600px; height:auto; }}
  svg .grid {{ stroke:var(--border); }} svg .tick {{ fill:var(--muted); font-size:12px; }}
  svg .rule {{ stroke:var(--fail); stroke-dasharray:6 4; }}
  svg .rule-label {{ fill:var(--fail); font-size:12px; }}
  svg .chosen {{ stroke:var(--muted); stroke-dasharray:2 4; }}
  svg polyline {{ fill:none; stroke-width:2.5; }}
  svg polyline.gender {{ stroke:var(--gender); }} svg polyline.region {{ stroke:var(--region); }}
  svg circle.gender {{ fill:var(--gender); }} svg circle.region {{ fill:var(--region); }}
  .legend span {{ margin-right:18px; font-size:13px; }}
  .legend i {{ display:inline-block; width:14px; height:3px; margin-right:6px;
    vertical-align:middle; }}
</style></head>
<body><main>
<h1>Ethical Guardian Suite: hiring demo</h1>
<div class="muted">Past hiring bias strength: {bias:g}. Each row links to a full report.</div>

<section class="panel">
<h2>Lifecycle reports</h2>
<div class="wrap"><table>
<thead><tr><th>Phase</th><th>Subject</th><th>Risk</th><th>Tier</th><th>Outcome</th>
<th>Files</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>
</section>

<section class="panel">
<h2>How to read this</h2>
<p>The <strong>data phase</strong> examines the company's hiring history before any agent
exists. The <strong>testing phase</strong> runs each agent on fresh simulated applicants and
compares it with a fair reference agent. A disparity found in the data (biased labels, or a
proxy such as postcode) should reappear in agents trained on that data, and disappear when
the suggested fix is applied.</p>
</section>

<section class="panel">
<h2>Historical parity against bias strength</h2>
<p class="muted">Demographic parity ratio of past hiring decisions for the same simulated
applicants as bias increases. The dotted line marks the bias used in this run.</p>
<div class="legend"><span><i style="background:var(--gender)"></i>gender</span>
<span><i style="background:var(--region)"></i>region</span></div>
{_chart(curve, bias)}
</section>
</main></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
