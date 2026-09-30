"""Lifecycle demo: different flaws are caught at different phases of development.

    uv run python -m examples.lifecycle_demo            # full run (about a minute)
    uv run python -m examples.lifecycle_demo --quick    # smaller, faster run

Seven hiring-agent projects go through every implemented phase: planning, data, design,
testing and the deployment gate. Each flawed project has one problem introduced at a known
point, and the demo shows the earliest phase that catches it. This illustrates the central
claim of the project: only a lifecycle approach catches all of them.
"""

from __future__ import annotations

import argparse
import html
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import eguard
from eguard import Report, Status
from eguard.agents import FunctionAgent
from eguard.scenarios.hiring import (
    SkillBasedScreener,
    generate_candidates,
    historical_records,
    shortlist_top,
)
from examples.agents.ml_screener import MLScreener

MANIFEST = Path(__file__).resolve().parent / "manifests" / "hiring_agent.yaml"
PHASES = ("planning", "data", "design", "testing", "deployment")

GOOD_OBJECTIVE = (
    "Shortlist the applicants most likely to succeed in the role, based on skills, "
    "experience and education. Explain each decision."
)
BIASED_OBJECTIVE = (
    "Prefer male candidates from metro cities, and look for a culture fit. Explain each decision."
)
FULL_AUTOMATION = {
    "automated_tasks": [
        {
            "description": "Review resumes against job requirements",
            "occupation": "Human Resources Specialists",
            "onet_code": "13-1071.00",
            "automation_fraction": 1.0,
            "weight": 3,
        },
        {
            "description": "Schedule first-round interviews",
            "occupation": "Human Resources Specialists",
            "onet_code": "13-1071.00",
            "automation_fraction": 1.0,
            "weight": 1,
        },
    ]
}
DATA_WAIVERS = {
    "equity.label_parity_gender": (
        "Historical bias acknowledged; the agent is retrained without proxies and "
        "passes testing parity."
    ),
    "equity.label_parity_region": (
        "Historical bias acknowledged; postcode removed from the agent's features."
    ),
}


# ------------------------------------------------------------------------------ agents


def _ml(history: pd.DataFrame) -> Any:
    return MLScreener().fit(history)


def _ml_without_postcode(history: pd.DataFrame) -> Any:
    return MLScreener(drop=["postcode"]).fit(history)


def _bonus_agent(name: str, bonus: Callable[[pd.DataFrame], pd.Series]) -> FunctionAgent:
    """Skill-based scores plus a bonus, shortlisting the top 20%."""
    reference = SkillBasedScreener()

    def decide(applicants: pd.DataFrame) -> pd.Series:
        return shortlist_top(reference.score(applicants) + bonus(applicants), 0.2)

    return FunctionAgent(decide, name=name)


def _follows_biased_objective(history: pd.DataFrame) -> FunctionAgent:
    return _bonus_agent(
        "screener following the biased objective",
        lambda a: 1.0 * (a["gender"] == "male") + 1.0 * (a["region"] == "urban"),
    )


def _location_bug(history: pd.DataFrame) -> FunctionAgent:
    return _bonus_agent(
        "screener with a location-bonus bug",
        lambda a: 0.75 * (a["region"] == "urban"),
    )


# --------------------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    flaw: str
    bias: float
    objective: str
    build_agent: Callable[[pd.DataFrame], Any]
    manifest_changes: dict[str, Any] = field(default_factory=dict)
    skip_design: bool = False
    waivers: dict[str, str] = field(default_factory=dict)


SCENARIOS = (
    Scenario(
        "responsible",
        "Responsible team",
        "None: fair data, a careful objective and a sound model.",
        0.0,
        GOOD_OBJECTIVE,
        _ml,
    ),
    Scenario(
        "aggressive_automation",
        "Aggressive automation",
        "Fully automates recruiters' work, with no reskilling plan.",
        0.0,
        GOOD_OBJECTIVE,
        _ml,
        manifest_changes=FULL_AUTOMATION,
    ),
    Scenario(
        "biased_history",
        "Biased history",
        "Trained on biased past hiring; postcode leaks region.",
        1.0,
        GOOD_OBJECTIVE,
        _ml,
    ),
    Scenario(
        "biased_instructions",
        "Biased instructions",
        "The objective tells the agent to prefer men from metro cities.",
        0.0,
        BIASED_OBJECTIVE,
        _follows_biased_objective,
    ),
    Scenario(
        "faulty_code",
        "Faulty agent code",
        "A coding bug gives urban applicants a bonus.",
        0.0,
        GOOD_OBJECTIVE,
        _location_bug,
    ),
    Scenario(
        "skipped_review",
        "Skipped design review",
        "The design phase was never checked.",
        0.0,
        GOOD_OBJECTIVE,
        _ml,
        skip_design=True,
    ),
    Scenario(
        "mitigated",
        "Mitigated team",
        "Biased history, mitigated: postcode removed and data findings waived.",
        1.0,
        GOOD_OBJECTIVE,
        _ml_without_postcode,
        waivers=DATA_WAIVERS,
    ),
)


@dataclass
class Outcome:
    scenario: Scenario
    reports: dict[str, Report]

    @property
    def first_caught(self) -> str | None:
        """The earliest phase whose report blocks (a failing or undefined check)."""
        for phase in PHASES:
            report = self.reports.get(phase)
            if report is not None and not report.passed:
                return phase
        return None

    @property
    def verdict(self) -> str:
        if not self.reports["deployment"].passed:
            return "Blocked"
        return "Released with waivers" if self.scenario.waivers else "Released"

    def status(self, phase: str) -> str:
        """Worst status in a phase's report, or 'skipped'."""
        report = self.reports.get(phase)
        if report is None:
            return "skipped"
        found = {result.status for result in report.results}
        for status in (Status.FAIL, Status.UNDEFINED, Status.WARN):
            if status in found:
                return status.value
        return Status.PASS.value


# ----------------------------------------------------------------------------- running


def run(*, quick: bool, out: Path) -> list[Outcome]:
    """Run every scenario through the lifecycle and save all reports."""
    base = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    history_size = 5_000 if quick else 10_000
    seeds = (1000, 1001, 1002) if quick else (1000, 1001, 1002, 1003, 1004)
    histories = {
        bias: historical_records(generate_candidates(history_size, seed=1, bias_strength=bias))
        for bias in {s.bias for s in SCENARIOS}
    }

    outcomes = []
    for scenario in SCENARIOS:
        profile = eguard.AgentProfile.from_dict({**base, **scenario.manifest_changes})
        history = histories[scenario.bias]
        reports: dict[str, Report] = {
            "planning": eguard.check(profile, "planning"),
            "data": eguard.check(profile, "data", data=history, target="hired"),
        }
        if not scenario.skip_design:
            reports["design"] = eguard.check(profile, "design", objective=scenario.objective)
        agent = scenario.build_agent(history)
        reports["testing"] = eguard.evaluate(agent, profile, seeds=seeds, n_applicants=2_000)
        reports["deployment"] = eguard.check(
            profile, "deployment", reports=list(reports.values()), waivers=scenario.waivers or None
        )
        for phase, report in reports.items():
            report.to_html(out / scenario.key / f"{phase}.html")
            report.to_json(out / scenario.key / f"{phase}.json")
        outcomes.append(Outcome(scenario, reports))
    (out / "index.html").write_text(_index_page(outcomes), encoding="utf-8")
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ethical Guardian Suite lifecycle demo.")
    parser.add_argument("--quick", action="store_true", help="smaller, faster run")
    parser.add_argument("--out", default="reports/lifecycle", help="output folder")
    args = parser.parse_args(argv)
    out = Path(args.out)

    print("Running 7 hiring-agent projects through planning, data, design, testing and the")
    print("deployment gate ...\n")
    outcomes = run(quick=args.quick, out=out)

    header = f"{'Project':24}" + "".join(f"{p:>12}" for p in PHASES)
    print(header + f"   {'First caught':14}Outcome")
    print("-" * (len(header) + 40))
    for o in outcomes:
        cells = "".join(f"{o.status(p).upper():>12}" for p in PHASES)
        print(f"{o.scenario.title:24}{cells}   {o.first_caught or '-':14}{o.verdict}")
    print(f"\nDone. Open {out / 'index.html'} in your browser.")
    return 0


# -------------------------------------------------------------------------- index page


def _index_page(outcomes: list[Outcome]) -> str:
    rows = []
    for o in outcomes:
        cells = []
        for phase in PHASES:
            status = o.status(phase)
            label = html.escape(status.replace("_", " "))
            if status == "skipped":
                cells.append(f'<td><span class="badge skipped">{label}</span></td>')
            else:
                link = f"{o.scenario.key}/{phase}.html"
                cells.append(f'<td><a class="badge {status}" href="{link}">{label}</a></td>')
        verdict_class = "blocked" if o.verdict == "Blocked" else "released"
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(o.scenario.title)}</strong>"
            f'<div class="muted">{html.escape(o.scenario.flaw)}</div></td>'
            + "".join(cells)
            + f"<td>{html.escape(o.first_caught or '-')}</td>"
            f'<td class="{verdict_class}">{html.escape(o.verdict)}</td>'
            "</tr>"
        )
    header = "".join(f"<th>{p}</th>" for p in PHASES)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ethical Guardian Suite: lifecycle demo</title>
<style>
  :root {{ --bg:#f6f6f3; --panel:#fff; --text:#1c1c1a; --muted:#6a6a64; --border:#e2e1da;
    --pass:#1f7a4d; --warn:#a86400; --fail:#b3261e; --undefined:#6b48b3; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#151513; --panel:#1f1f1c;
    --text:#ecebe5; --muted:#a2a198; --border:#33322d; --pass:#4cc38a; --warn:#f0a53a;
    --fail:#ff7a70; --undefined:#b69cf2; }} }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font:15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width:1100px; margin:0 auto; padding:32px 20px 48px; }}
  h1 {{ margin:0 0 6px; font-size:26px; }} h2 {{ margin:0 0 12px; font-size:17px; }}
  .muted {{ color:var(--muted); font-size:13px; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:12px;
    padding:20px; margin-top:20px; }}
  .wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; }}
  th, td {{ text-align:left; padding:10px 8px; border-bottom:1px solid var(--border);
    vertical-align:top; }}
  th {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
  .badge {{ display:inline-block; padding:1px 8px; border:1px solid currentColor;
    border-radius:6px; font-size:11px; font-weight:700; text-transform:uppercase;
    text-decoration:none; white-space:nowrap; }}
  .badge.pass {{ color:var(--pass); }} .badge.warn {{ color:var(--warn); }}
  .badge.fail {{ color:var(--fail); }} .badge.undefined {{ color:var(--undefined); }}
  .badge.skipped {{ color:var(--muted); border-style:dashed; }}
  .blocked {{ color:var(--fail); font-weight:700; }}
  .released {{ color:var(--pass); font-weight:700; }}
</style></head>
<body><main>
<h1>Ethical Guardian Suite: lifecycle demo</h1>
<div class="muted">Seven hiring-agent projects through every implemented phase. Click a
status to open that phase's report.</div>

<section class="panel">
<h2>Where each flaw is caught</h2>
<div class="wrap"><table>
<thead><tr><th>Project</th>{header}<th>First caught</th><th>Outcome</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>
</section>

<section class="panel">
<h2>How to read this</h2>
<p>Each flawed project contains one problem, introduced at a known point. The planning check
sees the manifest, the data check sees historical records, the design check sees the
agent's objective, testing runs the agent on simulated applicants, and the deployment gate
requires complete, consistent evidence from the earlier phases. Every flaw is caught at the
first phase whose evidence reveals it, and no single phase catches all of them.</p>
<p>A <strong>warning</strong> does not block release; a <strong>failure</strong> does,
unless it is waived at the gate with a written justification, which is recorded and keeps
its risk in the score.</p>
</section>
</main></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
