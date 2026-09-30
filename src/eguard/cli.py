"""Command-line interface: run lifecycle checks from a terminal or a CI pipeline.

    eguard check planning   --manifest agent.yaml
    eguard check data       --manifest agent.yaml --data history.csv --target hired
    eguard check design     --manifest agent.yaml --prompt-file prompt.txt
    eguard check deployment --manifest agent.yaml --reports reports/*.json
    eguard show report.json

Each check prints a summary and saves JSON and HTML reports. Exit codes:
    0  passed (warnings allowed unless --fail-on warn)
    1  blocked: a check failed or was undefined (or warned, with --fail-on warn)
    2  invalid input: manifest, configuration, files or arguments
The testing phase needs a running agent, so it is run from Python with eguard.evaluate();
its saved JSON report can be passed to the deployment gate.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import yaml

import eguard
from eguard.config import GuardConfig
from eguard.exceptions import ConfigError, EvaluationFailed, ManifestError
from eguard.manifest import AgentProfile
from eguard.results import Report

EXIT_PASSED = 0
EXIT_BLOCKED = 1
EXIT_INVALID = 2


class UsageError(Exception):
    """Invalid command-line input; reported with exit code 2."""


def run() -> None:
    """Entry point of the `eguard` command."""
    sys.exit(main())


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the command and return its exit code."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits with 2 on bad arguments, 0 on --help
        return int(exc.code or 0)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (ManifestError, ConfigError, UsageError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID


# ------------------------------------------------------------------------------ parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eguard",
        description="Ethical Guardian Suite: lifecycle checks for AI agents.",
    )
    parser.add_argument("--version", action="version", version=f"eguard {eguard.__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", required=True, type=Path, help="agent manifest (YAML)")
    common.add_argument("--config", type=Path, help="policy file overriding defaults (YAML)")
    common.add_argument(
        "--out", type=Path, default=Path("reports"), help="folder for reports (default: reports)"
    )
    common.add_argument(
        "--fail-on",
        choices=("fail", "warn"),
        default="fail",
        help="lowest status that blocks (default: fail)",
    )

    check = commands.add_parser("check", help="run the checks for one lifecycle phase")
    phases = check.add_subparsers(dest="phase", required=True)

    planning = phases.add_parser(
        "planning", parents=[common], help="check the manifest before any code exists"
    )
    planning.set_defaults(handler=_check_planning)

    data = phases.add_parser("data", parents=[common], help="check historical or training data")
    data.add_argument("--data", required=True, type=Path, help="CSV file, one row per case")
    data.add_argument("--target", required=True, help="binary outcome column, e.g. hired")
    data.add_argument("--exclude", nargs="*", default=[], help="columns never tested as proxies")
    data.set_defaults(handler=_check_data)

    design = phases.add_parser(
        "design", parents=[common], help="scan a prompt or objective for risky instructions"
    )
    design.add_argument("--prompt", help="prompt text")
    design.add_argument("--prompt-file", type=Path, help="file containing the prompt")
    design.add_argument("--objective", help="objective text")
    design.add_argument("--objective-file", type=Path, help="file containing the objective")
    design.set_defaults(handler=_check_design)

    deployment = phases.add_parser(
        "deployment", parents=[common], help="decide on release from earlier reports"
    )
    deployment.add_argument(
        "--reports", required=True, nargs="+", type=Path, help="earlier JSON reports"
    )
    deployment.add_argument(
        "--require",
        nargs="*",
        help="phases that must have a report (default: planning design testing)",
    )
    deployment.add_argument(
        "--waivers", type=Path, help="YAML mapping failing check IDs to justifications"
    )
    deployment.set_defaults(handler=_check_deployment)

    show = commands.add_parser("show", help="print the summary of a saved JSON report")
    show.add_argument("report", type=Path, help="JSON report")
    show.add_argument("--html", type=Path, help="also save the report as HTML here")
    show.set_defaults(handler=_show)
    return parser


# ---------------------------------------------------------------------------- handlers


def _load(args: argparse.Namespace) -> tuple[AgentProfile, GuardConfig | None]:
    profile = AgentProfile.from_yaml(args.manifest)
    config = GuardConfig.from_yaml(args.config) if args.config else None
    return profile, config


def _check_planning(args: argparse.Namespace) -> int:
    profile, config = _load(args)
    return _finish(eguard.check(profile, "planning", config=config), args)


def _check_data(args: argparse.Namespace) -> int:
    profile, config = _load(args)
    data = pd.read_csv(args.data)
    report = eguard.check(
        profile, "data", data=data, target=args.target, exclude=args.exclude, config=config
    )
    return _finish(report, args)


def _check_design(args: argparse.Namespace) -> int:
    profile, config = _load(args)
    prompt = _text(args.prompt, args.prompt_file, "prompt")
    objective = _text(args.objective, args.objective_file, "objective")
    if prompt is None and objective is None:
        raise UsageError("give --prompt/--prompt-file and/or --objective/--objective-file to scan.")
    report = eguard.check(profile, "design", prompt=prompt, objective=objective, config=config)
    return _finish(report, args)


def _check_deployment(args: argparse.Namespace) -> int:
    profile, config = _load(args)
    waivers = _read_waivers(args.waivers) if args.waivers else None
    report = eguard.check(
        profile,
        "deployment",
        reports=args.reports,
        require=args.require,
        waivers=waivers,
        config=config,
    )
    return _finish(report, args)


def _show(args: argparse.Namespace) -> int:
    report = Report.from_json(args.report)
    print(report.summary())
    if args.html:
        report.to_html(args.html)
        print(f"\nSaved {args.html}")
    return EXIT_PASSED


# ----------------------------------------------------------------------------- helpers


def _text(value: str | None, path: Path | None, name: str) -> str | None:
    if value is not None and path is not None:
        raise UsageError(f"use either --{name} or --{name}-file, not both.")
    if path is not None:
        return path.read_text(encoding="utf-8")
    return value


def _read_waivers(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise UsageError(f"{path}: waivers must map check IDs to written justifications.")
    return data


def _finish(report: Report, args: argparse.Namespace) -> int:
    """Print the summary, save JSON and HTML, and turn the outcome into an exit code."""
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"{report.phase.value}.json"
    html_path = args.out / f"{report.phase.value}.html"
    report.to_json(json_path)
    report.to_html(html_path)
    print(report.summary())
    print(f"\nSaved {json_path} and {html_path}")
    try:
        report.raise_for_status(fail_on=args.fail_on)
    except EvaluationFailed:
        print("Result: BLOCKED (exit code 1)")
        return EXIT_BLOCKED
    print("Result: PASSED")
    return EXIT_PASSED
