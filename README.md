# Ethical Guardian Suite

A lifecycle-adaptive framework for the ethical and economic evaluation of autonomous AI agents.

Companies building AI agents use this library at each stage of agent development, from
planning to monitoring, to check for economic harms such as income inequality, unfair
treatment of groups, job displacement and market instability.

> Status: early development (B.E. project, PICT, 2026–27).

## Quick demo

```bash
uv run python -m examples.hiring_demo
open reports/demo/index.html
```

The demo generates a company's hiring history with biased past decisions, checks it in the
**data phase** before any agent exists, trains ML screeners on it, and evaluates them in the
**testing phase** against a fair reference agent. It shows the data check predicting a
problem (a postcode proxy for region), the evaluation confirming it in the trained agent, and
the suggested fix removing it. Options: `--quick`, `--bias 0.5`, and `--llm` (requires
Ollama).

## Developer setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                        # install dependencies into .venv
uv run pre-commit install      # enable commit checks (once per clone)
uv run pytest                  # run tests
```

## Project layout

```
src/eguard/
├── metrics/     # inequality, fairness, labour and market metrics
├── manifest/    # agent manifest schema
└── results.py   # CheckResult and Report objects
tests/           # unit tests
examples/        # example manifests and agents
docs/            # metric definitions and API design
```
