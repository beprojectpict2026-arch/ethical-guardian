# Ethical Guardian Suite

A lifecycle-adaptive framework for the ethical and economic evaluation of autonomous AI agents.

Companies building AI agents use this library at each stage of agent development, from
planning to monitoring, to check for economic harms such as income inequality, unfair
treatment of groups, job displacement and market instability.

> Status: early development (B.E. project, PICT, 2026–27).

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
