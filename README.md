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

## Command line

```bash
eguard check planning   --manifest agent.yaml
eguard check data       --manifest agent.yaml --data history.csv --target hired
eguard check design     --manifest agent.yaml --prompt-file prompt.txt
eguard check deployment --manifest agent.yaml --reports reports/*.json
eguard show reports/planning.json
```

Each check prints a summary, saves JSON and HTML reports to `reports/` (`--out` to
change), and exits with **0** (passed), **1** (blocked) or **2** (invalid input). Use
`--config policy.yaml` to apply your own thresholds and weights, and `--fail-on warn`
to block on warnings too. The testing phase runs the agent itself, from Python with
`eguard.evaluate()`; its saved JSON report can be passed to the deployment gate.

### In CI

Because a blocked check exits with code 1, it stops a CI pipeline. For example, in a
GitHub Actions job:

```yaml
- name: Ethical Guardian deployment gate
  run: >
    eguard check deployment --manifest agent.yaml
    --reports reports/planning.json reports/design.json reports/testing.json
```

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
