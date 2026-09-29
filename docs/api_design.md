# API Design

**Status:** v1 (end of Stage 1). This document describes how companies will use the
Ethical Guardian Suite and the data structures every part of the library shares. Sections
marked **Implemented** exist in the code today; sections marked **Planned** are the agreed
design that later stages will build. Changes to the public API are made here first, in a
reviewed pull request.

---

## 1. Design principles

1. **One call per lifecycle phase.** A developer should be able to add an evaluation to any
   stage of agent development with a single function call.
2. **Every check returns the same structure.** All checks produce `CheckResult` objects
   collected in a `Report`, whatever the phase or agent type. Reports can therefore be
   compared across phases, stored, and turned into dashboards.
3. **Scores come from deterministic code only.** Metrics, thresholds and statuses are
   computed by tested code with explicit random seeds. LLMs may later be used to write
   plain-language explanations and remediation text, but never to compute a score or a
   pass/fail status. This keeps results reproducible.
4. **Fail loudly, never silently.** Invalid input raises an error. A metric that cannot be
   computed is reported as `undefined`, never treated as a pass.
5. **CI-friendly.** Any report can fail a build, so evaluations can act as deployment gates.

---

## 2. Package layout

| Module | Contents | Status |
|---|---|---|
| `eguard` | Top-level API: `AgentProfile`, `check`, `evaluate`, `monitor`, exceptions | Partly implemented |
| `eguard.metrics` | Pure metric functions (see `docs/metric_definitions.md`) | **Implemented** |
| `eguard.manifest` | `AgentProfile` and related schema classes | **Implemented** |
| `eguard.exceptions` | `EguardWarning`, `ManifestError`, `EvaluationFailed` | Partly implemented |
| `eguard.results` | `Phase`, `Status`, `Threshold`, `CheckResult`, `Report` | Planned (Stage 2) |
| `eguard.checks` | One module per phase group: `planning`, `data`, `design`, `testing`, `monitoring` | Planned (Stage 2 onward) |
| `eguard.scenarios` | Simulated environments: `hiring`, later `customer_support`, `pricing` | Planned (Stage 2 onward) |
| `eguard.agents` | Agent interfaces (protocols) that a company's agent is wrapped to fit | Planned (Stage 2) |
| `eguard.reporting` | JSON and HTML report output | Planned (Stage 2) |
| `eguard.cli` | `eguard` command-line tool | Planned (Stage 3) |

**Public vs. internal.** Only names listed in a module's `__all__` are public. Modules and
names starting with an underscore (e.g. `eguard.metrics._validation`) are internal and may
change at any time. While the version is `0.x`, the public API may still change between
minor versions; every change is recorded in the changelog below.

---

## 3. Lifecycle phases

**Planned:** `eguard.results.Phase`, a string enum with the eight phases of the agent
development lifecycle. Each phase uses the evidence that exists at that point.

| Phase | Evidence available | Entry point | Typical checks |
|---|---|---|---|
| `planning` | Manifest only | `check` | Job displacement score from `automated_tasks`; risk tier; missing information |
| `data` | Training or historical data | `check` | Group representation; DPR of historical labels; proxy attributes |
| `design` | System prompt, objective or reward | `check` | Risky incentives (e.g. profit-only objectives); protected attributes used as inputs |
| `development` | A callable agent | `evaluate` | Short simulation, few seeds (smoke test) |
| `training` | Agent checkpoints | `evaluate` | Same metrics per checkpoint, to track drift during training |
| `testing` | Final candidate agent | `evaluate` | Full simulation, many seeds, stress scenarios, counterfactual tests |
| `deployment` | Testing report + thresholds | `check` | Gate: fail if any required check failed or is undefined |
| `monitoring` | Live decision logs | `monitor`, `check` | Metric drift against the testing baseline |

---

## 4. Core data structures (Planned, Stage 2)

### 4.1 `Status`

String enum describing the outcome of one check:

| Value | Meaning |
|---|---|
| `pass` | Metric within threshold |
| `warn` | Metric past the warning threshold but within the failure threshold |
| `fail` | Metric past the failure threshold |
| `undefined` | Metric could not be computed (e.g. `nan` from a group with no positives). **Never counts as a pass.** |
| `not_applicable` | The check's dimension does not apply to this agent (per its manifest) |

### 4.2 `Threshold`

```python
class Threshold(BaseModel):
    warn_at: float
    fail_at: float
    direction: Literal["higher_is_worse", "lower_is_worse"]
```

Example: demographic parity ratio uses `lower_is_worse` with `warn_at=0.9`,
`fail_at=0.8` (the four-fifths rule). Default thresholds follow
`docs/metric_definitions.md`; all are overridable through configuration (Section 7).

### 4.3 `CheckResult`

One metric evaluated against one threshold.

| Field | Type | Description |
|---|---|---|
| `check_id` | `str` | Stable identifier, `<dimension>.<metric>`, e.g. `equity.demographic_parity_ratio` |
| `dimension` | `Dimension` | One of the five dimensions (from `eguard.manifest`) |
| `phase` | `Phase` | Phase in which the check ran |
| `value` | `float \| None` | Metric value; `None` if undefined or not applicable |
| `threshold` | `Threshold \| None` | Threshold applied, if any |
| `status` | `Status` | Outcome |
| `risk` | `float \| None` | Risk value in `[0, 1]` from the metric-to-risk mapping in `metric_definitions.md` §7.2 |
| `message` | `str` | One plain-language sentence, e.g. "Group B is shortlisted at 40% of group A's rate." |
| `evidence` | `dict` | Supporting numbers, e.g. per-group selection rates and group sizes |
| `remediation` | `list[str]` | Suggested fixes; empty when the check passes |

### 4.4 `Report`

All results for one agent at one phase.

| Field / method | Description |
|---|---|
| `agent_name`, `agent_version` | Copied from the manifest |
| `phase` | Phase evaluated |
| `results` | `list[CheckResult]` |
| `dimension_risks` | `dict[Dimension, float \| None]`: maximum risk per applicable dimension |
| `composite_risk` | Weighted composite over applicable dimensions (`metric_definitions.md` §7.3) |
| `risk_tier` | `low`, `moderate`, `high` or `critical` |
| `metadata` | eguard version, random seeds, configuration used, timestamp: everything needed to reproduce the report |
| `passed` | `True` only if no result is `fail` or `undefined` |
| `summary()` | Short text summary for terminals and logs |
| `to_json(path)` / `from_json(path)` | Save and reload; reports are compared across phases |
| `to_html(path)` | Human-readable report |
| `raise_for_status(fail_on="fail")` | Raises `EvaluationFailed` if any result is at or worse than `fail_on` (`"fail"` or `"warn"`). Used as a CI gate |

`EvaluationFailed` subclasses `AssertionError`, so a failing report also fails a pytest test.

---

## 5. Entry points

### 5.1 `AgentProfile` (**Implemented**)

```python
import eguard

profile = eguard.AgentProfile.from_yaml("agent_manifest.yaml")
profile.applicable_dimensions  # frozenset of Dimension
```

Raises `ManifestError` with one readable line per problem. Full schema:
`src/eguard/manifest/schema.py`; examples: `examples/manifests/`.

### 5.2 `check` (Planned)

For phases where the evidence is static: a manifest, a dataset, a prompt, or earlier reports.

```python
def check(
    profile: AgentProfile,
    phase: Phase | str,
    *,
    data: pd.DataFrame | None = None,  # phase "data"
    target: str | None = None,  # label column in `data`
    prompt: str | None = None,  # phase "design"
    objective: str | None = None,  # phase "design"
    reports: list[Report] | None = None,  # phase "deployment"
    logs: pd.DataFrame | None = None,  # phase "monitoring"
    config: GuardConfig | None = None,
) -> Report: ...
```

Passing evidence that the phase does not use, or omitting evidence it requires, raises
`ValueError` naming the missing or unexpected argument. Protected attributes are taken from
the manifest, so they are never repeated at the call site.

### 5.3 `evaluate` (Planned)

For phases where the agent is run inside a simulated scenario.

```python
def evaluate(
    agent: Agent,
    profile: AgentProfile,
    phase: Phase | str = "testing",
    *,
    scenario: str | None = None,  # defaults to the scenario for profile.agent_type
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    config: GuardConfig | None = None,
) -> Report: ...
```

For each seed, the scenario runs twice with identical randomness: once with the reference
agent (baseline) and once with the agent under test. Metrics are reported as differences
from the baseline where the definitions require it (e.g. Gini delta), with results
aggregated across seeds. The phase controls scale: `development` uses a small population and
one seed; `testing` uses the full configuration.

### 5.4 `monitor` (Planned, Stage 3)

Wraps a deployed agent's decision function and records each decision for later checks.

```python
@eguard.monitor(profile, sink="logs/decisions.jsonl")
def decide(request): ...
```

The decorator never changes the function's result and never blocks it. Logging failures
emit an `EguardWarning` rather than breaking production. Recorded logs are passed to
`check(profile, "monitoring", logs=...)`, which compares metrics against the testing
baseline and reports drift.

---

## 6. Agent interfaces (Planned, Stage 2)

A company wraps its agent in a small adapter matching one protocol per agent type. The
library never needs to know how the agent works internally: rule-based, ML or LLM.

```python
class ScreeningAgent(Protocol):  # agent_type: hiring
    def decide(self, candidates: pd.DataFrame) -> pd.Series:
        ...
        # returns one boolean per candidate: shortlisted or not


class SupportAgent(Protocol):  # agent_type: customer_support
    def respond(self, query: SupportQuery) -> SupportResponse: ...


class PricingAgent(Protocol):  # agent_type: pricing
    def set_prices(self, market: MarketState) -> dict[str, float]: ...
```

The hiring protocol is finalised in Stage 2; the other two are sketches and will be fixed
when their scenarios are built. Candidate tables passed to `decide` include protected
attributes, since real systems often receive them; whether the agent uses them is exactly
what the checks measure.

---

## 7. Configuration (Planned)

`GuardConfig` holds everything that is a choice rather than a definition: thresholds,
dimension weights, risk-tier boundaries, reference constants ($\Delta G_{\text{ref}}$,
$\text{CV}_{\text{ref}}$), scenario sizes and default seeds. Defaults match
`docs/metric_definitions.md`. A project can override them in a YAML file:

```yaml
# guard_config.yaml
weights:
  equity: 0.35
  labour: 0.25
thresholds:
  equity.demographic_parity_ratio: {warn_at: 0.9, fail_at: 0.8, direction: lower_is_worse}
```

The configuration used is copied into every report's metadata, so a report can always be
reproduced.

---

## 8. Command-line tool and CI (Planned, Stage 3)

```bash
eguard check --manifest agent.yaml --phase data --data train.csv --target hired
eguard check --manifest agent.yaml --phase deployment --reports reports/testing.json
```

| Exit code | Meaning |
|---|---|
| `0` | All checks passed (or only warnings, unless `--fail-on warn`) |
| `1` | At least one check failed or was undefined |
| `2` | Invalid input: manifest, data or arguments |

Every command also writes the report as JSON and HTML (`--out reports/`), so CI systems can
store them as build artifacts.

In Python test suites, the same gate is one line:

```python
def test_screener_is_fair():
    report = eguard.evaluate(MyScreener(), profile, phase="testing")
    report.raise_for_status()
```

---

## 9. Errors and warnings

| Class | Base | Raised when | Status |
|---|---|---|---|
| `ValueError` | built-in | Invalid input to a metric or check | **Implemented** (metrics) |
| `ManifestError` | `ValueError` | A manifest cannot be read or is invalid | **Implemented** |
| `EguardWarning` | `UserWarning` | Result is undefined (`nan`) or statistically unstable (small groups) | **Implemented** |
| `EvaluationFailed` | `AssertionError` | `Report.raise_for_status()` finds failing checks | Planned |

---

## 10. End-to-end example (target for Stage 2)

```python
import eguard
import pandas as pd

profile = eguard.AgentProfile.from_yaml("examples/manifests/hiring_agent.yaml")

# Planning: before any code exists
print(eguard.check(profile, "planning").summary())

# Data: historical hiring records
history = pd.read_csv("hiring_history.csv")
data_report = eguard.check(profile, "data", data=history, target="hired")

# Testing: run the agent in the simulated labour market
test_report = eguard.evaluate(MyScreener(), profile, phase="testing")
test_report.to_html("reports/testing.html")

# Deployment gate
gate = eguard.check(profile, "deployment", reports=[data_report, test_report])
gate.raise_for_status()
```

---

## Changelog

| Version | Change |
|---|---|
| v1 | Initial API design: phases, `Status`, `Threshold`, `CheckResult`, `Report`, `check`, `evaluate`, `monitor`, agent protocols, configuration, CLI exit codes, error classes. |
