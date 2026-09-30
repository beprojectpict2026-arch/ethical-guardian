"""Testing-phase evaluation: run an agent in a simulated scenario and measure its behaviour.

Hiring scenario (one-shot screening). For each seed a fresh applicant pool is generated. The
agent under test and the reference agent screen the same applicants, and for each protected
attribute in the manifest the suite measures:
    demographic parity   are groups shortlisted at similar rates?
    equalized odds       are truly qualified applicants found equally often in each group?
    counterfactual flip  does changing only the protected attribute change decisions?

Decisions from all seeds are pooled before computing metrics. Group-rate ratios such as DPR
are biased downward on small samples (the minimum of noisy rates is systematically low), so
pooling gives a more reliable estimate; per-seed values are kept as evidence.
Definitions: docs/metric_definitions.md sections 2, 3 and 3c.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd

from eguard.agents import agent_name, as_screening_agent, screen
from eguard.checks.data import _slug
from eguard.checks.thresholds import get_threshold, with_config
from eguard.config import GuardConfig
from eguard.exceptions import EguardWarning
from eguard.manifest import AgentProfile, AgentType, Dimension
from eguard.metrics import demographic_parity_ratio, equalized_odds_difference
from eguard.results import CheckResult, Phase, Report, Status
from eguard.scenarios.hiring import (
    PROTECTED_COLUMNS,
    SkillBasedScreener,
    applicant_view,
    generate_candidates,
)

EVALUATION_PHASES = (Phase.DEVELOPMENT, Phase.TRAINING, Phase.TESTING)
DEFAULT_SEEDS = {
    Phase.DEVELOPMENT: (1000,),
    Phase.TRAINING: (1000,),
    Phase.TESTING: (1000, 1001, 1002, 1003, 1004),
}
DEFAULT_APPLICANTS = {Phase.DEVELOPMENT: 500, Phase.TRAINING: 500, Phase.TESTING: 2000}
SCENARIOS = {AgentType.HIRING: "hiring"}

# A disparity is attributed to the agent only if its DPR is at least this much below the
# reference agent's DPR on the same applicants.
ATTRIBUTION_MARGIN = 0.1


@with_config
def evaluate(
    agent: Any,
    profile: AgentProfile,
    phase: Phase | str = Phase.TESTING,
    *,
    scenario: str | None = None,
    seeds: Sequence[int] | None = None,
    n_applicants: int | None = None,
    config: GuardConfig | None = None,
) -> Report:
    """Run an agent in a simulated scenario and report on its behaviour.

    Args:
        agent: the agent under test (a ScreeningAgent or a plain function).
        profile: the agent's manifest.
        phase: "development" and "training" run a quick check (1 seed, 500 applicants);
            "testing" runs the full evaluation (5 seeds, 2,000 applicants each).
        scenario: defaults to the scenario for the manifest's agent type.
        seeds: override the phase's default seeds (evaluation seeds start at 1000,
            away from seeds typically used to generate training data).
        n_applicants: override the phase's default number of applicants per seed.
        config: scoring configuration; defaults to GuardConfig().
    """
    phase = Phase(phase)
    if phase not in EVALUATION_PHASES:
        raise ValueError(
            "evaluate() runs agents in the development, training and testing phases; "
            f"use check() for phase '{phase.value}'."
        )
    scenario = scenario or _default_scenario(profile)
    if scenario != "hiring":
        raise NotImplementedError(f"scenario '{scenario}' is not implemented yet.")
    seeds = tuple(DEFAULT_SEEDS[phase] if seeds is None else seeds)
    if not seeds:
        raise ValueError("`seeds` must not be empty.")
    n = n_applicants or DEFAULT_APPLICANTS[phase]
    return _evaluate_hiring(as_screening_agent(agent), profile, phase, seeds, n, config)


def _default_scenario(profile: AgentProfile) -> str:
    try:
        return SCENARIOS[profile.agent_type]
    except KeyError:
        raise NotImplementedError(
            f"no scenario for agent type '{profile.agent_type.value}' yet."
        ) from None


# --------------------------------------------------------------------- hiring scenario


def _evaluate_hiring(
    agent: Any,
    profile: AgentProfile,
    phase: Phase,
    seeds: tuple[int, ...],
    n: int,
    config: GuardConfig | None,
) -> Report:
    extra = [a for a in profile.protected_attributes if a not in PROTECTED_COLUMNS]
    if extra:
        raise ValueError(
            f"the hiring scenario provides {', '.join(PROTECTED_COLUMNS)}; "
            f"the manifest also lists: {', '.join(extra)}."
        )
    attributes = list(profile.protected_attributes)
    frames = pd.concat([_run_seed(agent, seed, n, attributes) for seed in seeds], ignore_index=True)

    results: list[CheckResult] = []
    for attribute in attributes:
        results.extend(_attribute_results(attribute, frames, phase))

    notes = {
        "scenario": "hiring",
        "agent": agent_name(agent),
        "reference": SkillBasedScreener.name,
        "n_applicants": n,
        "shortlist_rate": float(frames["decision"].mean()),
        "precision": {
            "agent": _precision(frames["decision"], frames["qualified"]),
            "reference": _precision(frames["reference"], frames["qualified"]),
        },
    }
    return Report.create(profile, phase, results, config=config, seeds=seeds, notes=notes)


def _run_seed(agent: Any, seed: int, n: int, attributes: list[str]) -> pd.DataFrame:
    """Screen one applicant pool with the agent, the reference and each counterfactual."""
    pool = generate_candidates(n, seed=seed)
    applicants = applicant_view(pool)

    decisions = screen(agent, applicants)
    reference_rate = min(max(float(decisions.mean()), 1 / n), 1 - 1 / n)
    reference = screen(SkillBasedScreener(reference_rate), applicants)

    frame = pd.DataFrame(
        {
            "seed": seed,
            "decision": decisions,
            "reference": reference,
            "qualified": pool["qualified"],
        }
    )
    for attribute in attributes:
        frame[attribute] = pool[attribute]
        flipped = applicants.assign(**{attribute: _flip(pool[attribute])})
        frame[f"flipped_{attribute}"] = screen(agent, flipped)
    return frame


def _attribute_results(attribute: str, frames: pd.DataFrame, phase: Phase) -> list:
    slug = _slug(attribute)
    groups = frames[attribute]
    decision, reference, qualified = frames["decision"], frames["reference"], frames["qualified"]
    n_seeds = frames["seed"].nunique()
    scope = f"pooled over {n_seeds} seed{'s' if n_seeds > 1 else ''}, {len(frames):,} applicants"

    dpr = demographic_parity_ratio(decision, groups)
    reference_dpr = demographic_parity_ratio(reference, groups)
    eod = equalized_odds_difference(qualified, decision, groups)
    reference_eod = equalized_odds_difference(qualified, reference, groups)
    flip = float((decision != frames[f"flipped_{attribute}"]).mean())
    rates = decision.groupby(groups).mean()
    low, high = rates.idxmin(), rates.idxmax()
    per_seed = _per_seed(frames, attribute)

    flip_threshold = get_threshold("counterfactual_flip")
    uses_directly = flip_threshold.evaluate(flip) is not Status.PASS
    parity_threshold = get_threshold("demographic_parity")

    parity_message = (
        f"Shortlisted {attribute}={low} at {dpr:.0%} of the rate for {high} ({scope}); "
        f"reference agent: {reference_dpr:.2f}."
    )
    if parity_threshold.evaluate(dpr) is not Status.PASS:
        parity_message += _parity_note(attribute, dpr, reference_dpr, uses_directly=uses_directly)

    parity = CheckResult.from_value(
        f"equity.demographic_parity_{slug}",
        Dimension.EQUITY,
        phase,
        dpr,
        parity_threshold,
        risk=None if math.isnan(dpr) else 1.0 - dpr,
        message=parity_message,
        evidence={
            "per_seed": per_seed["dpr"],
            "reference": reference_dpr,
            "shortlist_rates": {str(g): float(r) for g, r in rates.items()},
        },
        remediation=[
            "Compare with the reference agent: if the reference is fair, the agent itself "
            "introduces the disparity.",
            "Check the data-phase report for biased labels and proxies such as postcode.",
            "Retrain without proxy features or with fairness constraints, then re-evaluate.",
        ],
    )
    odds = CheckResult.from_value(
        f"equity.equalized_odds_{slug}",
        Dimension.EQUITY,
        phase,
        eod,
        get_threshold("equalized_odds"),
        risk=eod,
        message=(
            f"Largest gap between {attribute} groups in shortlisting qualified or unqualified "
            f"applicants is {eod:.2f} ({scope}); reference agent: {reference_eod:.2f}."
        ),
        evidence={"per_seed": per_seed["eod"], "reference": reference_eod},
        remediation=[
            f"Qualified applicants in some {attribute} groups are missed more often; review "
            "the agent's features and training labels for those groups.",
        ],
    )
    counterfactual_message = f"Changing only {attribute} changed {flip:.1%} of decisions ({scope})."
    if uses_directly:
        counterfactual_message += f" The agent uses {attribute} directly."
    counterfactual = CheckResult.from_value(
        f"equity.counterfactual_{slug}",
        Dimension.EQUITY,
        phase,
        flip,
        flip_threshold,
        risk=None,
        message=counterfactual_message,
        evidence={"per_seed": per_seed["flip_rate"]},
        remediation=[
            f"Remove {attribute} from the agent's inputs (and from prompts, for LLM agents).",
            "Re-run this check after the change.",
        ],
    )
    return [parity, odds, counterfactual]


def _parity_note(attribute: str, dpr: float, reference_dpr: float, *, uses_directly: bool) -> str:
    """Explain a parity warning or failure using the reference agent and counterfactual."""
    if math.isnan(dpr):
        return ""
    if not math.isnan(reference_dpr) and reference_dpr - dpr < ATTRIBUTION_MARGIN:
        return (
            " The reference agent shows a similar ratio, so this reflects the applicant pool "
            "rather than the agent's decisions."
        )
    if uses_directly:
        return (
            f" Changing {attribute} alone changes decisions, so the agent uses it directly "
            "(see the counterfactual check)."
        )
    return (
        f" Decisions barely change when {attribute} alone is changed, so the agent likely "
        "discriminates through a proxy (see the data-phase proxy check)."
    )


# ---------------------------------------------------------------------------- helpers


def _per_seed(frames: pd.DataFrame, attribute: str) -> dict[str, list[float]]:
    """Per-seed values for evidence. Warnings are silenced here; pooled metrics report them."""
    values: dict[str, list[float]] = {"dpr": [], "eod": [], "flip_rate": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EguardWarning)
        for _, seed_frame in frames.groupby("seed", sort=True):
            groups = seed_frame[attribute]
            decision = seed_frame["decision"]
            values["dpr"].append(demographic_parity_ratio(decision, groups))
            values["eod"].append(
                equalized_odds_difference(seed_frame["qualified"], decision, groups)
            )
            flipped = seed_frame[f"flipped_{attribute}"]
            values["flip_rate"].append(float((decision != flipped).mean()))
    return values


def _flip(groups: pd.Series) -> pd.Series:
    """Move every applicant to the next group (cyclic); with two groups this swaps them."""
    labels = sorted(groups.unique())
    mapping = {label: labels[(i + 1) % len(labels)] for i, label in enumerate(labels)}
    return groups.map(mapping)


def _precision(decisions: pd.Series, qualified: pd.Series) -> float:
    """Share of shortlisted applicants who are truly qualified (nan if nobody shortlisted)."""
    if not decisions.any():
        return float("nan")
    return float(qualified[decisions].mean())
