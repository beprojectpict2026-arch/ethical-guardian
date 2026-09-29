"""Synthetic job-candidate pools with controllable historical hiring bias.

Each candidate has observable features, protected attributes, a hidden ground-truth
qualification, and a historical ``hired`` label produced by a simulated past hiring process.
``bias_strength`` sets how strongly that process penalised the targeted groups, reproducing
how biased history gets into training data.

True qualification never depends on protected attributes, so any group disparity in
``hired`` comes from the simulated bias alone. Because the correct answer is known, checks
can be validated against it.
"""

from __future__ import annotations

import math
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

SKILL_COLUMNS = ("skill_technical", "skill_communication", "skill_domain")
FEATURE_COLUMNS = (*SKILL_COLUMNS, "experience_years", "education", "postcode")
PROTECTED_COLUMNS = ("gender", "region")
HIDDEN_COLUMNS = ("qualification_score", "qualified")
LABEL_COLUMN = "hired"
EDUCATION_LEVELS = ("diploma", "bachelor", "master")

# Illustrative postcodes: one set per region. Postcode acts as a proxy for region.
URBAN_POSTCODES = tuple(f"4110{i:02d}" for i in range(1, 21))
RURAL_POSTCODES = tuple(f"4125{i:02d}" for i in range(1, 21))

# How much each attribute contributes to true qualification (after standardising).
QUALIFICATION_WEIGHTS = {
    "skill_technical": 0.35,
    "skill_communication": 0.20,
    "skill_domain": 0.25,
    "experience_years": 0.10,
    "education": 0.10,
}

_ALLOWED_TARGETS = {"gender": {"female", "male"}, "region": {"urban", "rural"}}


class CandidatePoolConfig(BaseModel):
    """Population settings for the candidate generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    female_share: float = Field(default=0.4, gt=0.0, lt=1.0)
    rural_share: float = Field(default=0.4, gt=0.0, lt=1.0)
    education_probabilities: tuple[float, float, float] = (0.3, 0.5, 0.2)
    qualified_share: float = Field(
        default=0.3, gt=0.0, lt=1.0, description="Share of candidates truly qualified."
    )
    hire_rate: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Share hired by the past process."
    )
    label_noise: float = Field(
        default=0.3, ge=0.0, description="Recruiter noise in past decisions, in SD units."
    )
    proxy_strength: float = Field(
        default=0.9,
        ge=0.5,
        le=1.0,
        description="Probability a postcode matches the candidate's region (0.5 = no proxy).",
    )
    bias_targets: dict[str, str] = Field(
        default_factory=lambda: {"gender": "female", "region": "rural"},
        description="Groups penalised by the biased past process.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        probs = self.education_probabilities
        if any(p < 0 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-9):
            raise ValueError("education_probabilities must be non-negative and sum to 1")
        for attribute, group in self.bias_targets.items():
            if group not in _ALLOWED_TARGETS.get(attribute, set()):
                raise ValueError(
                    f"invalid bias target {attribute}={group}; "
                    "allowed: gender=female|male, region=urban|rural"
                )
        return self


def generate_candidates(
    n: int,
    *,
    seed: int,
    bias_strength: float = 0.0,
    config: CandidatePoolConfig | None = None,
) -> pd.DataFrame:
    """Generate a candidate pool with a historical hiring label.

    Args:
        n: number of candidates (at least 10).
        seed: random seed. The same seed gives the same candidates and the same noise,
            so changing only ``bias_strength`` isolates the effect of bias.
        bias_strength: penalty, in standard deviations of qualification, applied by the past
            hiring process to each targeted group a candidate belongs to. 0 = fair history.
        config: population settings; defaults to ``CandidatePoolConfig()``.

    Returns:
        DataFrame with ``candidate_id``, feature, protected, hidden and label columns.
        ``df.attrs["generator"]`` records the seed, bias strength and configuration.
    """
    if n < 10:
        raise ValueError("`n` must be at least 10.")
    if not 0.0 <= bias_strength <= 3.0:
        raise ValueError("`bias_strength` must be in [0, 3].")
    cfg = config or CandidatePoolConfig()
    rng = np.random.default_rng(seed)

    gender = np.where(rng.random(n) < cfg.female_share, "female", "male")
    region = np.where(rng.random(n) < cfg.rural_share, "rural", "urban")

    skills = {col: np.clip(rng.normal(60, 15, n), 0, 100).round(1) for col in SKILL_COLUMNS}
    experience = np.clip(rng.gamma(shape=2.0, scale=2.5, size=n), 0, 30).round(1)
    education = rng.choice(EDUCATION_LEVELS, size=n, p=cfg.education_probabilities)

    matches_region = rng.random(n) < cfg.proxy_strength
    uses_rural_code = (region == "rural") == matches_region
    postcode = np.where(
        uses_rural_code, rng.choice(RURAL_POSTCODES, n), rng.choice(URBAN_POSTCODES, n)
    )

    education_cat = pd.Categorical(education, categories=EDUCATION_LEVELS, ordered=True)
    inputs = {
        **skills,
        "experience_years": experience,
        "education": education_cat.codes.astype(float),
    }
    score = sum(w * _standardise(inputs[col]) for col, w in QUALIFICATION_WEIGHTS.items())
    score = _standardise(np.asarray(score))
    qualified = _top_share(score, cfg.qualified_share)

    # Noise is drawn before bias is applied, so it is identical across bias levels.
    noise = rng.normal(0.0, 1.0, n)
    groups = {"gender": gender, "region": region}
    penalty = np.zeros(n)
    for attribute, group in cfg.bias_targets.items():
        penalty += groups[attribute] == group
    past_hiring_score = score + cfg.label_noise * noise - bias_strength * penalty
    hired = _top_share(past_hiring_score, cfg.hire_rate)

    df = pd.DataFrame(
        {
            "candidate_id": np.arange(1, n + 1),
            **skills,
            "experience_years": experience,
            "education": education_cat,
            "postcode": postcode,
            "gender": gender,
            "region": region,
            "qualification_score": score.round(4),
            "qualified": qualified,
            LABEL_COLUMN: hired,
        }
    )
    df.attrs["generator"] = {
        "name": "hiring.generate_candidates",
        "seed": seed,
        "bias_strength": bias_strength,
        "config": cfg.model_dump(),
    }
    return df


def historical_records(df: pd.DataFrame) -> pd.DataFrame:
    """What a company holds about past applicants: everything except hidden ground truth."""
    return df.drop(columns=list(HIDDEN_COLUMNS))


def applicant_view(df: pd.DataFrame) -> pd.DataFrame:
    """What an agent sees about new applicants: no ground truth and no hiring outcome."""
    return df.drop(columns=[*HIDDEN_COLUMNS, LABEL_COLUMN])


def _standardise(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - values.mean()) / std


def _top_share(score: np.ndarray, share: float) -> np.ndarray:
    """1 for the top `share` of scores (at least one), 0 otherwise. Ties broken by order."""
    k = max(1, round(share * score.size))
    order = np.argsort(-score, kind="stable")
    selected = np.zeros(score.size, dtype=int)
    selected[order[:k]] = 1
    return selected
