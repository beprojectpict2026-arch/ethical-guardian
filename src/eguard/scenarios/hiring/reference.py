"""Reference screening agent: the fair baseline every hiring agent is compared against.

It scores applicants only on job-relevant features (skills, experience, education), never on
protected attributes or postcode, and shortlists a fixed share. Evaluations run it on the same
applicants as the agent under test, so differences can be attributed to the agent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eguard.scenarios.hiring.generator import EDUCATION_LEVELS, SKILL_COLUMNS

REFERENCE_WEIGHTS = {
    "skill_technical": 0.25,
    "skill_communication": 0.25,
    "skill_domain": 0.25,
    "experience_years": 0.15,
    "education": 0.10,
}


def shortlist_top(scores: pd.Series, rate: float, *, seed: int = 0) -> pd.Series:
    """Shortlist the top `rate` share of applicants by score (at least one if any).

    Ties are broken at random with a fixed seed, so no applicant is favoured merely for
    appearing earlier in the data.
    """
    if not 0.0 < rate < 1.0:
        raise ValueError("`rate` must be between 0 and 1.")
    result = pd.Series(False, index=scores.index)
    n = len(scores)
    if n == 0:
        return result
    if scores.isna().any():
        raise ValueError("scores must not contain missing values.")
    k = max(1, round(rate * n))
    tiebreak = np.random.default_rng(seed).random(n)
    order = np.lexsort((tiebreak, -scores.to_numpy(dtype=float)))
    result.iloc[order[:k]] = True
    return result


class SkillBasedScreener:
    """Shortlists the top `shortlist_rate` of applicants by a job-relevant merit score."""

    name = "reference-skill-based"

    def __init__(self, shortlist_rate: float = 0.2):
        if not 0.0 < shortlist_rate < 1.0:
            raise ValueError("`shortlist_rate` must be between 0 and 1.")
        self.shortlist_rate = shortlist_rate

    def score(self, applicants: pd.DataFrame) -> pd.Series:
        """Weighted sum of standardised skills, experience and education."""
        missing = [col for col in REFERENCE_WEIGHTS if col not in applicants.columns]
        if missing:
            raise ValueError(f"applicants are missing column(s): {', '.join(missing)}.")
        levels = applicants["education"].astype(str)
        unknown = sorted(set(levels) - set(EDUCATION_LEVELS))
        if unknown:
            raise ValueError(
                f"unknown education level(s) {unknown}; expected one of {EDUCATION_LEVELS}."
            )
        rank = {level: float(i) for i, level in enumerate(EDUCATION_LEVELS)}
        inputs = {col: applicants[col].astype(float) for col in SKILL_COLUMNS}
        inputs["experience_years"] = applicants["experience_years"].astype(float)
        inputs["education"] = levels.map(rank)
        return sum(weight * _zscore(inputs[col]) for col, weight in REFERENCE_WEIGHTS.items())

    def decide(self, applicants: pd.DataFrame) -> pd.Series:
        if applicants.empty:
            return pd.Series(False, index=applicants.index)
        return shortlist_top(self.score(applicants), self.shortlist_rate)


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return values * 0.0
    return (values - values.mean()) / std
