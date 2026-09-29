"""Example ML screener: logistic regression trained on a company's historical hiring records.

This reproduces the pattern behind Amazon's scrapped recruiting tool: a model trained on
biased past decisions learns the bias. By default the model is "unaware" (protected
attributes are removed before training), yet it can still discriminate through proxies such
as postcode. That is exactly what the checks are designed to reveal.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from eguard.scenarios.hiring import EDUCATION_LEVELS, PROTECTED_COLUMNS, shortlist_top

NUMERIC_FEATURES = ("skill_technical", "skill_communication", "skill_domain", "experience_years")


class MLScreener:
    """Logistic regression screener. Call `fit` on historical records before `decide`."""

    def __init__(
        self,
        *,
        use_protected: bool = False,
        drop: Sequence[str] = (),
        shortlist_rate: float = 0.2,
    ):
        self.use_protected = use_protected
        self.drop = tuple(drop)
        self.shortlist_rate = shortlist_rate
        self.name = "ml-screener-aware" if use_protected else "ml-screener"
        self._model: Pipeline | None = None

    def fit(self, records: pd.DataFrame, target: str = "hired") -> Self:
        """Train on historical records (features plus the past outcome column)."""
        numeric = [c for c in NUMERIC_FEATURES if c not in self.drop]
        categorical = ["postcode", *(PROTECTED_COLUMNS if self.use_protected else ())]
        categorical = [c for c in categorical if c not in self.drop]

        steps = [("numeric", StandardScaler(), numeric)]
        if "education" not in self.drop:
            steps.append(
                ("education", OrdinalEncoder(categories=[list(EDUCATION_LEVELS)]), ["education"])
            )
        if categorical:
            steps.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))

        self._model = Pipeline(
            [
                ("prepare", ColumnTransformer(steps)),
                ("classify", LogisticRegression(max_iter=1000)),
            ]
        )
        self._model.fit(records, records[target])
        return self

    def decide(self, applicants: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("MLScreener must be fitted before deciding; call fit() first.")
        if applicants.empty:
            return pd.Series(False, index=applicants.index)
        probability = self._model.predict_proba(applicants)[:, 1]
        return shortlist_top(pd.Series(probability, index=applicants.index), self.shortlist_rate)
