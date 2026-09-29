"""Tests for the reference screener and shortlist helper."""

import numpy as np
import pandas as pd
import pytest

from eguard.agents import ScreeningAgent, screen
from eguard.metrics import demographic_parity_ratio
from eguard.scenarios.hiring import (
    SkillBasedScreener,
    applicant_view,
    generate_candidates,
    shortlist_top,
)

# ------------------------------------------------------------------------ shortlist_top


def test_shortlists_exact_share():
    scores = pd.Series(np.arange(100, dtype=float))
    result = shortlist_top(scores, 0.2)
    assert result.sum() == 20
    assert result.iloc[80:].all()


def test_at_least_one_is_shortlisted():
    assert shortlist_top(pd.Series([1.0, 2.0, 3.0]), 0.01).sum() == 1


def test_ties_are_broken_randomly_but_reproducibly():
    scores = pd.Series([50.0] * 100)
    first = shortlist_top(scores, 0.2, seed=0)
    assert first.equals(shortlist_top(scores, 0.2, seed=0))
    assert not first.equals(shortlist_top(scores, 0.2, seed=1))
    assert not first.iloc[:20].all()  # not simply the first 20 rows


def test_empty_scores():
    assert shortlist_top(pd.Series([], dtype=float), 0.2).empty


@pytest.mark.parametrize(
    ("scores", "rate", "message"),
    [
        (pd.Series([1.0, np.nan]), 0.5, "missing"),
        (pd.Series([1.0, 2.0]), 0.0, "between 0 and 1"),
        (pd.Series([1.0, 2.0]), 1.0, "between 0 and 1"),
    ],
)
def test_invalid_shortlist_input(scores, rate, message):
    with pytest.raises(ValueError, match=message):
        shortlist_top(scores, rate)


# ------------------------------------------------------------------ SkillBasedScreener


@pytest.fixture(scope="module")
def pool():
    return generate_candidates(20_000, seed=5)


def test_is_a_screening_agent():
    assert isinstance(SkillBasedScreener(), ScreeningAgent)


def test_shortlists_twenty_percent(pool):
    assert screen(SkillBasedScreener(), applicant_view(pool)).sum() == 4_000


def test_is_fair_on_protected_attributes(pool):
    decisions = screen(SkillBasedScreener(), applicant_view(pool))
    for attribute in ("gender", "region"):
        assert demographic_parity_ratio(decisions, pool[attribute]) > 0.9


def test_mostly_shortlists_qualified_applicants(pool):
    decisions = screen(SkillBasedScreener(), applicant_view(pool))
    assert pool.loc[decisions, "qualified"].mean() > 0.7


def test_ignores_protected_attributes_and_postcode(pool):
    applicants = applicant_view(pool).head(1_000)
    altered = applicants.assign(gender="female", region="rural", postcode="000000")
    agent = SkillBasedScreener()
    assert screen(agent, applicants).equals(screen(agent, altered))


def test_empty_applicants():
    applicants = applicant_view(generate_candidates(20, seed=0)).iloc[0:0]
    assert screen(SkillBasedScreener(), applicants).empty


def test_constant_feature_does_not_break_scoring():
    applicants = applicant_view(generate_candidates(50, seed=0)).assign(experience_years=5.0)
    assert screen(SkillBasedScreener(), applicants).sum() == 10


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda a: a.drop(columns=["skill_domain"]), "missing column"),
        (lambda a: a.assign(education="phd"), "unknown education"),
    ],
)
def test_invalid_applicants(mutate, message):
    applicants = applicant_view(generate_candidates(50, seed=0))
    with pytest.raises(ValueError, match=message):
        SkillBasedScreener().score(mutate(applicants))


def test_invalid_shortlist_rate():
    with pytest.raises(ValueError, match="shortlist_rate"):
        SkillBasedScreener(shortlist_rate=1.5)
