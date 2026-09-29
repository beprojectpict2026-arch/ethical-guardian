"""Tests for the example ML and LLM screeners (no network: the LLM client is faked)."""

import json
import re

import pandas as pd
import pytest
from examples.agents.llm_screener import LLMScreener, describe_applicants, parse_scores
from examples.agents.ml_screener import MLScreener

from eguard import AgentError
from eguard.agents import screen
from eguard.metrics import demographic_parity_ratio
from eguard.scenarios.hiring import applicant_view, generate_candidates, historical_records

# --------------------------------------------------------------------------- ML screener


@pytest.fixture(scope="module")
def history():
    return historical_records(generate_candidates(10_000, seed=1, bias_strength=1.0))


@pytest.fixture(scope="module")
def pool():
    return generate_candidates(10_000, seed=2)


def dprs(agent, pool):
    decisions = screen(agent, applicant_view(pool))
    return (
        demographic_parity_ratio(decisions, pool["gender"]),
        demographic_parity_ratio(decisions, pool["region"]),
    )


def test_ml_must_be_fitted_first(pool):
    with pytest.raises(AgentError, match="fitted"):
        screen(MLScreener(), applicant_view(pool))


def test_unaware_model_learns_region_bias_through_postcode(history, pool):
    gender, region = dprs(MLScreener().fit(history), pool)
    assert gender > 0.85
    assert region < 0.8


def test_removing_postcode_removes_region_bias(history, pool):
    _, region = dprs(MLScreener(drop=["postcode"]).fit(history), pool)
    assert region > 0.85


def test_aware_model_copies_historical_bias(history, pool):
    gender, region = dprs(MLScreener(use_protected=True).fit(history), pool)
    assert gender < 0.8
    assert region < 0.8


def test_ml_shortlist_rate_and_names(history, pool):
    agent = MLScreener().fit(history)
    assert screen(agent, applicant_view(pool)).sum() == 2_000
    assert agent.name == "ml-screener"
    assert MLScreener(use_protected=True).name == "ml-screener-aware"


def test_ml_can_drop_education(history, pool):
    agent = MLScreener(drop=["education", "postcode"]).fit(history)
    assert screen(agent, applicant_view(pool)).sum() == 2_000


def test_ml_empty_applicants(history, pool):
    agent = MLScreener().fit(history)
    assert screen(agent, applicant_view(pool).iloc[0:0]).empty


# -------------------------------------------------------------------------- LLM screener

LINE = re.compile(r"^(\d+)\. .*technical skill: (\d+)/100", re.MULTILINE)


class FakeClient:
    """Scores each applicant by technical skill; can return bad answers first."""

    def __init__(self, bad_answers=0):
        self.bad_answers = bad_answers
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append(user)
        if self.bad_answers > 0:
            self.bad_answers -= 1
            return "not json"
        scores = [{"id": int(i), "score": float(s)} for i, s in LINE.findall(user)]
        return json.dumps({"scores": scores})


@pytest.fixture
def applicants():
    return applicant_view(generate_candidates(25, seed=3))


def test_llm_scores_follow_the_model(applicants, tmp_path):
    agent = LLMScreener(client=FakeClient(), cache_dir=tmp_path)
    scores = agent.score(applicants)
    assert scores.round().tolist() == applicants["skill_technical"].round().tolist()


def test_llm_batches_requests(applicants, tmp_path):
    client = FakeClient()
    agent = LLMScreener(client=client, batch_size=10, cache_dir=tmp_path)
    screen(agent, applicants)
    assert len(client.prompts) == 3
    assert agent.stats["calls"] == 3


def test_llm_cache_avoids_repeat_calls(applicants, tmp_path):
    first = LLMScreener(client=FakeClient(), cache_dir=tmp_path)
    decisions = screen(first, applicants)
    client = FakeClient()
    second = LLMScreener(client=client, cache_dir=tmp_path)
    assert screen(second, applicants).equals(decisions)
    assert client.prompts == []
    assert second.stats["cache_hits"] == 3


def test_llm_without_cache(applicants):
    agent = LLMScreener(client=FakeClient(), cache_dir=None)
    assert screen(agent, applicants).sum() == 5


def test_llm_retries_once_after_a_bad_answer(applicants, tmp_path):
    client = FakeClient(bad_answers=1)
    agent = LLMScreener(client=client, batch_size=25, cache_dir=tmp_path)
    assert screen(agent, applicants).sum() == 5
    assert len(client.prompts) == 2
    assert "Return exactly 25 entries" in client.prompts[1]


def test_llm_gives_up_and_does_not_cache_bad_answers(applicants, tmp_path):
    agent = LLMScreener(client=FakeClient(bad_answers=5), batch_size=25, cache_dir=tmp_path)
    with pytest.raises(AgentError, match="not valid JSON"):
        screen(agent, applicants)
    assert list(tmp_path.iterdir()) == []


def test_protected_attributes_can_be_withheld(applicants):
    assert "gender:" in describe_applicants(applicants)
    hidden = describe_applicants(applicants, include_protected=False)
    assert "gender:" not in hidden
    assert "region:" not in hidden
    assert "postcode:" in hidden


def test_llm_empty_applicants(applicants):
    agent = LLMScreener(client=FakeClient(), cache_dir=None)
    assert screen(agent, applicants.iloc[0:0]).empty


def test_llm_name():
    assert LLMScreener("qwen2.5:3b", client=FakeClient()).name == "llm-screener (qwen2.5:3b)"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "not valid JSON"),
        ('["a list"]', "no 'scores' list"),
        ('{"scores": [{"id": 1}]}', "invalid score entry"),
        ('{"scores": [{"id": 1, "score": 50}]}', r"missing scores for applicant\(s\) \[2\]"),
    ],
)
def test_parse_scores_rejects_bad_responses(content, message):
    with pytest.raises(ValueError, match=message):
        parse_scores(content, pd.Index([10, 11]))


def test_parse_scores_clips_and_accepts_strings():
    content = '{"scores": [{"id": "1", "score": "150"}, {"id": 2, "score": -5}]}'
    assert parse_scores(content, pd.Index([10, 11])).tolist() == [100.0, 0.0]
