"""Tests for the screening-agent interface (eguard.agents)."""

import numpy as np
import pandas as pd
import pytest

from eguard import AgentError
from eguard.agents import (
    DECISION_NAME,
    FunctionAgent,
    ScreeningAgent,
    agent_name,
    as_screening_agent,
    screen,
)


@pytest.fixture
def applicants():
    return pd.DataFrame(
        {"skill": [90, 40, 75, 10], "region": ["urban", "rural", "rural", "urban"]},
        index=[10, 11, 12, 13],
    )


class Threshold:
    name = "threshold-70"

    def decide(self, applicants):
        return applicants["skill"] >= 70


def top_skill(applicants):
    return applicants["skill"] >= 80


# ---------------------------------------------------------------------------- wrapping


def test_class_with_decide_is_a_screening_agent():
    assert isinstance(Threshold(), ScreeningAgent)
    assert as_screening_agent(Threshold()).name == "threshold-70"


def test_function_is_wrapped():
    agent = as_screening_agent(top_skill)
    assert isinstance(agent, FunctionAgent)
    assert isinstance(agent, ScreeningAgent)
    assert agent.name == "top_skill"


def test_function_agent_custom_name():
    assert FunctionAgent(top_skill, name="strict").name == "strict"


def test_non_agent_rejected():
    with pytest.raises(TypeError, match="not a screening agent"):
        as_screening_agent(42)


def test_agent_name_falls_back_to_class_name():
    class Nameless:
        def decide(self, applicants):
            return [True] * len(applicants)

    assert agent_name(Nameless()) == "Nameless"


# ------------------------------------------------------------------------------ output


def test_series_output(applicants):
    decisions = screen(Threshold(), applicants)
    assert decisions.tolist() == [True, False, True, False]
    assert decisions.name == DECISION_NAME
    assert decisions.dtype == bool
    assert decisions.index.equals(applicants.index)


@pytest.mark.parametrize(
    "output",
    [
        [True, False, True, False],
        [1, 0, 1, 0],
        np.array([True, False, True, False]),
        np.array([1.0, 0.0, 1.0, 0.0]),
    ],
)
def test_list_and_array_outputs(applicants, output):
    assert screen(lambda a: output, applicants).tolist() == [True, False, True, False]


def test_shuffled_series_is_realigned(applicants):
    def shuffled(a):
        return (a["skill"] >= 70).iloc[::-1]

    assert screen(shuffled, applicants).tolist() == [True, False, True, False]


def test_empty_applicants(applicants):
    assert screen(Threshold(), applicants.iloc[0:0]).empty


# ---------------------------------------------------------------------------- safety


def test_agent_cannot_modify_callers_data(applicants):
    def vandal(a):
        a["skill"] = 0
        return [True] * len(a)

    screen(vandal, applicants)
    assert applicants["skill"].tolist() == [90, 40, 75, 10]


def test_agent_exception_becomes_agent_error(applicants):
    def crashes(a):
        raise KeyError("experience")

    with pytest.raises(AgentError, match="'crashes' raised KeyError"):
        screen(crashes, applicants)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (None, "returned None"),
        ([True, False], "2 decisions for 4 applicants"),
        ([True, None, True, False], "missing"),
        ([1, 2, 0, 1], "True/False or 1/0"),
        (["yes", "no", "yes", "no"], "True/False or 1/0"),
        (pd.Series([True] * 4, index=[0, 1, 2, 3]), "different applicants"),
        (pd.Series([True] * 3, index=[10, 11, 12]), "different applicants"),
    ],
)
def test_invalid_output_rejected(applicants, output, message):
    with pytest.raises(AgentError, match=message):
        screen(lambda a: output, applicants)


def test_non_agent_passed_to_screen(applicants):
    with pytest.raises(TypeError):
        screen("not an agent", applicants)
