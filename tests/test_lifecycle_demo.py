"""The lifecycle demo: each flaw is first caught at its intended phase."""

import pytest
from examples.lifecycle_demo import PHASES, main, run

EXPECTED = {
    "responsible": (None, "Released"),
    "aggressive_automation": ("planning", "Blocked"),
    "biased_history": ("data", "Blocked"),
    "biased_instructions": ("design", "Blocked"),
    "faulty_code": ("testing", "Blocked"),
    "skipped_review": ("deployment", "Blocked"),
    "mitigated": ("data", "Released with waivers"),
}


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    out = tmp_path_factory.mktemp("lifecycle")
    return out, {o.scenario.key: o for o in run(quick=True, out=out)}


@pytest.mark.parametrize("key", list(EXPECTED))
def test_each_flaw_is_caught_at_its_phase(outcomes, key):
    _, by_key = outcomes
    first_caught, verdict = EXPECTED[key]
    assert by_key[key].first_caught == first_caught
    assert by_key[key].verdict == verdict


def test_skipped_design_is_shown_as_skipped(outcomes):
    _, by_key = outcomes
    assert by_key["skipped_review"].status("design") == "skipped"


def test_reports_and_index_are_saved(outcomes):
    out, by_key = outcomes
    for key, outcome in by_key.items():
        for phase in outcome.reports:
            assert (out / key / f"{phase}.html").is_file()
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "Where each flaw is caught" in index
    assert all(phase in index for phase in PHASES)


def test_main_prints_the_table(tmp_path, capsys):
    assert main(["--quick", "--out", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "First caught" in output
    assert "Released with waivers" in output
