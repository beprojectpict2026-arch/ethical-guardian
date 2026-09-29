"""Smoke test: the end-to-end demo runs and writes every report and the index page."""

from examples.hiring_demo import main


def test_quick_demo_writes_reports(tmp_path, capsys):
    assert main(["--quick", "--out", str(tmp_path)]) == 0

    names = {path.name for path in tmp_path.iterdir()}
    assert "index.html" in names
    assert "01_data.html" in names
    assert len([n for n in names if n.endswith(".html")]) == 6
    assert len([n for n in names if n.endswith(".json")]) == 5

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "ML screener, no postcode" in index
    assert "<svg" in index
    assert "Blocked" in index

    assert "Done." in capsys.readouterr().out
