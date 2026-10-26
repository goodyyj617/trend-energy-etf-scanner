"""The experiment log (docs/research/lab/experiments.json) that the '실험 기록' page reads."""
import pandas as pd

from lab import experiments as xp


def test_registry_entries_are_complete_and_figures_committed():
    rows = xp.load()
    assert rows and [e["date"] for e in rows] == sorted((e["date"] for e in rows), reverse=True)
    assert len({e["id"] for e in rows}) == len(rows)
    for e in rows:
        assert e["why"].strip() and e["results"]
        for f in e["figures"]:
            assert (xp.ROOT / f["path"]).exists(), f["path"]
            assert f["path"].startswith("docs/research/lab/figures/") and f["path"].endswith(".png")
        if e.get("memo"):
            assert (xp.ROOT / e["memo"]).exists(), e["memo"]


def test_chart_frame_reads_summaries(tmp_path):
    pd.DataFrame({"sample": ["full", "recent"], "entry": "always", "layer": "hold", "score": "s_rs",
                  "median_goal_excess": [0.01, -0.02]}).to_csv(tmp_path / "summary.csv", index=False)
    frame, kind, _ = xp.chart_frame({"kind": "scores"}, tmp_path)
    assert kind == "excess" and set(frame["표본"]) == {"1993–2023", "2017–2023"} and len(frame) == 2
    assert xp.chart_frame({"kind": "screens"}, tmp_path) is None
    assert xp.results_folder({"results_dir": "x"}, "") is None
