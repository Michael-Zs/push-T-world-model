import pytest

from push_t_jepa.evaluate import summarize_distances


def test_summarize_distances_reports_mean_improvement():
    summary = summarize_distances([(0.4, 0.3), (0.2, 0.25)])
    assert summary["count"] == 2
    assert summary["mean_initial_distance"] == pytest.approx(0.3)
    assert summary["mean_final_distance"] == pytest.approx(0.275)
    assert summary["mean_improvement"] == pytest.approx(0.025)
