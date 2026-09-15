"""make_figures: the module draws without artifacts (Figure 1) and the test layout names every input."""

from benchmarks.dars_eval import make_figures


def test_figure1_draws_without_artifacts(tmp_path):
    assert make_figures.fig1({"out": tmp_path}, n_boot=10) == []
    assert (tmp_path / "fig1_architecture.pdf").stat().st_size > 0
    assert (tmp_path / "fig1_architecture.png").stat().st_size > 0


def test_test_layout_names_every_input_the_figures_read():
    lay = make_figures.layout("test")
    for key in ("e1_recall", "e1_reader", "e2_prec", "e2_dyn", "e2_noise", "e2_sources", "lme_evict",
                "e4", "e5", "e6", "e8", "e9_eval", "e9_analysis", "e10_eval"):
        assert key in lay, key
    assert set(lay["e10_eval"]) == {"test_in", "test_out"}
    assert len(lay["lme_evict"]["random"]) == 5          # five random-eviction seeds, as in run_test_phase.sh
