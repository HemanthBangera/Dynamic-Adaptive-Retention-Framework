"""The clustered AUROC must agree with DeLong for independent items and widen for clustered ones."""

import numpy as np

from benchmarks.dars_eval.stats import (
    auroc_cluster_bootstrap, auroc_cluster_bootstrap_paired, auroc_delong, auroc_delong_paired,
)


def _data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.45
    s = y * 0.8 + rng.normal(size=n)
    return s, y


def test_point_estimate_equals_delong():
    s, y = _data()
    assert abs(auroc_cluster_bootstrap(s, y, n_boot=200)["auc"] - auroc_delong(s, y)["auc"]) < 1e-12


def test_singleton_clusters_give_roughly_the_delong_interval():
    s, y = _data()
    boot = auroc_cluster_bootstrap(s, y, n_boot=1000, seed=1)
    dl = auroc_delong(s, y)
    width_b, width_d = boot["ci_hi"] - boot["ci_lo"], dl["ci_hi"] - dl["ci_lo"]
    assert abs(width_b - width_d) / width_d < 0.2


def test_duplicated_items_within_clusters_widen_the_interval():
    s, y = _data(n=300)
    # every item repeated 10 times: i.i.d. resampling pretends there are 3,000 independent items
    s10, y10 = np.repeat(s, 10), np.repeat(y, 10)
    clusters = np.repeat(np.arange(300), 10)
    iid = auroc_cluster_bootstrap(s10, y10, n_boot=600, seed=2)
    clustered = auroc_cluster_bootstrap(s10, y10, clusters, n_boot=600, seed=2)
    assert (clustered["ci_hi"] - clustered["ci_lo"]) > 2 * (iid["ci_hi"] - iid["ci_lo"])


def test_paired_difference_matches_delong_sign_and_detects_a_real_gap():
    rng = np.random.default_rng(3)
    n = 2000
    y = rng.random(n) < 0.5
    good = y * 1.0 + rng.normal(size=n)
    weak = y * 0.3 + rng.normal(size=n)
    boot = auroc_cluster_bootstrap_paired(good, weak, y, n_boot=500, seed=4)
    dl = auroc_delong_paired(good, weak, y)
    assert abs(boot["diff"] - dl["diff"]) < 1e-12
    assert boot["ci_lo"] > 0 and boot["p_value"] < 0.01


def test_identical_scores_have_zero_difference_and_p_one():
    s, y = _data(n=400)
    out = auroc_cluster_bootstrap_paired(s, s, y, n_boot=200)
    assert out["diff"] == 0 and out["p_value"] == 1.0
