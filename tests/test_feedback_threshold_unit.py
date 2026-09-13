"""Degrading feedback offline must be an exact resample of the stored verdict counts."""

import numpy as np

from benchmarks.dars_eval.feedback_threshold import crossing, resample_counts, utility


def test_no_noise_leaves_the_counts_untouched():
    s = np.array([0, 3, 17, 2])
    f = np.array([6, 1, 0, 9])
    assert np.array_equal(resample_counts(s, f, 0.0, np.random.default_rng(0)), s)


def test_flipping_every_verdict_swaps_successes_and_failures():
    s = np.array([0, 3, 17, 2])
    f = np.array([6, 1, 0, 9])
    assert np.array_equal(resample_counts(s, f, 1.0, np.random.default_rng(0)), f)


def test_expected_successes_match_the_binomial_mean():
    # each of the s successes survives with probability 1-eps, each of the f failures
    # becomes a success with probability eps, so E[s'] = s(1-eps) + f*eps
    rng = np.random.default_rng(7)
    s = np.full(4000, 10)
    f = np.full(4000, 30)
    eps = 0.25
    got = resample_counts(s, f, eps, rng).mean()
    assert abs(got - (10 * (1 - eps) + 30 * eps)) < 0.2


def test_noise_never_changes_the_number_of_verdicts():
    rng = np.random.default_rng(3)
    s = np.array([5, 0, 12])
    f = np.array([7, 4, 1])
    s2 = resample_counts(s, f, 0.3, rng)
    f2 = (s + f) - s2
    assert np.array_equal(s2 + f2, s + f)
    assert (s2 >= 0).all() and (f2 >= 0).all()


def test_utility_is_the_laplace_smoothed_rate():
    assert utility(np.array([0]), np.array([0]))[0] == 0.5
    assert utility(np.array([8]), np.array([0]))[0] == 9 / 10
    assert utility(np.array([0]), np.array([8]))[0] == 1 / 10


def test_crossing_finds_where_the_curve_falls_through_a_baseline():
    curve = [{"eps": 0.0, "auroc": 0.90}, {"eps": 0.1, "auroc": 0.80},
             {"eps": 0.2, "auroc": 0.60}, {"eps": 0.3, "auroc": 0.50}]
    at = crossing(curve, "auroc", 0.70, "higher")
    assert 0.1 < at < 0.2                    # interpolates between the bracketing points
    assert crossing(curve, "auroc", 0.40, "higher") is None      # never falls through
    lower = [{"eps": 0.0, "harmful_deletion": 0.05}, {"eps": 0.2, "harmful_deletion": 0.20}]
    assert crossing(lower, "harmful_deletion", 0.10, "lower") is not None


def test_subsampling_keeps_the_expected_share_of_verdicts():
    from benchmarks.dars_eval.feedback_threshold import subsample_counts
    rng = np.random.default_rng(11)
    s = np.full(5000, 8)
    f = np.full(5000, 4)
    s2, f2 = subsample_counts(s, f, 0.25, rng)
    assert abs(s2.mean() - 2.0) < 0.1 and abs(f2.mean() - 1.0) < 0.1
    assert np.array_equal(subsample_counts(s, f, 1.0, rng)[0], s)


def test_asymmetric_resample_reduces_to_identity_and_full_flip():
    from benchmarks.dars_eval.feedback_threshold import asymmetric_resample
    s = np.array([0, 3, 17, 2])
    f = np.array([6, 1, 0, 9])
    assert np.array_equal(asymmetric_resample(s, f, 0.0, 0.0, np.random.default_rng(0)), s)
    assert np.array_equal(asymmetric_resample(s, f, 1.0, 1.0, np.random.default_rng(0)), f)
    rng = np.random.default_rng(5)
    got = asymmetric_resample(np.full(4000, 10), np.full(4000, 30), 0.4, 0.1, rng).mean()
    assert abs(got - (10 * 0.6 + 30 * 0.1)) < 0.2


def test_judge_flip_rates_recover_precision_and_recall():
    from benchmarks.dars_eval.feedback_threshold import judge_flip_rates
    r = judge_flip_rates(precision=0.6, recall=0.5, base_rate=0.4)
    tp, fp = 0.4 * (1 - r["fnr"]), 0.6 * r["fpr"]
    assert abs(tp / (tp + fp) - 0.6) < 1e-12 and abs(r["fnr"] - 0.5) < 1e-12
