from __future__ import annotations

import numpy as np
import pytest

from diffusion_best_of_n.controller import (
    ALLOW_HIGH_N,
    AUDIT_ROLLOUTS,
    BLOCK_HIGH_N,
    CALIBRATE_SCORER,
    HIGH_N_BLOCKED,
    INCREASE_DIVERSITY,
    INCREASE_N,
    LATENCY_LIMITED,
    LOW_DIVERSITY_STOP,
    STOP_EARLY,
    UNDERPOWERED_AUDIT,
    AuditThenSampleConfig,
    audit_then_sample,
    audit_then_sample_adaptive,
    empirical_bernstein_lcb,
    empirical_bernstein_ucb,
    validate_repair_with_bounds,
)


def synthetic_trajectories(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    traj = rng.normal(size=(int(n), 4, 2))
    traj[:, :, 0] += np.linspace(-1.0, 1.0, int(n))[:, None]
    return traj


def audited_diversity(n: int) -> dict[str, float]:
    return {
        "effective_sample_diversity": float(n),
        "duplicate_collapse_rate": 0.0,
        "trajectory_cluster_entropy": 1.0,
        "trajectory_cluster_count": 8.0,
    }


def controller_config() -> AuditThenSampleConfig:
    return AuditThenSampleConfig(
        bootstrap_trials=40,
        confidence_method="both",
        min_effective_diversity=1.5,
        min_tail_rank_correlation=0.10,
        min_score_utility_correlation=0.10,
    )


def test_empirical_bernstein_bounds_are_conservative_and_tighten_for_constants():
    values = np.ones(24)
    assert empirical_bernstein_lcb(values[:12], delta=0.01) == pytest.approx(1.0)
    assert empirical_bernstein_lcb(values, delta=0.01) == pytest.approx(1.0)
    mixed = np.linspace(0.0, 1.0, 32)
    assert empirical_bernstein_lcb(mixed, delta=0.01) <= float(np.mean(mixed))
    assert empirical_bernstein_ucb(mixed, delta=0.01) >= float(np.mean(mixed))


def test_controller_allows_high_n_only_when_all_lower_bound_gates_are_positive():
    n = 96
    utilities = np.linspace(-1.0, 1.0, n)
    scores = utilities + np.random.default_rng(1).normal(scale=0.01, size=n)
    result = audit_then_sample(
        synthetic_trajectories(n, seed=1),
        scores,
        utilities,
        n_values=[1, 4, 16, 64],
        k_values=[4, 8],
        lambda_cost=0.0001,
        diversity_diagnostics=audited_diversity(72),
        config=controller_config(),
        seed=1,
    )
    diag = result.confidence_diagnostics
    assert result.decision_label == ALLOW_HIGH_N
    assert result.action_recommendation == INCREASE_N
    assert diag["admit_high_N"] is True
    assert diag["utility_gain_lcb"] > 0.0
    assert diag["tail_utility_lcb"] > 0.0
    assert diag["latency_adjusted_gain_lcb"] > 0.0


@pytest.mark.parametrize("mode", ["anti", "shuffled", "adversarial_tail", "duplicated_artifact"])
def test_controller_never_admits_high_n_for_bad_tails_or_duplicates(mode: str):
    n = 96
    rng = np.random.default_rng(2)
    utilities = np.linspace(-1.0, 1.0, n)
    trajectories = synthetic_trajectories(n, seed=2)
    diversity = audited_diversity(72)
    if mode == "anti":
        scores = -utilities + rng.normal(scale=0.01, size=n)
    elif mode == "shuffled":
        scores = rng.permutation(utilities)
    elif mode == "adversarial_tail":
        scores = utilities.copy()
        scores[utilities < np.quantile(utilities, 0.25)] += 4.0
    else:
        scores = utilities.copy()
        scores[:24] = 5.0
        utilities[:24] = -1.0
        trajectories[:72] = trajectories[0]
        diversity = {
            "effective_sample_diversity": 4.0,
            "duplicate_collapse_rate": 0.75,
            "trajectory_cluster_entropy": 0.1,
            "trajectory_cluster_count": 1.0,
        }
    result = audit_then_sample(
        trajectories,
        scores,
        utilities,
        n_values=[1, 4, 16, 64],
        k_values=[4],
        lambda_cost=0.0,
        diversity_diagnostics=diversity,
        config=controller_config(),
        seed=2,
    )
    assert result.confidence_diagnostics["admit_high_N"] is False
    assert result.selected_n == 1
    assert result.action_recommendation in {BLOCK_HIGH_N, CALIBRATE_SCORER, INCREASE_DIVERSITY}
    assert result.decision_label in {HIGH_N_BLOCKED, "alignment_audit_failed", LOW_DIVERSITY_STOP}


def test_underpowered_and_effective_n_limited_audits_abstain_even_with_good_points():
    n = 96
    utilities = np.linspace(-1.0, 1.0, n)
    scores = utilities + np.random.default_rng(3).normal(scale=0.01, size=n)
    result = audit_then_sample(
        synthetic_trajectories(n, seed=3),
        scores,
        utilities,
        n_values=[1, 8, 64],
        k_values=[8],
        lambda_cost=0.0,
        diversity_diagnostics={**audited_diversity(8), "effective_sample_diversity": 8.0},
        config=controller_config(),
        seed=3,
    )
    assert result.decision_label == UNDERPOWERED_AUDIT
    assert result.action_recommendation == AUDIT_ROLLOUTS
    assert result.confidence_diagnostics["abstention_reason"] == "underpowered_effective_n"


def test_unknown_utility_mode_never_admits_high_n():
    n = 64
    scores = np.linspace(-1.0, 1.0, n)
    result = audit_then_sample(
        synthetic_trajectories(n, seed=4),
        scores,
        None,
        n_values=[1, 8, 64],
        k_values=[8],
        diversity_diagnostics=audited_diversity(64),
        config=controller_config(),
        seed=4,
    )
    assert result.action_recommendation == AUDIT_ROLLOUTS
    assert result.confidence_diagnostics["admit_high_N"] is False
    assert result.confidence_diagnostics["abstention_reason"] == "unknown_utility"


def test_controller_prefers_smaller_nk_when_latency_adjusted_lcb_is_negative():
    n = 96
    utilities = np.linspace(0.0, 0.02, n)
    result = audit_then_sample(
        synthetic_trajectories(n, seed=5),
        utilities,
        utilities,
        n_values=[1, 8, 64],
        k_values=[1, 16],
        lambda_cost=0.03,
        diversity_diagnostics=audited_diversity(96),
        config=controller_config(),
        seed=5,
    )
    assert result.decision_label == LATENCY_LIMITED
    assert result.action_recommendation == STOP_EARLY
    assert (result.selected_n, result.selected_k) == (1, 1)
    assert result.confidence_diagnostics["latency_adjusted_gain_lcb"] <= 0.0


def test_affine_fallback_repair_passes_only_with_held_out_lower_bound_gates():
    n = 160
    rng = np.random.default_rng(6)
    utilities = np.linspace(-1.0, 1.0, n)
    scores = -utilities + rng.normal(scale=0.015, size=n)
    repair = validate_repair_with_bounds(
        scores,
        utilities,
        n_values=[1, 64],
        seed=6,
        min_tail_improvement=0.50,
        config=AuditThenSampleConfig(use_effective_n_for_bounds=False),
    )
    assert repair.success
    assert repair.repair_method == "affine"
    assert repair.calibration.slope < 0.0
    assert repair.repaired_tail_rank_correlation > repair.original_tail_rank_correlation
    assert repair.repaired_tail_utility_lcb > 0.0
    assert repair.repaired_latency_adjusted_gain_lcb > 0.0


@pytest.mark.parametrize("mode", ["random", "drift"])
def test_repair_fails_safely_for_random_scores_and_calibration_drift(mode: str):
    n = 160
    rng = np.random.default_rng(7)
    utilities = np.linspace(-1.0, 1.0, n)
    if mode == "random":
        scores = rng.normal(size=n)
    else:
        scores = utilities.copy()
        scores[n // 2 :] = -scores[n // 2 :]
        scores += rng.normal(scale=0.02, size=n)
    repair = validate_repair_with_bounds(
        scores,
        utilities,
        n_values=[1, 64],
        seed=7,
        min_tail_improvement=0.50,
        config=AuditThenSampleConfig(use_effective_n_for_bounds=False),
    )
    assert not repair.success
    assert repair.recommendation == BLOCK_HIGH_N


def test_adaptive_sampling_stops_when_gain_ucb_is_below_latency_cost():
    n = 96
    utilities = np.linspace(0.0, 0.02, n)
    scores = utilities + np.random.default_rng(8).normal(scale=0.0001, size=n)
    result = audit_then_sample_adaptive(
        synthetic_trajectories(n, seed=8),
        scores,
        utilities,
        batch_size=16,
        n_values=[1, 16, 32, 64, 96],
        k_values=[1, 16],
        lambda_cost=0.03,
        diversity_diagnostics=audited_diversity(96),
        config=controller_config(),
        seed=8,
    )
    assert result.decision_label == LATENCY_LIMITED
    assert result.action_recommendation == STOP_EARLY
    assert result.confidence_diagnostics["adaptive_stopped"] is True
    assert result.confidence_diagnostics["adaptive_stopping_savings"] > 0.0
