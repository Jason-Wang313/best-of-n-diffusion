from __future__ import annotations

import numpy as np
import pytest

from diffusion_audit.alignment import (
    high_n_regret,
    score_utility_correlation,
    tail_rank_correlation,
    top_score_tail_real_utility,
)
from diffusion_audit.deployment import (
    ALLOW_HIGH_N,
    BLOCK_HIGH_N,
    CALIBRATE_RERANKER,
    INCREASE_DIVERSITY,
    REDUCE_DENOISING_STEPS,
    STOP_EARLY,
    GateInputs,
    deployment_gate,
)
from diffusion_audit.latency import (
    latency_adjusted_utility,
    stop_rule_for_k,
    stop_rule_for_n,
    total_budget,
    utility_per_diffusion_step,
)


def test_alignment_metrics_expected_signs_and_tail_utility():
    scores = np.arange(10, dtype=float)
    utilities = np.arange(10, dtype=float)
    assert score_utility_correlation(scores, utilities) > 0.99
    assert tail_rank_correlation(scores, utilities, tail_fraction=0.4) > 0.99
    assert top_score_tail_real_utility(scores, utilities, tail_fraction=0.2) == pytest.approx(8.5)
    assert score_utility_correlation(scores, -utilities) < -0.99


def test_high_n_regret_detection():
    oracle = {1: 0.0, 8: 2.0, 64: 3.0}
    reranker = {1: 0.0, 8: 0.5, 64: -0.5}
    assert high_n_regret(oracle, reranker) == pytest.approx(3.5)


def test_latency_metrics_and_stop_rules():
    assert total_budget(4, 8) == 32
    assert utility_per_diffusion_step(16.0, 4, 8) == pytest.approx(0.5)
    assert latency_adjusted_utility(1.0, n=4, k=8, lambda_cost=0.01) == pytest.approx(0.68)
    assert stop_rule_for_n({1: 0.0, 2: 0.2, 4: 0.205, 8: 0.21}, min_marginal_gain=0.01) == 2
    assert stop_rule_for_k({2: 0.0, 4: 0.2, 8: 0.201}, min_marginal_gain=0.01) == 4


def test_deployment_gate_decisions():
    assert deployment_gate(
        GateInputs(
            diversity=5.0,
            collapse_rate=0.0,
            score_utility_correlation=0.9,
            tail_rank_correlation=0.8,
            high_n_real_change=-0.1,
            high_n_regret=0.1,
            latency_gain=0.1,
            k_marginal_gain=0.1,
        )
    ) == BLOCK_HIGH_N
    assert deployment_gate(
        GateInputs(5.0, 0.0, -0.2, -0.4, 0.1, 0.1, 0.1, 0.1)
    ) == CALIBRATE_RERANKER
    assert deployment_gate(
        GateInputs(1.0, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1)
    ) == INCREASE_DIVERSITY
    assert deployment_gate(
        GateInputs(5.0, 0.0, 0.8, 0.8, 0.1, 0.1, -0.1, 0.1)
    ) == STOP_EARLY
    assert deployment_gate(
        GateInputs(5.0, 0.0, 0.8, 0.8, 0.1, 0.1, 0.1, 0.0)
    ) == REDUCE_DENOISING_STEPS
    assert deployment_gate(
        GateInputs(5.0, 0.0, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1)
    ) == ALLOW_HIGH_N
