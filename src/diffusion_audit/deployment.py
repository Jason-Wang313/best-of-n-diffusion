"""Conservative deployment gates for high-N diffusion trajectory reranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ALLOW_HIGH_N = "allow_high_n"
STOP_EARLY = "stop_early"
INCREASE_DIVERSITY = "increase_diversity"
CALIBRATE_RERANKER = "calibrate_reranker"
REDUCE_DENOISING_STEPS = "reduce_denoising_steps"
BLOCK_HIGH_N = "block_high_n"


@dataclass(frozen=True)
class GateInputs:
    """Metrics used by the deployment gate."""

    diversity: float
    collapse_rate: float
    score_utility_correlation: float
    tail_rank_correlation: float
    high_n_real_change: float
    high_n_regret: float
    latency_gain: float
    k_marginal_gain: float


def deployment_gate(inputs: GateInputs) -> str:
    """Return one deployment action from the required gate vocabulary."""

    if inputs.high_n_real_change < -1e-9 or inputs.high_n_regret > 0.75:
        return BLOCK_HIGH_N
    if inputs.score_utility_correlation < 0.15 or inputs.tail_rank_correlation < 0.0:
        return CALIBRATE_RERANKER
    if inputs.collapse_rate > 0.65 or inputs.diversity < 1.5:
        return INCREASE_DIVERSITY
    if inputs.latency_gain < 0.0:
        return STOP_EARLY
    if inputs.k_marginal_gain < 0.005:
        return REDUCE_DENOISING_STEPS
    return ALLOW_HIGH_N


def gate_from_metrics(**kwargs) -> str:
    """Convenience wrapper that accepts the metric names as keyword args."""

    values = GateInputs(**kwargs)
    if not all(np.isfinite(float(v)) for v in values.__dict__.values()):
        return CALIBRATE_RERANKER
    return deployment_gate(values)
