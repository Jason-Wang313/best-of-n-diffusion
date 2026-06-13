"""Denoising-budget and latency-adjusted utility metrics."""

from __future__ import annotations

from collections.abc import Mapping


def total_budget(n: int, k: int) -> int:
    """Total denoising step budget B = N x K."""

    if int(n) < 1 or int(k) < 1:
        raise ValueError("n and k must be >= 1")
    return int(n) * int(k)


def utility_per_diffusion_step(utility: float, n: int, k: int) -> float:
    """Real utility divided by total diffusion steps."""

    return float(utility) / float(total_budget(n, k))


def latency_cost(n: int, k: int, cost_per_step: float = 1.0, overhead: float = 0.0) -> float:
    if cost_per_step < 0.0 or overhead < 0.0:
        raise ValueError("cost parameters must be non-negative")
    return float(overhead) + float(cost_per_step) * float(total_budget(n, k))


def latency_adjusted_utility(
    utility: float,
    n: int,
    k: int,
    lambda_cost: float,
    cost_per_step: float = 1.0,
    overhead: float = 0.0,
) -> float:
    """Utility minus lambda times diffusion latency cost."""

    if lambda_cost < 0.0:
        raise ValueError("lambda_cost must be non-negative")
    return float(utility) - float(lambda_cost) * latency_cost(n, k, cost_per_step, overhead)


def stop_rule_for_n(
    curve: Mapping[int, float],
    min_marginal_gain: float = 0.01,
) -> int:
    """Return the previous N once the next marginal gain is too small."""

    if not curve:
        raise ValueError("curve must be non-empty")
    ordered = sorted((int(n), float(v)) for n, v in curve.items())
    previous_n, previous_value = ordered[0]
    for n, value in ordered[1:]:
        if value - previous_value < float(min_marginal_gain):
            return int(previous_n)
        previous_n, previous_value = n, value
    return int(ordered[-1][0])


def stop_rule_for_k(
    curve: Mapping[int, float],
    min_marginal_gain: float = 0.01,
) -> int:
    """Return the previous K once the next marginal gain is too small."""

    return stop_rule_for_n(curve, min_marginal_gain=min_marginal_gain)
