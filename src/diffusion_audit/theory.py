"""Finite maximum-score trajectory-selection laws for diffusion policy reranking.

The law here is intentionally agnostic to how candidates are produced. In this
repository the candidates are action trajectories sampled by a diffusion-like
policy, and the law is used to diagnose how a reranker score selects real
trajectory utility from a finite candidate pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TieGroup:
    """A group of equal scores in ascending-score order."""

    score: float
    start: int
    stop: int
    rank_min: int
    rank_max: int


def _as_1d_float(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_n_values(n_values: Iterable[int]) -> list[int]:
    out = [int(n) for n in n_values]
    if not out:
        raise ValueError("n_values must be non-empty")
    if any(n < 1 for n in out):
        raise ValueError("all N values must be >= 1")
    return out


def _sorted_tie_groups(scores: np.ndarray) -> tuple[np.ndarray, list[TieGroup]]:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    groups: list[TieGroup] = []
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        groups.append(
            TieGroup(
                score=float(sorted_scores[i]),
                start=i,
                stop=j,
                rank_min=i + 1,
                rank_max=j,
            )
        )
        i = j
    return order, groups


def utility_max_selection_finite(
    scores: Iterable[float] | np.ndarray,
    utilities: Iterable[float] | np.ndarray,
    n_values: Iterable[int],
) -> dict[int, float]:
    """Exact finite-pool expected utility under maximum-score selection.

    Candidates are sampled independently with replacement from the finite pool.
    The maximum-score sample is selected. If the selected maximum score is tied
    within the sampled set, the selected candidate is random within that top
    score group, so the group contributes its mean real utility.
    """

    score_arr = _as_1d_float(scores, "scores")
    utility_arr = _as_1d_float(utilities, "utilities")
    if score_arr.shape != utility_arr.shape:
        raise ValueError("scores and utilities must have the same shape")
    ns = _as_n_values(n_values)

    pool_size = score_arr.size
    order, groups = _sorted_tie_groups(score_arr)
    sorted_utilities = utility_arr[order]
    out: dict[int, float] = {}
    for n in ns:
        value = 0.0
        for group in groups:
            selected_mass = (group.rank_max / pool_size) ** n - (
                (group.rank_min - 1) / pool_size
            ) ** n
            group_mean = float(np.mean(sorted_utilities[group.start : group.stop]))
            value += group_mean * selected_mass
        out[int(n)] = float(value)
    return out


def binary_max_selection_finite(
    scores: Iterable[float] | np.ndarray,
    success: Iterable[bool | int | float] | np.ndarray,
    n_values: Iterable[int],
) -> dict[int, float]:
    """Exact finite-pool selected success probability for binary tasks."""

    success_arr = _as_1d_float(success, "success")
    if not np.all((success_arr == 0.0) | (success_arr == 1.0)):
        raise ValueError("success must contain only 0/1 values")
    return utility_max_selection_finite(scores, success_arr, n_values)


def selected_score_max_selection_finite(
    scores: Iterable[float] | np.ndarray,
    n_values: Iterable[int],
) -> dict[int, float]:
    """Exact expected selected score under maximum-score selection."""

    score_arr = _as_1d_float(scores, "scores")
    return utility_max_selection_finite(score_arr, score_arr, n_values)


def simulate_max_selection(
    scores: Iterable[float] | np.ndarray,
    utilities: Iterable[float] | np.ndarray,
    n: int,
    trials: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Monte Carlo selected utilities with random tie handling."""

    score_arr = _as_1d_float(scores, "scores")
    utility_arr = _as_1d_float(utilities, "utilities")
    if score_arr.shape != utility_arr.shape:
        raise ValueError("scores and utilities must have the same shape")
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    if int(trials) < 1:
        raise ValueError("trials must be >= 1")

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, score_arr.size, size=(int(trials), int(n)))
    sampled_scores = score_arr[sample_idx]
    max_scores = np.max(sampled_scores, axis=1)
    selected_idx = np.empty(int(trials), dtype=int)
    for row in range(int(trials)):
        tied_positions = np.flatnonzero(sampled_scores[row] == max_scores[row])
        chosen_position = int(rng.choice(tied_positions))
        selected_idx[row] = sample_idx[row, chosen_position]
    return utility_arr[selected_idx]


def exact_law_prediction_error(
    exact: dict[int, float],
    estimates: dict[int, float],
) -> dict[str, float]:
    """Mean and maximum absolute error between two curves."""

    keys = sorted(set(exact).intersection(estimates))
    if not keys:
        raise ValueError("curves must share at least one N value")
    errs = np.asarray([abs(float(exact[k]) - float(estimates[k])) for k in keys])
    return {"mae": float(np.mean(errs)), "max_abs": float(np.max(errs))}
