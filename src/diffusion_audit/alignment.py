"""Score-utility alignment diagnostics for high-N diffusion reranking."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _as_pair(scores, utilities) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float)
    u = np.asarray(utilities, dtype=float)
    if s.ndim != 1 or u.ndim != 1:
        raise ValueError("scores and utilities must be one-dimensional")
    if s.size == 0:
        raise ValueError("scores and utilities must be non-empty")
    if s.shape != u.shape:
        raise ValueError("scores and utilities must have the same shape")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(u)):
        raise ValueError("scores and utilities must be finite")
    return s, u


def average_ranks(values) -> np.ndarray:
    """Average ranks with rank 1 assigned to the smallest value."""

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    sorted_arr = arr[order]
    i = 0
    while i < arr.size:
        j = i + 1
        while j < arr.size and sorted_arr[j] == sorted_arr[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def pearson_corr(x, y) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2:
        return float("nan")
    x_center = x_arr - float(np.mean(x_arr))
    y_center = y_arr - float(np.mean(y_arr))
    denom = float(np.linalg.norm(x_center) * np.linalg.norm(y_center))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x_center, y_center) / denom)


def score_utility_correlation(scores, utilities, method: str = "spearman") -> float:
    """Correlation between reranker score and real utility."""

    s, u = _as_pair(scores, utilities)
    if method == "pearson":
        return pearson_corr(s, u)
    if method == "spearman":
        return pearson_corr(average_ranks(s), average_ranks(u))
    raise ValueError("method must be 'spearman' or 'pearson'")


def top_score_tail_mask(scores, tail_fraction: float = 0.2, min_count: int = 2) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    k = min(s.size, max(int(np.ceil(s.size * float(tail_fraction))), int(min_count)))
    order = np.argsort(s, kind="mergesort")
    mask = np.zeros(s.size, dtype=bool)
    mask[order[-k:]] = True
    return mask


def top_score_tail_real_utility(scores, utilities, tail_fraction: float = 0.2) -> float:
    """Mean real utility among the top-score tail."""

    s, u = _as_pair(scores, utilities)
    return float(np.mean(u[top_score_tail_mask(s, tail_fraction=tail_fraction)]))


def tail_rank_correlation(scores, utilities, tail_fraction: float = 0.2) -> float:
    """Spearman correlation restricted to the high-score tail."""

    s, u = _as_pair(scores, utilities)
    mask = top_score_tail_mask(s, tail_fraction=tail_fraction)
    if int(np.sum(mask)) < 2:
        return float("nan")
    return score_utility_correlation(s[mask], u[mask], method="spearman")


def oracle_minus_reranker_gap(
    oracle_curve: Mapping[int, float],
    reranker_curve: Mapping[int, float],
) -> dict[int, float]:
    """Oracle selected utility minus reranker selected utility for shared N."""

    return {
        int(n): float(oracle_curve[n]) - float(reranker_curve[n])
        for n in sorted(set(oracle_curve).intersection(reranker_curve))
    }


def high_n_regret(
    oracle_curve: Mapping[int, float],
    reranker_curve: Mapping[int, float],
) -> float:
    """Oracle-minus-reranker gap at the largest shared N."""

    gaps = oracle_minus_reranker_gap(oracle_curve, reranker_curve)
    if not gaps:
        raise ValueError("curves must share at least one N value")
    return float(gaps[max(gaps)])


def high_n_real_change(curve: Mapping[int, float]) -> float:
    """Real utility at largest N minus real utility at smallest N."""

    if not curve:
        raise ValueError("curve must be non-empty")
    keys = sorted(curve)
    return float(curve[keys[-1]]) - float(curve[keys[0]])
