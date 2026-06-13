"""Diversity diagnostics for sampled action trajectory pools."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _as_trajectories(trajectories) -> np.ndarray:
    arr = np.asarray(trajectories, dtype=float)
    if arr.ndim < 2:
        raise ValueError("trajectories must have candidate and feature dimensions")
    if arr.shape[0] == 0:
        raise ValueError("trajectory pool must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("trajectories must be finite")
    return arr


def pairwise_action_trajectory_distance(trajectories, metric: str = "l2") -> np.ndarray:
    """Pairwise distances between flattened action trajectories."""

    arr = _as_trajectories(trajectories)
    flat = arr.reshape(arr.shape[0], -1)
    diff = flat[:, None, :] - flat[None, :, :]
    if metric == "l2":
        return np.sqrt(np.sum(diff * diff, axis=-1))
    if metric == "l1":
        return np.sum(np.abs(diff), axis=-1)
    raise ValueError("metric must be 'l2' or 'l1'")


def mean_pairwise_distance(trajectories) -> float:
    distances = pairwise_action_trajectory_distance(trajectories)
    n = distances.shape[0]
    if n < 2:
        return 0.0
    return float(np.mean(distances[np.triu_indices(n, k=1)]))


def effective_sample_diversity(trajectories, sigma: float | None = None) -> float:
    """Kernel effective sample count, equal to 1 for exact collapse."""

    distances = pairwise_action_trajectory_distance(trajectories)
    n = distances.shape[0]
    if n == 1:
        return 1.0
    nonzero = distances[distances > 0]
    if sigma is None:
        sigma = float(np.median(nonzero)) if nonzero.size else 1.0
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma must be positive and finite")
    sim = np.exp(-(distances**2) / (2.0 * sigma**2))
    return float((n * n) / np.sum(sim))


def mode_coverage(mode_ids: Iterable[int], expected_modes: int | Iterable[int]) -> float:
    """Fraction of expected action modes represented in the candidate pool."""

    observed = {int(x) for x in mode_ids}
    if isinstance(expected_modes, int):
        expected = set(range(int(expected_modes)))
    else:
        expected = {int(x) for x in expected_modes}
    if not expected:
        raise ValueError("expected_modes must be non-empty")
    return float(len(observed.intersection(expected)) / len(expected))


def duplicate_collapse_rate(trajectories, tolerance: float = 1e-6) -> float:
    """Fraction of candidates that are duplicates within a distance tolerance."""

    arr = _as_trajectories(trajectories)
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    unique: list[np.ndarray] = []
    for traj in arr:
        if not any(float(np.linalg.norm(traj - item)) <= tolerance for item in unique):
            unique.append(traj)
    return float(1.0 - len(unique) / arr.shape[0])


def marginal_diversity_gain(trajectories, n_values: Iterable[int]) -> dict[int, float]:
    """Incremental mean-pairwise-distance gain for growing prefix pools."""

    arr = _as_trajectories(trajectories)
    out: dict[int, float] = {}
    prev = None
    for n in sorted(int(v) for v in n_values):
        if n < 1 or n > arr.shape[0]:
            raise ValueError("all N values must be between 1 and the pool size")
        current = mean_pairwise_distance(arr[:n])
        out[n] = 0.0 if prev is None else float(current - prev)
        prev = current
    return out


def trajectory_cluster_ids(
    trajectories,
    distance_threshold: float | None = None,
    max_clusters: int = 8,
) -> np.ndarray:
    """Greedy trajectory clusters for mode coverage diagnostics."""

    arr = _as_trajectories(trajectories)
    flat = arr.reshape(arr.shape[0], -1)
    if int(max_clusters) < 1:
        raise ValueError("max_clusters must be >= 1")
    if distance_threshold is None:
        distances = pairwise_action_trajectory_distance(arr)
        nonzero = distances[distances > 0]
        distance_threshold = float(np.quantile(nonzero, 0.35)) if nonzero.size else 1e-6
    if not np.isfinite(distance_threshold) or float(distance_threshold) < 0.0:
        raise ValueError("distance_threshold must be non-negative and finite")

    centers: list[np.ndarray] = []
    labels = np.empty(arr.shape[0], dtype=int)
    for i, row in enumerate(flat):
        if not centers:
            centers.append(row.copy())
            labels[i] = 0
            continue
        dists = np.asarray([np.linalg.norm(row - center) for center in centers], dtype=float)
        nearest = int(np.argmin(dists))
        if dists[nearest] > float(distance_threshold) and len(centers) < int(max_clusters):
            centers.append(row.copy())
            labels[i] = len(centers) - 1
        else:
            labels[i] = nearest
    return labels


def cluster_entropy(labels: Iterable[int]) -> float:
    """Normalized entropy over discovered trajectory clusters."""

    arr = np.asarray(list(labels), dtype=int)
    if arr.size == 0:
        raise ValueError("labels must be non-empty")
    _, counts = np.unique(arr, return_counts=True)
    probs = counts.astype(float) / float(np.sum(counts))
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    denom = np.log(max(len(counts), 2))
    return float(entropy / denom)


def marginal_new_mode_discovery(labels: Iterable[int], n_values: Iterable[int]) -> dict[int, float]:
    """Number of newly discovered trajectory clusters in each larger prefix."""

    arr = np.asarray(list(labels), dtype=int)
    if arr.size == 0:
        raise ValueError("labels must be non-empty")
    out: dict[int, float] = {}
    prev_seen: set[int] = set()
    for n in sorted(int(v) for v in n_values):
        if n < 1 or n > arr.size:
            raise ValueError("all N values must be between 1 and the number of labels")
        seen = {int(x) for x in arr[:n]}
        out[n] = float(len(seen - prev_seen))
        prev_seen = seen
    return out


def diversity_summary(trajectories, mode_ids=None, expected_modes: int | Iterable[int] | None = None) -> dict[str, float]:
    """Compact diversity summary for one candidate pool."""

    labels = trajectory_cluster_ids(trajectories)
    summary = {
        "mean_pairwise_distance": mean_pairwise_distance(trajectories),
        "effective_sample_diversity": effective_sample_diversity(trajectories),
        "duplicate_collapse_rate": duplicate_collapse_rate(trajectories),
        "trajectory_cluster_count": float(len(set(int(x) for x in labels))),
        "trajectory_cluster_entropy": cluster_entropy(labels),
    }
    if mode_ids is not None and expected_modes is not None:
        summary["mode_coverage"] = mode_coverage(mode_ids, expected_modes)
    return summary
