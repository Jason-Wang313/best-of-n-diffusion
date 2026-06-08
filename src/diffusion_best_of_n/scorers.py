"""Reranker and critic scores for diffusion action trajectory pools."""

from __future__ import annotations

import numpy as np

from diffusion_best_of_n.toy_control import ToyObservation, trajectory_utilities


def trajectory_features(obs: ToyObservation, trajectories: np.ndarray) -> np.ndarray:
    traj = np.asarray(trajectories, dtype=float)
    final_pos = obs.block[None, :] + np.sum(traj, axis=1) * (obs.friction / obs.mass)
    final_dist = np.linalg.norm(final_pos - obs.goal[None, :], axis=1)
    energy = np.mean(np.sum(traj * traj, axis=2), axis=1)
    first_norm = np.linalg.norm(traj[:, 0, :], axis=1)
    positions = obs.block[None, None, :] + np.cumsum(traj * (obs.friction / obs.mass), axis=1)
    obstacle_dist = np.min(np.linalg.norm(positions - obs.obstacle[None, None, :], axis=2), axis=1)
    smoothness = np.mean(np.sum(np.diff(traj, axis=1) ** 2, axis=2), axis=1) if traj.shape[1] > 1 else np.zeros(traj.shape[0])
    return np.column_stack(
        [
            -final_dist,
            -energy,
            first_norm,
            -obstacle_dist,
            -smoothness,
            np.ones(traj.shape[0]),
        ]
    )


def random_scores(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=int(n))


def oracle_scores(obs: ToyObservation, trajectories: np.ndarray) -> np.ndarray:
    return trajectory_utilities(obs, trajectories)


def aligned_scores(obs: ToyObservation, trajectories: np.ndarray, seed: int | None = None, noise: float = 0.015) -> np.ndarray:
    rng = np.random.default_rng(seed)
    utility = oracle_scores(obs, trajectories)
    return utility + rng.normal(scale=float(noise), size=utility.shape)


def weakly_aligned_scores(
    obs: ToyObservation,
    trajectories: np.ndarray,
    seed: int | None = None,
    noise: float = 0.12,
) -> np.ndarray:
    """A scorer with positive mean correlation but weaker tail reliability."""

    rng = np.random.default_rng(seed)
    utility = oracle_scores(obs, trajectories)
    features = trajectory_features(obs, trajectories)
    nuisance = 0.20 * features[:, 2] - 0.10 * features[:, 1]
    return 0.45 * utility + nuisance + rng.normal(scale=float(noise), size=utility.shape)


def anti_correlated_scores(
    obs: ToyObservation,
    trajectories: np.ndarray,
    seed: int | None = None,
    noise: float = 0.02,
) -> np.ndarray:
    """Negative control: selected score improves while real utility should fall."""

    rng = np.random.default_rng(seed)
    utility = oracle_scores(obs, trajectories)
    return -utility + rng.normal(scale=float(noise), size=utility.shape)


def diffusion_likelihood_proxy(obs: ToyObservation, trajectories: np.ndarray) -> np.ndarray:
    """A plausible internal score favoring smooth, low-energy denoised actions."""

    features = trajectory_features(obs, trajectories)
    return 0.58 * features[:, 0] + 0.25 * features[:, 1] + 0.17 * features[:, 4]


def misaligned_tail_scores(obs: ToyObservation, trajectories: np.ndarray, seed: int | None = None) -> np.ndarray:
    """A scorer that likes risky high-score tails near hidden obstacles."""

    rng = np.random.default_rng(seed)
    features = trajectory_features(obs, trajectories)
    return (
        0.05 * features[:, 0]
        + 2.35 * features[:, 3]
        + 0.70 * features[:, 2]
        - 0.28 * features[:, 1]
        + rng.normal(scale=0.02, size=features.shape[0])
    )


def tail_only_misaligned_scores(obs: ToyObservation, trajectories: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Score whose average correlation can look harmless while the top tail is bad."""

    rng = np.random.default_rng(seed)
    utility = oracle_scores(obs, trajectories)
    features = trajectory_features(obs, trajectories)
    risky = features[:, 3] + 0.35 * features[:, 2]
    cutoff = np.quantile(risky, 0.72)
    tail_bonus = np.maximum(risky - cutoff, 0.0) * 6.0
    return utility + tail_bonus + rng.normal(scale=0.018, size=utility.shape)


def behavior_cloning_critic(obs: ToyObservation, trajectories: np.ndarray) -> np.ndarray:
    """Score closeness to the direct expert mode."""

    features = trajectory_features(obs, trajectories)
    return 0.85 * features[:, 0] + 0.15 * features[:, 1]


def fit_linear_value_critic(features: np.ndarray, utilities: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    y = np.asarray(utilities, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be 2D and utilities must match rows")
    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1])
    return np.linalg.solve(xtx, x.T @ y)


def apply_linear_critic(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=float) @ np.asarray(weights, dtype=float)


def ensemble_value_critic(
    features: np.ndarray,
    utilities: np.ndarray,
    *,
    seed: int = 0,
    members: int = 5,
    pilot_fraction: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap linear critics; returns mean prediction and disagreement."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(utilities, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be 2D and utilities must match rows")
    rng = np.random.default_rng(seed)
    n_pilot = max(x.shape[1] + 1, int(np.ceil(x.shape[0] * float(pilot_fraction))))
    preds = []
    for _ in range(int(members)):
        idx = rng.choice(np.arange(x.shape[0]), size=n_pilot, replace=True)
        weights = fit_linear_value_critic(x[idx], y[idx], ridge=1e-3)
        preds.append(apply_linear_critic(x, weights))
    stack = np.vstack(preds)
    return np.mean(stack, axis=0), np.std(stack, axis=0)


def uncertainty_aware_critic(
    obs: ToyObservation,
    trajectories: np.ndarray,
    *,
    seed: int = 0,
    pilot_fraction: float = 0.35,
    uncertainty_penalty: float = 0.50,
) -> np.ndarray:
    """Value critic that penalizes ensemble disagreement and feature extrapolation."""

    features = trajectory_features(obs, trajectories)
    utilities = oracle_scores(obs, trajectories)
    mean_pred, disagreement = ensemble_value_critic(
        features,
        utilities,
        seed=seed,
        pilot_fraction=pilot_fraction,
    )
    rng = np.random.default_rng(seed + 11)
    pilot = rng.choice(np.arange(features.shape[0]), size=max(6, int(features.shape[0] * pilot_fraction)), replace=False)
    center = np.mean(features[pilot], axis=0)
    spread = np.std(features[pilot], axis=0) + 1e-6
    extrapolation = np.linalg.norm((features - center[None, :]) / spread[None, :], axis=1)
    return mean_pred - float(uncertainty_penalty) * disagreement - 0.015 * extrapolation


def calibrated_critic(obs: ToyObservation, trajectories: np.ndarray, pilot_fraction: float = 0.35) -> np.ndarray:
    """Small pilot-rollout calibrated critic for the same candidate pool."""

    features = trajectory_features(obs, trajectories)
    utilities = oracle_scores(obs, trajectories)
    n_pilot = max(4, int(np.ceil(features.shape[0] * float(pilot_fraction))))
    order = np.argsort(misaligned_tail_scores(obs, trajectories, seed=17), kind="mergesort")
    pilot_idx = np.unique(np.r_[order[: n_pilot // 2], order[-n_pilot:]])
    weights = fit_linear_value_critic(features[pilot_idx], utilities[pilot_idx])
    return apply_linear_critic(features, weights)
