"""PushT simulator wrappers for CPU-light Best-of-N benchmark evidence."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


PUSHT_ENV_ID = "gym_pusht/PushT-v0"
PUSHT_ACTION_LOW = 0.0
PUSHT_ACTION_HIGH = 512.0


@dataclass(frozen=True)
class PushTRollout:
    utility: float
    max_coverage: float
    final_coverage: float
    success: bool
    steps: int
    runtime_seconds: float


def _require_pusht():
    try:
        import gymnasium as gym
        import gym_pusht  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only when dependency is missing
        raise RuntimeError(
            "PushT benchmark requires gym-pusht and gymnasium. Install requirements.txt; "
            "pymunk must be <7 for gym-pusht 0.1.6."
        ) from exc
    return gym


def pusht_obs_to_features(obs: np.ndarray) -> np.ndarray:
    """Normalize PushT low-dimensional observation to diffusion conditioning features."""

    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape[0] < 5:
        raise ValueError("PushT observation must have at least 5 entries")
    theta = float(arr[4])
    return np.asarray(
        [
            arr[0] / 512.0,
            arr[1] / 512.0,
            arr[2] / 512.0,
            arr[3] / 512.0,
            np.sin(theta),
            np.cos(theta),
        ],
        dtype=np.float32,
    )


def _interp(points: list[np.ndarray], horizon: int) -> np.ndarray:
    horizon = int(horizon)
    if len(points) < 2:
        raise ValueError("at least two points are required")
    segments = len(points) - 1
    weights = np.linspace(0.0, float(segments), horizon)
    actions = np.empty((horizon, 2), dtype=float)
    for i, value in enumerate(weights):
        seg = min(int(np.floor(value)), segments - 1)
        frac = value - seg
        actions[i] = (1.0 - frac) * points[seg] + frac * points[seg + 1]
    return np.clip(actions, PUSHT_ACTION_LOW, PUSHT_ACTION_HIGH)


def pusht_mode_template(obs: np.ndarray, horizon: int, mode: int, goal_xy: np.ndarray | None = None) -> np.ndarray:
    """Heuristic absolute-action trajectory modes for PushT."""

    arr = np.asarray(obs, dtype=float)
    agent = arr[:2]
    block = arr[2:4]
    goal = np.asarray(goal_xy if goal_xy is not None else [256.0, 256.0], dtype=float)
    to_goal = goal - block
    norm = np.linalg.norm(to_goal)
    normal = np.asarray([-to_goal[1], to_goal[0]], dtype=float) / max(norm, 1e-6)
    approach = block - 0.22 * to_goal
    if mode == 0:
        points = [agent, approach, block, goal]
    elif mode == 1:
        points = [agent, approach + 60.0 * normal, block + 52.0 * normal, goal]
    elif mode == 2:
        points = [agent, approach - 60.0 * normal, block - 52.0 * normal, goal]
    elif mode == 3:
        overshoot = np.clip(goal + 0.45 * to_goal, PUSHT_ACTION_LOW, PUSHT_ACTION_HIGH)
        points = [agent, block, overshoot, goal]
    elif mode == 4:
        corner = np.asarray([470.0, 42.0], dtype=float)
        points = [agent, corner, corner + np.asarray([-40.0, 80.0]), corner]
    else:
        raise ValueError("mode must be between 0 and 4")
    return _interp([np.asarray(p, dtype=float) for p in points], horizon).astype(float)


def make_pusht_expert_dataset(
    *,
    states: int,
    candidates_per_state: int,
    horizon: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate heuristic PushT demonstrations from reset observations."""

    gym = _require_pusht()
    env = gym.make(PUSHT_ENV_ID, render_mode=None)
    rng = np.random.default_rng(seed)
    obs_rows = []
    action_rows = []
    try:
        for state_idx in range(int(states)):
            obs, info = env.reset(seed=int(seed) * 1000 + state_idx)
            goal_xy = np.asarray(info.get("goal_pose", [256.0, 256.0])[:2], dtype=float)
            for _ in range(int(candidates_per_state)):
                mode = int(rng.choice([0, 1, 2, 3], p=[0.42, 0.24, 0.24, 0.10]))
                actions = pusht_mode_template(obs, horizon, mode, goal_xy=goal_xy)
                actions = actions + rng.normal(scale=5.0, size=actions.shape)
                obs_rows.append(pusht_obs_to_features(obs))
                action_rows.append(np.clip(actions, PUSHT_ACTION_LOW, PUSHT_ACTION_HIGH))
    finally:
        env.close()
    return np.asarray(obs_rows, dtype=np.float32), np.asarray(action_rows, dtype=np.float32)


def evaluate_pusht_trajectory(seed: int, trajectory: np.ndarray) -> PushTRollout:
    """Execute one absolute-action trajectory in PushT and return a scalar utility."""

    gym = _require_pusht()
    env = gym.make(PUSHT_ENV_ID, render_mode=None)
    traj = np.asarray(trajectory, dtype=np.float32)
    start = time.perf_counter()
    max_coverage = 0.0
    final_coverage = 0.0
    success = False
    steps = 0
    try:
        env.reset(seed=int(seed))
        for action in traj:
            _, _, terminated, truncated, info = env.step(np.clip(action, PUSHT_ACTION_LOW, PUSHT_ACTION_HIGH))
            final_coverage = float(info.get("coverage", 0.0))
            max_coverage = max(max_coverage, final_coverage)
            success = success or bool(info.get("is_success", False))
            steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    smoothness = float(np.mean(np.sum(np.diff(traj, axis=0) ** 2, axis=1))) if traj.shape[0] > 1 else 0.0
    utility = max_coverage + (0.35 if success else 0.0) - 0.000002 * smoothness
    return PushTRollout(
        utility=float(utility),
        max_coverage=float(max_coverage),
        final_coverage=float(final_coverage),
        success=bool(success),
        steps=int(steps),
        runtime_seconds=float(time.perf_counter() - start),
    )


def evaluate_pusht_pool(seed: int, trajectories: np.ndarray) -> tuple[np.ndarray, list[PushTRollout]]:
    """Execute candidate trajectories from the same PushT reset seed."""

    gym = _require_pusht()
    env = gym.make(PUSHT_ENV_ID, render_mode=None)
    rollouts: list[PushTRollout] = []
    try:
        for trajectory in np.asarray(trajectories, dtype=float):
            traj = np.asarray(trajectory, dtype=np.float32)
            start = time.perf_counter()
            max_coverage = 0.0
            final_coverage = 0.0
            success = False
            steps = 0
            env.reset(seed=int(seed))
            for action in traj:
                _, _, terminated, truncated, info = env.step(np.clip(action, PUSHT_ACTION_LOW, PUSHT_ACTION_HIGH))
                final_coverage = float(info.get("coverage", 0.0))
                max_coverage = max(max_coverage, final_coverage)
                success = success or bool(info.get("is_success", False))
                steps += 1
                if terminated or truncated:
                    break
            smoothness = float(np.mean(np.sum(np.diff(traj, axis=0) ** 2, axis=1))) if traj.shape[0] > 1 else 0.0
            utility = max_coverage + (0.35 if success else 0.0) - 0.000002 * smoothness
            rollouts.append(
                PushTRollout(
                    utility=float(utility),
                    max_coverage=float(max_coverage),
                    final_coverage=float(final_coverage),
                    success=bool(success),
                    steps=int(steps),
                    runtime_seconds=float(time.perf_counter() - start),
                )
            )
    finally:
        env.close()
    return np.asarray([item.utility for item in rollouts], dtype=float), rollouts


def pusht_trajectory_features(obs: np.ndarray, trajectories: np.ndarray) -> np.ndarray:
    """Feature map for lightweight PushT rerankers."""

    arr = np.asarray(obs, dtype=float)
    traj = np.asarray(trajectories, dtype=float)
    block = arr[2:4]
    goal = np.asarray([256.0, 256.0], dtype=float)
    first_dist_block = np.linalg.norm(traj[:, 0, :] - block[None, :], axis=1)
    final_dist_goal = np.linalg.norm(traj[:, -1, :] - goal[None, :], axis=1)
    motion = np.mean(np.linalg.norm(np.diff(traj, axis=1), axis=2), axis=1) if traj.shape[1] > 1 else np.zeros(traj.shape[0])
    smoothness = np.mean(np.sum(np.diff(traj, axis=1) ** 2, axis=2), axis=1) if traj.shape[1] > 1 else np.zeros(traj.shape[0])
    corner_dist = np.linalg.norm(traj[:, -1, :] - np.asarray([470.0, 42.0])[None, :], axis=1)
    return np.column_stack(
        [
            -first_dist_block / 512.0,
            -final_dist_goal / 512.0,
            -motion / 512.0,
            -smoothness / (512.0**2),
            -corner_dist / 512.0,
            np.ones(traj.shape[0]),
        ]
    )


def pusht_behavior_cloning_score(obs: np.ndarray, trajectories: np.ndarray) -> np.ndarray:
    """Score closeness to the best of the simple PushT demonstration modes."""

    traj = np.asarray(trajectories, dtype=float)
    templates = np.asarray([pusht_mode_template(obs, traj.shape[1], mode) for mode in [0, 1, 2]], dtype=float)
    dists = np.mean(np.sum((traj[:, None, :, :] - templates[None, :, :, :]) ** 2, axis=3), axis=2)
    return -np.min(dists, axis=1) / (512.0**2)


def pusht_misaligned_corner_score(trajectories: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Pathological PushT scorer that prefers visually dramatic corner-reaching actions."""

    rng = np.random.default_rng(seed)
    traj = np.asarray(trajectories, dtype=float)
    corner = np.asarray([470.0, 42.0], dtype=float)
    corner_bonus = -np.linalg.norm(traj[:, -1, :] - corner[None, :], axis=1) / 512.0
    motion = np.mean(np.linalg.norm(np.diff(traj, axis=1), axis=2), axis=1) / 512.0 if traj.shape[1] > 1 else 0.0
    return 1.4 * corner_bonus + 0.35 * motion + rng.normal(scale=0.01, size=traj.shape[0])
