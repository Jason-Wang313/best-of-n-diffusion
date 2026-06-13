from __future__ import annotations

import numpy as np
import pytest

from diffusion_audit.benchmarks.pusht import (
    evaluate_pusht_trajectory,
    pusht_behavior_cloning_score,
    pusht_mode_template,
    pusht_obs_to_features,
)


def test_pusht_features_templates_and_short_rollout():
    try:
        import gymnasium as gym
        import gym_pusht  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PushT dependency unavailable: {exc}")

    env = gym.make("gym_pusht/PushT-v0", render_mode=None)
    try:
        obs, _ = env.reset(seed=0)
    finally:
        env.close()

    features = pusht_obs_to_features(obs)
    assert features.shape == (6,)
    assert np.all(np.isfinite(features))

    trajectories = np.asarray([pusht_mode_template(obs, horizon=6, mode=mode) for mode in [0, 1, 2]])
    assert trajectories.shape == (3, 6, 2)
    assert trajectories.min() >= 0.0
    assert trajectories.max() <= 512.0

    scores = pusht_behavior_cloning_score(obs, trajectories)
    assert scores.shape == (3,)
    rollout = evaluate_pusht_trajectory(0, trajectories[0])
    assert rollout.steps > 0
    assert np.isfinite(rollout.utility)
