from __future__ import annotations

import numpy as np

from diffusion_best_of_n.action_ddpm import (
    diffusion_internal_scores,
    sample_consistency_trajectories,
    sample_ddim_trajectories,
    sample_ddpm_trajectories,
    train_epsilon_denoiser,
)
from diffusion_best_of_n.diffusion_lite import make_expert_dataset


def test_action_ddpm_sampling_shapes_and_reproducibility():
    obs, actions = make_expert_dataset(states=3, candidates_per_state=3, horizon=4, seed=7)
    policy, result = train_epsilon_denoiser(
        obs,
        actions,
        epochs=2,
        seed=11,
        diffusion_steps=8,
        hidden=32,
        batch_size=16,
    )
    assert result.target == "epsilon"
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)

    first = sample_ddim_trajectories(policy, obs[0], n=5, k=4, seed=13)
    second = sample_ddim_trajectories(policy, obs[0], n=5, k=4, seed=13)
    assert first.shape == (5, 4, 2)
    assert np.allclose(first, second)

    ddpm = sample_ddpm_trajectories(policy, obs[0], n=5, k=4, seed=14)
    one_step = sample_consistency_trajectories(policy, obs[0], n=5, seed=15)
    assert ddpm.shape == first.shape
    assert one_step.shape == first.shape

    scores = diffusion_internal_scores(policy, obs[0], first, seed=16, probes=2)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
