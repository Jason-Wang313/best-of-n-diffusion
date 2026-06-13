from __future__ import annotations

import numpy as np

from diffusion_audit.scorers import aligned_scores
from diffusion_audit.toy_control import make_observations, sample_diffusion_like_pool


def test_controlled_sampler_deterministic_with_fixed_seed():
    obs = make_observations(1, seed=10)[0]
    a = sample_diffusion_like_pool(
        obs,
        n_candidates=12,
        horizon=4,
        denoising_steps=6,
        temperature=0.7,
        diversity=0.5,
        mode_coverage_value=0.8,
        seed=123,
    )
    b = sample_diffusion_like_pool(
        obs,
        n_candidates=12,
        horizon=4,
        denoising_steps=6,
        temperature=0.7,
        diversity=0.5,
        mode_coverage_value=0.8,
        seed=123,
    )
    np.testing.assert_allclose(a.trajectories, b.trajectories)
    np.testing.assert_array_equal(a.mode_ids, b.mode_ids)


def test_aligned_scorer_deterministic_with_fixed_seed():
    obs = make_observations(1, seed=11)[0]
    pool = sample_diffusion_like_pool(
        obs,
        n_candidates=8,
        horizon=4,
        denoising_steps=8,
        temperature=0.6,
        diversity=0.4,
        mode_coverage_value=0.6,
        seed=5,
    )
    a = aligned_scores(obs, pool.trajectories, seed=99)
    b = aligned_scores(obs, pool.trajectories, seed=99)
    np.testing.assert_allclose(a, b)
