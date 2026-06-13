from __future__ import annotations

import numpy as np

from diffusion_audit.diffusion_lite import (
    IMAGE_SIZE,
    make_image_expert_dataset,
    render_observation_image,
)
from diffusion_audit.toy_control import make_observations


def test_render_observation_image_has_expected_shape_and_range():
    obs = make_observations(1, seed=123)[0]
    image = render_observation_image(obs, image_size=IMAGE_SIZE, visual_regime="id", seed=5)
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert image.dtype == np.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


def test_visual_ood_regimes_change_rendered_pixels():
    obs = make_observations(1, seed=124, ood="hidden_obstacle")[0]
    base = render_observation_image(obs, visual_regime="id", seed=7)
    hidden = render_observation_image(obs, visual_regime="hidden_obstacle", seed=7)
    noisy = render_observation_image(obs, visual_regime="observation_noise", seed=7)
    assert not np.allclose(base, hidden)
    assert not np.allclose(base, noisy)


def test_make_image_expert_dataset_pairs_images_and_actions():
    images, actions = make_image_expert_dataset(
        states=2,
        candidates_per_state=3,
        horizon=4,
        seed=10,
        image_size=16,
    )
    assert images.shape == (6, 3, 16, 16)
    assert actions.shape == (6, 4, 2)
