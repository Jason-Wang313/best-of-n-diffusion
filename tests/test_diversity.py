from __future__ import annotations

import numpy as np
import pytest

from diffusion_best_of_n.diversity import (
    cluster_entropy,
    duplicate_collapse_rate,
    effective_sample_diversity,
    marginal_diversity_gain,
    marginal_new_mode_discovery,
    mean_pairwise_distance,
    mode_coverage,
    pairwise_action_trajectory_distance,
    trajectory_cluster_ids,
)


def test_pairwise_action_trajectory_distance_correctness():
    trajectories = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[3.0, 4.0], [0.0, 0.0]],
        ]
    )
    distances = pairwise_action_trajectory_distance(trajectories)
    assert distances.shape == (2, 2)
    assert distances[0, 1] == pytest.approx(5.0)
    assert mean_pairwise_distance(trajectories) == pytest.approx(5.0)


def test_collapse_detection_and_effective_diversity():
    collapsed = np.zeros((5, 3, 2), dtype=float)
    assert duplicate_collapse_rate(collapsed) == pytest.approx(0.8)
    assert effective_sample_diversity(collapsed) == pytest.approx(1.0)


def test_mode_coverage():
    assert mode_coverage([0, 2, 2], expected_modes=4) == pytest.approx(0.5)
    assert mode_coverage([10, 20], expected_modes=[10, 20, 30]) == pytest.approx(2.0 / 3.0)


def test_marginal_diversity_gain_detects_new_spread():
    trajectories = np.array(
        [
            [[0.0, 0.0]],
            [[0.1, 0.0]],
            [[3.0, 0.0]],
            [[6.0, 0.0]],
        ]
    )
    gains = marginal_diversity_gain(trajectories, [1, 2, 4])
    assert gains[1] == 0.0
    assert gains[4] > gains[2]


def test_cluster_entropy_and_new_mode_discovery():
    trajectories = np.array(
        [
            [[0.0, 0.0]],
            [[0.05, 0.0]],
            [[3.0, 0.0]],
            [[3.1, 0.0]],
            [[7.0, 0.0]],
        ]
    )
    labels = trajectory_cluster_ids(trajectories, distance_threshold=0.2, max_clusters=4)
    assert len(set(labels.tolist())) == 3
    assert cluster_entropy(labels) > 0.8
    discovery = marginal_new_mode_discovery(labels, [1, 2, 3, 5])
    assert discovery[1] == pytest.approx(1.0)
    assert discovery[2] == pytest.approx(0.0)
    assert discovery[3] == pytest.approx(1.0)
    assert discovery[5] == pytest.approx(1.0)
