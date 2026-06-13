"""CPU-light 2D manipulation task and diffusion-like trajectory sampler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyObservation:
    block: np.ndarray
    goal: np.ndarray
    obstacle: np.ndarray
    friction: float = 1.0
    mass: float = 1.0
    action_noise: float = 0.0
    hidden_penalty: bool = False

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.block[0],
                self.block[1],
                self.goal[0],
                self.goal[1],
                self.obstacle[0],
                self.obstacle[1],
                self.friction,
                self.mass,
                self.action_noise,
                float(self.hidden_penalty),
            ],
            dtype=float,
        )


@dataclass
class CandidatePool:
    observation: ToyObservation
    trajectories: np.ndarray
    mode_ids: np.ndarray
    denoising_steps: int
    temperature: float
    metadata: dict[str, float | str | bool]


MODE_NAMES = {
    0: "direct",
    1: "arc_up",
    2: "arc_down",
    3: "overshoot",
    4: "obstacle_grazing",
}


def make_observations(count: int, seed: int, ood: str = "id") -> list[ToyObservation]:
    """Generate deterministic toy manipulation observations."""

    rng = np.random.default_rng(seed)
    observations: list[ToyObservation] = []
    for _ in range(int(count)):
        block = rng.uniform(-0.8, 0.8, size=2)
        goal_shift = rng.uniform(-0.9, 0.9, size=2)
        if np.linalg.norm(goal_shift) < 0.35:
            goal_shift = goal_shift + np.array([0.45, -0.25])
        goal = np.clip(block + goal_shift, -1.1, 1.1)
        obstacle = 0.5 * (block + goal) + rng.normal(scale=0.08, size=2)
        friction = 1.0
        mass = 1.0
        action_noise = 0.0
        hidden_penalty = False
        if ood == "changed_friction":
            friction = 0.68
        elif ood == "changed_mass":
            mass = 1.45
        elif ood == "action_noise":
            action_noise = 0.035
        elif ood == "hidden_obstacle":
            hidden_penalty = True
        elif ood == "shifted_goal":
            goal = np.clip(goal + np.array([0.45, -0.35]), -1.3, 1.3)
        observations.append(
            ToyObservation(
                block=np.asarray(block, dtype=float),
                goal=np.asarray(goal, dtype=float),
                obstacle=np.asarray(obstacle, dtype=float),
                friction=friction,
                mass=mass,
                action_noise=action_noise,
                hidden_penalty=hidden_penalty,
            )
        )
    return observations


def mode_template(obs: ToyObservation, horizon: int, mode: int) -> np.ndarray:
    """Clean action sequence for a named mode."""

    delta = (obs.goal - obs.block) / max(int(horizon), 1)
    actions = np.tile(delta, (int(horizon), 1))
    progress = np.linspace(-1.0, 1.0, int(horizon))[:, None]
    normal = np.asarray([-delta[1], delta[0]], dtype=float)
    norm = np.linalg.norm(normal)
    if norm > 1e-12:
        normal = normal / norm
    if mode == 1:
        actions = actions + 0.16 * normal * np.sin(np.linspace(0.0, np.pi, int(horizon)))[:, None]
    elif mode == 2:
        actions = actions - 0.16 * normal * np.sin(np.linspace(0.0, np.pi, int(horizon)))[:, None]
    elif mode == 3:
        actions = 1.25 * actions - 0.08 * progress * delta
    elif mode == 4:
        # A deceptively smooth mode that tends to pass through the obstacle.
        toward_obstacle = (obs.obstacle - obs.block) / max(int(horizon), 1)
        actions = 0.62 * actions + 0.38 * np.tile(toward_obstacle, (int(horizon), 1))
    return actions.astype(float)


def rollout_final_position(obs: ToyObservation, trajectory: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Execute an action trajectory in the toy dynamics."""

    rng = np.random.default_rng(seed)
    action_arr = np.asarray(trajectory, dtype=float)
    gain = obs.friction / obs.mass
    if obs.action_noise > 0.0:
        action_arr = action_arr + rng.normal(scale=obs.action_noise, size=action_arr.shape)
    return obs.block + gain * np.sum(action_arr, axis=0)


def trajectory_utility(obs: ToyObservation, trajectory: np.ndarray) -> float:
    """Real utility after executing the trajectory."""

    traj = np.asarray(trajectory, dtype=float)
    positions = obs.block + np.cumsum(traj * (obs.friction / obs.mass), axis=0)
    final_dist = float(np.linalg.norm(positions[-1] - obs.goal))
    action_energy = float(np.mean(np.sum(traj * traj, axis=1)))
    obstacle_dist = np.min(np.linalg.norm(positions - obs.obstacle, axis=1))
    threshold = 0.34 if obs.hidden_penalty else 0.25
    obstacle_penalty = max(0.0, threshold - float(obstacle_dist)) * 3.8
    smoothness = float(np.mean(np.sum(np.diff(traj, axis=0) ** 2, axis=1))) if traj.shape[0] > 1 else 0.0
    return -final_dist - 0.08 * action_energy - obstacle_penalty - 0.03 * smoothness


def trajectory_utilities(obs: ToyObservation, trajectories: np.ndarray) -> np.ndarray:
    return np.asarray([trajectory_utility(obs, t) for t in trajectories], dtype=float)


def sample_mode_ids(
    rng: np.random.Generator,
    n: int,
    diversity: float,
    mode_coverage: float,
    collapsed: bool,
    biased_bad_mode: bool,
) -> np.ndarray:
    if collapsed:
        return np.zeros(int(n), dtype=int)
    available = max(1, min(len(MODE_NAMES), int(np.ceil(float(mode_coverage) * len(MODE_NAMES)))))
    probs = np.ones(available, dtype=float)
    if biased_bad_mode and available >= 5:
        probs[:] = 0.12
        probs[4] = 0.52
    elif available >= 4:
        probs[0] = 0.40 + 0.20 * (1.0 - float(diversity))
        probs[1:] = (1.0 - probs[0]) / (available - 1)
    probs = probs / np.sum(probs)
    return rng.choice(np.arange(available), size=int(n), p=probs).astype(int)


def sample_diffusion_like_pool(
    obs: ToyObservation,
    n_candidates: int,
    horizon: int,
    denoising_steps: int,
    temperature: float,
    diversity: float,
    mode_coverage_value: float,
    seed: int,
    collapsed: bool = False,
    biased_bad_mode: bool = False,
    low_k_noise: bool = False,
) -> CandidatePool:
    """Generate trajectories via a diffusion-like noise-to-action process."""

    rng = np.random.default_rng(seed)
    n_candidates = int(n_candidates)
    horizon = int(horizon)
    denoising_steps = int(denoising_steps)
    if n_candidates < 1 or horizon < 1 or denoising_steps < 1:
        raise ValueError("n_candidates, horizon, and denoising_steps must be >= 1")

    modes = sample_mode_ids(
        rng,
        n_candidates,
        diversity=diversity,
        mode_coverage=mode_coverage_value,
        collapsed=collapsed,
        biased_bad_mode=biased_bad_mode,
    )
    alpha = 1.0 - np.exp(-denoising_steps / 7.0)
    if low_k_noise:
        alpha *= 0.70
    base_noise = float(temperature) * (0.24 + 0.18 * float(diversity))
    residual_noise = base_noise / np.sqrt(denoising_steps)
    trajectories = np.empty((n_candidates, horizon, 2), dtype=float)
    if collapsed and float(diversity) <= 0.015 and float(temperature) <= 0.06:
        clean = mode_template(obs, horizon, 0)
        trajectories[:] = clean[None, :, :]
        modes[:] = 0
        return CandidatePool(
            observation=obs,
            trajectories=trajectories,
            mode_ids=modes,
            denoising_steps=denoising_steps,
            temperature=float(temperature),
            metadata={
                "diversity": float(diversity),
                "mode_coverage": float(mode_coverage_value),
                "collapsed": bool(collapsed),
                "biased_bad_mode": bool(biased_bad_mode),
                "low_k_noise": bool(low_k_noise),
            },
        )
    shared_noise = rng.normal(scale=residual_noise * 0.08, size=(horizon, 2)) if collapsed else 0.0
    for i, mode in enumerate(modes):
        clean = mode_template(obs, horizon, int(mode))
        initial = rng.normal(scale=base_noise, size=(horizon, 2))
        denoised = alpha * clean + (1.0 - alpha) * initial
        innovation = rng.normal(scale=residual_noise * max(float(diversity), 0.03), size=(horizon, 2))
        if collapsed:
            innovation = 0.15 * innovation + shared_noise
        trajectories[i] = denoised + innovation
    return CandidatePool(
        observation=obs,
        trajectories=trajectories,
        mode_ids=modes,
        denoising_steps=denoising_steps,
        temperature=float(temperature),
        metadata={
            "diversity": float(diversity),
            "mode_coverage": float(mode_coverage_value),
            "collapsed": bool(collapsed),
            "biased_bad_mode": bool(biased_bad_mode),
            "low_k_noise": bool(low_k_noise),
        },
    )
