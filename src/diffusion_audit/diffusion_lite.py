"""Tiny learned Diffusion Policy-lite models for CPU tests and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from diffusion_audit.toy_control import ToyObservation, make_observations, mode_template, trajectory_utility


IMAGE_SIZE = 32
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class MLPDenoiser(nn.Module):
    def __init__(self, obs_dim: int, horizon: int, action_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        in_dim = obs_dim + self.horizon * self.action_dim + 1
        out_dim = self.horizon * self.action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, obs: torch.Tensor, noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat = noisy.reshape(noisy.shape[0], -1)
        x = torch.cat([obs, flat, t.reshape(-1, 1)], dim=1)
        return self.net(x).reshape(noisy.shape)


class TinyImageDenoiser(nn.Module):
    """A small CNN encoder followed by the same style of MLP denoising head."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        horizon: int,
        action_dim: int = 2,
        hidden: int = 64,
        embedding_dim: int = 32,
    ):
        super().__init__()
        channels, _, _ = image_shape
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(int(channels), 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(24, embedding_dim),
            nn.Tanh(),
        )
        in_dim = embedding_dim + self.horizon * self.action_dim + 1
        out_dim = self.horizon * self.action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def denoise_from_embedding(self, emb: torch.Tensor, noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat = noisy.reshape(noisy.shape[0], -1)
        x = torch.cat([emb, flat, t.reshape(-1, 1)], dim=1)
        return self.net(x).reshape(noisy.shape)

    def forward(self, obs: torch.Tensor, noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(obs)
        return self.denoise_from_embedding(emb, noisy, t)


@dataclass
class LiteTrainingResult:
    initial_loss: float
    final_loss: float
    epochs: int
    conditioning: str = "state"


def _draw_disk(image: np.ndarray, xy: np.ndarray, color: np.ndarray, radius: int, alpha: float = 1.0) -> None:
    h, w = image.shape[1:]
    px = int(round((float(xy[0]) + 1.4) / 2.8 * (w - 1)))
    py = int(round((1.4 - float(xy[1])) / 2.8 * (h - 1)))
    px = int(np.clip(px, 0, w - 1))
    py = int(np.clip(py, 0, h - 1))
    for y in range(max(0, py - radius), min(h, py + radius + 1)):
        for x in range(max(0, px - radius), min(w, px + radius + 1)):
            if (x - px) ** 2 + (y - py) ** 2 <= radius**2:
                image[:, y, x] = (1.0 - alpha) * image[:, y, x] + alpha * color


def render_observation_image(
    obs: ToyObservation,
    *,
    image_size: int = IMAGE_SIZE,
    visual_regime: str = "id",
    seed: int = 0,
    distractor_count: int = 4,
) -> np.ndarray:
    """Render a toy observation as a 32x32 RGB tensor in CHW format."""

    rng = np.random.default_rng(seed)
    size = int(image_size)
    image = np.full((3, size, size), 0.035, dtype=np.float32)
    colors = {
        "block": np.asarray([0.95, 0.20, 0.18], dtype=np.float32),
        "goal": np.asarray([0.20, 0.85, 0.26], dtype=np.float32),
        "obstacle": np.asarray([0.20, 0.38, 0.95], dtype=np.float32),
        "distractor": np.asarray([0.95, 0.82, 0.20], dtype=np.float32),
    }
    if visual_regime == "shifted_colors":
        colors = {
            "block": np.asarray([0.20, 0.85, 0.95], dtype=np.float32),
            "goal": np.asarray([0.92, 0.80, 0.18], dtype=np.float32),
            "obstacle": np.asarray([0.92, 0.25, 0.86], dtype=np.float32),
            "distractor": np.asarray([0.85, 0.85, 0.85], dtype=np.float32),
        }

    _draw_disk(image, obs.goal, colors["goal"], radius=max(2, size // 12), alpha=0.85)
    if visual_regime != "hidden_obstacle":
        _draw_disk(image, obs.obstacle, colors["obstacle"], radius=max(2, size // 13), alpha=0.78)
    _draw_disk(image, obs.block, colors["block"], radius=max(1, size // 15), alpha=0.95)

    if visual_regime == "distractors":
        for _ in range(int(distractor_count)):
            xy = rng.uniform(-1.15, 1.15, size=2)
            color = colors["distractor"] * rng.uniform(0.65, 1.0)
            _draw_disk(image, xy, color.astype(np.float32), radius=max(1, size // 18), alpha=0.65)

    if visual_regime == "observation_noise":
        image = image + rng.normal(scale=0.08, size=image.shape).astype(np.float32)

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def make_expert_dataset(
    states: int,
    candidates_per_state: int,
    horizon: int,
    seed: int,
    multimodal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic expert demonstrations as state/action sequences."""

    rng = np.random.default_rng(seed)
    obs_rows = []
    action_rows = []
    observations = make_observations(states, seed=seed, ood="id")
    for obs in observations:
        modes = [0, 1, 2] if multimodal else [0]
        for _ in range(int(candidates_per_state)):
            mode = int(rng.choice(modes))
            clean = mode_template(obs, horizon, mode)
            clean = clean + rng.normal(scale=0.015, size=clean.shape)
            obs_rows.append(obs.as_array())
            action_rows.append(clean)
    return np.asarray(obs_rows, dtype=np.float32), np.asarray(action_rows, dtype=np.float32)


def make_image_expert_dataset(
    states: int,
    candidates_per_state: int,
    horizon: int,
    seed: int,
    multimodal: bool = True,
    image_size: int = IMAGE_SIZE,
    visual_regime: str = "id",
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic expert demonstrations as image/action sequences."""

    rng = np.random.default_rng(seed)
    image_rows = []
    action_rows = []
    observations = make_observations(states, seed=seed, ood="id")
    for state_idx, obs in enumerate(observations):
        image = render_observation_image(
            obs,
            image_size=image_size,
            visual_regime=visual_regime,
            seed=seed * 1000 + state_idx,
        )
        modes = [0, 1, 2] if multimodal else [0]
        for _ in range(int(candidates_per_state)):
            mode = int(rng.choice(modes))
            clean = mode_template(obs, horizon, mode)
            clean = clean + rng.normal(scale=0.015, size=clean.shape)
            image_rows.append(image)
            action_rows.append(clean)
    return np.asarray(image_rows, dtype=np.float32), np.asarray(action_rows, dtype=np.float32)


def train_denoiser(
    obs: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    seed: int,
    lr: float = 2e-3,
    batch_size: int = 128,
) -> tuple[MLPDenoiser | TinyImageDenoiser, LiteTrainingResult]:
    """Train either a state-conditioned MLP or image-conditioned CNN denoiser."""

    torch.manual_seed(int(seed))
    obs_arr = np.asarray(obs, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32)
    if obs_arr.ndim == 2:
        model: MLPDenoiser | TinyImageDenoiser = MLPDenoiser(
            obs_dim=obs_arr.shape[1],
            horizon=actions_arr.shape[1],
            action_dim=actions_arr.shape[2],
        )
        conditioning = "state"
    elif obs_arr.ndim == 4:
        model = TinyImageDenoiser(
            image_shape=(obs_arr.shape[1], obs_arr.shape[2], obs_arr.shape[3]),
            horizon=actions_arr.shape[1],
            action_dim=actions_arr.shape[2],
        )
        conditioning = "image"
    else:
        raise ValueError("obs must be a 2D state matrix or 4D image tensor")

    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32)
    clean_t = torch.as_tensor(actions_arr, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    opt_step = 0

    def loss_once() -> torch.Tensor:
        t = torch.rand(clean_t.shape[0], 1)
        noise = torch.randn_like(clean_t)
        noisy = (1.0 - t.reshape(-1, 1, 1)) * clean_t + t.reshape(-1, 1, 1) * noise
        pred = model(obs_t, noisy, t)
        return torch.mean((pred - clean_t) ** 2)

    with torch.no_grad():
        initial = float(loss_once().detach().cpu().item())
    n = clean_t.shape[0]
    for _ in range(int(epochs)):
        order = rng.permutation(n)
        for start in range(0, n, int(batch_size)):
            idx = torch.as_tensor(order[start : start + int(batch_size)], dtype=torch.long)
            batch_obs = obs_t[idx]
            batch_clean = clean_t[idx]
            t = torch.rand(batch_clean.shape[0], 1)
            noise = torch.randn_like(batch_clean)
            noisy = (1.0 - t.reshape(-1, 1, 1)) * batch_clean + t.reshape(-1, 1, 1) * noise
            pred = model(batch_obs, noisy, t)
            loss = torch.mean((pred - batch_clean) ** 2)
            model.zero_grad(set_to_none=True)
            loss.backward()
            opt_step += 1
            with torch.no_grad():
                for i, p in enumerate(params):
                    if p.grad is None:
                        continue
                    grad = p.grad
                    m[i].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    v[i].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    m_hat = m[i] / (1.0 - beta1**opt_step)
                    v_hat = v[i] / (1.0 - beta2**opt_step)
                    p.addcdiv_(m_hat, torch.sqrt(v_hat).add_(eps), value=-float(lr))
    with torch.no_grad():
        final = float(loss_once().detach().cpu().item())
    return model, LiteTrainingResult(
        initial_loss=initial,
        final_loss=final,
        epochs=int(epochs),
        conditioning=conditioning,
    )


def sample_denoised_trajectories(
    model: MLPDenoiser,
    obs: ToyObservation,
    n: int,
    k: int,
    temperature: float,
    seed: int,
) -> np.ndarray:
    """Iteratively denoise noise into action trajectories conditioned on state."""

    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed) % (2**31 - 1))
    model.eval()
    horizon = model.horizon
    current = rng.normal(scale=float(temperature), size=(int(n), horizon, model.action_dim)).astype(np.float32)
    obs_batch = np.repeat(obs.as_array()[None, :].astype(np.float32), int(n), axis=0)
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
    with torch.no_grad():
        x = torch.as_tensor(current, dtype=torch.float32)
        for step in range(int(k), 0, -1):
            t_value = np.full((int(n), 1), step / max(int(k), 1), dtype=np.float32)
            t = torch.as_tensor(t_value, dtype=torch.float32)
            pred = model(obs_t, x, t)
            blend = 1.0 / float(step)
            x = (1.0 - blend) * x + blend * pred
            if step > 1:
                x = x + torch.randn_like(x) * (float(temperature) * 0.015 / np.sqrt(step))
        return x.detach().cpu().numpy().astype(float)


def sample_image_denoised_trajectories(
    model: TinyImageDenoiser,
    obs_image: np.ndarray,
    n: int,
    k: int,
    temperature: float,
    seed: int,
) -> np.ndarray:
    """Iteratively denoise action trajectories conditioned on a rendered image."""

    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed) % (2**31 - 1))
    model.eval()
    horizon = model.horizon
    current = rng.normal(scale=float(temperature), size=(int(n), horizon, model.action_dim)).astype(np.float32)
    image = np.asarray(obs_image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError("obs_image must have shape C,H,W")
    image_batch = np.repeat(image[None, :, :, :], int(n), axis=0)
    obs_t = torch.as_tensor(image_batch, dtype=torch.float32)
    with torch.no_grad():
        x = torch.as_tensor(current, dtype=torch.float32)
        emb = model.encoder(obs_t)
        for step in range(int(k), 0, -1):
            t_value = np.full((int(n), 1), step / max(int(k), 1), dtype=np.float32)
            t = torch.as_tensor(t_value, dtype=torch.float32)
            pred = model.denoise_from_embedding(emb, x, t)
            blend = 1.0 / float(step)
            x = (1.0 - blend) * x + blend * pred
            if step > 1:
                x = x + torch.randn_like(x) * (float(temperature) * 0.015 / np.sqrt(step))
        return x.detach().cpu().numpy().astype(float)


def _copy_with_block(obs: ToyObservation, block: np.ndarray) -> ToyObservation:
    return ToyObservation(
        block=np.asarray(block, dtype=float),
        goal=np.asarray(obs.goal, dtype=float),
        obstacle=np.asarray(obs.obstacle, dtype=float),
        friction=obs.friction,
        mass=obs.mass,
        action_noise=obs.action_noise,
        hidden_penalty=obs.hidden_penalty,
    )


def receding_horizon_utility(
    model: MLPDenoiser,
    obs: ToyObservation,
    n: int,
    k: int,
    scorer: Callable[[ToyObservation, np.ndarray], np.ndarray],
    horizon: int,
    rollout_steps: int,
    temperature: float,
    seed: int,
) -> float:
    """Simple receding-horizon execution: sample, rank, execute first action."""

    current = _copy_with_block(obs, obs.block)
    executed = []
    for step in range(int(rollout_steps)):
        trajs = sample_denoised_trajectories(
            model, current, n=n, k=k, temperature=temperature, seed=seed + 1009 * step
        )
        scores = scorer(current, trajs)
        chosen = trajs[int(np.argmax(scores))]
        action = chosen[0]
        executed.append(action)
        new_block = current.block + action * (current.friction / current.mass)
        current = _copy_with_block(current, new_block)
    full = np.asarray(executed, dtype=float)
    if full.shape[0] < horizon:
        pad = np.zeros((horizon - full.shape[0], 2), dtype=float)
        full = np.vstack([full, pad])
    return trajectory_utility(obs, full[:horizon])


def receding_horizon_utility_image(
    model: TinyImageDenoiser,
    obs: ToyObservation,
    n: int,
    k: int,
    scorer: Callable[[ToyObservation, np.ndarray], np.ndarray],
    horizon: int,
    rollout_steps: int,
    temperature: float,
    seed: int,
    image_size: int = IMAGE_SIZE,
    visual_regime: str = "id",
) -> float:
    """Receding-horizon execution with a freshly rendered image observation."""

    current = _copy_with_block(obs, obs.block)
    executed = []
    for step in range(int(rollout_steps)):
        image = render_observation_image(
            current,
            image_size=image_size,
            visual_regime=visual_regime,
            seed=seed + 701 * step,
        )
        trajs = sample_image_denoised_trajectories(
            model, image, n=n, k=k, temperature=temperature, seed=seed + 1009 * step
        )
        scores = scorer(current, trajs)
        chosen = trajs[int(np.argmax(scores))]
        action = chosen[0]
        executed.append(action)
        new_block = current.block + action * (current.friction / current.mass)
        current = _copy_with_block(current, new_block)
    full = np.asarray(executed, dtype=float)
    if full.shape[0] < horizon:
        pad = np.zeros((horizon - full.shape[0], 2), dtype=float)
        full = np.vstack([full, pad])
    return trajectory_utility(obs, full[:horizon])
