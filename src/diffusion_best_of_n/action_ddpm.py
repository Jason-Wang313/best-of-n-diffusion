"""CPU-light DDPM/DDIM action diffusion for trajectory reranking evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class VectorEpsilonModel(nn.Module):
    """Predict diffusion noise for an observation-conditioned action sequence."""

    def __init__(
        self,
        obs_dim: int,
        horizon: int,
        action_dim: int = 2,
        hidden: int = 96,
        time_dim: int = 1,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        in_dim = self.obs_dim + self.horizon * self.action_dim + int(time_dim)
        out_dim = self.horizon * self.action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, obs: torch.Tensor, noisy_actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat = noisy_actions.reshape(noisy_actions.shape[0], -1)
        x = torch.cat([obs, flat, t.reshape(-1, 1)], dim=1)
        return self.net(x).reshape(noisy_actions.shape)


@dataclass(frozen=True)
class DDPMSchedule:
    timesteps: int
    betas: np.ndarray
    alphas: np.ndarray
    alpha_bars: np.ndarray


@dataclass
class ActionDiffusionPolicy:
    model: VectorEpsilonModel
    schedule: DDPMSchedule
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @property
    def horizon(self) -> int:
        return self.model.horizon

    @property
    def action_dim(self) -> int:
        return self.model.action_dim


@dataclass(frozen=True)
class DiffusionTrainingResult:
    initial_loss: float
    final_loss: float
    epochs: int
    diffusion_steps: int
    target: str = "epsilon"


def make_ddpm_schedule(
    diffusion_steps: int = 32,
    beta_start: float = 1e-4,
    beta_end: float = 0.04,
) -> DDPMSchedule:
    """Create a small linear noise schedule with a t=0 sentinel."""

    steps = int(diffusion_steps)
    if steps < 1:
        raise ValueError("diffusion_steps must be >= 1")
    beta_values = np.linspace(float(beta_start), float(beta_end), steps, dtype=np.float32)
    betas = np.concatenate([np.asarray([0.0], dtype=np.float32), beta_values])
    alphas = 1.0 - betas
    alpha_bars = np.ones(steps + 1, dtype=np.float32)
    alpha_bars[1:] = np.cumprod(alphas[1:])
    return DDPMSchedule(timesteps=steps, betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def _standardize(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    return (arr - mean) / std, mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32)


def _adam_step(
    params: list[torch.Tensor],
    m: list[torch.Tensor],
    v: list[torch.Tensor],
    opt_step: int,
    lr: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
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


def train_epsilon_denoiser(
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    seed: int,
    diffusion_steps: int = 32,
    lr: float = 1.5e-3,
    batch_size: int = 128,
    hidden: int = 96,
) -> tuple[ActionDiffusionPolicy, DiffusionTrainingResult]:
    """Train a DDPM-style epsilon predictor for action trajectories."""

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    obs_arr = np.asarray(obs, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32)
    if obs_arr.ndim != 2:
        raise ValueError("obs must be a 2D matrix")
    if actions_arr.ndim != 3:
        raise ValueError("actions must have shape rows,horizon,action_dim")
    if obs_arr.shape[0] != actions_arr.shape[0]:
        raise ValueError("obs and actions must have the same number of rows")

    obs_norm, obs_mean, obs_std = _standardize(obs_arr)
    flat_actions = actions_arr.reshape(actions_arr.shape[0], -1)
    action_norm_flat, action_mean_flat, action_std_flat = _standardize(flat_actions)
    action_norm = action_norm_flat.reshape(actions_arr.shape)
    action_mean = action_mean_flat.reshape(actions_arr.shape[1], actions_arr.shape[2])
    action_std = action_std_flat.reshape(actions_arr.shape[1], actions_arr.shape[2])

    schedule = make_ddpm_schedule(diffusion_steps=diffusion_steps)
    model = VectorEpsilonModel(
        obs_dim=obs_arr.shape[1],
        horizon=actions_arr.shape[1],
        action_dim=actions_arr.shape[2],
        hidden=hidden,
    )
    obs_t = torch.as_tensor(obs_norm, dtype=torch.float32)
    x0_t = torch.as_tensor(action_norm, dtype=torch.float32)
    alpha_bars = torch.as_tensor(schedule.alpha_bars, dtype=torch.float32)
    n = x0_t.shape[0]

    eval_gen = torch.Generator().manual_seed(int(seed) + 991)
    eval_t_idx = torch.randint(1, schedule.timesteps + 1, (n,), dtype=torch.long, generator=eval_gen)
    eval_eps = torch.randn(x0_t.shape, dtype=torch.float32, generator=eval_gen)
    eval_ab = alpha_bars[eval_t_idx].reshape(-1, 1, 1)
    eval_x_t = torch.sqrt(eval_ab) * x0_t + torch.sqrt(1.0 - eval_ab) * eval_eps
    eval_t_scaled = eval_t_idx.to(torch.float32) / float(schedule.timesteps)

    def loss_once() -> torch.Tensor:
        pred = model(obs_t, eval_x_t, eval_t_scaled)
        return torch.mean((pred - eval_eps) ** 2)

    with torch.no_grad():
        initial = float(loss_once().detach().cpu().item())

    params = [p for p in model.parameters() if p.requires_grad]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    opt_step = 0
    for _ in range(int(epochs)):
        order = rng.permutation(n)
        for start in range(0, n, int(batch_size)):
            idx = torch.as_tensor(order[start : start + int(batch_size)], dtype=torch.long)
            batch_obs = obs_t[idx]
            batch_x0 = x0_t[idx]
            t_idx = torch.randint(1, schedule.timesteps + 1, (batch_x0.shape[0],), dtype=torch.long)
            eps = torch.randn_like(batch_x0)
            ab = alpha_bars[t_idx].reshape(-1, 1, 1)
            x_t = torch.sqrt(ab) * batch_x0 + torch.sqrt(1.0 - ab) * eps
            t_scaled = t_idx.to(torch.float32) / float(schedule.timesteps)
            pred = model(batch_obs, x_t, t_scaled)
            loss = torch.mean((pred - eps) ** 2)
            model.zero_grad(set_to_none=True)
            loss.backward()
            opt_step += 1
            _adam_step(params, m, v, opt_step, lr)

    with torch.no_grad():
        final = float(loss_once().detach().cpu().item())

    policy = ActionDiffusionPolicy(
        model=model,
        schedule=schedule,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean.astype(np.float32),
        action_std=action_std.astype(np.float32),
    )
    return policy, DiffusionTrainingResult(
        initial_loss=initial,
        final_loss=final,
        epochs=int(epochs),
        diffusion_steps=int(diffusion_steps),
    )


def _obs_batch(policy: ActionDiffusionPolicy, obs_vec: np.ndarray, n: int) -> torch.Tensor:
    obs = np.asarray(obs_vec, dtype=np.float32)
    if obs.ndim != 1 or obs.shape[0] != policy.obs_mean.shape[0]:
        raise ValueError("obs_vec has incompatible shape")
    norm = (obs - policy.obs_mean) / policy.obs_std
    return torch.as_tensor(np.repeat(norm[None, :], int(n), axis=0), dtype=torch.float32)


def _denormalize_actions(policy: ActionDiffusionPolicy, x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy().astype(np.float32)
    return (arr * policy.action_std[None, :, :] + policy.action_mean[None, :, :]).astype(float)


def _sampling_times(schedule: DDPMSchedule, k: int) -> list[int]:
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    values = np.linspace(schedule.timesteps, 1, min(k, schedule.timesteps))
    times = []
    for value in values:
        t = int(round(float(value)))
        if t not in times:
            times.append(max(1, min(schedule.timesteps, t)))
    if times[-1] != 1:
        times.append(1)
    return times


def _predict_x0(policy: ActionDiffusionPolicy, obs_t: torch.Tensor, x_t: torch.Tensor, t_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    schedule = policy.schedule
    t_scaled = torch.full((x_t.shape[0],), float(t_idx) / float(schedule.timesteps), dtype=torch.float32)
    eps = policy.model(obs_t, x_t, t_scaled)
    ab = float(schedule.alpha_bars[int(t_idx)])
    x0 = (x_t - np.sqrt(max(1.0 - ab, 0.0)) * eps) / max(np.sqrt(ab), 1e-6)
    return x0, eps


def sample_ddim_trajectories(
    policy: ActionDiffusionPolicy,
    obs_vec: np.ndarray,
    *,
    n: int,
    k: int,
    seed: int,
    temperature: float = 1.0,
) -> np.ndarray:
    """Fast deterministic DDIM-style sampling from the trained epsilon model."""

    torch.manual_seed(int(seed) % (2**31 - 1))
    rng = np.random.default_rng(seed)
    policy.model.eval()
    obs_t = _obs_batch(policy, obs_vec, int(n))
    init = rng.normal(scale=float(temperature), size=(int(n), policy.horizon, policy.action_dim)).astype(np.float32)
    x = torch.as_tensor(init, dtype=torch.float32)
    times = _sampling_times(policy.schedule, k)
    with torch.no_grad():
        for idx, t_idx in enumerate(times):
            prev_t = times[idx + 1] if idx + 1 < len(times) else 0
            x0, eps = _predict_x0(policy, obs_t, x, t_idx)
            if prev_t == 0:
                x = x0
            else:
                ab_prev = float(policy.schedule.alpha_bars[prev_t])
                x = np.sqrt(ab_prev) * x0 + np.sqrt(max(1.0 - ab_prev, 0.0)) * eps
    return _denormalize_actions(policy, x)


def sample_ddpm_trajectories(
    policy: ActionDiffusionPolicy,
    obs_vec: np.ndarray,
    *,
    n: int,
    k: int,
    seed: int,
    temperature: float = 1.0,
    eta: float = 0.18,
) -> np.ndarray:
    """Stochastic DDPM-like sampling on a possibly shortened timestep schedule."""

    torch.manual_seed(int(seed) % (2**31 - 1))
    rng = np.random.default_rng(seed)
    policy.model.eval()
    obs_t = _obs_batch(policy, obs_vec, int(n))
    init = rng.normal(scale=float(temperature), size=(int(n), policy.horizon, policy.action_dim)).astype(np.float32)
    x = torch.as_tensor(init, dtype=torch.float32)
    times = _sampling_times(policy.schedule, k)
    with torch.no_grad():
        for idx, t_idx in enumerate(times):
            prev_t = times[idx + 1] if idx + 1 < len(times) else 0
            x0, eps = _predict_x0(policy, obs_t, x, t_idx)
            if prev_t == 0:
                x = x0
            else:
                ab_prev = float(policy.schedule.alpha_bars[prev_t])
                base = np.sqrt(ab_prev) * x0 + np.sqrt(max(1.0 - ab_prev, 0.0)) * eps
                noise = torch.randn_like(x) * (float(eta) * float(temperature) * np.sqrt(max(1.0 - ab_prev, 0.0)))
                x = base + noise
    return _denormalize_actions(policy, x)


def sample_consistency_trajectories(
    policy: ActionDiffusionPolicy,
    obs_vec: np.ndarray,
    *,
    n: int,
    seed: int,
    temperature: float = 1.0,
) -> np.ndarray:
    """One-step distilled variant: predict x0 directly from terminal noise."""

    torch.manual_seed(int(seed) % (2**31 - 1))
    rng = np.random.default_rng(seed)
    policy.model.eval()
    obs_t = _obs_batch(policy, obs_vec, int(n))
    init = rng.normal(scale=float(temperature), size=(int(n), policy.horizon, policy.action_dim)).astype(np.float32)
    x = torch.as_tensor(init, dtype=torch.float32)
    with torch.no_grad():
        x0, _ = _predict_x0(policy, obs_t, x, policy.schedule.timesteps)
    return _denormalize_actions(policy, x0)


def diffusion_internal_scores(
    policy: ActionDiffusionPolicy,
    obs_vec: np.ndarray,
    trajectories: np.ndarray,
    *,
    seed: int = 0,
    probes: int = 3,
) -> np.ndarray:
    """Likelihood-style score from negative epsilon-prediction residual."""

    torch.manual_seed(int(seed) % (2**31 - 1))
    traj = np.asarray(trajectories, dtype=np.float32)
    if traj.ndim != 3:
        raise ValueError("trajectories must have shape n,horizon,action_dim")
    x0 = (traj - policy.action_mean[None, :, :]) / policy.action_std[None, :, :]
    x0_t = torch.as_tensor(x0, dtype=torch.float32)
    obs_t = _obs_batch(policy, obs_vec, traj.shape[0])
    alpha_bars = torch.as_tensor(policy.schedule.alpha_bars, dtype=torch.float32)
    losses = []
    with torch.no_grad():
        for probe in range(int(probes)):
            t_idx = torch.randint(1, policy.schedule.timesteps + 1, (traj.shape[0],), dtype=torch.long)
            eps = torch.randn_like(x0_t)
            ab = alpha_bars[t_idx].reshape(-1, 1, 1)
            x_t = torch.sqrt(ab) * x0_t + torch.sqrt(1.0 - ab) * eps
            t_scaled = t_idx.to(torch.float32) / float(policy.schedule.timesteps)
            pred = policy.model(obs_t, x_t, t_scaled)
            losses.append(torch.mean((pred - eps) ** 2, dim=(1, 2)).detach().cpu().numpy())
    return -np.mean(np.asarray(losses, dtype=np.float32), axis=0).astype(float)
