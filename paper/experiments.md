# Experiments

## Family A: Controlled Diffusion-Like Action Sampler

A 2D reaching/pushing environment generates trajectory pools with controllable `N`, `K`, temperature, diversity, mode coverage, collapse, and biased modes.

Required regimes:

- low-diversity high-quality sampler;
- high-diversity aligned sampler;
- high-diversity misaligned sampler;
- collapsed sampler;
- noisy low-`K` sampler;
- expensive high-`K` sampler.

## Family B: Learned Diffusion Policy-Lite

A small MLP denoiser learns to map noisy action trajectories plus state observations to clean expert action sequences. A second CPU-light variant renders 32x32 toy observations of the block, goal, obstacle, and distractors, then conditions the same denoising head on a tiny CNN embedding. The full run trains three learned seeds and writes seed-level aggregates, confidence intervals, ID/OOD curves, and receding-horizon execution.

Visual OOD regimes:

- distractors;
- shifted colors;
- observation noise;
- hidden obstacle.

## Family C: Scorer/Reranker Comparison

Scorers include random selection, diffusion likelihood proxy, behavior-cloning critic, value critic from pilot rollouts, calibrated critic, misaligned tail scorer, and oracle real-utility selector. The calibration repair map reports both strong-repair and no-strong-repair regimes, so the promotable claim is limited to repair in at least one synthetic setting.

## Family D: N Versus K Budget Law

A phase diagram sweeps `N` and `K` and reports real utility, total budget `B = N x K`, utility per diffusion step, and latency-adjusted utility.

## Family E: True Action DDPM/DDIM

An epsilon-prediction action diffusion model trains on multimodal action trajectories. Evaluation compares DDIM fast sampling, stochastic DDPM-style sampling, one-step consistency-style sampling, and the clean-target denoiser ablation under shared `N` and `K` grids. Scorers include diffusion-internal residual score, behavior cloning, pilot value critic, calibrated critic, weakly aligned score, anti-correlated score, tail-only misaligned score, and oracle real utility.

## Family F: PushT Simulator Benchmark

The PushT path uses `gym_pusht/PushT-v0` with actual simulator rollout utility for sampled action trajectories. Training demonstrations are heuristic and CPU-friendly; the claim is benchmark-path evidence for the reranking law, not full-scale visual Diffusion Policy validation. Regimes include aligned sampling, low-diversity sampling, and high-temperature misaligned-score failure.
