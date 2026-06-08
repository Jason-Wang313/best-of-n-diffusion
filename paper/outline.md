# Paper Outline

## Title

How Many Diffusion Trajectories Should a Robot Sample? Inference-Time Selection Laws for Diffusion Policies

## Thesis

Best-of-N inference is valuable for diffusion action policies only when sampled trajectories are sufficiently diverse, the reranker is aligned with real utility in the upper score tail, and denoising latency does not dominate the utility gain.

## Sections

1. Introduction: inference-time trajectory selection and diffusion tail over-selection.
2. Formal setup: finite tie-aware Best-of-N law for `(S, U)` action-trajectory pools.
3. Diagnostics: diversity, alignment, denoising/latency, deployment gate.
4. Experiments: controlled sampler, scorer comparison with calibration success/failure map, `N` versus `K`, supporting learned state/image Diffusion Policy-lite, primary true action DDPM/DDIM, and PushT simulator benchmark with rollout coverage/success.
5. Audit readiness: CI-backed claim gates, artifact inventory, CPU-only scope.
6. Limitations: CPU simulator evidence, toy learned model, no broad real-robot claim.
