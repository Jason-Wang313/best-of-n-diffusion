# Paper Outline

## Title

How Many Diffusion Trajectories Should a Robot Sample? Inference-Time Selection Laws for Diffusion Policies

## Thesis

max-over-`N` obeys measurable selection laws, and an Audit-Then-Sample controller can conservatively admit or reject extra diffusion trajectories. Extra samples are valuable only when sampled trajectories are sufficiently diverse, the reranker's upper score tail has audited real utility support, and denoising latency does not dominate the lower-bound utility gain.

## Sections

1. Introduction: inference-time trajectory selection and diffusion tail over-selection.
2. Formal setup: finite tie-aware trajectory-selection law for `(S, U)` action-trajectory pools.
3. Diagnostics and controller: diversity, alignment, denoising/latency, empirical-Bernstein gates, Audit-Then-Sample, adaptive stopping, and held-out calibration repair.
4. Experiments: controlled sampler, Audit-Then-Sample negative controls, scorer comparison with calibration success/failure map, `N` versus `K`, supporting learned state/image Diffusion Policy-lite, primary true action DDPM/DDIM, and PushT simulator benchmark with rollout coverage/success.
5. Audit readiness: CI-backed claim gates, artifact inventory, CPU-only scope.
6. Limitations: CPU simulator evidence, toy learned model, no broad real-robot claim.
