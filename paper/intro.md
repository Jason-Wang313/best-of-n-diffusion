# Introduction

Diffusion Policy-style controllers can sample multiple action trajectories for the same observation. This creates a practical inference-time question: how many trajectories should be sampled before selecting one with a critic or reranker?

The answer is not "as many as possible." Larger `N` increases upper-tail selection pressure. If the upper score tail is aligned with real task utility, selected utility can improve. If the scorer rewards diffusion artifacts, risky modes, or low-latency shortcuts that do not transfer to real utility, larger `N` can select bad outliers.

This paper studies that tradeoff with finite Best-of-N laws, diversity diagnostics, scorer alignment diagnostics, latency-adjusted utility, and Audit-Then-Sample, an inference-time controller that admits high `N` only when conservative lower-bound gates pass and otherwise stops, audits rollouts, repairs, increases diversity, reduces `K`, or blocks high-`N` selection. The experiments are CPU-light by design: controlled samplers isolate mechanisms, controller negative controls test the admission and abstention path, learned toy policies provide supporting state and 32x32 image-conditioning evidence, true epsilon-prediction action diffusion tests faithful DDPM/DDIM sampling, and PushT provides a lightweight simulator benchmark path with actual rollout utility, coverage, success, and runtime artifacts.

The contributions are intentionally separated. First, we give a finite, tie-aware selection law for action-trajectory pools. Second, we use it in a conservative Audit-Then-Sample controller that certifies `increase_N` only under lower-bound evidence and abstains otherwise. Third, we test faithful epsilon-prediction DDPM/DDIM and PushT simulator trajectories, so the strongest diffusion-policy wording is not carried by toy samplers alone. Fourth, we maintain a claim audit that ties headline text to CSV/JSON artifacts, confidence intervals, negative controls, and non-claim boundaries.

The scope is CPU simulation. We do not claim real-robot validation, universal high-`N` improvement, production-scale visual policy quality, or a hardware safety certificate.
