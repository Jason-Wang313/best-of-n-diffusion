# Introduction

Diffusion Policy-style controllers can sample multiple action trajectories for the same observation. This creates a practical inference-time question: how many trajectories should be sampled before selecting one with a critic or reranker?

The answer is not "as many as possible." Larger `N` increases upper-tail selection pressure. If the upper score tail is aligned with real task utility, selected utility can improve. If the scorer rewards diffusion artifacts, risky modes, or low-latency shortcuts that do not transfer to real utility, larger `N` can select bad outliers.

This paper studies that tradeoff with finite Best-of-N laws, diversity diagnostics, scorer alignment diagnostics, and latency-adjusted utility. The experiments are CPU-light by design: controlled samplers isolate mechanisms, learned toy policies provide supporting state and 32x32 image-conditioning evidence, true epsilon-prediction action diffusion tests faithful DDPM/DDIM sampling, and PushT provides a lightweight simulator benchmark path with actual rollout utility, coverage, success, and runtime artifacts.
