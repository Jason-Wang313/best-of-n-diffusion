# Claims

Claims are promotable only when supported by `results/claims_status.json` and corresponding CSV/JSON artifacts.

## Promotable Claim Categories

1. The finite tie-aware Best-of-N law is implemented and tested.
2. High `N` can help aligned diffusion trajectory selection.
3. High `N` can hurt or saturate under scorer misalignment.
4. Low sample diversity reduces the marginal value of increasing `N`.
5. The `N` versus `K` denoising-budget tradeoff is measured.
6. Latency-adjusted utility can prefer smaller `N` or smaller `K`.
7. A calibrated scorer repairs high-`N` selection in at least one synthetic regime.
8. The project is not a WAM clone.
9. The project is not a JEPA clone.
10. Major promoted claims are backed by CSV/JSON artifacts.
11. The learned Diffusion Policy-lite experiment includes state-conditioned and 32x32 image-conditioned variants with seed-level summaries, visual OOD regimes, confidence intervals, and receding-horizon evaluation.
12. A true epsilon-prediction action DDPM/DDIM policy reproduces the Best-of-N law with stochastic DDPM-style sampling, one-step consistency-style sampling, and the older clean-target denoiser held as an ablation.
13. A PushT simulator benchmark path reranks actual sampled action trajectories using real environment rollout utility and shows aligned gains, low-diversity saturation, and misaligned-scorer failure/gap behavior.
14. Runtime recommendations are backed by both abstract `N x K` budget sweeps and measured wall-clock runtime for true diffusion and PushT sampling/rollouts.
15. Global diffusion-policy wording is promoted only when controlled, learned-lite, true-DDPM, and PushT tiers all pass.

## Non-Claims

This repository does not claim real-robot validation, universal Diffusion Policy improvement, production-scale visual policy quality, or a production deployment rule. The PushT path is simulator evidence trained from heuristic demonstrations, not a full visual Diffusion Policy benchmark suite. The calibration map is evidence for "repairs at least one regime," not evidence that calibration always repairs high-`N` selection.
