# Method

For each observation `o`, sample `N` action trajectories from either a controlled diffusion-like generator or a learned action diffusion policy. Each trajectory has denoising metadata, a reranker score `S(o, tau)`, and measured real utility `U(o, tau)`.

The method has four parts:

- exact finite trajectory-search curves for selected score and selected real utility;
- trajectory-pool diversity metrics: pairwise distance, effective diversity, mode coverage, collapse rate, marginal diversity gain;
- score-utility alignment metrics: correlation, top-score-tail utility, tail rank correlation, high-`N` regret, oracle-reranker gap;
- paired seed/state confidence intervals for promoted high-minus-low and scorer-gap effects;
- latency-adjusted selection: `U - lambda C(N, K)` with stop rules over `N` and `K`.

The inference-time controller is Audit-Then-Sample. It audits diversity, tail alignment, high-minus-low utility gain, high-score-tail utility lift, and latency-adjusted utility before admitting larger `N`. Admission uses one-sided empirical-Bernstein lower confidence bounds with a total risk budget `delta`; bootstrap intervals are retained only as diagnostics. It returns selected `N,K`, a decision label, confidence diagnostics, and one action recommendation: `increase_N`, `stop_early`, `reduce_K`, `calibrate_scorer`, `audit_rollouts`, `increase_diversity`, or `block_high_N`.

The repair path fits monotone isotonic calibration with affine fallback on pilot candidates with measured utilities and validates the repaired score on held-out candidates. Repair is admitted only when the same held-out lower-bound gates pass. If average fit improves but the held-out upper-tail utility lower bounds do not pass, repair is marked failed and high `N` remains blocked.

Proposition. Under the bounded finite-audit assumptions of the empirical-Bernstein gates, Audit-Then-Sample admits high `N` only when the audited latency-adjusted high-minus-low gain is positive with probability at least `1 - delta`. The proposition is a conservative certification-and-abstention claim, not a claim that high `N` improves every task.

In the full audit artifacts, this proposition is checked operationally rather than asserted as deployment safety: 16 `increase_N` admissions all have positive utility, tail, and latency-adjusted lower bounds, while 240 of 256 decision rows abstain, stop, audit, increase diversity, or block. The false-admit rate in harmful negative controls is `0.0`.

The learned toy policy has two conditioning paths: a state-vector MLP denoiser and a 32x32 rendered-observation tiny-CNN denoiser. Both generate horizon-length action sequences through iterative denoising and are evaluated with the same max-selection and receding-horizon diagnostics. This tier tests the pipeline on learned generators but is not the main source for global diffusion-policy wording.

The true action diffusion tier trains an epsilon-prediction DDPM objective over action trajectories, then compares DDIM fast sampling, stochastic DDPM-style sampling, a one-step consistency-style variant, and the older clean-target denoiser as an ablation. The PushT tier uses the same epsilon-policy machinery with low-dimensional PushT observations and evaluates candidate utility, max coverage, final coverage, and success by actual `gym_pusht/PushT-v0` simulator rollout.
