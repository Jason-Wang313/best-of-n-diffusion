# Method

For each observation `o`, sample `N` action trajectories from either a controlled diffusion-like generator or a learned action diffusion policy. Each trajectory has denoising metadata, a reranker score `S(o, tau)`, and measured real utility `U(o, tau)`.

The method has four parts:

- exact finite Best-of-N curves for selected score and selected real utility;
- trajectory-pool diversity metrics: pairwise distance, effective diversity, mode coverage, collapse rate, marginal diversity gain;
- score-utility alignment metrics: correlation, top-score-tail utility, tail rank correlation, high-`N` regret, oracle-reranker gap;
- paired seed/state confidence intervals for promoted high-minus-low and scorer-gap effects;
- latency-adjusted selection: `U - lambda C(N, K)` with stop rules over `N` and `K`.

The deployment gate returns one of: `allow_high_n`, `stop_early`, `increase_diversity`, `calibrate_reranker`, `reduce_denoising_steps`, or `block_high_n`.

The learned toy policy has two conditioning paths: a state-vector MLP denoiser and a 32x32 rendered-observation tiny-CNN denoiser. Both generate horizon-length action sequences through iterative denoising and are evaluated with the same Best-of-N and receding-horizon diagnostics. This tier tests the pipeline on learned generators but is not the main source for global diffusion-policy wording.

The true action diffusion tier trains an epsilon-prediction DDPM objective over action trajectories, then compares DDIM fast sampling, stochastic DDPM-style sampling, a one-step consistency-style variant, and the older clean-target denoiser as an ablation. The PushT tier uses the same epsilon-policy machinery with low-dimensional PushT observations and evaluates candidate utility, max coverage, final coverage, and success by actual `gym_pusht/PushT-v0` simulator rollout.
