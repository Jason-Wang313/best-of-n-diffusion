# Theory

## Finite Tie-Aware Best-of-N Law

For one observation `o`, a diffusion policy samples action trajectories `tau_i ~ pi_theta(tau | o)`. Each trajectory has a scalar reranker score `S(o, tau_i)` and real utility `U(o, tau_i)`.

Best-of-N selects `tau* = argmax_i S(o, tau_i)`. On a finite candidate pool sampled with replacement, the exact expected selected utility is determined by the finite joint distribution of `(S, U)`.

If score tie group `g` occupies sorted ranks `r_min(g)` through `r_max(g)` among `m` finite candidates, then its contribution at sample count `N` is:

`mean_U(g) * [(r_max(g) / m)^N - ((r_min(g) - 1) / m)^N]`.

The implementation in `src/diffusion_best_of_n/theory.py` supports real-valued utilities, binary success/failure utilities, exact selected-score curves, and Monte Carlo simulation with random tie handling.

## Diffusion-Specific Reading

The theorem alone does not say a diffusion policy improves with larger `N`. It says the value of larger `N` is a property of the joint distribution of sampled trajectory score and real utility.

For diffusion action generation, the relevant mechanisms are:

- sample diversity and mode coverage;
- denoising steps `K` and residual low-`K` noise;
- scorer/reranker alignment in the upper score tail;
- latency-adjusted utility `E[U(o, tau*) - lambda C(N, K)]`.

The central failure mode is diffusion tail over-selection: a misaligned scorer can make the selected score improve with `N` while selected real utility saturates or decreases.

## Diffusion-Specific Corollaries

- Expected selected real utility depends on upper-tail score-utility alignment, not only average score-utility correlation.
- Low sample diversity bounds the marginal value of increasing `N`; exact collapse makes extra samples nearly redundant.
- A misaligned high-score tail can make selected real utility decrease with `N` even while selected score increases monotonically.
- Under latency-adjusted utility, the optimum can occur at finite `N` or finite `K`, not at the largest tested budget.
- Under a fixed budget `B = N x K`, the best allocation depends on marginal diversity gained from another candidate versus marginal quality gained from another denoising step.
- Monotonic selected score is not evidence of monotonic real utility unless the upper score tail is calibrated to real task utility.
