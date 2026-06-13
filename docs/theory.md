# Theory

## Finite Tie-Aware Trajectory-Selection Law

For one observation `o`, a diffusion policy samples action trajectories `tau_i ~ pi_theta(tau | o)`. Each trajectory has a scalar reranker score `S(o, tau_i)` and real utility `U(o, tau_i)`.

Maximum-score trajectory selection chooses `tau* = argmax_i S(o, tau_i)`. On a finite candidate pool sampled with replacement, the exact expected selected utility is determined by the finite joint distribution of `(S, U)`.

If score tie group `g` occupies sorted ranks `r_min(g)` through `r_max(g)` among `m` finite candidates, then its contribution at sample count `N` is:

`mean_U(g) * [(r_max(g) / m)^N - ((r_min(g) - 1) / m)^N]`.

The implementation in `src/diffusion_audit/theory.py` supports real-valued utilities, binary success/failure utilities, exact selected-score curves, and Monte Carlo simulation with random tie handling.

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

## First-Principles Controller Propositions

1. **Aligned tails permit high `N` only under lower-bound evidence.** If candidate diversity is effective and the empirical-Bernstein lower bounds on high-minus-low utility gain, top-score-tail utility lift, and latency-adjusted gain are all positive, high `N` is admissible for the audited finite distribution.
2. **Anti-aligned tails block high `N`.** If the upper score tail is anti-aligned with real utility, max-over-`N` increases selected score while selected real utility can fall. The conservative action is `block_high_N`.
3. **Collapsed pools use effective `N`.** When effective sample diversity is near one or duplicate collapse is high, nominal `N` is replaced by a low effective sample count and the action is `increase_diversity`, not further sampling.
4. **Latency creates a finite optimum.** With objective `U - lambda C(N,K)`, the best `N,K` may be smaller than the raw-utility optimum. The controller can therefore emit `stop_early` or `reduce_K`.
5. **Repair is audited on held-out candidates.** Isotonic or affine calibration is admitted only when held-out lower-bound gates pass; otherwise the controller falls back to low `N` or blocks high `N`.

## Conservative Admission Guarantee

Fix an audited candidate distribution with measured utilities, diversity diagnostics, and runtime costs. Let the controller allocate total risk `delta` across its one-sided empirical-Bernstein gates. If the bounded-audit assumptions used by those gates hold, then with probability at least `1 - delta`, every `increase_N` decision has positive audited latency-adjusted high-minus-low gain under the finite audited distribution. The guarantee is intentionally one-sided: the controller may abstain, stop early, request rollouts, increase diversity, or block high `N` even when a point estimate looks favorable.

This is not a universal high-`N` improvement theorem. Without audited utilities or a valid bound linking scores to real utility, no method can certify that increasing `N` improves real task utility. In that unknown-utility mode, Audit-Then-Sample recommends `audit_rollouts` or `calibrate_scorer`; it does not admit high `N`.

Full-run audit numbers from `results/audit_then_sample_summary.json`: controller false-admit rate `0.0`, abstention rate `0.9375`, empirical-Bernstein lower-bound coverage `1.0`, and adaptive stopping savings mean `0.667`. The decision table has 16 admitted `increase_N` rows; their minimum utility-gain, tail-utility, and latency-adjusted-gain lower bounds are `0.833`, `0.802`, and `0.827`, respectively.
