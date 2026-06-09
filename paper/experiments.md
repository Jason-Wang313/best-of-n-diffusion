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

## Family A2: Audit-Then-Sample Controller

The controller audit runs synthetic action-trajectory pools that isolate the inference-time fix behavior. Regimes include high-diversity aligned sampling, anti-correlated scoring, shuffled scoring, tail-misaligned and adversarial-tail scoring, noisy rollout utility, hidden OOD dynamics, duplicated high-score artifacts, correlated candidate pools, calibration drift, small underpowered audits, collapsed sampling, missing utility, latency-limited aligned sampling, and latency spikes.

The decision table records the risk budget, effective `N` used for confidence bounds, utility-gain LCB, tail-utility LCB, latency-adjusted-gain LCB, high-score-tail harm UCB, admission flag, abstention reason, and false-admit negative-control flag. The calibration table tests isotonic repair, affine fallback, random-score failure, and calibration-drift failure, with success allowed only when held-out lower-bound gates pass. Adaptive rows report early stopping when the upper confidence bound on further gain is below the latency cost.

Required artifacts:

- `results/tables/audit_then_sample_decisions.csv`;
- `results/tables/audit_then_sample_calibration.csv`;
- `results/figures/audit_then_sample_decision_regions.png`.

Full-run audit summary: false-admit rate `0.0`, abstention rate `0.9375`, lower-bound coverage `1.0`, repair-bound validation fraction `1.0`, repair-failure-control fraction `1.0`, and adaptive stopping savings mean `0.667` from `results/audit_then_sample_summary.json`. The decision table has 256 rows: 16 `increase_N` admissions, 112 `block_high_N` actions, 48 `audit_rollouts`, 48 `stop_early`, and 32 `increase_diversity` actions. Every admitted `increase_N` row has positive lower-bound evidence, with minimum utility-gain LCB `0.833`, tail-utility LCB `0.802`, and latency-adjusted-gain LCB `0.827`. The calibration table has 19 success rows and 45 failure or negative-control rows.

## Family B: Supporting Learned Diffusion Policy-Lite

A small MLP denoiser learns to map noisy action trajectories plus state observations to clean expert action sequences. A second CPU-light variant renders 32x32 toy observations of the block, goal, obstacle, and distractors, then conditions the same denoising head on a tiny CNN embedding. The full run trains three learned seeds and writes seed-level aggregates, confidence intervals, ID/OOD curves, and receding-horizon execution. These results support the diagnostic pipeline but are not the central evidence for full diffusion-policy wording.

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

An epsilon-prediction action diffusion model trains on multimodal action trajectories. Evaluation compares DDIM fast sampling, stochastic DDPM-style sampling, one-step consistency-style sampling, and the clean-target denoiser ablation under shared `N` and `K` grids. Scorers include diffusion-internal residual score, behavior cloning, pilot value critic, calibrated critic, weakly aligned score, anti-correlated score, tail-only misaligned score, and oracle real utility. The full run uses four seeds and three evaluation states so key true-DDPM CI rows have at least 12 paired units.

Full-run evidence: DDIM oracle high-minus-low selected-real gain is `0.370` with CI low `0.326`; stochastic DDPM-style oracle gain is `0.381` with CI low `0.336`; anti-correlated scoring changes selected real utility by `-0.406` with CI high `-0.371`. The true-diffusion runtime table has 540 rows.

## Family F: PushT Simulator Benchmark

The PushT path uses `gym_pusht/PushT-v0` with actual simulator rollout utility for sampled action trajectories. Training demonstrations are heuristic and CPU-friendly; the claim is benchmark-path evidence for the reranking law, not full-scale visual Diffusion Policy validation. Regimes include aligned sampling, low-diversity sampling, and high-temperature misaligned-score failure. The full run uses four seeds, three evaluation episodes, horizon 20, 16 candidates, and `K = 1, 8, 16`, then reports selected utility, max coverage, final coverage, success, seed-level summaries, and runtime.

Full-run evidence: aligned PushT oracle selected-utility gain is `0.121` with CI low `0.0576`; selected max-coverage gain is `0.103` with CI low `0.0381`; selected final-coverage gain is `0.0216` with CI low `0.00099`; selected success gain is `0.0`. The artifact contains 2,880 simulator rollout rows, 2,100 rollout-metric seed rows, 315 rollout-metric effect rows, and 180 runtime rows.
