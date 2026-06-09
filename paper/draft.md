# How Many Diffusion Trajectories Should a Robot Sample? Inference-Time Selection Laws for Diffusion Policies

## Abstract

Diffusion action policies can generate many candidate trajectories for the same observation, making Best-of-N inference an attractive test-time tool. This paper asks when that extra sampling is actually worth using and what to do when it is not. We study a finite setting in which an observation-conditioned generator proposes action trajectories, a scorer or reranker selects the top candidate, and performance is measured by task utility after optional latency cost. The central result is not that larger `N` always helps. Best-of-`N` obeys measurable selection laws, and high-`N` sampling is useful only when the audited candidate pool supports it: Audit-Then-Sample admits extra diffusion trajectories only when conservative lower-bound gates pass and otherwise abstains, audits, repairs, stops early, increases diversity, reduces `K`, or blocks high-`N` selection.

We combine a tie-aware finite Best-of-N law with diagnostics for diversity, upper-tail alignment, latency-adjusted utility, confidence-bound gates, and an inference-time repair path. The evidence is CPU-simulation evidence: controlled action samplers isolate mechanisms, a learned Diffusion Policy-lite tier checks the pipeline on learned state and tiny-image denoisers, a true epsilon-prediction action DDPM/DDIM tier tests faithful diffusion sampling, and a PushT simulator tier evaluates actual rollout utility, coverage, success, and runtime. We characterize, diagnose, control, and repair inference-time selection behavior within this scope; we do not claim real-robot validation, universal high-`N` improvement, or full visual-policy validation.

## 1. Introduction

Diffusion Policy-style controllers are naturally stochastic. Given an observation, the policy can sample multiple action sequences, score them with a critic, value model, likelihood proxy, behavior-cloning score, or hand-built reranker, and execute the best one. This creates a practical inference-time question:

How many diffusion trajectories should a robot sample?

The naive answer is "more." If the score is well aligned with task utility, sampling more candidates increases the chance of finding a high-utility trajectory. But the same selection pressure can amplify errors. A scorer that rewards artifacts, shortcuts, risky modes, or distribution-tail oddities can become worse as `N` increases because Best-of-N searches harder through the scorer's upper tail. The issue is not just average score-utility correlation. The decisive behavior often lives in the extreme candidates that become available only when sampling more trajectories.

This paper studies that inference-time selection problem as a finite selection law over sampled action trajectories. For each observation `o`, a generator produces a finite pool of candidate trajectories. Each candidate has a score `S(o, tau)` and real measured utility `U(o, tau)`. Best-of-N samples a subset of size `N`, selects the candidate with maximum score, and receives its utility. This framing lets us separate three mechanisms:

1. Diversity: larger `N` has little value if additional samples are duplicates or near-duplicates.
2. Upper-tail alignment: larger `N` helps only if high-scoring tail candidates are also high-utility tail candidates.
3. Latency: larger `N` and denoising depth `K` can lose after runtime cost, even when raw utility improves.

The intended contribution is a diagnose-predict-fix framework, not a new robot benchmark suite. We keep the scope narrow and auditable. The strongest claims must pass `scripts/claim_audit.py`, which writes `results/ideal_metrics_status.json` and checks for supported claims, overclaims, low-power warnings, negative controls, runtime evidence, controller/fix evidence, and PushT rollout metrics.

**Contributions.**

1. We give a finite, tie-aware Best-of-N selection law for action-trajectory pools, separating selected-score improvement from selected real-utility improvement.
2. We build Audit-Then-Sample, a conservative inference-time controller that certifies `increase_N` only under diversity, tail-utility, utility-gain, and latency-adjusted lower-bound gates, and otherwise abstains or recommends a repair action.
3. We test the mechanism beyond controlled samplers with learned Diffusion Policy-lite, true epsilon-prediction DDPM/DDIM action sampling, and a PushT simulator path with actual rollout utility, coverage, success, and runtime.
4. We provide an audit infrastructure that maps headline claims to CSV/JSON artifacts, negative controls, confidence intervals, and scope gates.

**Scope and non-claims.**
The paper claims conditional inference-time evidence under audited CPU simulation regimes. It does not claim real-robot validation, universal high-`N` improvement, production-scale visual Diffusion Policy performance, or that calibration always repairs a bad scorer. The PushT path is simulator evidence from low-dimensional observations and heuristic demonstrations, not a full visual imitation-learning benchmark. Audit-Then-Sample is a certification-and-abstention controller for measured candidate pools, not a hardware safety certificate.

## 2. Setup

For a fixed observation `o`, let the policy or sampler produce a finite candidate pool:

```text
P(o) = {(tau_i, S_i, U_i)} for i = 1,...,M.
```

`tau_i` is an action trajectory, `S_i` is a selection score, and `U_i` is the real utility measured by the task simulator or toy control objective. For a sample count `N <= M`, Best-of-N draws a subset of `N` candidates from the pool and executes the candidate with highest score. The expected selected score and expected selected real utility are finite-pool quantities, with ties handled by averaging over the tied top-score group.

The finite law is implemented in `src/diffusion_best_of_n/theory.py` and tested in `tests/test_theory.py`. The empirical curves use:

- selected score: `E[S(argmax S)]`;
- selected real utility: `E[U(argmax S)]`;
- oracle selected utility: `E[U(argmax U)]`;
- high-`N` regret: oracle selected utility minus selected utility;
- paired high-minus-low effects over seed/state or seed/episode units.

The paper's language uses "diffusion policy" only for learned action generators that satisfy the local validity checklist in `docs/diffusion_policy_validity_checklist.md`. Controlled hand-designed samplers are labeled separately.

## 3. Mechanisms

The following first-principles propositions organize the empirical tests.

**Proposition 1: selected-score monotonicity.** For any fixed finite pool and any scorer `S`, the expected selected score under Best-of-`N` is nondecreasing in `N`. This is a property of the maximum operator and does not imply that selected real utility is nondecreasing.

**Proposition 2: aligned upper tails.** If the scorer's high-score tail is aligned with real utility, larger `N` increases the probability of selecting high-utility candidates. The relevant condition is tail alignment, not only average score-utility correlation.

**Proposition 3: anti-aligned or tail-misaligned scorers.** If the high-score tail is anti-aligned with utility, increasing `N` can decrease selected real utility while selected score still rises. Random, shuffled, anti-correlated, and tail-only misaligned scorers are negative controls for this failure.

**Proposition 4: low-diversity saturation.** If candidate trajectories are duplicates or near-duplicates, nominal `N` overstates the useful search size. The controller therefore uses effective sample diversity and collapse rate as gates on the effective `N`.

**Proposition 5: finite latency-optimal `N,K`.** Under `U - lambda C(N,K)`, the optimal sample count and denoising depth can be finite and strictly smaller than the largest available budget.

### 3.1 Diversity

Increasing `N` can only help if new samples explore meaningfully different action trajectories. A collapsed sampler may produce many copies of the same mode; in that case, the expected selected utility saturates quickly. We measure diversity using mean pairwise trajectory distance, effective sample diversity, duplicate collapse rate, cluster count, cluster entropy, and marginal new-mode discovery.

The controlled sampler creates regimes where diversity is intentionally high or low. These regimes are not meant to be realistic robot policies; they isolate the diversity mechanism. The learned and true diffusion tiers then check whether the same diagnostics remain useful when trajectories come from trained denoisers.

Primary artifacts:

- Diversity curves: `results/tables/controlled_sampler_diversity.csv`
- True diffusion diversity: `results/tables/true_diffusion_diversity.csv`
- PushT diversity: `results/tables/pusht_diversity.csv`

Full-run evidence: the controlled low-diversity high-minus-low selected-utility effect is approximately `0.000` with 95% CI containing zero (`[-4.79e-16, 4.23e-16]`), while high-diversity aligned selection gains `0.601` selected real utility with CI low `0.550`. In PushT, the low-diversity oracle gain is small (`0.0195`, CI low `0.00037`), matching the saturation story without promoting it as a broad benchmark win.

### 3.2 Upper-Tail Alignment

Best-of-N is an upper-tail operator. It does not merely improve average score; it selects the maximum score among sampled candidates. Therefore the relevant question is whether the scorer's high-score tail is aligned with high real utility. A scorer can have reasonable average correlation while still failing in the tail.

We report score-utility correlation, tail rank correlation, top-score-tail real utility, high-`N` regret, and oracle-minus-scorer gaps. Negative controls are essential: anti-correlated and tail-only misaligned scorers should fail as `N` increases, while oracle or calibrated scorers should improve or expose the gap.

Primary artifacts:

- Scorer comparison: `results/figures/scorer_comparison.png`
- Scorer gap CIs: `results/tables/scorer_comparison_effect_cis.csv`
- True diffusion scorer gaps: `results/tables/true_diffusion_scorer_gap_cis.csv`
- PushT scorer gaps: `results/tables/pusht_scorer_gap_cis.csv`

Full-run evidence: true action diffusion has an oracle-minus-tail-only gap of `0.174` at high `N` with CI low `0.085`; PushT has an oracle-minus-misaligned high-`N` gap of `0.141` with CI low `0.085`. Negative controls also show harm: true-DDPM anti-correlated scoring changes selected real utility by `-0.406` (CI high `-0.371`), and PushT misaligned scoring changes selected real utility by `-0.048` (CI high `-0.031`).

### 3.3 Latency

Sampling more trajectories and running more denoising steps costs time. We model a latency-adjusted utility:

```text
U_latency = U - lambda * C(N, K),
```

where `K` is the number of denoising steps and `C(N, K)` is an inference-cost proxy or measured runtime. This creates an inference-time controller gate: high `N` is allowed only when diversity and alignment are good enough and the lower confidence bound on the latency-adjusted objective remains favorable.

The budget sweep studies `N x K` tradeoffs in a controlled setting, while the true action diffusion and PushT tiers record measured wall-clock runtime per candidate. The final recommendation is a conditional rule, not a universal prescription: increase `N` when diversity and upper-tail utility lower bounds are high and latency permits; otherwise stop early, audit rollouts, calibrate the scorer, increase diversity, reduce `K`, or block high-`N` selection.

Primary artifacts:

- Budget phase diagram: `results/figures/nk_budget_phase_diagram.png`
- Audit-Then-Sample decision regions: `results/figures/audit_then_sample_decision_regions.png`
- True diffusion runtime: `results/figures/true_diffusion_runtime.png`
- True diffusion sampler comparison: `results/figures/true_diffusion_sampler_comparison.png`
- PushT runtime table: `results/tables/pusht_runtime.csv`

Full-run evidence: the latency-adjusted budget sweep selects `N=32, K=2` rather than the largest `N=32, K=32` corner, using only `6.25%` of the largest tested `N x K` budget and improving the latency-adjusted objective by `3.31` with CI low `3.23`. Measured runtime tables contain 540 true-diffusion rows (`0.18` to `13.42` ms per candidate) and 180 PushT rows (`104.78` to `373.44` ms per candidate, including rollout cost).

### 3.4 Audit-Then-Sample Controller

Audit-Then-Sample is the inference-time controller that turns the diagnostics into an action. Its inputs are candidate trajectories, scorer values, optional measured rollout utilities from pilot/audit candidates, sampler metadata, runtime measurements, candidate-diversity diagnostics, and a user-chosen latency weight `lambda`. Its outputs are a selected `N,K`, a decision label, confidence diagnostics, and one action recommendation from `increase_N`, `stop_early`, `reduce_K`, `calibrate_scorer`, `audit_rollouts`, `increase_diversity`, and `block_high_N`.

```text
Algorithm 1: Audit-Then-Sample
Input: candidate trajectories tau_1:M, scores S_1:M, optional audited utilities U_1:M,
       candidate N grid, K grid, runtime model C(N,K), latency weight lambda
Output: selected N,K, decision label, diagnostics, action recommendation

1. Compute diversity diagnostics: effective sample diversity, collapse rate, mode entropy.
2. If diversity is below threshold, set N to the low audit value and recommend increase_diversity.
3. If audited utilities are unavailable, set N to the low audit value and recommend audit_rollouts or calibrate_scorer.
4. Estimate score-utility correlation, upper-tail rank correlation, and top-score-tail real utility lift.
5. Compute finite-pool selected-utility curves and latency-adjusted objective U_N - lambda C(N,K).
6. Permit larger N only if:
   a. effective diversity is sufficient;
   b. utility-gain, tail-utility, and latency-adjusted empirical-Bernstein lower bounds are positive;
   c. high-score-tail harm is not plausible under the upper confidence gate.
7. If the tail is anti-aligned or high-N utility is harmful, recommend block_high_N.
8. If alignment is weak but not proven harmful, fit monotone isotonic calibration with affine fallback on pilot candidates.
9. Validate repair on held-out candidates; use it only if the same lower-bound gates pass.
10. Choose the latency-adjusted best N,K among admitted actions; otherwise stop early or reduce K.
```

Primary artifacts:

- `results/tables/audit_then_sample_decisions.csv`
- `results/tables/audit_then_sample_calibration.csv`
- `results/figures/audit_then_sample_decision_regions.png`

Full-run audit summary from `results/audit_then_sample_summary.json`: false-admit rate `0.0`; abstention rate `0.9375`; empirical-Bernstein lower-bound coverage `1.0`; adaptive stopping savings mean `0.667`; calibration success rows `19`; calibration failure rows `45`; controller/fix gate supported.

## 4. Experiments

### 4.1 Controlled Diffusion-Like Sampler

The controlled sampler generates 2D action trajectories with known diversity, mode coverage, denoising budget, and scorer alignment. It is used to verify the finite law and isolate failure modes. The key regimes are high-diversity aligned selection, high-diversity misaligned selection, low-diversity saturation, collapsed sampling, noisy low-`K` sampling, and expensive high-`K` sampling.

Primary artifacts:

- `results/tables/controlled_sampler_curves.csv`
- `results/tables/controlled_sampler_effect_cis.csv`
- `results/figures/controlled_sampler_curves.png`

Expected final text after rerun: aligned high-diversity selection improves selected real utility; low diversity saturates; misaligned tail selection can reduce selected real utility at high `N`.

### 4.2 Audit-Then-Sample Controller and Negative Controls

The controller audit evaluates explicit inference-time decisions. It includes high-diversity aligned pools, anti-correlated scorers, shuffled scorers, tail-misaligned and adversarial-tail scorers, duplicated high-score artifacts, correlated pools, hidden OOD dynamics, small underpowered audits, missing utility, latency-limited and latency-spike pools, adaptive stopping, isotonic repair, affine fallback, random-score failed repair, and calibration-drift failed repair. The claim is that the controller can prevent unsupported high-`N` admission in these audited regimes, not that it is a production robot safety layer.

Primary artifacts:

- `results/tables/audit_then_sample_decisions.csv`
- `results/tables/audit_then_sample_calibration.csv`
- `results/figures/audit_then_sample_decision_regions.png`

Concise controller audit:

| Audit item | Full-run value | Artifact |
|---|---:|---|
| Decision rows | 256 | `results/tables/audit_then_sample_decisions.csv` |
| `increase_N` admissions | 16 (6.2%) | `results/tables/audit_then_sample_decisions.csv` |
| Abstentions or non-`increase_N` actions | 240 (93.8%) | `results/audit_then_sample_summary.json` |
| `block_high_N` actions | 112 (43.8%) | `results/tables/audit_then_sample_decisions.csv` |
| False admits in harmful negative controls | 0 | `results/audit_then_sample_summary.json` |
| Lower-bound coverage | 1.0 | `results/audit_then_sample_summary.json` |
| Adaptive stopping savings | 0.667 | `results/audit_then_sample_summary.json` |

Every admitted `increase_N` row has positive lower-bound evidence: minimum utility-gain LCB `0.833`, minimum tail-utility LCB `0.802`, and minimum latency-adjusted-gain LCB `0.827`. Calibration is deliberately asymmetric: 19 held-out repair rows pass and recommend `increase_N`, while 45 rows fail or are negative controls and recommend `block_high_N`.

### 4.3 Scorer and Calibration Comparison

We compare random selection, diffusion likelihood proxy, behavior-cloning critic, pilot value critic, calibrated critic, misaligned tail scorer, and oracle real-utility selector. The calibration map is intentionally limited: it records at least one regime where calibration repairs a bad scorer and at least one regime where it does not produce a strong repair. The claim is not that calibration always works.

Primary artifacts:

- `results/tables/scorer_comparison_curves.csv`
- `results/tables/calibration_repair_map.csv`
- `results/figures/scorer_comparison.png`

### 4.4 N Versus K Budget Sweep

The `N x K` sweep reports raw real utility, budget `B = N x K`, utility per diffusion step, and latency-adjusted utility. This family supplies the abstract latency law that later connects to measured runtime in true diffusion and PushT.

Primary artifacts:

- `results/tables/nk_budget_phase.csv`
- `results/tables/nk_budget_latency_effect_ci.csv`
- `results/figures/nk_budget_phase_diagram.png`

### 4.5 Supporting Learned Diffusion Policy-Lite

The learned-lite tier trains small denoisers that generate horizon-length action sequences. One path conditions on state vectors; the other renders 32x32 toy observations and uses a tiny CNN encoder. The purpose is to test whether the diagnostic pipeline applies to learned noise-to-action generators with state and small-image conditioning.

This tier is supporting evidence. It should not carry the central diffusion-policy claim by itself, because it is intentionally small and toy-like.

Primary artifacts:

- `results/tables/learned_policy_lite_training.csv`
- `results/tables/learned_policy_lite_effect_cis.csv`
- `results/tables/learned_policy_lite_receding_horizon.csv`
- `results/figures/learned_policy_lite_ood.png`
- `results/figures/toy_image_observations.png`

Full-run evidence: for calibrated `K=4`, the state-conditioned learned-lite path reports high-minus-low selected-real gains from `0.114` to `0.167` with minimum CI low `0.092`; the image-conditioned path reports gains from `0.150` to `0.154` with minimum CI low `0.107`. This supports the diagnostic pipeline on learned state and small-image denoisers, not a broad visual-policy claim.

### 4.6 True Action DDPM/DDIM

The main learned diffusion tier trains an epsilon-prediction DDPM objective over action trajectories. It evaluates three primary sampler families:

- `ddim_eps`: fast DDIM-style sampling;
- `ddpm_eps`: stochastic DDPM-style sampling;
- `consistency_1step`: one-step consistency-style variant.

The older clean-target denoiser remains as `clean_target_ablation`. It is useful for comparison but is not the main diffusion-policy claim.

The full run is configured for four seeds and three evaluation states, giving 12 paired seed-state units for key CI rows. The experiment reports selected utility curves, diversity, runtime, sampler comparison, and negative controls.

Primary artifacts:

- `results/tables/true_diffusion_curves.csv`
- `results/tables/true_diffusion_effect_cis.csv`
- `results/tables/true_diffusion_runtime.csv`
- `results/tables/true_diffusion_sampler_comparison.csv`
- `results/figures/true_diffusion_survival.png`
- `results/figures/true_diffusion_runtime.png`
- `results/figures/true_diffusion_sampler_comparison.png`

Full-run evidence: DDIM oracle high-minus-low gain is `0.370` with CI low `0.326`; stochastic DDPM-style oracle gain is `0.381` with CI low `0.336`; anti-correlated scoring changes selected real utility by `-0.406` with CI high `-0.371`. The critical CI rows use 12 paired seed-state units.

### 4.7 PushT Simulator Benchmark

The PushT tier uses `gym_pusht/PushT-v0` with low-dimensional observations and heuristic demonstrations for CPU-feasible training. Candidate trajectories are evaluated by actual simulator rollout. The selected metrics are:

- scalar rollout utility;
- max coverage;
- final coverage;
- success;
- sample and rollout runtime.

The full run is configured for four seeds, three evaluation episodes, horizon 20, 16 candidates, and `K = 1, 8, 16`, producing 12 paired seed-episode units for key CI rows. The benchmark includes aligned, low-diversity, and high-temperature misaligned regimes. It is simulator evidence for the inference-time law, not a full visual imitation-learning benchmark.

Primary artifacts:

- `results/tables/pusht_curves.csv`
- `results/tables/pusht_rollouts.csv`
- `results/tables/pusht_rollout_metric_effect_cis.csv`
- `results/tables/pusht_rollout_metric_seed_aggregate.csv`
- `results/tables/pusht_runtime.csv`
- `results/figures/pusht_best_of_n.png`

Full-run evidence: PushT aligned oracle selected-utility gain is `0.121` with CI low `0.0576`; selected max-coverage gain is `0.103` with CI low `0.0381`; selected final-coverage gain is `0.0216` with CI low `0.00099`; selected success gain is `0.0`. The artifact contains 2,880 simulator rollout rows, 2,100 rollout-metric seed rows, and 12 paired seed-episode units for critical CI rows.

## 5. Audit and Claim Discipline

The repository uses an explicit claim audit to prevent accidental overstatement. The audit writes:

- `results/claims_status.json`
- `results/claims_status.md`
- `results/ideal_metrics_status.json`
- `results/ideal_metrics_status.md`

The audit splits evidence into toy-controlled, controller/fix, learned-policy-lite, true-DDPM, and PushT gates. Global diffusion-policy wording requires the true-DDPM gate and PushT rollout-metric gate. Fix wording requires the controller/fix gate, which combines diversity, tail-alignment, latency, calibration/repair, and negative-control evidence. Learned-lite results remain useful supporting evidence, but they do not by themselves justify broad diffusion-policy language.

The reviewer-skepticism checklist requires:

- true DDPM survives;
- PushT survives with rollout metrics;
- no real-robot overclaim;
- no full visual-policy overclaim;
- runtime evidence present;
- controller/fix evidence present;
- negative controls present;
- no full-run low-power warning.

Full-run audit result: `all_strong=true`, `num_supported=18`, `num_partial=0`, `num_unsupported=0`, and `low_statistical_power.warning=null`. The reviewer-skepticism checklist passes for true DDPM, PushT, runtime, controller/fix evidence, negative controls, overclaim checks, and statistical power.

Reviewer concern/evidence matrix:

| Likely concern | Paper answer | Direct artifact |
|---|---|---|
| "High N helps everywhere" | Misaligned controlled and PushT scorers have negative selected-real changes. | `results/claims_status.md`, claims 3 and 14 |
| "Controller admits unsafe high N" | Harmful negative controls have zero false admits; all admitted rows have positive utility/tail/latency LCBs. | `results/audit_then_sample_summary.json`; `results/tables/audit_then_sample_decisions.csv` |
| "This is only a toy sampler" | Strong wording requires true-DDPM and PushT gates; learned-lite and controlled tiers are supporting context. | `results/ideal_metrics_status.json`, claims 13-18 |
| "Runtime ignored" | Budget sweep and measured true-diffusion/PushT runtime are audited. | `results/tables/nk_budget_latency_effect_ci.csv`; `results/tables/true_diffusion_runtime.csv`; `results/tables/pusht_runtime.csv` |
| "Repair always works" | Calibration has both success and failure rows; repair is used only under held-out lower-bound gates. | `results/tables/audit_then_sample_calibration.csv` |
| "Robot validity overstated" | Scope excludes real robots, hardware safety, production visual policies, and full visual PushT imitation learning. | `paper/limitations.md`; `results/ideal_metrics_status.json` |

### What Would Falsify This?

The main empirical claim would be falsified if any harmful negative-control row were admitted as `increase_N`, if an admitted `increase_N` row had a nonpositive utility, tail, or latency-adjusted lower bound, or if rerunning the claim audit produced unsupported global diffusion-policy wording. The true-DDPM tier would be weakened if DDIM or DDPM oracle gains lost positive lower bounds or anti-correlated scoring no longer showed harmful selection pressure. The PushT tier would be weakened if actual simulator rollout metrics lost the aligned utility or coverage lower bounds, or if the misaligned scorer no longer exposed a positive oracle gap. The controller claim would also fail under distribution shift unless fresh audits re-establish diversity, tail utility, runtime, and negative-control evidence.

## 6. Discussion

The experiments support a conditional view of Best-of-N inference. The same act of sampling more diffusion trajectories can be beneficial, neutral, or harmful. It is beneficial when the generator produces diverse candidates and the scorer's upper tail tracks real utility. It is neutral when diversity collapses. It is harmful when the scorer's tail rewards artifacts or risky behaviors. It can also be rejected after a latency adjustment even when raw utility improves.

This suggests that deployments should treat `N` as a controlled inference-time knob, not a default maximization target. A practical system should estimate diversity, audit scorer tail alignment, measure runtime, and maintain negative-control tests. When those diagnostics fail, the right action is not to sample more. It is to recalibrate the scorer, improve candidate diversity, reduce denoising depth, stop early, or block high-`N` selection.

## 7. Limitations

The evidence is CPU simulation evidence. The controlled sampler is hand-designed. The learned-lite tier is intentionally small. The image-conditioned path uses 32x32 toy renderings and a tiny CNN. The true action diffusion tier is faithful to epsilon-prediction DDPM/DDIM action sampling, but it is trained on a small toy manipulation dataset. PushT is a simulator benchmark path with low-dimensional observations and heuristic demonstrations.

We do not claim real-robot validation. We do not claim universal Diffusion Policy improvement. We do not claim that high `N` always helps. We do not claim that calibration always repairs a bad scorer. We do not claim full visual-policy validation from the PushT path.

## 8. Conclusion

Best-of-N inference for diffusion action policies is governed by diversity, upper-tail alignment, and latency. More samples are worth using only when candidate diversity supplies new useful options, scorer tails select genuinely high-utility trajectories, and runtime cost does not dominate. The paper's contribution is a finite law, Audit-Then-Sample controller, repair path, and auditable evidence stack for deciding when extra diffusion trajectories are worth sampling.
