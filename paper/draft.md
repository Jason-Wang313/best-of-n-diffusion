# How Many Diffusion Trajectories Should a Robot Sample? Inference-Time Selection Laws for Diffusion Policies

## Abstract

Diffusion action policies can generate many candidate trajectories for the same observation, making Best-of-N inference an attractive test-time tool. This paper asks when that extra sampling is actually worth using. We study a finite setting in which an observation-conditioned generator proposes action trajectories, a scorer or reranker selects the top candidate, and performance is measured by task utility after optional latency cost. The central result is not that larger `N` always helps. Larger `N` is useful only when three mechanisms line up: the candidate pool contains meaningful diversity, the upper tail of the scorer is aligned with real utility, and the denoising/runtime cost does not erase the gain.

We combine a tie-aware finite Best-of-N law with diagnostics for diversity, upper-tail alignment, and latency-adjusted utility. The evidence is CPU-simulation evidence: controlled action samplers isolate mechanisms, a learned Diffusion Policy-lite tier checks the pipeline on learned state and tiny-image denoisers, a true epsilon-prediction action DDPM/DDIM tier tests faithful diffusion sampling, and a PushT simulator tier evaluates actual rollout utility, coverage, success, and runtime. We characterize, diagnose, and control inference-time selection behavior; we do not claim real-robot validation, universal high-`N` improvement, or full visual-policy validation.

## 1. Introduction

Diffusion Policy-style controllers are naturally stochastic. Given an observation, the policy can sample multiple action sequences, score them with a critic, value model, likelihood proxy, behavior-cloning score, or hand-built reranker, and execute the best one. This creates a practical inference-time question:

How many diffusion trajectories should a robot sample?

The naive answer is "more." If the score is well aligned with task utility, sampling more candidates increases the chance of finding a high-utility trajectory. But the same selection pressure can amplify errors. A scorer that rewards artifacts, shortcuts, risky modes, or distribution-tail oddities can become worse as `N` increases because Best-of-N searches harder through the scorer's upper tail. The issue is not just average score-utility correlation. The decisive behavior often lives in the extreme candidates that become available only when sampling more trajectories.

This paper studies that inference-time selection problem as a finite selection law over sampled action trajectories. For each observation `o`, a generator produces a finite pool of candidate trajectories. Each candidate has a score `S(o, tau)` and real measured utility `U(o, tau)`. Best-of-N samples a subset of size `N`, selects the candidate with maximum score, and receives its utility. This framing lets us separate three mechanisms:

1. Diversity: larger `N` has little value if additional samples are duplicates or near-duplicates.
2. Upper-tail alignment: larger `N` helps only if high-scoring tail candidates are also high-utility tail candidates.
3. Latency: larger `N` and denoising depth `K` can lose after runtime cost, even when raw utility improves.

The intended contribution is a diagnostic and evidence framework, not a new robot benchmark suite. We keep the scope narrow and auditable. The strongest claims must pass `scripts/claim_audit.py`, which writes `results/ideal_metrics_status.json` and checks for supported claims, overclaims, low-power warnings, negative controls, runtime evidence, and PushT rollout metrics.

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

### 3.1 Diversity

Increasing `N` can only help if new samples explore meaningfully different action trajectories. A collapsed sampler may produce many copies of the same mode; in that case, the expected selected utility saturates quickly. We measure diversity using mean pairwise trajectory distance, effective sample diversity, duplicate collapse rate, cluster count, cluster entropy, and marginal new-mode discovery.

The controlled sampler creates regimes where diversity is intentionally high or low. These regimes are not meant to be realistic robot policies; they isolate the diversity mechanism. The learned and true diffusion tiers then check whether the same diagnostics remain useful when trajectories come from trained denoisers.

Artifact placeholders:

- Diversity curves: `results/tables/controlled_sampler_diversity.csv`
- True diffusion diversity: `results/tables/true_diffusion_diversity.csv`
- PushT diversity: `results/tables/pusht_diversity.csv`

Final draft insertion: `[FINAL RERUN NUMBER: low-diversity high-minus-low selected utility from results/ideal_metrics_status.json]`.

### 3.2 Upper-Tail Alignment

Best-of-N is an upper-tail operator. It does not merely improve average score; it selects the maximum score among sampled candidates. Therefore the relevant question is whether the scorer's high-score tail is aligned with high real utility. A scorer can have reasonable average correlation while still failing in the tail.

We report score-utility correlation, tail rank correlation, top-score-tail real utility, high-`N` regret, and oracle-minus-scorer gaps. Negative controls are essential: anti-correlated and tail-only misaligned scorers should fail as `N` increases, while oracle or calibrated scorers should improve or expose the gap.

Artifact placeholders:

- Scorer comparison: `results/figures/scorer_comparison.png`
- Scorer gap CIs: `results/tables/scorer_comparison_effect_cis.csv`
- True diffusion scorer gaps: `results/tables/true_diffusion_scorer_gap_cis.csv`
- PushT scorer gaps: `results/tables/pusht_scorer_gap_cis.csv`

Final draft insertion: `[FINAL RERUN NUMBER: true-DDPM oracle-minus-tail gap and PushT oracle-minus-misaligned gap]`.

### 3.3 Latency

Sampling more trajectories and running more denoising steps costs time. We model a latency-adjusted utility:

```text
U_latency = U - lambda * C(N, K),
```

where `K` is the number of denoising steps and `C(N, K)` is an inference-cost proxy or measured runtime. This creates a deployment gate: high `N` is allowed only when diversity and alignment are good enough and the latency-adjusted objective remains favorable.

The budget sweep studies `N x K` tradeoffs in a controlled setting, while the true action diffusion and PushT tiers record measured wall-clock runtime per candidate. The final recommendation is a conditional rule, not a universal prescription: increase `N` when diversity and upper-tail alignment are high and latency permits; otherwise stop early, calibrate the scorer, increase diversity, reduce `K`, or block high-`N` selection.

Artifact placeholders:

- Budget phase diagram: `results/figures/nk_budget_phase_diagram.png`
- True diffusion runtime: `results/figures/true_diffusion_runtime.png`
- True diffusion sampler comparison: `results/figures/true_diffusion_sampler_comparison.png`
- PushT runtime table: `results/tables/pusht_runtime.csv`

Final draft insertion: `[FINAL RERUN NUMBER: latency-adjusted best budget and measured sampler runtime ranges]`.

## 4. Experiments

### 4.1 Controlled Diffusion-Like Sampler

The controlled sampler generates 2D action trajectories with known diversity, mode coverage, denoising budget, and scorer alignment. It is used to verify the finite law and isolate failure modes. The key regimes are high-diversity aligned selection, high-diversity misaligned selection, low-diversity saturation, collapsed sampling, noisy low-`K` sampling, and expensive high-`K` sampling.

Primary artifacts:

- `results/tables/controlled_sampler_curves.csv`
- `results/tables/controlled_sampler_effect_cis.csv`
- `results/figures/controlled_sampler_curves.png`

Expected final text after rerun: aligned high-diversity selection improves selected real utility; low diversity saturates; misaligned tail selection can reduce selected real utility at high `N`.

### 4.2 Scorer and Calibration Comparison

We compare random selection, diffusion likelihood proxy, behavior-cloning critic, pilot value critic, calibrated critic, misaligned tail scorer, and oracle real-utility selector. The calibration map is intentionally limited: it records at least one regime where calibration repairs a bad scorer and at least one regime where it does not produce a strong repair. The claim is not that calibration always works.

Primary artifacts:

- `results/tables/scorer_comparison_curves.csv`
- `results/tables/calibration_repair_map.csv`
- `results/figures/scorer_comparison.png`

### 4.3 N Versus K Budget Sweep

The `N x K` sweep reports raw real utility, budget `B = N x K`, utility per diffusion step, and latency-adjusted utility. This family supplies the abstract latency law that later connects to measured runtime in true diffusion and PushT.

Primary artifacts:

- `results/tables/nk_budget_phase.csv`
- `results/tables/nk_budget_latency_effect_ci.csv`
- `results/figures/nk_budget_phase_diagram.png`

### 4.4 Supporting Learned Diffusion Policy-Lite

The learned-lite tier trains small denoisers that generate horizon-length action sequences. One path conditions on state vectors; the other renders 32x32 toy observations and uses a tiny CNN encoder. The purpose is to test whether the diagnostic pipeline applies to learned noise-to-action generators with state and small-image conditioning.

This tier is supporting evidence. It should not carry the central diffusion-policy claim by itself, because it is intentionally small and toy-like.

Primary artifacts:

- `results/tables/learned_policy_lite_training.csv`
- `results/tables/learned_policy_lite_effect_cis.csv`
- `results/tables/learned_policy_lite_receding_horizon.csv`
- `results/figures/learned_policy_lite_ood.png`
- `results/figures/toy_image_observations.png`

Final draft insertion: `[FINAL RERUN NUMBER: learned-lite state/image calibrated K=4 gains and CI lower bounds]`.

### 4.5 True Action DDPM/DDIM

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

Final draft insertion: `[FINAL RERUN NUMBER: DDIM oracle gain, DDPM oracle gain, anti-correlated high-minus-low change, min CI units]`.

### 4.6 PushT Simulator Benchmark

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

Final draft insertion: `[FINAL RERUN NUMBER: PushT aligned utility gain, selected max/final coverage CI rows, success CI row, rollout rows, min CI units]`.

## 5. Audit and Claim Discipline

The repository uses an explicit claim audit to prevent accidental overstatement. The audit writes:

- `results/claims_status.json`
- `results/claims_status.md`
- `results/ideal_metrics_status.json`
- `results/ideal_metrics_status.md`

The audit splits evidence into toy-controlled, learned-policy-lite, true-DDPM, and PushT gates. Global diffusion-policy wording requires the true-DDPM gate and PushT rollout-metric gate. Learned-lite results remain useful supporting evidence, but they do not by themselves justify broad diffusion-policy language.

The reviewer-skepticism checklist requires:

- true DDPM survives;
- PushT survives with rollout metrics;
- no real-robot overclaim;
- no full visual-policy overclaim;
- runtime evidence present;
- negative controls present;
- no full-run low-power warning.

Final draft insertion: `[FINAL RERUN NUMBER: all_strong, num_partial, num_unsupported, low_statistical_power.warning from results/ideal_metrics_status.json]`.

## 6. Discussion

The experiments support a conditional view of Best-of-N inference. The same act of sampling more diffusion trajectories can be beneficial, neutral, or harmful. It is beneficial when the generator produces diverse candidates and the scorer's upper tail tracks real utility. It is neutral when diversity collapses. It is harmful when the scorer's tail rewards artifacts or risky behaviors. It can also be rejected after a latency adjustment even when raw utility improves.

This suggests that deployments should treat `N` as a controlled inference-time knob, not a default maximization target. A practical system should estimate diversity, audit scorer tail alignment, measure runtime, and maintain negative-control tests. When those diagnostics fail, the right action is not to sample more. It is to recalibrate the scorer, improve candidate diversity, reduce denoising depth, or block high-`N` selection.

## 7. Limitations

The evidence is CPU simulation evidence. The controlled sampler is hand-designed. The learned-lite tier is intentionally small. The image-conditioned path uses 32x32 toy renderings and a tiny CNN. The true action diffusion tier is faithful to epsilon-prediction DDPM/DDIM action sampling, but it is trained on a small toy manipulation dataset. PushT is a simulator benchmark path with low-dimensional observations and heuristic demonstrations.

We do not claim real-robot validation. We do not claim universal Diffusion Policy improvement. We do not claim that high `N` always helps. We do not claim that calibration always repairs a bad scorer. We do not claim full visual-policy validation from the PushT path.

## 8. Conclusion

Best-of-N inference for diffusion action policies is governed by three mechanisms: diversity, upper-tail alignment, and latency. More samples are worth using only when candidate diversity supplies new useful options, scorer tails select genuinely high-utility trajectories, and runtime cost does not dominate. The paper's contribution is a finite law, diagnostic suite, and auditable evidence stack for deciding when extra diffusion trajectories are worth sampling.

