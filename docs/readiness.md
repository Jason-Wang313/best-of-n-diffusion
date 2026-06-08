# Audit Readiness

This repository is ready to make only claims that pass `scripts/run_claim_audit.sh`. The strongest current claim is the tiered inference-time law: high `N` is useful only when candidate diversity and upper-tail scorer alignment are good enough to justify the added denoising/runtime cost. Global diffusion-policy wording now depends on the true-DDPM and PushT rollout-metric gates, with toy and learned-lite tiers used as diagnostics and supporting context.

The weakest remaining claim is external validity beyond CPU simulation: the repo now includes true epsilon-prediction action diffusion and a PushT simulator path, but it still does not establish real-robot performance, production-scale visual manipulation quality, or universal high-`N` improvement.

## Artifact Inventory

Core summaries:

- `results/controlled_sampler_summary.json`
- `results/scorer_comparison_summary.json`
- `results/nk_budget_summary.json`
- `results/learned_policy_lite_summary.json`
- `results/true_diffusion_summary.json`
- `results/pusht_summary.json`
- `results/ideal_metrics_status.json`

Primary tables:

- `results/tables/controlled_sampler_curves.csv`
- `results/tables/controlled_sampler_seed_aggregate.csv`
- `results/tables/controlled_sampler_effect_cis.csv`
- `results/tables/scorer_comparison_curves.csv`
- `results/tables/scorer_comparison_seed_aggregate.csv`
- `results/tables/scorer_comparison_effect_cis.csv`
- `results/tables/calibration_repair_map.csv`
- `results/tables/nk_budget_phase.csv`
- `results/tables/nk_budget_latency_effect_ci.csv`
- `results/tables/learned_policy_lite_curves.csv`
- `results/tables/learned_policy_lite_seed_aggregate.csv`
- `results/tables/learned_policy_lite_effect_cis.csv`
- `results/tables/learned_policy_lite_receding_horizon.csv`
- `results/tables/true_diffusion_curves.csv`
- `results/tables/true_diffusion_seed_aggregate.csv`
- `results/tables/true_diffusion_effect_cis.csv`
- `results/tables/true_diffusion_runtime.csv`
- `results/tables/true_diffusion_sampler_comparison.csv`
- `results/tables/pusht_curves.csv`
- `results/tables/pusht_seed_aggregate.csv`
- `results/tables/pusht_effect_cis.csv`
- `results/tables/pusht_runtime.csv`
- `results/tables/pusht_rollouts.csv`
- `results/tables/pusht_rollout_metric_seed_aggregate.csv`
- `results/tables/pusht_rollout_metric_aggregate.csv`
- `results/tables/pusht_rollout_metric_effect_cis.csv`

Primary figures:

- `results/figures/controlled_sampler_curves.png`
- `results/figures/scorer_comparison.png`
- `results/figures/nk_budget_phase_diagram.png`
- `results/figures/learned_policy_lite_ood.png`
- `results/figures/toy_image_observations.png`
- `results/figures/true_diffusion_survival.png`
- `results/figures/true_diffusion_runtime.png`
- `results/figures/true_diffusion_sampler_comparison.png`
- `results/figures/pusht_best_of_n.png`

## Scope

All acceptance runs are CPU-only by default. The image-conditioned model uses 32x32 toy renderings and a tiny CNN encoder. PushT is the single lightweight simulator benchmark path and uses low-dimensional observations plus heuristic demonstrations for training. The full run is configured for 12 paired seed-state or seed-episode CI units in the true-DDPM and PushT tiers. Robot hardware, GPU-scale vision training, and real-world deployment validation are out of scope.
