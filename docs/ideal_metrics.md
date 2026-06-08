# Ideal Metrics Gate

`scripts/claim_audit.py` treats `SUPPORTED` as a strong result, not a minimal pass. It also writes `results/ideal_metrics_status.json` and `results/ideal_metrics_status.md`. For promoted empirical effects, point estimates must be backed by bootstrap or paired seed/state confidence intervals.

The strong gate currently requires:

- aligned high-diversity selection: selected score gain and real utility gain at least `0.50`, CI lower bounds at least `0.35`, score-utility correlation at least `0.99`, tail rank correlation at least `0.95`, and high-`N` regret at most `0.01`;
- misaligned selection: selected score gain at least `0.35` while real utility changes by at most `-0.25`, with high-`N` regret at least `0.80`, and CI evidence that the real-utility change remains negative;
- low-diversity regime: absolute high-minus-low `N` real gain at most `0.02`, its CI contained in `[-0.02, 0.02]`, effective sample diversity at most `1.05`, and duplicate collapse rate at least `0.98`;
- `N` versus `K` sweep: at least six `N` values, five `K` values, and thirty phase-grid rows;
- latency rule: best latency-adjusted policy uses at most ten percent of the high-budget corner and improves latency-adjusted utility by at least `2.0`, with paired CI lower bound at least `1.50`;
- calibration repair: calibrated scorer recovers at least `0.80` real utility and at least seventy-five percent of the oracle-minus-misaligned gap, with a calibration map containing at least one strong repair row and at least one no-strong-repair row;
- learned Diffusion Policy-lite: all validity checklist items true, four visual OOD regimes, state and image conditioning, 32x32 images, full-run training over at least three seeds, loss ratios at most `0.30` for state and `0.65` for image, aggregate and seed-aggregate rows at least `200`, receding-horizon rows at least `20`, and calibrated `K=4` real gains at least `0.08` with CI lower bound at least `0.04` across state/image ID/OOD regimes. This tier is supporting evidence.
- true action diffusion: epsilon-prediction target, DDIM, stochastic DDPM-style sampling, one-step consistency-style sampling, clean-target ablation, diffusion-internal/oracle/adversarial scorers, seed-level CI rows with at least 12 paired seed-state units in full runs, measured runtime rows, a sampler-comparison table/figure, positive aligned oracle gains for DDIM and DDPM, and negative real-utility change for an anti-correlated score;
- PushT benchmark: `gym_pusht/PushT-v0`, simulator rollout utility, DDIM/DDPM/consistency samplers, oracle/value/misaligned scorers, rollout-level CSV inventory, selected utility/max-coverage/final-coverage/success curves, seed-level CI rows with at least 12 paired seed-episode units in full runs, aligned oracle gain, low-diversity evidence, a scorer-gap failure case, and measured wall-clock runtime;
- low-power warning: full-run critical CI rows for true diffusion and PushT must report at least 12 paired units, while smoke results remain permissive and are marked as smoke in `results/ideal_metrics_status.json`;
- tiered global gate: true action diffusion and PushT rollout-metric evidence must be `SUPPORTED`; toy-controlled and learned Diffusion Policy-lite gates are retained as diagnostic/supporting tiers, but toy-only or learned-lite-only evidence cannot promote a global diffusion-policy claim.
- reviewer-skepticism checklist: true DDPM survives, PushT survives, no real-robot overclaim, no full visual-policy overclaim, runtime evidence present, negative controls present, and no full-run low-power warning.

These thresholds are CPU evidence gates. They are not real-robot deployment guarantees.
