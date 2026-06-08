# Diffusion Policy Validity Checklist

A result may call itself "Diffusion Policy-style" only if it includes:

- stochastic trajectory generation;
- iterative denoising or diffusion-like noise-to-action generation;
- conditioning on observation/state;
- action sequence generation rather than only one-step action prediction;
- evaluation under receding-horizon or trajectory-execution setting.

If an experiment uses a hand-designed sampler, label it:

`controlled diffusion-like action sampler`

not full Diffusion Policy.

If an experiment trains a learned denoising policy, label it:

`learned Diffusion Policy-lite`

If an experiment trains epsilon prediction with DDIM/DDPM-style sampling, label it:

`true action DDPM/DDIM`

In paper claims, treat `ddim_eps`, `ddpm_eps`, and `consistency_1step` as the primary sampler families for the true diffusion tier. Treat `clean_target_ablation` as an ablation only.

The learned toy experiment in this repository includes state-conditioned and 32x32 image-conditioned variants. The image path uses a tiny CNN encoder over rendered block/goal/obstacle observations with visual OOD regimes, not a full visual robotics benchmark.

The PushT experiment is a simulator benchmark path with actual rollout utility over sampled action trajectories. Promoted PushT claims must include selected utility, max coverage, final coverage, success, runtime, and seed-level robustness artifacts. It is not real-robot validation and not a full visual Diffusion Policy benchmark.

The learned and benchmark experiments are CPU-feasible models, not evidence that full-scale robot Diffusion Policies benefit universally from high `N`.
