# Differentiation From WAM And JEPA

## What Theorem Is Reused

The reused mathematical object is the finite tie-aware Best-of-N law. For a fixed finite pool of candidates with score `S` and real utility `U`, the expected selected utility under maximum-score selection is exactly determined by the finite joint distribution of `(S, U)`, including score ties.

## What Is New Scientifically

This repository instantiates that theorem for diffusion action trajectory inference:

- stochastic action-sequence generation conditioned on observation/state;
- iterative denoising, true epsilon-prediction DDPM/DDIM sampling, and one-step consistency-style ablation;
- sample diversity, mode coverage, duplicate collapse, and marginal diversity gain;
- denoising-step/sample-count tradeoffs through `B = N x K`;
- scorer alignment in the high-score tail;
- diffusion tail over-selection and latency-adjusted deployment gates.

The new failure mode is diffusion tail over-selection / diversity-selection-latency tradeoff.

## What WAM Claims Are Forbidden

Do not frame the project as imagined rollout selection, imagined-vs-real dynamics mismatch, world-action-model training, or WAM-based planning. The controlled sampler is a diffusion-like action sampler, not an imagined rollout model.

## What JEPA Claims Are Forbidden

Do not frame the project as latent prediction, latent score learning, representation-space planning, or latent-real rank distortion as the main object. Latent-real rank distortion is a JEPA-specific prior-project object, not the central claim here.

## What Diffusion-Specific Experiments Must Exist

The repository must include:

- controlled diffusion-like action sampler regimes for aligned, misaligned, collapsed, low-diversity, low-`K`, and high-`K` sampling;
- learned Diffusion Policy-lite denoising model with state conditioning and horizon action sequences;
- true action DDPM/DDIM artifact with clean-target denoiser held as an ablation;
- one lightweight robotics simulator benchmark path with actual sampled-trajectory rollout utility;
- scorer/reranker comparison showing that the value of `N` depends on score-utility alignment;
- `N` versus `K` phase diagram with latency-adjusted real utility;
- claim audit that checks the above artifacts.

## What Would Make This Project Fail As A Clone

The project fails as a clone if it can only be summarized as "same Best-of-N theorem, same experiments, different variable names." It must produce diffusion-specific artifacts about action trajectory diversity, denoising budget, reranker alignment, high-`N` bad-tail selection, and latency-adjusted selection.
