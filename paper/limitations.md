# Limitations

The experiments are CPU-light. They are designed to isolate Best-of-N selection laws and diagnostics, not to establish real-robot performance.

The controlled diffusion-like sampler is not a full Diffusion Policy. It is labeled separately so that hand-designed sampling results are not confused with learned denoising-policy results.

The learned Diffusion Policy-lite model is intentionally small. Its role is to test whether the diagnostic pipeline applies to a learned noise-to-action generator, not to claim production-scale robot policy quality.

The image-conditioned variant uses 32x32 toy renderings and a tiny CNN. It is evidence that the diagnostic pipeline can handle image observations on CPU, not evidence for full visual manipulation benchmarks.

The true action diffusion tier is a faithful epsilon-prediction/DDIM/DDPM action sampler, but it is still trained on a small toy manipulation dataset. The PushT tier is a simulator benchmark path with actual rollout utility and coverage/success reporting, but uses low-dimensional observations and heuristic demonstrations rather than full visual imitation learning.

Calibration is not guaranteed to repair every scorer. The claim audit only promotes calibration repair when a generated artifact shows improvement in at least one regime and also records regimes without strong repair.
