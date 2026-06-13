# Limitations

The experiments are CPU-light. They are designed to isolate trajectory-selection laws and diagnostics, not to establish real-robot performance.

The controlled diffusion-like sampler is not a full Diffusion Policy. It is labeled separately so that hand-designed sampling results are not confused with learned denoising-policy results.

The learned Diffusion Policy-lite model is intentionally small. Its role is to test whether the diagnostic pipeline applies to a learned noise-to-action generator, not to claim production-scale robot policy quality.

The image-conditioned variant uses 32x32 toy renderings and a tiny CNN. It is evidence that the diagnostic pipeline can handle image observations on CPU, not evidence for full visual manipulation benchmarks.

The true action diffusion tier is a faithful epsilon-prediction/DDIM/DDPM action sampler, but it is still trained on a small toy manipulation dataset. The PushT tier is a simulator benchmark path with actual rollout utility and coverage/success reporting, but uses low-dimensional observations and heuristic demonstrations rather than full visual imitation learning.

Calibration is not guaranteed to repair every scorer. The claim audit only promotes calibration repair when held-out lower-bound gates pass in at least one regime and also records random-score, drift, or OOD-style regimes without strong repair.

Audit-Then-Sample is an inference-time controller for audited CPU simulation regimes. It is not a production deployment rule, a hardware safety layer, or evidence that high-`N` selection is safe without fresh diversity, tail-alignment, runtime, and negative-control audits. Its "near-perfect" target is prevention of unsupported or harmful high-`N` admission in the audited stress regimes, not universal performance improvement.

No controller can certify high-`N` utility from scores alone when real utility is unknown and no valid utility bound is available. In that case the correct output is abstention: audit more rollouts, calibrate the scorer, increase diversity, stop early, or block high `N`.

A revision or deployment claim should be considered falsified if a harmful negative-control row is admitted as `increase_N`, if an admitted row has a nonpositive utility, tail, or latency-adjusted lower bound, if true-DDPM or PushT critical CI rows fall below their evidence thresholds, or if the claim audit reports unsupported global diffusion-policy wording.
