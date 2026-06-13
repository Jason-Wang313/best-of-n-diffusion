from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffusion_audit.action_ddpm import (
    diffusion_internal_scores,
    sample_consistency_trajectories,
    sample_ddim_trajectories,
    sample_ddpm_trajectories,
    train_epsilon_denoiser,
)
from diffusion_audit.benchmarks.pusht import (
    PUSHT_ACTION_HIGH,
    PUSHT_ACTION_LOW,
    PUSHT_ENV_ID,
    evaluate_pusht_pool,
    make_pusht_expert_dataset,
    pusht_behavior_cloning_score,
    pusht_misaligned_corner_score,
    pusht_obs_to_features,
    pusht_trajectory_features,
)
from diffusion_audit.diversity import diversity_summary, marginal_new_mode_discovery, trajectory_cluster_ids
from diffusion_audit.evaluation import curve_rows, evaluate_pool
from diffusion_audit.io import results_dir, write_json
from diffusion_audit.scorers import apply_linear_critic, fit_linear_value_critic, random_scores
from diffusion_audit.stats import bootstrap_mean_ci, mean_ci_columns, paired_high_minus_low_ci
from diffusion_audit.theory import utility_max_selection_finite


N_VALUES = [1, 2, 4, 8, 16, 32]
DEFAULT_K_VALUES = [1, 4, 8, 16]
ROLLOUT_METRIC_VALUE_COLS = [
    "exact_selected_max_coverage",
    "exact_selected_final_coverage",
    "exact_selected_success",
]
REGIMES = {
    "pusht_aligned": {"temperature": 0.95, "seed_offset": 31_000},
    "pusht_low_diversity": {"temperature": 0.10, "seed_offset": 32_000},
    "pusht_high_temp_misaligned": {"temperature": 1.30, "seed_offset": 33_000},
}


def reset_obs(seed: int) -> np.ndarray:
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    env = gym.make(PUSHT_ENV_ID, render_mode=None)
    try:
        obs, _ = env.reset(seed=int(seed))
    finally:
        env.close()
    return np.asarray(obs, dtype=np.float32)


def pusht_scores(policy, raw_obs: np.ndarray, trajectories: np.ndarray, utilities: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    features = pusht_trajectory_features(raw_obs, trajectories)
    rng = np.random.default_rng(seed)
    n_pilot = min(trajectories.shape[0], max(features.shape[1] + 1, trajectories.shape[0] // 3))
    pilot = rng.choice(np.arange(trajectories.shape[0]), size=n_pilot, replace=False)
    weights = fit_linear_value_critic(features[pilot], utilities[pilot], ridge=1e-3)
    value_scores = apply_linear_critic(features, weights)
    uncertainty = np.linalg.norm(features - np.mean(features[pilot], axis=0, keepdims=True), axis=1)
    obs_features = pusht_obs_to_features(raw_obs)
    return {
        "random_sample_selection": random_scores(trajectories.shape[0], seed=seed),
        "diffusion_internal_score": diffusion_internal_scores(policy, obs_features, trajectories, seed=seed + 7, probes=1),
        "behavior_cloning_critic": pusht_behavior_cloning_score(raw_obs, trajectories),
        "learned_value_critic_from_pilot_rollouts": value_scores,
        "uncertainty_aware_critic": value_scores - 0.08 * uncertainty,
        "misaligned_corner_scorer": pusht_misaligned_corner_score(trajectories, seed=seed + 17)
        - 1.8 * pusht_behavior_cloning_score(raw_obs, trajectories),
        "oracle_real_utility_selector": utilities,
    }


def selected_rollout_metric_curves(scores: np.ndarray, rollouts: list, n_values: list[int]) -> dict[str, dict[int, float]]:
    """Compute exact maximum-score selected rollout metrics for the score-ranked pool."""

    metric_arrays = {
        "exact_selected_max_coverage": np.asarray([item.max_coverage for item in rollouts], dtype=float),
        "exact_selected_final_coverage": np.asarray([item.final_coverage for item in rollouts], dtype=float),
        "exact_selected_success": np.asarray([float(item.success) for item in rollouts], dtype=float),
    }
    return {
        name: utility_max_selection_finite(scores, values, n_values)
        for name, values in metric_arrays.items()
    }


def effect_ci_table(curves: pd.DataFrame, n_values: list[int], value_cols: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    low_n = min(n_values)
    high_n = max(n_values)
    value_cols = value_cols or ["exact_selected_real", "exact_selected_score", "high_n_regret"]
    for group_key, group in curves.groupby(["sampler", "regime", "scorer", "K"]):
        sampler, regime, scorer, k = group_key
        for value_col in value_cols:
            rows.append(
                {
                    "sampler": sampler,
                    "regime": regime,
                    "scorer": scorer,
                    "K": int(k),
                    "metric": value_col,
                    **paired_high_minus_low_ci(
                        group,
                        unit_cols=["seed", "episode_idx"],
                        value_col=value_col,
                        low_n=low_n,
                        high_n=high_n,
                        seed=6100 + len(sampler) + len(regime) + len(scorer) + int(k),
                    ),
                }
            )
    return pd.DataFrame(rows)


def scorer_gap_ci(
    curves: pd.DataFrame,
    *,
    sampler: str,
    regime: str,
    better_scorer: str,
    worse_scorer: str,
    k: int,
    n: int,
    seed: int,
) -> dict:
    sub = curves[
        (curves["sampler"] == sampler)
        & (curves["regime"] == regime)
        & (curves["K"] == int(k))
        & (curves["N"] == int(n))
    ]
    pivot = sub.pivot_table(index=["seed", "episode_idx"], columns="scorer", values="exact_selected_real", aggfunc="mean")
    if better_scorer not in pivot.columns or worse_scorer not in pivot.columns:
        values = []
    else:
        values = (pivot[better_scorer] - pivot[worse_scorer]).dropna().to_numpy(dtype=float)
    ci = bootstrap_mean_ci(values, seed=seed)
    ci["effect"] = f"{better_scorer}_minus_{worse_scorer}"
    ci["sampler"] = sampler
    ci["regime"] = regime
    ci["K"] = int(k)
    ci["N"] = int(n)
    return ci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--train-states", type=int, default=10)
    parser.add_argument("--train-candidates", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--diffusion-steps", type=int, default=24)
    parser.add_argument("--mc-trials", type=int, default=40)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES))
    args = parser.parse_args()

    out_dir = results_dir()
    n_values = [n for n in N_VALUES if n <= args.candidates]
    k_values = sorted({int(k) for k in args.k_values if int(k) >= 1})
    rows: list[dict] = []
    diversity_rows: list[dict] = []
    runtime_rows: list[dict] = []
    rollout_rows: list[dict] = []
    training_rows: list[dict] = []

    for seed in args.seeds:
        train_obs, train_actions = make_pusht_expert_dataset(
            states=args.train_states,
            candidates_per_state=args.train_candidates,
            horizon=args.horizon,
            seed=seed,
        )
        policy, result = train_epsilon_denoiser(
            train_obs,
            train_actions,
            epochs=args.epochs,
            seed=seed + 700,
            diffusion_steps=args.diffusion_steps,
            hidden=96,
            lr=1.2e-3,
        )
        training_rows.append(
            {
                "seed": seed,
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "loss_ratio": result.final_loss / max(result.initial_loss, 1e-12),
                "epochs": result.epochs,
                "diffusion_steps": result.diffusion_steps,
                "target": result.target,
                "benchmark": "PushT",
            }
        )

        for regime in args.regimes:
            cfg = REGIMES[regime]
            for episode_idx in range(args.eval_episodes):
                env_seed = int(cfg["seed_offset"] + seed * 100 + episode_idx)
                raw_obs = reset_obs(env_seed)
                obs_features = pusht_obs_to_features(raw_obs)
                for k in k_values:
                    sampler_specs = [("ddim_eps", k)]
                    if k == min(k_values):
                        sampler_specs.append(("consistency_1step", 1))
                    if k == max(k_values):
                        sampler_specs.append(("ddpm_eps", k))
                    for sampler, sampler_k in sampler_specs:
                        sample_seed = seed * 100_000 + episode_idx * 1000 + sampler_k * 19 + len(regime) + len(sampler)
                        start = time.perf_counter()
                        if sampler == "ddim_eps":
                            trajectories = sample_ddim_trajectories(
                                policy,
                                obs_features,
                                n=args.candidates,
                                k=sampler_k,
                                seed=sample_seed,
                                temperature=cfg["temperature"],
                            )
                        elif sampler == "ddpm_eps":
                            trajectories = sample_ddpm_trajectories(
                                policy,
                                obs_features,
                                n=args.candidates,
                                k=sampler_k,
                                seed=sample_seed,
                                temperature=cfg["temperature"],
                            )
                        elif sampler == "consistency_1step":
                            trajectories = sample_consistency_trajectories(
                                policy,
                                obs_features,
                                n=args.candidates,
                                seed=sample_seed,
                                temperature=cfg["temperature"],
                            )
                        else:
                            raise ValueError(f"unknown sampler {sampler}")
                        sample_runtime = time.perf_counter() - start
                        trajectories = np.clip(trajectories, PUSHT_ACTION_LOW, PUSHT_ACTION_HIGH)

                        rollout_start = time.perf_counter()
                        utilities, rollouts = evaluate_pusht_pool(env_seed, trajectories)
                        rollout_runtime = time.perf_counter() - rollout_start
                        labels = trajectory_cluster_ids(trajectories)
                        div = diversity_summary(trajectories)
                        new_modes = marginal_new_mode_discovery(labels, n_values)
                        runtime_rows.append(
                            {
                                "seed": seed,
                                "episode_idx": episode_idx,
                                "env_seed": env_seed,
                                "regime": regime,
                                "sampler": sampler,
                                "K": sampler_k,
                                "candidates": args.candidates,
                                "sample_runtime_seconds": sample_runtime,
                                "rollout_runtime_seconds": rollout_runtime,
                                "runtime_per_candidate_ms": (sample_runtime + rollout_runtime) * 1000.0 / max(args.candidates, 1),
                            }
                        )
                        for cand_idx, rollout in enumerate(rollouts):
                            rollout_rows.append(
                                {
                                    "seed": seed,
                                    "episode_idx": episode_idx,
                                    "env_seed": env_seed,
                                    "regime": regime,
                                    "sampler": sampler,
                                    "K": sampler_k,
                                    "candidate_idx": cand_idx,
                                    "utility": rollout.utility,
                                    "max_coverage": rollout.max_coverage,
                                    "final_coverage": rollout.final_coverage,
                                    "success": rollout.success,
                                    "steps": rollout.steps,
                                    "runtime_seconds": rollout.runtime_seconds,
                                }
                            )
                        diversity_rows.append(
                            {
                                "seed": seed,
                                "episode_idx": episode_idx,
                                "env_seed": env_seed,
                                "regime": regime,
                                "sampler": sampler,
                                "K": sampler_k,
                                "temperature": cfg["temperature"],
                                "new_modes_at_high_n": new_modes[max(n_values)],
                                **div,
                            }
                        )

                        scores_by_name = pusht_scores(policy, raw_obs, trajectories, utilities, seed=sample_seed)
                        for scorer, scores in scores_by_name.items():
                            payload = evaluate_pool(
                                scores,
                                utilities,
                                n_values,
                                mc_trials=args.mc_trials,
                                seed=sample_seed + len(scorer),
                            )
                            rollout_curves = selected_rollout_metric_curves(scores, rollouts, n_values)
                            curve_payload = curve_rows(
                                family="E_pusht_benchmark",
                                regime=regime,
                                scorer=scorer,
                                seed=seed,
                                eval_payload=payload,
                                extra={
                                    "episode_idx": episode_idx,
                                    "env_seed": env_seed,
                                    "sampler": sampler,
                                    "K": sampler_k,
                                    "temperature": cfg["temperature"],
                                    "sample_runtime_seconds": sample_runtime,
                                    "rollout_runtime_seconds": rollout_runtime,
                                    "runtime_per_candidate_ms": (sample_runtime + rollout_runtime) * 1000.0 / max(args.candidates, 1),
                                    **div,
                                    "new_modes_at_high_n": new_modes[max(n_values)],
                                },
                            )
                            for row in curve_payload:
                                n = int(row["N"])
                                for metric_col, metric_curve in rollout_curves.items():
                                    row[metric_col] = float(metric_curve[n])
                            rows.extend(curve_payload)

    curves = pd.DataFrame(rows)
    diversity = pd.DataFrame(diversity_rows)
    runtime = pd.DataFrame(runtime_rows)
    rollouts = pd.DataFrame(rollout_rows)
    training = pd.DataFrame(training_rows)
    numeric = [
        "exact_selected_real",
        "exact_selected_score",
        "mc_selected_real",
        "oracle_selected_real",
        "score_utility_correlation",
        "tail_rank_correlation",
        "top_score_tail_real_utility",
        "high_n_regret",
        "real_change_high_minus_low",
        "score_change_high_minus_low",
        "sample_runtime_seconds",
        "rollout_runtime_seconds",
        "runtime_per_candidate_ms",
        "mean_pairwise_distance",
        "effective_sample_diversity",
        "duplicate_collapse_rate",
        "trajectory_cluster_count",
        "trajectory_cluster_entropy",
        "new_modes_at_high_n",
        *ROLLOUT_METRIC_VALUE_COLS,
    ]
    seed_agg = curves.groupby(["seed", "sampler", "regime", "scorer", "N", "K"], as_index=False)[numeric].mean()
    agg = mean_ci_columns(
        curves,
        group_cols=["sampler", "regime", "scorer", "N", "K"],
        numeric_cols=numeric,
        seed=7100,
    )
    effect_cis = effect_ci_table(
        curves,
        n_values,
        ["exact_selected_real", "exact_selected_score", "high_n_regret", *ROLLOUT_METRIC_VALUE_COLS],
    )
    rollout_metric_seed_agg = seed_agg[
        ["seed", "sampler", "regime", "scorer", "N", "K", "exact_selected_real", *ROLLOUT_METRIC_VALUE_COLS]
    ].copy()
    rollout_metric_agg = mean_ci_columns(
        curves,
        group_cols=["sampler", "regime", "scorer", "N", "K"],
        numeric_cols=["exact_selected_real", *ROLLOUT_METRIC_VALUE_COLS],
        seed=7150,
    )
    rollout_metric_effect_cis = effect_cis[effect_cis["metric"].isin(ROLLOUT_METRIC_VALUE_COLS)].copy()
    high_n = max(n_values)
    key_k = max(k_values)
    gap_df = pd.DataFrame(
        [
            scorer_gap_ci(
                curves,
                sampler="ddim_eps",
                regime="pusht_high_temp_misaligned",
                better_scorer="oracle_real_utility_selector",
                worse_scorer="misaligned_corner_scorer",
                k=key_k,
                n=high_n,
                seed=7200,
            ),
            scorer_gap_ci(
                curves,
                sampler="ddim_eps",
                regime="pusht_high_temp_misaligned",
                better_scorer="learned_value_critic_from_pilot_rollouts",
                worse_scorer="misaligned_corner_scorer",
                k=key_k,
                n=high_n,
                seed=7300,
            ),
        ]
    )

    curves.to_csv(out_dir / "tables" / "pusht_curves.csv", index=False)
    seed_agg.to_csv(out_dir / "tables" / "pusht_seed_aggregate.csv", index=False)
    agg.to_csv(out_dir / "tables" / "pusht_aggregate.csv", index=False)
    effect_cis.to_csv(out_dir / "tables" / "pusht_effect_cis.csv", index=False)
    gap_df.to_csv(out_dir / "tables" / "pusht_scorer_gap_cis.csv", index=False)
    diversity.to_csv(out_dir / "tables" / "pusht_diversity.csv", index=False)
    runtime.to_csv(out_dir / "tables" / "pusht_runtime.csv", index=False)
    rollouts.to_csv(out_dir / "tables" / "pusht_rollouts.csv", index=False)
    training.to_csv(out_dir / "tables" / "pusht_training.csv", index=False)
    rollout_metric_seed_agg.to_csv(out_dir / "tables" / "pusht_rollout_metric_seed_aggregate.csv", index=False)
    rollout_metric_agg.to_csv(out_dir / "tables" / "pusht_rollout_metric_aggregate.csv", index=False)
    rollout_metric_effect_cis.to_csv(out_dir / "tables" / "pusht_rollout_metric_effect_cis.csv", index=False)

    def agg_value(sampler: str, regime: str, scorer: str, k: int, n: int, col: str) -> float:
        part = agg[
            (agg["sampler"] == sampler)
            & (agg["regime"] == regime)
            & (agg["scorer"] == scorer)
            & (agg["K"] == int(k))
            & (agg["N"] == int(n))
        ]
        return float(part.iloc[0][col]) if len(part) else float("nan")

    low_n = min(n_values)
    aligned_gain = agg_value("ddim_eps", "pusht_aligned", "oracle_real_utility_selector", key_k, high_n, "exact_selected_real") - agg_value(
        "ddim_eps", "pusht_aligned", "oracle_real_utility_selector", key_k, low_n, "exact_selected_real"
    )
    low_div_gain = agg_value("ddim_eps", "pusht_low_diversity", "oracle_real_utility_selector", key_k, high_n, "exact_selected_real") - agg_value(
        "ddim_eps", "pusht_low_diversity", "oracle_real_utility_selector", key_k, low_n, "exact_selected_real"
    )
    misaligned_change = agg_value("ddim_eps", "pusht_high_temp_misaligned", "misaligned_corner_scorer", key_k, high_n, "exact_selected_real") - agg_value(
        "ddim_eps", "pusht_high_temp_misaligned", "misaligned_corner_scorer", key_k, low_n, "exact_selected_real"
    )
    oracle_gap = agg_value("ddim_eps", "pusht_high_temp_misaligned", "oracle_real_utility_selector", key_k, high_n, "exact_selected_real") - agg_value(
        "ddim_eps", "pusht_high_temp_misaligned", "misaligned_corner_scorer", key_k, high_n, "exact_selected_real"
    )
    value_gap = agg_value("ddim_eps", "pusht_high_temp_misaligned", "learned_value_critic_from_pilot_rollouts", key_k, high_n, "exact_selected_real") - agg_value(
        "ddim_eps", "pusht_high_temp_misaligned", "misaligned_corner_scorer", key_k, high_n, "exact_selected_real"
    )
    aligned_max_coverage_gain = agg_value("ddim_eps", "pusht_aligned", "oracle_real_utility_selector", key_k, high_n, "exact_selected_max_coverage") - agg_value(
        "ddim_eps", "pusht_aligned", "oracle_real_utility_selector", key_k, low_n, "exact_selected_max_coverage"
    )
    aligned_final_coverage_gain = agg_value("ddim_eps", "pusht_aligned", "oracle_real_utility_selector", key_k, high_n, "exact_selected_final_coverage") - agg_value(
        "ddim_eps", "pusht_aligned", "oracle_real_utility_selector", key_k, low_n, "exact_selected_final_coverage"
    )
    aligned_success_gain = agg_value("ddim_eps", "pusht_aligned", "oracle_real_utility_selector", key_k, high_n, "exact_selected_success") - agg_value(
        "ddim_eps", "pusht_aligned", "oracle_real_utility_selector", key_k, low_n, "exact_selected_success"
    )
    summary = {
        "artifact_tables": {
            "curves": "results/tables/pusht_curves.csv",
            "seed_aggregate": "results/tables/pusht_seed_aggregate.csv",
            "aggregate": "results/tables/pusht_aggregate.csv",
            "effect_cis": "results/tables/pusht_effect_cis.csv",
            "scorer_gap_cis": "results/tables/pusht_scorer_gap_cis.csv",
            "diversity": "results/tables/pusht_diversity.csv",
            "runtime": "results/tables/pusht_runtime.csv",
            "rollouts": "results/tables/pusht_rollouts.csv",
            "training": "results/tables/pusht_training.csv",
            "rollout_metric_seed_aggregate": "results/tables/pusht_rollout_metric_seed_aggregate.csv",
            "rollout_metric_aggregate": "results/tables/pusht_rollout_metric_aggregate.csv",
            "rollout_metric_effect_cis": "results/tables/pusht_rollout_metric_effect_cis.csv",
        },
        "benchmark": "PushT",
        "env_id": PUSHT_ENV_ID,
        "actual_simulator_rollouts": True,
        "actual_rollout_metric_curves": True,
        "heuristic_demonstrations_for_training": True,
        "diffusion_policy_validity_checklist": {
            "true_epsilon_prediction": True,
            "ddim_fast_sampling": True,
            "stochastic_ddpm_sampling": True,
            "actual_environment_rollout_utility": True,
            "trajectory_reranking_over_sampled_actions": True,
        },
        "sampler_families": sorted(curves["sampler"].unique().tolist()),
        "scorers": sorted(curves["scorer"].unique().tolist()),
        "regimes": sorted(args.regimes),
        "n_values": n_values,
        "k_values": k_values,
        "num_training_seeds": int(training["seed"].nunique()),
        "loss_decreased_all_seeds": bool((training["final_loss"] < training["initial_loss"]).all()),
        "max_loss_ratio": float(training["loss_ratio"].max()),
        "pusht_aligned_oracle_gain_high_minus_low": float(aligned_gain),
        "pusht_low_diversity_oracle_gain_high_minus_low": float(low_div_gain),
        "pusht_misaligned_real_change_high_minus_low": float(misaligned_change),
        "pusht_oracle_minus_misaligned_high_n": float(oracle_gap),
        "pusht_value_minus_misaligned_high_n": float(value_gap),
        "pusht_aligned_max_coverage_gain_high_minus_low": float(aligned_max_coverage_gain),
        "pusht_aligned_final_coverage_gain_high_minus_low": float(aligned_final_coverage_gain),
        "pusht_aligned_success_gain_high_minus_low": float(aligned_success_gain),
        "rollout_metric_columns": ROLLOUT_METRIC_VALUE_COLS,
        "rollout_metric_effect_rows": int(len(rollout_metric_effect_cis)),
        "rollout_metric_seed_rows": int(len(rollout_metric_seed_agg)),
        "runtime_rows": int(len(runtime)),
        "rollout_rows": int(len(rollouts)),
        "mean_candidate_rollout_coverage": float(rollouts["max_coverage"].mean()) if len(rollouts) else float("nan"),
        "mean_candidate_rollout_success": float(rollouts["success"].astype(float).mean()) if len(rollouts) else float("nan"),
        "measured_wall_clock_runtime": True,
    }
    write_json(out_dir / "pusht_summary.json", summary)

    plot_specs = [
        ("pusht_aligned", "oracle_real_utility_selector", "aligned oracle"),
        ("pusht_low_diversity", "oracle_real_utility_selector", "low-div oracle"),
        ("pusht_high_temp_misaligned", "oracle_real_utility_selector", "hot oracle"),
        ("pusht_high_temp_misaligned", "misaligned_corner_scorer", "hot misaligned"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharex=True)
    metric_specs = [
        ("exact_selected_real", "rollout utility"),
        ("exact_selected_max_coverage", "max coverage"),
        ("exact_selected_success", "success probability"),
    ]
    for ax, (metric_col, ylabel) in zip(axes, metric_specs, strict=True):
        for regime, scorer, label in plot_specs:
            part = agg[
                (agg["sampler"] == "ddim_eps")
                & (agg["K"] == key_k)
                & (agg["regime"] == regime)
                & (agg["scorer"] == scorer)
            ].sort_values("N")
            if len(part):
                ax.plot(part["N"], part[metric_col], marker="o", linewidth=1.4, label=label)
        ax.axvline(1, color="0.55", linestyle=":", linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18, linewidth=0.6)
    axes[0].set_title("utility")
    axes[1].set_title("coverage")
    axes[2].set_title("success")
    axes[0].legend(fontsize=6.4, ncol=1)
    fig.suptitle("PushT benchmark: selected rollout outcomes versus sample count", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "pusht_max_selection.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
