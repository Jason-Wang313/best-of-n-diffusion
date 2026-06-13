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
from diffusion_audit.diffusion_lite import make_expert_dataset, sample_denoised_trajectories, train_denoiser
from diffusion_audit.diversity import (
    diversity_summary,
    marginal_new_mode_discovery,
    trajectory_cluster_ids,
)
from diffusion_audit.evaluation import curve_rows, evaluate_pool
from diffusion_audit.io import results_dir, write_json
from diffusion_audit.scorers import (
    anti_correlated_scores,
    apply_linear_critic,
    behavior_cloning_critic,
    calibrated_critic,
    diffusion_likelihood_proxy,
    ensemble_value_critic,
    fit_linear_value_critic,
    oracle_scores,
    random_scores,
    tail_only_misaligned_scores,
    trajectory_features,
    uncertainty_aware_critic,
    weakly_aligned_scores,
)
from diffusion_audit.stats import bootstrap_mean_ci, mean_ci_columns, paired_high_minus_low_ci
from diffusion_audit.toy_control import make_observations, trajectory_utilities


N_VALUES = [1, 2, 4, 8, 16, 32]
DEFAULT_K_VALUES = [1, 2, 4, 8, 16, 32]
REGIMES = {
    "id": {"ood": "id", "temperature": 0.95},
    "low_diversity": {"ood": "id", "temperature": 0.12},
    "hidden_obstacle": {"ood": "hidden_obstacle", "temperature": 1.20},
    "changed_friction": {"ood": "changed_friction", "temperature": 0.95},
    "changed_mass": {"ood": "changed_mass", "temperature": 0.95},
    "action_noise": {"ood": "action_noise", "temperature": 1.05},
    "shifted_goal": {"ood": "shifted_goal", "temperature": 1.00},
}


def true_diffusion_scorers(
    policy,
    obs,
    trajectories: np.ndarray,
    utilities: np.ndarray,
    seed: int,
    include_extended: bool,
) -> dict[str, np.ndarray]:
    features = trajectory_features(obs, trajectories)
    rng = np.random.default_rng(seed)
    n_pilot = min(trajectories.shape[0], max(features.shape[1] + 1, trajectories.shape[0] // 4))
    pilot = rng.choice(np.arange(trajectories.shape[0]), size=n_pilot, replace=False)
    pilot_weights = fit_linear_value_critic(features[pilot], utilities[pilot], ridge=1e-3)
    scores = {
        "random_sample_selection": random_scores(trajectories.shape[0], seed=seed),
        "diffusion_internal_score": diffusion_internal_scores(policy, obs.as_array(), trajectories, seed=seed + 3, probes=1),
        "diffusion_likelihood_proxy": diffusion_likelihood_proxy(obs, trajectories),
        "behavior_cloning_critic": behavior_cloning_critic(obs, trajectories),
        "learned_value_critic_from_pilot_rollouts": apply_linear_critic(features, pilot_weights),
        "calibrated_critic": calibrated_critic(obs, trajectories),
        "weakly_aligned_score": weakly_aligned_scores(obs, trajectories, seed=seed + 31),
        "tail_only_misaligned_score": tail_only_misaligned_scores(obs, trajectories, seed=seed + 37),
        "anti_correlated_score": anti_correlated_scores(obs, trajectories, seed=seed + 41),
        "oracle_real_utility_selector": oracle_scores(obs, trajectories),
    }
    if include_extended:
        ensemble_mean, ensemble_uncertainty = ensemble_value_critic(features, utilities, seed=seed + 17)
        scores["ensemble_value_critic"] = ensemble_mean
        scores["uncertainty_aware_critic"] = uncertainty_aware_critic(obs, trajectories, seed=seed + 29)
        scores["ensemble_uncertainty_negative_control"] = -ensemble_uncertainty
    return scores


def effect_ci_table(curves: pd.DataFrame, n_values: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    low_n = min(n_values)
    high_n = max(n_values)
    group_cols = ["sampler", "regime", "scorer", "K"]
    for group_key, group in curves.groupby(group_cols):
        sampler, regime, scorer, k = group_key
        for value_col in ["exact_selected_real", "exact_selected_score", "high_n_regret"]:
            rows.append(
                {
                    "sampler": sampler,
                    "regime": regime,
                    "scorer": scorer,
                    "K": int(k),
                    "metric": value_col,
                    **paired_high_minus_low_ci(
                        group,
                        unit_cols=["seed", "state_idx"],
                        value_col=value_col,
                        low_n=low_n,
                        high_n=high_n,
                        seed=4400 + len(sampler) + len(regime) + len(scorer) + int(k),
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
    pivot = sub.pivot_table(index=["seed", "state_idx"], columns="scorer", values="exact_selected_real", aggfunc="mean")
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--train-states", type=int, default=14)
    parser.add_argument("--train-candidates", type=int, default=8)
    parser.add_argument("--eval-states", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--mc-trials", type=int, default=80)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES))
    parser.add_argument("--extended-scorers", action="store_true")
    args = parser.parse_args()

    out_dir = results_dir()
    n_values = [n for n in N_VALUES if n <= args.candidates]
    k_values = sorted({int(k) for k in args.k_values if int(k) >= 1})
    rows: list[dict] = []
    diversity_rows: list[dict] = []
    runtime_rows: list[dict] = []
    training_rows: list[dict] = []

    for seed in args.seeds:
        train_obs, train_actions = make_expert_dataset(
            states=args.train_states,
            candidates_per_state=args.train_candidates,
            horizon=args.horizon,
            seed=seed,
            multimodal=True,
        )
        policy, eps_result = train_epsilon_denoiser(
            train_obs,
            train_actions,
            epochs=args.epochs,
            seed=seed,
            diffusion_steps=args.diffusion_steps,
            hidden=80,
        )
        training_rows.append(
            {
                "seed": seed,
                "model": "epsilon_ddpm",
                "initial_loss": eps_result.initial_loss,
                "final_loss": eps_result.final_loss,
                "loss_ratio": eps_result.final_loss / max(eps_result.initial_loss, 1e-12),
                "epochs": eps_result.epochs,
                "diffusion_steps": eps_result.diffusion_steps,
                "target": eps_result.target,
            }
        )
        clean_model, clean_result = train_denoiser(
            train_obs,
            train_actions,
            epochs=max(6, args.epochs // 2),
            seed=10_000 + seed,
            lr=2.0e-3,
        )
        training_rows.append(
            {
                "seed": seed,
                "model": "clean_target_ablation",
                "initial_loss": clean_result.initial_loss,
                "final_loss": clean_result.final_loss,
                "loss_ratio": clean_result.final_loss / max(clean_result.initial_loss, 1e-12),
                "epochs": clean_result.epochs,
                "diffusion_steps": 0,
                "target": "clean_action",
            }
        )

        for regime in args.regimes:
            cfg = REGIMES[regime]
            observations = make_observations(args.eval_states, seed=6200 + seed, ood=cfg["ood"])
            for state_idx, obs in enumerate(observations):
                for k in k_values:
                    sampler_specs = [
                        ("ddim_eps", k),
                        ("clean_target_ablation", k),
                    ]
                    if k >= 4:
                        sampler_specs.append(("ddpm_eps", k))
                    if k == min(k_values):
                        sampler_specs.append(("consistency_1step", 1))

                    for sampler, sampler_k in sampler_specs:
                        sample_seed = seed * 100_000 + state_idx * 1000 + sampler_k * 17 + len(sampler) + len(regime)
                        start = time.perf_counter()
                        if sampler == "ddim_eps":
                            trajectories = sample_ddim_trajectories(
                                policy,
                                obs.as_array(),
                                n=args.candidates,
                                k=sampler_k,
                                temperature=cfg["temperature"],
                                seed=sample_seed,
                            )
                        elif sampler == "ddpm_eps":
                            trajectories = sample_ddpm_trajectories(
                                policy,
                                obs.as_array(),
                                n=args.candidates,
                                k=sampler_k,
                                temperature=cfg["temperature"],
                                seed=sample_seed,
                            )
                        elif sampler == "consistency_1step":
                            trajectories = sample_consistency_trajectories(
                                policy,
                                obs.as_array(),
                                n=args.candidates,
                                temperature=cfg["temperature"],
                                seed=sample_seed,
                            )
                        elif sampler == "clean_target_ablation":
                            trajectories = sample_denoised_trajectories(
                                clean_model,
                                obs,
                                n=args.candidates,
                                k=sampler_k,
                                temperature=cfg["temperature"],
                                seed=sample_seed,
                            )
                        else:
                            raise ValueError(f"unknown sampler {sampler}")
                        elapsed = time.perf_counter() - start
                        utilities = trajectory_utilities(obs, trajectories)
                        labels = trajectory_cluster_ids(trajectories)
                        div = diversity_summary(trajectories)
                        new_modes = marginal_new_mode_discovery(labels, n_values)
                        runtime_rows.append(
                            {
                                "seed": seed,
                                "state_idx": state_idx,
                                "regime": regime,
                                "sampler": sampler,
                                "K": sampler_k,
                                "candidates": args.candidates,
                                "runtime_seconds": elapsed,
                                "runtime_per_candidate_ms": elapsed * 1000.0 / max(args.candidates, 1),
                                "mean_pairwise_distance": div["mean_pairwise_distance"],
                                "effective_sample_diversity": div["effective_sample_diversity"],
                            }
                        )
                        diversity_rows.append(
                            {
                                "seed": seed,
                                "state_idx": state_idx,
                                "regime": regime,
                                "sampler": sampler,
                                "K": sampler_k,
                                "temperature": cfg["temperature"],
                                "new_modes_at_high_n": new_modes[max(n_values)],
                                **div,
                            }
                        )
                        scores_by_name = true_diffusion_scorers(
                            policy,
                            obs,
                            trajectories,
                            utilities,
                            seed=sample_seed,
                            include_extended=args.extended_scorers,
                        )
                        for scorer, scores in scores_by_name.items():
                            payload = evaluate_pool(
                                scores,
                                utilities,
                                n_values,
                                mc_trials=args.mc_trials,
                                seed=sample_seed + len(scorer),
                            )
                            rows.extend(
                                curve_rows(
                                    family="D_true_action_diffusion",
                                    regime=regime,
                                    scorer=scorer,
                                    seed=seed,
                                    eval_payload=payload,
                                    extra={
                                        "state_idx": state_idx,
                                        "sampler": sampler,
                                        "K": sampler_k,
                                        "temperature": cfg["temperature"],
                                        "runtime_seconds": elapsed,
                                        "runtime_per_candidate_ms": elapsed * 1000.0 / max(args.candidates, 1),
                                        **div,
                                        "new_modes_at_high_n": new_modes[max(n_values)],
                                    },
                                )
                            )

    curves = pd.DataFrame(rows)
    diversity = pd.DataFrame(diversity_rows)
    runtime = pd.DataFrame(runtime_rows)
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
        "runtime_seconds",
        "runtime_per_candidate_ms",
        "mean_pairwise_distance",
        "effective_sample_diversity",
        "duplicate_collapse_rate",
        "trajectory_cluster_count",
        "trajectory_cluster_entropy",
        "new_modes_at_high_n",
    ]
    seed_agg = curves.groupby(["seed", "sampler", "regime", "scorer", "N", "K"], as_index=False)[numeric].mean()
    agg = mean_ci_columns(
        curves,
        group_cols=["sampler", "regime", "scorer", "N", "K"],
        numeric_cols=numeric,
        seed=5100,
    )
    effect_cis = effect_ci_table(curves, n_values)
    high_n = max(n_values)
    gap_rows = [
        scorer_gap_ci(
            curves,
            sampler="ddim_eps",
            regime="hidden_obstacle",
            better_scorer="oracle_real_utility_selector",
            worse_scorer="tail_only_misaligned_score",
            k=max(k_values),
            n=high_n,
            seed=5200,
        ),
        scorer_gap_ci(
            curves,
            sampler="ddim_eps",
            regime="hidden_obstacle",
            better_scorer="calibrated_critic",
            worse_scorer="tail_only_misaligned_score",
            k=max(k_values),
            n=high_n,
            seed=5300,
        ),
    ]
    gap_df = pd.DataFrame(gap_rows)
    runtime_summary = runtime.groupby(["sampler", "K"], as_index=False).agg(
        runtime_per_candidate_ms=("runtime_per_candidate_ms", "mean"),
        runtime_rows=("runtime_per_candidate_ms", "size"),
        mean_pairwise_distance=("mean_pairwise_distance", "mean"),
        effective_sample_diversity=("effective_sample_diversity", "mean"),
    )
    sampler_comparison = effect_cis[
        (effect_cis["regime"] == "id")
        & (effect_cis["scorer"] == "oracle_real_utility_selector")
        & (effect_cis["metric"] == "exact_selected_real")
    ].merge(runtime_summary, on=["sampler", "K"], how="left")
    sampler_comparison["sampler_role"] = np.where(
        sampler_comparison["sampler"].eq("clean_target_ablation"),
        "ablation",
        "primary",
    )

    curves.to_csv(out_dir / "tables" / "true_diffusion_curves.csv", index=False)
    seed_agg.to_csv(out_dir / "tables" / "true_diffusion_seed_aggregate.csv", index=False)
    agg.to_csv(out_dir / "tables" / "true_diffusion_aggregate.csv", index=False)
    effect_cis.to_csv(out_dir / "tables" / "true_diffusion_effect_cis.csv", index=False)
    gap_df.to_csv(out_dir / "tables" / "true_diffusion_scorer_gap_cis.csv", index=False)
    diversity.to_csv(out_dir / "tables" / "true_diffusion_diversity.csv", index=False)
    runtime.to_csv(out_dir / "tables" / "true_diffusion_runtime.csv", index=False)
    training.to_csv(out_dir / "tables" / "true_diffusion_training.csv", index=False)
    sampler_comparison.to_csv(out_dir / "tables" / "true_diffusion_sampler_comparison.csv", index=False)

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
    key_k = max(k_values)
    ddim_oracle_gain = agg_value("ddim_eps", "id", "oracle_real_utility_selector", key_k, high_n, "exact_selected_real") - agg_value(
        "ddim_eps", "id", "oracle_real_utility_selector", key_k, low_n, "exact_selected_real"
    )
    ddpm_oracle_gain = agg_value("ddpm_eps", "id", "oracle_real_utility_selector", key_k, high_n, "exact_selected_real") - agg_value(
        "ddpm_eps", "id", "oracle_real_utility_selector", key_k, low_n, "exact_selected_real"
    )
    low_div_gain = agg_value("ddim_eps", "low_diversity", "oracle_real_utility_selector", key_k, high_n, "exact_selected_real") - agg_value(
        "ddim_eps", "low_diversity", "oracle_real_utility_selector", key_k, low_n, "exact_selected_real"
    )
    hidden_tail_change = agg_value("ddim_eps", "hidden_obstacle", "tail_only_misaligned_score", key_k, high_n, "exact_selected_real") - agg_value(
        "ddim_eps", "hidden_obstacle", "tail_only_misaligned_score", key_k, low_n, "exact_selected_real"
    )
    anti_change = agg_value("ddim_eps", "id", "anti_correlated_score", key_k, high_n, "exact_selected_real") - agg_value(
        "ddim_eps", "id", "anti_correlated_score", key_k, low_n, "exact_selected_real"
    )
    best_runtime = runtime.groupby(["sampler", "K"], as_index=False)["runtime_per_candidate_ms"].mean()
    fastest = best_runtime.iloc[int(np.argmin(best_runtime["runtime_per_candidate_ms"].to_numpy()))].to_dict()
    slowest = best_runtime.iloc[int(np.argmax(best_runtime["runtime_per_candidate_ms"].to_numpy()))].to_dict()

    summary = {
        "artifact_tables": {
            "curves": "results/tables/true_diffusion_curves.csv",
            "seed_aggregate": "results/tables/true_diffusion_seed_aggregate.csv",
            "aggregate": "results/tables/true_diffusion_aggregate.csv",
            "effect_cis": "results/tables/true_diffusion_effect_cis.csv",
            "scorer_gap_cis": "results/tables/true_diffusion_scorer_gap_cis.csv",
            "diversity": "results/tables/true_diffusion_diversity.csv",
            "runtime": "results/tables/true_diffusion_runtime.csv",
            "training": "results/tables/true_diffusion_training.csv",
            "sampler_comparison": "results/tables/true_diffusion_sampler_comparison.csv",
        },
        "diffusion_policy_validity_checklist": {
            "true_epsilon_prediction": True,
            "ddim_fast_sampling": True,
            "stochastic_ddpm_sampling": True,
            "one_step_consistency_variant": True,
            "clean_target_ablation_kept": True,
            "action_sequence_generation": True,
            "same_total_compute_budget_grid": True,
        },
        "primary_samplers": ["ddim_eps", "ddpm_eps", "consistency_1step"],
        "ablation_samplers": ["clean_target_ablation"],
        "sampler_families": sorted(curves["sampler"].unique().tolist()),
        "scorers": sorted(curves["scorer"].unique().tolist()),
        "regimes": sorted(args.regimes),
        "n_values": n_values,
        "k_values": k_values,
        "num_training_seeds": int(training["seed"].nunique()),
        "loss_decreased_all_epsilon_seeds": bool(
            (
                training[training["model"] == "epsilon_ddpm"]["final_loss"]
                < training[training["model"] == "epsilon_ddpm"]["initial_loss"]
            ).all()
        ),
        "epsilon_ddpm_max_loss_ratio": float(training[training["model"] == "epsilon_ddpm"]["loss_ratio"].max()),
        "ddim_oracle_gain_high_minus_low": float(ddim_oracle_gain),
        "ddpm_oracle_gain_high_minus_low": float(ddpm_oracle_gain),
        "low_diversity_oracle_gain_high_minus_low": float(low_div_gain),
        "hidden_tail_misaligned_real_change_high_minus_low": float(hidden_tail_change),
        "anti_correlated_real_change_high_minus_low": float(anti_change),
        "runtime_rows": int(len(runtime)),
        "sampler_comparison_rows": int(len(sampler_comparison)),
        "fastest_sampler_k": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in fastest.items()},
        "slowest_sampler_k": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in slowest.items()},
        "measured_wall_clock_runtime": True,
    }
    write_json(out_dir / "true_diffusion_summary.json", summary)

    subset = agg[
        (agg["sampler"].isin(["ddim_eps", "ddpm_eps", "clean_target_ablation"]))
        & (agg["regime"].isin(["id", "hidden_obstacle"]))
        & (agg["scorer"].isin(["oracle_real_utility_selector", "tail_only_misaligned_score"]))
        & (agg["K"] == key_k)
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for (sampler, regime, scorer), part in subset.groupby(["sampler", "regime", "scorer"]):
        label = f"{sampler}:{regime}:{scorer.replace('_', ' ')[:18]}"
        ax.plot(part["N"], part["exact_selected_real"], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N sampled action trajectories")
    ax.set_ylabel("Exact selected real utility")
    ax.set_title("Faithful action diffusion: trajectory search survives and fails by scorer tail")
    ax.legend(fontsize=5.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "true_diffusion_survival.png", dpi=160)
    plt.close(fig)

    pivot = best_runtime.pivot(index="sampler", columns="K", values="runtime_per_candidate_ms")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("K denoising steps")
    ax.set_title("Measured sampler runtime per candidate")
    fig.colorbar(image, ax=ax, label="ms / trajectory")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "true_diffusion_runtime.png", dpi=160)
    plt.close(fig)

    comp = sampler_comparison.sort_values(["sampler_role", "sampler", "K"]).copy()
    labels = [f"{row.sampler}\nK={int(row.K)}" for row in comp.itertuples()]
    colors = ["#3d6fb6" if role == "primary" else "#9a9a9a" for role in comp["sampler_role"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    x = np.arange(len(comp))
    axes[0].bar(x, comp["mean"].to_numpy(dtype=float), color=colors)
    axes[0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].set_ylabel("max-selection utility gain")
    axes[0].set_title("oracle high-N effect")
    axes[1].bar(x, comp["runtime_per_candidate_ms"].to_numpy(dtype=float), color=colors)
    axes[1].set_ylabel("ms / trajectory")
    axes[1].set_title("measured sampling runtime")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.suptitle("True action diffusion sampler comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "true_diffusion_sampler_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
