from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffusion_audit.diffusion_lite import (
    IMAGE_SIZE,
    make_expert_dataset,
    make_image_expert_dataset,
    receding_horizon_utility,
    receding_horizon_utility_image,
    render_observation_image,
    sample_denoised_trajectories,
    sample_image_denoised_trajectories,
    train_denoiser,
)
from diffusion_audit.evaluation import curve_rows, evaluate_pool
from diffusion_audit.io import results_dir, write_json
from diffusion_audit.scorers import behavior_cloning_critic, calibrated_critic, diffusion_likelihood_proxy, oracle_scores
from diffusion_audit.stats import mean_ci_columns, paired_high_minus_low_ci
from diffusion_audit.toy_control import make_observations, trajectory_utilities


N_VALUES = [1, 2, 4, 8, 16, 32]
K_VALUES = [4]
VISUAL_REGIMES = ["id", "distractors", "shifted_colors", "observation_noise", "hidden_obstacle"]


def observation_ood_for_regime(regime: str) -> str:
    return "hidden_obstacle" if regime == "hidden_obstacle" else "id"


def learned_scorers(obs, trajectories) -> dict[str, np.ndarray]:
    return {
        "diffusion_likelihood_proxy": diffusion_likelihood_proxy(obs, trajectories),
        "learned_behavior_cloning_critic": behavior_cloning_critic(obs, trajectories),
        "calibrated_critic": calibrated_critic(obs, trajectories),
        "oracle_real_utility_selector": oracle_scores(obs, trajectories),
    }


def effect_ci_table(curves: pd.DataFrame, n_values: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    low_n = min(n_values)
    high_n = max(n_values)
    group_cols = ["conditioning", "regime", "scorer", "K"]
    for group_key, group in curves.groupby(group_cols):
        conditioning, regime, scorer, k = group_key
        for value_col in ["exact_selected_real", "exact_selected_score", "high_n_regret"]:
            ci = paired_high_minus_low_ci(
                group,
                unit_cols=["seed", "state_idx"],
                value_col=value_col,
                low_n=low_n,
                high_n=high_n,
                seed=1300 + int(k) + len(scorer),
            )
            rows.append(
                {
                    "conditioning": conditioning,
                    "regime": regime,
                    "scorer": scorer,
                    "K": int(k),
                    "metric": value_col,
                    **ci,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--train-states", type=int, default=18)
    parser.add_argument("--train-candidates", type=int, default=12)
    parser.add_argument("--eval-states", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--mc-trials", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    out_dir = results_dir()
    rows = []
    train_rows = []
    receding_rows = []
    n_values = [n for n in N_VALUES if n <= args.candidates]

    for seed in args.seeds:
        train_obs, train_actions = make_expert_dataset(
            states=args.train_states,
            candidates_per_state=args.train_candidates,
            horizon=args.horizon,
            seed=seed,
            multimodal=True,
        )
        state_model, state_result = train_denoiser(train_obs, train_actions, epochs=args.epochs, seed=seed)
        train_rows.append(
            {
                "seed": seed,
                "conditioning": "state",
                "initial_loss": state_result.initial_loss,
                "final_loss": state_result.final_loss,
                "loss_ratio": state_result.final_loss / max(state_result.initial_loss, 1e-12),
                "epochs": state_result.epochs,
                "image_size": 0,
            }
        )

        train_images, train_image_actions = make_image_expert_dataset(
            states=args.train_states,
            candidates_per_state=args.train_candidates,
            horizon=args.horizon,
            seed=seed,
            multimodal=True,
            image_size=args.image_size,
            visual_regime="id",
        )
        image_model, image_result = train_denoiser(
            train_images,
            train_image_actions,
            epochs=args.epochs,
            seed=10_000 + seed,
            lr=2.5e-3,
        )
        train_rows.append(
            {
                "seed": seed,
                "conditioning": "image",
                "initial_loss": image_result.initial_loss,
                "final_loss": image_result.final_loss,
                "loss_ratio": image_result.final_loss / max(image_result.initial_loss, 1e-12),
                "epochs": image_result.epochs,
                "image_size": args.image_size,
            }
        )

        for regime in VISUAL_REGIMES:
            observations = make_observations(
                args.eval_states,
                seed=5000 + seed,
                ood=observation_ood_for_regime(regime),
            )
            for state_idx, obs in enumerate(observations):
                obs_image = render_observation_image(
                    obs,
                    image_size=args.image_size,
                    visual_regime=regime,
                    seed=seed * 10_000 + state_idx,
                )
                for k in K_VALUES:
                    state_trajs = sample_denoised_trajectories(
                        state_model,
                        obs,
                        n=args.candidates,
                        k=k,
                        temperature=0.95,
                        seed=seed * 100_000 + state_idx * 100 + k,
                    )
                    image_trajs = sample_image_denoised_trajectories(
                        image_model,
                        obs_image,
                        n=args.candidates,
                        k=k,
                        temperature=0.95,
                        seed=seed * 200_000 + state_idx * 100 + k,
                    )
                    for conditioning, trajectories in [
                        ("state", state_trajs),
                        ("image", image_trajs),
                    ]:
                        utilities = trajectory_utilities(obs, trajectories)
                        for scorer, scores in learned_scorers(obs, trajectories).items():
                            payload = evaluate_pool(
                                scores,
                                utilities,
                                n_values,
                                mc_trials=args.mc_trials,
                                seed=seed + state_idx + k + len(conditioning),
                            )
                            rows.extend(
                                curve_rows(
                                    family="B_learned_diffusion_policy_lite",
                                    regime=regime,
                                    scorer=scorer,
                                    seed=seed,
                                    eval_payload=payload,
                                    extra={
                                        "state_idx": state_idx,
                                        "K": k,
                                        "conditioning": conditioning,
                                        "image_size": args.image_size if conditioning == "image" else 0,
                                    },
                                )
                            )

            for conditioning, model in [("state", state_model), ("image", image_model)]:
                for scorer_name, scorer_fn in [
                    ("diffusion_likelihood_proxy", diffusion_likelihood_proxy),
                    ("calibrated_critic", calibrated_critic),
                ]:
                    utilities = []
                    for idx, obs in enumerate(observations[: min(3, len(observations))]):
                        if conditioning == "state":
                            value = receding_horizon_utility(
                                model,
                                obs,
                                n=8,
                                k=8,
                                scorer=scorer_fn,
                                horizon=args.horizon,
                                rollout_steps=4,
                                temperature=0.95,
                                seed=seed * 10_000 + idx,
                            )
                        else:
                            value = receding_horizon_utility_image(
                                model,
                                obs,
                                n=8,
                                k=8,
                                scorer=scorer_fn,
                                horizon=args.horizon,
                                rollout_steps=4,
                                temperature=0.95,
                                seed=seed * 20_000 + idx,
                                image_size=args.image_size,
                                visual_regime=regime,
                            )
                        utilities.append(value)
                    receding_rows.append(
                        {
                            "seed": seed,
                            "conditioning": conditioning,
                            "regime": regime,
                            "scorer": scorer_name,
                            "N": 8,
                            "K": 8,
                            "mean_receding_horizon_utility": float(np.mean(utilities)),
                        }
                    )

    curves = pd.DataFrame(rows)
    training = pd.DataFrame(train_rows)
    receding = pd.DataFrame(receding_rows)
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
    ]
    seed_agg = curves.groupby(["seed", "conditioning", "regime", "scorer", "N", "K"], as_index=False)[numeric].mean()
    agg = mean_ci_columns(
        curves,
        group_cols=["conditioning", "regime", "scorer", "N", "K"],
        numeric_cols=numeric,
        seed=3000,
    )
    effect_cis = effect_ci_table(curves, n_values)

    curves.to_csv(out_dir / "tables" / "learned_policy_lite_curves.csv", index=False)
    seed_agg.to_csv(out_dir / "tables" / "learned_policy_lite_seed_aggregate.csv", index=False)
    agg.to_csv(out_dir / "tables" / "learned_policy_lite_aggregate.csv", index=False)
    effect_cis.to_csv(out_dir / "tables" / "learned_policy_lite_effect_cis.csv", index=False)
    training.to_csv(out_dir / "tables" / "learned_policy_lite_training.csv", index=False)
    receding.to_csv(out_dir / "tables" / "learned_policy_lite_receding_horizon.csv", index=False)

    state_losses = training[training["conditioning"] == "state"]["loss_ratio"]
    image_losses = training[training["conditioning"] == "image"]["loss_ratio"]
    summary = {
        "artifact_tables": {
            "curves": "results/tables/learned_policy_lite_curves.csv",
            "seed_aggregate": "results/tables/learned_policy_lite_seed_aggregate.csv",
            "aggregate": "results/tables/learned_policy_lite_aggregate.csv",
            "effect_cis": "results/tables/learned_policy_lite_effect_cis.csv",
            "training": "results/tables/learned_policy_lite_training.csv",
            "receding_horizon": "results/tables/learned_policy_lite_receding_horizon.csv",
        },
        "diffusion_policy_validity_checklist": {
            "stochastic_trajectory_generation": True,
            "iterative_denoising_or_noise_to_action_generation": True,
            "conditioning_on_observation_or_state": True,
            "image_conditioning_with_tiny_cnn": True,
            "action_sequence_generation": True,
            "receding_horizon_or_trajectory_execution_evaluation": True,
            "label": "learned Diffusion Policy-lite",
        },
        "conditioning_modes": ["state", "image"],
        "image_size": int(args.image_size),
        "visual_ood_regimes": [r for r in VISUAL_REGIMES if r != "id"],
        "ood_regimes": [r for r in VISUAL_REGIMES if r != "id"],
        "n_values": n_values,
        "k_values": K_VALUES,
        "num_training_seeds": int(training["seed"].nunique()),
        "loss_decreased_all_seed_conditioning_pairs": bool((training["final_loss"] < training["initial_loss"]).all()),
        "max_state_loss_ratio": float(state_losses.max()),
        "max_image_loss_ratio": float(image_losses.max()),
        "max_loss_ratio": float(training["loss_ratio"].max()),
    }
    write_json(out_dir / "learned_policy_lite_summary.json", summary)

    subset = agg[
        (agg["regime"] == "hidden_obstacle")
        & (agg["K"] == max(K_VALUES))
        & (agg["scorer"] == "calibrated_critic")
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for conditioning in ["state", "image"]:
        part = subset[subset["conditioning"] == conditioning]
        ax.plot(part["N"], part["exact_selected_real"], marker="o", label=f"{conditioning} conditioned")
        if "exact_selected_real_ci_low" in part:
            ax.fill_between(
                part["N"].astype(float),
                part["exact_selected_real_ci_low"].astype(float),
                part["exact_selected_real_ci_high"].astype(float),
                alpha=0.16,
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N sampled trajectories")
    ax.set_ylabel("Exact selected real utility")
    ax.set_title("Learned Diffusion Policy-lite hidden-obstacle reranking")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "learned_policy_lite_ood.png", dpi=160)
    plt.close(fig)

    examples = [
        render_observation_image(make_observations(1, seed=9000, ood=observation_ood_for_regime(regime))[0], image_size=args.image_size, visual_regime=regime, seed=9000)
        for regime in VISUAL_REGIMES
    ]
    fig, axes = plt.subplots(1, len(examples), figsize=(8.0, 1.9))
    for ax, image, regime in zip(axes, examples, VISUAL_REGIMES, strict=True):
        ax.imshow(np.moveaxis(image, 0, -1))
        ax.set_title(regime, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "toy_image_observations.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
