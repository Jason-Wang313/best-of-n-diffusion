from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffusion_audit.evaluation import curve_rows, evaluate_pool
from diffusion_audit.io import results_dir, write_json
from diffusion_audit.scorers import (
    apply_linear_critic,
    behavior_cloning_critic,
    calibrated_critic,
    diffusion_likelihood_proxy,
    fit_linear_value_critic,
    misaligned_tail_scores,
    oracle_scores,
    random_scores,
    trajectory_features,
)
from diffusion_audit.stats import bootstrap_mean_ci, mean_ci_columns, paired_high_minus_low_ci
from diffusion_audit.toy_control import make_observations, sample_diffusion_like_pool, trajectory_utilities


N_VALUES = [1, 2, 4, 8, 16, 32, 64]

CALIBRATION_REGIMES = {
    "hidden_obstacle_high_diversity": {
        "ood": "hidden_obstacle",
        "denoising_steps": 12,
        "temperature": 0.95,
        "diversity": 1.0,
        "mode_coverage": 1.0,
        "collapsed": False,
        "biased_bad_mode": True,
    },
    "id_high_diversity": {
        "ood": "id",
        "denoising_steps": 12,
        "temperature": 0.85,
        "diversity": 0.85,
        "mode_coverage": 1.0,
        "collapsed": False,
        "biased_bad_mode": False,
    },
    "low_diversity_collapsed": {
        "ood": "id",
        "denoising_steps": 18,
        "temperature": 0.05,
        "diversity": 0.01,
        "mode_coverage": 0.22,
        "collapsed": True,
        "biased_bad_mode": False,
    },
}


def scorer_dict(obs, trajectories, utilities, seed: int) -> dict[str, np.ndarray]:
    features = trajectory_features(obs, trajectories)
    rng = np.random.default_rng(seed)
    pilot = rng.choice(np.arange(trajectories.shape[0]), size=max(8, trajectories.shape[0] // 5), replace=False)
    pilot_weights = fit_linear_value_critic(features[pilot], utilities[pilot])
    return {
        "random_sample_selection": random_scores(trajectories.shape[0], seed=seed),
        "diffusion_likelihood_proxy": diffusion_likelihood_proxy(obs, trajectories),
        "learned_behavior_cloning_critic": behavior_cloning_critic(obs, trajectories),
        "learned_value_critic_from_pilot_rollouts": apply_linear_critic(features, pilot_weights),
        "calibrated_critic": calibrated_critic(obs, trajectories),
        "misaligned_tail_scorer": misaligned_tail_scores(obs, trajectories, seed=seed + 5),
        "oracle_real_utility_selector": oracle_scores(obs, trajectories),
    }


def scorer_gap_ci(
    curves: pd.DataFrame,
    *,
    regime: str,
    better_scorer: str,
    worse_scorer: str,
    n: int,
    value_col: str,
    seed: int,
) -> dict:
    sub = curves[(curves["regime"] == regime) & (curves["N"] == int(n))]
    pivot = sub.pivot_table(index=["seed", "state_idx"], columns="scorer", values=value_col, aggfunc="mean")
    if better_scorer not in pivot.columns or worse_scorer not in pivot.columns:
        values = []
    else:
        values = (pivot[better_scorer] - pivot[worse_scorer]).dropna().to_numpy(dtype=float)
    ci = bootstrap_mean_ci(values, seed=seed)
    ci["N"] = int(n)
    ci["value_col"] = value_col
    ci["effect"] = f"{better_scorer}_minus_{worse_scorer}"
    return ci


def effect_ci_table(curves: pd.DataFrame, n_values: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    low_n = min(n_values)
    high_n = max(n_values)
    for regime in CALIBRATION_REGIMES:
        misaligned = curves[(curves["regime"] == regime) & (curves["scorer"] == "misaligned_tail_scorer")]
        for value_col in ["exact_selected_real", "exact_selected_score"]:
            rows.append(
                {
                    "regime": regime,
                    "metric": value_col,
                    **paired_high_minus_low_ci(
                        misaligned,
                        unit_cols=["seed", "state_idx"],
                        value_col=value_col,
                        low_n=low_n,
                        high_n=high_n,
                        seed=700 + len(regime) + len(value_col),
                    ),
                }
            )
        for better, worse in [
            ("calibrated_critic", "misaligned_tail_scorer"),
            ("oracle_real_utility_selector", "misaligned_tail_scorer"),
        ]:
            rows.append(
                {
                    "regime": regime,
                    "metric": "exact_selected_real",
                    **scorer_gap_ci(
                        curves,
                        regime=regime,
                        better_scorer=better,
                        worse_scorer=worse,
                        n=high_n,
                        value_col="exact_selected_real",
                        seed=800 + len(regime) + len(better),
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--mc-trials", type=int, default=300)
    args = parser.parse_args()

    out_dir = results_dir()
    n_values = [n for n in N_VALUES if n <= args.candidates]
    rows = []
    for seed in args.seeds:
        for regime, cfg in CALIBRATION_REGIMES.items():
            observations = make_observations(args.states, seed=800 + seed, ood=cfg["ood"])
            for state_idx, obs in enumerate(observations):
                pool = sample_diffusion_like_pool(
                    obs,
                    n_candidates=args.candidates,
                    horizon=args.horizon,
                    denoising_steps=cfg["denoising_steps"],
                    temperature=cfg["temperature"],
                    diversity=cfg["diversity"],
                    mode_coverage_value=cfg["mode_coverage"],
                    seed=54_000 + seed * 101 + state_idx + len(regime),
                    collapsed=cfg["collapsed"],
                    biased_bad_mode=cfg["biased_bad_mode"],
                )
                utilities = trajectory_utilities(obs, pool.trajectories)
                for scorer, scores in scorer_dict(obs, pool.trajectories, utilities, seed=seed * 1000 + state_idx).items():
                    payload = evaluate_pool(scores, utilities, n_values, mc_trials=args.mc_trials, seed=seed + state_idx)
                    rows.extend(
                        curve_rows(
                            family="C_scorer_reranker_comparison",
                            regime=regime,
                            scorer=scorer,
                            seed=seed,
                            eval_payload=payload,
                            extra={"state_idx": state_idx, "K": cfg["denoising_steps"]},
                        )
                    )

    curves = pd.DataFrame(rows)
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
    seed_agg = curves.groupby(["seed", "regime", "scorer", "N"], as_index=False)[numeric].mean()
    agg = mean_ci_columns(curves, group_cols=["regime", "scorer", "N"], numeric_cols=numeric, seed=900)
    effect_cis = effect_ci_table(curves, n_values)
    curves.to_csv(out_dir / "tables" / "scorer_comparison_curves.csv", index=False)
    seed_agg.to_csv(out_dir / "tables" / "scorer_comparison_seed_aggregate.csv", index=False)
    agg.to_csv(out_dir / "tables" / "scorer_comparison_aggregate.csv", index=False)
    effect_cis.to_csv(out_dir / "tables" / "scorer_comparison_effect_cis.csv", index=False)

    low_n = min(n_values)
    high_n = max(n_values)

    def value(regime: str, scorer: str, n: int, col: str) -> float:
        return float(agg[(agg["regime"] == regime) & (agg["scorer"] == scorer) & (agg["N"] == n)].iloc[0][col])

    calibration_map = []
    for regime in CALIBRATION_REGIMES:
        oracle_gap = value(regime, "oracle_real_utility_selector", high_n, "exact_selected_real") - value(
            regime, "misaligned_tail_scorer", high_n, "exact_selected_real"
        )
        repair = value(regime, "calibrated_critic", high_n, "exact_selected_real") - value(
            regime, "misaligned_tail_scorer", high_n, "exact_selected_real"
        )
        repair_fraction = repair / max(oracle_gap, 1e-12)
        status = "strong_repair" if repair >= 0.80 and oracle_gap >= 0.90 and repair_fraction >= 0.75 else "no_strong_repair"
        calibration_map.append(
            {
                "regime": regime,
                "N": high_n,
                "calibrated_minus_misaligned_high_n": repair,
                "oracle_minus_misaligned_high_n": oracle_gap,
                "repair_fraction_of_oracle_gap": repair_fraction,
                "repair_status": status,
            }
        )
    calibration_map_df = pd.DataFrame(calibration_map)
    calibration_map_df.to_csv(out_dir / "tables" / "calibration_repair_map.csv", index=False)

    summary = {
        "artifact_tables": {
            "curves": "results/tables/scorer_comparison_curves.csv",
            "seed_aggregate": "results/tables/scorer_comparison_seed_aggregate.csv",
            "aggregate": "results/tables/scorer_comparison_aggregate.csv",
            "effect_cis": "results/tables/scorer_comparison_effect_cis.csv",
            "calibration_repair_map": "results/tables/calibration_repair_map.csv",
        },
        "scorers": sorted(agg["scorer"].unique().tolist()),
        "calibration_regimes": sorted(CALIBRATION_REGIMES),
        "calibration_map": calibration_map,
        "oracle_real_gain_high_minus_low": value("hidden_obstacle_high_diversity", "oracle_real_utility_selector", high_n, "exact_selected_real")
        - value("hidden_obstacle_high_diversity", "oracle_real_utility_selector", low_n, "exact_selected_real"),
        "misaligned_score_gain_high_minus_low": value("hidden_obstacle_high_diversity", "misaligned_tail_scorer", high_n, "exact_selected_score")
        - value("hidden_obstacle_high_diversity", "misaligned_tail_scorer", low_n, "exact_selected_score"),
        "misaligned_real_change_high_minus_low": value("hidden_obstacle_high_diversity", "misaligned_tail_scorer", high_n, "exact_selected_real")
        - value("hidden_obstacle_high_diversity", "misaligned_tail_scorer", low_n, "exact_selected_real"),
        "oracle_minus_misaligned_high_n": value("hidden_obstacle_high_diversity", "oracle_real_utility_selector", high_n, "exact_selected_real")
        - value("hidden_obstacle_high_diversity", "misaligned_tail_scorer", high_n, "exact_selected_real"),
        "calibrated_minus_misaligned_high_n": value("hidden_obstacle_high_diversity", "calibrated_critic", high_n, "exact_selected_real")
        - value("hidden_obstacle_high_diversity", "misaligned_tail_scorer", high_n, "exact_selected_real"),
        "effect_cis": {
            f"{row.regime}:{row.effect}:{row.metric}": {
                "mean": float(row["mean"]),
                "se": float(row["se"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "n": int(row["n"]),
            }
            for _, row in effect_cis.iterrows()
        },
        "aligned_oracle_helps_but_misaligned_hurts_or_saturates": bool(
            value("hidden_obstacle_high_diversity", "oracle_real_utility_selector", high_n, "exact_selected_real")
            > value("hidden_obstacle_high_diversity", "oracle_real_utility_selector", low_n, "exact_selected_real")
            and value("hidden_obstacle_high_diversity", "misaligned_tail_scorer", high_n, "exact_selected_real")
            <= value("hidden_obstacle_high_diversity", "misaligned_tail_scorer", low_n, "exact_selected_real") + 0.02
        ),
    }
    write_json(out_dir / "scorer_comparison_summary.json", summary)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for scorer in [
        "random_sample_selection",
        "diffusion_likelihood_proxy",
        "learned_behavior_cloning_critic",
        "calibrated_critic",
        "misaligned_tail_scorer",
        "oracle_real_utility_selector",
    ]:
        subset = agg[(agg["regime"] == "hidden_obstacle_high_diversity") & (agg["scorer"] == scorer)]
        ax.plot(subset["N"], subset["exact_selected_real"], marker="o", label=scorer)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N sampled trajectories")
    ax.set_ylabel("Exact selected real utility")
    ax.set_title("Scorer controls the value of N")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "scorer_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
