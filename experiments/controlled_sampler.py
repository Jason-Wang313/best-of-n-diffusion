from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffusion_audit.diversity import diversity_summary, marginal_diversity_gain
from diffusion_audit.evaluation import curve_rows, evaluate_pool
from diffusion_audit.io import results_dir, write_json
from diffusion_audit.scorers import aligned_scores, diffusion_likelihood_proxy, misaligned_tail_scores
from diffusion_audit.stats import mean_ci_columns, paired_high_minus_low_ci
from diffusion_audit.toy_control import make_observations, sample_diffusion_like_pool, trajectory_utilities


N_VALUES = [1, 2, 4, 8, 16, 32, 64]


REGIMES = {
    "low_diversity_high_quality": {
        "k": 32,
        "temperature": 0.05,
        "diversity": 0.01,
        "mode_coverage": 0.22,
        "collapsed": True,
        "biased_bad_mode": False,
        "low_k_noise": False,
        "scorer": "aligned",
    },
    "high_diversity_aligned": {
        "k": 16,
        "temperature": 0.88,
        "diversity": 0.95,
        "mode_coverage": 1.0,
        "collapsed": False,
        "biased_bad_mode": False,
        "low_k_noise": False,
        "scorer": "aligned",
    },
    "high_diversity_misaligned": {
        "k": 16,
        "temperature": 0.95,
        "diversity": 1.0,
        "mode_coverage": 1.0,
        "collapsed": False,
        "biased_bad_mode": True,
        "low_k_noise": False,
        "scorer": "misaligned_tail",
    },
    "collapsed_sampler": {
        "k": 18,
        "temperature": 0.34,
        "diversity": 0.03,
        "mode_coverage": 0.2,
        "collapsed": True,
        "biased_bad_mode": False,
        "low_k_noise": False,
        "scorer": "aligned",
    },
    "noisy_low_k_sampler": {
        "k": 2,
        "temperature": 1.10,
        "diversity": 0.90,
        "mode_coverage": 1.0,
        "collapsed": False,
        "biased_bad_mode": True,
        "low_k_noise": True,
        "scorer": "diffusion_likelihood_proxy",
    },
    "expensive_high_k_sampler": {
        "k": 32,
        "temperature": 0.62,
        "diversity": 0.55,
        "mode_coverage": 0.8,
        "collapsed": False,
        "biased_bad_mode": False,
        "low_k_noise": False,
        "scorer": "aligned",
    },
}


def score_pool(name: str, obs, trajectories: np.ndarray, seed: int) -> np.ndarray:
    if name == "aligned":
        return aligned_scores(obs, trajectories, seed=seed, noise=0.018)
    if name == "misaligned_tail":
        return misaligned_tail_scores(obs, trajectories, seed=seed)
    if name == "diffusion_likelihood_proxy":
        return diffusion_likelihood_proxy(obs, trajectories)
    raise ValueError(f"unknown scorer {name}")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
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
        "mean_pairwise_distance",
        "effective_sample_diversity",
        "duplicate_collapse_rate",
        "mode_coverage_observed",
        "marginal_diversity_gain_high_n",
    ]
    return mean_ci_columns(df, group_cols=["regime", "scorer", "N"], numeric_cols=numeric, seed=100)


def effect_ci_table(curves: pd.DataFrame, n_values: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    low_n = min(n_values)
    high_n = max(n_values)
    for regime, scorer in [
        ("high_diversity_aligned", "aligned"),
        ("high_diversity_misaligned", "misaligned_tail"),
        ("low_diversity_high_quality", "aligned"),
    ]:
        group = curves[(curves["regime"] == regime) & (curves["scorer"] == scorer)]
        for value_col in [
            "exact_selected_real",
            "exact_selected_score",
            "high_n_regret",
            "effective_sample_diversity",
            "duplicate_collapse_rate",
        ]:
            ci = paired_high_minus_low_ci(
                group,
                unit_cols=["seed", "state_idx"],
                value_col=value_col,
                low_n=low_n,
                high_n=high_n,
                seed=500 + len(regime) + len(value_col),
            )
            rows.append({"regime": regime, "scorer": scorer, "metric": value_col, **ci})
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
    rows: list[dict] = []
    diversity_rows: list[dict] = []
    n_values = [n for n in N_VALUES if n <= args.candidates]

    for seed in args.seeds:
        observations = make_observations(args.states, seed=100 + seed, ood="id")
        for state_idx, obs in enumerate(observations):
            for regime, cfg in REGIMES.items():
                pool = sample_diffusion_like_pool(
                    obs,
                    n_candidates=args.candidates,
                    horizon=args.horizon,
                    denoising_steps=cfg["k"],
                    temperature=cfg["temperature"],
                    diversity=cfg["diversity"],
                    mode_coverage_value=cfg["mode_coverage"],
                    collapsed=cfg["collapsed"],
                    biased_bad_mode=cfg["biased_bad_mode"],
                    low_k_noise=cfg["low_k_noise"],
                    seed=10_000 * seed + 101 * state_idx + len(regime),
                )
                utilities = trajectory_utilities(obs, pool.trajectories)
                scores = score_pool(
                    cfg["scorer"],
                    obs,
                    pool.trajectories,
                    seed=90_000 + seed + state_idx,
                )
                eval_payload = evaluate_pool(scores, utilities, n_values, mc_trials=args.mc_trials, seed=seed + state_idx)
                div = diversity_summary(pool.trajectories, pool.mode_ids, expected_modes=5)
                marg = marginal_diversity_gain(pool.trajectories, n_values)
                extra = {
                    "state_idx": state_idx,
                    "K": cfg["k"],
                    "temperature": cfg["temperature"],
                    "configured_diversity": cfg["diversity"],
                    "mean_pairwise_distance": div["mean_pairwise_distance"],
                    "effective_sample_diversity": div["effective_sample_diversity"],
                    "duplicate_collapse_rate": div["duplicate_collapse_rate"],
                    "mode_coverage_observed": div["mode_coverage"],
                    "marginal_diversity_gain_high_n": marg[max(n_values)],
                }
                rows.extend(
                    curve_rows(
                        family="A_controlled_diffusion_like_sampler",
                        regime=regime,
                        scorer=cfg["scorer"],
                        seed=seed,
                        eval_payload=eval_payload,
                        extra=extra,
                    )
                )
                diversity_rows.append({"regime": regime, "seed": seed, "state_idx": state_idx, **extra})

    curves = pd.DataFrame(rows)
    div_df = pd.DataFrame(diversity_rows)
    seed_agg = curves.groupby(["seed", "regime", "scorer", "N"], as_index=False)[
        [
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
            "mean_pairwise_distance",
            "effective_sample_diversity",
            "duplicate_collapse_rate",
            "mode_coverage_observed",
            "marginal_diversity_gain_high_n",
        ]
    ].mean()
    agg = aggregate(curves)
    effect_cis = effect_ci_table(curves, n_values)
    curves.to_csv(out_dir / "tables" / "controlled_sampler_curves.csv", index=False)
    seed_agg.to_csv(out_dir / "tables" / "controlled_sampler_seed_aggregate.csv", index=False)
    div_df.to_csv(out_dir / "tables" / "controlled_sampler_diversity.csv", index=False)
    agg.to_csv(out_dir / "tables" / "controlled_sampler_aggregate.csv", index=False)
    effect_cis.to_csv(out_dir / "tables" / "controlled_sampler_effect_cis.csv", index=False)

    def row(regime: str, n: int) -> pd.Series:
        return agg[(agg["regime"] == regime) & (agg["N"] == n)].iloc[0]

    low_n = min(n_values)
    high_n = max(n_values)
    aligned_gain = float(row("high_diversity_aligned", high_n)["exact_selected_real"] - row("high_diversity_aligned", low_n)["exact_selected_real"])
    aligned_score_gain = float(row("high_diversity_aligned", high_n)["exact_selected_score"] - row("high_diversity_aligned", low_n)["exact_selected_score"])
    bad_real_change = float(row("high_diversity_misaligned", high_n)["exact_selected_real"] - row("high_diversity_misaligned", low_n)["exact_selected_real"])
    bad_score_gain = float(row("high_diversity_misaligned", high_n)["exact_selected_score"] - row("high_diversity_misaligned", low_n)["exact_selected_score"])
    low_div_gain = float(row("low_diversity_high_quality", high_n)["exact_selected_real"] - row("low_diversity_high_quality", low_n)["exact_selected_real"])
    high_div_gain = aligned_gain
    summary = {
        "artifact_tables": {
            "curves": "results/tables/controlled_sampler_curves.csv",
            "seed_aggregate": "results/tables/controlled_sampler_seed_aggregate.csv",
            "aggregate": "results/tables/controlled_sampler_aggregate.csv",
            "diversity": "results/tables/controlled_sampler_diversity.csv",
            "effect_cis": "results/tables/controlled_sampler_effect_cis.csv",
        },
        "n_values": n_values,
        "regimes": list(REGIMES),
        "aligned_score_gain_high_minus_low": aligned_score_gain,
        "aligned_real_gain_high_minus_low": aligned_gain,
        "misaligned_score_gain_high_minus_low": bad_score_gain,
        "misaligned_real_change_high_minus_low": bad_real_change,
        "misaligned_high_n_regret": float(row("high_diversity_misaligned", high_n)["high_n_regret"]),
        "low_diversity_real_gain_high_minus_low": low_div_gain,
        "high_diversity_real_gain_high_minus_low": high_div_gain,
        "low_diversity_has_small_marginal_value": bool(low_div_gain < 0.35 * max(high_div_gain, 1e-9)),
        "effect_cis": {
            f"{row.regime}:{row.metric}": {
                "mean": float(row["mean"]),
                "se": float(row["se"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "n": int(row["n"]),
            }
            for _, row in effect_cis.iterrows()
        },
        "selected_score_improves_real_improves_regime": "high_diversity_aligned",
        "selected_score_improves_real_saturates_or_decreases_regime": "high_diversity_misaligned",
        "low_diversity_low_gain_regime": "low_diversity_high_quality",
    }
    write_json(out_dir / "controlled_sampler_summary.json", summary)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for regime in ["low_diversity_high_quality", "high_diversity_aligned", "high_diversity_misaligned", "collapsed_sampler"]:
        subset = agg[agg["regime"] == regime]
        ax.plot(subset["N"], subset["exact_selected_real"], marker="o", label=regime)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N sampled trajectories")
    ax.set_ylabel("Exact selected real utility")
    ax.set_title("Controlled diffusion-like sampler")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "controlled_sampler_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
