from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffusion_audit.io import results_dir, write_json
from diffusion_audit.latency import latency_adjusted_utility, total_budget, utility_per_diffusion_step
from diffusion_audit.scorers import aligned_scores
from diffusion_audit.stats import bootstrap_mean_ci, mean_ci_columns
from diffusion_audit.theory import utility_max_selection_finite
from diffusion_audit.toy_control import make_observations, sample_diffusion_like_pool, trajectory_utilities


N_GRID = [1, 2, 4, 8, 16, 32]
K_GRID = [2, 4, 8, 16, 32]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--lambda-cost", type=float, default=0.0035)
    args = parser.parse_args()

    out_dir = results_dir()
    rows = []
    n_values = [n for n in N_GRID if n <= args.max_candidates]
    for seed in args.seeds:
        observations = make_observations(args.states, seed=1200 + seed, ood="id")
        for state_idx, obs in enumerate(observations):
            for k in K_GRID:
                pool = sample_diffusion_like_pool(
                    obs,
                    n_candidates=args.max_candidates,
                    horizon=args.horizon,
                    denoising_steps=k,
                    temperature=0.92,
                    diversity=0.85,
                    mode_coverage_value=1.0,
                    seed=77_000 + seed * 100 + state_idx * 17 + k,
                    low_k_noise=k <= 4,
                )
                utilities = trajectory_utilities(obs, pool.trajectories)
                scores = aligned_scores(obs, pool.trajectories, seed=seed + state_idx + k, noise=0.02)
                real_curve = utility_max_selection_finite(scores, utilities, n_values)
                for n in n_values:
                    real = float(real_curve[n])
                    adjusted = latency_adjusted_utility(real, n=n, k=k, lambda_cost=args.lambda_cost)
                    rows.append(
                        {
                            "seed": seed,
                            "state_idx": state_idx,
                            "N": n,
                            "K": k,
                            "B": total_budget(n, k),
                            "real_utility": real,
                            "latency_adjusted_utility": adjusted,
                            "utility_per_diffusion_step": utility_per_diffusion_step(real, n, k),
                            "lambda_cost": args.lambda_cost,
                        }
                    )

    grid = pd.DataFrame(rows)
    numeric = ["B", "real_utility", "latency_adjusted_utility", "utility_per_diffusion_step"]
    seed_agg = grid.groupby(["seed", "N", "K"], as_index=False)[numeric].mean()
    agg = mean_ci_columns(grid, group_cols=["N", "K"], numeric_cols=numeric, seed=1100)
    grid.to_csv(out_dir / "tables" / "nk_budget_curves.csv", index=False)
    seed_agg.to_csv(out_dir / "tables" / "nk_budget_seed_aggregate.csv", index=False)
    agg.to_csv(out_dir / "tables" / "nk_budget_phase.csv", index=False)
    best_real = agg.iloc[int(np.argmax(agg["real_utility"].to_numpy()))].to_dict()
    best_latency = agg.iloc[int(np.argmax(agg["latency_adjusted_utility"].to_numpy()))].to_dict()
    high_budget = agg[(agg["N"] == max(n_values)) & (agg["K"] == max(K_GRID))].iloc[0].to_dict()
    latency_pairs = grid[
        (
            (grid["N"] == int(best_latency["N"]))
            & (grid["K"] == int(best_latency["K"]))
        )
        | ((grid["N"] == int(high_budget["N"])) & (grid["K"] == int(high_budget["K"])))
    ].pivot_table(index=["seed", "state_idx"], columns=["N", "K"], values="latency_adjusted_utility", aggfunc="mean")
    best_key = (int(best_latency["N"]), int(best_latency["K"]))
    high_key = (int(high_budget["N"]), int(high_budget["K"]))
    if best_key in latency_pairs.columns and high_key in latency_pairs.columns:
        latency_effects = (latency_pairs[best_key] - latency_pairs[high_key]).dropna().to_numpy(dtype=float)
    else:
        latency_effects = np.asarray([], dtype=float)
    latency_ci = bootstrap_mean_ci(latency_effects, seed=1200)
    latency_effect_df = pd.DataFrame(
        [
            {
                "effect": "best_latency_adjusted_minus_high_budget_corner",
                "best_N": int(best_latency["N"]),
                "best_K": int(best_latency["K"]),
                "high_N": int(high_budget["N"]),
                "high_K": int(high_budget["K"]),
                **latency_ci,
            }
        ]
    )
    latency_effect_df.to_csv(out_dir / "tables" / "nk_budget_latency_effect_ci.csv", index=False)
    summary = {
        "artifact_tables": {
            "curves": "results/tables/nk_budget_curves.csv",
            "seed_aggregate": "results/tables/nk_budget_seed_aggregate.csv",
            "phase": "results/tables/nk_budget_phase.csv",
            "latency_effect_ci": "results/tables/nk_budget_latency_effect_ci.csv",
        },
        "n_grid": n_values,
        "k_grid": K_GRID,
        "lambda_cost": args.lambda_cost,
        "best_real": {k: float(v) for k, v in best_real.items()},
        "best_latency_adjusted": {k: float(v) for k, v in best_latency.items()},
        "high_budget_corner": {k: float(v) for k, v in high_budget.items()},
        "latency_prefers_smaller_budget_than_max": bool(best_latency["B"] < high_budget["B"]),
        "latency_best_differs_from_real_best": bool(
            int(best_latency["N"]) != int(best_real["N"]) or int(best_latency["K"]) != int(best_real["K"])
        ),
        "latency_best_minus_high_budget_ci": latency_ci,
    }
    write_json(out_dir / "nk_budget_summary.json", summary)

    pivot = agg.pivot(index="K", columns="N", values="latency_adjusted_utility").sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(x)) for x in pivot.index])
    ax.set_xlabel("N sampled trajectories")
    ax.set_ylabel("K denoising steps")
    ax.set_title("Latency-adjusted utility over N and K")
    fig.colorbar(image, ax=ax, label="utility - latency cost")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "nk_budget_phase_diagram.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
