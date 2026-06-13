from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from diffusion_audit.controller import (
    AUDIT_ROLLOUTS,
    BLOCK_HIGH_N,
    CALIBRATE_SCORER,
    INCREASE_DIVERSITY,
    INCREASE_N,
    LATENCY_LIMITED,
    LOW_DIVERSITY_STOP,
    REDUCE_K,
    STOP_EARLY,
    AuditThenSampleConfig,
    audit_then_sample,
    audit_then_sample_adaptive,
    validate_repair_with_bounds,
)
from diffusion_audit.io import results_dir, write_json


REGIMES = [
    "high_diversity_aligned",
    "anti_correlated_scorer",
    "shuffled_scorer",
    "tail_misaligned_scorer",
    "adversarial_tail_scorer",
    "noisy_rollout_utility",
    "hidden_ood_dynamics",
    "duplicated_high_score_artifacts",
    "correlated_candidate_pool",
    "calibration_drift",
    "small_audit_underpowered",
    "collapsed_sampler",
    "latency_limited_aligned",
    "latency_spike",
    "missing_utility",
]

HARMFUL_NEGATIVE_CONTROLS = {
    "anti_correlated_scorer",
    "tail_misaligned_scorer",
    "adversarial_tail_scorer",
    "hidden_ood_dynamics",
    "duplicated_high_score_artifacts",
    "correlated_candidate_pool",
    "calibration_drift",
    "small_audit_underpowered",
    "collapsed_sampler",
    "latency_spike",
    "missing_utility",
    "adaptive_latency_limited",
}

UNDERPOWERED_REGIMES = {
    "correlated_candidate_pool",
    "small_audit_underpowered",
}


def n_grid(max_candidates: int) -> list[int]:
    values = [1, 2, 4, 8, 16, 32, 64, 96, 128]
    out = [n for n in values if n <= int(max_candidates)]
    if out[-1] != int(max_candidates):
        out.append(int(max_candidates))
    return out


def _compressed_aligned_utilities(n: int, rng: np.random.Generator) -> np.ndarray:
    base = np.linspace(-1.0, 1.0, int(n))
    utilities = np.where(base > 0.55, 0.90 + 0.08 * (base - 0.55) / 0.45, base)
    rng.shuffle(utilities)
    return utilities.astype(float)


def _base_trajectories(utilities: np.ndarray, horizon: int, rng: np.random.Generator) -> np.ndarray:
    n = utilities.size
    trajectories = rng.normal(scale=0.55, size=(n, int(horizon), 2))
    trajectories[:, :, 0] += 0.12 * utilities[:, None]
    trajectories[:, :, 1] += rng.normal(scale=0.04, size=(n, int(horizon)))
    return trajectories.astype(float)


def synthetic_action_pool(
    regime: str,
    n_candidates: int,
    horizon: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, float], float, dict[Any, float] | None]:
    rng = np.random.default_rng(seed)
    n = int(n_candidates)
    utilities = _compressed_aligned_utilities(n, rng)
    trajectories = _base_trajectories(utilities, horizon, rng)
    scores = utilities + rng.normal(scale=0.002, size=n)
    diversity = {
        "effective_sample_diversity": float(max(24, int(0.75 * n))),
        "duplicate_collapse_rate": 0.0,
        "trajectory_cluster_entropy": 1.0,
        "trajectory_cluster_count": 8.0,
    }
    lambda_cost = 0.0001
    runtime_measurements: dict[Any, float] | None = None

    if regime == "anti_correlated_scorer":
        scores = -utilities + rng.normal(scale=0.002, size=n)
    elif regime == "shuffled_scorer":
        scores = rng.permutation(utilities)
    elif regime == "tail_misaligned_scorer":
        scores = utilities + rng.normal(scale=0.002, size=n)
        low_utility = utilities < np.quantile(utilities, 0.25)
        scores[low_utility] += 4.0 + np.linspace(0.0, 0.4, int(np.sum(low_utility)))
    elif regime == "adversarial_tail_scorer":
        scores = utilities + rng.normal(scale=0.002, size=n)
        worst = np.argsort(utilities, kind="mergesort")[: max(4, n // 6)]
        scores[worst] = 5.0 + np.linspace(0.0, 1.0, worst.size)
    elif regime == "noisy_rollout_utility":
        scores = utilities + rng.normal(scale=0.02, size=n)
        utilities = utilities + rng.normal(scale=0.45, size=n)
    elif regime == "hidden_ood_dynamics":
        scores = utilities + rng.normal(scale=0.002, size=n)
        hidden_tail = scores >= np.quantile(scores, 0.80)
        utilities = utilities.copy()
        utilities[hidden_tail] -= 1.4
    elif regime == "duplicated_high_score_artifacts":
        duplicated = max(12, n // 3)
        trajectories[:duplicated] = trajectories[0]
        utilities = utilities.copy()
        utilities[:duplicated] = -1.0
        scores = utilities + rng.normal(scale=0.002, size=n)
        scores[:duplicated] = 5.0 + rng.normal(scale=0.001, size=duplicated)
        diversity = {
            "effective_sample_diversity": 4.0,
            "duplicate_collapse_rate": 0.80,
            "trajectory_cluster_entropy": 0.10,
            "trajectory_cluster_count": 1.0,
        }
    elif regime == "correlated_candidate_pool":
        clean = np.mean(trajectories, axis=0)
        trajectories = clean[None, :, :] + rng.normal(scale=0.002, size=trajectories.shape)
        diversity = {
            "effective_sample_diversity": 6.0,
            "duplicate_collapse_rate": 0.20,
            "trajectory_cluster_entropy": 0.25,
            "trajectory_cluster_count": 2.0,
        }
    elif regime == "calibration_drift":
        scores = utilities.copy()
        half = n // 2
        scores[half:] = -scores[half:]
        scores += rng.normal(scale=0.02, size=n)
    elif regime == "small_audit_underpowered":
        diversity = {
            "effective_sample_diversity": 8.0,
            "duplicate_collapse_rate": 0.0,
            "trajectory_cluster_entropy": 0.8,
            "trajectory_cluster_count": 4.0,
        }
    elif regime == "collapsed_sampler":
        clean = np.mean(trajectories, axis=0)
        trajectories[:] = clean[None, :, :]
        utilities = np.linspace(0.20, 0.22, n)
        scores = utilities.copy()
        diversity = {
            "effective_sample_diversity": 1.0,
            "duplicate_collapse_rate": 0.98,
            "trajectory_cluster_entropy": 0.0,
            "trajectory_cluster_count": 1.0,
        }
    elif regime in {"latency_limited_aligned", "latency_spike"}:
        utilities = np.linspace(0.0, 0.02, n)
        rng.shuffle(utilities)
        trajectories = _base_trajectories(utilities, horizon, rng)
        scores = utilities + rng.normal(scale=0.0002, size=n)
        lambda_cost = 0.03
        if regime == "latency_spike":
            runtime_measurements = {"runtime_per_candidate_ms": 4.0}
    elif regime == "missing_utility":
        return trajectories, scores.astype(float), None, diversity, lambda_cost, runtime_measurements
    elif regime != "high_diversity_aligned":
        raise ValueError(f"unknown regime {regime}")

    return trajectories.astype(float), scores.astype(float), utilities.astype(float), diversity, lambda_cost, runtime_measurements


def add_decision_region_figure(rows: pd.DataFrame, out_path) -> None:
    diversity = np.linspace(0.0, 80.0, 180)
    tail_lcb = np.linspace(-0.6, 1.0, 180)
    div_grid, tail_grid = np.meshgrid(diversity, tail_lcb)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharex=True, sharey=True)
    cmap = ListedColormap(["#8d99ae", "#d95f02", "#7570b3", "#1b9e77"])
    labels = ["underpowered", "tail or harm fail", "latency stop", "increase N"]
    for ax, latency_pressure in zip(axes, [0.02, 0.20, 0.60], strict=True):
        gain_margin = tail_grid - latency_pressure
        region = np.full_like(div_grid, 3, dtype=int)
        region[div_grid < 12.0] = 0
        region[(div_grid >= 12.0) & (tail_grid <= 0.0)] = 1
        region[(div_grid >= 12.0) & (tail_grid > 0.0) & (gain_margin <= 0.0)] = 2
        ax.imshow(
            region,
            origin="lower",
            extent=[diversity.min(), diversity.max(), tail_lcb.min(), tail_lcb.max()],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=3,
        )
        ax.set_title(f"latency pressure={latency_pressure:.2f}")
        ax.set_xlabel("effective N for bounds")
    axes[0].set_ylabel("tail utility LCB")
    point_colors = {
        INCREASE_N: "#1b9e77",
        BLOCK_HIGH_N: "#7570b3",
        CALIBRATE_SCORER: "#7570b3",
        INCREASE_DIVERSITY: "#8d99ae",
        AUDIT_ROLLOUTS: "#8d99ae",
    }
    for _, row in rows.iterrows():
        tail_value = row.get("tail_utility_lcb", -0.55)
        tail_float = float(tail_value) if pd.notna(tail_value) else -0.55
        latency_cost = float(row.get("lambda_cost", 0.0) or 0.0)
        axes[min(2, int(latency_cost > 0.01) * 2)].scatter(
            float(row.get("effective_n_for_bounds", 0.0) or 0.0),
            tail_float if np.isfinite(tail_float) else -0.55,
            s=18,
            c=point_colors.get(row["action_recommendation"], "#d95f02"),
            edgecolor="white",
            linewidth=0.35,
        )
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(i), label=labels[i], markersize=8)
        for i in range(4)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def fraction(rows: list[dict], predicate: Callable[[dict], bool]) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if predicate(row)) / len(rows))


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index)
    return frame[column].fillna(False).astype(bool)


def _add_controller_row(
    rows: list[dict],
    *,
    seed: int,
    state_idx: int,
    regime: str,
    result,
    negative_control: bool,
    adaptive: bool = False,
) -> None:
    payload = {
        "seed": seed,
        "state_idx": state_idx,
        "regime": regime,
        "negative_control": bool(negative_control),
        "adaptive_row": bool(adaptive),
        **result.as_dict(),
    }
    payload["false_admit_negative_control"] = bool(
        negative_control and payload.get("admit_high_N") is True
    )
    rows.append(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=40)
    args = parser.parse_args()

    out_dir = results_dir()
    ns = n_grid(args.candidates)
    k_values = [1, 8, 32]
    cfg = AuditThenSampleConfig(
        bootstrap_trials=int(args.bootstrap),
        confidence_method="both",
        min_effective_diversity=1.5,
        min_tail_rank_correlation=0.10,
        min_score_utility_correlation=0.10,
    )
    decision_rows: list[dict] = []
    calibration_rows: list[dict] = []

    for seed in args.seeds:
        for state_idx in range(int(args.states)):
            for regime in REGIMES:
                trajectories, scores, utilities, diversity, lambda_cost, runtime_measurements = synthetic_action_pool(
                    regime,
                    args.candidates,
                    args.horizon,
                    seed=100_000 + 997 * seed + 17 * state_idx + len(regime),
                )
                result = audit_then_sample(
                    trajectories,
                    scores,
                    utilities,
                    n_values=ns,
                    k_values=k_values,
                    lambda_cost=lambda_cost,
                    runtime_measurements=runtime_measurements,
                    diversity_diagnostics=diversity,
                    config=cfg,
                    seed=200_000 + 541 * seed + state_idx,
                )
                _add_controller_row(
                    decision_rows,
                    seed=seed,
                    state_idx=state_idx,
                    regime=regime,
                    result=result,
                    negative_control=regime in HARMFUL_NEGATIVE_CONTROLS,
                )

            adaptive_trajectories, adaptive_scores, adaptive_utilities, adaptive_div, _, _ = synthetic_action_pool(
                "latency_limited_aligned",
                args.candidates,
                args.horizon,
                seed=300_000 + 997 * seed + state_idx,
            )
            adaptive_result = audit_then_sample_adaptive(
                adaptive_trajectories,
                adaptive_scores,
                adaptive_utilities,
                batch_size=max(16, min(32, args.candidates // 2)),
                n_values=ns,
                k_values=k_values,
                lambda_cost=0.03,
                diversity_diagnostics=adaptive_div,
                config=cfg,
                seed=310_000 + 541 * seed + state_idx,
            )
            _add_controller_row(
                decision_rows,
                seed=seed,
                state_idx=state_idx,
                regime="adaptive_latency_limited",
                result=adaptive_result,
                negative_control=True,
                adaptive=True,
            )

            repair_n = max(160, int(args.candidates) * 3)
            repair_utilities = np.linspace(-1.0, 1.0, repair_n)
            repair_rng = np.random.default_rng(400_000 + 37 * seed + state_idx)
            repair_specs = [
                (
                    "anti_correlated_affine_repair",
                    -repair_utilities + repair_rng.normal(scale=0.015, size=repair_n),
                    AuditThenSampleConfig(use_effective_n_for_bounds=False, repair_method="auto"),
                    0.50,
                ),
                (
                    "monotone_isotonic_repair",
                    repair_utilities + repair_rng.normal(scale=0.05, size=repair_n),
                    AuditThenSampleConfig(use_effective_n_for_bounds=False, repair_method="isotonic"),
                    0.0,
                ),
                (
                    "random_score_failed_repair",
                    repair_rng.normal(size=repair_n),
                    AuditThenSampleConfig(use_effective_n_for_bounds=False, repair_method="auto"),
                    0.50,
                ),
                (
                    "calibration_drift_failed_repair",
                    np.r_[
                        repair_utilities[: repair_n // 2],
                        -repair_utilities[repair_n // 2 :],
                    ]
                    + repair_rng.normal(scale=0.02, size=repair_n),
                    AuditThenSampleConfig(use_effective_n_for_bounds=False, repair_method="auto"),
                    0.50,
                ),
            ]
            for repair_regime, repair_scores, repair_cfg, min_tail_improvement in repair_specs:
                repair = validate_repair_with_bounds(
                    repair_scores,
                    repair_utilities,
                    n_values=[min(ns), max(ns)],
                    seed=500_000 + 37 * seed + state_idx,
                    min_tail_improvement=min_tail_improvement,
                    config=repair_cfg,
                )
                calibration_rows.append(
                    {
                        "seed": seed,
                        "state_idx": state_idx,
                        "repair_regime": repair_regime,
                        "negative_control": repair_regime.endswith("failed_repair"),
                        **repair.as_dict(),
                    }
                )

    decisions = pd.DataFrame(decision_rows)
    calibrations = pd.DataFrame(calibration_rows)
    decisions.to_csv(out_dir / "tables" / "audit_then_sample_decisions.csv", index=False)
    calibrations.to_csv(out_dir / "tables" / "audit_then_sample_calibration.csv", index=False)
    add_decision_region_figure(decisions, out_dir / "figures" / "audit_then_sample_decision_regions.png")

    by_regime = {
        regime: decisions[decisions["regime"] == regime].to_dict("records")
        for regime in sorted(set(decisions["regime"].tolist()))
    }
    admitted = decisions[_bool_series(decisions, "admit_high_N")]
    negative_rows = decisions[_bool_series(decisions, "negative_control")]
    false_admits = decisions[_bool_series(decisions, "false_admit_negative_control")]
    underpowered_rows = decisions[decisions["regime"].isin(UNDERPOWERED_REGIMES)]
    adaptive_rows = decisions[_bool_series(decisions, "adaptive_row")]
    latency_lcb_positive = bool(
        len(admitted) == 0
        or (
            (admitted["latency_adjusted_gain_lcb"].astype(float) > 0.0)
            & (admitted["utility_gain_lcb"].astype(float) > 0.0)
            & (admitted["tail_utility_lcb"].astype(float) > 0.0)
        ).all()
    )
    repair_successes = calibrations[_bool_series(calibrations, "success")]
    repair_failures = calibrations[~_bool_series(calibrations, "success")]
    repair_bound_valid = bool(
        len(repair_successes) == 0
        or (
            (repair_successes["repaired_tail_utility_lcb"].astype(float) > 0.0)
            & (repair_successes["repaired_latency_adjusted_gain_lcb"].astype(float) > 0.0)
            & (repair_successes["repaired_utility_gain_lcb"].astype(float) > 0.0)
        ).all()
    )
    failure_controls_failed = bool(
        len(calibrations[calibrations["negative_control"].astype(bool)]) > 0
        and not calibrations[calibrations["negative_control"].astype(bool)]["success"].astype(bool).any()
    )
    effective_ratios = decisions.apply(
        lambda row: float(row.get("effective_n_for_bounds", 0.0) or 0.0)
        / max(float(row.get("audit_units", args.candidates) or args.candidates), 1.0),
        axis=1,
    )
    adaptive_savings = adaptive_rows.get("adaptive_stopping_savings", pd.Series(dtype=float)).astype(float)
    adaptive_stopped = (
        adaptive_rows["adaptive_stopped"].map(lambda value: str(value).strip().lower() == "true")
        if "adaptive_stopped" in adaptive_rows.columns
        else pd.Series([False] * len(adaptive_rows), index=adaptive_rows.index)
    )
    summary = {
        "artifact_tables": {
            "decisions": "results/tables/audit_then_sample_decisions.csv",
            "calibration": "results/tables/audit_then_sample_calibration.csv",
        },
        "artifact_figures": {
            "decision_regions": "results/figures/audit_then_sample_decision_regions.png",
        },
        "n_values": ns,
        "k_values": k_values,
        "regimes": REGIMES,
        "action_vocabulary": [
            INCREASE_N,
            STOP_EARLY,
            REDUCE_K,
            CALIBRATE_SCORER,
            AUDIT_ROLLOUTS,
            INCREASE_DIVERSITY,
            BLOCK_HIGH_N,
        ],
        "risk_delta": cfg.risk_delta,
        "confidence_method": cfg.confidence_method,
        "aligned_allow_high_n_fraction": fraction(
            by_regime["high_diversity_aligned"],
            lambda row: row["action_recommendation"] == INCREASE_N
            and bool(row.get("admit_high_N"))
            and float(row.get("latency_adjusted_gain_lcb", -1.0)) > 0.0,
        ),
        "anti_correlated_block_fraction": fraction(
            by_regime["anti_correlated_scorer"],
            lambda row: row["action_recommendation"] == BLOCK_HIGH_N and int(row["selected_N"]) == min(ns),
        ),
        "tail_misaligned_block_fraction": fraction(
            by_regime["tail_misaligned_scorer"],
            lambda row: row["action_recommendation"] == BLOCK_HIGH_N and int(row["selected_N"]) == min(ns),
        ),
        "shuffled_repair_or_block_fraction": fraction(
            by_regime["shuffled_scorer"],
            lambda row: row["action_recommendation"] in {CALIBRATE_SCORER, BLOCK_HIGH_N}
            and int(row["selected_N"]) == min(ns),
        ),
        "collapsed_stop_early_fraction": fraction(
            by_regime["collapsed_sampler"],
            lambda row: row["decision_label"] == LOW_DIVERSITY_STOP
            and row["action_recommendation"] == INCREASE_DIVERSITY
            and int(row["selected_N"]) == min(ns),
        ),
        "latency_limited_small_budget_fraction": fraction(
            by_regime["latency_limited_aligned"],
            lambda row: row["decision_label"] in {LATENCY_LIMITED, "high_N_blocked"}
            and int(row["selected_N"]) == min(ns)
            and row["action_recommendation"] in {STOP_EARLY, BLOCK_HIGH_N}
            and int(row["selected_K"]) in set(k_values),
        ),
        "false_admit_rate": float(len(false_admits) / max(len(negative_rows), 1)),
        "false_admit_count": int(len(false_admits)),
        "false_block_rate": float(
            1.0
            - fraction(
                by_regime["high_diversity_aligned"],
                lambda row: bool(row.get("admit_high_N")),
            )
        ),
        "abstention_rate": float(1.0 - len(admitted) / max(len(decisions), 1)),
        "lower_bound_coverage": float(latency_lcb_positive),
        "underpowered_abstain_fraction": fraction(
            underpowered_rows.to_dict("records"),
            lambda row: row["action_recommendation"] in {AUDIT_ROLLOUTS, INCREASE_DIVERSITY, BLOCK_HIGH_N}
            and not bool(row.get("admit_high_N")),
        ),
        "calibration_success_rows": int(len(repair_successes)),
        "calibration_failure_rows": int(len(repair_failures)),
        "repair_bound_validation_fraction": float(repair_bound_valid),
        "repair_failure_control_fraction": float(failure_controls_failed),
        "repair_success_methods": sorted(set(repair_successes.get("repair_method", pd.Series(dtype=str)).astype(str))),
        "effective_n_ratio_mean": float(np.mean(effective_ratios)) if len(effective_ratios) else 0.0,
        "effective_n_ratio_min": float(np.min(effective_ratios)) if len(effective_ratios) else 0.0,
        "adaptive_stopping_savings_mean": float(np.mean(adaptive_savings)) if len(adaptive_savings) else 0.0,
        "adaptive_stop_fraction": float(
            np.mean(adaptive_stopped) if len(adaptive_stopped) else 0.0
        ),
        "confidence_gates_present": bool(
            {
                "risk_delta",
                "effective_n_for_bounds",
                "utility_gain_lcb",
                "tail_utility_lcb",
                "latency_adjusted_gain_lcb",
                "admit_high_N",
                "abstention_reason",
                "false_admit_negative_control",
            }.issubset(decisions.columns)
        ),
        "negative_controls": sorted(HARMFUL_NEGATIVE_CONTROLS.union({"random_score_failed_repair"})),
        "controller_conservative_certification": bool(
            len(false_admits) == 0
            and latency_lcb_positive
            and repair_bound_valid
            and failure_controls_failed
            and fraction(
                underpowered_rows.to_dict("records"),
                lambda row: row["action_recommendation"] in {AUDIT_ROLLOUTS, INCREASE_DIVERSITY, BLOCK_HIGH_N}
                and not bool(row.get("admit_high_N")),
            )
            == 1.0
        ),
    }
    write_json(out_dir / "audit_then_sample_summary.json", summary)


if __name__ == "__main__":
    main()
