"""Audit-Then-Sample inference controller for Best-of-N diffusion reranking."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from diffusion_best_of_n.alignment import (
    score_utility_correlation,
    tail_rank_correlation,
    top_score_tail_mask,
)
from diffusion_best_of_n.diversity import diversity_summary, trajectory_cluster_ids
from diffusion_best_of_n.latency import latency_cost
from diffusion_best_of_n.theory import utility_best_of_n_finite


INCREASE_N = "increase_N"
STOP_EARLY = "stop_early"
REDUCE_K = "reduce_K"
CALIBRATE_SCORER = "calibrate_scorer"
AUDIT_ROLLOUTS = "audit_rollouts"
INCREASE_DIVERSITY = "increase_diversity"
INCREASE_TEMPERATURE = "increase_temperature"
CLUSTER_BALANCE = "cluster_balance"
MODE_QUOTA = "mode_quota"
RESAMPLE = "resample"
BLOCK_HIGH_N = "block_high_N"

ALLOW_HIGH_N = "allow_high_N"
LOW_DIVERSITY_STOP = "stop_early_low_diversity"
ALIGNMENT_AUDIT_FAILED = "alignment_audit_failed"
UNDERPOWERED_AUDIT = "underpowered_audit"
LATENCY_LIMITED = "latency_limited"
HIGH_N_BLOCKED = "high_N_blocked"
CALIBRATION_REPAIRED = "calibration_repaired"


GateName = Literal[
    "utility_gain",
    "tail_utility",
    "latency_adjusted_gain",
    "tail_harm",
    "repair",
]


@dataclass(frozen=True)
class AuditThenSampleConfig:
    """Thresholds for conservative high-N admission."""

    min_effective_diversity: float = 2.0
    max_duplicate_collapse_rate: float = 0.65
    min_tail_rank_correlation: float = 0.20
    min_score_utility_correlation: float = 0.15
    min_gain_lcb: float = 0.0
    min_block_harm: float = 0.0
    confidence: float = 0.95
    bootstrap_trials: int = 200
    runtime_cost_per_step: float = 1.0
    runtime_overhead: float = 0.0
    tail_fraction: float = 0.20
    risk_delta: float = 0.01
    gate_delta_allocation: Mapping[str, float] | None = None
    min_audit_units: int = 12
    min_effect_size: float = 0.0
    confidence_method: Literal["empirical_bernstein", "bootstrap", "both"] = "empirical_bernstein"
    repair_method: Literal["isotonic", "affine", "auto"] = "auto"
    use_effective_n_for_bounds: bool = True
    cluster_balance_candidates: bool = False


@dataclass(frozen=True)
class AuditThenSampleResult:
    """Controller output and auditable diagnostics."""

    selected_n: int
    selected_k: int
    decision_label: str
    action_recommendation: str
    confidence_diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_N": int(self.selected_n),
            "selected_K": int(self.selected_k),
            "decision_label": self.decision_label,
            "action_recommendation": self.action_recommendation,
            **self.confidence_diagnostics,
        }


@dataclass(frozen=True)
class AffineCalibration:
    """One-dimensional score-to-utility affine repair."""

    slope: float
    intercept: float

    def predict(self, scores: Iterable[float] | np.ndarray) -> np.ndarray:
        return float(self.slope) * np.asarray(scores, dtype=float) + float(self.intercept)


@dataclass(frozen=True)
class IsotonicCalibration:
    """One-dimensional monotone isotonic score-to-utility repair."""

    thresholds: np.ndarray
    values: np.ndarray

    def predict(self, scores: Iterable[float] | np.ndarray) -> np.ndarray:
        score_arr = np.asarray(scores, dtype=float)
        return np.interp(
            score_arr,
            self.thresholds,
            self.values,
            left=float(self.values[0]),
            right=float(self.values[-1]),
        )


@dataclass(frozen=True)
class CalibrationRepairResult:
    """Held-out calibration repair audit."""

    success: bool
    original_tail_rank_correlation: float
    repaired_tail_rank_correlation: float
    original_high_n_gain: float
    repaired_high_n_gain: float
    calibration: AffineCalibration | IsotonicCalibration
    pilot_count: int
    holdout_count: int
    recommendation: str
    repair_method: str
    risk_delta: float
    effective_n_for_bounds: int
    original_utility_gain_lcb: float
    repaired_utility_gain_lcb: float
    original_tail_utility_lcb: float
    repaired_tail_utility_lcb: float
    original_latency_adjusted_gain_lcb: float
    repaired_latency_adjusted_gain_lcb: float
    repaired_score_lcb_radius: float

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": bool(self.success),
            "original_tail_rank_correlation": float(self.original_tail_rank_correlation),
            "repaired_tail_rank_correlation": float(self.repaired_tail_rank_correlation),
            "original_high_n_gain": float(self.original_high_n_gain),
            "repaired_high_n_gain": float(self.repaired_high_n_gain),
            "pilot_count": int(self.pilot_count),
            "holdout_count": int(self.holdout_count),
            "recommendation": self.recommendation,
            "repair_method": self.repair_method,
            "risk_delta": float(self.risk_delta),
            "effective_n_for_bounds": int(self.effective_n_for_bounds),
            "original_utility_gain_lcb": float(self.original_utility_gain_lcb),
            "repaired_utility_gain_lcb": float(self.repaired_utility_gain_lcb),
            "original_tail_utility_lcb": float(self.original_tail_utility_lcb),
            "repaired_tail_utility_lcb": float(self.repaired_tail_utility_lcb),
            "original_latency_adjusted_gain_lcb": float(self.original_latency_adjusted_gain_lcb),
            "repaired_latency_adjusted_gain_lcb": float(self.repaired_latency_adjusted_gain_lcb),
            "repaired_score_lcb_radius": float(self.repaired_score_lcb_radius),
        }
        if isinstance(self.calibration, AffineCalibration):
            payload.update(
                {
                    "calibration_slope": float(self.calibration.slope),
                    "calibration_intercept": float(self.calibration.intercept),
                    "isotonic_knots": 0,
                }
            )
        else:
            payload.update(
                {
                    "calibration_slope": float("nan"),
                    "calibration_intercept": float("nan"),
                    "isotonic_knots": int(self.calibration.thresholds.size),
                }
            )
        return payload


def _as_1d(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_n_values(n_values: Iterable[int] | None, pool_size: int) -> list[int]:
    if n_values is None:
        out: list[int] = []
        n = 1
        while n <= int(pool_size):
            out.append(n)
            n *= 2
        if out[-1] != int(pool_size):
            out.append(int(pool_size))
        return out
    out = sorted({int(n) for n in n_values})
    if not out or any(n < 1 for n in out):
        raise ValueError("n_values must contain positive integers")
    return out


def _as_k_values(k_values: Iterable[int] | None, sampler_metadata: Mapping[str, Any] | None) -> list[int]:
    if k_values is not None:
        out = sorted({int(k) for k in k_values})
    elif sampler_metadata and "denoising_steps" in sampler_metadata:
        out = [int(sampler_metadata["denoising_steps"])]
    else:
        out = [1]
    if not out or any(n < 1 for n in out):
        raise ValueError("k_values must contain positive integers")
    return out


def _runtime_lookup(
    runtime_measurements: Mapping[Any, float] | None,
    n: int,
    k: int,
    config: AuditThenSampleConfig,
) -> float:
    if runtime_measurements:
        for key in [(int(n), int(k)), f"{int(n)},{int(k)}", f"N={int(n)},K={int(k)}"]:
            if key in runtime_measurements:
                return float(runtime_measurements[key])
        for key in [int(k), str(int(k))]:
            if key in runtime_measurements:
                return float(n) * float(runtime_measurements[key])
        if "runtime_per_candidate_ms" in runtime_measurements:
            return float(n) * float(runtime_measurements["runtime_per_candidate_ms"])
    return latency_cost(
        int(n),
        int(k),
        cost_per_step=config.runtime_cost_per_step,
        overhead=config.runtime_overhead,
    )


def _ci_from_values(values: np.ndarray, confidence: float) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    alpha = 1.0 - float(confidence)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, alpha / 2.0)),
        "ci_high": float(np.quantile(arr, 1.0 - alpha / 2.0)),
    }


def empirical_bernstein_radius(
    values: Iterable[float] | np.ndarray,
    *,
    delta: float = 0.01,
    n_eff: int | None = None,
    value_range: float | None = None,
) -> float:
    """One-sided empirical-Bernstein radius for bounded scalar observations."""

    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size if n_eff is None else min(max(int(n_eff), 0), arr.size))
    if n < 2:
        return float("nan")
    delta = float(delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if value_range is None:
        span = float(np.max(arr) - np.min(arr))
    else:
        span = float(value_range)
    if span < 0.0 or not np.isfinite(span):
        raise ValueError("value_range must be non-negative and finite")
    if span == 0.0:
        return 0.0
    variance = float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0
    log_term = float(np.log(3.0 / delta))
    return float(np.sqrt(2.0 * variance * log_term / n) + 3.0 * span * log_term / max(n - 1, 1))


def empirical_bernstein_lcb(
    values: Iterable[float] | np.ndarray,
    *,
    delta: float = 0.01,
    n_eff: int | None = None,
    value_range: float | None = None,
) -> float:
    """One-sided empirical-Bernstein lower confidence bound on the mean."""

    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    radius = empirical_bernstein_radius(arr, delta=delta, n_eff=n_eff, value_range=value_range)
    if not np.isfinite(radius):
        return float("nan")
    return float(np.mean(arr) - radius)


def empirical_bernstein_ucb(
    values: Iterable[float] | np.ndarray,
    *,
    delta: float = 0.01,
    n_eff: int | None = None,
    value_range: float | None = None,
) -> float:
    """One-sided empirical-Bernstein upper confidence bound on the mean."""

    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    radius = empirical_bernstein_radius(arr, delta=delta, n_eff=n_eff, value_range=value_range)
    if not np.isfinite(radius):
        return float("nan")
    return float(np.mean(arr) + radius)


def _empirical_bernstein_summary(
    values: Iterable[float] | np.ndarray,
    *,
    delta: float,
    n_eff: int | None,
    value_range: float | None = None,
) -> dict[str, float | int]:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "n_eff": 0,
            "mean": float("nan"),
            "radius": float("nan"),
            "lcb": float("nan"),
            "ucb": float("nan"),
        }
    used_n = int(arr.size if n_eff is None else min(max(int(n_eff), 0), arr.size))
    radius = empirical_bernstein_radius(arr, delta=delta, n_eff=used_n, value_range=value_range)
    mean = float(np.mean(arr))
    return {
        "n": int(arr.size),
        "n_eff": int(used_n),
        "mean": mean,
        "radius": float(radius),
        "lcb": float(mean - radius) if np.isfinite(radius) else float("nan"),
        "ucb": float(mean + radius) if np.isfinite(radius) else float("nan"),
    }


def _gate_delta(config: AuditThenSampleConfig, gate: GateName) -> float:
    if config.gate_delta_allocation and gate in config.gate_delta_allocation:
        value = float(config.gate_delta_allocation[gate])
        if not 0.0 < value < 1.0:
            raise ValueError("gate delta allocations must be in (0, 1)")
        return value
    return float(config.risk_delta) / 5.0


def _objective_grid(
    scores: np.ndarray,
    utilities: np.ndarray,
    n_values: list[int],
    k_values: list[int],
    lambda_cost: float,
    runtime_measurements: Mapping[Any, float] | None,
    config: AuditThenSampleConfig,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    raw_curve = utility_best_of_n_finite(scores, utilities, n_values)
    adjusted: dict[tuple[int, int], float] = {}
    for n in n_values:
        for k in k_values:
            runtime = _runtime_lookup(runtime_measurements, n, k, config)
            adjusted[(int(n), int(k))] = float(raw_curve[int(n)]) - float(lambda_cost) * runtime
    return raw_curve, adjusted


def _best_pair(adjusted: Mapping[tuple[int, int], float], allowed_n: Iterable[int] | None = None) -> tuple[int, int]:
    allowed = {int(n) for n in allowed_n} if allowed_n is not None else None
    rows = [
        (key, float(value))
        for key, value in adjusted.items()
        if allowed is None or int(key[0]) in allowed
    ]
    if not rows:
        raise ValueError("no objective rows available for the requested N set")
    return max(rows, key=lambda item: (item[1], -item[0][0], -item[0][1]))[0]


def _selection_weights(scores: np.ndarray, n: int) -> np.ndarray:
    score_arr = np.asarray(scores, dtype=float)
    order = np.argsort(score_arr, kind="mergesort")
    sorted_scores = score_arr[order]
    weights = np.zeros(score_arr.size, dtype=float)
    pool_size = float(score_arr.size)
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        mass = (j / pool_size) ** int(n) - (i / pool_size) ** int(n)
        weights[order[i:j]] = mass / float(j - i)
        i = j
    return weights


def _utility_span(utilities: np.ndarray) -> float:
    if utilities.size < 2:
        return 0.0
    return float(np.max(utilities) - np.min(utilities))


def _bound_n(
    *,
    audit_n: int,
    effective_sample_diversity: float,
    config: AuditThenSampleConfig,
) -> int:
    if not config.use_effective_n_for_bounds:
        return int(audit_n)
    if not np.isfinite(effective_sample_diversity):
        return 0
    return int(min(int(audit_n), max(0, int(np.floor(float(effective_sample_diversity))))))


def _bootstrap_high_n_gain_ci(
    scores: np.ndarray,
    utilities: np.ndarray,
    n_values: list[int],
    k_values: list[int],
    lambda_cost: float,
    runtime_measurements: Mapping[Any, float] | None,
    config: AuditThenSampleConfig,
    seed: int,
) -> dict[str, float | int]:
    if config.bootstrap_trials <= 0 or config.confidence_method == "empirical_bernstein":
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    low_n = min(n_values)
    high_ns = [n for n in n_values if n > low_n]
    if not high_ns:
        return {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    gains = []
    for _ in range(int(config.bootstrap_trials)):
        idx = rng.integers(0, scores.size, size=scores.size)
        _, adjusted = _objective_grid(
            scores[idx],
            utilities[idx],
            n_values,
            k_values,
            lambda_cost,
            runtime_measurements,
            config,
        )
        low_pair = _best_pair(adjusted, allowed_n=[low_n])
        high_pair = _best_pair(adjusted, allowed_n=high_ns)
        gains.append(float(adjusted[high_pair] - adjusted[low_pair]))
    return _ci_from_values(np.asarray(gains, dtype=float), config.confidence)


def _bootstrap_tail_corr_ci(
    scores: np.ndarray,
    utilities: np.ndarray,
    config: AuditThenSampleConfig,
    seed: int,
) -> dict[str, float | int]:
    if config.bootstrap_trials <= 0 or config.confidence_method == "empirical_bernstein":
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(config.bootstrap_trials)):
        idx = rng.integers(0, scores.size, size=scores.size)
        try:
            values.append(
                tail_rank_correlation(
                    scores[idx],
                    utilities[idx],
                    tail_fraction=config.tail_fraction,
                )
            )
        except ValueError:
            values.append(float("nan"))
    return _ci_from_values(np.asarray(values, dtype=float), config.confidence)


def _gate_metrics(
    scores: np.ndarray,
    utilities: np.ndarray,
    n_values: list[int],
    k_values: list[int],
    lambda_cost: float,
    runtime_measurements: Mapping[Any, float] | None,
    config: AuditThenSampleConfig,
    *,
    effective_sample_diversity: float,
    seed: int,
) -> dict[str, Any]:
    low_n = min(n_values)
    high_ns = [n for n in n_values if n > low_n]
    raw_curve, adjusted = _objective_grid(
        scores,
        utilities,
        n_values,
        k_values,
        float(lambda_cost),
        runtime_measurements,
        config,
    )
    low_pair = _best_pair(adjusted, allowed_n=[low_n])
    high_pair = _best_pair(adjusted, allowed_n=high_ns) if high_ns else low_pair
    best_pair = _best_pair(adjusted)

    low_weights = _selection_weights(scores, int(low_pair[0]))
    high_weights = _selection_weights(scores, int(high_pair[0]))
    low_utility = float(np.dot(low_weights, utilities))
    high_utility = float(np.dot(high_weights, utilities))
    utility_gain = float(high_utility - low_utility)
    latency_delta = float(lambda_cost) * (
        _runtime_lookup(runtime_measurements, high_pair[0], high_pair[1], config)
        - _runtime_lookup(runtime_measurements, low_pair[0], low_pair[1], config)
    )
    latency_adjusted_gain = float(utility_gain - latency_delta)
    utility_range = _utility_span(utilities)
    effective_n = _bound_n(
        audit_n=int(scores.size),
        effective_sample_diversity=effective_sample_diversity,
        config=config,
    )
    tail_mask = top_score_tail_mask(
        scores,
        tail_fraction=config.tail_fraction,
        min_count=min(int(config.min_audit_units), int(scores.size)),
    )
    tail_values = utilities[tail_mask] - low_utility
    tail_n_eff = int(min(int(np.sum(tail_mask)), effective_n))
    utility_radius_values = tail_values if tail_values.size >= 2 else utilities
    utility_radius_summary = _empirical_bernstein_summary(
        utility_radius_values,
        delta=_gate_delta(config, "utility_gain"),
        n_eff=min(effective_n, int(utility_radius_values.size)),
        value_range=None,
    )
    utility_radius = float(utility_radius_summary["radius"])
    utility_gain_lcb = float(utility_gain - utility_radius) if np.isfinite(utility_radius) else float("nan")
    utility_gain_ucb = float(utility_gain + utility_radius) if np.isfinite(utility_radius) else float("nan")
    latency_gain_lcb = (
        float(latency_adjusted_gain - utility_radius) if np.isfinite(utility_radius) else float("nan")
    )
    latency_gain_ucb = (
        float(latency_adjusted_gain + utility_radius) if np.isfinite(utility_radius) else float("nan")
    )

    tail_summary = _empirical_bernstein_summary(
        tail_values,
        delta=_gate_delta(config, "tail_utility"),
        n_eff=tail_n_eff,
        value_range=None,
    )
    tail_harm_summary = _empirical_bernstein_summary(
        -tail_values,
        delta=_gate_delta(config, "tail_harm"),
        n_eff=tail_n_eff,
        value_range=None,
    )
    corr = score_utility_correlation(scores, utilities)
    tail_corr = tail_rank_correlation(scores, utilities, tail_fraction=config.tail_fraction)
    gain_ci = _bootstrap_high_n_gain_ci(
        scores,
        utilities,
        n_values,
        k_values,
        float(lambda_cost),
        runtime_measurements,
        config,
        seed=seed + 17,
    )
    tail_ci = _bootstrap_tail_corr_ci(scores, utilities, config, seed=seed + 31)
    return {
        "score_utility_correlation": float(corr),
        "tail_rank_correlation": float(tail_corr),
        "tail_rank_correlation_ci_low": float(tail_ci["ci_low"]),
        "tail_rank_correlation_ci_high": float(tail_ci["ci_high"]),
        "high_N_gain": float(latency_adjusted_gain),
        "high_N_gain_ci_low": float(gain_ci["ci_low"]),
        "high_N_gain_ci_high": float(gain_ci["ci_high"]),
        "raw_high_N_real_change": float(raw_curve[max(n_values)] - raw_curve[low_n]),
        "utility_gain_point": float(utility_gain),
        "utility_gain_lcb": float(utility_gain_lcb),
        "utility_gain_ucb": float(utility_gain_ucb),
        "tail_utility_lcb": float(tail_summary["lcb"]),
        "tail_utility_ucb": float(tail_summary["ucb"]),
        "tail_utility_mean": float(tail_summary["mean"]),
        "high_score_tail_harm_ucb": float(tail_harm_summary["ucb"]),
        "latency_adjusted_gain_point": float(latency_adjusted_gain),
        "latency_adjusted_gain_lcb": float(latency_gain_lcb),
        "latency_adjusted_gain_ucb": float(latency_gain_ucb),
        "empirical_bernstein_radius": float(utility_radius),
        "effective_n_for_bounds": int(effective_n),
        "tail_effective_n_for_bounds": int(tail_n_eff),
        "audit_units": int(scores.size),
        "tail_audit_units": int(np.sum(tail_mask)),
        "low_N_latency_adjusted_utility": float(adjusted[low_pair]),
        "best_high_N_latency_adjusted_utility": float(adjusted[high_pair]),
        "best_latency_adjusted_utility": float(adjusted[best_pair]),
        "best_high_N_pair": {"N": int(high_pair[0]), "K": int(high_pair[1])},
        "low_N_pair": {"N": int(low_pair[0]), "K": int(low_pair[1])},
        "best_pair": {"N": int(best_pair[0]), "K": int(best_pair[1])},
        "utility_bound_delta": float(_gate_delta(config, "utility_gain")),
        "tail_utility_bound_delta": float(_gate_delta(config, "tail_utility")),
        "latency_bound_delta": float(_gate_delta(config, "latency_adjusted_gain")),
        "tail_harm_bound_delta": float(_gate_delta(config, "tail_harm")),
    }


def _diversity_repair_recommendation(diversity: Mapping[str, float], config: AuditThenSampleConfig) -> str:
    collapse = float(diversity.get("duplicate_collapse_rate", float("nan")))
    entropy = float(diversity.get("trajectory_cluster_entropy", float("nan")))
    cluster_count = float(diversity.get("trajectory_cluster_count", float("nan")))
    if np.isfinite(collapse) and collapse > config.max_duplicate_collapse_rate:
        return RESAMPLE
    if np.isfinite(entropy) and entropy < 0.45:
        return CLUSTER_BALANCE
    if np.isfinite(cluster_count) and cluster_count < 2.0:
        return MODE_QUOTA
    return INCREASE_TEMPERATURE


def cluster_balanced_candidate_indices(
    trajectories: Any,
    scores: Iterable[float] | np.ndarray,
    *,
    max_candidates: int | None = None,
) -> np.ndarray:
    """Return candidate indices interleaving trajectory clusters before reranking."""

    score_arr = _as_1d(scores, "scores")
    traj_arr = np.asarray(trajectories, dtype=float)
    if traj_arr.shape[0] != score_arr.size:
        raise ValueError("trajectories and scores must have the same candidate dimension")
    labels = trajectory_cluster_ids(traj_arr)
    cluster_ids = sorted({int(label) for label in labels})
    per_cluster: dict[int, list[int]] = {}
    for cluster_id in cluster_ids:
        members = np.flatnonzero(labels == cluster_id)
        ordered = members[np.argsort(score_arr[members], kind="mergesort")[::-1]]
        per_cluster[cluster_id] = [int(i) for i in ordered]
    out: list[int] = []
    while any(per_cluster.values()):
        for cluster_id in cluster_ids:
            if per_cluster[cluster_id]:
                out.append(per_cluster[cluster_id].pop(0))
                if max_candidates is not None and len(out) >= int(max_candidates):
                    return np.asarray(out, dtype=int)
    return np.asarray(out, dtype=int)


def fit_affine_score_calibration(
    scores: Iterable[float] | np.ndarray,
    utilities: Iterable[float] | np.ndarray,
    *,
    ridge: float = 1e-8,
) -> AffineCalibration:
    """Fit an affine score-to-utility repair on audited candidates."""

    s = _as_1d(scores, "scores")
    u = _as_1d(utilities, "utilities")
    if s.shape != u.shape:
        raise ValueError("scores and utilities must have the same shape")
    x = np.column_stack([s, np.ones_like(s)])
    xtx = x.T @ x + float(ridge) * np.eye(2)
    slope, intercept = np.linalg.solve(xtx, x.T @ u)
    return AffineCalibration(float(slope), float(intercept))


def _pava(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    levels: list[float] = []
    level_weights: list[float] = []
    starts: list[int] = []
    stops: list[int] = []
    for i, (value, weight) in enumerate(zip(y, weights, strict=True)):
        levels.append(float(value))
        level_weights.append(float(weight))
        starts.append(i)
        stops.append(i + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_weight = level_weights[-2] + level_weights[-1]
            merged = (levels[-2] * level_weights[-2] + levels[-1] * level_weights[-1]) / total_weight
            levels[-2] = float(merged)
            level_weights[-2] = float(total_weight)
            stops[-2] = stops[-1]
            levels.pop()
            level_weights.pop()
            starts.pop()
            stops.pop()
    fitted = np.empty_like(y, dtype=float)
    for level, start, stop in zip(levels, starts, stops, strict=True):
        fitted[start:stop] = level
    return fitted


def fit_isotonic_score_calibration(
    scores: Iterable[float] | np.ndarray,
    utilities: Iterable[float] | np.ndarray,
) -> IsotonicCalibration:
    """Fit a monotone nondecreasing isotonic score-to-utility repair."""

    s = _as_1d(scores, "scores")
    u = _as_1d(utilities, "utilities")
    if s.shape != u.shape:
        raise ValueError("scores and utilities must have the same shape")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_utilities = u[order]
    thresholds: list[float] = []
    means: list[float] = []
    weights: list[float] = []
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        thresholds.append(float(sorted_scores[i]))
        means.append(float(np.mean(sorted_utilities[i:j])))
        weights.append(float(j - i))
        i = j
    fitted = _pava(np.asarray(means, dtype=float), np.asarray(weights, dtype=float))
    return IsotonicCalibration(np.asarray(thresholds, dtype=float), fitted)


def _calibration_residual_radius(
    calibration: AffineCalibration | IsotonicCalibration,
    scores: np.ndarray,
    utilities: np.ndarray,
    config: AuditThenSampleConfig,
) -> float:
    residuals = utilities - calibration.predict(scores)
    summary = _empirical_bernstein_summary(
        np.abs(residuals),
        delta=_gate_delta(config, "repair"),
        n_eff=residuals.size,
        value_range=None,
    )
    radius = float(summary["ucb"])
    return radius if np.isfinite(radius) else float("inf")


def _validate_single_repair(
    calibration: AffineCalibration | IsotonicCalibration,
    repair_method: str,
    original_scores: np.ndarray,
    holdout_utilities: np.ndarray,
    pilot_scores: np.ndarray,
    pilot_utilities: np.ndarray,
    ns: list[int],
    config: AuditThenSampleConfig,
    *,
    seed: int,
    min_tail_improvement: float,
    min_repaired_tail_correlation: float,
) -> CalibrationRepairResult:
    repaired_pred = calibration.predict(original_scores)
    residual_radius = _calibration_residual_radius(calibration, pilot_scores, pilot_utilities, config)
    repaired_scores = repaired_pred - residual_radius
    effective_n = int(holdout_utilities.size)
    original_metrics = _gate_metrics(
        original_scores,
        holdout_utilities,
        ns,
        [1],
        0.0,
        None,
        config,
        effective_sample_diversity=float(effective_n),
        seed=seed + 101,
    )
    repaired_metrics = _gate_metrics(
        repaired_scores,
        holdout_utilities,
        ns,
        [1],
        0.0,
        None,
        config,
        effective_sample_diversity=float(effective_n),
        seed=seed + 211,
    )
    original_tail = float(original_metrics["tail_rank_correlation"])
    repaired_tail = float(repaired_metrics["tail_rank_correlation"])
    original_gain = float(original_metrics["utility_gain_point"])
    repaired_gain = float(repaired_metrics["utility_gain_point"])
    repaired_gate_passed = (
        np.isfinite(repaired_tail)
        and repaired_tail >= float(min_repaired_tail_correlation)
        and repaired_tail - original_tail >= float(min_tail_improvement)
        and repaired_gain >= original_gain
        and float(repaired_metrics["utility_gain_lcb"]) > config.min_effect_size
        and float(repaired_metrics["tail_utility_lcb"]) > config.min_effect_size
        and float(repaired_metrics["latency_adjusted_gain_lcb"]) > config.min_effect_size
        and float(repaired_metrics["high_score_tail_harm_ucb"]) <= config.min_block_harm
    )
    return CalibrationRepairResult(
        success=bool(repaired_gate_passed),
        original_tail_rank_correlation=original_tail,
        repaired_tail_rank_correlation=repaired_tail,
        original_high_n_gain=original_gain,
        repaired_high_n_gain=repaired_gain,
        calibration=calibration,
        pilot_count=int(pilot_scores.size),
        holdout_count=int(holdout_utilities.size),
        recommendation=INCREASE_N if repaired_gate_passed else BLOCK_HIGH_N,
        repair_method=repair_method,
        risk_delta=float(config.risk_delta),
        effective_n_for_bounds=int(repaired_metrics["effective_n_for_bounds"]),
        original_utility_gain_lcb=float(original_metrics["utility_gain_lcb"]),
        repaired_utility_gain_lcb=float(repaired_metrics["utility_gain_lcb"]),
        original_tail_utility_lcb=float(original_metrics["tail_utility_lcb"]),
        repaired_tail_utility_lcb=float(repaired_metrics["tail_utility_lcb"]),
        original_latency_adjusted_gain_lcb=float(original_metrics["latency_adjusted_gain_lcb"]),
        repaired_latency_adjusted_gain_lcb=float(repaired_metrics["latency_adjusted_gain_lcb"]),
        repaired_score_lcb_radius=float(residual_radius),
    )


def validate_repair_with_bounds(
    scores: Iterable[float] | np.ndarray,
    utilities: Iterable[float] | np.ndarray,
    *,
    n_values: Iterable[int],
    pilot_fraction: float = 0.50,
    seed: int = 0,
    min_tail_improvement: float = 0.25,
    min_repaired_tail_correlation: float = 0.20,
    config: AuditThenSampleConfig | None = None,
) -> CalibrationRepairResult:
    """Fit on pilot candidates and validate repair with held-out EB gates."""

    cfg = config or AuditThenSampleConfig(use_effective_n_for_bounds=False)
    s = _as_1d(scores, "scores")
    u = _as_1d(utilities, "utilities")
    if s.shape != u.shape:
        raise ValueError("scores and utilities must have the same shape")
    if s.size < 6:
        raise ValueError("at least six candidates are required for a repair audit")
    ns = _as_n_values(n_values, s.size)
    rng = np.random.default_rng(seed)
    order = rng.permutation(s.size)
    pilot_count = min(s.size - 2, max(3, int(np.ceil(s.size * float(pilot_fraction)))))
    pilot_idx = order[:pilot_count]
    holdout_idx = order[pilot_count:]
    pilot_scores = s[pilot_idx]
    pilot_utilities = u[pilot_idx]
    holdout_scores = s[holdout_idx]
    holdout_utilities = u[holdout_idx]
    candidates: list[CalibrationRepairResult] = []
    if cfg.repair_method in {"isotonic", "auto"}:
        isotonic = fit_isotonic_score_calibration(pilot_scores, pilot_utilities)
        candidates.append(
            _validate_single_repair(
                isotonic,
                "isotonic",
                holdout_scores,
                holdout_utilities,
                pilot_scores,
                pilot_utilities,
                ns,
                cfg,
                seed=seed,
                min_tail_improvement=min_tail_improvement,
                min_repaired_tail_correlation=min_repaired_tail_correlation,
            )
        )
    if cfg.repair_method in {"affine", "auto"}:
        affine = fit_affine_score_calibration(pilot_scores, pilot_utilities)
        candidates.append(
            _validate_single_repair(
                affine,
                "affine",
                holdout_scores,
                holdout_utilities,
                pilot_scores,
                pilot_utilities,
                ns,
                cfg,
                seed=seed + 19,
                min_tail_improvement=min_tail_improvement,
                min_repaired_tail_correlation=min_repaired_tail_correlation,
            )
        )
    successful = [item for item in candidates if item.success]
    if successful:
        return max(
            successful,
            key=lambda item: (
                item.repaired_latency_adjusted_gain_lcb,
                item.repaired_tail_utility_lcb,
                item.repaired_tail_rank_correlation,
            ),
        )
    return max(
        candidates,
        key=lambda item: (
            item.repaired_latency_adjusted_gain_lcb,
            item.repaired_tail_utility_lcb,
            item.repaired_tail_rank_correlation,
        ),
    )


def validate_calibration_repair(
    scores: Iterable[float] | np.ndarray,
    utilities: Iterable[float] | np.ndarray,
    *,
    n_values: Iterable[int],
    pilot_fraction: float = 0.50,
    seed: int = 0,
    min_tail_improvement: float = 0.25,
    min_repaired_tail_correlation: float = 0.20,
) -> CalibrationRepairResult:
    """Backward-compatible wrapper for the stronger held-out repair audit."""

    return validate_repair_with_bounds(
        scores,
        utilities,
        n_values=n_values,
        pilot_fraction=pilot_fraction,
        seed=seed,
        min_tail_improvement=min_tail_improvement,
        min_repaired_tail_correlation=min_repaired_tail_correlation,
        config=AuditThenSampleConfig(use_effective_n_for_bounds=False),
    )


def audit_then_sample(
    trajectories: Any,
    scores: Iterable[float] | np.ndarray,
    utilities: Iterable[float] | np.ndarray | None = None,
    *,
    n_values: Iterable[int] | None = None,
    k_values: Iterable[int] | None = None,
    lambda_cost: float = 0.0,
    runtime_measurements: Mapping[Any, float] | None = None,
    diversity_diagnostics: Mapping[str, float] | None = None,
    sampler_metadata: Mapping[str, Any] | None = None,
    config: AuditThenSampleConfig | None = None,
    seed: int = 0,
) -> AuditThenSampleResult:
    """Choose `N,K` and return a conservative high-N action recommendation."""

    cfg = config or AuditThenSampleConfig()
    score_arr = _as_1d(scores, "scores")
    traj_arr = np.asarray(trajectories, dtype=float)
    if traj_arr.shape[0] != score_arr.size:
        raise ValueError("trajectories and scores must have the same candidate dimension")
    if cfg.cluster_balance_candidates:
        order = cluster_balanced_candidate_indices(traj_arr, score_arr)
        traj_arr = traj_arr[order]
        score_arr = score_arr[order]
        if utilities is not None:
            utilities = np.asarray(utilities, dtype=float)[order]
    ns = _as_n_values(n_values, score_arr.size)
    ks = _as_k_values(k_values, sampler_metadata)
    low_n = min(ns)
    min_k = min(ks)

    div = dict(diversity_diagnostics or diversity_summary(traj_arr))
    eff_div = float(div.get("effective_sample_diversity", float("nan")))
    collapse = float(div.get("duplicate_collapse_rate", float("nan")))
    diversity_ok = bool(
        eff_div >= cfg.min_effective_diversity
        and collapse <= cfg.max_duplicate_collapse_rate
    )
    base_diag: dict[str, Any] = {
        "low_N": int(low_n),
        "max_N": int(max(ns)),
        "min_K": int(min_k),
        "max_K": int(max(ks)),
        "lambda_cost": float(lambda_cost),
        "risk_delta": float(cfg.risk_delta),
        "min_effect_size": float(cfg.min_effect_size),
        "confidence_method": cfg.confidence_method,
        "repair_method": cfg.repair_method,
        "use_effective_n_for_bounds": bool(cfg.use_effective_n_for_bounds),
        "effective_sample_diversity": eff_div,
        "duplicate_collapse_rate": collapse,
        "diversity_gate_passed": diversity_ok,
        "admit_high_N": False,
        "abstention_reason": "",
        "false_admit_negative_control": False,
    }
    if cfg.cluster_balance_candidates:
        base_diag["cluster_balance_applied"] = True
    if sampler_metadata:
        base_diag["sampler_metadata"] = dict(sampler_metadata)

    if not diversity_ok:
        recommendation = _diversity_repair_recommendation(div, cfg)
        base_diag["abstention_reason"] = "diversity_gate_failed"
        base_diag["diversity_repair_recommendation"] = recommendation
        return AuditThenSampleResult(
            selected_n=low_n,
            selected_k=min_k,
            decision_label=LOW_DIVERSITY_STOP,
            action_recommendation=INCREASE_DIVERSITY,
            confidence_diagnostics=base_diag,
        )

    effective_n = _bound_n(
        audit_n=int(score_arr.size),
        effective_sample_diversity=eff_div,
        config=cfg,
    )
    base_diag["effective_n_for_bounds"] = int(effective_n)
    if utilities is None:
        base_diag["alignment_gate_passed"] = False
        base_diag["alignment_reason"] = "no_measured_rollout_utilities"
        base_diag["abstention_reason"] = "unknown_utility"
        return AuditThenSampleResult(
            selected_n=low_n,
            selected_k=min_k,
            decision_label=ALIGNMENT_AUDIT_FAILED,
            action_recommendation=AUDIT_ROLLOUTS,
            confidence_diagnostics=base_diag,
        )

    utility_arr = _as_1d(utilities, "utilities")
    if utility_arr.shape != score_arr.shape:
        raise ValueError("scores and utilities must have the same shape")
    if float(lambda_cost) < 0.0:
        raise ValueError("lambda_cost must be non-negative")

    metrics = _gate_metrics(
        score_arr,
        utility_arr,
        ns,
        ks,
        float(lambda_cost),
        runtime_measurements,
        cfg,
        effective_sample_diversity=eff_div,
        seed=seed,
    )
    base_diag.update(metrics)
    underpowered = (
        int(metrics["effective_n_for_bounds"]) < int(cfg.min_audit_units)
        or int(metrics["tail_effective_n_for_bounds"]) < int(cfg.min_audit_units)
    )
    base_diag["underpowered_audit"] = bool(underpowered)
    if underpowered:
        base_diag["alignment_gate_passed"] = False
        base_diag["latency_gate_passed"] = False
        base_diag["abstention_reason"] = "underpowered_effective_n"
        low_pair = metrics["low_N_pair"]
        return AuditThenSampleResult(
            selected_n=int(low_pair["N"]),
            selected_k=int(low_pair["K"]),
            decision_label=UNDERPOWERED_AUDIT,
            action_recommendation=AUDIT_ROLLOUTS,
            confidence_diagnostics=base_diag,
        )

    corr = float(metrics["score_utility_correlation"])
    tail_corr = float(metrics["tail_rank_correlation"])
    utility_gain_lcb = float(metrics["utility_gain_lcb"])
    tail_utility_lcb = float(metrics["tail_utility_lcb"])
    latency_gain_lcb = float(metrics["latency_adjusted_gain_lcb"])
    tail_harm_ucb = float(metrics["high_score_tail_harm_ucb"])
    raw_high_change = float(metrics["raw_high_N_real_change"])
    hard_negative = (
        (np.isfinite(corr) and corr < -cfg.min_score_utility_correlation)
        or (np.isfinite(tail_corr) and tail_corr < -0.05)
        or raw_high_change < -max(cfg.min_block_harm, 0.0)
        or (np.isfinite(tail_harm_ucb) and tail_harm_ucb > cfg.min_block_harm)
    )
    if hard_negative:
        base_diag["alignment_gate_passed"] = False
        base_diag["latency_gate_passed"] = False
        base_diag["abstention_reason"] = "tail_harm_plausible"
        low_pair = metrics["low_N_pair"]
        return AuditThenSampleResult(
            selected_n=int(low_pair["N"]),
            selected_k=int(low_pair["K"]),
            decision_label=HIGH_N_BLOCKED,
            action_recommendation=BLOCK_HIGH_N,
            confidence_diagnostics=base_diag,
        )

    alignment_ok = (
        np.isfinite(corr)
        and np.isfinite(tail_corr)
        and np.isfinite(tail_utility_lcb)
        and corr >= cfg.min_score_utility_correlation
        and tail_corr >= cfg.min_tail_rank_correlation
        and tail_utility_lcb > cfg.min_effect_size
    )
    base_diag["alignment_gate_passed"] = bool(alignment_ok)
    if not alignment_ok:
        base_diag["abstention_reason"] = "alignment_or_tail_utility_lcb_failed"
        low_pair = metrics["low_N_pair"]
        return AuditThenSampleResult(
            selected_n=int(low_pair["N"]),
            selected_k=int(low_pair["K"]),
            decision_label=ALIGNMENT_AUDIT_FAILED,
            action_recommendation=CALIBRATE_SCORER,
            confidence_diagnostics=base_diag,
        )

    utility_ok = np.isfinite(utility_gain_lcb) and utility_gain_lcb > cfg.min_effect_size
    latency_ok = np.isfinite(latency_gain_lcb) and latency_gain_lcb > cfg.min_effect_size
    base_diag["utility_gain_gate_passed"] = bool(utility_ok)
    base_diag["latency_gate_passed"] = bool(latency_ok)
    if not utility_ok or not latency_ok:
        best_pair = metrics["best_pair"]
        low_pair = metrics["low_N_pair"]
        adjusted_best = float(metrics["best_latency_adjusted_utility"])
        adjusted_low = float(metrics["low_N_latency_adjusted_utility"])
        action = REDUCE_K if int(best_pair["K"]) < max(ks) and int(best_pair["N"]) > low_n else STOP_EARLY
        base_diag["abstention_reason"] = (
            "utility_gain_lcb_failed" if not utility_ok else "latency_adjusted_gain_lcb_failed"
        )
        return AuditThenSampleResult(
            selected_n=int(best_pair["N"]) if adjusted_best >= adjusted_low else int(low_pair["N"]),
            selected_k=int(best_pair["K"]) if adjusted_best >= adjusted_low else int(low_pair["K"]),
            decision_label=LATENCY_LIMITED,
            action_recommendation=action,
            confidence_diagnostics=base_diag,
        )

    high_pair = metrics["best_high_N_pair"]
    base_diag["admit_high_N"] = True
    return AuditThenSampleResult(
        selected_n=int(high_pair["N"]),
        selected_k=int(high_pair["K"]),
        decision_label=ALLOW_HIGH_N,
        action_recommendation=INCREASE_N,
        confidence_diagnostics=base_diag,
    )


def audit_then_sample_adaptive(
    trajectories: Any,
    scores: Iterable[float] | np.ndarray,
    utilities: Iterable[float] | np.ndarray | None = None,
    *,
    batch_size: int = 16,
    n_values: Iterable[int] | None = None,
    k_values: Iterable[int] | None = None,
    lambda_cost: float = 0.0,
    runtime_measurements: Mapping[Any, float] | None = None,
    diversity_diagnostics: Mapping[str, float] | None = None,
    sampler_metadata: Mapping[str, Any] | None = None,
    config: AuditThenSampleConfig | None = None,
    seed: int = 0,
) -> AuditThenSampleResult:
    """Batchwise Audit-Then-Sample with conservative early stopping."""

    cfg = config or AuditThenSampleConfig()
    score_arr = _as_1d(scores, "scores")
    traj_arr = np.asarray(trajectories, dtype=float)
    if traj_arr.shape[0] != score_arr.size:
        raise ValueError("trajectories and scores must have the same candidate dimension")
    if utilities is not None:
        utility_arr: np.ndarray | None = _as_1d(utilities, "utilities")
        if utility_arr.shape != score_arr.shape:
            raise ValueError("scores and utilities must have the same shape")
    else:
        utility_arr = None
    max_n = int(score_arr.size)
    checkpoints = list(_as_n_values(n_values, max_n) if n_values is not None else range(int(batch_size), max_n + 1, int(batch_size)))
    if checkpoints[-1] != max_n:
        checkpoints.append(max_n)
    checkpoints = sorted({int(n) for n in checkpoints if int(n) >= 1 and int(n) <= max_n})
    min_checkpoint = max(int(batch_size), int(cfg.min_audit_units), 2)
    final_result: AuditThenSampleResult | None = None
    for checkpoint in checkpoints:
        if checkpoint < min_checkpoint and checkpoint != max_n:
            continue
        prefix_n_values = [n for n in checkpoints if n <= checkpoint]
        if 1 not in prefix_n_values:
            prefix_n_values.insert(0, 1)
        prefix_diversity = dict(diversity_diagnostics or diversity_summary(traj_arr[:checkpoint]))
        if diversity_diagnostics and "effective_sample_diversity" in diversity_diagnostics:
            prefix_diversity["effective_sample_diversity"] = min(
                float(diversity_diagnostics["effective_sample_diversity"]),
                float(checkpoint),
            )
        result = audit_then_sample(
            traj_arr[:checkpoint],
            score_arr[:checkpoint],
            None if utility_arr is None else utility_arr[:checkpoint],
            n_values=prefix_n_values,
            k_values=k_values,
            lambda_cost=lambda_cost,
            runtime_measurements=runtime_measurements,
            diversity_diagnostics=prefix_diversity,
            sampler_metadata=sampler_metadata,
            config=cfg,
            seed=seed + checkpoint,
        )
        diag = dict(result.confidence_diagnostics)
        diag["adaptive_checked_N"] = int(checkpoint)
        diag["adaptive_max_N"] = int(max_n)
        diag["adaptive_stopping_savings"] = float(max(0.0, 1.0 - checkpoint / max(max_n, 1)))
        result = AuditThenSampleResult(
            selected_n=result.selected_n,
            selected_k=result.selected_k,
            decision_label=result.decision_label,
            action_recommendation=result.action_recommendation,
            confidence_diagnostics=diag,
        )
        final_result = result
        gain_ucb = float(diag.get("latency_adjusted_gain_ucb", float("nan")))
        if np.isfinite(gain_ucb) and gain_ucb <= cfg.min_effect_size:
            diag["adaptive_stopped"] = True
            diag["abstention_reason"] = "adaptive_gain_ucb_below_cost"
            return AuditThenSampleResult(
                selected_n=min(prefix_n_values),
                selected_k=result.selected_k,
                decision_label=LATENCY_LIMITED,
                action_recommendation=STOP_EARLY,
                confidence_diagnostics=diag,
            )
        if result.action_recommendation in {BLOCK_HIGH_N, INCREASE_DIVERSITY, AUDIT_ROLLOUTS}:
            diag["adaptive_stopped"] = True
            return result
    if final_result is None:
        return audit_then_sample(
            traj_arr,
            score_arr,
            utility_arr,
            n_values=n_values,
            k_values=k_values,
            lambda_cost=lambda_cost,
            runtime_measurements=runtime_measurements,
            diversity_diagnostics=diversity_diagnostics,
            sampler_metadata=sampler_metadata,
            config=cfg,
            seed=seed,
        )
    final_diag = dict(final_result.confidence_diagnostics)
    final_diag.setdefault("adaptive_stopped", False)
    return AuditThenSampleResult(
        selected_n=final_result.selected_n,
        selected_k=final_result.selected_k,
        decision_label=final_result.decision_label,
        action_recommendation=final_result.action_recommendation,
        confidence_diagnostics=final_diag,
    )
