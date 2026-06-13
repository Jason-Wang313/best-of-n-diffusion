from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS = Path(os.environ.get("DIFFUSION_AUDIT_RESULTS_DIR", ROOT / "results")).expanduser()
if not RESULTS.is_absolute():
    RESULTS = ROOT / RESULTS


def load_json(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def csv_columns(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return set(csv.DictReader(handle).fieldnames or [])


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def ci_units(row: dict[str, Any]) -> int:
    try:
        return int(float(row.get("n", 0)))
    except (TypeError, ValueError):
        return 0


def status(strong: bool, weak: bool = False) -> str:
    if strong:
        return "SUPPORTED"
    if weak:
        return "PARTIAL"
    return "UNSUPPORTED"


def is_smoke_results() -> bool:
    return any(part.lower() == "smoke" for part in RESULTS.parts)


def first_row(rows: list[dict[str, str]], **matches: Any) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in matches.items()):
            return row
    return {}


def ci_ok(
    row: dict[str, Any],
    *,
    mean_min: float | None = None,
    mean_max: float | None = None,
    low_min: float | None = None,
    high_max: float | None = None,
    min_n: int = 2,
) -> bool:
    if not row:
        return False
    n = int(f(row, "n", 0.0))
    if n < int(min_n):
        return False
    mean = f(row, "mean")
    lo = f(row, "ci_low")
    hi = f(row, "ci_high")
    if mean_min is not None and mean < float(mean_min):
        return False
    if mean_max is not None and mean > float(mean_max):
        return False
    if low_min is not None and lo < float(low_min):
        return False
    if high_max is not None and hi > float(high_max):
        return False
    return True


def add(claims: list[dict[str, Any]], category: str, claim: str, stat: str, evidence: Any) -> None:
    claims.append(
        {
            "id": len(claims) + 1,
            "category": category,
            "claim": claim,
            "status": stat,
            "evidence": evidence if isinstance(evidence, str) else json.dumps(evidence, sort_keys=True),
        }
    )


def forbidden_hits() -> list[dict[str, Any]]:
    surfaces = [
        ROOT / "README.md",
        ROOT / "docs" / "claims.md",
        ROOT / "docs" / "theory.md",
        ROOT / "paper" / "intro.md",
        ROOT / "paper" / "limitations.md",
        ROOT / "paper" / "draft.md",
    ]
    patterns = [
        re.compile(r"\bvalidated on real robots\b", re.I),
        re.compile(r"\breal[- ]robot validation\b", re.I),
        re.compile(r"\buniversal diffusion policy improvement\b", re.I),
        re.compile(r"\btrajectory search always helps\b", re.I),
        re.compile(r"\bsolves robot manipulation\b", re.I),
        re.compile(r"\bfull[- ]visual diffusion policy validation\b", re.I),
        re.compile(r"\bfull[- ]scale visual manipulation validation\b", re.I),
        re.compile(r"\bproduction[- ]scale visual policy quality\b", re.I),
    ]
    guards = ["do not claim", "not claim", "forbidden", "unsupported", "limitation", "future work"]
    hits: list[dict[str, Any]] = []
    for path in surfaces:
        text = read_text(path)
        guarded_section = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            lower = line.lower()
            if line.lstrip().startswith("#"):
                guarded_section = any(g in lower for g in guards)
            if guarded_section or any(g in lower for g in guards):
                continue
            for pat in patterns:
                if pat.search(line):
                    hits.append({"path": str(path.relative_to(ROOT)), "line": lineno, "text": line.strip()})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    controlled = load_json("controlled_sampler_summary.json")
    controller = load_json("audit_then_sample_summary.json")
    scorer = load_json("scorer_comparison_summary.json")
    nk = load_json("nk_budget_summary.json")
    learned = load_json("learned_policy_lite_summary.json")
    true_diffusion = load_json("true_diffusion_summary.json")
    pusht = load_json("pusht_summary.json")
    controlled_agg = csv_rows(RESULTS / "tables" / "controlled_sampler_aggregate.csv")
    controlled_div = csv_rows(RESULTS / "tables" / "controlled_sampler_diversity.csv")
    controlled_effect_cis = csv_rows(RESULTS / "tables" / "controlled_sampler_effect_cis.csv")
    controller_decisions = csv_rows(RESULTS / "tables" / "audit_then_sample_decisions.csv")
    controller_calibration = csv_rows(RESULTS / "tables" / "audit_then_sample_calibration.csv")
    scorer_agg = csv_rows(RESULTS / "tables" / "scorer_comparison_aggregate.csv")
    scorer_effect_cis = csv_rows(RESULTS / "tables" / "scorer_comparison_effect_cis.csv")
    calibration_map = csv_rows(RESULTS / "tables" / "calibration_repair_map.csv")
    nk_latency_effect = csv_rows(RESULTS / "tables" / "nk_budget_latency_effect_ci.csv")
    learned_agg = csv_rows(RESULTS / "tables" / "learned_policy_lite_aggregate.csv")
    learned_seed_agg = csv_rows(RESULTS / "tables" / "learned_policy_lite_seed_aggregate.csv")
    learned_effect_cis = csv_rows(RESULTS / "tables" / "learned_policy_lite_effect_cis.csv")
    learned_training = csv_rows(RESULTS / "tables" / "learned_policy_lite_training.csv")
    learned_receding = csv_rows(RESULTS / "tables" / "learned_policy_lite_receding_horizon.csv")
    true_effect_cis = csv_rows(RESULTS / "tables" / "true_diffusion_effect_cis.csv")
    true_gap_cis = csv_rows(RESULTS / "tables" / "true_diffusion_scorer_gap_cis.csv")
    true_training = csv_rows(RESULTS / "tables" / "true_diffusion_training.csv")
    true_runtime = csv_rows(RESULTS / "tables" / "true_diffusion_runtime.csv")
    true_agg = csv_rows(RESULTS / "tables" / "true_diffusion_aggregate.csv")
    true_sampler_comparison = csv_rows(RESULTS / "tables" / "true_diffusion_sampler_comparison.csv")
    pusht_effect_cis = csv_rows(RESULTS / "tables" / "pusht_effect_cis.csv")
    pusht_gap_cis = csv_rows(RESULTS / "tables" / "pusht_scorer_gap_cis.csv")
    pusht_training = csv_rows(RESULTS / "tables" / "pusht_training.csv")
    pusht_runtime = csv_rows(RESULTS / "tables" / "pusht_runtime.csv")
    pusht_rollouts = csv_rows(RESULTS / "tables" / "pusht_rollouts.csv")
    pusht_agg = csv_rows(RESULTS / "tables" / "pusht_aggregate.csv")
    pusht_rollout_metric_effect_cis = csv_rows(RESULTS / "tables" / "pusht_rollout_metric_effect_cis.csv")
    pusht_rollout_metric_seed_agg = csv_rows(RESULTS / "tables" / "pusht_rollout_metric_seed_aggregate.csv")
    pusht_rollout_metric_agg = csv_rows(RESULTS / "tables" / "pusht_rollout_metric_aggregate.csv")
    diff_doc = read_text(ROOT / "docs" / "differentiation_from_wam_jepa.md").lower()
    checklist_doc = read_text(ROOT / "docs" / "diffusion_policy_validity_checklist.md").lower()
    theory_doc = read_text(ROOT / "docs" / "theory.md").lower()

    claims: list[dict[str, Any]] = []
    strong_metrics: dict[str, Any] = {}

    def agg_row(rows: list[dict[str, str]], **matches: Any) -> dict[str, str]:
        for row in rows:
            ok = True
            for key, value in matches.items():
                if str(row.get(key)) != str(value):
                    ok = False
                    break
            if ok:
                return row
        return {}

    theorem_ok = (
        (ROOT / "src" / "diffusion_audit" / "theory.py").exists()
        and (ROOT / "tests" / "test_theory.py").exists()
        and "tie-aware" in theory_doc
        and "finite" in theory_doc
        and "test_anti_aligned_score_degrades_selected_real_utility" in read_text(ROOT / "tests" / "test_theory.py")
        and "test_tie_aware_max_selection_uses_tie_group_mean_utility" in read_text(ROOT / "tests" / "test_theory.py")
    )
    add(
        claims,
        "finite_law",
        "Finite tie-aware trajectory-selection law is implemented and tested.",
        status(theorem_ok),
        "src/diffusion_audit/theory.py; tests/test_theory.py; docs/theory.md",
    )

    aligned_high = agg_row(
        controlled_agg,
        regime="high_diversity_aligned",
        scorer="aligned",
        N=max(controlled.get("n_values") or [64]),
    )
    aligned_weak = (
        controlled.get("aligned_score_gain_high_minus_low", 0.0) > 0.01
        and controlled.get("aligned_real_gain_high_minus_low", 0.0) > 0.01
    )
    aligned_real_ci = first_row(
        controlled_effect_cis,
        regime="high_diversity_aligned",
        metric="exact_selected_real",
    )
    aligned_score_ci = first_row(
        controlled_effect_cis,
        regime="high_diversity_aligned",
        metric="exact_selected_score",
    )
    aligned_strong = (
        controlled.get("aligned_score_gain_high_minus_low", 0.0) >= 0.50
        and controlled.get("aligned_real_gain_high_minus_low", 0.0) >= 0.50
        and f(aligned_high, "score_utility_correlation") >= 0.99
        and f(aligned_high, "score_utility_correlation_ci_low") >= 0.985
        and f(aligned_high, "tail_rank_correlation") >= 0.95
        and f(aligned_high, "tail_rank_correlation_ci_low") >= 0.90
        and f(aligned_high, "high_n_regret") <= 0.01
        and ci_ok(aligned_real_ci, mean_min=0.50, low_min=0.35, min_n=3)
        and ci_ok(aligned_score_ci, mean_min=0.50, low_min=0.35, min_n=3)
    )
    strong_metrics["aligned_selection"] = {
        "score_gain": controlled.get("aligned_score_gain_high_minus_low"),
        "real_gain": controlled.get("aligned_real_gain_high_minus_low"),
        "real_gain_ci": aligned_real_ci,
        "score_gain_ci": aligned_score_ci,
        "correlation": f(aligned_high, "score_utility_correlation"),
        "correlation_ci_low": f(aligned_high, "score_utility_correlation_ci_low"),
        "tail_rank_correlation": f(aligned_high, "tail_rank_correlation"),
        "tail_rank_correlation_ci_low": f(aligned_high, "tail_rank_correlation_ci_low"),
        "high_n_regret": f(aligned_high, "high_n_regret"),
        "thresholds": {
            "score_gain_min": 0.50,
            "real_gain_min": 0.50,
            "ci_low_gain_min": 0.35,
            "correlation_min": 0.99,
            "correlation_ci_low_min": 0.985,
            "tail_rank_min": 0.95,
            "tail_rank_ci_low_min": 0.90,
            "regret_max": 0.01,
        },
    }
    add(
        claims,
        "controlled_sampler",
        "High N can help aligned diffusion trajectory selection.",
        status(aligned_strong, aligned_weak),
        strong_metrics["aligned_selection"],
    )

    misaligned_weak = (
        controlled.get("misaligned_score_gain_high_minus_low", 0.0) > 0.01
        and controlled.get("misaligned_real_change_high_minus_low", 1.0) <= 0.02
        and controlled.get("misaligned_high_n_regret", 0.0) > 0.05
    ) or (
        scorer.get("misaligned_score_gain_high_minus_low", 0.0) > 0.01
        and scorer.get("misaligned_real_change_high_minus_low", 1.0) <= 0.02
    )
    controlled_misaligned_real_ci = first_row(
        controlled_effect_cis,
        regime="high_diversity_misaligned",
        metric="exact_selected_real",
    )
    controlled_misaligned_score_ci = first_row(
        controlled_effect_cis,
        regime="high_diversity_misaligned",
        metric="exact_selected_score",
    )
    scorer_misaligned_real_ci = first_row(
        scorer_effect_cis,
        regime="hidden_obstacle_high_diversity",
        effect="high_minus_low",
        metric="exact_selected_real",
    )
    scorer_misaligned_score_ci = first_row(
        scorer_effect_cis,
        regime="hidden_obstacle_high_diversity",
        effect="high_minus_low",
        metric="exact_selected_score",
    )
    scorer_oracle_gap_ci = first_row(
        scorer_effect_cis,
        regime="hidden_obstacle_high_diversity",
        effect="oracle_real_utility_selector_minus_misaligned_tail_scorer",
    )
    misaligned_strong = (
        controlled.get("misaligned_score_gain_high_minus_low", 0.0) >= 0.35
        and controlled.get("misaligned_real_change_high_minus_low", 1.0) <= -0.25
        and controlled.get("misaligned_high_n_regret", 0.0) >= 0.80
        and scorer.get("misaligned_score_gain_high_minus_low", 0.0) >= 0.35
        and scorer.get("misaligned_real_change_high_minus_low", 1.0) <= -0.25
        and scorer.get("oracle_minus_misaligned_high_n", 0.0) >= 0.90
        and ci_ok(controlled_misaligned_score_ci, mean_min=0.35, low_min=0.20, min_n=3)
        and ci_ok(controlled_misaligned_real_ci, mean_max=-0.25, high_max=-0.12, min_n=3)
        and ci_ok(scorer_misaligned_score_ci, mean_min=0.35, low_min=0.20, min_n=3)
        and ci_ok(scorer_misaligned_real_ci, mean_max=-0.25, high_max=-0.12, min_n=3)
        and ci_ok(scorer_oracle_gap_ci, mean_min=0.90, low_min=0.65, min_n=3)
    )
    strong_metrics["misaligned_tail_selection"] = {
        "controlled_score_gain": controlled.get("misaligned_score_gain_high_minus_low"),
        "controlled_real_change": controlled.get("misaligned_real_change_high_minus_low"),
        "controlled_high_n_regret": controlled.get("misaligned_high_n_regret"),
        "controlled_score_gain_ci": controlled_misaligned_score_ci,
        "controlled_real_change_ci": controlled_misaligned_real_ci,
        "scorer_score_gain": scorer.get("misaligned_score_gain_high_minus_low"),
        "scorer_real_change": scorer.get("misaligned_real_change_high_minus_low"),
        "oracle_minus_misaligned_high_n": scorer.get("oracle_minus_misaligned_high_n"),
        "scorer_score_gain_ci": scorer_misaligned_score_ci,
        "scorer_real_change_ci": scorer_misaligned_real_ci,
        "oracle_minus_misaligned_high_n_ci": scorer_oracle_gap_ci,
        "thresholds": {
            "controlled_score_gain_min": 0.35,
            "controlled_real_change_max": -0.25,
            "controlled_regret_min": 0.80,
            "scorer_score_gain_min": 0.35,
            "scorer_real_change_max": -0.25,
            "oracle_gap_min": 0.90,
            "score_ci_low_min": 0.20,
            "real_change_ci_high_max": -0.12,
            "oracle_gap_ci_low_min": 0.65,
        },
    }
    add(
        claims,
        "controlled_sampler",
        "High N can hurt or saturate under scorer misalignment.",
        status(misaligned_strong, misaligned_weak),
        strong_metrics["misaligned_tail_selection"],
    )

    low_div_rows = [row for row in controlled_div if row.get("regime") == "low_diversity_high_quality"]
    low_div_eff = [f(row, "effective_sample_diversity") for row in low_div_rows]
    low_div_collapse = [f(row, "duplicate_collapse_rate") for row in low_div_rows]
    low_div_weak = bool(controlled.get("low_diversity_has_small_marginal_value"))
    low_div_gain_ci = first_row(
        controlled_effect_cis,
        regime="low_diversity_high_quality",
        metric="exact_selected_real",
    )
    high_div_gain_ci = first_row(
        controlled_effect_cis,
        regime="high_diversity_aligned",
        metric="exact_selected_real",
    )
    low_div_strong = (
        abs(controlled.get("low_diversity_real_gain_high_minus_low", 1.0)) <= 0.02
        and controlled.get("high_diversity_real_gain_high_minus_low", 0.0) >= 0.50
        and bool(low_div_eff)
        and max(low_div_eff) <= 1.05
        and bool(low_div_collapse)
        and min(low_div_collapse) >= 0.98
        and ci_ok(low_div_gain_ci, min_n=3)
        and f(low_div_gain_ci, "ci_low") >= -0.02
        and f(low_div_gain_ci, "ci_high") <= 0.02
        and ci_ok(high_div_gain_ci, mean_min=0.50, low_min=0.35, min_n=3)
    )
    strong_metrics["low_diversity"] = {
        "low_diversity_gain": controlled.get("low_diversity_real_gain_high_minus_low"),
        "high_diversity_gain": controlled.get("high_diversity_real_gain_high_minus_low"),
        "low_diversity_gain_ci": low_div_gain_ci,
        "high_diversity_gain_ci": high_div_gain_ci,
        "max_effective_sample_diversity": max(low_div_eff) if low_div_eff else None,
        "min_duplicate_collapse_rate": min(low_div_collapse) if low_div_collapse else None,
        "thresholds": {
            "abs_low_gain_max": 0.02,
            "low_gain_ci_bounds": [-0.02, 0.02],
            "high_diversity_gain_min": 0.50,
            "high_diversity_gain_ci_low_min": 0.35,
            "effective_diversity_max": 1.05,
            "duplicate_collapse_rate_min": 0.98,
        },
    }
    add(
        claims,
        "diversity",
        "Low diversity reduces marginal value of N.",
        status(low_div_strong, low_div_weak),
        strong_metrics["low_diversity"],
    )

    nk_phase_path = RESULTS / "tables" / "nk_budget_phase.csv"
    nk_rows = csv_rows(nk_phase_path)
    unique_n = {row.get("N") for row in nk_rows}
    unique_k = {row.get("K") for row in nk_rows}
    nk_weak = bool(nk) and {"N", "K", "B", "real_utility", "latency_adjusted_utility"}.issubset(csv_columns(nk_phase_path))
    nk_strong = nk_weak and len(unique_n) >= 6 and len(unique_k) >= 5 and csv_row_count(nk_phase_path) >= 30
    strong_metrics["nk_tradeoff"] = {
        "rows": csv_row_count(nk_phase_path),
        "unique_N": len(unique_n),
        "unique_K": len(unique_k),
        "thresholds": {"rows_min": 30, "unique_N_min": 6, "unique_K_min": 5},
    }
    add(
        claims,
        "denoising_budget",
        "N versus K tradeoff is measured.",
        status(nk_strong, nk_weak),
        strong_metrics["nk_tradeoff"],
    )

    latency_weak = bool(nk.get("latency_prefers_smaller_budget_than_max")) and bool(
        nk.get("latency_best_differs_from_real_best")
    )
    best_latency = nk.get("best_latency_adjusted") or {}
    high_budget = nk.get("high_budget_corner") or {}
    latency_margin = f(high_budget, "latency_adjusted_utility") - f(best_latency, "latency_adjusted_utility")
    budget_ratio = f(best_latency, "B") / max(f(high_budget, "B"), 1e-12)
    latency_effect_ci = first_row(
        nk_latency_effect,
        effect="best_latency_adjusted_minus_high_budget_corner",
    )
    latency_strong = (
        latency_weak
        and budget_ratio <= 0.10
        and latency_margin <= -2.0
        and ci_ok(latency_effect_ci, mean_min=2.0, low_min=1.50, min_n=3)
    )
    strong_metrics["latency_adjustment"] = {
        "best_latency_adjusted": best_latency,
        "high_budget_corner": high_budget,
        "best_budget_to_high_budget_ratio": budget_ratio,
        "high_minus_best_latency_adjusted_utility": latency_margin,
        "best_minus_high_latency_adjusted_ci": latency_effect_ci,
        "thresholds": {
            "best_budget_ratio_max": 0.10,
            "high_minus_best_latency_adjusted_utility_max": -2.0,
            "best_minus_high_ci_low_min": 1.50,
        },
    }
    add(
        claims,
        "latency",
        "Latency-adjusted utility can prefer smaller N or K.",
        status(latency_strong, latency_weak),
        strong_metrics["latency_adjustment"],
    )

    repair_weak = scorer.get("calibrated_minus_misaligned_high_n", 0.0) > 0.05
    repair_fraction = scorer.get("calibrated_minus_misaligned_high_n", 0.0) / max(
        scorer.get("oracle_minus_misaligned_high_n", 0.0), 1e-12
    )
    repair_ci = first_row(
        scorer_effect_cis,
        regime="hidden_obstacle_high_diversity",
        effect="calibrated_critic_minus_misaligned_tail_scorer",
    )
    oracle_gap_ci = first_row(
        scorer_effect_cis,
        regime="hidden_obstacle_high_diversity",
        effect="oracle_real_utility_selector_minus_misaligned_tail_scorer",
    )
    calibration_successes = [row for row in calibration_map if row.get("repair_status") == "strong_repair"]
    calibration_failures = [row for row in calibration_map if row.get("repair_status") == "no_strong_repair"]
    repair_strong = (
        scorer.get("calibrated_minus_misaligned_high_n", 0.0) >= 0.80
        and scorer.get("oracle_minus_misaligned_high_n", 0.0) >= 0.90
        and repair_fraction >= 0.75
        and ci_ok(repair_ci, mean_min=0.80, low_min=0.60, min_n=3)
        and ci_ok(oracle_gap_ci, mean_min=0.90, low_min=0.65, min_n=3)
        and len(calibration_successes) >= 1
        and len(calibration_failures) >= 1
    )
    strong_metrics["calibration_repair"] = {
        "calibrated_minus_misaligned_high_n": scorer.get("calibrated_minus_misaligned_high_n"),
        "oracle_minus_misaligned_high_n": scorer.get("oracle_minus_misaligned_high_n"),
        "repair_fraction_of_oracle_gap": repair_fraction,
        "calibrated_minus_misaligned_ci": repair_ci,
        "oracle_minus_misaligned_ci": oracle_gap_ci,
        "calibration_map": calibration_map,
        "thresholds": {
            "calibrated_gain_min": 0.80,
            "oracle_gap_min": 0.90,
            "repair_fraction_min": 0.75,
            "calibrated_gain_ci_low_min": 0.60,
            "oracle_gap_ci_low_min": 0.65,
            "success_rows_min": 1,
            "non_repair_rows_min": 1,
        },
    }
    add(
        claims,
        "calibration",
        "Calibrated scorer repairs high-N selection in at least one regime.",
        status(repair_strong, repair_weak),
        strong_metrics["calibration_repair"],
    )

    controller_actions = set(controller.get("action_vocabulary") or [])
    required_controller_actions = {
        "increase_N",
        "stop_early",
        "reduce_K",
        "calibrate_scorer",
        "audit_rollouts",
        "increase_diversity",
        "block_high_N",
    }
    controller_decision_columns = csv_columns(RESULTS / "tables" / "audit_then_sample_decisions.csv")
    controller_calibration_columns = csv_columns(RESULTS / "tables" / "audit_then_sample_calibration.csv")
    controller_negative_controls = set(controller.get("negative_controls") or [])
    required_controller_negative_controls = {
        "anti_correlated_scorer",
        "adversarial_tail_scorer",
        "tail_misaligned_scorer",
        "hidden_ood_dynamics",
        "duplicated_high_score_artifacts",
        "correlated_candidate_pool",
        "calibration_drift",
        "collapsed_sampler",
        "latency_spike",
        "missing_utility",
        "random_score_failed_repair",
    }
    controller_required_bound_columns = {
        "risk_delta",
        "effective_n_for_bounds",
        "utility_gain_lcb",
        "tail_utility_lcb",
        "latency_adjusted_gain_lcb",
        "admit_high_N",
        "abstention_reason",
        "false_admit_negative_control",
    }
    calibration_required_bound_columns = {
        "repair_regime",
        "success",
        "repair_method",
        "original_tail_rank_correlation",
        "repaired_tail_rank_correlation",
        "original_utility_gain_lcb",
        "repaired_utility_gain_lcb",
        "original_tail_utility_lcb",
        "repaired_tail_utility_lcb",
        "original_latency_adjusted_gain_lcb",
        "repaired_latency_adjusted_gain_lcb",
    }

    def truthy(row: dict[str, Any], key: str) -> bool:
        value = row.get(key)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}

    controller_admitted_rows = [row for row in controller_decisions if truthy(row, "admit_high_N")]
    controller_false_admit_rows = [row for row in controller_decisions if truthy(row, "false_admit_negative_control")]
    controller_underpowered_rows = [
        row
        for row in controller_decisions
        if row.get("regime") in {"correlated_candidate_pool", "small_audit_underpowered"}
    ]
    admitted_bounds_positive = all(
        f(row, "utility_gain_lcb") > 0.0
        and f(row, "tail_utility_lcb") > 0.0
        and f(row, "latency_adjusted_gain_lcb") > 0.0
        for row in controller_admitted_rows
    )
    underpowered_abstain_rows = bool(controller_underpowered_rows) and all(
        not truthy(row, "admit_high_N")
        and row.get("action_recommendation") in {"audit_rollouts", "increase_diversity", "block_high_N"}
        for row in controller_underpowered_rows
    )
    repair_success_rows = [row for row in controller_calibration if truthy(row, "success")]
    repair_negative_rows = [row for row in controller_calibration if truthy(row, "negative_control")]
    repair_success_bound_rows = bool(repair_success_rows) and all(
        f(row, "repaired_utility_gain_lcb") > 0.0
        and f(row, "repaired_tail_utility_lcb") > 0.0
        and f(row, "repaired_latency_adjusted_gain_lcb") > 0.0
        for row in repair_success_rows
    )
    repair_negative_controls_fail = bool(repair_negative_rows) and all(
        not truthy(row, "success") for row in repair_negative_rows
    )
    controller_weak = (
        bool(controller)
        and required_controller_actions.issubset(controller_actions)
        and {"selected_N", "selected_K", "decision_label", "action_recommendation"}.issubset(controller_decision_columns)
        and controller_required_bound_columns.issubset(controller_decision_columns)
        and calibration_required_bound_columns.issubset(controller_calibration_columns)
        and len(controller_decisions) > 0
        and len(controller_calibration) > 0
    )
    controller_strong = (
        controller_weak
        and f(controller, "aligned_allow_high_n_fraction") >= 0.75
        and f(controller, "anti_correlated_block_fraction") >= 0.75
        and f(controller, "tail_misaligned_block_fraction") >= 0.75
        and f(controller, "shuffled_repair_or_block_fraction") >= 0.75
        and f(controller, "collapsed_stop_early_fraction") >= 0.75
        and f(controller, "latency_limited_small_budget_fraction") >= 0.75
        and f(controller, "false_admit_rate") == 0.0
        and int(f(controller, "false_admit_count", 1.0)) == 0
        and admitted_bounds_positive
        and underpowered_abstain_rows
        and int(f(controller, "calibration_success_rows", 0.0)) >= 1
        and int(f(controller, "calibration_failure_rows", 0.0)) >= 1
        and repair_success_bound_rows
        and repair_negative_controls_fail
        and f(controller, "repair_bound_validation_fraction") == 1.0
        and f(controller, "repair_failure_control_fraction") == 1.0
        and f(controller, "adaptive_stop_fraction") >= 0.75
        and f(controller, "adaptive_stopping_savings_mean") > 0.0
        and bool(controller.get("controller_conservative_certification"))
        and bool(controller.get("confidence_gates_present"))
        and required_controller_negative_controls.issubset(controller_negative_controls)
        and (RESULTS / "figures" / "audit_then_sample_decision_regions.png").exists()
    )
    strong_metrics["audit_then_sample_controller"] = {
        "aligned_allow_high_n_fraction": controller.get("aligned_allow_high_n_fraction"),
        "anti_correlated_block_fraction": controller.get("anti_correlated_block_fraction"),
        "tail_misaligned_block_fraction": controller.get("tail_misaligned_block_fraction"),
        "shuffled_repair_or_block_fraction": controller.get("shuffled_repair_or_block_fraction"),
        "collapsed_stop_early_fraction": controller.get("collapsed_stop_early_fraction"),
        "latency_limited_small_budget_fraction": controller.get("latency_limited_small_budget_fraction"),
        "false_admit_rate": controller.get("false_admit_rate"),
        "false_admit_count": controller.get("false_admit_count"),
        "admitted_rows": len(controller_admitted_rows),
        "admitted_bounds_positive": admitted_bounds_positive,
        "underpowered_abstain_rows": underpowered_abstain_rows,
        "lower_bound_coverage": controller.get("lower_bound_coverage"),
        "calibration_success_rows": controller.get("calibration_success_rows"),
        "calibration_failure_rows": controller.get("calibration_failure_rows"),
        "repair_bound_validation_fraction": controller.get("repair_bound_validation_fraction"),
        "repair_failure_control_fraction": controller.get("repair_failure_control_fraction"),
        "repair_success_bound_rows": repair_success_bound_rows,
        "repair_negative_controls_fail": repair_negative_controls_fail,
        "repair_success_methods": controller.get("repair_success_methods"),
        "effective_n_ratio_mean": controller.get("effective_n_ratio_mean"),
        "effective_n_ratio_min": controller.get("effective_n_ratio_min"),
        "adaptive_stop_fraction": controller.get("adaptive_stop_fraction"),
        "adaptive_stopping_savings_mean": controller.get("adaptive_stopping_savings_mean"),
        "controller_conservative_certification": controller.get("controller_conservative_certification"),
        "confidence_gates_present": controller.get("confidence_gates_present"),
        "action_vocabulary": sorted(controller_actions),
        "negative_controls": sorted(controller_negative_controls),
        "false_admit_rows": controller_false_admit_rows[:5],
        "decision_rows": len(controller_decisions),
        "calibration_rows": len(controller_calibration),
        "decision_columns": sorted(controller_decision_columns),
        "calibration_columns": sorted(controller_calibration_columns),
        "thresholds": {
            "decision_fraction_min": 0.75,
            "false_admit_rate_max": 0.0,
            "admitted_lcbs_must_be_positive": True,
            "underpowered_cases_must_abstain": True,
            "calibration_success_rows_min": 1,
            "calibration_failure_rows_min": 1,
            "repair_success_requires_positive_heldout_lcbs": True,
            "adaptive_stop_fraction_min": 0.75,
            "requires_confidence_gates": True,
            "requires_negative_controls": sorted(required_controller_negative_controls),
        },
    }
    add(
        claims,
        "controller_fix",
        "Audit-Then-Sample chooses when to increase N, stop early, repair, increase diversity, reduce K, or block high-N selection.",
        status(controller_strong, controller_weak),
        strong_metrics["audit_then_sample_controller"],
    )

    not_wam_ok = (
        "what theorem is reused" in diff_doc
        and "diffusion tail over-selection" in diff_doc
        and "denoising" in diff_doc
        and "sample diversity" in diff_doc
        and "what wam claims are forbidden" in diff_doc
    )
    add(
        claims,
        "differentiation",
        "Project is not a WAM clone.",
        status(not_wam_ok),
        "docs/differentiation_from_wam_jepa.md centers diffusion action trajectories rather than imagined rollouts.",
    )

    not_jepa_ok = (
        "what jepa claims are forbidden" in diff_doc
        and "action sequence" in checklist_doc
        and "learned diffusion policy-lite" in checklist_doc
        and "latent-real rank distortion" in diff_doc
    )
    add(
        claims,
        "differentiation",
        "Project is not a JEPA clone.",
        status(not_jepa_ok),
        "docs/differentiation_from_wam_jepa.md forbids latent-prediction framing as the central contribution.",
    )

    required_artifacts = [
        RESULTS / "controlled_sampler_summary.json",
        RESULTS / "audit_then_sample_summary.json",
        RESULTS / "scorer_comparison_summary.json",
        RESULTS / "nk_budget_summary.json",
        RESULTS / "learned_policy_lite_summary.json",
        RESULTS / "tables" / "controlled_sampler_aggregate.csv",
        RESULTS / "tables" / "controlled_sampler_seed_aggregate.csv",
        RESULTS / "tables" / "controlled_sampler_effect_cis.csv",
        RESULTS / "tables" / "audit_then_sample_decisions.csv",
        RESULTS / "tables" / "audit_then_sample_calibration.csv",
        RESULTS / "tables" / "scorer_comparison_aggregate.csv",
        RESULTS / "tables" / "scorer_comparison_seed_aggregate.csv",
        RESULTS / "tables" / "scorer_comparison_effect_cis.csv",
        RESULTS / "tables" / "calibration_repair_map.csv",
        RESULTS / "tables" / "nk_budget_phase.csv",
        RESULTS / "tables" / "nk_budget_seed_aggregate.csv",
        RESULTS / "tables" / "nk_budget_latency_effect_ci.csv",
        RESULTS / "tables" / "learned_policy_lite_aggregate.csv",
        RESULTS / "tables" / "learned_policy_lite_seed_aggregate.csv",
        RESULTS / "tables" / "learned_policy_lite_effect_cis.csv",
        RESULTS / "true_diffusion_summary.json",
        RESULTS / "tables" / "true_diffusion_aggregate.csv",
        RESULTS / "tables" / "true_diffusion_seed_aggregate.csv",
        RESULTS / "tables" / "true_diffusion_effect_cis.csv",
        RESULTS / "tables" / "true_diffusion_scorer_gap_cis.csv",
        RESULTS / "tables" / "true_diffusion_runtime.csv",
        RESULTS / "tables" / "true_diffusion_training.csv",
        RESULTS / "tables" / "true_diffusion_sampler_comparison.csv",
        RESULTS / "pusht_summary.json",
        RESULTS / "tables" / "pusht_aggregate.csv",
        RESULTS / "tables" / "pusht_seed_aggregate.csv",
        RESULTS / "tables" / "pusht_effect_cis.csv",
        RESULTS / "tables" / "pusht_scorer_gap_cis.csv",
        RESULTS / "tables" / "pusht_runtime.csv",
        RESULTS / "tables" / "pusht_rollouts.csv",
        RESULTS / "tables" / "pusht_rollout_metric_seed_aggregate.csv",
        RESULTS / "tables" / "pusht_rollout_metric_aggregate.csv",
        RESULTS / "tables" / "pusht_rollout_metric_effect_cis.csv",
        RESULTS / "figures" / "nk_budget_phase_diagram.png",
        RESULTS / "figures" / "audit_then_sample_decision_regions.png",
        RESULTS / "figures" / "true_diffusion_survival.png",
        RESULTS / "figures" / "true_diffusion_runtime.png",
        RESULTS / "figures" / "true_diffusion_sampler_comparison.png",
        RESULTS / "figures" / "pusht_max_selection.png",
    ]
    table_min_rows = {
        "controlled_sampler_aggregate.csv": 42,
        "audit_then_sample_decisions.csv": 20 if is_smoke_results() else 80,
        "audit_then_sample_calibration.csv": 4 if is_smoke_results() else 20,
        "scorer_comparison_aggregate.csv": 49,
        "nk_budget_phase.csv": 30,
        "learned_policy_lite_aggregate.csv": 200,
        "learned_policy_lite_effect_cis.csv": 80,
        "true_diffusion_aggregate.csv": 80 if is_smoke_results() else 200,
        "true_diffusion_effect_cis.csv": 60 if is_smoke_results() else 150,
        "true_diffusion_sampler_comparison.csv": 4 if is_smoke_results() else 6,
        "pusht_aggregate.csv": 40 if is_smoke_results() else 90,
        "pusht_effect_cis.csv": 30 if is_smoke_results() else 70,
        "pusht_rollout_metric_effect_cis.csv": 15 if is_smoke_results() else 60,
        "pusht_rollout_metric_seed_aggregate.csv": 20 if is_smoke_results() else 120,
    }
    table_rows_ok = all(
        csv_row_count(RESULTS / "tables" / name) >= minimum
        for name, minimum in table_min_rows.items()
    )
    figure_sizes_ok = all(
        (RESULTS / "figures" / name).exists() and (RESULTS / "figures" / name).stat().st_size >= 10_000
        for name in [
            "controlled_sampler_curves.png",
            "audit_then_sample_decision_regions.png",
            "scorer_comparison.png",
            "nk_budget_phase_diagram.png",
            "learned_policy_lite_ood.png",
            "toy_image_observations.png",
            "true_diffusion_survival.png",
            "true_diffusion_runtime.png",
            "true_diffusion_sampler_comparison.png",
            "pusht_max_selection.png",
        ]
    )
    artifacts_weak = all(path.exists() and path.stat().st_size > 0 for path in required_artifacts)
    artifacts_strong = artifacts_weak and table_rows_ok and figure_sizes_ok
    strong_metrics["artifact_backing"] = {
        "required_artifacts_exist": artifacts_weak,
        "table_min_rows": table_min_rows,
        "actual_table_rows": {
            name: csv_row_count(RESULTS / "tables" / name) for name in table_min_rows
        },
        "figure_sizes_ok": figure_sizes_ok,
    }
    add(
        claims,
        "artifact_backing",
        "All major claims are backed by CSV/JSON artifacts.",
        status(artifacts_strong, artifacts_weak),
        strong_metrics["artifact_backing"],
    )

    learned_checklist = learned.get("diffusion_policy_validity_checklist") or {}
    min_learned_seeds = 1 if is_smoke_results() else 3
    learned_conditionings = {row.get("conditioning") for row in learned_agg}
    learned_training_seeds = {row.get("seed") for row in learned_training}
    learned_weak = all(
        learned_checklist.get(key) is True
        for key in [
            "stochastic_trajectory_generation",
            "iterative_denoising_or_noise_to_action_generation",
            "conditioning_on_observation_or_state",
            "action_sequence_generation",
            "receding_horizon_or_trajectory_execution_evaluation",
        ]
    ) and len(learned.get("ood_regimes") or []) >= 4 and {"state", "image"}.issubset(learned_conditionings)
    k4_calibrated_gains: dict[str, list[float]] = {"state": [], "image": []}
    k4_calibrated_ci_lows: dict[str, list[float]] = {"state": [], "image": []}
    k4_calibrated_corr: dict[str, list[float]] = {"state": [], "image": []}
    k4_calibrated_tail: dict[str, list[float]] = {"state": [], "image": []}
    k4_effect_rows = []
    for conditioning in ["state", "image"]:
        for regime in sorted({row.get("regime") for row in learned_agg}):
            ci_row = first_row(
                learned_effect_cis,
                conditioning=conditioning,
                regime=regime,
                scorer="calibrated_critic",
                K=4,
                metric="exact_selected_real",
            )
            if ci_row:
                k4_effect_rows.append(ci_row)
                k4_calibrated_ci_lows[conditioning].append(f(ci_row, "ci_low"))
                k4_calibrated_gains[conditioning].append(f(ci_row, "mean"))
            rows = [
                row
                for row in learned_agg
                if row.get("conditioning") == conditioning
                and row.get("regime") == regime
                and row.get("scorer") == "calibrated_critic"
                and row.get("K") == "4"
            ]
            if rows:
                hi = max(rows, key=lambda row: f(row, "N"))
                k4_calibrated_corr[conditioning].append(f(hi, "score_utility_correlation"))
                k4_calibrated_tail[conditioning].append(f(hi, "tail_rank_correlation"))
    loss_ratios = [f(row, "loss_ratio") for row in learned_training]
    state_loss_ratios = [f(row, "loss_ratio") for row in learned_training if row.get("conditioning") == "state"]
    image_loss_ratios = [f(row, "loss_ratio") for row in learned_training if row.get("conditioning") == "image"]
    learned_strong = (
        learned_weak
        and bool(loss_ratios)
        and bool(state_loss_ratios)
        and bool(image_loss_ratios)
        and max(state_loss_ratios) <= 0.30
        and max(image_loss_ratios) <= 0.65
        and bool(learned.get("loss_decreased_all_seed_conditioning_pairs"))
        and len(learned_agg) >= 200
        and len(learned_seed_agg) >= 200
        and len(learned_receding) >= 20
        and len(learned_training_seeds) >= min_learned_seeds
        and int(learned.get("image_size") or 0) == 32
        and len(k4_effect_rows) >= 10
        and all(ci_ok(row, mean_min=0.08, low_min=0.04, min_n=3) for row in k4_effect_rows)
        and min(k4_calibrated_corr["state"] or [0.0]) >= 0.95
        and min(k4_calibrated_tail["state"] or [0.0]) >= 0.85
        and min(k4_calibrated_corr["image"] or [0.0]) >= 0.90
        and min(k4_calibrated_tail["image"] or [0.0]) >= 0.50
    )
    strong_metrics["learned_policy_lite"] = {
        "checklist": learned_checklist,
        "conditioning_modes": sorted(learned_conditionings),
        "image_size": learned.get("image_size"),
        "ood_regimes": learned.get("ood_regimes"),
        "num_training_seeds": len(learned_training_seeds),
        "min_training_seeds_required": min_learned_seeds,
        "max_state_loss_ratio": max(state_loss_ratios) if state_loss_ratios else None,
        "max_image_loss_ratio": max(image_loss_ratios) if image_loss_ratios else None,
        "aggregate_rows": len(learned_agg),
        "seed_aggregate_rows": len(learned_seed_agg),
        "receding_rows": len(learned_receding),
        "k4_calibrated_real_gains": k4_calibrated_gains,
        "k4_calibrated_real_gain_ci_lows": k4_calibrated_ci_lows,
        "k4_calibrated_correlations": k4_calibrated_corr,
        "k4_calibrated_tail_correlations": k4_calibrated_tail,
        "k4_effect_ci_rows": k4_effect_rows,
        "thresholds": {
            "state_loss_ratio_max": 0.30,
            "image_loss_ratio_max": 0.65,
            "aggregate_rows_min": 200,
            "seed_aggregate_rows_min": 200,
            "receding_rows_min": 20,
            "image_size": 32,
            "effect_ci_mean_min": 0.08,
            "effect_ci_low_min": 0.04,
            "state_correlation_min": 0.95,
            "state_tail_correlation_min": 0.85,
            "image_correlation_min": 0.90,
            "image_tail_correlation_min": 0.50,
        },
    }
    add(
        claims,
        "learned_diffusion_policy_lite",
        "Learned experiment satisfies the Diffusion Policy-style validity checklist and includes multi-seed state/image ID/OOD evaluation.",
        status(learned_strong, learned_weak),
        strong_metrics["learned_policy_lite"],
    )

    min_true_units = 1 if is_smoke_results() else 12
    min_true_training_seeds = 1 if is_smoke_results() else 4
    true_checklist = true_diffusion.get("diffusion_policy_validity_checklist") or {}
    true_samplers = set(true_diffusion.get("sampler_families") or [])
    true_primary_samplers = set(true_diffusion.get("primary_samplers") or [])
    true_ablation_samplers = set(true_diffusion.get("ablation_samplers") or [])
    true_scorers = set(true_diffusion.get("scorers") or [])
    true_training_seeds = {row.get("seed") for row in true_training if row.get("model") == "epsilon_ddpm"}
    true_k_values = [int(v) for v in true_diffusion.get("k_values") or []]
    true_key_k = max(true_k_values) if true_k_values else 0
    true_ddim_oracle_ci = first_row(
        true_effect_cis,
        sampler="ddim_eps",
        regime="id",
        scorer="oracle_real_utility_selector",
        K=true_key_k,
        metric="exact_selected_real",
    )
    true_ddpm_oracle_ci = first_row(
        true_effect_cis,
        sampler="ddpm_eps",
        regime="id",
        scorer="oracle_real_utility_selector",
        K=true_key_k,
        metric="exact_selected_real",
    )
    true_anti_ci = first_row(
        true_effect_cis,
        sampler="ddim_eps",
        regime="id",
        scorer="anti_correlated_score",
        K=true_key_k,
        metric="exact_selected_real",
    )
    true_tail_ci = first_row(
        true_effect_cis,
        sampler="ddim_eps",
        regime="hidden_obstacle",
        scorer="tail_only_misaligned_score",
        K=true_key_k,
        metric="exact_selected_real",
    )
    true_gap_ci = first_row(
        true_gap_cis,
        sampler="ddim_eps",
        regime="hidden_obstacle",
        effect="oracle_real_utility_selector_minus_tail_only_misaligned_score",
        K=true_key_k,
    )
    true_weak = (
        all(true_checklist.get(key) is True for key in ["true_epsilon_prediction", "ddim_fast_sampling", "stochastic_ddpm_sampling"])
        and {"ddim_eps", "ddpm_eps", "consistency_1step", "clean_target_ablation"}.issubset(true_samplers)
        and {"ddim_eps", "ddpm_eps", "consistency_1step"}.issubset(true_primary_samplers)
        and {"clean_target_ablation"}.issubset(true_ablation_samplers)
        and {"diffusion_internal_score", "oracle_real_utility_selector", "anti_correlated_score", "tail_only_misaligned_score"}.issubset(true_scorers)
        and len(true_training_seeds) >= min_true_training_seeds
        and len(true_runtime) > 0
        and len(true_sampler_comparison) >= (4 if is_smoke_results() else 6)
    )
    true_strong = (
        true_weak
        and f(true_diffusion, "ddim_oracle_gain_high_minus_low") >= 0.01
        and f(true_diffusion, "ddpm_oracle_gain_high_minus_low") >= 0.0
        and f(true_diffusion, "anti_correlated_real_change_high_minus_low") <= -0.01
        and len(true_agg) >= (80 if is_smoke_results() else 200)
        and csv_row_count(RESULTS / "tables" / "true_diffusion_curves.csv") >= (300 if is_smoke_results() else 1200)
        and ci_ok(true_ddim_oracle_ci, mean_min=0.01, min_n=min_true_units)
        and ci_ok(true_ddpm_oracle_ci, mean_min=0.0, min_n=min_true_units)
        and ci_ok(true_anti_ci, mean_max=-0.01, min_n=min_true_units)
        and bool(true_tail_ci)
        and bool(true_gap_ci)
        and bool(true_diffusion.get("measured_wall_clock_runtime"))
        and (RESULTS / "figures" / "true_diffusion_sampler_comparison.png").exists()
    )
    strong_metrics["true_action_diffusion"] = {
        "checklist": true_checklist,
        "primary_samplers": sorted(true_primary_samplers),
        "ablation_samplers": sorted(true_ablation_samplers),
        "sampler_families": sorted(true_samplers),
        "scorers": sorted(true_scorers),
        "num_training_rows": len(true_training),
        "num_training_seeds": len(true_training_seeds),
        "num_runtime_rows": len(true_runtime),
        "sampler_comparison_rows": len(true_sampler_comparison),
        "aggregate_rows": len(true_agg),
        "curve_rows": csv_row_count(RESULTS / "tables" / "true_diffusion_curves.csv"),
        "ddim_oracle_gain": true_diffusion.get("ddim_oracle_gain_high_minus_low"),
        "ddpm_oracle_gain": true_diffusion.get("ddpm_oracle_gain_high_minus_low"),
        "anti_correlated_change": true_diffusion.get("anti_correlated_real_change_high_minus_low"),
        "hidden_tail_change": true_diffusion.get("hidden_tail_misaligned_real_change_high_minus_low"),
        "ddim_oracle_ci": true_ddim_oracle_ci,
        "ddpm_oracle_ci": true_ddpm_oracle_ci,
        "anti_correlated_ci": true_anti_ci,
        "tail_misaligned_ci": true_tail_ci,
        "oracle_tail_gap_ci": true_gap_ci,
        "thresholds": {
            "ddim_oracle_gain_min": 0.01,
            "ddpm_oracle_gain_min": 0.0,
            "anti_correlated_change_max": -0.01,
            "min_ci_units": min_true_units,
            "min_training_seeds": min_true_training_seeds,
            "sampler_comparison_rows_min": 4 if is_smoke_results() else 6,
        },
    }
    add(
        claims,
        "true_action_diffusion",
        "trajectory-search effects survive a true epsilon-prediction DDPM/DDIM action diffusion policy, with one-step and clean-target ablations.",
        status(true_strong, true_weak),
        strong_metrics["true_action_diffusion"],
    )

    min_pusht_units = 1 if is_smoke_results() else 12
    min_pusht_training_seeds = 1 if is_smoke_results() else 4
    pusht_checklist = pusht.get("diffusion_policy_validity_checklist") or {}
    pusht_samplers = set(pusht.get("sampler_families") or [])
    pusht_scorers = set(pusht.get("scorers") or [])
    pusht_training_seeds = {row.get("seed") for row in pusht_training}
    pusht_k_values = [int(v) for v in pusht.get("k_values") or []]
    pusht_key_k = max(pusht_k_values) if pusht_k_values else 0
    pusht_aligned_ci = first_row(
        pusht_effect_cis,
        sampler="ddim_eps",
        regime="pusht_aligned",
        scorer="oracle_real_utility_selector",
        K=pusht_key_k,
        metric="exact_selected_real",
    )
    pusht_low_div_ci = first_row(
        pusht_effect_cis,
        sampler="ddim_eps",
        regime="pusht_low_diversity",
        scorer="oracle_real_utility_selector",
        K=pusht_key_k,
        metric="exact_selected_real",
    )
    pusht_misaligned_ci = first_row(
        pusht_effect_cis,
        sampler="ddim_eps",
        regime="pusht_high_temp_misaligned",
        scorer="misaligned_corner_scorer",
        K=pusht_key_k,
        metric="exact_selected_real",
    )
    pusht_gap_ci = first_row(
        pusht_gap_cis,
        sampler="ddim_eps",
        regime="pusht_high_temp_misaligned",
        effect="oracle_real_utility_selector_minus_misaligned_corner_scorer",
        K=pusht_key_k,
    )
    pusht_rollout_max_cov_ci = first_row(
        pusht_rollout_metric_effect_cis,
        sampler="ddim_eps",
        regime="pusht_aligned",
        scorer="oracle_real_utility_selector",
        K=pusht_key_k,
        metric="exact_selected_max_coverage",
    )
    pusht_rollout_final_cov_ci = first_row(
        pusht_rollout_metric_effect_cis,
        sampler="ddim_eps",
        regime="pusht_aligned",
        scorer="oracle_real_utility_selector",
        K=pusht_key_k,
        metric="exact_selected_final_coverage",
    )
    pusht_rollout_success_ci = first_row(
        pusht_rollout_metric_effect_cis,
        sampler="ddim_eps",
        regime="pusht_aligned",
        scorer="oracle_real_utility_selector",
        K=pusht_key_k,
        metric="exact_selected_success",
    )
    pusht_rollout_metric_columns = set(pusht.get("rollout_metric_columns") or [])
    pusht_rollout_metrics_weak = (
        bool(pusht.get("actual_rollout_metric_curves"))
        and {"exact_selected_max_coverage", "exact_selected_final_coverage", "exact_selected_success"}.issubset(pusht_rollout_metric_columns)
        and {"exact_selected_max_coverage", "exact_selected_final_coverage", "exact_selected_success"}.issubset(
            csv_columns(RESULTS / "tables" / "pusht_curves.csv")
        )
        and len(pusht_rollout_metric_effect_cis) > 0
        and len(pusht_rollout_metric_seed_agg) > 0
        and len(pusht_rollout_metric_agg) > 0
    )
    pusht_rollout_metrics_strong = (
        pusht_rollout_metrics_weak
        and ci_ok(pusht_rollout_max_cov_ci, min_n=min_pusht_units)
        and ci_ok(pusht_rollout_final_cov_ci, min_n=min_pusht_units)
        and ci_ok(pusht_rollout_success_ci, min_n=min_pusht_units)
    )
    pusht_weak = (
        pusht.get("benchmark") == "PushT"
        and pusht.get("env_id") == "gym_pusht/PushT-v0"
        and bool(pusht.get("actual_simulator_rollouts"))
        and all(pusht_checklist.get(key) is True for key in ["true_epsilon_prediction", "actual_environment_rollout_utility", "trajectory_reranking_over_sampled_actions"])
        and {"ddim_eps", "ddpm_eps", "consistency_1step"}.issubset(pusht_samplers)
        and {"oracle_real_utility_selector", "learned_value_critic_from_pilot_rollouts", "misaligned_corner_scorer"}.issubset(pusht_scorers)
        and len(pusht_training_seeds) >= min_pusht_training_seeds
        and len(pusht_runtime) > 0
        and len(pusht_rollouts) > 0
        and pusht_rollout_metrics_weak
    )
    pusht_strong = (
        pusht_weak
        and f(pusht, "pusht_aligned_oracle_gain_high_minus_low") >= 0.002
        and f(pusht, "pusht_oracle_minus_misaligned_high_n") >= 0.0
        and len(pusht_agg) >= (40 if is_smoke_results() else 90)
        and csv_row_count(RESULTS / "tables" / "pusht_curves.csv") >= (120 if is_smoke_results() else 300)
        and int(f(pusht, "rollout_rows", 0.0)) >= (16 if is_smoke_results() else 48)
        and ci_ok(pusht_aligned_ci, mean_min=0.0, min_n=min_pusht_units)
        and bool(pusht_low_div_ci)
        and bool(pusht_misaligned_ci)
        and bool(pusht_gap_ci)
        and pusht_rollout_metrics_strong
        and bool(pusht.get("measured_wall_clock_runtime"))
    )
    strong_metrics["pusht_benchmark"] = {
        "checklist": pusht_checklist,
        "sampler_families": sorted(pusht_samplers),
        "scorers": sorted(pusht_scorers),
        "num_training_rows": len(pusht_training),
        "num_training_seeds": len(pusht_training_seeds),
        "num_runtime_rows": len(pusht_runtime),
        "num_rollout_rows": len(pusht_rollouts),
        "rollout_metric_effect_rows": len(pusht_rollout_metric_effect_cis),
        "rollout_metric_seed_rows": len(pusht_rollout_metric_seed_agg),
        "aggregate_rows": len(pusht_agg),
        "curve_rows": csv_row_count(RESULTS / "tables" / "pusht_curves.csv"),
        "aligned_oracle_gain": pusht.get("pusht_aligned_oracle_gain_high_minus_low"),
        "low_diversity_oracle_gain": pusht.get("pusht_low_diversity_oracle_gain_high_minus_low"),
        "misaligned_real_change": pusht.get("pusht_misaligned_real_change_high_minus_low"),
        "oracle_minus_misaligned": pusht.get("pusht_oracle_minus_misaligned_high_n"),
        "aligned_oracle_ci": pusht_aligned_ci,
        "low_diversity_ci": pusht_low_div_ci,
        "misaligned_ci": pusht_misaligned_ci,
        "oracle_gap_ci": pusht_gap_ci,
        "rollout_max_coverage_ci": pusht_rollout_max_cov_ci,
        "rollout_final_coverage_ci": pusht_rollout_final_cov_ci,
        "rollout_success_ci": pusht_rollout_success_ci,
        "rollout_metrics_supported": pusht_rollout_metrics_strong,
        "thresholds": {
            "aligned_oracle_gain_min": 0.002,
            "oracle_minus_misaligned_min": 0.0,
            "min_ci_units": min_pusht_units,
            "min_training_seeds": min_pusht_training_seeds,
        },
    }
    add(
        claims,
        "pusht_benchmark",
        "A credible PushT simulator benchmark shows the same reranking law over actual sampled action trajectories.",
        status(pusht_strong, pusht_weak),
        strong_metrics["pusht_benchmark"],
    )

    strong_metrics["pusht_rollout_metrics"] = {
        "metric_columns": sorted(pusht_rollout_metric_columns),
        "curves_columns": sorted(csv_columns(RESULTS / "tables" / "pusht_curves.csv")),
        "effect_rows": len(pusht_rollout_metric_effect_cis),
        "seed_rows": len(pusht_rollout_metric_seed_agg),
        "aggregate_rows": len(pusht_rollout_metric_agg),
        "max_coverage_ci": pusht_rollout_max_cov_ci,
        "final_coverage_ci": pusht_rollout_final_cov_ci,
        "success_ci": pusht_rollout_success_ci,
        "thresholds": {"min_ci_units": min_pusht_units},
    }
    add(
        claims,
        "pusht_rollout_metrics",
        "PushT reports selected rollout coverage and success trajectory-search curves from actual simulator rollouts.",
        status(pusht_rollout_metrics_strong, pusht_rollout_metrics_weak),
        strong_metrics["pusht_rollout_metrics"],
    )

    runtime_tier_strong = (
        latency_strong
        and len(true_runtime) > 0
        and len(pusht_runtime) > 0
        and bool(true_diffusion.get("measured_wall_clock_runtime"))
        and bool(pusht.get("measured_wall_clock_runtime"))
        and {"runtime_per_candidate_ms", "K", "sampler"}.issubset(csv_columns(RESULTS / "tables" / "true_diffusion_runtime.csv"))
        and {"runtime_per_candidate_ms", "K", "sampler"}.issubset(csv_columns(RESULTS / "tables" / "pusht_runtime.csv"))
    )
    strong_metrics["measured_runtime_allocation"] = {
        "nk_latency_supported": latency_strong,
        "true_runtime_rows": len(true_runtime),
        "pusht_runtime_rows": len(pusht_runtime),
        "true_runtime_columns": sorted(csv_columns(RESULTS / "tables" / "true_diffusion_runtime.csv")),
        "pusht_runtime_columns": sorted(csv_columns(RESULTS / "tables" / "pusht_runtime.csv")),
    }
    add(
        claims,
        "latency",
        "N versus K recommendations are backed by both abstract budget sweeps and measured sampler/benchmark runtime.",
        status(runtime_tier_strong, len(true_runtime) > 0 and len(pusht_runtime) > 0),
        strong_metrics["measured_runtime_allocation"],
    )

    overclaims = forbidden_hits()
    real_robot_overclaim_hits = [
        hit for hit in overclaims if "robot" in hit.get("text", "").lower() or "real" in hit.get("text", "").lower()
    ]
    visual_overclaim_hits = [
        hit
        for hit in overclaims
        if "visual" in hit.get("text", "").lower()
        or "production-scale" in hit.get("text", "").lower()
        or "full-scale" in hit.get("text", "").lower()
    ]
    critical_ci_units = {
        "true_ddim_oracle": ci_units(true_ddim_oracle_ci),
        "true_ddpm_oracle": ci_units(true_ddpm_oracle_ci),
        "true_anti_correlated": ci_units(true_anti_ci),
        "true_oracle_tail_gap": ci_units(true_gap_ci),
        "pusht_aligned_oracle": ci_units(pusht_aligned_ci),
        "pusht_oracle_gap": ci_units(pusht_gap_ci),
        "pusht_rollout_max_coverage": ci_units(pusht_rollout_max_cov_ci),
        "pusht_rollout_final_coverage": ci_units(pusht_rollout_final_cov_ci),
        "pusht_rollout_success": ci_units(pusht_rollout_success_ci),
    }
    required_critical_ci_units = 1 if is_smoke_results() else 12
    underpowered_ci_units = {
        name: units for name, units in critical_ci_units.items() if units < required_critical_ci_units
    }
    low_statistical_power_warning = None
    if not is_smoke_results() and underpowered_ci_units:
        low_statistical_power_warning = (
            f"Full-run critical CI units below {required_critical_ci_units}: {underpowered_ci_units}"
        )
    low_statistical_power = {
        "is_smoke_results": is_smoke_results(),
        "required_critical_ci_units": required_critical_ci_units,
        "critical_ci_units": critical_ci_units,
        "underpowered_ci_units": underpowered_ci_units,
        "warning": low_statistical_power_warning,
    }

    claim_gates = {
        "toy_controlled": {
            "supported": aligned_strong and misaligned_strong and low_div_strong,
            "aligned_selection": aligned_strong,
            "misaligned_selection": misaligned_strong,
            "low_diversity": low_div_strong,
        },
        "learned_policy_lite": {
            "supported": learned_strong,
            "role": "supporting_evidence",
        },
        "true_ddpm": {
            "supported": true_strong,
            "required_for_global_diffusion_policy_wording": True,
        },
        "pusht": {
            "supported": pusht_strong and pusht_rollout_metrics_strong,
            "benchmark_supported": pusht_strong,
            "rollout_metrics_supported": pusht_rollout_metrics_strong,
            "required_for_global_diffusion_policy_wording": True,
        },
        "controller_fix": {
            "supported": controller_strong and low_div_strong and misaligned_strong and latency_strong and repair_strong,
            "audit_then_sample_supported": controller_strong,
            "diversity_gate_supported": low_div_strong,
            "tail_alignment_gate_supported": misaligned_strong and aligned_strong,
            "latency_gate_supported": latency_strong,
            "calibration_repair_supported": repair_strong,
            "negative_controls_supported": required_controller_negative_controls.issubset(controller_negative_controls),
            "required_for_fix_wording": True,
        },
    }
    strong_metrics["claim_gates"] = claim_gates

    negative_controls_present = (
        misaligned_strong
        and "anti_correlated_score" in true_scorers
        and bool(true_anti_ci)
        and "tail_only_misaligned_score" in true_scorers
        and bool(true_tail_ci)
        and "misaligned_corner_scorer" in pusht_scorers
        and bool(pusht_misaligned_ci)
    )
    reviewer_skepticism_checklist = {
        "true_ddpm_survives": true_strong,
        "pusht_survives": pusht_strong and pusht_rollout_metrics_strong,
        "controller_fix_tier_supported": claim_gates["controller_fix"]["supported"],
        "no_real_robot_overclaim": not real_robot_overclaim_hits,
        "no_full_visual_policy_overclaim": not visual_overclaim_hits,
        "runtime_evidence_present": runtime_tier_strong,
        "negative_controls_present": negative_controls_present,
        "low_power_warning_absent": low_statistical_power_warning is None,
    }
    reviewer_skepticism_strong = all(reviewer_skepticism_checklist.values())

    tiered_global_strong = true_strong and pusht_strong and pusht_rollout_metrics_strong
    strong_metrics["tiered_global_claim_gate"] = {
        "toy_controlled_supported": claim_gates["toy_controlled"]["supported"],
        "learned_policy_lite_supported_as_context": learned_strong,
        "audit_then_sample_controller_supported": controller_strong,
        "controller_fix_gate_supported": claim_gates["controller_fix"]["supported"],
        "true_action_diffusion_supported": true_strong,
        "pusht_benchmark_supported": pusht_strong,
        "pusht_rollout_metrics_supported": pusht_rollout_metrics_strong,
        "toy_only_explanation_blocked": true_strong and pusht_strong and pusht_rollout_metrics_strong,
        "survives_true_ddpm": true_strong,
        "survives_real_benchmark_path": pusht_strong,
        "requires_true_ddpm_and_pusht_rollout_metrics": True,
    }
    add(
        claims,
        "tiered_claim_gate",
        "Global diffusion-policy wording is promoted only when true-DDPM and PushT rollout-metric tiers are supported.",
        status(tiered_global_strong),
        strong_metrics["tiered_global_claim_gate"],
    )

    strong_metrics["reviewer_skepticism_checklist"] = {
        **reviewer_skepticism_checklist,
        "real_robot_overclaim_hits": real_robot_overclaim_hits,
        "visual_overclaim_hits": visual_overclaim_hits,
        "low_statistical_power": low_statistical_power,
    }
    add(
        claims,
        "reviewer_skepticism",
        "Reviewer-skepticism checklist passes for true DDPM, PushT, runtime, negative controls, overclaims, and statistical power.",
        status(reviewer_skepticism_strong),
        strong_metrics["reviewer_skepticism_checklist"],
    )

    payload = {
        "claims": claims,
        "claims_by_category": {},
        "claim_gates": claim_gates,
        "low_statistical_power": low_statistical_power,
        "reviewer_skepticism_checklist": strong_metrics["reviewer_skepticism_checklist"],
        "overclaims": overclaims,
        "num_supported": sum(c["status"] == "SUPPORTED" for c in claims),
        "num_partial": sum(c["status"] == "PARTIAL" for c in claims),
        "num_unsupported": sum(c["status"] == "UNSUPPORTED" for c in claims),
        "num_strong": sum(c["status"] == "SUPPORTED" for c in claims),
        "all_strong": all(c["status"] == "SUPPORTED" for c in claims) and not overclaims and reviewer_skepticism_strong,
        "strong_metrics": strong_metrics,
    }
    for claim in claims:
        payload["claims_by_category"].setdefault(claim["category"], []).append(claim)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "claims_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULTS / "ideal_metrics_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# Claims Status", ""]
    for claim in claims:
        lines.append(
            f"- Claim {claim['id']}: **{claim['status']}** - {claim['claim']} Evidence: {claim['evidence']}"
        )
    if overclaims:
        lines.extend(["", "## Overclaims"])
        for hit in overclaims:
            lines.append(f"- {hit['path']}:{hit['line']} {hit['text']}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "claims_status.md").write_text(text, encoding="utf-8")
    (RESULTS / "ideal_metrics_status.md").write_text(text, encoding="utf-8")
    print(text)

    if args.fail_on_error and (
        any(c["status"] != "SUPPORTED" for c in claims) or overclaims
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
