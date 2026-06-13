"""Shared exact-curve evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from diffusion_audit.alignment import (
    high_n_regret,
    score_utility_correlation,
    tail_rank_correlation,
    top_score_tail_real_utility,
)
from diffusion_audit.theory import (
    selected_score_max_selection_finite,
    simulate_max_selection,
    utility_max_selection_finite,
)


def evaluate_pool(scores, utilities, n_values: Iterable[int], mc_trials: int = 300, seed: int = 0) -> dict:
    """Evaluate exact and Monte Carlo trajectory-selection diagnostics for one pool."""

    ns = [int(n) for n in n_values]
    score_arr = np.asarray(scores, dtype=float)
    utility_arr = np.asarray(utilities, dtype=float)
    real_curve = utility_max_selection_finite(score_arr, utility_arr, ns)
    score_curve = selected_score_max_selection_finite(score_arr, ns)
    oracle_curve = utility_max_selection_finite(utility_arr, utility_arr, ns)
    mc = {
        n: float(np.mean(simulate_max_selection(score_arr, utility_arr, n, trials=mc_trials, seed=seed + int(n))))
        for n in ns
    }
    max_n = max(ns)
    min_n = min(ns)
    return {
        "real_curve": real_curve,
        "score_curve": score_curve,
        "oracle_curve": oracle_curve,
        "mc_curve": mc,
        "score_utility_correlation": score_utility_correlation(score_arr, utility_arr),
        "tail_rank_correlation": tail_rank_correlation(score_arr, utility_arr),
        "top_score_tail_real_utility": top_score_tail_real_utility(score_arr, utility_arr),
        "high_n_regret": high_n_regret(oracle_curve, real_curve),
        "real_change": float(real_curve[max_n] - real_curve[min_n]),
        "score_change": float(score_curve[max_n] - score_curve[min_n]),
    }


def curve_rows(
    family: str,
    regime: str,
    scorer: str,
    seed: int,
    eval_payload: dict,
    extra: dict | None = None,
) -> list[dict]:
    rows = []
    extra = extra or {}
    for n in sorted(eval_payload["real_curve"]):
        rows.append(
            {
                "family": family,
                "regime": regime,
                "scorer": scorer,
                "seed": int(seed),
                "N": int(n),
                "exact_selected_real": float(eval_payload["real_curve"][n]),
                "exact_selected_score": float(eval_payload["score_curve"][n]),
                "mc_selected_real": float(eval_payload["mc_curve"][n]),
                "oracle_selected_real": float(eval_payload["oracle_curve"][n]),
                "score_utility_correlation": float(eval_payload["score_utility_correlation"]),
                "tail_rank_correlation": float(eval_payload["tail_rank_correlation"]),
                "top_score_tail_real_utility": float(eval_payload["top_score_tail_real_utility"]),
                "high_n_regret": float(eval_payload["high_n_regret"]),
                "real_change_high_minus_low": float(eval_payload["real_change"]),
                "score_change_high_minus_low": float(eval_payload["score_change"]),
                **extra,
            }
        )
    return rows
