from __future__ import annotations

import pandas as pd
import pytest

from diffusion_audit.stats import bootstrap_mean_ci, paired_high_minus_low_ci


def test_bootstrap_mean_ci_reports_positive_margin():
    ci = bootstrap_mean_ci([1.0, 1.2, 1.4, 1.6], seed=4, n_boot=200)
    assert ci["n"] == 4
    assert ci["mean"] > 1.0
    assert ci["ci_low"] > 0.9
    assert ci["ci_high"] > ci["ci_low"]


def test_paired_high_minus_low_ci_uses_unit_pairing():
    rows = pd.DataFrame(
        [
            {"seed": 1, "state_idx": 0, "N": 1, "utility": 0.0},
            {"seed": 1, "state_idx": 0, "N": 4, "utility": 0.4},
            {"seed": 1, "state_idx": 1, "N": 1, "utility": 0.2},
            {"seed": 1, "state_idx": 1, "N": 4, "utility": 0.7},
        ]
    )
    ci = paired_high_minus_low_ci(
        rows,
        unit_cols=["seed", "state_idx"],
        value_col="utility",
        low_n=1,
        high_n=4,
        seed=8,
    )
    assert ci["n"] == 2
    assert ci["mean"] == pytest.approx(0.45)
    assert ci["low_n"] == 1
    assert ci["high_n"] == 4
