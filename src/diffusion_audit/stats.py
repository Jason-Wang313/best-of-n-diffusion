"""Small uncertainty summaries for CPU-light experiment artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import NormalDist

import numpy as np
import pandas as pd


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    seed: int = 0,
    n_boot: int = 500,
) -> dict[str, float | int]:
    """Return a bootstrap mean/SE/CI summary for one vector of values."""

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "confidence": float(confidence),
        }
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    if arr.size == 1:
        lo = hi = mean
    else:
        rng = np.random.default_rng(seed)
        draws = rng.choice(arr, size=(int(n_boot), arr.size), replace=True).mean(axis=1)
        alpha = 1.0 - float(confidence)
        lo = float(np.quantile(draws, alpha / 2.0))
        hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return {
        "n": int(arr.size),
        "mean": mean,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "confidence": float(confidence),
    }


def paired_high_minus_low_effects(
    rows: pd.DataFrame,
    *,
    unit_cols: list[str],
    value_col: str,
    low_n: int,
    high_n: int,
) -> np.ndarray:
    """Compute high-N minus low-N effects after pairing rows by unit."""

    sub = rows[rows["N"].isin([int(low_n), int(high_n)])]
    if sub.empty:
        return np.asarray([], dtype=float)
    pivot = sub.pivot_table(index=unit_cols, columns="N", values=value_col, aggfunc="mean")
    if int(low_n) not in pivot.columns or int(high_n) not in pivot.columns:
        return np.asarray([], dtype=float)
    effects = pivot[int(high_n)] - pivot[int(low_n)]
    return effects.dropna().to_numpy(dtype=float)


def paired_high_minus_low_ci(
    rows: pd.DataFrame,
    *,
    unit_cols: list[str],
    value_col: str,
    low_n: int,
    high_n: int,
    seed: int = 0,
) -> dict[str, float | int]:
    effects = paired_high_minus_low_effects(
        rows,
        unit_cols=unit_cols,
        value_col=value_col,
        low_n=int(low_n),
        high_n=int(high_n),
    )
    payload = bootstrap_mean_ci(effects, seed=seed)
    payload["low_n"] = int(low_n)
    payload["high_n"] = int(high_n)
    payload["effect"] = "high_minus_low"
    payload["value_col"] = value_col
    return payload


def mean_ci_columns(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    numeric_cols: list[str],
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate means and lightweight normal-approximation CI columns."""

    rows: list[dict[str, float | int | str]] = []
    grouped = frame.groupby(group_cols, dropna=False)
    z = NormalDist().inv_cdf(0.5 + 0.5 * 0.95)
    for group_key, group in grouped:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, float | int | str] = dict(zip(group_cols, key_values, strict=True))
        row["n_rows"] = int(len(group))
        for col in numeric_cols:
            arr = group[col].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                mean = se = lo = hi = float("nan")
            else:
                mean = float(np.mean(arr))
                se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
                lo = mean - z * se
                hi = mean + z * se
            row[col] = mean
            row[f"{col}_se"] = se
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)
