"""Audit-Then-Sample diagnostics for diffusion action trajectory search."""

from diffusion_audit.controller import (
    audit_then_sample,
    audit_then_sample_adaptive,
    empirical_bernstein_lcb,
    empirical_bernstein_ucb,
    fit_isotonic_score_calibration,
    validate_calibration_repair,
    validate_repair_with_bounds,
)
from diffusion_audit.theory import (
    binary_max_selection_finite,
    selected_score_max_selection_finite,
    simulate_max_selection,
    utility_max_selection_finite,
)

__all__ = [
    "audit_then_sample",
    "audit_then_sample_adaptive",
    "binary_max_selection_finite",
    "empirical_bernstein_lcb",
    "empirical_bernstein_ucb",
    "fit_isotonic_score_calibration",
    "selected_score_max_selection_finite",
    "simulate_max_selection",
    "utility_max_selection_finite",
    "validate_calibration_repair",
    "validate_repair_with_bounds",
]
