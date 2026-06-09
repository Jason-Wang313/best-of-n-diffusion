"""Best-of-N laws and diagnostics for diffusion action trajectory sampling."""

from diffusion_best_of_n.controller import (
    audit_then_sample,
    audit_then_sample_adaptive,
    empirical_bernstein_lcb,
    empirical_bernstein_ucb,
    fit_isotonic_score_calibration,
    validate_calibration_repair,
    validate_repair_with_bounds,
)
from diffusion_best_of_n.theory import (
    binary_best_of_n_finite,
    selected_score_best_of_n_finite,
    simulate_best_of_n,
    utility_best_of_n_finite,
)

__all__ = [
    "audit_then_sample",
    "audit_then_sample_adaptive",
    "binary_best_of_n_finite",
    "empirical_bernstein_lcb",
    "empirical_bernstein_ucb",
    "fit_isotonic_score_calibration",
    "selected_score_best_of_n_finite",
    "simulate_best_of_n",
    "utility_best_of_n_finite",
    "validate_calibration_repair",
    "validate_repair_with_bounds",
]
