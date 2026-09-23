"""Appearance residual: regularised least squares (theory §9) and low-rank compression."""

from .lowrank import LowRankResidual, compress_residual, rank_for_error, spectrum_report
from .solver import ResidualResult, ResidualView, conjugate_gradient, solve_residual

__all__ = [
    "LowRankResidual",
    "ResidualResult",
    "ResidualView",
    "compress_residual",
    "conjugate_gradient",
    "rank_for_error",
    "solve_residual",
    "spectrum_report",
]
