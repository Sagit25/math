"""Clinically interpretable measures: volume curve and ejection fraction.

Proposal §8.3 asks for these for a specific reason: they separate *visual* quality
from *quantitative* validity.  A viewer can look convincing while the underlying
segmentation is systematically off, and the volume curve is what exposes that.

.. math:: \\mathrm{EF} = \\frac{V_{\\mathrm{ED}} - V_{\\mathrm{ES}}}{V_{\\mathrm{ED}}}\\times 100\\%
   \\qquad \\text{(Eq. 41)}

Two volume estimators are provided.  The hard count ``#\\{\\phi>0\\}\\cdot h_xh_yh_z`` is
what a label-based reference computes, so it is the one to use for like-for-like
comparison against ACDC labels.  The smooth estimator :math:`\\sum H_\\varepsilon(\\phi)`
is sub-voxel accurate and much less noisy frame to frame, which matters on a stack
with :math:`h_z = 8\\,\\mathrm{mm}` where a single slice is ~6% of the cavity - but it
is *not* what the reference does, so mixing the two would bias the comparison.
Both are returned and it is stated which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..core.grid import Grid
from ..levelset.operators import heaviside_eps

__all__ = [
    "VolumeCurve",
    "volume_from_levelset",
    "volume_from_mask",
    "volume_curve",
    "ejection_fraction",
    "volume_curve_report",
]

_MM3_PER_ML = 1000.0


def volume_from_mask(mask: Tensor, grid: Grid) -> float:
    """Cavity volume in ml from a binary mask."""
    return float(mask.to(torch.bool).sum().item()) * grid.voxel_volume_mm3 / _MM3_PER_ML


def volume_from_levelset(
    phi: Tensor, grid: Grid, *, smooth_eps: float | None = None
) -> float:
    """Cavity volume in ml from a level set (``phi > 0`` inside).

    With ``smooth_eps`` set, integrates :math:`H_\\varepsilon(\\phi)` for a sub-voxel
    estimate; otherwise counts voxels, matching a label-based reference.
    """
    if smooth_eps is None:
        return float((phi > 0).sum().item()) * grid.voxel_volume_mm3 / _MM3_PER_ML
    occ = heaviside_eps(phi, float(smooth_eps))
    return float(occ.sum().item()) * grid.voxel_volume_mm3 / _MM3_PER_ML


@dataclass
class VolumeCurve:
    """Per-frame cavity volumes and the derived indices."""

    volumes_ml: list[float]
    ed_index: int
    es_index: int

    @property
    def v_ed(self) -> float:
        return self.volumes_ml[self.ed_index]

    @property
    def v_es(self) -> float:
        return self.volumes_ml[self.es_index]

    @property
    def stroke_volume_ml(self) -> float:
        return self.v_ed - self.v_es

    @property
    def ef_percent(self) -> float:
        return (self.v_ed - self.v_es) / max(self.v_ed, 1e-12) * 100.0

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "volumes_ml": list(self.volumes_ml),
            "ed_index": float(self.ed_index),
            "es_index": float(self.es_index),
            "v_ed_ml": self.v_ed,
            "v_es_ml": self.v_es,
            "stroke_volume_ml": self.stroke_volume_ml,
            "ef_percent": self.ef_percent,
        }


def volume_curve(
    phis: Sequence[Tensor], grid: Grid, *, smooth_eps: float | None = None
) -> VolumeCurve:
    """Volume curve over a cardiac cycle, with ED/ES taken as the extrema."""
    vols = [volume_from_levelset(p, grid, smooth_eps=smooth_eps) for p in phis]
    ed = int(max(range(len(vols)), key=lambda t: vols[t]))
    es = int(min(range(len(vols)), key=lambda t: vols[t]))
    return VolumeCurve(volumes_ml=vols, ed_index=ed, es_index=es)


def ejection_fraction(v_ed: float, v_es: float) -> float:
    """Eq. (41), in percent."""
    return (v_ed - v_es) / max(v_ed, 1e-12) * 100.0


def volume_curve_report(
    phis: Sequence[Tensor],
    grid: Grid,
    *,
    reference_volumes_ml: Sequence[float] | None = None,
    smooth_eps: float | None = None,
) -> dict[str, float | list[float]]:
    """Volume curve with hard and smooth estimators, optionally against a reference.

    When ``reference_volumes_ml`` is supplied the absolute EF error and the mean
    absolute volume error are reported.  EF error in **percentage points** is the
    clinically meaningful figure, so it is labelled as such rather than as a ratio.
    """
    hard = volume_curve(phis, grid, smooth_eps=None)
    out: dict[str, float | list[float]] = {f"hard/{k}": v for k, v in hard.to_dict().items()}

    if smooth_eps is not None:
        soft = volume_curve(phis, grid, smooth_eps=smooth_eps)
        out.update({f"smooth/{k}": v for k, v in soft.to_dict().items()})

    if reference_volumes_ml is not None:
        ref = list(reference_volumes_ml)
        if len(ref) != len(hard.volumes_ml):
            raise ValueError(
                f"reference has {len(ref)} frames, prediction has {len(hard.volumes_ml)}"
            )
        errs = [abs(a - b) for a, b in zip(hard.volumes_ml, ref)]
        ref_ed = max(ref)
        ref_es = min(ref)
        out["ref/ef_percent"] = ejection_fraction(ref_ed, ref_es)
        out["error/ef_percentage_points"] = abs(hard.ef_percent - out["ref/ef_percent"])
        out["error/volume_mae_ml"] = sum(errs) / max(1, len(errs))
        out["error/volume_max_ml"] = max(errs)
        out["error/volume_bias_ml"] = sum(
            a - b for a, b in zip(hard.volumes_ml, ref)
        ) / max(1, len(ref))
    return out
