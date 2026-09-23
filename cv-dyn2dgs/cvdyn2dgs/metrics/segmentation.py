"""Segmentation and surface-accuracy metrics (proposal §8.3, Eq. 39).

Dice, HD95 and ASSD are the standard trio for cardiac segmentation and are what
ACDC/M&Ms-2 comparisons are reported in.  Two implementation points matter for
honesty:

* **Everything is in millimetres.**  Surface distances computed in voxel index
  units would be meaningless on a stack with :math:`h_z/h_x \\approx 6`.
* **Surfaces are sub-voxel.**  Boundary-voxel centres quantise distances to the
  grid, which flatters a method whose errors are below one voxel.  When a
  marching-tetrahedra mesh is available its vertices are used instead, so HD95 can
  resolve errors finer than :math:`h`.

:func:`e_surf` is the surfel-specific measure of proposal Eq. (39),
:math:`E_{\\mathrm{surf}} = \\frac1N \\sum_i |\\phi_t(p^t_i)|`.  It answers a
different question from Dice: it asks whether the *anchors landed on the level set*,
not whether the level set is anatomically right.  Both are needed - anchors can sit
perfectly on a wrong surface, or be scattered off a correct one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..core.grid import Grid

__all__ = [
    "SurfaceDistanceReport",
    "dice",
    "mask_surface_points",
    "surface_distances",
    "surface_distance_report",
    "hausdorff95",
    "assd",
    "e_surf",
]


def dice(pred: Tensor, target: Tensor) -> float:
    """Dice similarity coefficient :math:`2|A\\cap B| / (|A| + |B|)`.

    Returns ``1.0`` when both masks are empty, which is the conventional (and
    defensible) choice: a method that correctly predicts "nothing here" on a slice
    with no cavity should not be punished.
    """
    a = pred.to(torch.bool)
    b = target.to(torch.bool)
    inter = float((a & b).sum().item())
    sa, sb = float(a.sum().item()), float(b.sum().item())
    if sa + sb == 0.0:
        return 1.0
    return 2.0 * inter / (sa + sb)


def mask_surface_points(mask: Tensor, grid: Grid) -> Tensor:
    """World-mm centres of boundary voxels of a binary mask.

    A voxel is on the boundary when at least one 6-neighbour has the opposite
    label.  Quantised to the grid; prefer mesh vertices when available.
    """
    m = mask.to(torch.bool)
    boundary = torch.zeros_like(m)
    for axis in range(3):
        d = -3 + axis
        n = m.shape[d]
        if n < 2:
            continue
        diff = m.narrow(d, 1, n - 1) != m.narrow(d, 0, n - 1)
        lo = torch.zeros_like(m)
        hi = torch.zeros_like(m)
        lo.narrow(d, 0, n - 1).copy_(diff)
        hi.narrow(d, 1, n - 1).copy_(diff)
        boundary |= (lo | hi) & m
    idx = torch.nonzero(boundary, as_tuple=False)
    if idx.numel() == 0:
        return torch.zeros((0, 3), device=mask.device, dtype=torch.float32)
    return grid.voxel_to_world(idx.to(torch.float32))


@torch.no_grad()
def surface_distances(
    points_a: Tensor, points_b: Tensor, *, chunk: int = 4096
) -> Tensor:
    """Distance from every point of ``A`` to the nearest point of ``B``, in mm.

    Chunked brute force: exact, dependency-free, and fast enough because cardiac
    surfaces have :math:`O(10^4)` points.
    """
    if points_a.numel() == 0 or points_b.numel() == 0:
        return torch.full((max(1, points_a.shape[0]),), float("nan"), device=points_a.device)
    out = torch.empty((points_a.shape[0],), device=points_a.device, dtype=points_a.dtype)
    for lo in range(0, points_a.shape[0], chunk):
        hi = min(lo + chunk, points_a.shape[0])
        out[lo:hi] = torch.cdist(points_a[lo:hi], points_b).min(dim=1).values
    return out


@dataclass
class SurfaceDistanceReport:
    """Symmetric surface-distance summary, all values in mm."""

    hd95_mm: float
    hd_max_mm: float
    assd_mm: float
    mean_a_to_b_mm: float
    mean_b_to_a_mm: float
    n_points_a: int
    n_points_b: int

    def to_dict(self) -> dict[str, float]:
        return {
            "hd95_mm": self.hd95_mm,
            "hd_max_mm": self.hd_max_mm,
            "assd_mm": self.assd_mm,
            "mean_a_to_b_mm": self.mean_a_to_b_mm,
            "mean_b_to_a_mm": self.mean_b_to_a_mm,
            "n_points_a": float(self.n_points_a),
            "n_points_b": float(self.n_points_b),
        }


@torch.no_grad()
def surface_distance_report(points_a: Tensor, points_b: Tensor) -> SurfaceDistanceReport:
    """Compute HD95, max Hausdorff and ASSD between two surface point sets.

    HD95 is the *maximum* of the two directed 95th percentiles, and ASSD averages
    both directed means weighted by point count.  Reporting a single direction
    hides one-sided failures (e.g. a surface that covers the target but bulges
    elsewhere).
    """
    d_ab = surface_distances(points_a, points_b)
    d_ba = surface_distances(points_b, points_a)
    q = torch.tensor(0.95, device=d_ab.device, dtype=d_ab.dtype)
    hd95 = max(float(d_ab.quantile(q).item()), float(d_ba.quantile(q).item()))
    hd_max = max(float(d_ab.max().item()), float(d_ba.max().item()))
    n_a, n_b = int(points_a.shape[0]), int(points_b.shape[0])
    assd_val = (float(d_ab.sum().item()) + float(d_ba.sum().item())) / max(1, n_a + n_b)
    return SurfaceDistanceReport(
        hd95_mm=hd95,
        hd_max_mm=hd_max,
        assd_mm=assd_val,
        mean_a_to_b_mm=float(d_ab.mean().item()),
        mean_b_to_a_mm=float(d_ba.mean().item()),
        n_points_a=n_a,
        n_points_b=n_b,
    )


def hausdorff95(points_a: Tensor, points_b: Tensor) -> float:
    """HD95 in mm (see :func:`surface_distance_report`)."""
    return surface_distance_report(points_a, points_b).hd95_mm


def assd(points_a: Tensor, points_b: Tensor) -> float:
    """Average symmetric surface distance in mm."""
    return surface_distance_report(points_a, points_b).assd_mm


def e_surf(residual_mm: Tensor) -> dict[str, float]:
    """:math:`E_{\\mathrm{surf}}` of proposal Eq. (39), with tail statistics.

    The mean alone can hide a small population of badly mis-projected anchors -
    exactly the branch-confusion failure mode of proposal §6.3 - so the 95th
    percentile and maximum are reported alongside it.
    """
    r = residual_mm.abs()
    return {
        "e_surf_mm": float(r.mean().item()),
        "e_surf_p95_mm": float(r.quantile(0.95).item()),
        "e_surf_max_mm": float(r.max().item()),
        "e_surf_std_mm": float(r.std(unbiased=False).item()),
    }
