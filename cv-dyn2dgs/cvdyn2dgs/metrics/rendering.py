"""Surface-rendering quality metrics (proposal §8.3).

These are the metrics that decide RQ5 (is 2DGS better than a mesh on the same
surface?) and RQ6 (is a 2-D disk better than a thin 3-D Gaussian?).  Because both
questions are comparisons at equal geometry, every metric here is computed against a
*reference rendering of the same* :math:`\\Gamma_t` - not against the anatomy.

The theory predicts what each one should expose:

==============================  =================================================
``silhouette_iou`` / boundary F  planar-disk footprint error :math:`O(\\kappa s^2)`
                                 (Prop. 8.3) and the exactness of perspective-correct
                                 intersection (Prop. 8.4)
``depth_rmse``                   ray-plane conditioning :math:`1/|n^\\top d_q|`
                                 (Prop. 8.1) and local-coordinate error (Prop. 8.2)
``normal_angular_error``         unit-normal error :math:`O(h_{\\max}^2)` (Prop. 7.3)
                                 plus the :math:`O(\\kappa s)` normal deviation of
                                 Prop. 8.3
``hole_fraction`` / overlap      surfel coverage, the failure mode of proposal §6.4
==============================  =================================================

A note on where the mesh baseline is structurally different: a mesh produces binary
coverage, so its silhouette IoU is computed on a hard mask while the surfel alpha is
thresholded.  Sweeping the threshold (:func:`silhouette_iou_sweep`) rather than
fixing it at 0.5 avoids handing either representation an advantage from an arbitrary
cut-off.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "silhouette_iou",
    "silhouette_iou_sweep",
    "boundary_f_score",
    "depth_rmse",
    "normal_angular_error",
    "coverage_report",
    "CoverageReport",
]


def silhouette_iou(alpha: Tensor, target_mask: Tensor, *, threshold: float = 0.5) -> float:
    """IoU between the thresholded rendered alpha and the reference silhouette."""
    pred = alpha >= float(threshold)
    tgt = target_mask.to(torch.bool)
    inter = float((pred & tgt).sum().item())
    union = float((pred | tgt).sum().item())
    return inter / union if union > 0 else 1.0


def silhouette_iou_sweep(
    alpha: Tensor, target_mask: Tensor, *, thresholds: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
) -> dict[str, float]:
    """IoU across several alpha thresholds, plus the best value and its threshold.

    Reporting ``best_iou`` separately from ``iou@0.5`` keeps the soft-vs-binary
    comparison fair: a soft silhouette should not be penalised merely because 0.5 is
    not its optimal cut.
    """
    out: dict[str, float] = {}
    best, best_t = -1.0, 0.5
    for t in thresholds:
        v = silhouette_iou(alpha, target_mask, threshold=t)
        out[f"iou@{t}"] = v
        if v > best:
            best, best_t = v, t
    out["best_iou"] = best
    out["best_threshold"] = best_t
    return out


def _binary_boundary(mask: Tensor) -> Tensor:
    """1-pixel boundary of a 2-D binary mask."""
    m = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
    eroded = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
    return ((m - eroded) > 0.5).squeeze(0).squeeze(0)


def boundary_f_score(
    pred_mask: Tensor, target_mask: Tensor, *, tolerance_px: int = 2
) -> dict[str, float]:
    """Boundary F-score with a pixel tolerance.

    Precision counts predicted boundary pixels within ``tolerance_px`` of a
    reference boundary pixel; recall is the mirror image.  Dilation by a square
    structuring element implements the tolerance.  This is far more sensitive to
    silhouette quality than IoU, which is dominated by the interior.
    """
    pb = _binary_boundary(pred_mask).to(torch.float32)
    tb = _binary_boundary(target_mask).to(torch.float32)
    k = 2 * int(tolerance_px) + 1

    def dilate(x: Tensor) -> Tensor:
        return F.max_pool2d(x.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0).squeeze(0)

    pb_d, tb_d = dilate(pb), dilate(tb)
    n_p, n_t = float(pb.sum().item()), float(tb.sum().item())
    if n_p == 0 and n_t == 0:
        return {"boundary_precision": 1.0, "boundary_recall": 1.0, "boundary_f": 1.0}
    precision = float((pb * tb_d).sum().item()) / max(n_p, 1e-8)
    recall = float((tb * pb_d).sum().item()) / max(n_t, 1e-8)
    f = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f": f}


def depth_rmse(
    depth_pred: Tensor,
    depth_ref: Tensor,
    *,
    valid: Tensor | None = None,
) -> dict[str, float]:
    """Depth error in mm, restricted to pixels where both renderings hit a surface.

    Restricting to jointly valid pixels is essential: comparing a hit against a miss
    measures silhouette disagreement, which the IoU metrics already cover, and would
    otherwise swamp the depth number.
    """
    if valid is None:
        valid = torch.ones_like(depth_pred, dtype=torch.bool)
    v = valid.to(torch.bool)
    if int(v.sum().item()) == 0:
        return {"depth_rmse_mm": float("nan"), "depth_mae_mm": float("nan"), "depth_valid_px": 0.0}
    d = (depth_pred - depth_ref)[v]
    return {
        "depth_rmse_mm": float(torch.sqrt((d * d).mean()).item()),
        "depth_mae_mm": float(d.abs().mean().item()),
        "depth_p95_mm": float(d.abs().quantile(0.95).item()),
        "depth_valid_px": float(int(v.sum().item())),
    }


def normal_angular_error(
    normal_pred: Tensor,
    normal_ref: Tensor,
    *,
    valid: Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Angle between predicted and reference normals, in degrees.

    ``normal_pred`` / ``normal_ref`` are ``(3, H, W)``; both are renormalised first
    because the rasterisers output alpha-weighted sums rather than unit vectors.

    This is the quantity Prop. 7.3 bounds by
    :math:`\\arcsin(O(h_{\\max}^2))`; ``experiments/theory_checks.py`` sweeps
    :math:`h` and fits the exponent.
    """
    if valid is None:
        valid = torch.ones(normal_pred.shape[1:], dtype=torch.bool, device=normal_pred.device)
    v = valid.to(torch.bool)
    if int(v.sum().item()) == 0:
        return {"normal_mean_deg": float("nan"), "normal_median_deg": float("nan"), "normal_valid_px": 0.0}

    a = normal_pred / normal_pred.norm(dim=0, keepdim=True).clamp_min(eps)
    b = normal_ref / normal_ref.norm(dim=0, keepdim=True).clamp_min(eps)
    cos = (a * b).sum(dim=0).clamp(-1.0, 1.0)[v]
    deg = torch.rad2deg(torch.acos(cos))
    return {
        "normal_mean_deg": float(deg.mean().item()),
        "normal_median_deg": float(deg.median().item()),
        "normal_p95_deg": float(deg.quantile(0.95).item()),
        "normal_valid_px": float(int(v.sum().item())),
    }


@dataclass
class CoverageReport:
    """How well the disks cover the surface they sit on (proposal §6.4, §8.3)."""

    hole_fraction: float
    """Reference-silhouette pixels where the rendered alpha stayed below threshold -
    gaps opened by normal projection into expanding regions."""

    spill_fraction: float
    """Rendered pixels outside the reference silhouette - disks overhanging the edge."""

    overlap_fraction: float
    """Covered pixels with more than ``overlap_k`` contributing surfels - the
    clustering side of the same failure mode."""

    mean_contributors: float
    max_contributors: float
    covered_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {
            "hole_fraction": self.hole_fraction,
            "spill_fraction": self.spill_fraction,
            "overlap_fraction": self.overlap_fraction,
            "mean_contributors": self.mean_contributors,
            "max_contributors": self.max_contributors,
            "covered_fraction": self.covered_fraction,
        }


def coverage_report(
    alpha: Tensor,
    n_contributing: Tensor,
    target_mask: Tensor,
    *,
    alpha_threshold: float = 0.5,
    overlap_k: int = 8,
) -> CoverageReport:
    """Hole / spill / overlap statistics against a reference silhouette."""
    tgt = target_mask.to(torch.bool)
    covered = alpha >= float(alpha_threshold)
    n_tgt = float(tgt.sum().item())
    n_cov = float(covered.sum().item())

    holes = float((tgt & (~covered)).sum().item())
    spill = float(((~tgt) & covered).sum().item())
    over = float((covered & (n_contributing > int(overlap_k))).sum().item())

    contrib = (
        n_contributing[covered].to(torch.float32)
        if n_cov > 0
        else torch.zeros(1, device=n_contributing.device)
    )
    return CoverageReport(
        hole_fraction=holes / max(n_tgt, 1.0),
        spill_fraction=spill / max(n_tgt, 1.0),
        overlap_fraction=over / max(n_cov, 1.0),
        mean_contributors=float(contrib.mean().item()),
        max_contributors=float(contrib.max().item()),
        covered_fraction=n_cov / float(alpha.numel()),
    )
