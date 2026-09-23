"""Surfel density control: repulsion, densification, pruning.

Motivation (proposal §6.4)
-------------------------
Repeated normal projection is *not* measure preserving.  Where the surface
contracts, anchors crowd together; where it expands, gaps open.  Neither is
visible in the surface residual :math:`E_{\\mathrm{surf}}` (Eq. 39) - anchors can
sit exactly on :math:`\\Gamma_t` and still be badly distributed - so it has to be
measured separately and corrected.

Three corrections are implemented, all optional and all ablatable:

``tangential_repulsion``
    A few steps of nearest-neighbour repulsion **restricted to the tangent
    plane**, followed by re-projection with Eq. (6.3).  Removing the normal
    component before moving, and re-projecting afterwards, guarantees anchors
    stay on :math:`\\Gamma_t`: the correction cannot trade surface accuracy for
    distribution quality.

``densify``
    Adds children in gaps, detected purely geometrically from the
    nearest-neighbour distance distribution.  Growth is capped, because an
    uncapped densifier would silently invalidate the storage comparison of
    Eq. (36)-(38) by inflating :math:`N`.

``prune``
    Removes near-transparent and heavily redundant surfels.

The cost of all three is reported so the ablation of proposal §8.4 can separate
their benefit from their overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..core.config import SurfelConfig
from ..core.grid import Grid
from ..levelset.operators import gradient_central
from .canonical import surface_normals_at
from .model import SurfelSet2D
from .projection import project_to_surface

__all__ = [
    "DensityStats",
    "knn",
    "nearest_neighbour_stats",
    "tangential_repulsion",
    "densify",
    "prune",
    "apply_density_control",
]


@dataclass
class DensityStats:
    """Distribution quality of the anchor set."""

    nn_mean_mm: float
    nn_std_mm: float
    nn_min_mm: float
    nn_p05_mm: float
    nn_p95_mm: float
    clustering_ratio: float
    """``nn_p05 / nn_mean`` - small values mean anchors are piling up."""

    gap_ratio: float
    """``nn_p95 / nn_mean`` - large values mean gaps are opening."""

    n_surfels: int

    def to_dict(self) -> dict[str, float]:
        return {
            "nn_mean_mm": self.nn_mean_mm,
            "nn_std_mm": self.nn_std_mm,
            "nn_min_mm": self.nn_min_mm,
            "nn_p05_mm": self.nn_p05_mm,
            "nn_p95_mm": self.nn_p95_mm,
            "clustering_ratio": self.clustering_ratio,
            "gap_ratio": self.gap_ratio,
            "n_surfels": float(self.n_surfels),
        }


@torch.no_grad()
def knn(points: Tensor, k: int = 1, *, chunk: int = 2048) -> tuple[Tensor, Tensor]:
    """Chunked exact k-nearest-neighbours, excluding self.

    Returns ``(dists, idx)`` each ``(N, k)``.  Brute force keeps the code
    dependency-free; ``chunk`` bounds peak memory at ``chunk * N`` floats.
    """
    n = points.shape[0]
    if n <= 1:
        z = torch.zeros((n, k), device=points.device, dtype=points.dtype)
        return z, torch.zeros((n, k), device=points.device, dtype=torch.long)
    k_eff = min(int(k), n - 1)

    d_out = torch.empty((n, k_eff), device=points.device, dtype=points.dtype)
    i_out = torch.empty((n, k_eff), device=points.device, dtype=torch.long)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        d = torch.cdist(points[lo:hi], points)  # (c, N)
        rows = torch.arange(lo, hi, device=points.device)
        d[torch.arange(hi - lo, device=points.device), rows] = float("inf")
        vals, idx = torch.topk(d, k_eff, dim=1, largest=False)
        d_out[lo:hi] = vals
        i_out[lo:hi] = idx
    return d_out, i_out


@torch.no_grad()
def nearest_neighbour_stats(anchors: Tensor, *, chunk: int = 2048) -> DensityStats:
    """Nearest-neighbour distance statistics of the anchor cloud."""
    d, _ = knn(anchors, k=1, chunk=chunk)
    d1 = d[:, 0]
    mean = float(d1.mean().item())
    return DensityStats(
        nn_mean_mm=mean,
        nn_std_mm=float(d1.std(unbiased=False).item()),
        nn_min_mm=float(d1.min().item()),
        nn_p05_mm=float(d1.quantile(0.05).item()),
        nn_p95_mm=float(d1.quantile(0.95).item()),
        clustering_ratio=float(d1.quantile(0.05).item()) / max(mean, 1e-12),
        gap_ratio=float(d1.quantile(0.95).item()) / max(mean, 1e-12),
        n_surfels=int(anchors.shape[0]),
    )


@torch.no_grad()
def tangential_repulsion(
    surfels: SurfelSet2D,
    phi: Tensor,
    grid: Grid,
    cfg: SurfelConfig,
    *,
    k: int = 6,
    grad_phi: Tensor | None = None,
    spacing_aware: bool = True,
) -> dict[str, float]:
    """Spread anchors within the tangent plane, then re-project onto the surface.

    The displacement is a normalised inverse-distance repulsion from the ``k``
    nearest neighbours, with the component along :math:`n^t_i` removed.  After
    moving, Eq. (6.3) is re-applied, so the final anchors satisfy the same
    surface-residual bound as without repulsion.

    Modifies ``surfels`` in place.  Returns timing/effect statistics.
    """
    if cfg.repulsion_steps <= 0 or surfels.n < 2:
        return {"repulsion_steps": 0.0}
    if grad_phi is None:
        grad_phi = gradient_central(phi, grid.spacing, spacing_aware=spacing_aware)

    before = nearest_neighbour_stats(surfels.anchor)
    p = surfels.anchor.clone()
    radius = surfels.scale.max(dim=1).values  # (N,)

    for _ in range(int(cfg.repulsion_steps)):
        d, idx = knn(p, k=k)
        nb = p[idx]  # (N, k, 3)
        diff = p.unsqueeze(1) - nb  # (N, k, 3)
        dist = d.clamp_min(1e-9)  # (N, k)

        # Only neighbours closer than the disk radius exert a force.
        target = radius.unsqueeze(1)
        w = (1.0 - dist / target.clamp_min(1e-9)).clamp_min(0.0)  # (N, k)
        force = (diff / dist.unsqueeze(-1) * w.unsqueeze(-1)).sum(dim=1)  # (N, 3)

        # Strip the normal component: the correction is tangential by construction.
        n = surfels.normal
        force = force - (force * n).sum(-1, keepdim=True) * n

        step = cfg.repulsion_strength * force * radius.unsqueeze(-1)
        # Never move further than the local disk radius in one step.
        slen = step.norm(dim=-1, keepdim=True)
        cap = radius.unsqueeze(-1)
        step = step * torch.where(slen > cap, cap / slen.clamp_min(1e-12), torch.ones_like(slen))
        p = p + step

    proj = project_to_surface(
        p,
        phi,
        grid,
        iters=max(1, cfg.projection_iters),
        eps=cfg.projection_eps,
        max_step_mm=cfg.projection_max_step_mm,
        grad_phi=grad_phi,
        spacing_aware=spacing_aware,
    )
    normals = surface_normals_at(
        proj.anchor, phi, grid, eps=cfg.normal_eps, grad_phi=grad_phi, spacing_aware=spacing_aware
    )
    surfels.set_geometry(anchor=proj.anchor, normal=normals)

    after = nearest_neighbour_stats(surfels.anchor)
    return {
        "repulsion_steps": float(cfg.repulsion_steps),
        "nn_std_before_mm": before.nn_std_mm,
        "nn_std_after_mm": after.nn_std_mm,
        "clustering_ratio_before": before.clustering_ratio,
        "clustering_ratio_after": after.clustering_ratio,
        "gap_ratio_before": before.gap_ratio,
        "gap_ratio_after": after.gap_ratio,
        "e_surf_after_mm": float(proj.residual_mm.mean().item()),
    }


@torch.no_grad()
def densify(
    surfels: SurfelSet2D,
    phi: Tensor,
    grid: Grid,
    cfg: SurfelConfig,
    *,
    n_initial: int,
    gap_factor: float = 1.6,
    grad_phi: Tensor | None = None,
    spacing_aware: bool = True,
) -> tuple[SurfelSet2D, dict[str, float]]:
    """Insert surfels into gaps, respecting a global growth cap.

    A surfel is a gap candidate when its nearest-neighbour distance exceeds
    ``gap_factor`` times the median.  One child is placed at the midpoint towards
    that neighbour and projected onto :math:`\\Gamma_t`.

    ``cfg.densify_max_growth`` caps ``N`` at ``(1 + growth) * n_initial`` where
    ``n_initial`` is the canonical count, so that the compression ratio of
    Eq. (38) cannot be improved by quietly adding primitives.
    """
    max_n = int((1.0 + cfg.densify_max_growth) * n_initial)
    if surfels.n >= max_n:
        return surfels, {"densified": 0.0, "capped": 1.0, "n_surfels": float(surfels.n)}

    d, idx = knn(surfels.anchor, k=1)
    d1 = d[:, 0]
    median = d1.median()
    candidates = torch.nonzero(d1 > gap_factor * median, as_tuple=False).squeeze(1)
    if candidates.numel() == 0:
        return surfels, {"densified": 0.0, "capped": 0.0, "n_surfels": float(surfels.n)}

    budget = max_n - surfels.n
    if candidates.numel() > budget:
        # Prefer the widest gaps.
        order = torch.argsort(d1[candidates], descending=True)
        candidates = candidates[order[:budget]]

    parent = candidates
    partner = idx[candidates, 0]
    mid = 0.5 * (surfels.anchor[parent] + surfels.anchor[partner])

    if grad_phi is None:
        grad_phi = gradient_central(phi, grid.spacing, spacing_aware=spacing_aware)
    proj = project_to_surface(
        mid,
        phi,
        grid,
        iters=max(1, cfg.projection_iters),
        eps=cfg.projection_eps,
        max_step_mm=cfg.projection_max_step_mm,
        grad_phi=grad_phi,
        spacing_aware=spacing_aware,
    )
    new_anchor = proj.anchor
    new_normal = surface_normals_at(
        new_anchor, phi, grid, eps=cfg.normal_eps, grad_phi=grad_phi, spacing_aware=spacing_aware
    )

    # Inherit the parent's frame, re-orthogonalised against the child's normal.
    e1p = surfels.e1[parent]
    e1n = e1p - (e1p * new_normal).sum(-1, keepdim=True) * new_normal
    e1n = e1n / e1n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    e2n = torch.cross(new_normal, e1n, dim=-1)

    children = SurfelSet2D(
        anchor=new_anchor,
        e1=e1n,
        e2=e2n,
        normal=new_normal,
        scale=surfels.scale[parent],
        amplitude=surfels.amplitude.detach()[parent],
        opacity=surfels.opacity.detach()[parent],
        scale_min_mm=surfels.scale_min_mm,
        scale_max_mm=surfels.scale_max_mm,
    )
    merged = surfels.concat(children)
    return merged, {
        "densified": float(children.n),
        "capped": 0.0,
        "n_surfels": float(merged.n),
    }


@torch.no_grad()
def prune(
    surfels: SurfelSet2D,
    cfg: SurfelConfig,
    *,
    min_keep: int = 1,
) -> tuple[SurfelSet2D, dict[str, float]]:
    """Drop near-transparent and heavily redundant surfels.

    Redundancy is measured geometrically: a surfel whose nearest neighbour is
    closer than ``prune_overlap_thresh`` times its own radius contributes almost
    nothing that the neighbour does not already cover.  Only one of each such
    pair is removed (the one with the lower opacity), so the operation cannot
    cascade into a hole.
    """
    n0 = surfels.n
    keep = surfels.opacity.detach() >= cfg.prune_opacity_thresh

    if cfg.prune_overlap_thresh > 0 and n0 > 1:
        d, idx = knn(surfels.anchor, k=1)
        radius = surfels.scale.max(dim=1).values
        too_close = d[:, 0] < cfg.prune_overlap_thresh * radius
        partner = idx[:, 0]
        opa = surfels.opacity.detach()
        # Break ties by index so exactly one of each pair is dropped.
        loser = too_close & (
            (opa < opa[partner]) | ((opa == opa[partner]) & (torch.arange(n0, device=opa.device) < partner))
        )
        keep = keep & (~loser)

    if int(keep.sum().item()) < min_keep:
        return surfels, {"pruned": 0.0, "n_surfels": float(n0)}

    index = torch.nonzero(keep, as_tuple=False).squeeze(1)
    out = surfels.gather(index)
    return out, {"pruned": float(n0 - out.n), "n_surfels": float(out.n)}


@torch.no_grad()
def apply_density_control(
    surfels: SurfelSet2D,
    phi: Tensor,
    grid: Grid,
    cfg: SurfelConfig,
    *,
    n_initial: int,
    grad_phi: Tensor | None = None,
    spacing_aware: bool = True,
) -> tuple[SurfelSet2D, dict[str, float]]:
    """Run whichever density corrections the config enables, in order.

    Order matters: prune first (cheap, shrinks the working set), then repel
    (redistributes), then densify (fills what is left).
    """
    stats: dict[str, float] = {}
    out = surfels

    if cfg.prune_enabled:
        out, s = prune(out, cfg)
        stats.update({f"prune/{k}": v for k, v in s.items()})

    if cfg.repulsion_enabled:
        s = tangential_repulsion(
            out, phi, grid, cfg, grad_phi=grad_phi, spacing_aware=spacing_aware
        )
        stats.update({f"repulsion/{k}": v for k, v in s.items()})

    if cfg.densify_enabled:
        out, s = densify(
            out,
            phi,
            grid,
            cfg,
            n_initial=n_initial,
            grad_phi=grad_phi,
            spacing_aware=spacing_aware,
        )
        stats.update({f"densify/{k}": v for k, v in s.items()})

    stats.update({f"density/{k}": v for k, v in nearest_neighbour_stats(out.anchor).to_dict().items()})
    return out, stats
