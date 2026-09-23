"""Surface normal projection of surfel anchors.

Theory §6 derives the minimal normal step by imposing the first-order surface
condition of Eq. (6.5) and picking the least-norm displacement, Eq. (6.6).  The
result, Eq. (6.2), and its iteration, Eq. (6.3), are:

.. math::
    d^t_i = -\\frac{\\phi_t(p^{t-1}_i)}{\\|\\nabla_h\\phi_t(p^{t-1}_i)\\|^2 + \\varepsilon}
            \\nabla_h\\phi_t(p^{t-1}_i),
    \\qquad p^{t,(k+1)}_i = p^{t,(k)}_i + d^{t,(k)}_i .

Key properties implemented and instrumented here:

* **Prop. 6.2** - near an SDF (:math:`\\|\\nabla\\phi\\|\\approx1`) the step reduces to
  a standard Newton step and 1-2 iterations suffice, so ``K_p`` defaults to 2.
* **Lemma 6.3** - the residual contracts quadratically,
  :math:`|\\phi(p^{k+1})| \\le \\frac{M}{2\\|g\\|^2}|\\phi(p^k)|^2`.
  :func:`project_to_surface` records the residual after every iteration so the
  quadratic rate can be *measured* (see ``experiments/theory_checks.py``).
* **Remark on eps** - a non-zero :math:`\\varepsilon` leaves a first-order residual
  proportional to :math:`\\varepsilon/(\\|g\\|^2+\\varepsilon)`; this is reported as
  ``eps_residual_estimate``.

What this is *not*
------------------
The projection chooses the geometrically nearest surface: the tangential
component is deliberately left undetermined (Eq. 6.6 is under-determined and we
take the least-norm solution).  It therefore reproduces *surface shape motion*,
never myocardial material-point correspondence, and must not be read as strain
(proposal §2.8, §4.2, §10.3).

A guard against the dominant failure mode of proposal §6.3 - projecting onto a
*different* nearby surface branch - is included: steps are clipped to a trust
region and anchors whose residual grows are rolled back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from ..core.grid import Grid, trilinear_sample, trilinear_sample_vector
from ..levelset.operators import gradient_central

__all__ = ["ProjectionResult", "project_to_surface", "surface_seed_points", "e_surf"]

ProjectionMode = Literal["normal", "closest_point"]


@dataclass
class ProjectionResult:
    """Projected anchors plus everything needed to audit Eq. (6.3)."""

    anchor: Tensor
    """``(N, 3)`` projected anchors in world mm."""

    residual_mm: Tensor
    """``(N,)`` final :math:`|\\phi_t(p^t_i)|`. Its mean is :math:`E_{\\mathrm{surf}}`
    of proposal Eq. (39)."""

    grad_norm: Tensor
    """``(N,)`` :math:`\\|\\nabla_h\\phi_t(p^t_i)\\|`; should be ~1 near an SDF."""

    total_step_mm: Tensor
    """``(N,)`` cumulative displacement over all iterations."""

    residual_history: list[float] = field(default_factory=list)
    """Mean :math:`|\\phi|` after each iteration, starting with iteration 0
    (i.e. before any step).  Used for the quadratic-rate check of Lemma 6.3."""

    clipped_fraction: float = 0.0
    """Fraction of steps that hit the trust region - a proxy for how often the
    branch-confusion failure mode of proposal §6.3 was suppressed."""

    rolled_back_fraction: float = 0.0
    """Fraction of anchors whose residual increased and were reverted."""

    out_of_grid_fraction: float = 0.0

    eps_residual_estimate: float = 0.0
    """Mean of :math:`\\varepsilon/(\\|g\\|^2+\\varepsilon)` - the first-order
    residual the regularisation leaves behind (remark to Lemma 6.3)."""

    def summary(self) -> dict[str, float]:
        return {
            "e_surf_mm": float(self.residual_mm.mean().item()),
            "e_surf_p95_mm": float(self.residual_mm.quantile(0.95).item()),
            "e_surf_max_mm": float(self.residual_mm.max().item()),
            "grad_norm_mean": float(self.grad_norm.mean().item()),
            "total_step_mean_mm": float(self.total_step_mm.mean().item()),
            "total_step_max_mm": float(self.total_step_mm.max().item()),
            "clipped_fraction": self.clipped_fraction,
            "rolled_back_fraction": self.rolled_back_fraction,
            "out_of_grid_fraction": self.out_of_grid_fraction,
            "eps_residual_estimate": self.eps_residual_estimate,
        }


def e_surf(residual_mm: Tensor) -> Tensor:
    """:math:`E_{\\mathrm{surf}} = \\frac1N\\sum_i |\\phi_t(p^t_i)|`, proposal Eq. (39)."""
    return residual_mm.abs().mean()


@torch.no_grad()
def project_to_surface(
    anchors: Tensor,
    phi: Tensor,
    grid: Grid,
    *,
    iters: int = 2,
    eps: float = 1e-6,
    max_step_mm: float = 8.0,
    mode: ProjectionMode = "normal",
    grad_phi: Tensor | None = None,
    surface_points: Tensor | None = None,
    spacing_aware: bool = True,
    rollback_on_increase: bool = True,
    chunk: int = 4096,
) -> ProjectionResult:
    """Project anchors onto :math:`\\Gamma_t = \\{\\phi_t = 0\\}`.

    Parameters
    ----------
    anchors:
        ``(N, 3)`` world-mm anchors from the previous frame, :math:`p^{t-1}_i`.
    phi:
        ``(nx, ny, nz)`` current level set, ``phi > 0`` inside.
    iters:
        :math:`K_p` of Eq. (6.3).
    max_step_mm:
        Trust region on a single step's length.
    mode:
        ``"normal"`` implements Eq. (6.3).  ``"closest_point"`` is the ablation
        baseline of proposal §8.2 that drops the normal constraint and takes the
        nearest point of a precomputed surface point cloud.  Note the asymmetry
        in *cost*: the normal step is a couple of interpolations per anchor,
        while closest-point needs a spatial search (here a chunked brute force),
        which is itself part of why the paper prefers the normal step.
    grad_phi:
        Optional precomputed ``(3, nx, ny, nz)`` gradient, to avoid recomputing it
        per frame.
    surface_points:
        ``(M, 3)`` world-mm points on :math:`\\Gamma_t`; required for
        ``mode="closest_point"``.

    Returns
    -------
    :class:`ProjectionResult`
    """
    if anchors.dim() != 2 or anchors.shape[1] != 3:
        raise ValueError(f"anchors must be (N,3), got {tuple(anchors.shape)}")

    if mode == "closest_point":
        if surface_points is None:
            raise ValueError('mode="closest_point" requires surface_points')
        return _closest_point_projection(
            anchors, phi, grid, surface_points, chunk=chunk, spacing_aware=spacing_aware
        )

    if grad_phi is None:
        grad_phi = gradient_central(phi, grid.spacing, spacing_aware=spacing_aware)

    p = anchors.clone()
    start = anchors.clone()
    n_pts = p.shape[0]
    clipped_total = 0
    rolled_total = 0
    history: list[float] = []

    def residual_at(x: Tensor) -> tuple[Tensor, Tensor]:
        vox = grid.world_to_voxel(x)
        val = trilinear_sample(phi, vox)
        g = trilinear_sample_vector(grad_phi, vox)
        return val, g

    val, g = residual_at(p)
    history.append(float(val.abs().mean().item()))

    for _ in range(int(iters)):
        gsq = (g * g).sum(dim=-1)
        # Eq. (6.2) with the eps-regularised denominator.
        step = -(val / (gsq + eps)).unsqueeze(-1) * g

        # Trust region (proposal §6.3 failure-mode guard).
        slen = step.norm(dim=-1, keepdim=True)
        over = (slen > max_step_mm).squeeze(-1)
        clipped_total += int(over.sum().item())
        scale = torch.where(
            slen > max_step_mm, torch.full_like(slen, max_step_mm) / slen.clamp_min(1e-12),
            torch.ones_like(slen),
        )
        step = step * scale

        cand = p + step
        cand_val, cand_g = residual_at(cand)

        if rollback_on_increase:
            worse = cand_val.abs() > val.abs()
            rolled_total += int(worse.sum().item())
            keep = (~worse).unsqueeze(-1)
            p = torch.where(keep, cand, p)
            val = torch.where(worse, val, cand_val)
            g = torch.where(keep, cand_g, g)
        else:
            p, val, g = cand, cand_val, cand_g

        history.append(float(val.abs().mean().item()))

    gnorm = g.norm(dim=-1)
    in_grid = grid.inside_mask(p, margin_vox=0.0)
    denom = max(1, n_pts * int(iters))

    return ProjectionResult(
        anchor=p,
        residual_mm=val.abs(),
        grad_norm=gnorm,
        total_step_mm=(p - start).norm(dim=-1),
        residual_history=history,
        clipped_fraction=clipped_total / denom,
        rolled_back_fraction=rolled_total / denom,
        out_of_grid_fraction=float((~in_grid).to(torch.float32).mean().item()),
        eps_residual_estimate=float((eps / (gnorm**2 + eps)).mean().item()),
    )


@torch.no_grad()
def _closest_point_projection(
    anchors: Tensor,
    phi: Tensor,
    grid: Grid,
    surface_points: Tensor,
    *,
    chunk: int,
    spacing_aware: bool,
) -> ProjectionResult:
    """Unconstrained nearest-surface-point projection (ablation baseline).

    Brute force in chunks.  A spatial index would be faster but the purpose here
    is to isolate the *accuracy* difference against Eq. (6.3), and to make the
    cost gap visible rather than hide it.
    """
    out = torch.empty_like(anchors)
    for lo in range(0, anchors.shape[0], chunk):
        hi = min(lo + chunk, anchors.shape[0])
        d = torch.cdist(anchors[lo:hi], surface_points)  # (c, M)
        idx = d.argmin(dim=1)
        out[lo:hi] = surface_points[idx]

    grad_phi = gradient_central(phi, grid.spacing, spacing_aware=spacing_aware)
    vox = grid.world_to_voxel(out)
    val = trilinear_sample(phi, vox)
    g = trilinear_sample_vector(grad_phi, vox)
    return ProjectionResult(
        anchor=out,
        residual_mm=val.abs(),
        grad_norm=g.norm(dim=-1),
        total_step_mm=(out - anchors).norm(dim=-1),
        residual_history=[float(val.abs().mean().item())],
        out_of_grid_fraction=float((~grid.inside_mask(out)).to(torch.float32).mean().item()),
    )


@torch.no_grad()
def surface_seed_points(
    phi: Tensor,
    grid: Grid,
    *,
    max_points: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Voxel-resolution point cloud on :math:`\\{\\phi = 0\\}`.

    Every axis-aligned edge whose endpoints straddle zero contributes one point,
    placed at the linearly interpolated crossing.  Simple, sub-voxel accurate and
    fully vectorised.  Used to seed the closest-point baseline and as a fallback
    when no mesh is available.

    Returns
    -------
    ``(M, 3)`` world-mm points.  ``M`` is roughly three times the number of
    surface voxels, since the three axes are collected independently.
    """
    chunks: list[Tensor] = []
    for axis in range(3):
        d = -3 + axis
        n = phi.shape[d]
        if n < 2:
            continue
        a = phi.narrow(d, 0, n - 1)
        b = phi.narrow(d, 1, n - 1)
        crossing = (a * b) < 0
        if not bool(crossing.any()):
            continue
        denom = a - b
        t = torch.where(denom.abs() > 1e-12, a / denom, torch.full_like(denom, 0.5)).clamp(0.0, 1.0)

        # nonzero() on the reduced-shape mask yields the low-side voxel index.
        idx = torch.nonzero(crossing, as_tuple=False).to(phi.dtype)  # (M, 3)
        offs = torch.zeros_like(idx)
        offs[:, axis] = t[crossing]
        chunks.append(grid.voxel_to_world(idx + offs))

    if not chunks:
        return torch.zeros((0, 3), device=phi.device, dtype=phi.dtype)
    pts = torch.cat(chunks, dim=0)

    if max_points is not None and pts.shape[0] > max_points:
        perm = torch.randperm(pts.shape[0], device=pts.device, generator=generator)
        pts = pts[perm[:max_points]]
    return pts
