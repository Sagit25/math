"""Build the canonical surfel set :math:`\\mathcal{G}^{2D}_0` on the frame-0 surface.

Sampling strategy
-----------------
Anchors are drawn **area-uniformly** from the marching-tetrahedra mesh of
:math:`\\Gamma_0`: a face is chosen with probability proportional to its area and
a point is drawn uniformly inside it.  This gives an unbiased, roughly
blue-noise-free but density-uniform covering of the surface, which is what keeps
the hole/overlap statistics of proposal §6.4 under control from the start.  The
anchors are then tightened onto :math:`\\{\\phi_0 = 0\\}` with Eq. (6.3), so the
mesh is only a *sampler*, never part of the stored representation.

Scale initialisation
--------------------
For :math:`N` disks covering area :math:`A`, the mean inter-anchor spacing is
:math:`\\approx \\sqrt{A/N}`, so the base radius is

    ``s = scale_init_factor * sqrt(A / N)``.

``scale_init_factor`` trades the hole fraction against overlap; both are measured
rather than assumed (proposal §8.3).

With ``curvature_adaptive_scale=True`` (the ``v3`` preset) the radius is further
capped so that the planar-disk approximation error of Prop. 8.3,
:math:`\\tfrac12 \\kappa s^2`, stays under a budget:

.. math::
    s_i \\le \\sqrt{2\\,\\text{budget} / |\\kappa_i|}.

This is a direct, quantitative use of the error analysis: disks shrink exactly
where the surface bends, and nowhere else.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..core.config import SurfelConfig
from ..core.grid import Grid, trilinear_sample, trilinear_sample_vector
from ..levelset.mesh_extract import TriangleMesh, marching_tetrahedra
from ..levelset.operators import curvature, gradient_central
from .model import SurfelSet2D
from .projection import project_to_surface
from .transport import initial_tangent_frame

__all__ = ["initialize_canonical_surfels", "sample_mesh_surface", "surface_normals_at"]


def sample_mesh_surface(
    mesh: TriangleMesh,
    n_points: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Draw ``n_points`` area-uniform samples from a triangle mesh.

    Sampling is performed on the CPU with an explicit generator and the indices
    are then moved to the mesh device, so results are bit-reproducible regardless
    of the compute device.

    Returns
    -------
    ``(points, total_area)`` with ``points`` of shape ``(n_points, 3)``.
    """
    if mesh.n_faces == 0:
        raise ValueError("cannot sample an empty mesh; is the level set degenerate?")

    areas = mesh.face_areas()
    total_area = areas.sum()
    cdf = torch.cumsum(areas.detach().to("cpu", torch.float64), dim=0)
    cdf = cdf / cdf[-1].clamp_min(1e-30)

    # seed_everything() returns a CPU generator, so these draws are CPU by necessity and
    # every derived tensor is moved to the mesh's device explicitly below.
    u = torch.rand(n_points, generator=generator, dtype=torch.float64)
    face_idx = torch.searchsorted(cdf, u.clamp(max=1.0 - 1e-9)).clamp(max=mesh.n_faces - 1)
    face_idx = face_idx.to(mesh.faces.device)

    r1 = torch.rand(n_points, generator=generator, dtype=torch.float64)
    r2 = torch.rand(n_points, generator=generator, dtype=torch.float64)
    su = torch.sqrt(r1)
    b0 = (1.0 - su).to(mesh.vertices.dtype).to(mesh.vertices.device)
    b1 = (su * (1.0 - r2)).to(mesh.vertices.dtype).to(mesh.vertices.device)
    b2 = (su * r2).to(mesh.vertices.dtype).to(mesh.vertices.device)

    tri = mesh.vertices[mesh.faces[face_idx]]  # (n, 3, 3)
    pts = b0.unsqueeze(-1) * tri[:, 0] + b1.unsqueeze(-1) * tri[:, 1] + b2.unsqueeze(-1) * tri[:, 2]
    return pts, total_area


def surface_normals_at(
    points: Tensor,
    phi: Tensor,
    grid: Grid,
    *,
    eps: float = 1e-8,
    grad_phi: Tensor | None = None,
    spacing_aware: bool = True,
) -> Tensor:
    """Unit normals :math:`n_i = \\nabla_h\\phi / (\\|\\nabla_h\\phi\\| + \\varepsilon)`, Eq. (7.3).

    Prop. 7.3 bounds the angular error of this estimate by
    :math:`\\sin\\theta \\le \\|e_h\\| / \\|g\\| = O(h_{\\max}^2)`, and notes the
    ``eps`` regularisation changes only the magnitude, never the direction.
    """
    if grad_phi is None:
        grad_phi = gradient_central(phi, grid.spacing, spacing_aware=spacing_aware)
    g = trilinear_sample_vector(grad_phi, grid.world_to_voxel(points))
    return g / (g.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def initialize_canonical_surfels(
    phi0: Tensor,
    image0: Tensor,
    grid: Grid,
    cfg: SurfelConfig,
    *,
    generator: torch.Generator | None = None,
    mesh: TriangleMesh | None = None,
    opacity_init: float = 0.9,
    spacing_aware: bool = True,
) -> tuple[SurfelSet2D, dict[str, float]]:
    """Construct :math:`\\mathcal{G}^{2D}_0` (Def. 7.1) from the frame-0 surface.

    Parameters
    ----------
    phi0:
        ``(nx, ny, nz)`` level set of frame 0, ``phi > 0`` inside.
    image0:
        ``(nx, ny, nz)`` intensity volume :math:`I_0`; used to initialise the
        amplitudes :math:`a^0_i` from the MRI itself rather than from a constant.
    cfg:
        Surfel configuration (count, isotropy, scale policy).

    Returns
    -------
    ``(surfels, stats)``
    """
    if mesh is None:
        mesh = marching_tetrahedra(phi0, grid, spacing_aware=spacing_aware)
    if mesh.n_faces == 0:
        raise ValueError(
            "frame-0 surface is empty: check the sign convention (phi > 0 inside) "
            "and the Chan-Vese initialisation"
        )

    pts, total_area = sample_mesh_surface(mesh, int(cfg.n_surfels), generator=generator)

    grad_phi = gradient_central(phi0, grid.spacing, spacing_aware=spacing_aware)

    # Tighten onto {phi = 0} with Eq. (6.3) so the stored anchors are on the
    # level set, not on the mesh approximation of it.
    proj = project_to_surface(
        pts,
        phi0,
        grid,
        iters=max(1, cfg.projection_iters),
        eps=cfg.projection_eps,
        max_step_mm=cfg.projection_max_step_mm,
        grad_phi=grad_phi,
        spacing_aware=spacing_aware,
    )
    anchors = proj.anchor

    normals = surface_normals_at(
        anchors, phi0, grid, eps=cfg.normal_eps, grad_phi=grad_phi, spacing_aware=spacing_aware
    )
    e1, e2 = initial_tangent_frame(normals, eps=cfg.normal_eps)

    # ---- scales -----------------------------------------------------------
    area = float(total_area.item())
    base_r = cfg.scale_init_factor * math.sqrt(max(area, 1e-12) / max(1, cfg.n_surfels))
    scale = torch.full((anchors.shape[0], 2), base_r, device=anchors.device, dtype=anchors.dtype)

    kappa_at: Tensor | None = None
    if cfg.curvature_adaptive_scale:
        kappa_vol = curvature(phi0, grid.spacing, spacing_aware=spacing_aware)
        kappa_at = trilinear_sample(kappa_vol, grid.world_to_voxel(anchors)).abs()
        # Prop. 8.3: height error ~ 0.5 * kappa * s^2  <=  budget
        cap = torch.sqrt(2.0 * cfg.curvature_error_budget_mm / kappa_at.clamp_min(1e-6))
        scale = torch.minimum(scale, cap.unsqueeze(-1).expand(-1, 2))

    scale = scale.clamp(cfg.scale_min_mm, cfg.scale_max_mm)
    if cfg.isotropic:
        iso = scale.mean(dim=1, keepdim=True).expand(-1, 2).contiguous()
        scale = iso

    # ---- appearance -------------------------------------------------------
    amp = trilinear_sample(image0, grid.world_to_voxel(anchors)).unsqueeze(-1)
    opa = torch.full((anchors.shape[0],), float(opacity_init), device=anchors.device, dtype=anchors.dtype)

    surfels = SurfelSet2D(
        anchor=anchors,
        e1=e1,
        e2=e2,
        normal=normals,
        scale=scale,
        amplitude=amp,
        opacity=opa,
        scale_min_mm=cfg.scale_min_mm,
        scale_max_mm=cfg.scale_max_mm,
    )

    stats: dict[str, float] = {
        "n_surfels": float(surfels.n),
        "surface_area_mm2": area,
        "base_radius_mm": base_r,
        "mean_scale_mm": float(scale.mean().item()),
        "min_scale_mm": float(scale.min().item()),
        "max_scale_mm": float(scale.max().item()),
        "mesh_vertices": float(mesh.n_vertices),
        "mesh_faces": float(mesh.n_faces),
        "init_e_surf_mm": float(proj.residual_mm.mean().item()),
    }
    if kappa_at is not None:
        stats["mean_abs_curvature_1_per_mm"] = float(kappa_at.mean().item())
        stats["planar_disk_error_mm"] = float(
            (0.5 * kappa_at * scale.max(dim=1).values ** 2).mean().item()
        )
    return surfels, stats
