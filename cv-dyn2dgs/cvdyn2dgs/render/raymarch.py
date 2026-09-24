"""Ray-marched reference rendering of the Chan-Vese level set.

This module answers a question the proposal raises but does not spell out
operationally: *what exactly is the supervision target?*

Cine CMR gives a voxel volume, not photographs.  Proposal §7.1 and §10.3 restrict
supervision to "the mask, depth, normal and intensity of known slice planes", and
theory §11.3 stresses that 2DGS is being used as a surface renderer for a known
geometry, not as a scene reconstructor.  Concretely that means: for each pixel ray of
a known slice-plane camera, find where the ray first crosses
:math:`\\Gamma_t = \\{\\phi_t = 0\\}`, and read off

* ``hit``       - the reference silhouette, target for :math:`\\mathcal{L}_{\\mathrm{mask}}`,
* ``depth``     - the reference :math:`\\tau`, reference for depth RMSE,
* ``normal``    - :math:`\\nabla_h\\phi / \\|\\nabla_h\\phi\\|` at the crossing (Eq. 7.3),
  reference for the normal angular error of Prop. 7.3,
* ``intensity`` - the MRI value **at the surface**, target for
  :math:`\\mathcal{L}_{\\mathrm{app}}`.

Every one of these comes from the level set and the MRI alone, so the surfel fit is
never given information the stored representation does not already contain.  Crucially
this makes the mesh, thin-3DGS and 2DGS comparisons of RQ5/RQ6 fair: all three are
scored against the *same* reference rendering of the *same* surface.

Method: uniform sampling of :math:`\\phi` along each ray inside the grid's bounding
box to bracket the first outside-to-inside crossing, then bisection to refine.
Uniform-then-bisect rather than sphere tracing, because :math:`\\phi` is only
approximately a signed distance function between reinitialisations and sphere tracing
would overshoot when :math:`\\|\\nabla\\phi\\| > 1`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..core.grid import Grid, trilinear_sample, trilinear_sample_vector
from ..levelset.operators import gradient_central
from .camera import Camera

__all__ = ["SurfaceReference", "raymarch_levelset"]


@dataclass
class SurfaceReference:
    """Reference rendering of a level set through one camera."""

    hit: Tensor
    """``(H, W)`` boolean - the ray crossed the surface inside the grid."""

    depth: Tensor
    """``(H, W)`` distance along the unit ray to the first crossing, in mm.
    Zero where ``hit`` is ``False``."""

    normal: Tensor
    """``(3, H, W)`` unit surface normal at the crossing."""

    intensity: Tensor
    """``(C, H, W)`` MRI intensity sampled at the crossing."""

    point: Tensor
    """``(3, H, W)`` world-mm crossing position."""

    @property
    def mask_float(self) -> Tensor:
        return self.hit.to(self.depth.dtype)

    def summary(self) -> dict[str, float]:
        n_hit = int(self.hit.sum().item())
        return {
            "hit_fraction": n_hit / float(self.hit.numel()),
            "depth_mean_mm": float(self.depth[self.hit].mean().item()) if n_hit else float("nan"),
            "depth_min_mm": float(self.depth[self.hit].min().item()) if n_hit else float("nan"),
            "depth_max_mm": float(self.depth[self.hit].max().item()) if n_hit else float("nan"),
        }


def _ray_box_intersection(
    origins: Tensor, dirs: Tensor, box_min: Tensor, box_max: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Slab-method AABB intersection.

    ``origins`` broadcasts against ``dirs`` of shape ``(..., 3)``.

    Returns ``(t_near, t_far, valid)``; ``t_near`` is clamped to be non-negative so
    that a camera placed inside the box still marches forward.
    """
    eps = 1e-12
    inv_d = 1.0 / torch.where(dirs.abs() < eps, torch.full_like(dirs, eps), dirs)
    t0 = (box_min - origins) * inv_d
    t1 = (box_max - origins) * inv_d
    t_lo = torch.minimum(t0, t1)
    t_hi = torch.maximum(t0, t1)
    t_near = t_lo.max(dim=-1).values.clamp_min(0.0)
    t_far = t_hi.min(dim=-1).values
    return t_near, t_far, t_far > t_near


@torch.no_grad()
def raymarch_levelset(
    phi: Tensor,
    grid: Grid,
    camera: Camera,
    *,
    image: Tensor | None = None,
    step_mm: float | None = None,
    refine_iters: int = 12,
    grad_phi: Tensor | None = None,
    row_chunk: int = 64,
    spacing_aware: bool = True,
) -> SurfaceReference:
    """Render the reference surface for one camera.

    Parameters
    ----------
    phi:
        ``(nx, ny, nz)`` level set, ``phi > 0`` inside.
    image:
        ``(nx, ny, nz)`` or ``(C, nx, ny, nz)`` intensity volume to sample at the
        crossing.  When ``None`` the intensity output is all ones.
    step_mm:
        Marching step.  Defaults to half the smallest voxel spacing, which cannot
        step over a surface whose features are resolved by the grid.
    refine_iters:
        Bisection steps after bracketing.  12 iterations shrink the bracket by
        ``2^-12``, i.e. well below any spacing of practical interest.
    row_chunk:
        Image rows processed at once; bounds peak memory at
        ``row_chunk * W * n_steps`` samples.

    Returns
    -------
    :class:`SurfaceReference`
    """
    dev, dt = phi.device, phi.dtype
    h, w = int(camera.height), int(camera.width)
    if step_mm is None:
        step_mm = 0.5 * grid.h_min

    if image is None:
        img_vol = torch.ones((1,) + tuple(phi.shape), device=dev, dtype=dt)
    elif image.dim() == 3:
        img_vol = image.unsqueeze(0)
    else:
        img_vol = image
    n_ch = int(img_vol.shape[0])

    if grad_phi is None:
        grad_phi = gradient_central(phi, grid.spacing, spacing_aware=spacing_aware)

    box_min = grid.origin_tensor(dev, dt)
    box_max = box_min + torch.tensor(grid.extent_mm, device=dev, dtype=dt)

    origins_all, dirs_all = camera.rays()
    per_pixel_origin = origins_all.shape[0] != 1

    # device-ok: reduced to a Python float on the same line; never meets another tensor.
    diag = float(torch.tensor(grid.extent_mm).norm().item())
    n_steps = max(2, int(math.ceil(diag / float(step_mm))))

    hit = torch.zeros((h, w), device=dev, dtype=torch.bool)
    depth = torch.zeros((h, w), device=dev, dtype=dt)
    point = torch.zeros((3, h, w), device=dev, dtype=dt)

    s_grid = (torch.arange(n_steps, device=dev, dtype=dt) + 0.5) / n_steps  # (S,)

    for r0 in range(0, h, row_chunk):
        r1 = min(r0 + row_chunk, h)
        d = dirs_all[r0:r1]  # (R, W, 3)
        o = origins_all[r0:r1] if per_pixel_origin else origins_all.expand(r1 - r0, w, 3)

        t_near, t_far, valid = _ray_box_intersection(o, d, box_min, box_max)
        span = (t_far - t_near).clamp_min(0.0)  # (R, W)

        # Sample phi at n_steps points along the in-box segment of every ray.
        t_samples = t_near.unsqueeze(-1) + span.unsqueeze(-1) * s_grid  # (R, W, S)
        pts = o.unsqueeze(-2) + t_samples.unsqueeze(-1) * d.unsqueeze(-2)  # (R, W, S, 3)
        vals = trilinear_sample(phi, grid.world_to_voxel(pts))  # (R, W, S)
        vals = torch.where(valid.unsqueeze(-1), vals, torch.full_like(vals, -1.0))

        inside = vals > 0
        any_inside = inside.any(dim=-1)
        # First sample index that is inside.  `argmax` on a 0/1 tensor is NOT
        # guaranteed to return the first maximum in PyTorch, so use an explicit
        # min-over-indices, which is deterministic.
        step_ids = torch.arange(n_steps, device=dev).expand_as(inside)
        first = torch.where(inside, step_ids, torch.full_like(step_ids, n_steps)).min(dim=-1).values

        # Bracket: [t_lo, t_hi] with phi(t_lo) <= 0 < phi(t_hi).
        # Rays that never went inside get `first == n_steps`; clamp so the gather is
        # in range (their result is discarded via `ok` below).
        first = first.clamp(max=n_steps - 1)
        idx_hi = first
        idx_lo = (first - 1).clamp_min(0)
        t_hi_b = torch.gather(t_samples, -1, idx_hi.unsqueeze(-1)).squeeze(-1)
        t_lo_b = torch.gather(t_samples, -1, idx_lo.unsqueeze(-1)).squeeze(-1)
        # If the very first sample is already inside, extend the bracket backwards.
        t_lo_b = torch.where(first == 0, t_near, t_lo_b)

        ok = any_inside & valid
        for _ in range(int(refine_iters)):
            t_mid = 0.5 * (t_lo_b + t_hi_b)
            p_mid = o + t_mid.unsqueeze(-1) * d
            v_mid = trilinear_sample(phi, grid.world_to_voxel(p_mid))
            go_up = v_mid > 0  # inside -> move the upper bound down
            t_hi_b = torch.where(go_up, t_mid, t_hi_b)
            t_lo_b = torch.where(go_up, t_lo_b, t_mid)

        t_hit = 0.5 * (t_lo_b + t_hi_b)
        p_hit = o + t_hit.unsqueeze(-1) * d

        hit[r0:r1] = ok
        depth[r0:r1] = torch.where(ok, t_hit, torch.zeros_like(t_hit))
        point[:, r0:r1] = torch.where(
            ok.unsqueeze(-1), p_hit, torch.zeros_like(p_hit)
        ).permute(2, 0, 1)

    # Normals and intensity at the crossings.
    flat_pts = point.permute(1, 2, 0).reshape(-1, 3)
    vox = grid.world_to_voxel(flat_pts)
    g = trilinear_sample_vector(grad_phi, vox)
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    normal = g.reshape(h, w, 3).permute(2, 0, 1) * hit.unsqueeze(0)

    inten = trilinear_sample_vector(img_vol, vox).reshape(h, w, n_ch).permute(2, 0, 1)
    inten = inten * hit.unsqueeze(0)

    return SurfaceReference(
        hit=hit, depth=depth, normal=normal, intensity=inten, point=point
    )
