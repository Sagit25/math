"""Loss terms for canonical 2DGS fitting, proposal Eq. (33).

.. math::
    \\mathcal{L} = \\mathcal{L}_{\\mathrm{app}}
                 + \\lambda_m \\mathcal{L}_{\\mathrm{mask}}
                 + \\lambda_n \\mathcal{L}_{\\mathrm{normal}}
                 + \\lambda_d \\mathcal{L}_{\\mathrm{dist}}

An important structural point: **none of these losses can move the surface.**  The
anchors, tangent frames and normals are buffers driven by the Chan-Vese level set
(see :mod:`cvdyn2dgs.surfel.model`), so gradients only reach amplitude, opacity and
scale.  Proposal §6.6 states this explicitly - "the geometry is pinned to the
Chan-Vese surface, so this loss does not arbitrarily move the boundary" - and the
implementation enforces it rather than relying on a learning-rate of zero.

Consequently the geometry terms play a *different* role here than in ordinary
2DGS: they cannot improve the surface, only the way a fixed surface is covered.
``L_mask`` closes holes and trims overhang, ``L_normal`` discourages disks from
tilting away from the level-set normal, and ``L_dist`` discourages several disks
from contributing to one ray at different depths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .core.config import LossConfig
from .render.camera import Camera
from .render.raster2dgs import RenderOutput

__all__ = [
    "LossTerms",
    "appearance_loss",
    "mask_loss",
    "normal_from_depth",
    "normal_consistency_loss",
    "distortion_loss",
    "total_loss",
]


@dataclass
class LossTerms:
    """Individual terms plus the weighted total."""

    appearance: Tensor
    mask: Tensor
    normal: Tensor
    distortion: Tensor
    total: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "loss/appearance": float(self.appearance.detach().item()),
            "loss/mask": float(self.mask.detach().item()),
            "loss/normal": float(self.normal.detach().item()),
            "loss/distortion": float(self.distortion.detach().item()),
            "loss/total": float(self.total.detach().item()),
        }


def _masked_mean(x: Tensor, weight: Tensor | None) -> Tensor:
    if weight is None:
        return x.mean()
    denom = weight.sum().clamp_min(1e-8)
    return (x * weight).sum() / denom


def appearance_loss(
    rendered: Tensor,
    target: Tensor,
    *,
    kind: str = "l1",
    huber_delta: float = 0.05,
    weight: Tensor | None = None,
) -> Tensor:
    """:math:`\\mathcal{L}_{\\mathrm{app}}` between rendered and observed intensity.

    Parameters
    ----------
    rendered, target:
        ``(C, H, W)``.
    weight:
        Optional ``(H, W)`` weighting.  Passing the heart ROI here is what keeps
        the score from being dominated by empty background - the same concern
        proposal §8.3 raises for PSNR/SSIM, where whole-image and ROI numbers are
        reported separately.
    """
    if rendered.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(rendered.shape)} vs {tuple(target.shape)}")
    diff = rendered - target
    if kind == "l1":
        per_px = diff.abs().mean(dim=0)
    elif kind == "l2":
        per_px = (diff * diff).mean(dim=0)
    elif kind == "huber":
        per_px = F.huber_loss(
            rendered, target, reduction="none", delta=float(huber_delta)
        ).mean(dim=0)
    else:
        raise ValueError(f"unknown appearance loss {kind!r}")
    return _masked_mean(per_px, weight)


def mask_loss(
    alpha: Tensor,
    target_mask: Tensor,
    *,
    kind: str = "l2",
    weight: Tensor | None = None,
) -> Tensor:
    """:math:`\\mathcal{L}_{\\mathrm{mask}}` between the rendered alpha and the
    projected Chan-Vese silhouette.

    The target is produced by rendering the *same* :math:`\\Gamma_t` as an opaque
    mesh (or by projecting the mask volume), so this term asks only "do the disks
    cover the surface they sit on", never "is the surface right".
    """
    if alpha.shape != target_mask.shape:
        raise ValueError(f"shape mismatch: {tuple(alpha.shape)} vs {tuple(target_mask.shape)}")
    tgt = target_mask.to(alpha.dtype)
    if kind == "l2":
        per_px = (alpha - tgt) ** 2
    elif kind == "l1":
        per_px = (alpha - tgt).abs()
    elif kind == "bce":
        a = alpha.clamp(1e-6, 1.0 - 1e-6)
        per_px = -(tgt * torch.log(a) + (1.0 - tgt) * torch.log(1.0 - a))
    else:
        raise ValueError(f"unknown mask loss {kind!r}")
    return _masked_mean(per_px, weight)


def normal_from_depth(depth: Tensor, camera: Camera, *, eps: float = 1e-8) -> Tensor:
    """Estimate per-pixel normals from a rendered depth map.

    The depth produced by the rasterisers is :math:`\\tau`, a distance along the
    **unit** ray (Eq. 8.2), so back-projection is simply
    :math:`X(q) = o_c + \\tau(q)\\, d_q`.  Normals follow from the cross product of
    the screen-space derivatives of :math:`X`, computed with central differences
    and replicate padding at the border.

    Returns ``(3, H, W)``, unit-norm, oriented towards the camera.
    """
    h, w = depth.shape
    origins, dirs = camera.rays(height=h, width=w)
    pts = origins + depth.unsqueeze(-1) * dirs  # (H, W, 3)

    p = pts.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    pad = F.pad(p, (1, 1, 1, 1), mode="replicate")
    dx = pad[:, :, 1:-1, 2:] - pad[:, :, 1:-1, :-2]
    dy = pad[:, :, 2:, 1:-1] - pad[:, :, :-2, 1:-1]

    n = torch.cross(dx.squeeze(0), dy.squeeze(0), dim=0)  # (3,H,W)
    n = n / n.norm(dim=0, keepdim=True).clamp_min(eps)

    # Orient towards the camera: n . (-d) > 0
    view = -dirs.permute(2, 0, 1)
    flip = ((n * view).sum(dim=0, keepdim=True) < 0).to(n.dtype)
    return n * (1.0 - 2.0 * flip)


def normal_consistency_loss(
    out: RenderOutput,
    camera: Camera,
    *,
    reference_normal: Tensor | None = None,
    alpha_floor: float = 0.5,
) -> Tensor:
    """:math:`\\mathcal{L}_{\\mathrm{normal}}` - agreement between the surfel normals
    and the geometry implied by the rendered depth.

    Proposal §6.6 specifies the comparison as "the normal computed from the
    rendered depth versus Eq. (20)".  Here Eq. (20)/(7.3) enters through
    ``out.normal``, the alpha-weighted mean of the surfel normals, and the
    depth-derived normal comes from :func:`normal_from_depth`.  Supplying
    ``reference_normal`` replaces the rendered normal with an external one (e.g.
    the analytic level-set normal rasterised through the same camera).

    Only pixels with ``alpha >= alpha_floor`` contribute: depth is meaningless
    where nothing was hit, and its finite differences across a silhouette edge are
    dominated by the depth discontinuity rather than by surface orientation.
    """
    n_depth = normal_from_depth(out.depth, camera)
    n_surf = out.normalized_normal() if reference_normal is None else reference_normal
    cos = (n_depth * n_surf).sum(dim=0)
    weight = (out.alpha >= float(alpha_floor)).to(out.alpha.dtype)
    return _masked_mean(1.0 - cos, weight)


def distortion_loss(out: RenderOutput, *, alpha_floor: float = 0.1) -> Tensor:
    """:math:`\\mathcal{L}_{\\mathrm{dist}}` - the depth-distortion map, averaged.

    See :func:`cvdyn2dgs.render.raster2dgs._distortion` for the exact form and for
    why a weighted variance is used in place of the pairwise formulation.
    """
    weight = (out.alpha >= float(alpha_floor)).to(out.alpha.dtype)
    return _masked_mean(out.distortion, weight)


def total_loss(
    out: RenderOutput,
    target_image: Tensor,
    cfg: LossConfig,
    *,
    camera: Camera | None = None,
    target_mask: Tensor | None = None,
    roi_weight: Tensor | None = None,
    reference_normal: Tensor | None = None,
) -> LossTerms:
    """Assemble Eq. (33).

    Terms whose weight is zero are skipped entirely - no wasted rendering of
    depth-derived normals for the ``v1-minimal`` preset, which sets all three
    geometry weights to zero.
    """
    dev, dt = out.color.device, out.color.dtype
    zero = torch.zeros((), device=dev, dtype=dt)

    l_app = appearance_loss(
        out.color,
        target_image,
        kind=cfg.appearance,
        huber_delta=cfg.huber_delta,
        weight=roi_weight,
    )

    l_mask = zero
    if cfg.lambda_mask > 0.0:
        if target_mask is None:
            raise ValueError("lambda_mask > 0 requires target_mask")
        l_mask = mask_loss(out.alpha, target_mask)

    l_normal = zero
    if cfg.lambda_normal > 0.0:
        if camera is None:
            raise ValueError("lambda_normal > 0 requires camera")
        l_normal = normal_consistency_loss(out, camera, reference_normal=reference_normal)

    l_dist = zero
    if cfg.lambda_dist > 0.0:
        l_dist = distortion_loss(out)

    total = (
        l_app
        + cfg.lambda_mask * l_mask
        + cfg.lambda_normal * l_normal
        + cfg.lambda_dist * l_dist
    )
    return LossTerms(
        appearance=l_app, mask=l_mask, normal=l_normal, distortion=l_dist, total=total
    )
