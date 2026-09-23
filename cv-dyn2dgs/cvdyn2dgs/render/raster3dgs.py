"""Thin 3D Gaussian splatting baseline (EWA / affine projection).

This is the baseline that isolates *the primitive itself* (RQ6, proposal §8.2).
It places a 3-D Gaussian ellipsoid at every surfel anchor, with the same two
tangential scales and a small third scale along the normal:

.. math::
    \\Sigma = R \\,\\mathrm{diag}(s_1^2, s_2^2, (\\rho\\min(s_1,s_2))^2)\\, R^\\top,
    \\qquad R = [\\,e_1\\ e_2\\ n\\,],

so the only differences from :func:`cvdyn2dgs.render.raster2dgs.render_2dgs` are
the two the theory singles out:

1. **A normal-direction thickness exists** (:math:`\\rho > 0`).  Theory §1 calls
   this an unnecessary degree of freedom for a thin sheet and predicts surface
   blur; with :math:`\\rho \\to 0` the ellipsoid degenerates and the projected
   conic becomes ill-conditioned, which is itself informative.
2. **Projection is affine, not perspective-correct.**  The splat is approximated
   by a screen-space conic obtained from the Jacobian of the projection at the
   anchor, and its depth is *constant across the footprint*.  Prop. 8.4 predicts
   an error growing with the relative depth variation :math:`\\Delta z / z` over the
   footprint - steeply inclined and large splats suffer most.

Because the depth of a splat does not vary across its footprint, the rendered
depth map is piecewise constant per primitive.  That is not an implementation
shortcut: it is precisely the geometric deficiency the 2-D formulation removes,
and reporting depth RMSE / normal error against this baseline is how RQ6 is
answered.

Everything else - tiling, depth-sorted compositing, culling thresholds, output
buffers - is kept identical to the 2DGS path so the comparison isolates the
primitive rather than the engineering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..core.config import RenderConfig
from ..surfel.model import SurfelSet2D
from .camera import Camera
from .raster2dgs import RenderOutput, _tile_pixel_index

__all__ = ["render_thin_3dgs", "Thin3DGSConfig"]


@dataclass
class Thin3DGSConfig:
    """Extra knobs specific to the thin-ellipsoid baseline."""

    thickness_ratio: float = 0.1
    """:math:`\\rho` - normal scale as a fraction of the smaller tangential scale.

    ``0.1`` makes the ellipsoid genuinely thin while keeping :math:`\\Sigma`
    numerically invertible.  The ablation sweeps it to show the trade-off:
    thinner is geometrically better but worse conditioned.
    """

    low_pass_px2: float = 0.3
    """Screen-space dilation added to the projected conic.

    Standard practice in 3DGS: without it, splats smaller than a pixel alias
    badly.  Note this is an *additional* blur that the 2DGS path does not need.
    """


def _project_covariance(
    anchor: Tensor,
    e1: Tensor,
    e2: Tensor,
    normal: Tensor,
    scale: Tensor,
    camera: Camera,
    cfg3d: Thin3DGSConfig,
    scale_normal: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the ``(N, 2, 2)`` screen-space covariance and ``(N,)`` camera depth.

    ``scale_normal`` overrides the third (normal-direction) scale.  The thin-ellipsoid
    baseline ties it to the tangential scales via ``cfg3d.thickness_ratio``; a
    free-geometry 3DGS needs it to be an independent parameter, and
    :mod:`cvdyn2dgs.surfel.free3dgs` supplies it.
    """
    rot = torch.stack((e1, e2, normal), dim=-1)  # (N, 3, 3), columns
    if scale_normal is None:
        s_n = cfg3d.thickness_ratio * scale.min(dim=1).values
    else:
        s_n = scale_normal
    diag = torch.stack((scale[:, 0], scale[:, 1], s_n), dim=-1) ** 2  # (N, 3)
    sigma = rot @ torch.diag_embed(diag) @ rot.transpose(-1, -2)  # (N, 3, 3)

    w_rot = camera.R  # world -> camera
    sigma_cam = w_rot @ sigma @ w_rot.transpose(0, 1)

    cam = anchor @ w_rot.transpose(0, 1) + camera.t
    z = cam[:, 2]

    n_pts = anchor.shape[0]
    jac = torch.zeros((n_pts, 2, 3), device=anchor.device, dtype=anchor.dtype)
    if camera.orthographic:
        inv_px = 1.0 / camera.pixel_size_mm
        jac[:, 0, 0] = inv_px
        jac[:, 1, 1] = inv_px
    else:
        zz = z.clamp_min(1e-9)
        jac[:, 0, 0] = camera.fx / zz
        jac[:, 0, 2] = -camera.fx * cam[:, 0] / (zz * zz)
        jac[:, 1, 1] = camera.fy / zz
        jac[:, 1, 2] = -camera.fy * cam[:, 1] / (zz * zz)

    cov2d = jac @ sigma_cam @ jac.transpose(-1, -2)  # (N, 2, 2)
    eye2 = torch.eye(2, device=anchor.device, dtype=anchor.dtype)
    cov2d = cov2d + cfg3d.low_pass_px2 * eye2
    return cov2d, z


def render_thin_3dgs(
    surfels: SurfelSet2D,
    camera: Camera,
    cfg: RenderConfig | None = None,
    cfg3d: Thin3DGSConfig | None = None,
    *,
    compute_aux: bool = True,
    max_pairs: int = 60_000_000,
    scale_normal: Tensor | None = None,
) -> RenderOutput:
    """Rasterise the surfel set as thin 3-D Gaussians with affine projection.

    ``surfels`` is duck-typed: anything exposing ``anchor, e1, e2, normal, scale,
    amplitude, opacity, n, channels, device, dtype`` works.  That is how
    :class:`cvdyn2dgs.surfel.free3dgs.FreeGaussians3D` reuses this rasteriser without
    pretending to be a surface-pinned surfel set.

    ``scale_normal`` supplies an independent third scale; see
    :func:`_project_covariance`.
    """
    cfg = cfg or RenderConfig()
    cfg3d = cfg3d or Thin3DGSConfig()
    dev, dt = surfels.device, surfels.dtype
    tile = int(cfg.tile)
    h, w = int(camera.height), int(camera.width)
    n_ty = (h + tile - 1) // tile
    n_tx = (w + tile - 1) // tile
    hp, wp = n_ty * tile, n_tx * tile
    n_tiles = n_ty * n_tx
    n_ch = surfels.channels

    def _empty() -> RenderOutput:
        return RenderOutput(
            color=torch.full((n_ch, h, w), float(cfg.background), device=dev, dtype=dt),
            alpha=torch.zeros((h, w), device=dev, dtype=dt),
            depth=torch.zeros((h, w), device=dev, dtype=dt),
            normal=torch.zeros((3, h, w), device=dev, dtype=dt),
            distortion=torch.zeros((h, w), device=dev, dtype=dt),
            n_contributing=torch.zeros((h, w), device=dev, dtype=torch.int32),
            stats={"visible_surfels": 0.0, "pairs": 0.0},
        )

    if surfels.n == 0:
        return _empty()

    cov2d, z_cam = _project_covariance(
        surfels.anchor,
        surfels.e1,
        surfels.e2,
        surfels.normal,
        surfels.scale,
        camera,
        cfg3d,
        scale_normal=scale_normal,
    )
    uv, _ = camera.project(surfels.anchor)

    # Inverse of the symmetric 2x2 conic.
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = (a * c - b * b).clamp_min(1e-12)
    conic = torch.stack((c / det, -b / det, a / det), dim=-1)  # (N, 3) = [A, B, C]

    with torch.no_grad():
        mid = 0.5 * (a + c)
        disc = torch.sqrt((0.5 * (a - c)) ** 2 + b * b)
        lam_max = (mid + disc).clamp_min(1e-12)
        r_pix = float(cfg.gaussian_cutoff) * torch.sqrt(lam_max) + 1.0

        visible = (z_cam > float(cfg.near_mm)) & (det > 1e-10)
        if not bool(visible.any()):
            return _empty()
        vis_idx = torch.nonzero(visible, as_tuple=False).squeeze(1)

        ux, uy = uv[vis_idx, 0], uv[vis_idx, 1]
        rp = r_pix[vis_idx]
        x0, x1 = ux - rp, ux + rp
        y0, y1 = uy - rp, uy + rp
        on_screen = (x1 >= 0) & (y1 >= 0) & (x0 <= wp - 1) & (y0 <= hp - 1)
        if not bool(on_screen.any()):
            return _empty()
        keep = torch.nonzero(on_screen, as_tuple=False).squeeze(1)
        vis_idx = vis_idx[keep]
        x0, x1, y0, y1 = x0[keep], x1[keep], y0[keep], y1[keep]
        depth_vis = z_cam[vis_idx]

        tx0 = (x0 / tile).floor().clamp(0, n_tx - 1).long()
        tx1 = (x1 / tile).floor().clamp(0, n_tx - 1).long()
        ty0 = (y0 / tile).floor().clamp(0, n_ty - 1).long()
        ty1 = (y1 / tile).floor().clamp(0, n_ty - 1).long()
        wt, ht = tx1 - tx0 + 1, ty1 - ty0 + 1
        counts = wt * ht
        total = int(counts.sum().item())
        if total == 0:
            return _empty()
        if total > max_pairs:
            raise RuntimeError(f"{total} pairs exceeds max_pairs={max_pairs}")

        k_vis = int(vis_idx.numel())
        pair_surfel = torch.repeat_interleave(torch.arange(k_vis, device=dev), counts)
        cum = torch.cumsum(counts, 0) - counts
        local = torch.arange(total, device=dev) - torch.repeat_interleave(cum, counts)
        w_of = wt[pair_surfel]
        dy = torch.div(local, w_of, rounding_mode="floor")
        dx = local - dy * w_of
        tile_id = (ty0[pair_surfel] + dy) * n_tx + (tx0[pair_surfel] + dx)

        order = torch.argsort(depth_vis[pair_surfel])
        order = order[torch.argsort(tile_id[order], stable=True)]
        tid_s, sid_s = tile_id[order], pair_surfel[order]
        per_tile = torch.bincount(tid_s, minlength=n_tiles)
        starts = torch.cumsum(per_tile, 0) - per_tile
        rank = torch.arange(total, device=dev) - starts[tid_s]
        m_cap = int(cfg.max_per_tile)
        sel = rank < m_cap
        dense = torch.full((n_tiles, m_cap), -1, device=dev, dtype=torch.long)
        dense[tid_s[sel], rank[sel]] = sid_s[sel]
        truncated = int((~sel).sum().item())
        active_tiles = torch.nonzero(per_tile > 0, as_tuple=False).squeeze(1)
        tile_pix = _tile_pixel_index(n_ty, n_tx, tile, wp, dev)

        py = torch.arange(hp, device=dev, dtype=dt) + 0.5
        px = torch.arange(wp, device=dev, dtype=dt) + 0.5
        gy, gx = torch.meshgrid(py, px, indexing="ij")
        pix_xy = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)  # (Hp*Wp, 2)

    uv_v = uv[vis_idx]
    conic_v = conic[vis_idx]
    opa_v = surfels.opacity[vis_idx]
    amp_v = surfels.amplitude[vis_idx]
    normal_v = surfels.normal[vis_idx]
    depth_v = z_cam[vis_idx]

    out_pix: list[Tensor] = []
    out_color: list[Tensor] = []
    out_alpha: list[Tensor] = []
    out_depth: list[Tensor] = []
    out_normal: list[Tensor] = []
    out_dist: list[Tensor] = []
    out_count: list[Tensor] = []

    cutoff_sq = float(cfg.gaussian_cutoff) ** 2
    chunk = max(1, int(cfg.tile_chunk))

    for lo in range(0, int(active_tiles.numel()), chunk):
        tsel = active_tiles[lo : lo + chunk]
        idx = dense[tsel]
        valid = idx >= 0
        if not bool(valid.any()):
            continue
        sid = idx.clamp_min(0)
        pix = tile_pix[tsel]  # (B, P)

        xy = pix_xy[pix]  # (B, P, 2)
        ctr = uv_v[sid]  # (B, M, 2)
        d = xy.unsqueeze(1) - ctr.unsqueeze(2)  # (B, M, P, 2)
        cn = conic_v[sid]  # (B, M, 3)
        power = (
            cn[..., 0:1] * d[..., 0] ** 2
            + 2.0 * cn[..., 1:2] * d[..., 0] * d[..., 1]
            + cn[..., 2:3] * d[..., 1] ** 2
        )  # (B, M, P)

        ok = valid.unsqueeze(-1) & (power <= cutoff_sq)
        alpha = opa_v[sid].unsqueeze(-1) * torch.exp(-0.5 * power)
        alpha = torch.where(ok, alpha, torch.zeros_like(alpha))
        alpha = torch.where(alpha >= float(cfg.min_alpha), alpha, torch.zeros_like(alpha))
        alpha = alpha.clamp(max=1.0 - 1e-6)

        trans = torch.cumprod(1.0 - alpha, dim=1)
        trans_excl = torch.cat((torch.ones_like(trans[:, :1]), trans[:, :-1]), dim=1)
        weights = trans_excl * alpha
        alpha_sum = weights.sum(dim=1)

        out_color.append(torch.einsum("bmp,bmc->bpc", weights, amp_v[sid]))
        out_alpha.append(alpha_sum)
        out_pix.append(pix.reshape(-1))

        if compute_aux:
            # Depth is constant across a splat's footprint - the key deficiency.
            tau = depth_v[sid].unsqueeze(-1).expand_as(alpha)
            tau = torch.where(ok, tau, torch.zeros_like(tau))
            out_depth.append((weights * tau).sum(dim=1))
            out_normal.append(torch.einsum("bmp,bmk->bpk", weights, normal_v[sid]))
            mean = (weights * tau).sum(dim=1) / alpha_sum.clamp_min(1e-8)
            out_dist.append((weights * (tau - mean.unsqueeze(1)) ** 2).sum(dim=1))
            out_count.append((weights > float(cfg.min_alpha)).sum(dim=1).to(torch.int32))

    if not out_pix:
        return _empty()

    pix_all = torch.cat(out_pix, dim=0)
    n_pix_p = hp * wp

    def scatter(vals: list[Tensor], trailing: tuple[int, ...], dtype=None) -> Tensor:
        flat = torch.cat([v.reshape(-1, *trailing) for v in vals], dim=0)
        base = torch.zeros((n_pix_p, *trailing), device=dev, dtype=dtype or flat.dtype)
        return base.index_add(0, pix_all, flat)

    color_img = scatter(out_color, (n_ch,)).reshape(hp, wp, n_ch).permute(2, 0, 1)
    alpha_img = scatter(out_alpha, ()).reshape(hp, wp)
    if compute_aux:
        depth_img = scatter(out_depth, ()).reshape(hp, wp)
        normal_img = scatter(out_normal, (3,)).reshape(hp, wp, 3).permute(2, 0, 1)
        dist_img = scatter(out_dist, ()).reshape(hp, wp)
        count_img = scatter(out_count, (), dtype=torch.int32).reshape(hp, wp)
    else:
        depth_img = torch.zeros((hp, wp), device=dev, dtype=dt)
        normal_img = torch.zeros((3, hp, wp), device=dev, dtype=dt)
        dist_img = torch.zeros((hp, wp), device=dev, dtype=dt)
        count_img = torch.zeros((hp, wp), device=dev, dtype=torch.int32)

    bg = float(cfg.background)
    if bg != 0.0:
        color_img = color_img + bg * (1.0 - alpha_img).clamp_min(0.0).unsqueeze(0)

    return RenderOutput(
        color=color_img[:, :h, :w],
        alpha=alpha_img[:h, :w],
        depth=depth_img[:h, :w],
        normal=normal_img[:, :h, :w],
        distortion=dist_img[:h, :w],
        n_contributing=count_img[:h, :w],
        stats={
            "visible_surfels": float(vis_idx.numel()),
            "pairs": float(total),
            "pairs_truncated": float(truncated),
            "thickness_ratio": cfg3d.thickness_ratio,
        },
    )
