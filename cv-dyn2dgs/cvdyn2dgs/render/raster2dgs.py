"""Perspective-correct 2D Gaussian surfel rasterisation.

Implements Eq. (8.1)-(8.6) of the theory / Eq. (28)-(32) of the proposal.

For pixel ray :math:`r_q(\\tau) = o_c + \\tau d_q`, the surfel plane through
:math:`p^t_i` with normal :math:`n^t_i` is intersected **exactly**:

.. math::
    \\tau^t_i(q) = \\frac{(n^t_i)^\\top (p^t_i - o_c)}{(n^t_i)^\\top d_q},
    \\quad x^t_i(q) = o_c + \\tau^t_i(q)\\, d_q,
    \\quad u^t_i(q) = S_i^{-1}(E^t_i)^\\top\\!\\left(x^t_i(q) - p^t_i\\right),
    \\quad \\alpha^t_i(q) = o^t_i \\exp\\!\\left(-\\tfrac12\\|u^t_i(q)\\|^2\\right),

and the surfels hitting a pixel are composited front-to-back,

.. math::
    \\hat C_t(q) = \\sum_i \\Bigl(\\prod_{j<i}(1-\\alpha^t_j(q))\\Bigr)\\alpha^t_i(q)\\, c^t_i .

Prop. 8.4 is the reason for doing it this way: for a *planar* disk this is exact,
whereas affine screen-space projection leaves an error that grows with the
relative depth variation :math:`\\Delta z / z` across the footprint.  The only
geometric error left is the planar-disk approximation of the curved surface,
:math:`O(\\kappa s^2)` from Prop. 8.3.

Two rasterisers are provided:

:func:`render_2dgs`
    Tile-binned, vectorised, differentiable - the one used everywhere.
:func:`render_2dgs_reference`
    Brute force over all surfels with **exact per-pixel depth ordering**.  Far too
    slow for experiments, but it is the ground truth the tiled renderer is tested
    against on small inputs, which is how the two approximations of the fast path
    (per-tile anchor-depth ordering and ``max_per_tile`` truncation) are quantified
    rather than assumed harmless.

Implementation notes
--------------------
* The intersection point :math:`x^t_i(q)` is never materialised.  Expanding
  Eq. (8.4) as
  ``u_1 = (e_1·o_c + tau (e_1·d) - e_1·p) / s_1``
  keeps every intermediate at ``(B, M, P)`` instead of ``(B, M, P, 3)``.
* Culling follows Prop. 8.1: grazing-angle intersections with
  :math:`|(n^t_i)^\\top d_q| < c_0` are dropped because their depth has condition
  number :math:`1/c_0`, and :math:`\\tau \\le 0` is dropped as unphysical.
* Compositing uses an exclusive ``cumprod`` of :math:`1-\\alpha`.  Since
  ``opacity = sigmoid(...) < 1`` strictly and the Gaussian factor is positive,
  :math:`1-\\alpha > 0` always, so the product never hits zero and the backward
  pass is well behaved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..core.config import RenderConfig
from ..surfel.model import SurfelSet2D
from .camera import Camera

__all__ = [
    "RenderOutput",
    "WeightMatrix",
    "render_2dgs",
    "render_2dgs_reference",
    "render_weights",
]


@dataclass
class RenderOutput:
    """Rasterisation result and the auxiliary buffers the losses need."""

    color: Tensor
    """``(C, H, W)`` composited intensity, Eq. (8.6)."""

    alpha: Tensor
    """``(H, W)`` accumulated opacity - the rendered silhouette."""

    depth: Tensor
    """``(H, W)`` expected ray depth :math:`\\sum_i w_i \\tau_i` in mm."""

    normal: Tensor
    """``(3, H, W)`` expected surfel normal :math:`\\sum_i w_i n_i` (not renormalised)."""

    distortion: Tensor
    """``(H, W)`` depth-distortion map; see :func:`_distortion` for the exact form."""

    n_contributing: Tensor
    """``(H, W)`` number of surfels with non-negligible weight."""

    stats: dict[str, float] = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return int(self.color.shape[0])

    def normalized_normal(self, eps: float = 1e-8) -> Tensor:
        return self.normal / self.normal.norm(dim=0, keepdim=True).clamp_min(eps)

    def mean_depth(self, eps: float = 1e-8) -> Tensor:
        """Alpha-normalised depth, i.e. the expected depth of the *hit* surface."""
        return self.depth / self.alpha.clamp_min(eps)


def _tile_pixel_index(n_tiles_y: int, n_tiles_x: int, tile: int, width_p: int, device) -> Tensor:
    """``(n_tiles, tile*tile)`` flat pixel indices of the padded image."""
    py = torch.arange(tile, device=device)
    px = torch.arange(tile, device=device)
    gy, gx = torch.meshgrid(py, px, indexing="ij")
    off = (gy * width_p + gx).reshape(-1)  # (P,)

    ty = torch.arange(n_tiles_y, device=device)
    tx = torch.arange(n_tiles_x, device=device)
    origin = (ty.view(-1, 1) * tile * width_p) + (tx.view(1, -1) * tile)  # (nty, ntx)
    return origin.reshape(-1, 1) + off.view(1, -1)


@dataclass
class _TileGeom:
    """Padded tile layout for an image."""

    tile: int
    n_ty: int
    n_tx: int
    hp: int
    wp: int

    @property
    def n_tiles(self) -> int:
        return self.n_ty * self.n_tx

    @property
    def p_per_tile(self) -> int:
        return self.tile * self.tile

    @classmethod
    def build(cls, height: int, width: int, tile: int) -> "_TileGeom":
        n_ty = (int(height) + tile - 1) // tile
        n_tx = (int(width) + tile - 1) // tile
        return cls(tile=tile, n_ty=n_ty, n_tx=n_tx, hp=n_ty * tile, wp=n_tx * tile)


@dataclass
class _BinResult:
    """Outcome of assigning surfels to tiles."""

    vis_idx: Tensor
    """``(K,)`` indices of surfels that survived culling."""

    dense: Tensor
    """``(n_tiles, max_per_tile)`` depth-sorted indices *into the visible subset*,
    padded with ``-1``."""

    active_tiles: Tensor
    tile_pix: Tensor
    total_pairs: int
    truncated_pairs: int

    @property
    def empty(self) -> bool:
        return int(self.vis_idx.numel()) == 0 or self.total_pairs == 0


@torch.no_grad()
def _bin_surfels(
    anchor: Tensor,
    radius_mm: Tensor,
    camera: Camera,
    cfg: RenderConfig,
    geom: _TileGeom,
    *,
    max_pairs: int,
) -> _BinResult | None:
    """Cull surfels, bound them in screen space and bin them into depth-sorted tiles.

    The screen bound is deliberately conservative: a sphere of radius
    ``radius_mm`` around the anchor contains the disk, and it is evaluated at the
    nearest depth the disk can reach.  A tight bound on the projected ellipse
    would generate fewer candidate pairs, but an under-estimated bound silently
    drops contributions, which is a correctness bug rather than a performance one.
    """
    dev = anchor.device
    uv_a, z_a = camera.project(anchor)
    visible = z_a > (float(cfg.near_mm) + radius_mm)
    if not bool(visible.any()):
        return None
    vis_idx = torch.nonzero(visible, as_tuple=False).squeeze(1)

    z_eff = (z_a[vis_idx] - radius_mm[vis_idx]).clamp_min(float(cfg.near_mm))
    r_pix = radius_mm[vis_idx] * camera.pixel_scale_at(z_eff) + 1.0
    ux, uy = uv_a[vis_idx, 0], uv_a[vis_idx, 1]
    x0, x1 = ux - r_pix, ux + r_pix
    y0, y1 = uy - r_pix, uy + r_pix

    on_screen = (x1 >= 0) & (y1 >= 0) & (x0 <= geom.wp - 1) & (y0 <= geom.hp - 1)
    if not bool(on_screen.any()):
        return None
    keep = torch.nonzero(on_screen, as_tuple=False).squeeze(1)
    vis_idx = vis_idx[keep]
    x0, x1, y0, y1 = x0[keep], x1[keep], y0[keep], y1[keep]
    depth_vis = z_a[vis_idx]

    t = geom.tile
    tx0 = (x0 / t).floor().clamp(0, geom.n_tx - 1).long()
    tx1 = (x1 / t).floor().clamp(0, geom.n_tx - 1).long()
    ty0 = (y0 / t).floor().clamp(0, geom.n_ty - 1).long()
    ty1 = (y1 / t).floor().clamp(0, geom.n_ty - 1).long()

    wt, ht = tx1 - tx0 + 1, ty1 - ty0 + 1
    counts = wt * ht
    total = int(counts.sum().item())
    if total == 0:
        return None
    if total > max_pairs:
        raise RuntimeError(
            f"{total} (surfel, tile) pairs exceeds max_pairs={max_pairs}. "
            "The camera is likely inside the model or the surfel scales are too large."
        )

    k_vis = int(vis_idx.numel())
    pair_surfel = torch.repeat_interleave(torch.arange(k_vis, device=dev), counts)
    cum = torch.cumsum(counts, 0) - counts
    local = torch.arange(total, device=dev) - torch.repeat_interleave(cum, counts)
    w_of = wt[pair_surfel]
    dy = torch.div(local, w_of, rounding_mode="floor")
    dx = local - dy * w_of
    tile_id = (ty0[pair_surfel] + dy) * geom.n_tx + (tx0[pair_surfel] + dx)

    # Group by tile, depth-ascending within each tile.
    order = torch.argsort(depth_vis[pair_surfel])
    order = order[torch.argsort(tile_id[order], stable=True)]
    tid_s, sid_s = tile_id[order], pair_surfel[order]

    per_tile = torch.bincount(tid_s, minlength=geom.n_tiles)
    starts = torch.cumsum(per_tile, 0) - per_tile
    rank = torch.arange(total, device=dev) - starts[tid_s]
    m_cap = int(cfg.max_per_tile)
    sel = rank < m_cap

    dense = torch.full((geom.n_tiles, m_cap), -1, device=dev, dtype=torch.long)
    dense[tid_s[sel], rank[sel]] = sid_s[sel]

    return _BinResult(
        vis_idx=vis_idx,
        dense=dense,
        active_tiles=torch.nonzero(per_tile > 0, as_tuple=False).squeeze(1),
        tile_pix=_tile_pixel_index(geom.n_ty, geom.n_tx, t, geom.wp, dev),
        total_pairs=total,
        truncated_pairs=int((~sel).sum().item()),
    )


def _distortion(weights: Tensor, tau: Tensor, alpha_sum: Tensor, eps: float = 1e-8) -> Tensor:
    """Depth-distortion surrogate.

    The 2DGS paper penalises :math:`\\sum_i\\sum_j w_i w_j |\\tau_i - \\tau_j|`, an
    ``O(M^2)`` quantity.  Here the ``O(M)`` weighted variance about the expected
    depth is used instead:

    .. math:: \\sum_i w_i (\\tau_i - \\bar\\tau)^2, \\qquad
              \\bar\\tau = \\frac{\\sum_i w_i \\tau_i}{\\sum_i w_i}.

    Both vanish exactly when all contributions to a ray sit at one depth, which is
    the property the regulariser is there for; the variance form is simply cheaper.
    This substitution is a deliberate deviation from the reference implementation
    and is flagged here rather than buried.
    """
    mean = (weights * tau).sum(dim=1) / alpha_sum.clamp_min(eps)  # (B, P)
    return (weights * (tau - mean.unsqueeze(1)) ** 2).sum(dim=1)


def render_2dgs(
    surfels: SurfelSet2D,
    camera: Camera,
    cfg: RenderConfig | None = None,
    *,
    compute_aux: bool = True,
    max_pairs: int = 60_000_000,
) -> RenderOutput:
    """Rasterise a surfel set through a camera.

    Differentiable with respect to ``surfels.log_scale``, ``surfels.amplitude`` and
    ``surfels.opacity_logit``.  Geometry (anchor / frame / normal) is treated as a
    constant, because it is produced by the Chan-Vese surface rather than fitted
    (see :mod:`cvdyn2dgs.surfel.model`).

    Parameters
    ----------
    max_pairs:
        Safety valve on the number of (surfel, tile) pairs.  Exceeding it means the
        camera is extremely close or the scales are far too large; the call raises
        rather than silently exhausting memory.
    """
    cfg = cfg or RenderConfig()
    dev, dt = surfels.device, surfels.dtype
    h, w = int(camera.height), int(camera.width)
    geom = _TileGeom.build(h, w, int(cfg.tile))
    hp, wp = geom.hp, geom.wp
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

    scale_all = surfels.scale  # (N,2) differentiable
    opacity_all = surfels.opacity  # (N,)
    amp_all = surfels.amplitude  # (N,C)

    radius_mm = float(cfg.gaussian_cutoff) * scale_all.detach().max(dim=1).values
    binned = _bin_surfels(
        surfels.anchor, radius_mm, camera, cfg, geom, max_pairs=max_pairs
    )
    if binned is None or binned.empty:
        return _empty()
    vis_idx = binned.vis_idx
    dense = binned.dense
    active_tiles = binned.active_tiles
    tile_pix = binned.tile_pix
    total = binned.total_pairs
    truncated = binned.truncated_pairs

    # Gather the visible subset once; these views stay differentiable.
    anchor_v = surfels.anchor[vis_idx]
    normal_v = surfels.normal[vis_idx]
    e1_v = surfels.e1[vis_idx]
    e2_v = surfels.e2[vis_idx]
    scale_v = scale_all[vis_idx]
    opa_v = opacity_all[vis_idx]
    amp_v = amp_all[vis_idx]

    origins, dirs = camera.rays(height=hp, width=wp)
    dirs_flat = dirs.reshape(-1, 3)
    per_pixel_origin = origins.shape[0] != 1
    origins_flat = origins.reshape(-1, 3) if per_pixel_origin else origins.reshape(3)

    # Accumulators, gathered per chunk and scattered once at the end.
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
        idx = dense[tsel]  # (B, M)
        valid = idx >= 0
        if not bool(valid.any()):
            continue
        sid = idx.clamp_min(0)

        pix = tile_pix[tsel]  # (B, P)
        d = dirs_flat[pix]  # (B, P, 3)

        p = anchor_v[sid]  # (B, M, 3)
        n = normal_v[sid]
        e1 = e1_v[sid]
        e2 = e2_v[sid]
        s = scale_v[sid]  # (B, M, 2)
        o = opa_v[sid]  # (B, M)
        c = amp_v[sid]  # (B, M, C)

        # (n . d) -> (B, M, P)
        nd = torch.einsum("bmk,bpk->bmp", n, d)

        n_dot_p = (n * p).sum(-1)  # (B, M)
        e1_dot_p = (e1 * p).sum(-1)
        e2_dot_p = (e2 * p).sum(-1)
        e1d = torch.einsum("bmk,bpk->bmp", e1, d)
        e2d = torch.einsum("bmk,bpk->bmp", e2, d)

        if per_pixel_origin:
            o_c = origins_flat[pix]  # (B, P, 3)
            n_dot_o = torch.einsum("bmk,bpk->bmp", n, o_c)
            e1_dot_o = torch.einsum("bmk,bpk->bmp", e1, o_c)
            e2_dot_o = torch.einsum("bmk,bpk->bmp", e2, o_c)
            numer = n_dot_p.unsqueeze(-1) - n_dot_o
        else:
            o_c = origins_flat  # (3,)
            n_dot_o = (n * o_c).sum(-1)  # (B, M)
            e1_dot_o = (e1 * o_c).sum(-1)
            e2_dot_o = (e2 * o_c).sum(-1)
            numer = (n_dot_p - n_dot_o).unsqueeze(-1)
            e1_dot_o = e1_dot_o.unsqueeze(-1)
            e2_dot_o = e2_dot_o.unsqueeze(-1)

        # Prop. 8.1: cull grazing angles before dividing.
        ok = valid.unsqueeze(-1) & (nd.abs() >= float(cfg.cull_cos))
        nd_safe = torch.where(ok, nd, torch.ones_like(nd))
        tau = numer / nd_safe  # Eq. (8.2)
        ok = ok & (tau > float(cfg.near_mm))

        # Eq. (8.4), expanded to avoid materialising the 3-vector intersection.
        u1 = (e1_dot_o + tau * e1d - e1_dot_p.unsqueeze(-1)) / s[..., 0:1].clamp_min(1e-12)
        u2 = (e2_dot_o + tau * e2d - e2_dot_p.unsqueeze(-1)) / s[..., 1:2].clamp_min(1e-12)
        r2 = u1 * u1 + u2 * u2
        ok = ok & (r2 <= cutoff_sq)

        # Eq. (8.5)
        alpha = o.unsqueeze(-1) * torch.exp(-0.5 * r2)
        alpha = torch.where(ok, alpha, torch.zeros_like(alpha))
        alpha = torch.where(alpha >= float(cfg.min_alpha), alpha, torch.zeros_like(alpha))
        alpha = alpha.clamp(max=1.0 - 1e-6)

        # Front-to-back compositing, Eq. (8.6)
        one_minus = 1.0 - alpha  # strictly > 0
        trans = torch.cumprod(one_minus, dim=1)
        trans_excl = torch.cat(
            (torch.ones_like(trans[:, :1]), trans[:, :-1]), dim=1
        )  # (B, M, P)
        weights = trans_excl * alpha

        alpha_sum = weights.sum(dim=1)  # (B, P)
        color = torch.einsum("bmp,bmc->bpc", weights, c)
        out_color.append(color)
        out_alpha.append(alpha_sum)
        out_pix.append(pix.reshape(-1))

        if compute_aux:
            tau_m = torch.where(ok, tau, torch.zeros_like(tau))
            out_depth.append((weights * tau_m).sum(dim=1))
            out_normal.append(torch.einsum("bmp,bmk->bpk", weights, n))
            out_dist.append(_distortion(weights, tau_m, alpha_sum))
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
            "truncation_rate": float(truncated) / max(1, total),
            "active_tiles": float(active_tiles.numel()),
            "mean_per_tile": float(total) / max(1, int(active_tiles.numel())),
        },
    )


@torch.no_grad()
def render_2dgs_reference(
    surfels: SurfelSet2D,
    camera: Camera,
    cfg: RenderConfig | None = None,
    *,
    pixel_chunk: int = 4096,
) -> RenderOutput:
    """Brute-force rasteriser with exact per-pixel depth ordering.

    Evaluates every surfel against every pixel and sorts contributions by the
    exact :math:`\\tau^t_i(q)` of Eq. (8.2) before compositing.  This removes both
    approximations of :func:`render_2dgs` - the per-tile anchor-depth ordering and
    the ``max_per_tile`` truncation - and so serves as the reference the fast path
    is validated against.

    Cost is ``O(N * H * W)``; intended for small test images only.
    """
    cfg = cfg or RenderConfig()
    dev, dt = surfels.device, surfels.dtype
    h, w = int(camera.height), int(camera.width)
    n_ch = surfels.channels

    origins, dirs = camera.rays()
    dirs_flat = dirs.reshape(-1, 3)
    per_pixel_origin = origins.shape[0] != 1
    origins_flat = origins.reshape(-1, 3) if per_pixel_origin else origins.reshape(3)

    color = torch.zeros((h * w, n_ch), device=dev, dtype=dt)
    alpha = torch.zeros((h * w,), device=dev, dtype=dt)
    depth = torch.zeros((h * w,), device=dev, dtype=dt)
    normal = torch.zeros((h * w, 3), device=dev, dtype=dt)
    dist = torch.zeros((h * w,), device=dev, dtype=dt)
    count = torch.zeros((h * w,), device=dev, dtype=torch.int32)

    p, n = surfels.anchor, surfels.normal
    e1, e2, s = surfels.e1, surfels.e2, surfels.scale
    o, c = surfels.opacity, surfels.amplitude
    cutoff_sq = float(cfg.gaussian_cutoff) ** 2

    for lo in range(0, h * w, pixel_chunk):
        hi = min(lo + pixel_chunk, h * w)
        d = dirs_flat[lo:hi]  # (P, 3)
        oc = origins_flat[lo:hi] if per_pixel_origin else origins_flat

        nd = d @ n.transpose(0, 1)  # (P, N)
        n_dot_p = (n * p).sum(-1).unsqueeze(0)  # (1, N)
        if per_pixel_origin:
            n_dot_o = oc @ n.transpose(0, 1)
            e1_dot_o = oc @ e1.transpose(0, 1)
            e2_dot_o = oc @ e2.transpose(0, 1)
        else:
            n_dot_o = (n * oc).sum(-1).unsqueeze(0)
            e1_dot_o = (e1 * oc).sum(-1).unsqueeze(0)
            e2_dot_o = (e2 * oc).sum(-1).unsqueeze(0)

        ok = nd.abs() >= float(cfg.cull_cos)
        tau = (n_dot_p - n_dot_o) / torch.where(ok, nd, torch.ones_like(nd))
        ok = ok & (tau > float(cfg.near_mm))

        u1 = (e1_dot_o + tau * (d @ e1.transpose(0, 1)) - (e1 * p).sum(-1).unsqueeze(0)) / s[:, 0].clamp_min(1e-12)
        u2 = (e2_dot_o + tau * (d @ e2.transpose(0, 1)) - (e2 * p).sum(-1).unsqueeze(0)) / s[:, 1].clamp_min(1e-12)
        r2 = u1 * u1 + u2 * u2
        ok = ok & (r2 <= cutoff_sq)

        a = o.unsqueeze(0) * torch.exp(-0.5 * r2)
        a = torch.where(ok, a, torch.zeros_like(a))
        a = torch.where(a >= float(cfg.min_alpha), a, torch.zeros_like(a)).clamp(max=1.0 - 1e-6)

        # Exact per-pixel ordering by intersection depth.
        tau_sort = torch.where(ok, tau, torch.full_like(tau, float("inf")))
        order = torch.argsort(tau_sort, dim=1)
        a_s = torch.gather(a, 1, order)
        tau_s = torch.gather(torch.where(ok, tau, torch.zeros_like(tau)), 1, order)

        trans = torch.cumprod(1.0 - a_s, dim=1)
        trans_excl = torch.cat((torch.ones_like(trans[:, :1]), trans[:, :-1]), dim=1)
        wgt = trans_excl * a_s  # (P, N)

        c_s = c[order]  # (P, N, C)
        n_s = n[order]  # (P, N, 3)

        asum = wgt.sum(dim=1)
        color[lo:hi] = (wgt.unsqueeze(-1) * c_s).sum(dim=1)
        alpha[lo:hi] = asum
        depth[lo:hi] = (wgt * tau_s).sum(dim=1)
        normal[lo:hi] = (wgt.unsqueeze(-1) * n_s).sum(dim=1)
        mean_tau = (wgt * tau_s).sum(dim=1) / asum.clamp_min(1e-8)
        dist[lo:hi] = (wgt * (tau_s - mean_tau.unsqueeze(1)) ** 2).sum(dim=1)
        count[lo:hi] = (wgt > float(cfg.min_alpha)).sum(dim=1).to(torch.int32)

    bg = float(cfg.background)
    color_img = color.reshape(h, w, n_ch).permute(2, 0, 1)
    alpha_img = alpha.reshape(h, w)
    if bg != 0.0:
        color_img = color_img + bg * (1.0 - alpha_img).clamp_min(0.0).unsqueeze(0)

    return RenderOutput(
        color=color_img,
        alpha=alpha_img,
        depth=depth.reshape(h, w),
        normal=normal.reshape(h, w, 3).permute(2, 0, 1),
        distortion=dist.reshape(h, w),
        n_contributing=count.reshape(h, w),
        stats={"visible_surfels": float(surfels.n), "pairs": float(surfels.n * h * w)},
    )



# --------------------------------------------------------------------------- #
#  The rendering operator A_t as an explicit sparse matrix
# --------------------------------------------------------------------------- #
@dataclass
class WeightMatrix:
    """The linear rendering operator :math:`A_t` of Eq. (9.2), stored sparsely.

    Prop. 9.1 proves that with the geometry fixed, the compositing weights

    .. math:: w_i(q) = \\Bigl(\\prod_{j<i}(1-\\alpha^t_j(q))\\Bigr)\\alpha^t_i(q)

    do **not** depend on the amplitudes, so :math:`\\hat C_t = A_t a` is linear with
    :math:`A_t[q, i] = w_i(q)`.  That is what makes the residual estimation of
    Eq. (9.2) an honest linear least-squares problem rather than a linearisation.

    Storing :math:`A_t` explicitly (typically a few hundred thousand non-zeros)
    turns every conjugate-gradient iteration into two ``index_add`` calls instead
    of a full forward+backward render, which is what makes the 60-iteration solve
    of Eq. (9.3) affordable per frame.
    """

    surfel_index: Tensor
    """``(nnz,)`` column indices - into the **full** surfel array."""

    pixel_index: Tensor
    """``(nnz,)`` row indices - flat index into the unpadded ``H*W`` image."""

    weight: Tensor
    """``(nnz,)`` values :math:`w_i(q)`."""

    height: int
    width: int
    n_surfels: int
    channels: int

    @property
    def nnz(self) -> int:
        return int(self.weight.numel())

    def apply(self, amplitude: Tensor) -> Tensor:
        """:math:`A_t a` -> ``(C, H, W)``."""
        if amplitude.dim() == 1:
            amplitude = amplitude.unsqueeze(-1)
        contrib = self.weight.unsqueeze(-1) * amplitude[self.surfel_index]
        flat = torch.zeros(
            (self.height * self.width, amplitude.shape[1]),
            device=amplitude.device,
            dtype=amplitude.dtype,
        )
        flat = flat.index_add(0, self.pixel_index, contrib)
        return flat.reshape(self.height, self.width, -1).permute(2, 0, 1)

    def apply_transpose(self, image: Tensor) -> Tensor:
        """:math:`A_t^\\top r` -> ``(N, C)``. ``image`` is ``(C, H, W)``."""
        flat = image.permute(1, 2, 0).reshape(self.height * self.width, -1)
        contrib = self.weight.unsqueeze(-1) * flat[self.pixel_index]
        out = torch.zeros(
            (self.n_surfels, flat.shape[1]), device=image.device, dtype=image.dtype
        )
        return out.index_add(0, self.surfel_index, contrib)

    def alpha(self) -> Tensor:
        """``(H, W)`` accumulated opacity implied by these weights."""
        flat = torch.zeros(
            (self.height * self.width,), device=self.weight.device, dtype=self.weight.dtype
        )
        return flat.index_add(0, self.pixel_index, self.weight).reshape(self.height, self.width)

    def contributions_per_surfel(self) -> Tensor:
        """``(N,)`` :math:`\\sum_q w_i(q)` - how much each surfel is actually seen.

        Surfels with a near-zero value are unobserved from this view; Prop. 9.2's
        Tikhonov term is what keeps their residual well determined (the data term
        alone leaves them in the null space of :math:`A_t`).
        """
        out = torch.zeros(
            (self.n_surfels,), device=self.weight.device, dtype=self.weight.dtype
        )
        return out.index_add(0, self.surfel_index, self.weight)


@torch.no_grad()
def render_weights(
    surfels: SurfelSet2D,
    camera: Camera,
    cfg: RenderConfig | None = None,
    *,
    min_weight: float | None = None,
    max_pairs: int = 60_000_000,
) -> WeightMatrix:
    """Extract the sparse compositing weights :math:`w_i(q)` for one view.

    Uses exactly the same culling, binning, ordering and truncation as
    :func:`render_2dgs`, so the extracted operator is the one the renderer
    actually applies - not an idealised version of it.
    """
    cfg = cfg or RenderConfig()
    dev, dt = surfels.device, surfels.dtype
    h, w = int(camera.height), int(camera.width)
    geom = _TileGeom.build(h, w, int(cfg.tile))
    thresh = float(cfg.min_alpha if min_weight is None else min_weight)

    empty = WeightMatrix(
        surfel_index=torch.zeros((0,), device=dev, dtype=torch.long),
        pixel_index=torch.zeros((0,), device=dev, dtype=torch.long),
        weight=torch.zeros((0,), device=dev, dtype=dt),
        height=h,
        width=w,
        n_surfels=surfels.n,
        channels=surfels.channels,
    )
    if surfels.n == 0:
        return empty

    radius_mm = float(cfg.gaussian_cutoff) * surfels.scale.max(dim=1).values
    binned = _bin_surfels(surfels.anchor, radius_mm, camera, cfg, geom, max_pairs=max_pairs)
    if binned is None or binned.empty:
        return empty

    vis_idx = binned.vis_idx
    anchor_v = surfels.anchor[vis_idx]
    normal_v = surfels.normal[vis_idx]
    e1_v, e2_v = surfels.e1[vis_idx], surfels.e2[vis_idx]
    scale_v = surfels.scale[vis_idx]
    opa_v = surfels.opacity[vis_idx]

    origins, dirs = camera.rays(height=geom.hp, width=geom.wp)
    dirs_flat = dirs.reshape(-1, 3)
    per_pixel_origin = origins.shape[0] != 1
    origins_flat = origins.reshape(-1, 3) if per_pixel_origin else origins.reshape(3)

    cutoff_sq = float(cfg.gaussian_cutoff) ** 2
    chunk = max(1, int(cfg.tile_chunk))
    s_idx: list[Tensor] = []
    p_idx: list[Tensor] = []
    w_val: list[Tensor] = []

    for lo in range(0, int(binned.active_tiles.numel()), chunk):
        tsel = binned.active_tiles[lo : lo + chunk]
        idx = binned.dense[tsel]
        valid = idx >= 0
        if not bool(valid.any()):
            continue
        sid = idx.clamp_min(0)
        pix = binned.tile_pix[tsel]
        d = dirs_flat[pix]

        p, n = anchor_v[sid], normal_v[sid]
        e1, e2, s = e1_v[sid], e2_v[sid], scale_v[sid]
        o = opa_v[sid]

        nd = torch.einsum("bmk,bpk->bmp", n, d)
        e1d = torch.einsum("bmk,bpk->bmp", e1, d)
        e2d = torch.einsum("bmk,bpk->bmp", e2, d)
        n_dot_p = (n * p).sum(-1)
        e1_dot_p = (e1 * p).sum(-1)
        e2_dot_p = (e2 * p).sum(-1)

        if per_pixel_origin:
            o_c = origins_flat[pix]
            numer = n_dot_p.unsqueeze(-1) - torch.einsum("bmk,bpk->bmp", n, o_c)
            e1_dot_o = torch.einsum("bmk,bpk->bmp", e1, o_c)
            e2_dot_o = torch.einsum("bmk,bpk->bmp", e2, o_c)
        else:
            o_c = origins_flat
            numer = (n_dot_p - (n * o_c).sum(-1)).unsqueeze(-1)
            e1_dot_o = (e1 * o_c).sum(-1).unsqueeze(-1)
            e2_dot_o = (e2 * o_c).sum(-1).unsqueeze(-1)

        ok = valid.unsqueeze(-1) & (nd.abs() >= float(cfg.cull_cos))
        tau = numer / torch.where(ok, nd, torch.ones_like(nd))
        ok = ok & (tau > float(cfg.near_mm))
        u1 = (e1_dot_o + tau * e1d - e1_dot_p.unsqueeze(-1)) / s[..., 0:1].clamp_min(1e-12)
        u2 = (e2_dot_o + tau * e2d - e2_dot_p.unsqueeze(-1)) / s[..., 1:2].clamp_min(1e-12)
        r2 = u1 * u1 + u2 * u2
        ok = ok & (r2 <= cutoff_sq)

        alpha = o.unsqueeze(-1) * torch.exp(-0.5 * r2)
        alpha = torch.where(ok, alpha, torch.zeros_like(alpha))
        alpha = torch.where(alpha >= float(cfg.min_alpha), alpha, torch.zeros_like(alpha))
        alpha = alpha.clamp(max=1.0 - 1e-6)

        trans = torch.cumprod(1.0 - alpha, dim=1)
        trans_excl = torch.cat((torch.ones_like(trans[:, :1]), trans[:, :-1]), dim=1)
        weights = trans_excl * alpha  # (B, M, P)

        hit = weights > thresh
        if not bool(hit.any()):
            continue
        bi, mi, pi = torch.nonzero(hit, as_tuple=True)
        s_idx.append(vis_idx[sid[bi, mi]])
        p_idx.append(pix[bi, pi])
        w_val.append(weights[bi, mi, pi])

    if not w_val:
        return empty

    surfel_index = torch.cat(s_idx)
    padded_pixel = torch.cat(p_idx)
    weight = torch.cat(w_val)

    # Drop the padding region and re-index into the unpadded image.
    py = torch.div(padded_pixel, geom.wp, rounding_mode="floor")
    px = padded_pixel - py * geom.wp
    inside = (py < h) & (px < w)
    surfel_index = surfel_index[inside]
    weight = weight[inside]
    pixel_index = (py[inside] * w + px[inside])

    return WeightMatrix(
        surfel_index=surfel_index,
        pixel_index=pixel_index,
        weight=weight,
        height=h,
        width=w,
        n_surfels=surfels.n,
        channels=surfels.channels,
    )
