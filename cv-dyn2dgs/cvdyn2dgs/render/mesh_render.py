"""Z-buffered triangle rasteriser for the mesh-only baseline.

Proposal §4.3 and §5.8 insist the mesh comparison be *strong*.  This renderer is
therefore deliberately generous to the mesh:

* **Per-vertex normals from the level-set gradient**, barycentrically interpolated
  -> Phong-quality smooth shading, better than face normals.
* **Per-vertex amplitudes sampled from the MRI**, barycentrically interpolated ->
  a textured mesh, which is exactly the competitor proposal §4.3 names ("a
  sufficiently dense and smooth textured mesh can also reach high quality").
* **Perspective-correct interpolation** of depth and attributes.
* **Exact visibility** from a z-buffer, with no truncation analogous to the
  surfel rasteriser's ``max_per_tile``.

The one thing a mesh cannot do is produce a soft silhouette: coverage is binary
per pixel, so ``alpha`` is 0/1.  That is not a handicap imposed by this code, it is
the property under test - whether overlapping Gaussian kernels give a measurably
better boundary than triangle edges at a comparable primitive budget (RQ5).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..levelset.mesh_extract import TriangleMesh
from .camera import Camera
from .raster2dgs import RenderOutput

__all__ = ["render_mesh"]


@torch.no_grad()
def render_mesh(
    mesh: TriangleMesh,
    camera: Camera,
    *,
    vertex_amplitude: Tensor | None = None,
    background: float = 0.0,
    near_mm: float = 1e-3,
    max_pairs: int = 80_000_000,
) -> RenderOutput:
    """Rasterise a triangle mesh with a z-buffer.

    Parameters
    ----------
    vertex_amplitude:
        ``(V, C)`` per-vertex intensity.  If ``None`` the mesh is shaded with a
        constant 1.0, which corresponds to the "flat single-colour surface" case
        where proposal §4.3 expects a mesh to be entirely sufficient.

    Returns
    -------
    A :class:`RenderOutput` with the same fields as the surfel rasterisers, so all
    metrics apply unchanged.  ``distortion`` is identically zero: an opaque mesh
    has exactly one surface per ray, so the depth-distortion penalty that the 2DGS
    losses need has nothing to act on.
    """
    dev, dt = mesh.vertices.device, mesh.vertices.dtype
    h, w = int(camera.height), int(camera.width)
    n_ch = 1 if vertex_amplitude is None else int(vertex_amplitude.shape[1])
    n_pix = h * w

    def _empty() -> RenderOutput:
        return RenderOutput(
            color=torch.full((n_ch, h, w), float(background), device=dev, dtype=dt),
            alpha=torch.zeros((h, w), device=dev, dtype=dt),
            depth=torch.zeros((h, w), device=dev, dtype=dt),
            normal=torch.zeros((3, h, w), device=dev, dtype=dt),
            distortion=torch.zeros((h, w), device=dev, dtype=dt),
            n_contributing=torch.zeros((h, w), device=dev, dtype=torch.int32),
            stats={"triangles": 0.0, "pairs": 0.0},
        )

    if mesh.n_faces == 0:
        return _empty()

    if vertex_amplitude is None:
        vertex_amplitude = torch.ones((mesh.n_vertices, 1), device=dev, dtype=dt)

    uv, z = camera.project(mesh.vertices)  # (V,2), (V,)
    tri = mesh.faces  # (F,3)
    tuv = uv[tri]  # (F,3,2)
    tz = z[tri]  # (F,3)

    # ---- triangle culling -------------------------------------------------
    ax, ay = tuv[:, 0, 0], tuv[:, 0, 1]
    bx, by = tuv[:, 1, 0], tuv[:, 1, 1]
    cx, cy = tuv[:, 2, 0], tuv[:, 2, 1]
    area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    x0 = torch.minimum(torch.minimum(ax, bx), cx)
    x1 = torch.maximum(torch.maximum(ax, bx), cx)
    y0 = torch.minimum(torch.minimum(ay, by), cy)
    y1 = torch.maximum(torch.maximum(ay, by), cy)

    ok_tri = (
        (tz > near_mm).all(dim=1)
        & (area2.abs() > 1e-12)
        & (x1 >= 0)
        & (y1 >= 0)
        & (x0 <= w - 1)
        & (y0 <= h - 1)
    )
    if not bool(ok_tri.any()):
        return _empty()
    fsel = torch.nonzero(ok_tri, as_tuple=False).squeeze(1)

    tuv, tz, area2 = tuv[fsel], tz[fsel], area2[fsel]
    tri = tri[fsel]
    x0 = x0[fsel].floor().clamp(0, w - 1).long()
    x1 = x1[fsel].ceil().clamp(0, w - 1).long()
    y0 = y0[fsel].floor().clamp(0, h - 1).long()
    y1 = y1[fsel].ceil().clamp(0, h - 1).long()

    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    counts = bw * bh
    total = int(counts.sum().item())
    if total == 0:
        return _empty()
    if total > max_pairs:
        raise RuntimeError(
            f"{total} (triangle, pixel) pairs exceeds max_pairs={max_pairs}; "
            "reduce the image size or decimate the mesh"
        )

    # ---- enumerate (triangle, pixel) candidate pairs ----------------------
    n_tri = int(fsel.numel())
    pair_tri = torch.repeat_interleave(torch.arange(n_tri, device=dev), counts)
    cum = torch.cumsum(counts, 0) - counts
    local = torch.arange(total, device=dev) - torch.repeat_interleave(cum, counts)
    bw_of = bw[pair_tri]
    dy = torch.div(local, bw_of, rounding_mode="floor")
    dx = local - dy * bw_of
    px = x0[pair_tri] + dx
    py = y0[pair_tri] + dy

    pxf = px.to(dt) + 0.5
    pyf = py.to(dt) + 0.5

    a = tuv[pair_tri, 0]
    b = tuv[pair_tri, 1]
    c = tuv[pair_tri, 2]
    den = area2[pair_tri]

    l0 = ((b[:, 0] - pxf) * (c[:, 1] - pyf) - (b[:, 1] - pyf) * (c[:, 0] - pxf)) / den
    l1 = ((c[:, 0] - pxf) * (a[:, 1] - pyf) - (c[:, 1] - pyf) * (a[:, 0] - pxf)) / den
    l2 = 1.0 - l0 - l1

    eps = -1e-6
    inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
    if not bool(inside.any()):
        return _empty()
    isel = torch.nonzero(inside, as_tuple=False).squeeze(1)
    pair_tri, px, py = pair_tri[isel], px[isel], py[isel]
    l0, l1, l2 = l0[isel], l1[isel], l2[isel]

    # ---- depth and attribute interpolation --------------------------------
    zt = tz[pair_tri]  # (P, 3)
    bary = torch.stack((l0, l1, l2), dim=-1)
    if camera.orthographic:
        # Under parallel projection screen-space barycentrics are already correct;
        # applying the 1/z correction here would *introduce* error.
        depth = (bary * zt).sum(dim=-1)
        wbary = bary
    else:
        inv_z = bary / zt.clamp_min(1e-9)
        inv_sum = inv_z.sum(dim=-1).clamp_min(1e-12)
        depth = 1.0 / inv_sum
        wbary = inv_z / inv_sum.unsqueeze(-1)  # perspective-correct weights

    pix_id = py * w + px

    # ---- z-buffer ---------------------------------------------------------
    zbuf = torch.full((n_pix,), float("inf"), device=dev, dtype=dt)
    zbuf = zbuf.scatter_reduce(0, pix_id, depth, reduce="amin", include_self=True)
    winner_mask = depth <= zbuf[pix_id] * (1.0 + 1e-6) + 1e-9

    win_idx = torch.full((n_pix,), -1, device=dev, dtype=torch.long)
    cand = torch.nonzero(winner_mask, as_tuple=False).squeeze(1)
    # Last write wins; any of the tied-depth candidates is an acceptable choice.
    win_idx[pix_id[cand]] = cand
    covered = win_idx >= 0
    if not bool(covered.any()):
        return _empty()

    wsel = win_idx[covered]
    tsel = pair_tri[wsel]
    wb = wbary[wsel]  # (K, 3)
    vtri = tri[tsel]  # (K, 3) vertex ids

    vn = mesh.normals[vtri]  # (K, 3, 3)
    normal_px = (wb.unsqueeze(-1) * vn).sum(dim=1)
    normal_px = normal_px / normal_px.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    va = vertex_amplitude[vtri]  # (K, 3, C)
    color_px = (wb.unsqueeze(-1) * va).sum(dim=1)

    color = torch.zeros((n_pix, n_ch), device=dev, dtype=dt)
    alpha = torch.zeros((n_pix,), device=dev, dtype=dt)
    depth_img = torch.zeros((n_pix,), device=dev, dtype=dt)
    normal_img = torch.zeros((n_pix, 3), device=dev, dtype=dt)
    count = torch.zeros((n_pix,), device=dev, dtype=torch.int32)

    color[covered] = color_px
    alpha[covered] = 1.0
    depth_img[covered] = depth[wsel]
    normal_img[covered] = normal_px
    count[covered] = 1

    color_out = color.reshape(h, w, n_ch).permute(2, 0, 1)
    alpha_out = alpha.reshape(h, w)
    if background != 0.0:
        color_out = color_out + float(background) * (1.0 - alpha_out).unsqueeze(0)

    return RenderOutput(
        color=color_out,
        alpha=alpha_out,
        depth=depth_img.reshape(h, w),
        normal=normal_img.reshape(h, w, 3).permute(2, 0, 1),
        distortion=torch.zeros((h, w), device=dev, dtype=dt),
        n_contributing=count.reshape(h, w),
        stats={
            "triangles": float(n_tri),
            "pairs": float(total),
            "covered_pixels": float(int(covered.sum().item())),
            "vertices": float(mesh.n_vertices),
        },
    )
