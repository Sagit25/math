"""Storage accounting and compression ratio, proposal Eq. (36)-(38).

Storing every frame's 2DGS costs

.. math:: S_{\\mathrm{full}} = T N P_{2D}  \\qquad\\text{(Eq. 36)}

while the proposed representation :math:`\\mathcal{D} = \\{\\mathcal{G}^{2D}_0,
\\{\\Gamma_t\\}, \\{\\Delta a_t\\}\\}` costs

.. math:: S_{\\mathrm{ours}} = N P_{2D} + \\sum_t S(\\Gamma_t) + T N P_r
          \\qquad\\text{(Eq. 37)},
          \\qquad \\mathrm{CR} = S_{\\mathrm{full}} / S_{\\mathrm{ours}}
          \\qquad\\text{(Eq. 38)}.

The whole point of RQ3 is that this is **not** automatically a win: proposal §2.9
says outright that "if the surface files and residuals are too large there may be no
gain at all".  So this module is written to make it hard to flatter the method:

* :math:`S(\\Gamma_t)` is reported for all three storage modes of proposal §8.3
  (narrow-band SDF / binary mask / mesh), not just the cheapest one;
* accounting is in **bytes**, with index and occupancy overhead counted, not in
  idealised "parameter counts" that quietly omit the cost of saying *where* a
  sparse value lives;
* both a literal and a redundancy-exploiting :math:`P_{2D}` are reported, and the
  same choice is applied to numerator and denominator so the ratio stays fair;
* the break-even surface budget is computed explicitly, so one can see how large the
  per-frame surface is allowed to get before the method loses.

Baseline sizes for competing representations (independent per-frame 2DGS, per-frame
meshes) are computed by the same functions, so the comparison is apples to apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch
from torch import Tensor

from ..core.grid import Grid
from ..levelset.mesh_extract import TriangleMesh

__all__ = [
    "BYTES_FLOAT32",
    "BYTES_INT32",
    "SurfaceStorage",
    "StorageReport",
    "narrow_band_bytes",
    "mask_bytes",
    "mesh_bytes",
    "surface_storage",
    "storage_report",
]

BYTES_FLOAT32 = 4
BYTES_INT32 = 4

SurfaceMode = Literal["narrow_band_sdf", "mask", "mesh"]


@dataclass
class SurfaceStorage:
    """Cost of storing one frame's surface :math:`\\Gamma_t`, in bytes."""

    mode: str
    total_bytes: int
    value_bytes: int = 0
    overhead_bytes: int = 0
    n_elements: int = 0
    detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        out = {
            f"surface/{self.mode}/total_bytes": float(self.total_bytes),
            f"surface/{self.mode}/value_bytes": float(self.value_bytes),
            f"surface/{self.mode}/overhead_bytes": float(self.overhead_bytes),
            f"surface/{self.mode}/n_elements": float(self.n_elements),
        }
        out.update({f"surface/{self.mode}/{k}": v for k, v in self.detail.items()})
        return out


def narrow_band_bytes(phi: Tensor, grid: Grid, band_mm: float) -> SurfaceStorage:
    """Cost of a narrow-band signed distance field.

    Stores one ``float32`` per band voxel plus a **1-bit-per-voxel occupancy mask
    over the band's bounding box** so the reader knows which voxels are present.
    That bitmask is real cost and is counted; quoting only the value array would
    understate the size.
    """
    band = phi.abs() < float(band_mm)
    n_band = int(band.sum().item())

    # Bounding box of the band -> the bitmask only needs to cover that.
    nx, ny, nz = phi.shape
    ax = band.any(dim=2).any(dim=1)
    ay = band.any(dim=2).any(dim=0)
    az = band.any(dim=1).any(dim=0)

    def span(v: Tensor, n: int) -> int:
        idx = torch.nonzero(v, as_tuple=False)
        if idx.numel() == 0:
            return 0
        return int(idx.max().item() - idx.min().item()) + 1

    bbox = span(ax, nx) * span(ay, ny) * span(az, nz)
    value_bytes = n_band * BYTES_FLOAT32
    overhead = (bbox + 7) // 8 + 6 * BYTES_INT32  # bitmask + bbox corners
    return SurfaceStorage(
        mode="narrow_band_sdf",
        total_bytes=value_bytes + overhead,
        value_bytes=value_bytes,
        overhead_bytes=overhead,
        n_elements=n_band,
        detail={
            "band_mm": float(band_mm),
            "bbox_voxels": float(bbox),
            "band_fraction_of_volume": n_band / float(phi.numel()),
        },
    )


def mask_bytes(phi: Tensor, grid: Grid) -> SurfaceStorage:
    """Cost of a binary mask: 1 bit per voxel over the mask's bounding box.

    Cheapest option, but it quantises the surface to the voxel grid, which shows up
    as a worse HD95 and a blockier silhouette.  The size/quality trade-off is the
    point of reporting all three modes.
    """
    mask = phi > 0
    nx, ny, nz = phi.shape
    ax = mask.any(dim=2).any(dim=1)
    ay = mask.any(dim=2).any(dim=0)
    az = mask.any(dim=1).any(dim=0)

    def span(v: Tensor) -> int:
        idx = torch.nonzero(v, as_tuple=False)
        if idx.numel() == 0:
            return 0
        return int(idx.max().item() - idx.min().item()) + 1

    bbox = span(ax) * span(ay) * span(az)
    total = (bbox + 7) // 8 + 6 * BYTES_INT32
    return SurfaceStorage(
        mode="mask",
        total_bytes=total,
        value_bytes=(bbox + 7) // 8,
        overhead_bytes=6 * BYTES_INT32,
        n_elements=int(mask.sum().item()),
        detail={"bbox_voxels": float(bbox)},
    )


def mesh_bytes(mesh: TriangleMesh) -> SurfaceStorage:
    """Cost of a triangle mesh: 3 float32 per vertex + 3 int32 per face."""
    v_bytes = mesh.n_vertices * 3 * BYTES_FLOAT32
    f_bytes = mesh.n_faces * 3 * BYTES_INT32
    return SurfaceStorage(
        mode="mesh",
        total_bytes=v_bytes + f_bytes,
        value_bytes=v_bytes,
        overhead_bytes=f_bytes,
        n_elements=mesh.n_vertices,
        detail={"vertices": float(mesh.n_vertices), "faces": float(mesh.n_faces)},
    )


def surface_storage(
    phi: Tensor,
    grid: Grid,
    *,
    band_mm: float = 4.0,
    mesh: TriangleMesh | None = None,
) -> dict[str, SurfaceStorage]:
    """All available storage modes for one frame's surface."""
    out = {
        "narrow_band_sdf": narrow_band_bytes(phi, grid, band_mm),
        "mask": mask_bytes(phi, grid),
    }
    if mesh is not None:
        out["mesh"] = mesh_bytes(mesh)
    return out


@dataclass
class StorageReport:
    """Eq. (36)-(38) for one sequence, in bytes."""

    n_frames: int
    n_surfels: int
    channels: int
    p2d: int
    p_residual: int

    s_full_bytes: int
    """:math:`S_{\\mathrm{full}} = T N P_{2D}`, Eq. (36)."""

    canonical_bytes: int
    surface_bytes: int
    residual_bytes: int
    s_ours_bytes: int
    """:math:`S_{\\mathrm{ours}}`, Eq. (37)."""

    compression_ratio: float
    """:math:`\\mathrm{CR}`, Eq. (38). ``> 1`` means the method wins."""

    surface_mode: str
    residual_mode: str
    per_frame_surface_bytes: list[int] = field(default_factory=list)
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def break_even_surface_bytes_per_frame(self) -> float:
        """How large :math:`S(\\Gamma_t)` may become before ``CR`` drops to 1.

        Solving :math:`S_{\\mathrm{ours}} = S_{\\mathrm{full}}` for the per-frame
        surface budget:

        .. math::
            S^{\\ast} = \\frac{T N P_{2D} - N P_{2D} - T N P_r}{T}.

        A negative value means the residual alone already costs more than storing
        every frame's surfels - i.e. the compact representation cannot win at this
        configuration, which is exactly the outcome proposal §2.9 warns about.
        """
        if self.n_frames == 0:
            return 0.0
        return (self.s_full_bytes - self.canonical_bytes - self.residual_bytes) / self.n_frames

    def to_dict(self) -> dict[str, float | str | list[int]]:
        d: dict[str, float | str | list[int]] = {
            "n_frames": float(self.n_frames),
            "n_surfels": float(self.n_surfels),
            "p2d": float(self.p2d),
            "p_residual": float(self.p_residual),
            "s_full_bytes": float(self.s_full_bytes),
            "s_full_mb": self.s_full_bytes / 1024**2,
            "canonical_bytes": float(self.canonical_bytes),
            "surface_bytes": float(self.surface_bytes),
            "residual_bytes": float(self.residual_bytes),
            "s_ours_bytes": float(self.s_ours_bytes),
            "s_ours_mb": self.s_ours_bytes / 1024**2,
            "compression_ratio": self.compression_ratio,
            "surface_mode": self.surface_mode,
            "residual_mode": self.residual_mode,
            "break_even_surface_bytes_per_frame": self.break_even_surface_bytes_per_frame,
            "surface_share": self.surface_bytes / max(1, self.s_ours_bytes),
            "residual_share": self.residual_bytes / max(1, self.s_ours_bytes),
            "canonical_share": self.canonical_bytes / max(1, self.s_ours_bytes),
        }
        d.update(self.extras)
        return d


def storage_report(
    *,
    n_frames: int,
    n_surfels: int,
    channels: int,
    surface_bytes_per_frame: Sequence[int],
    p2d: int,
    p_residual: int,
    lowrank_rank: int | None = None,
    surface_mode: str = "narrow_band_sdf",
) -> StorageReport:
    """Assemble Eq. (36)-(38).

    Parameters
    ----------
    p2d:
        :math:`P_{2D}` - numbers per surfel.  Use the *same* value for
        :math:`S_{\\mathrm{full}}` and the canonical term; see
        :class:`cvdyn2dgs.surfel.model.StorageCount` for the literal and
        redundancy-exploiting choices.
    lowrank_rank:
        When set, the residual is stored factored (Eq. 9.5) at
        :math:`r(N + T)` values instead of :math:`T N P_r`.
    """
    t, n = int(n_frames), int(n_surfels)
    s_full = t * n * p2d * BYTES_FLOAT32
    canonical = n * p2d * BYTES_FLOAT32
    surface = int(sum(int(b) for b in surface_bytes_per_frame))

    if lowrank_rank is None:
        residual = t * n * p_residual * BYTES_FLOAT32
        res_mode = "dense_scalar"
    else:
        residual = int(lowrank_rank) * (n + t) * channels * BYTES_FLOAT32
        res_mode = f"lowrank_r{int(lowrank_rank)}"

    s_ours = canonical + surface + residual
    return StorageReport(
        n_frames=t,
        n_surfels=n,
        channels=channels,
        p2d=p2d,
        p_residual=p_residual,
        s_full_bytes=s_full,
        canonical_bytes=canonical,
        surface_bytes=surface,
        residual_bytes=residual,
        s_ours_bytes=s_ours,
        compression_ratio=s_full / max(1, s_ours),
        surface_mode=surface_mode,
        residual_mode=res_mode,
        per_frame_surface_bytes=[int(b) for b in surface_bytes_per_frame],
        extras={
            "mean_surface_bytes_per_frame": surface / max(1, t),
        },
    )
