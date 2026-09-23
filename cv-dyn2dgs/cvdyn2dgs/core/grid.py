"""Voxel grid geometry and interpolation.

Conventions used throughout CV-Dyn2DGS
--------------------------------------
* A 3-D volume is a tensor of shape ``(nx, ny, nz)`` indexed ``vol[i, j, k]``.
  This matches the paper's :math:`\\phi_{i,j,k}` notation (theory Eq. 4.12-4.15).
* Voxel spacing is ``h = (hx, hy, hz)`` in **millimetres**.  MRI short-axis cine
  is anisotropic with ``hz >= hx ~= hy`` (theory, Notation §2.1).
* World (physical) coordinates in mm are
  ``world = origin + (i * hx, j * hy, k * hz)``.
* Level-set sign convention (proposal Eq. 1, theory §2.1):
  ``phi > 0`` inside the heart, ``phi < 0`` outside, ``phi == 0`` on the surface.
  This is the *opposite* of the more common "inside is negative" SDF convention,
  so every comparison in this codebase is written explicitly against it.

All distances, curvatures and surface errors are computed in mm, never in voxel
index units (proposal §3.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

__all__ = ["Grid", "trilinear_sample", "trilinear_sample_vector"]


@dataclass(frozen=True)
class Grid:
    """An anisotropic, axis-aligned voxel grid.

    Parameters
    ----------
    shape:
        ``(nx, ny, nz)`` number of voxels along each axis.
    spacing:
        ``(hx, hy, hz)`` voxel spacing in mm.
    origin:
        World coordinate of voxel ``(0, 0, 0)`` in mm.
    """

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or len(self.spacing) != 3 or len(self.origin) != 3:
            raise ValueError("Grid.shape/spacing/origin must each have length 3")
        if any(int(n) < 2 for n in self.shape):
            raise ValueError(f"Grid needs at least 2 voxels per axis, got {self.shape}")
        if any(float(s) <= 0.0 for s in self.spacing):
            raise ValueError(f"Grid spacing must be positive, got {self.spacing}")

    # ------------------------------------------------------------------ sizes
    @property
    def nx(self) -> int:
        return int(self.shape[0])

    @property
    def ny(self) -> int:
        return int(self.shape[1])

    @property
    def nz(self) -> int:
        return int(self.shape[2])

    @property
    def numel(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def hx(self) -> float:
        return float(self.spacing[0])

    @property
    def hy(self) -> float:
        return float(self.spacing[1])

    @property
    def hz(self) -> float:
        return float(self.spacing[2])

    @property
    def h_max(self) -> float:
        """:math:`h_{\\max}` from theory Prop. 7.3 (normal angular error bound)."""
        return max(self.spacing)

    @property
    def h_min(self) -> float:
        return min(self.spacing)

    @property
    def voxel_volume_mm3(self) -> float:
        """Volume of one voxel in mm^3; used for LV volume / EF (Eq. 41)."""
        return self.hx * self.hy * self.hz

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return (
            (self.nx - 1) * self.hx,
            (self.ny - 1) * self.hy,
            (self.nz - 1) * self.hz,
        )

    # ------------------------------------------------------------- conversion
    def spacing_tensor(self, device=None, dtype=torch.float32) -> Tensor:
        return torch.tensor(self.spacing, device=device, dtype=dtype)

    def origin_tensor(self, device=None, dtype=torch.float32) -> Tensor:
        return torch.tensor(self.origin, device=device, dtype=dtype)

    def voxel_to_world(self, idx: Tensor) -> Tensor:
        """Continuous voxel indices ``(..., 3)`` -> world mm ``(..., 3)``."""
        h = self.spacing_tensor(idx.device, idx.dtype)
        o = self.origin_tensor(idx.device, idx.dtype)
        return idx * h + o

    def world_to_voxel(self, xyz: Tensor) -> Tensor:
        """World mm ``(..., 3)`` -> continuous voxel indices ``(..., 3)``."""
        h = self.spacing_tensor(xyz.device, xyz.dtype)
        o = self.origin_tensor(xyz.device, xyz.dtype)
        return (xyz - o) / h

    def center_world(self, device=None, dtype=torch.float32) -> Tensor:
        """World coordinate of the grid centre (handy for camera placement)."""
        idx = torch.tensor(
            [(self.nx - 1) / 2.0, (self.ny - 1) / 2.0, (self.nz - 1) / 2.0],
            device=device,
            dtype=dtype,
        )
        return self.voxel_to_world(idx)

    # ------------------------------------------------------------- meshgrids
    def voxel_meshgrid(self, device=None, dtype=torch.float32) -> Tensor:
        """``(3, nx, ny, nz)`` tensor of voxel indices."""
        ii = torch.arange(self.nx, device=device, dtype=dtype)
        jj = torch.arange(self.ny, device=device, dtype=dtype)
        kk = torch.arange(self.nz, device=device, dtype=dtype)
        gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing="ij")
        return torch.stack((gi, gj, gk), dim=0)

    def world_meshgrid(self, device=None, dtype=torch.float32) -> Tensor:
        """``(3, nx, ny, nz)`` tensor of world coordinates in mm."""
        vox = self.voxel_meshgrid(device=device, dtype=dtype)
        h = self.spacing_tensor(device, dtype).view(3, 1, 1, 1)
        o = self.origin_tensor(device, dtype).view(3, 1, 1, 1)
        return vox * h + o

    # ------------------------------------------------------------------ misc
    def with_shape(self, shape: Sequence[int]) -> "Grid":
        return Grid(tuple(int(s) for s in shape), self.spacing, self.origin)  # type: ignore[arg-type]

    def inside_mask(self, xyz_world: Tensor, margin_vox: float = 0.0) -> Tensor:
        """Boolean mask of points that lie within the grid (minus a margin)."""
        vox = self.world_to_voxel(xyz_world)
        hi = torch.tensor(
            [self.nx - 1.0, self.ny - 1.0, self.nz - 1.0],
            device=xyz_world.device,
            dtype=xyz_world.dtype,
        )
        lo = torch.full_like(hi, float(margin_vox))
        return ((vox >= lo) & (vox <= hi - margin_vox)).all(dim=-1)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Grid(shape={self.shape}, spacing_mm={self.spacing}, "
            f"origin_mm={self.origin})"
        )


# --------------------------------------------------------------------------- #
#  Interpolation
# --------------------------------------------------------------------------- #
def _gather_corners(vol: Tensor, i0: Tensor, j0: Tensor, k0: Tensor) -> Tensor:
    """Gather the 8 corner values for each query point.

    ``vol`` is ``(C, nx, ny, nz)``; returns ``(C, M, 8)`` ordered as
    ``(i0j0k0, i0j0k1, i0j1k0, i0j1k1, i1j0k0, i1j0k1, i1j1k0, i1j1k1)``.
    """
    C, nx, ny, nz = vol.shape
    flat = vol.reshape(C, -1)

    i1 = (i0 + 1).clamp_(max=nx - 1)
    j1 = (j0 + 1).clamp_(max=ny - 1)
    k1 = (k0 + 1).clamp_(max=nz - 1)

    def lin(i: Tensor, j: Tensor, k: Tensor) -> Tensor:
        return (i * ny + j) * nz + k

    idx = torch.stack(
        (
            lin(i0, j0, k0),
            lin(i0, j0, k1),
            lin(i0, j1, k0),
            lin(i0, j1, k1),
            lin(i1, j0, k0),
            lin(i1, j0, k1),
            lin(i1, j1, k0),
            lin(i1, j1, k1),
        ),
        dim=-1,
    )  # (M, 8)
    # (C, M, 8)
    return flat[:, idx.reshape(-1)].reshape(C, idx.shape[0], 8)


def trilinear_sample(vol: Tensor, pts_vox: Tensor) -> Tensor:
    """Trilinearly interpolate a scalar volume at continuous voxel coordinates.

    Border handling clamps to the edge voxel (replicate padding), which keeps the
    Newton-like surface projection of Eq. (6.3) well defined for anchors that
    drift marginally outside the grid.

    Parameters
    ----------
    vol:
        ``(nx, ny, nz)`` scalar volume.
    pts_vox:
        ``(..., 3)`` continuous voxel indices.

    Returns
    -------
    ``(...)`` interpolated values, differentiable w.r.t. both ``vol`` and
    ``pts_vox``.
    """
    if vol.dim() != 3:
        raise ValueError(f"trilinear_sample expects a 3-D volume, got {tuple(vol.shape)}")
    out = trilinear_sample_vector(vol.unsqueeze(0), pts_vox)
    return out.squeeze(-1) if out.shape[-1] == 1 else out[..., 0]


def trilinear_sample_vector(vol: Tensor, pts_vox: Tensor) -> Tensor:
    """Trilinearly interpolate a ``(C, nx, ny, nz)`` volume.

    Parameters
    ----------
    vol:
        ``(C, nx, ny, nz)`` multi-channel volume (e.g. ``C=3`` for
        :math:`\\nabla_h \\phi_t`).
    pts_vox:
        ``(..., 3)`` continuous voxel indices.

    Returns
    -------
    ``(..., C)`` interpolated values.
    """
    if vol.dim() != 4:
        raise ValueError(
            f"trilinear_sample_vector expects (C,nx,ny,nz), got {tuple(vol.shape)}"
        )
    C, nx, ny, nz = vol.shape
    batch_shape = pts_vox.shape[:-1]
    p = pts_vox.reshape(-1, 3)

    # Clamp query points into the valid interpolation domain.
    hi = torch.tensor([nx - 1.0, ny - 1.0, nz - 1.0], device=p.device, dtype=p.dtype)
    p = p.clamp(min=torch.zeros_like(hi), max=hi)

    base = torch.floor(p)
    # Keep the lower corner strictly inside so that +1 stays in range.
    base = base.clamp(
        min=torch.zeros_like(hi),
        max=torch.tensor([nx - 2.0, ny - 2.0, nz - 2.0], device=p.device, dtype=p.dtype),
    )
    frac = p - base  # in [0, 1]

    i0 = base[:, 0].long()
    j0 = base[:, 1].long()
    k0 = base[:, 2].long()

    corners = _gather_corners(vol, i0, j0, k0)  # (C, M, 8)

    fx = frac[:, 0].unsqueeze(0)  # (1, M)
    fy = frac[:, 1].unsqueeze(0)
    fz = frac[:, 2].unsqueeze(0)
    gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz

    w = torch.stack(
        (
            gx * gy * gz,
            gx * gy * fz,
            gx * fy * gz,
            gx * fy * fz,
            fx * gy * gz,
            fx * gy * fz,
            fx * fy * gz,
            fx * fy * fz,
        ),
        dim=-1,
    )  # (1, M, 8)

    vals = (corners * w).sum(dim=-1)  # (C, M)
    return vals.transpose(0, 1).reshape(*batch_shape, C)
