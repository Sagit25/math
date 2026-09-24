"""Serialisation of the stored representation, and verifiable size reporting.

Proposal §8.3 asks for :math:`S_{\\mathrm{ours}}` (Eq. 37) and the compression ratio
(Eq. 38).  A parameter count alone is easy to flatter, so this module does two things:

1. **Actually writes** the compact representation - canonical surfels, a packed
   narrow-band signed distance field per frame, and the residuals (dense or factored).
2. Reports the **theoretical** byte count from Eq. (36)-(37) *next to* the **measured**
   file size, so the two can be checked against each other.

If the measured size is much larger than the theoretical one, the theory is not wrong -
the serialisation has overhead - and saying so is more useful than only quoting the
favourable number.

The narrow band is stored as a bounding box, a 1-bit-per-voxel occupancy bitmask over
that box, and a dense ``float32`` array of the in-band values, in raster order.  That
is the same model the accounting in :mod:`cvdyn2dgs.metrics.storage` assumes, so the
two agree by construction rather than by coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..core.grid import Grid
from ..levelset.mesh_extract import marching_tetrahedra
from ..metrics.storage import StorageReport, storage_report, surface_storage
from ..surfel.model import SurfelSet2D
from .precompute import PrecomputedModel

__all__ = [
    "PackedBand",
    "pack_narrow_band",
    "unpack_narrow_band",
    "save_model",
    "load_model",
    "model_storage_report",
]

_FORMAT_VERSION = 1


@dataclass
class PackedBand:
    """A narrow-band signed distance field in its serialised form."""

    bbox_lo: Tensor
    """``(3,)`` int32 lower corner of the bounding box (voxel indices)."""

    bbox_shape: Tensor
    """``(3,)`` int32 box extent in voxels."""

    bitmask: Tensor
    """``(ceil(prod(bbox_shape)/8),)`` uint8 occupancy, 1 bit per box voxel."""

    values: Tensor
    """``(n_band,)`` float32 level-set values, in raster order of the set bits."""

    fill_outside: float
    """Constant magnitude used to reconstruct voxels outside the band; the sign comes
    from ``sign_mask``."""

    sign_mask: Tensor
    """``(ceil(numel/8),)`` uint8 sign of the *whole* volume, 1 = inside.

    Needed because the constant extension of Eq. (5.1) is sign-preserving: a reader
    must know which side of the surface an out-of-band voxel is on, or the mask and
    every volume measurement would be wrong.
    """

    def nbytes(self) -> int:
        return int(
            self.bbox_lo.numel() * 4
            + self.bbox_shape.numel() * 4
            + self.bitmask.numel()
            + self.values.numel() * 4
            + self.sign_mask.numel()
        )


def _pack_bits(flat_bool: Tensor) -> Tensor:
    """Pack a flat boolean tensor into uint8, 8 values per byte (LSB first)."""
    n = int(flat_bool.numel())
    pad = (-n) % 8
    if pad:
        flat_bool = torch.cat(
            (flat_bool, torch.zeros(pad, dtype=torch.bool, device=flat_bool.device))
        )
    bits = flat_bool.reshape(-1, 8).to(torch.int16)
    weights = (2 ** torch.arange(8, device=flat_bool.device, dtype=torch.int16)).reshape(1, 8)
    # Accumulate in int16: the row sum reaches 255, which would be at the very edge
    # of uint8 and is not worth risking.
    return (bits * weights).sum(dim=1).to(torch.uint8)


def _unpack_bits(packed: Tensor, n: int) -> Tensor:
    """Inverse of :func:`_pack_bits`."""
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8).reshape(1, 8)
    bits = (packed.reshape(-1, 1) >> shifts) & 1
    return bits.reshape(-1)[:n].to(torch.bool)


@torch.no_grad()
def pack_narrow_band(phi: Tensor, band_mm: float, *, fill_outside: float | None = None) -> PackedBand:
    """Serialise a level set as a narrow band plus a global sign mask."""
    band = phi.abs() < float(band_mm)
    nx, ny, nz = phi.shape

    idx = torch.nonzero(band, as_tuple=False)
    if idx.numel() == 0:
        # device-ok: the empty-band early return builds a record for SERIALISATION, which
        # is a CPU byte-level operation throughout (see _pack_bits below).
        lo = torch.zeros(3, dtype=torch.int32)
        shape = torch.zeros(3, dtype=torch.int32)  # device-ok: as above
        return PackedBand(
            bbox_lo=lo,
            bbox_shape=shape,
            # device-ok: empty buffers destined for a file, never used in arithmetic.
            bitmask=torch.zeros(0, dtype=torch.uint8),
            values=torch.zeros(0, dtype=torch.float32),  # device-ok: as above
            fill_outside=float(band_mm) if fill_outside is None else float(fill_outside),
            sign_mask=_pack_bits((phi > 0).reshape(-1).cpu()),
        )

    lo = idx.min(dim=0).values
    hi = idx.max(dim=0).values
    shape = hi - lo + 1

    sub_band = band[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]
    sub_phi = phi[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]

    flat_mask = sub_band.reshape(-1)
    values = sub_phi.reshape(-1)[flat_mask].to(torch.float32)

    return PackedBand(
        bbox_lo=lo.to(torch.int32).cpu(),
        bbox_shape=shape.to(torch.int32).cpu(),
        bitmask=_pack_bits(flat_mask.cpu()),
        values=values.cpu(),
        fill_outside=float(band_mm) if fill_outside is None else float(fill_outside),
        sign_mask=_pack_bits((phi > 0).reshape(-1).cpu()),
    )


@torch.no_grad()
def unpack_narrow_band(
    packed: PackedBand, grid: Grid, *, device=None, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Reconstruct a level set from its packed form.

    Out-of-band voxels get ``+fill_outside`` inside and ``-fill_outside`` outside,
    reproducing the sign-preserving constant extension of Eq. (5.1).  Reconstruction is
    therefore exact in the band and sign-exact everywhere - which is all the surfel
    projection and any mask/volume measurement ever need.
    """
    nx, ny, nz = grid.shape
    n = nx * ny * nz
    sign = _unpack_bits(packed.sign_mask.to(device or "cpu"), n).reshape(nx, ny, nz)
    phi = torch.where(
        sign,
        torch.full((nx, ny, nz), packed.fill_outside, device=sign.device, dtype=dtype),
        torch.full((nx, ny, nz), -packed.fill_outside, device=sign.device, dtype=dtype),
    )
    if packed.values.numel() == 0:
        return phi

    lo = packed.bbox_lo.tolist()
    shp = packed.bbox_shape.tolist()
    box_n = int(shp[0] * shp[1] * shp[2])
    mask = _unpack_bits(packed.bitmask.to(phi.device), box_n).reshape(*shp)

    sub = phi[lo[0] : lo[0] + shp[0], lo[1] : lo[1] + shp[1], lo[2] : lo[2] + shp[2]].clone()
    sub_flat = sub.reshape(-1)
    sub_flat[mask.reshape(-1)] = packed.values.to(device=phi.device, dtype=dtype)
    phi[lo[0] : lo[0] + shp[0], lo[1] : lo[1] + shp[1], lo[2] : lo[2] + shp[2]] = sub_flat.reshape(*shp)
    return phi


def _surfel_state(surfels: SurfelSet2D, *, minimal: bool = True) -> dict[str, Tensor]:
    """Serialise the canonical surfels.

    With ``minimal=True`` the second tangent axis is dropped, since
    :math:`e_2 = n \\times e_1` and :math:`n` follows from
    :math:`\\nabla_h\\phi_0` - matching ``StorageCount.p2d_minimal``.
    """
    state = {
        "anchor": surfels.anchor.detach().cpu(),
        "e1": surfels.e1.detach().cpu(),
        "scale": surfels.scale.detach().cpu(),
        "amplitude": surfels.amplitude.detach().cpu(),
        "opacity": surfels.opacity.detach().cpu(),
    }
    if not minimal:
        state["e2"] = surfels.e2.detach().cpu()
        state["normal"] = surfels.normal.detach().cpu()
    return state


@torch.no_grad()
def save_model(
    model: PrecomputedModel,
    path: str | Path,
    *,
    minimal_surfels: bool = True,
) -> dict[str, Any]:
    """Write :math:`\\mathcal{D}` to disk and report the measured size.

    Returns a dict with ``path``, ``file_bytes`` and a per-component byte breakdown.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    band_mm = model.config.narrow_band_store_mm

    bands = [pack_narrow_band(phi, band_mm) for phi in model.phis]
    payload: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "grid": {
            "shape": list(model.grid.shape),
            "spacing": list(model.grid.spacing),
            "origin": list(model.grid.origin),
        },
        "config_name": model.config.name,
        "band_mm": band_mm,
        "surfels": _surfel_state(model.surfels, minimal=minimal_surfels),
        "bands": [
            {
                "bbox_lo": b.bbox_lo,
                "bbox_shape": b.bbox_shape,
                "bitmask": b.bitmask,
                "values": b.values,
                "fill_outside": b.fill_outside,
                "sign_mask": b.sign_mask,
            }
            for b in bands
        ],
    }

    if model.lowrank is not None:
        payload["residual_mode"] = "lowrank"
        payload["residual"] = {
            "u": model.lowrank.u.detach().cpu(),
            "z": model.lowrank.z.detach().cpu(),
            "rank": model.lowrank.rank,
        }
    else:
        payload["residual_mode"] = "dense"
        payload["residual"] = {
            "delta": torch.stack([r.detach().cpu() for r in model.residuals], dim=0)
        }

    torch.save(payload, p)
    file_bytes = p.stat().st_size

    surfel_bytes = sum(int(v.numel() * v.element_size()) for v in payload["surfels"].values())
    band_bytes = sum(b.nbytes() for b in bands)
    if model.lowrank is not None:
        res_bytes = int(
            model.lowrank.u.numel() * 4 + model.lowrank.z.numel() * 4
        )
    else:
        res_bytes = int(sum(r.numel() * 4 for r in model.residuals))

    return {
        "path": str(p),
        "file_bytes": int(file_bytes),
        "file_mb": file_bytes / 1024**2,
        "component_bytes": {
            "surfels": surfel_bytes,
            "surfaces": band_bytes,
            "residual": res_bytes,
        },
        "serialisation_overhead_bytes": int(file_bytes - surfel_bytes - band_bytes - res_bytes),
    }


@torch.no_grad()
def load_model(path: str | Path, *, device=None, dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    """Read back a saved representation.

    Returns the raw payload with the level sets reconstructed, rather than a full
    :class:`PrecomputedModel`, because the diagnostics in that object are not stored -
    only what Eq. (6) says is stored.  That asymmetry is deliberate: it makes it
    impossible to accidentally evaluate using information the format does not carry.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if int(payload.get("format_version", 0)) != _FORMAT_VERSION:
        raise ValueError(
            f"unsupported format version {payload.get('format_version')!r}; "
            f"this build writes v{_FORMAT_VERSION}"
        )
    g = payload["grid"]
    grid = Grid(
        shape=tuple(int(x) for x in g["shape"]),
        spacing=tuple(float(x) for x in g["spacing"]),
        origin=tuple(float(x) for x in g["origin"]),
    )
    phis = [
        unpack_narrow_band(
            PackedBand(
                bbox_lo=b["bbox_lo"],
                bbox_shape=b["bbox_shape"],
                bitmask=b["bitmask"],
                values=b["values"],
                fill_outside=float(b["fill_outside"]),
                sign_mask=b["sign_mask"],
            ),
            grid,
            device=device,
            dtype=dtype,
        )
        for b in payload["bands"]
    ]
    return {
        "grid": grid,
        "phis": phis,
        "surfels": {k: v.to(device=device) for k, v in payload["surfels"].items()},
        "residual_mode": payload["residual_mode"],
        "residual": payload["residual"],
        "config_name": payload.get("config_name", ""),
        "band_mm": float(payload.get("band_mm", 0.0)),
    }


@torch.no_grad()
def model_storage_report(
    model: PrecomputedModel,
    *,
    surface_mode: str = "narrow_band_sdf",
    minimal_p2d: bool = True,
    include_mesh: bool = True,
) -> dict[str, Any]:
    """Eq. (36)-(38) for a precomputed model, in every surface storage mode.

    ``minimal_p2d`` selects :attr:`StorageCount.p2d_minimal` over
    :attr:`StorageCount.p2d_explicit`.  The same :math:`P_{2D}` is used for both
    :math:`S_{\\mathrm{full}}` and the canonical term, so the ratio is unaffected by
    the choice - but both are reported so the reader can check that.
    """
    count = model.surfels.storage()
    p2d = count.p2d_minimal if minimal_p2d else count.p2d_explicit

    per_mode: dict[str, list[int]] = {"narrow_band_sdf": [], "mask": [], "mesh": []}
    for phi in model.phis:
        mesh = marching_tetrahedra(phi, model.grid) if include_mesh else None
        store = surface_storage(
            phi, model.grid, band_mm=model.config.narrow_band_store_mm, mesh=mesh
        )
        for k, v in store.items():
            per_mode[k].append(v.total_bytes)

    reports: dict[str, StorageReport] = {}
    for mode, sizes in per_mode.items():
        if not sizes:
            continue
        reports[mode] = storage_report(
            n_frames=model.n_frames,
            n_surfels=model.surfels.n,
            channels=model.surfels.channels,
            surface_bytes_per_frame=sizes,
            p2d=p2d,
            p_residual=count.p_residual,
            lowrank_rank=None if model.lowrank is None else model.lowrank.rank,
            surface_mode=mode,
        )

    return {
        "p2d_minimal": count.p2d_minimal,
        "p2d_explicit": count.p2d_explicit,
        "p2d_used": p2d,
        "p_residual": count.p_residual,
        "by_surface_mode": {k: v.to_dict() for k, v in reports.items()},
        "primary": reports[surface_mode].to_dict() if surface_mode in reports else {},
    }
