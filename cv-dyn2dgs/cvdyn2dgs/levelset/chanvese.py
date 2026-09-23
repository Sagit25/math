"""Spacing-aware 3-D Chan-Vese with narrow-band warm starting.

Implements the alternating minimisation of theory §4:

1. fix :math:`\\phi_t`, solve for the region means in closed form (Eq. 4.3-4.4);
2. fix the means, take an explicit Euler step of the gradient flow (Eq. 4.16).

and the warm-start initialisation of Eq. (5.1),
:math:`\\phi_t^{(0)} = \\phi^*_{t-1}` restricted to :math:`\\{|\\phi^*_{t-1}| < b\\}`.

How the narrow band is realised
-------------------------------
A textbook narrow band keeps a sparse list of active voxels.  On a GPU that is
usually *slower* than dense arithmetic on a small box, so here the band is
realised as

* a dense **crop** around the bounding box of :math:`\\{|\\phi| < b\\}` plus a
  finite-difference halo - this is where the actual speed-up comes from, and
* a **masked update** inside that crop, so voxels outside the band keep their
  values (the sign-preserving constant extension of Eq. 5.1).

The region means of Eq. (4.3)-(4.4) are integrals over all of :math:`\\Omega`, so
they are deliberately recomputed on the **full** volume.  Computing them on the
crop would shrink the exterior to a thin shell and bias
:math:`c^{\\mathrm{out}}_t`.

Step size
---------
Eq. (4.16) is an explicit scheme, so the step is chosen adaptively as

    ``dt = cfl * h_min / max|speed|``

rather than fixed.  Lemma 5.4's linear-convergence rate assumes
:math:`\\alpha \\le 1/L`; a fixed step that violates it would make the measured
iteration counts of RQ1 meaningless.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from ..core.config import ChanVeseConfig
from ..core.grid import Grid
from .operators import (
    chanvese_energy,
    chanvese_speed,
    narrow_band_mask,
    region_means,
)
from .sdf import reinitialize

__all__ = ["ChanVeseResult", "solve_frame", "solve_sequence", "SequenceResult"]


@dataclass
class ChanVeseResult:
    """Outcome of solving one frame."""

    phi: Tensor
    iterations: int
    converged: bool
    c_in: float
    c_out: float
    time_ms: float
    energy_history: list[float] = field(default_factory=list)
    band_voxels: int = 0
    crop_voxels: int = 0
    full_voxels: int = 0
    iters_to_reference: int | None = None
    """Iteration at which ``||phi - phi_ref||`` first fell below the requested
    tolerance.  This is the :math:`K(e_0)` of Eq. (5.6) and is what RQ1 compares
    between warm and cold starts."""

    initial_distance_to_reference: float | None = None
    """:math:`e_0 = \\|\\phi^{(0)} - \\phi^*_t\\|` - the quantity Def. 5.1 bounds by
    :math:`\\eta` for a warm start and which is much larger for a cold start."""

    @property
    def crop_fraction(self) -> float:
        return self.crop_voxels / self.full_voxels if self.full_voxels else float("nan")


def _bbox_from_mask(mask: Tensor, pad: int) -> tuple[slice, slice, slice]:
    """Bounding box of a boolean mask, dilated by ``pad`` voxels and clipped."""
    nx, ny, nz = mask.shape
    along_x = mask.any(dim=2).any(dim=1)
    along_y = mask.any(dim=2).any(dim=0)
    along_z = mask.any(dim=1).any(dim=0)

    out: list[slice] = []
    for vec, n in ((along_x, nx), (along_y, ny), (along_z, nz)):
        nz_idx = torch.nonzero(vec, as_tuple=False)
        if nz_idx.numel() == 0:  # empty band -> use the whole axis
            out.append(slice(0, n))
            continue
        lo = int(nz_idx.min().item()) - pad
        hi = int(nz_idx.max().item()) + pad + 1
        out.append(slice(max(0, lo), min(n, hi)))
    return out[0], out[1], out[2]


def _band_l2(a: Tensor, b: Tensor, mask: Tensor, voxel_volume: float) -> float:
    """:math:`L^2(\\Omega)` distance restricted to a mask, in physical units."""
    d = (a - b)[mask]
    if d.numel() == 0:
        return 0.0
    return float(torch.sqrt((d * d).sum() * voxel_volume).item())


@torch.no_grad()
def solve_frame(
    image: Tensor,
    phi_init: Tensor,
    grid: Grid,
    cfg: ChanVeseConfig,
    *,
    spacing_aware: bool = True,
    max_iters: int | None = None,
    track_energy: bool = False,
    reference_phi: Tensor | None = None,
    tol_to_reference: float | None = None,
) -> ChanVeseResult:
    """Minimise :math:`E_{\\mathrm{CV}}` for a single frame.

    Parameters
    ----------
    image:
        ``(nx, ny, nz)`` intensity volume :math:`I_t`.  Normalise it to roughly
        ``[0, 1]`` beforehand so that ``mu`` is comparable across datasets.
    phi_init:
        Initial level set.  For :math:`t>0` this is the warm start of Eq. (5.1);
        for :math:`t=0` it comes from the ED label or an interior seed.
    reference_phi, tol_to_reference:
        If both are given, the solver records the first iteration at which the
        band-restricted :math:`L^2` distance to ``reference_phi`` drops below the
        tolerance - the empirical :math:`K(e_0)` of Eq. (5.6).

    Returns
    -------
    :class:`ChanVeseResult`.  ``phi`` is a fresh tensor; ``phi_init`` is untouched.
    """
    if image.shape != phi_init.shape:
        raise ValueError(f"image {tuple(image.shape)} and phi {tuple(phi_init.shape)} must match")
    if tuple(image.shape) != tuple(grid.shape):
        raise ValueError(f"image {tuple(image.shape)} does not match grid {grid.shape}")

    spacing = grid.spacing
    vox_vol = grid.voxel_volume_mm3 if spacing_aware else 1.0
    h_min = grid.h_min if spacing_aware else 1.0
    n_iters = int(max_iters if max_iters is not None else cfg.max_iters)

    phi = phi_init.clone()

    # ---- narrow-band crop (Eq. 5.1) --------------------------------------
    if cfg.warm_start and cfg.narrow_band_mm > 0:
        band_full = narrow_band_mask(phi, cfg.narrow_band_mm)
        halo = max(4, int(cfg.reinit_iters * cfg.cfl) + 3)
        sx, sy, sz = _bbox_from_mask(band_full, halo)
    else:
        sx, sy, sz = slice(0, grid.nx), slice(0, grid.ny), slice(0, grid.nz)
        band_full = torch.ones_like(phi, dtype=torch.bool)

    # Views into `phi`: in-place updates on the crop mutate the full volume.
    phi_c = phi[sx, sy, sz]
    img_c = image[sx, sy, sz]
    crop_voxels = int(phi_c.numel())

    t0 = time.perf_counter()

    e0: float | None = None
    iters_to_ref: int | None = None
    if reference_phi is not None:
        e0 = _band_l2(phi, reference_phi, band_full, vox_vol)

    energy_hist: list[float] = []
    converged = False
    used_iters = 0
    c_in = c_out = torch.zeros((), device=phi.device, dtype=phi.dtype)

    for it in range(n_iters):
        used_iters = it + 1

        # (1) region means on the FULL volume, Eq. (4.3)-(4.4)
        if it % cfg.check_every == 0 or it == 0:
            c_in, c_out = region_means(image, phi, cfg.eps_heaviside)

        # active band inside the crop
        band_c = narrow_band_mask(phi_c, cfg.narrow_band_mm) if cfg.narrow_band_mm > 0 else None

        # (2) gradient-flow speed, Eq. (4.7)
        speed = chanvese_speed(
            img_c,
            phi_c,
            c_in,
            c_out,
            spacing,
            mu=cfg.mu,
            lambda_in=cfg.lambda_in,
            lambda_out=cfg.lambda_out,
            eps=cfg.eps_heaviside,
            eps_div=cfg.eps_div,
            spacing_aware=spacing_aware,
        )
        if band_c is not None:
            speed = speed * band_c

        # (3) adaptive explicit Euler step, Eq. (4.16)
        smax = float(speed.abs().max().item())
        if smax <= 0.0:
            converged = True
            break
        dt = cfg.cfl * h_min / smax

        prev = phi_c.clone()
        phi_c.add_(dt * speed)

        # (4) keep phi close to an SDF so Prop. 6.2's 1-2 step projection holds
        if cfg.reinit_every > 0 and (it + 1) % cfg.reinit_every == 0:
            phi_c.copy_(
                reinitialize(phi_c, spacing, iters=cfg.reinit_iters, dt_scale=0.3)
            )

        # ---- diagnostics / stopping -------------------------------------
        if reference_phi is not None and tol_to_reference is not None and iters_to_ref is None:
            if _band_l2(phi, reference_phi, band_full, vox_vol) <= float(tol_to_reference):
                iters_to_ref = used_iters

        if (it + 1) % cfg.check_every == 0:
            if track_energy:
                energy_hist.append(
                    float(
                        chanvese_energy(
                            image,
                            phi,
                            c_in,
                            c_out,
                            spacing,
                            mu=cfg.mu,
                            lambda_in=cfg.lambda_in,
                            lambda_out=cfg.lambda_out,
                            eps=cfg.eps_heaviside,
                            spacing_aware=spacing_aware,
                            voxel_volume=vox_vol,
                        ).item()
                    )
                )
            denom = float(prev.abs().mean().item()) + 1e-12
            rel = float((phi_c - prev).abs().mean().item()) / denom
            if rel < cfg.tol_band_change:
                converged = True
                break

    dt_ms = (time.perf_counter() - t0) * 1e3

    return ChanVeseResult(
        phi=phi,
        iterations=used_iters,
        converged=converged,
        c_in=float(c_in.item()) if isinstance(c_in, Tensor) else float(c_in),
        c_out=float(c_out.item()) if isinstance(c_out, Tensor) else float(c_out),
        time_ms=dt_ms,
        energy_history=energy_hist,
        band_voxels=int(band_full.sum().item()),
        crop_voxels=crop_voxels,
        full_voxels=int(phi.numel()),
        iters_to_reference=iters_to_ref,
        initial_distance_to_reference=e0,
    )


@dataclass
class SequenceResult:
    """Per-frame level sets for one cardiac cycle plus timing bookkeeping."""

    phis: list[Tensor]
    per_frame: list[ChanVeseResult]

    @property
    def total_iterations(self) -> int:
        return sum(r.iterations for r in self.per_frame)

    @property
    def total_time_ms(self) -> float:
        return sum(r.time_ms for r in self.per_frame)

    def summary(self) -> dict[str, float]:
        return {
            "frames": float(len(self.per_frame)),
            "total_iterations": float(self.total_iterations),
            "mean_iterations": self.total_iterations / max(1, len(self.per_frame)),
            "total_time_ms": self.total_time_ms,
            "mean_time_ms": self.total_time_ms / max(1, len(self.per_frame)),
            "mean_crop_fraction": sum(r.crop_fraction for r in self.per_frame)
            / max(1, len(self.per_frame)),
        }


@torch.no_grad()
def solve_sequence(
    images: Sequence[Tensor],
    phi0_init: Tensor,
    grid: Grid,
    cfg: ChanVeseConfig,
    *,
    spacing_aware: bool = True,
    track_energy: bool = False,
) -> SequenceResult:
    """Solve the whole cardiac cycle with frame-to-frame warm starting.

    This is the "sequential 3-D Chan-Vese" of proposal §2.6 - deliberately *not*
    a coupled 4-D level set.  Frame 0 gets ``frame0_iters_scale`` times the
    iteration budget because it has no warm start available.

    When ``cfg.warm_start`` is ``False`` every frame restarts from ``phi0_init``,
    which is the cold-start baseline for RQ1.
    """
    phis: list[Tensor] = []
    results: list[ChanVeseResult] = []

    for t, img in enumerate(images):
        if t == 0:
            init = phi0_init
            budget = int(cfg.max_iters * cfg.frame0_iters_scale)
        elif cfg.warm_start:
            init = phis[-1]  # Eq. (5.1)
            budget = cfg.max_iters
        else:
            init = phi0_init
            budget = int(cfg.max_iters * cfg.frame0_iters_scale)

        res = solve_frame(
            img,
            init,
            grid,
            cfg,
            spacing_aware=spacing_aware,
            max_iters=budget,
            track_energy=track_energy,
        )
        phis.append(res.phi)
        results.append(res)

    return SequenceResult(phis=phis, per_frame=results)
