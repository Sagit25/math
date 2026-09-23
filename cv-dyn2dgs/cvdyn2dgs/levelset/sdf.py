"""Signed-distance utilities: reinitialisation, mask conversion, mid-time interpolation.

Two results from the theory drive this module.

1. **Prop. 3.1 (eikonal property).**  A true signed distance function satisfies
   :math:`\\|\\nabla\\phi\\| = 1`.  Prop. 6.2 leans on this to argue that the
   Newton-like projection of Eq. (6.3) reaches the surface in 1-2 steps, so the
   solver periodically reinitialises :math:`\\phi` towards an SDF.

2. **Prop. 10.1 (convex combinations are not SDFs).**  Linearly interpolating two
   level sets, Eq. (10.1), gives
   :math:`\\|\\nabla\\phi_\\tau\\|^2 = 1 - 2\\beta(1-\\beta)(1-\\cos\\omega)`,
   which is ``< 1`` unless the two gradients are aligned.  Prop. 10.2 turns that
   into a relative error injected into the projection.  :func:`interpolate_levelsets`
   therefore *returns the predicted violation alongside the measured one*, so the
   two can be compared numerically instead of taken on faith.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from .operators import (
    backward_diff,
    forward_diff,
    gradient_central,
    gradient_norm,
)

__all__ = [
    "reinitialize",
    "signed_distance_from_mask",
    "mask_from_levelset",
    "eikonal_residual",
    "interpolate_levelsets",
    "gradient_alignment_cos",
    "predicted_eikonal_norm",
]


def _godunov_grad_norm(phi: Tensor, sign: Tensor, spacing: Sequence[float]) -> Tensor:
    """Upwind :math:`\\|\\nabla\\phi\\|` for the reinitialisation PDE.

    Standard Godunov/Rouy-Tourin selection: for an outward-moving front
    (``sign > 0``) take ``max(D^-^+, D^+^-)``, and mirrored for ``sign < 0``.
    Using the naive central difference here makes the PDE unstable.
    """
    total = torch.zeros_like(phi)
    pos = sign > 0
    for axis in range(3):
        a = backward_diff(phi, axis, spacing)  # D^-
        b = forward_diff(phi, axis, spacing)  # D^+
        ap, bm = a.clamp_min(0.0), b.clamp_max(0.0)
        am, bp = a.clamp_max(0.0), b.clamp_min(0.0)
        fwd = torch.maximum(ap * ap, bm * bm)
        bwd = torch.maximum(am * am, bp * bp)
        total = total + torch.where(pos, fwd, bwd)
    return torch.sqrt(total.clamp_min(0.0) + 1e-20)


def reinitialize(
    phi: Tensor,
    spacing: Sequence[float],
    *,
    iters: int = 8,
    dt_scale: float = 0.3,
    freeze_sign: bool = True,
) -> Tensor:
    """Drive :math:`\\phi` towards a signed distance function without moving
    :math:`\\Gamma = \\{\\phi = 0\\}`.

    Solves :math:`\\partial_\\tau \\phi = S(\\phi_0)\\,(1 - \\|\\nabla\\phi\\|)` to
    steady state, where :math:`S` is a smoothed sign.  Smoothing the sign over one
    voxel is what keeps the zero level set pinned; a hard ``sign()`` lets the
    interface drift.

    Parameters
    ----------
    dt_scale:
        CFL factor; ``dt = dt_scale * min(h)``.  Must stay below ~0.5 for the
        upwind scheme to be stable.
    freeze_sign:
        Use the sign of the *initial* :math:`\\phi` throughout (recommended).
    """
    if iters <= 0:
        return phi
    h_min = min(float(s) for s in spacing)
    dt = float(dt_scale) * h_min
    phi0 = phi
    # Smoothed sign, Sussman et al.: S = phi / sqrt(phi^2 + h^2)
    sign0 = phi0 / torch.sqrt(phi0 * phi0 + h_min * h_min)

    out = phi.clone()
    for _ in range(int(iters)):
        sign = sign0 if freeze_sign else out / torch.sqrt(out * out + h_min * h_min)
        gnorm = _godunov_grad_norm(out, sign, spacing)
        out = out + dt * sign * (1.0 - gnorm)
    return out


def signed_distance_from_mask(
    mask: Tensor,
    spacing: Sequence[float],
    *,
    max_dist_mm: float | None = None,
    iters: int | None = None,
    dt_scale: float = 0.3,
) -> Tensor:
    """Build :math:`\\phi` from a binary mask with the paper's sign convention.

    ``phi > 0`` inside (``mask == True``), ``phi < 0`` outside (theory §2.1).

    The interface is placed half a voxel outside the last foreground voxel, then
    the reinitialisation PDE propagates distances outward.  Accuracy is only
    guaranteed within ``max_dist_mm`` of the surface, which is all the narrow-band
    solver and the surfel projection ever read.

    Notes
    -----
    This is an *approximate* Euclidean distance.  When SciPy is installed the
    tests cross-check it against ``scipy.ndimage.distance_transform_edt`` with
    anisotropic ``sampling=spacing``; agreement is expected to within one voxel
    near the surface, which is the region that matters.
    """
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    h = [float(s) for s in spacing]
    h_min = min(h)
    if max_dist_mm is None:
        max_dist_mm = 12.0 * h_min
    if iters is None:
        iters = int(math.ceil(2.5 * max_dist_mm / (dt_scale * h_min)))

    dtype = torch.float32
    inside = mask.to(dtype)
    sign = inside * 2.0 - 1.0  # +1 inside, -1 outside

    # Boundary voxels: a 6-neighbour has the opposite label.
    half = torch.full_like(inside, float(max_dist_mm))
    is_bnd = torch.zeros_like(mask)
    for axis in range(3):
        d = -3 + axis
        n = mask.shape[d]
        fwd_diff = mask.narrow(d, 1, n - 1) != mask.narrow(d, 0, n - 1)
        pad_hi = [1, 1, 1]
        pad_hi[axis] = 1
        # mark both sides of every sign change along this axis
        lo = torch.zeros_like(mask)
        hi = torch.zeros_like(mask)
        lo.narrow(d, 0, n - 1).copy_(fwd_diff)
        hi.narrow(d, 1, n - 1).copy_(fwd_diff)
        changed = lo | hi
        is_bnd |= changed
        half = torch.where(changed, torch.minimum(half, torch.full_like(half, 0.5 * h[axis])), half)

    phi = sign * torch.where(is_bnd, half, torch.full_like(half, float(max_dist_mm)))
    phi = reinitialize(phi, spacing, iters=iters, dt_scale=dt_scale, freeze_sign=True)
    return phi.clamp(-float(max_dist_mm), float(max_dist_mm))


def mask_from_levelset(phi: Tensor) -> Tensor:
    """``phi > 0`` -> inside (theory §2.1, proposal Eq. 1)."""
    return phi > 0


def eikonal_residual(phi: Tensor, spacing: Sequence[float]) -> Tensor:
    """:math:`\\|\\nabla_h \\phi\\| - 1` - how far :math:`\\phi` is from an SDF."""
    return gradient_norm(gradient_central(phi, spacing)) - 1.0


# --------------------------------------------------------------------------- #
#  Mid-time interpolation (proposal Eq. 35, theory §10)
# --------------------------------------------------------------------------- #
def gradient_alignment_cos(
    phi_a: Tensor, phi_b: Tensor, spacing: Sequence[float], *, eps: float = 1e-12
) -> Tensor:
    """:math:`\\cos\\omega` between :math:`\\nabla\\phi_t` and :math:`\\nabla\\phi_{t+1}`.

    This is the quantity that controls the eikonal violation in Prop. 10.1.
    """
    ga = gradient_central(phi_a, spacing)
    gb = gradient_central(phi_b, spacing)
    na = gradient_norm(ga).clamp_min(eps)
    nb = gradient_norm(gb).clamp_min(eps)
    return ((ga * gb).sum(dim=0) / (na * nb)).clamp(-1.0, 1.0)


def predicted_eikonal_norm(cos_omega: Tensor, beta: float) -> Tensor:
    """Prop. 10.1, Eq. (10.2):
    :math:`\\|\\nabla\\phi_\\tau\\|^2 = 1 - 2\\beta(1-\\beta)(1-\\cos\\omega)`.

    Returns the predicted **norm** (square root of the above), assuming both
    inputs were exact SDFs.
    """
    b = float(beta)
    sq = 1.0 - 2.0 * b * (1.0 - b) * (1.0 - cos_omega)
    return torch.sqrt(sq.clamp_min(0.0))


def interpolate_levelsets(
    phi_t: Tensor,
    phi_next: Tensor,
    beta: float,
    spacing: Sequence[float],
    *,
    residual_delta: Tensor | None = None,
    residual_next: Tensor | None = None,
    diagnose: bool = False,
) -> tuple[Tensor, Tensor | None, dict[str, Tensor] | None]:
    """Linear mid-time interpolation, Eq. (10.1) / proposal Eq. (35).

    .. math::
        \\phi_\\tau = (1-\\beta)\\phi_t + \\beta\\phi_{t+1}, \\qquad
        \\Delta a_\\tau = (1-\\beta)\\Delta a_t + \\beta \\Delta a_{t+1}

    **This is display-only.**  Theory §10.3 proves the interpolated level set is
    not the SDF of the true intermediate surface, so the result must not be read
    as a clinical reconstruction of an unobserved phase.  Setting ``diagnose=True``
    returns the measured eikonal norm next to the value Prop. 10.1 predicts, plus
    the relative projection error of Eq. (10.3).

    Returns
    -------
    ``(phi_tau, residual_tau, diagnostics_or_None)``
    """
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError(f"beta must lie in [0,1], got {beta}")
    b = float(beta)
    phi_tau = (1.0 - b) * phi_t + b * phi_next

    res_tau: Tensor | None = None
    if residual_delta is not None and residual_next is not None:
        res_tau = (1.0 - b) * residual_delta + b * residual_next

    diag: dict[str, Tensor] | None = None
    if diagnose:
        measured = gradient_norm(gradient_central(phi_tau, spacing))
        cos_omega = gradient_alignment_cos(phi_t, phi_next, spacing)
        predicted = predicted_eikonal_norm(cos_omega, b)
        # Eq. (10.3): relative error left in one projection step.
        denom = measured.clamp_min(1e-12) ** 2
        rel = ((measured**2 - 1.0).abs() / denom)
        diag = {
            "eikonal_norm_measured": measured,
            "eikonal_norm_predicted": predicted,
            "cos_omega": cos_omega,
            "projection_relative_error": rel,
        }
    return phi_tau, res_tau, diag
