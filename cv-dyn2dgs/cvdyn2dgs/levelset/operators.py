"""Spacing-aware differential operators and Chan-Vese terms.

This module implements the paper's **first theoretical contribution**: finite
difference operators that carry the anisotropic voxel spacing
:math:`h = (h_x, h_y, h_z)` explicitly (theory §4.4, Eq. 4.12-4.15).

Why it matters: short-axis cine CMR has :math:`h_z \\gg h_x \\approx h_y`.  A
spacing-agnostic stencil is equivalent to setting :math:`h_x=h_y=h_z=1`, which
over-weights the slice direction and biases the curvature-based regulariser
along :math:`z`.  The ``spacing_aware=False`` switch on :func:`gradient_central`
and :func:`curvature` exists purely so that this bias can be *measured* in the
ablation of proposal §8.4 rather than asserted.

Boundary conditions are the natural/Neumann conditions of Eq. (4.6), realised as
replicate padding: the first-order difference across the outer face is zero.

Discretisation choices
----------------------
* Normals (Eq. 7.3) and the projection step (Eq. 6.2) use the **central**
  gradient of Eq. (4.14), whose truncation error is :math:`O(h_{\\max}^2)` -
  exactly the rate Prop. 7.3 predicts for the normal angular error.
* The curvature term of Eq. (4.7) uses **forward** differences to build the
  normalised gradient field and **backward** differences for the divergence.
  Theory §4.4 specifies this pairing because the resulting discrete
  integration-by-parts identity preserves the sign structure of Eq. (4.9).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "heaviside_eps",
    "dirac_eps",
    "forward_diff",
    "backward_diff",
    "central_diff",
    "gradient_central",
    "gradient_forward",
    "gradient_norm",
    "divergence_backward",
    "curvature",
    "region_means",
    "chanvese_energy",
    "chanvese_speed",
    "narrow_band_mask",
]

# Spatial dims are always the trailing three: (..., nx, ny, nz)
_SPATIAL = (-3, -2, -1)


def _spacing(spacing: Sequence[float], axis: int, spacing_aware: bool) -> float:
    return float(spacing[axis]) if spacing_aware else 1.0


def _pad_replicate(t: Tensor, pads: Sequence[tuple[int, int]]) -> Tensor:
    """Replicate-pad the trailing three dims. ``pads`` is ``((x0,x1),(y0,y1),(z0,z1))``."""
    nx, ny, nz = t.shape[-3:]
    lead = t.shape[:-3]
    x = t.reshape(1, -1, nx, ny, nz)
    # F.pad orders the tuple from the LAST dim backwards.
    p = (pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1])
    y = F.pad(x, p, mode="replicate")
    return y.reshape(*lead, *y.shape[-3:])


# --------------------------------------------------------------------------- #
#  Smooth Heaviside / Dirac (theory Eq. 3.7-3.8)
# --------------------------------------------------------------------------- #
def heaviside_eps(z: Tensor, eps: float) -> Tensor:
    """:math:`H_\\varepsilon(z) = \\tfrac12\\left(1 + \\tfrac{2}{\\pi}\\arctan(z/\\varepsilon)\\right)`.

    Prop. 3.2: smooth, strictly increasing, valued in ``(0, 1)`` for every
    ``eps > 0``.  Strict positivity is what makes the denominators of Eq. (4.3)
    and (4.4) non-zero.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    return 0.5 * (1.0 + (2.0 / math.pi) * torch.atan(z / eps))


def dirac_eps(z: Tensor, eps: float) -> Tensor:
    """:math:`\\delta_\\varepsilon(z) = H'_\\varepsilon(z) = \\tfrac{1}{\\pi}\\tfrac{\\varepsilon}{\\varepsilon^2+z^2}`.

    Prop. 3.2 (2): integrates to 1 for every ``eps > 0``.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    return (eps / math.pi) / (eps * eps + z * z)


# --------------------------------------------------------------------------- #
#  Finite differences (theory Eq. 4.12-4.14)
# --------------------------------------------------------------------------- #
def forward_diff(vol: Tensor, axis: int, spacing: Sequence[float], *, spacing_aware: bool = True) -> Tensor:
    """:math:`D^+_a \\phi = (\\phi_{a+1} - \\phi_a) / h_a`, Eq. (4.12)."""
    h = _spacing(spacing, axis, spacing_aware)
    n = vol.shape[_SPATIAL[axis]]
    pads = [(0, 0), (0, 0), (0, 0)]
    pads[axis] = (0, 1)
    v = _pad_replicate(vol, pads)
    d = _SPATIAL[axis]
    return (v.narrow(d, 1, n) - v.narrow(d, 0, n)) / h


def backward_diff(vol: Tensor, axis: int, spacing: Sequence[float], *, spacing_aware: bool = True) -> Tensor:
    """:math:`D^-_a \\phi = (\\phi_a - \\phi_{a-1}) / h_a`, Eq. (4.13)."""
    h = _spacing(spacing, axis, spacing_aware)
    n = vol.shape[_SPATIAL[axis]]
    pads = [(0, 0), (0, 0), (0, 0)]
    pads[axis] = (1, 0)
    v = _pad_replicate(vol, pads)
    d = _SPATIAL[axis]
    return (v.narrow(d, 1, n) - v.narrow(d, 0, n)) / h


def central_diff(vol: Tensor, axis: int, spacing: Sequence[float], *, spacing_aware: bool = True) -> Tensor:
    """:math:`(\\phi_{a+1} - \\phi_{a-1}) / (2 h_a)`, Eq. (4.14). Truncation ``O(h^2)``."""
    h = _spacing(spacing, axis, spacing_aware)
    n = vol.shape[_SPATIAL[axis]]
    pads = [(0, 0), (0, 0), (0, 0)]
    pads[axis] = (1, 1)
    v = _pad_replicate(vol, pads)
    d = _SPATIAL[axis]
    return (v.narrow(d, 2, n) - v.narrow(d, 0, n)) / (2.0 * h)


def gradient_central(phi: Tensor, spacing: Sequence[float], *, spacing_aware: bool = True) -> Tensor:
    """:math:`\\nabla_h \\phi` by central differences -> ``(3, nx, ny, nz)``, Eq. (4.14).

    Used for normals (Eq. 7.3) and for the Newton-like projection (Eq. 6.2-6.3),
    because its :math:`O(h_{\\max}^2)` error is what bounds the normal angular
    error in Prop. 7.3, Eq. (7.9).
    """
    return torch.stack(
        [central_diff(phi, a, spacing, spacing_aware=spacing_aware) for a in range(3)],
        dim=0,
    )


def gradient_forward(phi: Tensor, spacing: Sequence[float], *, spacing_aware: bool = True) -> Tensor:
    """:math:`\\nabla_h \\phi` by forward differences -> ``(3, nx, ny, nz)``, Eq. (4.12)."""
    return torch.stack(
        [forward_diff(phi, a, spacing, spacing_aware=spacing_aware) for a in range(3)],
        dim=0,
    )


def gradient_norm(grad: Tensor, eps: float = 0.0) -> Tensor:
    """:math:`\\|\\nabla_h\\phi\\|_2` from a ``(3, ...)`` gradient stack."""
    return torch.sqrt((grad * grad).sum(dim=0).clamp_min(0.0) + eps * eps)


def divergence_backward(
    vec: Tensor, spacing: Sequence[float], *, spacing_aware: bool = True
) -> Tensor:
    """:math:`\\mathrm{div}_h V` by backward differences, Eq. (4.15).

    ``vec`` is ``(3, nx, ny, nz)``.
    """
    if vec.shape[0] != 3:
        raise ValueError(f"expected (3,nx,ny,nz), got {tuple(vec.shape)}")
    out = backward_diff(vec[0], 0, spacing, spacing_aware=spacing_aware)
    out = out + backward_diff(vec[1], 1, spacing, spacing_aware=spacing_aware)
    out = out + backward_diff(vec[2], 2, spacing, spacing_aware=spacing_aware)
    return out


def curvature(
    phi: Tensor,
    spacing: Sequence[float],
    *,
    eps: float = 1e-6,
    spacing_aware: bool = True,
) -> Tensor:
    """:math:`\\kappa_h = \\mathrm{div}_h\\!\\left(\\nabla_h\\phi / (\\|\\nabla_h\\phi\\| + \\varepsilon)\\right)`.

    The regularised denominator is Eq. (4.7)'s :math:`\\|\\nabla_h\\phi\\| +
    \\varepsilon`, which prevents blow-up where the gradient vanishes.  Forward
    differences build the field, backward differences take the divergence
    (theory §4.4).
    """
    g = gradient_forward(phi, spacing, spacing_aware=spacing_aware)
    nrm = gradient_norm(g) + eps
    return divergence_backward(g / nrm, spacing, spacing_aware=spacing_aware)


# --------------------------------------------------------------------------- #
#  Chan-Vese terms (theory Eq. 4.1, 4.3, 4.4, 4.7)
# --------------------------------------------------------------------------- #
def region_means(
    image: Tensor,
    phi: Tensor,
    eps: float,
    *,
    weight: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimal region means :math:`c^{\\mathrm{in}}_t, c^{\\mathrm{out}}_t`, Eq. (4.3)-(4.4).

    Closed form because the energy is quadratic in the means with :math:`\\phi`
    fixed; the second derivative :math:`2\\lambda\\int H_\\varepsilon > 0` makes
    them minimisers (theory §4.2).

    Parameters
    ----------
    weight:
        Optional domain restriction.  When the solver runs on a cropped box the
        means must still be computed over the region of interest; passing a
        weight keeps that explicit.
    """
    h_in = heaviside_eps(phi, eps)
    h_out = 1.0 - h_in
    if weight is not None:
        h_in = h_in * weight
        h_out = h_out * weight
    tiny = torch.finfo(image.dtype).tiny
    c_in = (image * h_in).sum() / h_in.sum().clamp_min(tiny)
    c_out = (image * h_out).sum() / h_out.sum().clamp_min(tiny)
    return c_in, c_out


def chanvese_energy(
    image: Tensor,
    phi: Tensor,
    c_in: Tensor | float,
    c_out: Tensor | float,
    spacing: Sequence[float],
    *,
    mu: float,
    lambda_in: float,
    lambda_out: float,
    eps: float,
    spacing_aware: bool = True,
    voxel_volume: float | None = None,
) -> Tensor:
    """Chan-Vese energy :math:`E_{\\mathrm{CV}}`, Eq. (4.1).

    Returned as a physical quantity: the integral is discretised with the voxel
    volume :math:`h_x h_y h_z`, so energies from grids of different resolution
    are comparable.  This is what the warm-start experiment of RQ1 monitors, and
    what Lemma 5.2's dissipation statement applies to.
    """
    if voxel_volume is None:
        voxel_volume = float(spacing[0]) * float(spacing[1]) * float(spacing[2]) if spacing_aware else 1.0
    g = gradient_central(phi, spacing, spacing_aware=spacing_aware)
    gnorm = gradient_norm(g)
    h_in = heaviside_eps(phi, eps)

    length = mu * (dirac_eps(phi, eps) * gnorm).sum()
    fid_in = lambda_in * (((image - c_in) ** 2) * h_in).sum()
    fid_out = lambda_out * (((image - c_out) ** 2) * (1.0 - h_in)).sum()
    return (length + fid_in + fid_out) * voxel_volume


def chanvese_speed(
    image: Tensor,
    phi: Tensor,
    c_in: Tensor | float,
    c_out: Tensor | float,
    spacing: Sequence[float],
    *,
    mu: float,
    lambda_in: float,
    lambda_out: float,
    eps: float,
    eps_div: float = 1e-6,
    spacing_aware: bool = True,
    return_parts: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Right-hand side of the gradient flow, Eq. (4.7).

    .. math::
        \\frac{\\partial \\phi_t}{\\partial s} =
        \\delta_\\varepsilon(\\phi_t)\\left[
            \\mu\\,\\mathrm{div}_h\\!\\left(\\frac{\\nabla_h\\phi_t}{\\|\\nabla_h\\phi_t\\|+\\varepsilon}\\right)
            - \\lambda_{\\mathrm{in}}(I_t - c^{\\mathrm{in}}_t)^2
            + \\lambda_{\\mathrm{out}}(I_t - c^{\\mathrm{out}}_t)^2
        \\right]

    Sign check (the convention trips people up): with :math:`\\phi>0` inside, a
    voxel that sits inside but whose intensity matches :math:`c^{\\mathrm{out}}`
    better than :math:`c^{\\mathrm{in}}` gives a negative bracket, so
    :math:`\\phi` decreases and the voxel leaves the interior.  That is the
    desired direction.
    """
    kappa = curvature(phi, spacing, eps=eps_div, spacing_aware=spacing_aware)
    data_in = lambda_in * (image - c_in) ** 2
    data_out = lambda_out * (image - c_out) ** 2
    bracket = mu * kappa - data_in + data_out
    speed = dirac_eps(phi, eps) * bracket
    if return_parts:
        return speed, {
            "curvature": kappa,
            "bracket": bracket,
            "data_in": data_in,
            "data_out": data_out,
        }
    return speed


def narrow_band_mask(phi: Tensor, band_mm: float) -> Tensor:
    """:math:`\\{|\\phi| < b\\}` - the active band of the warm start, Eq. (5.1)."""
    return phi.abs() < float(band_mm)
