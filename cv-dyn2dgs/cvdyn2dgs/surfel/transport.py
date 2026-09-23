"""Tangent-frame construction and frame-to-frame transport.

The unit normal fixes the tangent *plane* but not the orientation of the axes
inside it - a gauge freedom.  Theory §7.3 resolves it with a minimum-rotation
transport, Eq. (7.6)-(7.7):

.. math::
    \\bar e^t_{i,1} = e^{t-1}_{i,1} - (e^{t-1}_{i,1}\\cdot n^t_i)\\,n^t_i, \\qquad
    e^t_{i,1} = \\frac{\\bar e^t_{i,1}}{\\|\\bar e^t_{i,1}\\| + \\varepsilon}, \\qquad
    e^t_{i,2} = n^t_i \\times e^t_{i,1}.

Three theory results are implemented here as executable checks:

* **Lemma 7.4** - for ``eps = 0`` the result is an exact right-handed orthonormal
  frame; for ``eps > 0`` only the column *lengths* drift, by
  :math:`O(\\varepsilon/\\|\\bar e\\|)`, while orthogonality stays exact.
* **Prop. 7.7** - :math:`\\|\\bar e^t_{i,1}\\| = |\\sin\\psi|`, and a perturbation
  :math:`\\gamma` is amplified to :math:`\\gamma/|\\sin\\psi|`.  When
  :math:`|\\sin\\psi|` is small the transport is switched to the *second* axis.
  That fallback is always available: for an orthonormal pair,
  :math:`\\|\\bar e_1\\|^2 + \\|\\bar e_2\\|^2 \\ge 1`, so at least one axis has
  :math:`\\|\\bar e\\| \\ge 1/\\sqrt2`.
* **Prop. 7.5** - for isotropic disks the in-plane gauge is irrelevant, which is
  why ``v1-minimal`` can ignore transport entirely.

``mode="rodrigues"`` offers an alternative with **no degeneracy at all**: rotate
the whole previous frame by the minimal rotation carrying :math:`n^{t-1}` to
:math:`n^t`.  It is mathematically the same "minimum rotation" idea expressed as
a group action rather than a Gram-Schmidt projection, and its conditioning does
not depend on :math:`\\sin\\psi`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

__all__ = [
    "TransportDiagnostics",
    "initial_tangent_frame",
    "transport_tangent_frame",
    "minimal_rotation_matrix",
]

TransportMode = Literal["gram_schmidt", "rodrigues"]


@dataclass
class TransportDiagnostics:
    """Per-surfel quantities predicted by Prop. 7.7 and Lemma 7.4."""

    sin_psi: Tensor
    """:math:`\\|\\bar e^t_{i,1}\\| = |\\sin\\psi|`, Eq. (7.12)."""

    amplification: Tensor
    """:math:`1/(\\|\\bar e\\| + \\varepsilon)` - the bound of Eq. (7.13), capped
    at :math:`1/\\varepsilon` exactly as the remark to Prop. 7.7 describes."""

    fallback_mask: Tensor
    """Surfels where the first axis was degenerate and the second was used."""

    rotation_angle_rad: Tensor
    """Angle between :math:`e^{t-1}_{i,1}` and :math:`e^t_{i,1}` - the per-frame
    :math:`\\theta_t` whose sum bounds the orientation drift of Prop. 11.1,
    Eq. (11.1)."""

    def summary(self) -> dict[str, float]:
        return {
            "sin_psi_min": float(self.sin_psi.min().item()),
            "sin_psi_mean": float(self.sin_psi.mean().item()),
            "amplification_max": float(self.amplification.max().item()),
            "fallback_fraction": float(self.fallback_mask.to(torch.float32).mean().item()),
            "rotation_angle_mean_deg": float(
                torch.rad2deg(self.rotation_angle_rad).mean().item()
            ),
            "rotation_angle_max_deg": float(
                torch.rad2deg(self.rotation_angle_rad).max().item()
            ),
        }


def _normalize(v: Tensor, eps: float) -> Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def initial_tangent_frame(normal: Tensor, *, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Build a deterministic orthonormal tangent frame for frame 0.

    Any choice is admissible - Prop. 7.5 shows isotropic disks are invariant to
    it, and for anisotropic disks frame 0 *defines* the reference gauge that later
    frames transport.  To stay well conditioned the seed axis is taken as the
    world axis *least* aligned with the normal, so the cross product is never
    near-degenerate.

    Parameters
    ----------
    normal:
        ``(N, 3)`` unit normals.

    Returns
    -------
    ``(e1, e2)`` each ``(N, 3)``, with ``{n, e1, e2}`` right-handed orthonormal.
    """
    if normal.dim() != 2 or normal.shape[1] != 3:
        raise ValueError(f"normal must be (N,3), got {tuple(normal.shape)}")
    n = _normalize(normal, eps)

    # Pick the world axis with the smallest |n . axis|.
    a = n.abs()
    axis_id = a.argmin(dim=1)  # (N,)
    seed = torch.zeros_like(n)
    seed.scatter_(1, axis_id.unsqueeze(1), 1.0)

    e1 = torch.cross(seed, n, dim=-1)
    e1 = _normalize(e1, eps)
    e2 = torch.cross(n, e1, dim=-1)
    e2 = _normalize(e2, eps)
    return e1, e2


def minimal_rotation_matrix(n_from: Tensor, n_to: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Minimal rotation ``R`` with ``R @ n_from = n_to`` (Rodrigues).

    Returns ``(N, 3, 3)``.  The near-antipodal case (:math:`n_{to} = -n_{from}`)
    is handled by rotating :math:`\\pi` about any axis orthogonal to
    :math:`n_{from}`; that situation should not arise between adjacent cine frames
    and is guarded only for numerical safety.
    """
    a = _normalize(n_from, eps)
    b = _normalize(n_to, eps)
    v = torch.cross(a, b, dim=-1)
    s = v.norm(dim=-1, keepdim=True)
    c = (a * b).sum(dim=-1, keepdim=True)

    n_pts = a.shape[0]
    eye = torch.eye(3, device=a.device, dtype=a.dtype).expand(n_pts, 3, 3)

    # Skew-symmetric [v]_x
    zeros = torch.zeros_like(v[:, 0])
    kx = torch.stack(
        (
            torch.stack((zeros, -v[:, 2], v[:, 1]), dim=-1),
            torch.stack((v[:, 2], zeros, -v[:, 0]), dim=-1),
            torch.stack((-v[:, 1], v[:, 0], zeros), dim=-1),
        ),
        dim=-2,
    )  # (N,3,3)

    denom = (1.0 + c).clamp_min(eps).unsqueeze(-1)
    rot = eye + kx + kx @ kx / denom

    # Degenerate: antipodal (s ~ 0 and c < 0) -> rotate pi about a tangent axis.
    antipodal = (s.squeeze(-1) < 1e-6) & (c.squeeze(-1) < 0.0)
    if bool(antipodal.any()):
        t1, _ = initial_tangent_frame(a, eps=eps)
        # Householder-style pi rotation about t1: R = 2 t1 t1^T - I
        rot_anti = 2.0 * t1.unsqueeze(-1) @ t1.unsqueeze(-2) - eye
        rot = torch.where(antipodal.view(-1, 1, 1), rot_anti, rot)

    # Degenerate: identical (s ~ 0 and c > 0) -> identity
    identical = (s.squeeze(-1) < 1e-12) & (c.squeeze(-1) > 0.0)
    if bool(identical.any()):
        rot = torch.where(identical.view(-1, 1, 1), eye, rot)
    return rot


def transport_tangent_frame(
    e1_prev: Tensor,
    e2_prev: Tensor,
    normal_new: Tensor,
    *,
    normal_prev: Tensor | None = None,
    eps: float = 1e-8,
    degeneracy_thresh: float = 0.2,
    mode: TransportMode = "gram_schmidt",
    renormalize: bool = True,
) -> tuple[Tensor, Tensor, TransportDiagnostics]:
    """Transport a tangent frame onto a new tangent plane.

    Parameters
    ----------
    e1_prev, e2_prev:
        ``(N, 3)`` previous tangent axes.
    normal_new:
        ``(N, 3)`` unit normals :math:`n^t_i` from Eq. (7.3).
    normal_prev:
        Required for ``mode="rodrigues"``.
    degeneracy_thresh:
        Threshold on :math:`|\\sin\\psi|` below which the second axis is
        transported instead (remark to Prop. 7.7).
    renormalize:
        Re-divide by the achieved norm after the ``eps``-regularised step.  The
        remark to Lemma 7.4 recommends exactly this to recover unit length; the
        default is ``True`` because it costs one division and removes the
        ``O(eps/||e||)`` length drift entirely.

    Returns
    -------
    ``(e1, e2, diagnostics)``.
    """
    n = _normalize(normal_new, eps)

    if mode == "rodrigues":
        if normal_prev is None:
            raise ValueError('mode="rodrigues" needs normal_prev')
        rot = minimal_rotation_matrix(normal_prev, n, eps=eps)
        e1 = (rot @ e1_prev.unsqueeze(-1)).squeeze(-1)
        # Re-orthogonalise against the *exact* new normal to kill drift.
        e1 = e1 - (e1 * n).sum(-1, keepdim=True) * n
        bar_norm = e1.norm(dim=-1)
        e1 = _normalize(e1, eps)
        e2 = _normalize(torch.cross(n, e1, dim=-1), eps)
        fallback = torch.zeros(n.shape[0], dtype=torch.bool, device=n.device)
    elif mode == "gram_schmidt":
        # Eq. (7.6): remove the normal component from the previous first axis.
        bar1 = e1_prev - (e1_prev * n).sum(-1, keepdim=True) * n
        bar2 = e2_prev - (e2_prev * n).sum(-1, keepdim=True) * n
        n1 = bar1.norm(dim=-1)
        n2 = bar2.norm(dim=-1)

        # Prop. 7.7 remark: if axis 1 is degenerate, transport axis 2 instead.
        fallback = n1 < float(degeneracy_thresh)
        bar = torch.where(fallback.unsqueeze(-1), bar2, bar1)
        bar_norm = torch.where(fallback, n2, n1)

        # Eq. (7.7) with the eps-regularised denominator.
        e1 = bar / (bar_norm.unsqueeze(-1) + eps)
        if renormalize:
            e1 = _normalize(e1, eps)
        e2 = torch.cross(n, e1, dim=-1)
        if renormalize:
            e2 = _normalize(e2, eps)
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"unknown transport mode {mode!r}")

    # Diagnostics
    cos_rot = (e1 * e1_prev).sum(-1).clamp(-1.0, 1.0)
    diag = TransportDiagnostics(
        sin_psi=bar_norm.detach(),
        amplification=(1.0 / (bar_norm.detach() + eps)),
        fallback_mask=fallback,
        rotation_angle_rad=torch.acos(cos_rot.detach()),
    )
    return e1, e2, diag
