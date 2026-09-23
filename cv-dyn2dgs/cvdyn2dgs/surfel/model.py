"""The 2-D Gaussian surfel set :math:`\\mathcal{G}^{2D}_0` and its storage accounting.

Definition 7.1 of the theory:

.. math::
    \\mathcal{G}^{2D}_0 = \\bigl\\{ g^0_i = (p^0_i,\\ e^0_{i,1},\\ e^0_{i,2},\\
    s_{i,1},\\ s_{i,2},\\ a^0_i,\\ o^0_i) \\bigr\\}_{i=1}^{N}

The single most important structural property, and the one that separates this
from a 3-D Gaussian ellipsoid representation, is that **there is no
normal-direction scale**.  A surfel has exactly two tangential scales
:math:`s_{i,1}, s_{i,2}`; the normal direction is carried by the unit normal
:math:`n^t_i` and the tangent frame :math:`E^t_i`, never by a third extent.  That
is what makes the primitive dimension-matched to the 2-D surface
:math:`\\Gamma_t \\subset \\mathbb{R}^3` (theory §1, reason 1-2).

A surface point and its Gaussian weight are, by Eq. (7.5),

.. math::
    X^t_i(u) = p^t_i + E^t_i S_i u, \\qquad G_i(u) = \\exp(-\\tfrac12 u^\\top u).

Parameterisation notes
----------------------
* scales are stored as ``log_scale`` so they stay positive under gradient steps;
* opacity is stored as a logit so it stays in ``[0, 1]``;
* ``anchor``, ``e1``, ``e2`` and ``normal`` are **buffers, not parameters**.
  They are produced by the Chan-Vese surface via Eq. (6.3) and Eq. (7.6)-(7.7).
  Making them learnable would quietly turn the method into a free-form dynamic
  2DGS fit and invalidate the claim that the geometry comes from
  :math:`\\Gamma_t` alone (proposal §6.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["SurfelSet2D", "StorageCount"]


@dataclass(frozen=True)
class StorageCount:
    """Parameter counts used by the storage comparison of Eq. (36)-(38)."""

    n_surfels: int
    channels: int

    @property
    def p2d_explicit(self) -> int:
        """:math:`P_{2D}` counting every field literally: ``p, e1, e2, s, a, o``."""
        return 3 + 3 + 3 + 2 + self.channels + 1

    @property
    def p2d_minimal(self) -> int:
        """:math:`P_{2D}` exploiting redundancy.

        ``e2 = n x e1`` and ``n`` follows from :math:`\\nabla_h\\phi_t`, so the
        second tangent axis need not be stored: ``p, e1, s, a, o``.
        """
        return 3 + 3 + 2 + self.channels + 1

    @property
    def p_residual(self) -> int:
        """:math:`P_r` - a scalar amplitude residual per channel, Eq. (25)."""
        return self.channels


class SurfelSet2D(nn.Module):
    """A set of surface-aligned 2-D Gaussian disks.

    Parameters
    ----------
    anchor:
        ``(N, 3)`` disk centres :math:`p_i` in world mm.
    e1, e2:
        ``(N, 3)`` orthonormal tangent axes spanning the tangent plane.
    normal:
        ``(N, 3)`` unit normals :math:`n_i`.
    scale:
        ``(N, 2)`` tangential scales :math:`(s_{i,1}, s_{i,2})` in mm.
    amplitude:
        ``(N, C)`` appearance amplitudes :math:`a_i` (``C=1`` for grayscale MRI).
    opacity:
        ``(N,)`` opacities :math:`o_i \\in [0,1]`.
    """

    def __init__(
        self,
        anchor: Tensor,
        e1: Tensor,
        e2: Tensor,
        normal: Tensor,
        scale: Tensor,
        amplitude: Tensor,
        opacity: Tensor,
        *,
        scale_min_mm: float = 0.05,
        scale_max_mm: float = 6.0,
    ) -> None:
        super().__init__()
        n = anchor.shape[0]
        for name, t, shape in (
            ("anchor", anchor, (n, 3)),
            ("e1", e1, (n, 3)),
            ("e2", e2, (n, 3)),
            ("normal", normal, (n, 3)),
            ("scale", scale, (n, 2)),
        ):
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(t.shape)}")
        if amplitude.dim() == 1:
            amplitude = amplitude.unsqueeze(-1)
        if amplitude.shape[0] != n:
            raise ValueError("amplitude must have N rows")
        if opacity.shape != (n,):
            raise ValueError(f"opacity must be ({n},), got {tuple(opacity.shape)}")

        self.scale_min_mm = float(scale_min_mm)
        self.scale_max_mm = float(scale_max_mm)

        # Geometry: driven by the level set, never optimised (see module docstring).
        self.register_buffer("anchor", anchor.clone())
        self.register_buffer("e1", e1.clone())
        self.register_buffer("e2", e2.clone())
        self.register_buffer("normal", normal.clone())

        # Appearance / extent: fitted on frame 0 (proposal §6.6).
        clamped = scale.clamp(self.scale_min_mm, self.scale_max_mm)
        self.log_scale = nn.Parameter(torch.log(clamped))
        self.amplitude = nn.Parameter(amplitude.clone())
        self.opacity_logit = nn.Parameter(_inverse_sigmoid(opacity.clamp(1e-4, 1 - 1e-4)))

    # ------------------------------------------------------------------ sizes
    @property
    def n(self) -> int:
        return int(self.anchor.shape[0])

    @property
    def channels(self) -> int:
        return int(self.amplitude.shape[1])

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype

    # --------------------------------------------------------------- derived
    @property
    def scale(self) -> Tensor:
        """``(N, 2)`` positive tangential scales in mm."""
        return self.log_scale.exp().clamp(self.scale_min_mm, self.scale_max_mm)

    @property
    def opacity(self) -> Tensor:
        """``(N,)`` opacities in ``(0, 1)``."""
        return torch.sigmoid(self.opacity_logit)

    def frame_matrix(self) -> Tensor:
        """:math:`E^t_i = [e^t_{i,1}\\ e^t_{i,2}] \\in \\mathbb{R}^{3\\times2}`, Eq. (7.4)."""
        return torch.stack((self.e1, self.e2), dim=-1)

    def surface_point(self, u: Tensor) -> Tensor:
        """:math:`X^t_i(u) = p^t_i + E^t_i S_i u`, Eq. (7.5).

        ``u`` is ``(N, 2)`` in local disk coordinates.
        """
        s = self.scale
        return self.anchor + self.e1 * (s[:, 0:1] * u[:, 0:1]) + self.e2 * (s[:, 1:2] * u[:, 1:2])

    def disk_radius(self) -> Tensor:
        """``(N,)`` largest tangential extent, used for screen-space bounds."""
        return self.scale.max(dim=1).values

    def anisotropy_ratio(self) -> Tensor:
        """``(N,)`` :math:`\\max(s_1/s_2, s_2/s_1)`.

        Prop. 7.6 bounds the tangent-misalignment error by
        :math:`|s_1/s_2 - s_2/s_1|\\,|\\sin\\delta|`, so this is the quantity that
        says how much the transport accuracy matters for each surfel.
        """
        s = self.scale
        r = s[:, 0] / s[:, 1].clamp_min(1e-12)
        return torch.maximum(r, 1.0 / r.clamp_min(1e-12))

    # ------------------------------------------------------- geometry setters
    @torch.no_grad()
    def set_geometry(
        self,
        *,
        anchor: Tensor | None = None,
        normal: Tensor | None = None,
        e1: Tensor | None = None,
        e2: Tensor | None = None,
    ) -> None:
        """In-place geometry update from the level set (Eq. 6.3, 7.3, 7.6-7.7)."""
        if anchor is not None:
            self.anchor.copy_(anchor)
        if normal is not None:
            self.normal.copy_(normal)
        if e1 is not None:
            self.e1.copy_(e1)
        if e2 is not None:
            self.e2.copy_(e2)

    def frame_orthonormality_error(self) -> dict[str, Tensor]:
        """Residuals of the constraints in Eq. (7.4) / Lemma 7.4.

        Lemma 7.4 proves :math:`(E^t)^\\top E^t = I_2` and
        :math:`(E^t)^\\top n = 0` hold *exactly* for the ``eps = 0`` transport.
        Its remark notes that ``eps > 0`` perturbs only the column *lengths*,
        by :math:`O(\\varepsilon/\\|\\bar e\\|)`, and leaves orthogonality exact.
        These diagnostics let that prediction be checked numerically.
        """
        return {
            "e1_norm_err": (self.e1.norm(dim=-1) - 1.0).abs(),
            "e2_norm_err": (self.e2.norm(dim=-1) - 1.0).abs(),
            "e1_e2_dot": (self.e1 * self.e2).sum(-1).abs(),
            "e1_n_dot": (self.e1 * self.normal).sum(-1).abs(),
            "e2_n_dot": (self.e2 * self.normal).sum(-1).abs(),
            "n_norm_err": (self.normal.norm(dim=-1) - 1.0).abs(),
        }

    # ------------------------------------------------------------- selection
    @torch.no_grad()
    def gather(self, index: Tensor) -> "SurfelSet2D":
        """Return a new surfel set containing only ``index`` (used by pruning)."""
        return SurfelSet2D(
            self.anchor[index],
            self.e1[index],
            self.e2[index],
            self.normal[index],
            self.scale[index],
            self.amplitude.detach()[index],
            self.opacity.detach()[index],
            scale_min_mm=self.scale_min_mm,
            scale_max_mm=self.scale_max_mm,
        )

    @torch.no_grad()
    def concat(self, other: "SurfelSet2D") -> "SurfelSet2D":
        """Concatenate two sets (used by densification)."""
        return SurfelSet2D(
            torch.cat((self.anchor, other.anchor), 0),
            torch.cat((self.e1, other.e1), 0),
            torch.cat((self.e2, other.e2), 0),
            torch.cat((self.normal, other.normal), 0),
            torch.cat((self.scale, other.scale), 0),
            torch.cat((self.amplitude.detach(), other.amplitude.detach()), 0),
            torch.cat((self.opacity.detach(), other.opacity.detach()), 0),
            scale_min_mm=self.scale_min_mm,
            scale_max_mm=self.scale_max_mm,
        )

    @torch.no_grad()
    def clone_detached(self) -> "SurfelSet2D":
        return self.gather(torch.arange(self.n, device=self.device))

    # --------------------------------------------------------------- storage
    def storage(self) -> StorageCount:
        return StorageCount(n_surfels=self.n, channels=self.channels)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"N={self.n}, channels={self.channels}"


def _inverse_sigmoid(x: Tensor) -> Tensor:
    return torch.log(x / (1.0 - x))
