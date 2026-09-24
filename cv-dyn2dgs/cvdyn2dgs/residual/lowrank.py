"""Low-rank compression of the residual field, Eq. (9.5) / Prop. 9.3.

Storing a scalar residual per surfel per frame costs :math:`T N P_r` numbers, which
proposal §10.3 (limitation 5) flags as the point where the compact representation
stops being compact.  Eq. (27)/(9.5) proposes factoring it,

.. math:: \\Delta a^t_i \\approx \\sum_{\\ell=1}^{r} U_{i\\ell} z_{\\ell t},
          \\qquad r \\ll \\min(N, T),

which drops the cost to :math:`r(N + T)`.

Prop. 9.3 makes the error exact rather than heuristic: by Eckart-Young-Mirsky the
best rank-:math:`r` approximation of :math:`\\Delta A \\in \\mathbb{R}^{N\\times T}`
has

.. math:: \\min \\|\\Delta A - \\Delta A_r\\|_F^2 = \\sum_{\\ell>r}\\sigma_\\ell^2,
          \\qquad \\min\\|\\Delta A - \\Delta A_r\\|_2 = \\sigma_{r+1}.

So the compression error is *observable in advance* from the singular-value tail,
and :func:`rank_for_error` inverts the relation to pick the smallest rank meeting a
target.  Whether the tail actually decays fast for real cardiac data is an
empirical question, which is why :func:`spectrum_report` returns the full spectrum
rather than a single number.

Since :math:`T` (20-40 cine phases) is tiny compared with :math:`N`, the thin SVD
costs almost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "LowRankResidual",
    "compress_residual",
    "spectrum_report",
    "rank_for_error",
]


@dataclass
class LowRankResidual:
    """Factored residual field :math:`\\Delta A \\approx U Z`."""

    u: Tensor
    """``(N, r)`` spatial factors :math:`U_{i\\ell}`."""

    z: Tensor
    """``(r, T)`` temporal factors :math:`z_{\\ell t}`."""

    singular_values: Tensor
    """``(min(N,T),)`` full spectrum of the original matrix."""

    rank: int
    frobenius_error: float
    """:math:`\\|\\Delta A - U Z\\|_F` - measured, and compared against
    :math:`\\sqrt{\\sum_{\\ell>r}\\sigma_\\ell^2}` in the tests."""

    spectral_error: float
    """:math:`\\|\\Delta A - U Z\\|_2`, predicted to equal :math:`\\sigma_{r+1}`."""

    relative_frobenius_error: float

    def reconstruct(self) -> Tensor:
        """``(N, T)`` dense reconstruction."""
        return self.u @ self.z

    def frame(self, t: int) -> Tensor:
        """``(N,)`` residual for one frame, without materialising the full matrix."""
        return self.u @ self.z[:, t]

    def storage_floats(self) -> int:
        """``r * (N + T)`` - what actually goes to disk."""
        return int(self.rank * (self.u.shape[0] + self.z.shape[1]))

    def predicted_frobenius_error(self) -> float:
        """:math:`\\sqrt{\\sum_{\\ell>r}\\sigma_\\ell^2}` from Eq. (9.6)."""
        tail = self.singular_values[self.rank :]
        return float(torch.sqrt((tail * tail).sum()).item())

    def predicted_spectral_error(self) -> float:
        """:math:`\\sigma_{r+1}` from Eq. (9.6)."""
        if self.rank >= self.singular_values.numel():
            return 0.0
        return float(self.singular_values[self.rank].item())


def _stack(delta: Tensor | list[Tensor]) -> Tensor:
    """Normalise input to an ``(N, T)`` matrix."""
    if isinstance(delta, list):
        cols = []
        for d in delta:
            cols.append(d.reshape(-1) if d.dim() == 1 else d.reshape(d.shape[0], -1).squeeze(-1))
        return torch.stack(cols, dim=1)
    if delta.dim() == 2:
        return delta
    raise ValueError(f"expected (N,T) or a list of (N,), got {tuple(delta.shape)}")


def spectrum_report(delta: Tensor | list[Tensor]) -> dict[str, float | list[float]]:
    """Singular spectrum of the residual field, for deciding on a rank.

    Returned ``energy_fraction[k]`` is the fraction of squared Frobenius norm
    captured by the leading ``k+1`` components, i.e. exactly the complement of the
    tail in Eq. (9.6).
    """
    mat = _stack(delta)
    sv = torch.linalg.svdvals(mat.to(torch.float64))
    total = float((sv * sv).sum().item())
    cum = torch.cumsum(sv * sv, dim=0) / max(total, 1e-30)
    return {
        "singular_values": [float(v) for v in sv],
        "energy_fraction": [float(v) for v in cum],
        "total_frobenius": float(torch.sqrt((sv * sv).sum()).item()),
        # The threshold must live on the same device as cum, or searchsorted raises on GPU.
        "effective_rank_99": int(
            torch.searchsorted(
                cum, torch.tensor(0.99, dtype=cum.dtype, device=cum.device)
            ).item()
        ) + 1,
        "effective_rank_999": int(
            torch.searchsorted(
                cum, torch.tensor(0.999, dtype=cum.dtype, device=cum.device)
            ).item()
        ) + 1,
    }


def rank_for_error(delta: Tensor | list[Tensor], rel_frobenius_tol: float) -> int:
    """Smallest ``r`` with :math:`\\sqrt{\\sum_{\\ell>r}\\sigma_\\ell^2} \\le \\mathrm{tol}\\,\\|\\Delta A\\|_F`.

    A direct inversion of Prop. 9.3: the rank is chosen from the spectrum, not tuned.
    """
    mat = _stack(delta)
    sv = torch.linalg.svdvals(mat.to(torch.float64))
    sq = sv * sv
    total = sq.sum()
    if float(total.item()) <= 0.0:
        return 1
    tail = torch.flip(torch.cumsum(torch.flip(sq, dims=[0]), dim=0), dims=[0])
    # tail[r] = sum_{l >= r} sigma_l^2  (0-indexed) -> error of rank r
    target = (float(rel_frobenius_tol) ** 2) * float(total.item())
    ok = torch.nonzero(tail <= target, as_tuple=False)
    if ok.numel() == 0:
        return int(sv.numel())
    return int(ok[0].item())


def compress_residual(
    delta: Tensor | list[Tensor],
    rank: int | None = None,
    *,
    rel_frobenius_tol: float | None = None,
) -> LowRankResidual:
    """Truncated-SVD compression of the residual field.

    Exactly one of ``rank`` / ``rel_frobenius_tol`` should be supplied; if both are
    ``None`` the rank is set to the full rank (lossless, useful as a control in the
    ablation).

    The singular values are absorbed into ``z``, so ``u`` has orthonormal columns -
    convenient because it makes ``u`` reusable as a fixed spatial basis if one ever
    wants to stream frames.
    """
    mat = _stack(delta)
    n, t = mat.shape
    full = min(n, t)

    if rank is None and rel_frobenius_tol is not None:
        rank = rank_for_error(mat, rel_frobenius_tol)
    if rank is None:
        rank = full
    rank = max(1, min(int(rank), full))

    u_f, s_f, vh_f = torch.linalg.svd(mat.to(torch.float64), full_matrices=False)
    u = (u_f[:, :rank]).to(mat.dtype)
    z = (torch.diag(s_f[:rank]) @ vh_f[:rank]).to(mat.dtype)

    approx = u @ z
    err = mat - approx
    fro = float(err.norm().item())
    spec = float(torch.linalg.matrix_norm(err.to(torch.float64), ord=2).item())

    return LowRankResidual(
        u=u,
        z=z,
        singular_values=s_f.to(mat.dtype),
        rank=rank,
        frobenius_error=fro,
        spectral_error=spec,
        relative_frobenius_error=fro / max(float(mat.norm().item()), 1e-30),
    )
