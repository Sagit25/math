"""Regularised least-squares estimation of the appearance residual.

Theory §9 / proposal Eq. (25)-(26).  The amplitude of surfel :math:`i` at frame
:math:`t` is split into a canonical part and a small residual,

.. math:: a^t_i = a^0_i + \\Delta a^t_i,

and the residual is estimated from the observed slice images :math:`y_t` by

.. math::
    \\Delta a^*_t = \\arg\\min_{\\Delta a_t}
        \\bigl\\| W\\bigl(A_t(a^0 + \\Delta a_t) - y_t\\bigr)\\bigr\\|^2
        + \\lambda_a \\|\\Delta a_t\\|^2
        + \\lambda_T \\|\\Delta a_t - \\Delta a_{t-1}\\|^2 .

Because Prop. 9.1 makes :math:`A_t` genuinely linear in the amplitudes, the
stationarity condition is the linear system of Eq. (9.3),

.. math::
    \\underbrace{\\bigl(A_t^\\top W^\\top W A_t + (\\lambda_a + \\lambda_T) I\\bigr)}_{M}
    \\Delta a_t
    = A_t^\\top W^\\top W\\,(y_t - A_t a^0) + \\lambda_T \\Delta a_{t-1},

which Prop. 9.2 shows is **symmetric positive definite** as soon as
:math:`\\lambda_a + \\lambda_T > 0`, with

.. math::
    \\mathrm{cond}(M) \\le
    \\frac{\\sigma_{\\max}(A_t^\\top W^\\top W A_t) + \\lambda_a + \\lambda_T}
         {\\lambda_a + \\lambda_T}.

SPD is exactly what conjugate gradients needs, so the system is solved
matrix-free: ``M`` is never assembled, only applied.  The bound above is also
*reported* (via a power iteration for :math:`\\sigma_{\\max}`), because it is the
quantity that says how much the regularisation is buying in conditioning and
therefore how much bias it is trading in.

Why regularisation is not optional here: many surfels are invisible from the known
slice planes, so :math:`A_t` has a large null space and the data term alone leaves
their residual undetermined.  The Tikhonov term pins those to zero; the temporal
term is what suppresses frame-to-frame brightness flicker (Eq. 40).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
from torch import Tensor

from ..core.config import ResidualConfig
from ..render.raster2dgs import WeightMatrix

__all__ = ["ResidualView", "ResidualResult", "solve_residual", "conjugate_gradient"]


@dataclass
class ResidualView:
    """One observed view and the rendering operator that produced it."""

    weights: WeightMatrix
    target: Tensor
    """``(C, H, W)`` observed intensity :math:`y_t` for this view."""

    pixel_weight: Tensor | None = None
    """``(H, W)`` diagonal of :math:`W`.  A natural choice is the heart ROI, so
    background pixels do not dominate the fit."""

    def _w2(self) -> Tensor | None:
        if self.pixel_weight is None:
            return None
        return (self.pixel_weight * self.pixel_weight).unsqueeze(0)


@dataclass
class ResidualResult:
    """Solved residual plus solver diagnostics."""

    delta: Tensor
    """``(N, C)`` :math:`\\Delta a^*_t`."""

    iterations: int
    residual_norm: float
    """Final :math:`\\|M\\Delta a - b\\|_2`."""

    relative_residual: float
    sigma_max_estimate: float = 0.0
    condition_bound: float = 0.0
    """The right-hand side of Eq. (9.4)."""

    data_term_before: float = 0.0
    data_term_after: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        d = {
            "residual/iterations": float(self.iterations),
            "residual/residual_norm": self.residual_norm,
            "residual/relative_residual": self.relative_residual,
            "residual/sigma_max": self.sigma_max_estimate,
            "residual/condition_bound": self.condition_bound,
            "residual/data_before": self.data_term_before,
            "residual/data_after": self.data_term_after,
            "residual/delta_abs_mean": float(self.delta.abs().mean().item()),
            "residual/delta_abs_max": float(self.delta.abs().max().item()),
        }
        d.update(self.extras)
        return d


def conjugate_gradient(
    apply_m: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    x0: Tensor | None = None,
    max_iters: int = 60,
    tol: float = 1e-6,
) -> tuple[Tensor, int, float]:
    """Solve ``M x = b`` for SPD ``M`` given only ``apply_m``.

    Standard CG with a relative-residual stopping rule.  Valid precisely because
    Prop. 9.2 establishes that ``M`` is SPD; with ``lambda_a + lambda_T = 0`` the
    matrix is only semi-definite and this would stall.

    Returns ``(x, iterations, final_residual_norm)``.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - apply_m(x)
    p = r.clone()
    rs = (r * r).sum()
    b_norm = torch.sqrt((b * b).sum()).clamp_min(1e-30)

    it = 0
    for it in range(1, int(max_iters) + 1):
        if torch.sqrt(rs) / b_norm <= tol:
            it -= 1
            break
        ap = apply_m(p)
        denom = (p * ap).sum()
        if float(denom.abs().item()) < 1e-30:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = (r * r).sum()
        p = r + (rs_new / rs.clamp_min(1e-30)) * p
        rs = rs_new

    return x, it, float(torch.sqrt(rs).item())


def _apply_normal_operator(views: Sequence[ResidualView], v: Tensor) -> Tensor:
    """:math:`\\sum_{\\text{views}} A^\\top W^\\top W A\\, v`."""
    out = torch.zeros_like(v)
    for view in views:
        img = view.weights.apply(v)  # (C, H, W)
        w2 = view._w2()
        if w2 is not None:
            img = img * w2
        out = out + view.weights.apply_transpose(img)
    return out


@torch.no_grad()
def _estimate_sigma_max(
    views: Sequence[ResidualView], shape: tuple[int, int], *, iters: int = 20, generator=None
) -> float:
    """Power iteration for the largest eigenvalue of :math:`A^\\top W^\\top W A`."""
    dev = views[0].target.device
    dt = views[0].target.dtype
    # Draw on the CPU so a CPU generator is always valid, then move.
    v = torch.randn(shape, dtype=torch.float32, generator=generator).to(device=dev, dtype=dt)
    v = v / v.norm().clamp_min(1e-30)
    lam = 0.0
    for _ in range(int(iters)):
        w = _apply_normal_operator(views, v)
        nrm = w.norm()
        if float(nrm.item()) < 1e-30:
            return 0.0
        v = w / nrm
        lam = float(nrm.item())
    return lam


@torch.no_grad()
def solve_residual(
    views: Sequence[ResidualView],
    amplitude_base: Tensor,
    cfg: ResidualConfig,
    *,
    prev_delta: Tensor | None = None,
    estimate_condition: bool = False,
    generator: torch.Generator | None = None,
) -> ResidualResult:
    """Solve Eq. (9.3) for one frame.

    Parameters
    ----------
    views:
        One or more observed views with their rendering operators.  For cine CMR
        these are the known slice planes (proposal §7.1: supervision is restricted
        to masks, depth, normals and intensity on those planes).
    amplitude_base:
        ``(N, C)`` canonical amplitudes :math:`a^0`.
    prev_delta:
        ``(N, C)`` :math:`\\Delta a_{t-1}`; ``None`` at the first frame, in which
        case the temporal term degenerates to an extra Tikhonov term of weight
        ``lambda_T`` about zero.

    Returns
    -------
    :class:`ResidualResult`
    """
    if not views:
        raise ValueError("solve_residual needs at least one view")
    if not cfg.enabled:
        return ResidualResult(
            delta=torch.zeros_like(amplitude_base),
            iterations=0,
            residual_norm=0.0,
            relative_residual=0.0,
        )

    lam = float(cfg.lambda_a) + float(cfg.lambda_T)
    if lam <= 0.0:
        raise ValueError(
            "Prop. 9.2 requires lambda_a + lambda_T > 0 for the normal equations "
            "to be positive definite; got both zero"
        )

    prev = torch.zeros_like(amplitude_base) if prev_delta is None else prev_delta

    # Right-hand side of Eq. (9.3)
    rhs = torch.zeros_like(amplitude_base)
    data_before = 0.0
    for view in views:
        pred0 = view.weights.apply(amplitude_base)
        resid = view.target - pred0
        w2 = view._w2()
        weighted = resid * w2 if w2 is not None else resid
        rhs = rhs + view.weights.apply_transpose(weighted)
        data_before += float((weighted * resid).sum().item())
    rhs = rhs + float(cfg.lambda_T) * prev

    def apply_m(v: Tensor) -> Tensor:
        return _apply_normal_operator(views, v) + lam * v

    delta, iters, rnorm = conjugate_gradient(
        apply_m, rhs, max_iters=cfg.cg_iters, tol=cfg.cg_tol
    )

    data_after = 0.0
    for view in views:
        resid = view.target - view.weights.apply(amplitude_base + delta)
        w2 = view._w2()
        weighted = resid * w2 if w2 is not None else resid
        data_after += float((weighted * resid).sum().item())

    sigma_max = 0.0
    cond = 0.0
    if estimate_condition:
        sigma_max = _estimate_sigma_max(
            views, tuple(amplitude_base.shape), generator=generator
        )
        cond = (sigma_max + lam) / lam  # Eq. (9.4)

    b_norm = float(rhs.norm().item())
    return ResidualResult(
        delta=delta,
        iterations=iters,
        residual_norm=rnorm,
        relative_residual=rnorm / max(b_norm, 1e-30),
        sigma_max_estimate=sigma_max,
        condition_bound=cond,
        data_term_before=data_before,
        data_term_after=data_after,
        extras={
            "residual/data_reduction": 1.0 - data_after / max(data_before, 1e-30),
            "residual/unseen_surfel_fraction": float(
                sum(
                    float((v.weights.contributions_per_surfel() < 1e-6).to(torch.float32).mean().item())
                    for v in views
                )
                / len(views)
            ),
        },
    )
