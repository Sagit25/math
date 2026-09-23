"""Canonical-frame appearance fitting (proposal §6.6).

Only three quantities are fitted: the amplitudes :math:`a^0_i`, the opacities
:math:`o^0_i` and the tangential scales :math:`s_{i,1}, s_{i,2}`.  Anchors, tangent
frames and normals are buffers produced by the Chan-Vese surface via Eq. (6.3) and
Eq. (7.6)-(7.7), so no gradient can reach them - the claim that geometry comes from
:math:`\\Gamma_t` alone is enforced structurally, not by a zero learning rate.

Fitting the *scales* is deliberate and is what proposal §4.3 means by "estimate
intensity, opacity and scale directly through differentiable rasterisation": the disk
extent controls the hole/overlap trade-off, and letting the data choose it is
strictly better than the geometric initialisation from
:func:`cvdyn2dgs.surfel.canonical.initialize_canonical_surfels`.

Multi-view supervision comes from the known slice planes only (see
:mod:`cvdyn2dgs.render.raymarch`).  Losses from all views are summed, which weights
views by pixel count; that is the intended behaviour since every plane is an equally
valid observation of the same surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
from torch import Tensor

from ..core.config import FitConfig, LossConfig, RenderConfig
from ..core.runtime import StageTimer
from ..losses import total_loss
from ..render.camera import Camera
from ..render.raster2dgs import render_2dgs
from ..render.raymarch import SurfaceReference
from ..surfel.model import SurfelSet2D

__all__ = ["ViewTarget", "FitReport", "fit_appearance"]


@dataclass
class ViewTarget:
    """One supervision view: a known slice plane and its reference rendering."""

    camera: Camera
    reference: SurfaceReference

    @property
    def target_color(self) -> Tensor:
        return self.reference.intensity

    @property
    def target_mask(self) -> Tensor:
        return self.reference.hit

    def roi(self, dilate: bool = False) -> Tensor:
        """Heart ROI used to keep background from dominating the appearance score."""
        return self.reference.hit


@dataclass
class FitReport:
    """Optimisation trace and the final state of the fitted quantities."""

    iterations: int
    loss_history: list[float] = field(default_factory=list)
    term_history: list[dict[str, float]] = field(default_factory=list)
    final_terms: dict[str, float] = field(default_factory=dict)
    scale_mean_before_mm: float = 0.0
    scale_mean_after_mm: float = 0.0
    opacity_mean_before: float = 0.0
    opacity_mean_after: float = 0.0
    time_ms: float = 0.0

    def summary(self) -> dict[str, float]:
        d = {
            "fit/iterations": float(self.iterations),
            "fit/loss_initial": self.loss_history[0] if self.loss_history else float("nan"),
            "fit/loss_final": self.loss_history[-1] if self.loss_history else float("nan"),
            "fit/scale_mean_before_mm": self.scale_mean_before_mm,
            "fit/scale_mean_after_mm": self.scale_mean_after_mm,
            "fit/opacity_mean_before": self.opacity_mean_before,
            "fit/opacity_mean_after": self.opacity_mean_after,
            "fit/time_ms": self.time_ms,
        }
        d.update({f"fit/{k}": v for k, v in self.final_terms.items()})
        return d


def fit_appearance(
    surfels: SurfelSet2D,
    views: Sequence[ViewTarget],
    fit_cfg: FitConfig,
    loss_cfg: LossConfig,
    render_cfg: RenderConfig,
    *,
    timer: StageTimer | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> FitReport:
    """Fit amplitude / opacity / scale of a surfel set to a set of views.

    Modifies ``surfels`` in place.  Returns a :class:`FitReport` whose loss history
    is the primary sanity check: if it does not decrease, the geometry is wrong
    (usually a sign-convention slip in :math:`\\phi`) and no amount of appearance
    fitting will hide it.
    """
    if not views:
        raise ValueError("fit_appearance needs at least one view")

    params: list[dict] = [
        {"params": [surfels.amplitude], "lr": float(fit_cfg.lr_amplitude)},
        {"params": [surfels.opacity_logit], "lr": float(fit_cfg.lr_opacity)},
    ]
    if fit_cfg.lr_scale > 0:
        params.append({"params": [surfels.log_scale], "lr": float(fit_cfg.lr_scale)})
    opt = torch.optim.Adam(params, eps=1e-15)

    report = FitReport(iterations=int(fit_cfg.iters))
    report.scale_mean_before_mm = float(surfels.scale.detach().mean().item())
    report.opacity_mean_before = float(surfels.opacity.detach().mean().item())

    acc: dict[str, float] = {}
    ctx = timer.stage("fit/canonical") if timer is not None else _null_context()
    with ctx:
        for it in range(int(fit_cfg.iters)):
            opt.zero_grad(set_to_none=True)

            total = torch.zeros((), device=surfels.device, dtype=surfels.dtype)
            acc = {}
            for view in views:
                out = render_2dgs(surfels, view.camera, render_cfg, compute_aux=True)
                terms = total_loss(
                    out,
                    view.target_color,
                    loss_cfg,
                    camera=view.camera,
                    target_mask=view.target_mask.to(out.alpha.dtype),
                    roi_weight=None,
                )
                total = total + terms.total
                for k, v in terms.to_dict().items():
                    acc[k] = acc.get(k, 0.0) + v

            total.backward()
            if fit_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in params for p in g["params"]], float(fit_cfg.grad_clip)
                )
            opt.step()

            loss_val = float(total.detach().item())
            report.loss_history.append(loss_val)
            if fit_cfg.log_every > 0 and (it % fit_cfg.log_every == 0 or it == fit_cfg.iters - 1):
                report.term_history.append({"iter": float(it), **acc})
            if on_step is not None:
                on_step(it, loss_val)

        report.final_terms = dict(acc)

    if timer is not None:
        report.time_ms = timer.samples.get("fit/canonical", [0.0])[-1]
    report.scale_mean_after_mm = float(surfels.scale.detach().mean().item())
    report.opacity_mean_after = float(surfels.opacity.detach().mean().item())
    return report


class _null_context:
    """Stand-in for a timer stage when no timer was supplied."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc) -> None:
        return None
