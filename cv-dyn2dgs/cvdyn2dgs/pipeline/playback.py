"""Interactive playback: what the viewer does per frame, and how fast.

Proposal §2.10 and §3.3 are careful about the word "real-time": it does **not** mean
segmenting while the scanner acquires.  It means that after a one-off per-patient
precomputation, the stored representation supports playback, free camera rotation and
zoom at 30 FPS or better.  The per-frame budget is Eq. (34):

.. math:: T_{\\mathrm{frame}} = T_{\\mathrm{surface}} + T_{\\mathrm{project}}
          + T_{\\mathrm{orient}} + T_{\\mathrm{raster}} < 33.3\\ \\mathrm{ms},

and crucially **no Chan-Vese and no non-linear optimisation run at playback time**.

An honest wrinkle about seeking
-------------------------------
Eq. (6) stores only :math:`\\mathcal{G}^{2D}_0` and :math:`\\{\\Gamma_t\\}` - no
per-frame anchor positions.  So frame-:math:`t` geometry must be recoverable from
those two alone, which means projecting the *canonical* anchors onto
:math:`\\Gamma_t` (``mode="canonical"``).  Precomputation, by contrast, projected
anchors **sequentially**, :math:`p^{t-1} \\to p^t` (``mode="chained"``).

These two are not identical.  Sequential projection accumulates a path; direct
projection from frame 0 takes a single, much longer step, which is more likely to hit
the trust region or land on the wrong surface branch - precisely the failure mode
proposal §6.3 flags.  Rather than quietly picking one, :class:`PlaybackEngine`
implements both and :func:`compare_projection_modes` measures the gap.  If direct
projection degrades :math:`E_{\\mathrm{surf}}`, then honest storage accounting has to
either add per-frame anchors or restrict the viewer to sequential playback, and the
experiment says which.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch
from torch import Tensor

from ..core.config import RenderConfig
from ..core.runtime import synchronize
from ..levelset.operators import gradient_central
from ..levelset.sdf import interpolate_levelsets
from ..render.camera import Camera
from ..render.raster2dgs import RenderOutput, render_2dgs
from ..surfel.canonical import surface_normals_at
from ..surfel.model import SurfelSet2D
from ..surfel.projection import project_to_surface
from ..surfel.transport import transport_tangent_frame
from .precompute import PrecomputedModel

__all__ = [
    "FrameTiming",
    "PlaybackEngine",
    "PlaybackReport",
    "measure_playback",
    "compare_projection_modes",
]

ProjectionSource = Literal["canonical", "chained"]


@dataclass
class FrameTiming:
    """One frame's decomposition of Eq. (34), in milliseconds."""

    surface_ms: float = 0.0
    project_ms: float = 0.0
    orient_ms: float = 0.0
    raster_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.surface_ms + self.project_ms + self.orient_ms + self.raster_ms

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else float("inf")

    def to_dict(self) -> dict[str, float]:
        return {
            "surface_ms": self.surface_ms,
            "project_ms": self.project_ms,
            "orient_ms": self.orient_ms,
            "raster_ms": self.raster_ms,
            "total_ms": self.total_ms,
            "fps": self.fps,
        }


class PlaybackEngine:
    """Stateful viewer over a :class:`PrecomputedModel`.

    Holds a working copy of the surfel set whose geometry is rewritten each frame.
    The canonical set inside ``model`` is never mutated, so a model can drive several
    engines (e.g. one per projection mode) without interference.
    """

    def __init__(
        self,
        model: PrecomputedModel,
        *,
        render_cfg: RenderConfig | None = None,
        projection_source: ProjectionSource = "chained",
        cache_gradients: bool = True,
    ) -> None:
        self.model = model
        self.render_cfg = render_cfg or model.config.render
        self.projection_source = projection_source
        self.cache_gradients = cache_gradients

        self.work: SurfelSet2D = model.surfels.clone_detached()
        self._canonical_anchor = model.surfels.anchor.clone()
        self._canonical_e1 = model.surfels.e1.clone()
        self._canonical_e2 = model.surfels.e2.clone()
        self._canonical_normal = model.surfels.normal.clone()
        self._grad_cache: dict[int, Tensor] = {}
        self._last_frame: int | None = None

    # ------------------------------------------------------------------ misc
    def reset(self) -> None:
        """Return the working set to its canonical state."""
        self.work.set_geometry(
            anchor=self._canonical_anchor,
            e1=self._canonical_e1,
            e2=self._canonical_e2,
            normal=self._canonical_normal,
        )
        self._last_frame = None

    def _gradient(self, t: int, phi: Tensor) -> Tensor:
        if not self.cache_gradients:
            return gradient_central(phi, self.model.grid.spacing)
        if t not in self._grad_cache:
            self._grad_cache[t] = gradient_central(phi, self.model.grid.spacing)
        return self._grad_cache[t]

    # ----------------------------------------------------------------- frame
    @torch.no_grad()
    def step_to(
        self,
        t: int,
        camera: Camera,
        *,
        beta: float = 0.0,
        compute_aux: bool = True,
    ) -> tuple[RenderOutput, FrameTiming]:
        """Update geometry to frame ``t`` (+ ``beta``) and rasterise.

        Parameters
        ----------
        beta:
            Mid-time interpolation parameter of Eq. (10.1) / proposal Eq. (35).
            ``beta = 0`` shows the stored frame.  Non-zero values interpolate towards
            frame ``t+1`` for smoother playback - **display only**, since Prop. 10.1
            shows the interpolated level set is not the true intermediate surface.
        """
        cfg = self.model.config
        grid = self.model.grid
        timing = FrameTiming()
        dev = self.work.device

        # ---- T_surface: load (and optionally interpolate) the stored surface
        synchronize(dev)
        t0 = time.perf_counter()
        phi = self.model.phis[t]
        if beta > 0.0 and t + 1 < self.model.n_frames:
            phi, _, _ = interpolate_levelsets(
                phi, self.model.phis[t + 1], beta, grid.spacing
            )
            grad = gradient_central(phi, grid.spacing)
        else:
            grad = self._gradient(t, phi)
        synchronize(dev)
        timing.surface_ms = (time.perf_counter() - t0) * 1e3

        # ---- choose the anchors to project from
        if self.projection_source == "canonical" or self._last_frame is None:
            start_anchor = self._canonical_anchor
            start_e1, start_e2 = self._canonical_e1, self._canonical_e2
            start_normal = self._canonical_normal
        else:
            start_anchor = self.work.anchor
            start_e1, start_e2 = self.work.e1, self.work.e2
            start_normal = self.work.normal

        # ---- T_project: Eq. (6.3)
        synchronize(dev)
        t0 = time.perf_counter()
        proj = project_to_surface(
            start_anchor,
            phi,
            grid,
            iters=cfg.surfel.projection_iters,
            eps=cfg.surfel.projection_eps,
            max_step_mm=cfg.surfel.projection_max_step_mm,
            grad_phi=grad,
        )
        synchronize(dev)
        timing.project_ms = (time.perf_counter() - t0) * 1e3

        # ---- T_orient: Eq. (7.3) + (7.6)-(7.7)
        synchronize(dev)
        t0 = time.perf_counter()
        normal = surface_normals_at(
            proj.anchor, phi, grid, eps=cfg.surfel.normal_eps, grad_phi=grad
        )
        if cfg.surfel.isotropic:
            e1 = start_e1 - (start_e1 * normal).sum(-1, keepdim=True) * normal
            bad = e1.norm(dim=-1) < 1e-4
            e1 = torch.where(bad.unsqueeze(-1), start_e2, e1)
            e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            e2 = torch.cross(normal, e1, dim=-1)
        else:
            e1, e2, _ = transport_tangent_frame(
                start_e1,
                start_e2,
                normal,
                normal_prev=start_normal,
                eps=cfg.surfel.normal_eps,
                degeneracy_thresh=cfg.surfel.transport_degeneracy_thresh,
            )
        self.work.set_geometry(anchor=proj.anchor, normal=normal, e1=e1, e2=e2)
        synchronize(dev)
        timing.orient_ms = (time.perf_counter() - t0) * 1e3

        # ---- amplitude from the stored residual (no optimisation here)
        amp = self.model.amplitude_at(min(t, len(self.model.residuals) - 1))
        if amp.shape[0] == self.work.n:
            self.work.amplitude.data.copy_(amp)

        # ---- T_raster: Eq. (8.2)-(8.6)
        synchronize(dev)
        t0 = time.perf_counter()
        out = render_2dgs(self.work, camera, self.render_cfg, compute_aux=compute_aux)
        synchronize(dev)
        timing.raster_ms = (time.perf_counter() - t0) * 1e3

        self._last_frame = t
        return out, timing

    @torch.no_grad()
    def surface_residual(self, t: int) -> Tensor:
        """``|phi_t(p_i)|`` for the current working anchors - :math:`E_{\\mathrm{surf}}`."""
        from ..core.grid import trilinear_sample

        phi = self.model.phis[t]
        return trilinear_sample(phi, self.model.grid.world_to_voxel(self.work.anchor)).abs()


@dataclass
class PlaybackReport:
    """Aggregated playback timings and the 30 FPS verdict."""

    per_frame: list[FrameTiming] = field(default_factory=list)
    n_surfels: int = 0
    height: int = 0
    width: int = 0
    projection_source: str = ""
    extras: dict[str, float] = field(default_factory=dict)

    def _totals(self) -> list[float]:
        return sorted(f.total_ms for f in self.per_frame)

    @property
    def mean_ms(self) -> float:
        ts = [f.total_ms for f in self.per_frame]
        return sum(ts) / max(1, len(ts))

    @property
    def p95_ms(self) -> float:
        ts = self._totals()
        if not ts:
            return float("nan")
        pos = 0.95 * (len(ts) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ts) - 1)
        return ts[lo] + (ts[hi] - ts[lo]) * (pos - lo)

    @property
    def mean_fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else float("inf")

    @property
    def meets_30fps(self) -> bool:
        """Eq. (34): the success criterion is on **total frame latency**, and
        proposal §8.3 asks for the 95th percentile, not just the mean - a viewer that
        stutters every tenth frame is not interactive."""
        return self.p95_ms < 33.3

    STAGES = ("surface_ms", "project_ms", "orient_ms", "raster_ms")

    def _percentile(self, values: list[float], q: float) -> float:
        if not values:
            return float("nan")
        vs = sorted(values)
        pos = q * (len(vs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vs) - 1)
        return vs[lo] + (vs[hi] - vs[lo]) * (pos - lo)

    def stage_p95(self) -> dict[str, float]:
        """95th percentile **per stage**, not just of the total.

        Without this, a p95 total that misses the 33.3 ms budget says only *that* the
        viewer stutters, never *where*.  Since the four stages have completely different
        remedies - a slower level-set read, a projection that needed more Newton steps, a
        transport fallback, or a tile overflow in the rasteriser - the aggregate on its own
        cannot direct any fix.
        """
        return {
            s: self._percentile([getattr(f, s) for f in self.per_frame], 0.95)
            for s in self.STAGES
        }

    def slowest_stage_on_worst_frames(self, *, top_fraction: float = 0.05) -> dict[str, float]:
        """Which stage dominates the slowest frames.

        The stage with the largest *mean* cost is not necessarily the one responsible for
        the tail.  A rasteriser that is usually cheap but occasionally overflows its tile
        budget will lose on p95 while looking fine on the mean, so attribution is computed
        on the worst frames specifically.

        Returns the fraction of worst-frames each stage dominates, so a split cause is
        visible instead of being collapsed to one label.
        """
        if not self.per_frame:
            return {}
        n = max(1, int(round(top_fraction * len(self.per_frame))))
        worst = sorted(self.per_frame, key=lambda f: f.total_ms, reverse=True)[:n]
        counts = dict.fromkeys(self.STAGES, 0.0)
        for f in worst:
            dom = max(self.STAGES, key=lambda s: getattr(f, s))
            counts[dom] += 1.0
        return {f"worst_dominated_by_{s}": c / n for s, c in counts.items()}

    @property
    def headroom_ratio(self) -> float:
        """``33.3 / p95``.  Above 1 passes; the value says by how much.

        ``meets_30fps`` is a pre-registered pass/fail and stays that way, but a bare
        boolean cannot distinguish scraping past the budget from having threefold margin -
        and those imply very different conclusions about whether the result will survive a
        larger surfel count or a higher resolution.
        """
        p = self.p95_ms
        if not (p > 0) or p != p:  # zero, negative or NaN
            return float("nan")
        return 33.3 / p

    def summary(self) -> dict[str, float | str | bool]:
        def mean_of(attr: str) -> float:
            vals = [getattr(f, attr) for f in self.per_frame]
            return sum(vals) / max(1, len(vals))

        d: dict[str, float | str | bool] = {
            "frames_measured": float(len(self.per_frame)),
            "n_surfels": float(self.n_surfels),
            "resolution": f"{self.width}x{self.height}",
            "projection_source": self.projection_source,
            "mean_total_ms": self.mean_ms,
            "p95_total_ms": self.p95_ms,
            "max_total_ms": max((f.total_ms for f in self.per_frame), default=float("nan")),
            "mean_fps": self.mean_fps,
            "meets_30fps_p95": self.meets_30fps,
            "headroom_ratio": self.headroom_ratio,
            "mean_surface_ms": mean_of("surface_ms"),
            "mean_project_ms": mean_of("project_ms"),
            "mean_orient_ms": mean_of("orient_ms"),
            "mean_raster_ms": mean_of("raster_ms"),
        }
        d.update({f"p95_{k}": v for k, v in self.stage_p95().items()})
        d.update(self.slowest_stage_on_worst_frames())
        d.update(self.extras)
        return d


@torch.no_grad()
def measure_playback(
    model: PrecomputedModel,
    cameras: Sequence[Camera],
    *,
    render_cfg: RenderConfig | None = None,
    loops: int = 2,
    warmup: int = 3,
    projection_source: ProjectionSource = "chained",
    compute_aux: bool = False,
) -> PlaybackReport:
    """Measure per-frame playback latency over a camera path.

    Parameters
    ----------
    cameras:
        Camera per displayed frame; cycled if shorter than the sequence.  Sweeping the
        camera (e.g. with :meth:`Camera.orbit`) is the point - a static camera hides
        the cost of re-binning surfels when the view changes.
    warmup:
        Frames discarded before measuring.  Essential on CUDA, where the first calls
        include allocator growth and kernel autotuning.
    compute_aux:
        Depth/normal/distortion buffers are only needed for *quality* metrics, not for
        display, so FPS is measured with them off by default.  Leaving them on would
        understate the achievable frame rate.
    """
    engine = PlaybackEngine(
        model, render_cfg=render_cfg, projection_source=projection_source
    )
    cams = list(cameras)
    if not cams:
        raise ValueError("measure_playback needs at least one camera")

    n = model.n_frames
    order = [(t, cams[i % len(cams)]) for i in range(loops * n) for t in [i % n]]

    for i in range(min(warmup, len(order))):
        t, cam = order[i]
        engine.step_to(t, cam, compute_aux=compute_aux)

    report = PlaybackReport(
        n_surfels=engine.work.n,
        height=cams[0].height,
        width=cams[0].width,
        projection_source=projection_source,
    )
    for t, cam in order[warmup:]:
        _, timing = engine.step_to(t, cam, compute_aux=compute_aux)
        report.per_frame.append(timing)
    return report


@torch.no_grad()
def compare_projection_modes(
    model: PrecomputedModel, camera: Camera, *, render_cfg: RenderConfig | None = None
) -> dict[str, float]:
    """Quantify the chained-vs-canonical seek gap described in the module docstring.

    Walks the whole sequence twice and reports :math:`E_{\\mathrm{surf}}` for each
    mode.  A large ``canonical`` value means random seeking cannot be served from
    Eq. (6)'s storage alone, which is a finding about the representation - not a bug.
    """
    out: dict[str, float] = {}
    for mode in ("chained", "canonical"):
        engine = PlaybackEngine(
            model, render_cfg=render_cfg, projection_source=mode  # type: ignore[arg-type]
        )
        vals: list[float] = []
        worst = 0.0
        for t in range(model.n_frames):
            engine.step_to(t, camera, compute_aux=False)
            r = engine.surface_residual(t)
            vals.append(float(r.mean().item()))
            worst = max(worst, float(r.quantile(0.95).item()))
        out[f"{mode}/e_surf_mean_mm"] = sum(vals) / max(1, len(vals))
        out[f"{mode}/e_surf_p95_worst_mm"] = worst
        out[f"{mode}/e_surf_last_frame_mm"] = vals[-1] if vals else float("nan")
    out["canonical_minus_chained_mm"] = (
        out["canonical/e_surf_mean_mm"] - out["chained/e_surf_mean_mm"]
    )
    return out
