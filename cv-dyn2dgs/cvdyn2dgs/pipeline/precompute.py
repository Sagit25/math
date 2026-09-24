"""The CV-Dyn2DGS precomputation pipeline (proposal §2.12, §6.1).

One pass over a patient produces the stored representation of Eq. (6),

.. math:: \\mathcal{D} = \\bigl\\{\\mathcal{G}^{2D}_0,\\ \\{\\Gamma_t\\}_{t=0}^{T-1},\\
          \\{\\Delta a_t\\}_{t=0}^{T-1}\\bigr\\},

by the seven steps of proposal §2.12:

1. 3-D Chan-Vese on frame 0, seeded from the ED label or an interior seed;
2. canonical 2-D surfels on :math:`\\Gamma_0`, with amplitude/opacity/scale fitted;
3. for each later frame, a **warm-started** Chan-Vese solve (Eq. 5.1);
4. normal projection of every anchor onto the new surface (Eq. 6.3) and a tangent
   frame update (Eq. 7.6-7.7);
5. optional density control, then a compact amplitude residual (Eq. 9.3);
6. repeat to the last frame, storing only canonical surfels, surfaces and residuals;
7. playback re-derives everything else (see :mod:`cvdyn2dgs.pipeline.playback`).

Cost bookkeeping
----------------
Proposal Eq. (7)-(9) frames the efficiency claim as a **hypothesis to be tested**:

.. math:: C_{\\mathrm{ours}} = C_{\\mathrm{full}} +
          \\sum_{t=1}^{T-1}\\bigl(C^t_{\\mathrm{CV}} + C^t_{\\mathrm{project}}
          + C^t_{\\mathrm{orient}} + C^t_{\\mathrm{res}}\\bigr),

and the method is only faster if the per-frame bracket is cheaper than a full refit.
Every one of those four terms is timed separately here so Eq. (9) can be checked term
by term rather than asserted.  Setting ``cfg.refit_every_frame = True`` turns this same
function into the *Independent 2DGS* baseline, which makes the comparison
implementation-identical apart from the thing under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from ..core.config import PipelineConfig
from ..core.grid import Grid, trilinear_sample
from ..core.runtime import StageTimer
from ..levelset.chanvese import solve_frame
from ..levelset.mesh_extract import marching_tetrahedra
from ..levelset.operators import curvature, gradient_central
from ..metrics.storage import surface_storage
from ..render.camera import Camera
from ..render.raster2dgs import render_weights
from ..render.raymarch import raymarch_levelset
from ..residual.lowrank import LowRankResidual, compress_residual
from ..residual.solver import ResidualView, solve_residual
from ..surfel.canonical import initialize_canonical_surfels, surface_normals_at
from ..surfel.density import apply_density_control
from ..surfel.model import SurfelSet2D
from ..surfel.projection import project_to_surface
from ..surfel.transport import transport_tangent_frame
from .fitting import ViewTarget, fit_appearance

__all__ = [
    "PrecomputedModel",
    "FrameRecord",
    "make_supervision_cameras",
    "precompute",
]


def make_supervision_cameras(
    grid: Grid,
    *,
    axes: Sequence[int] = (2, 0, 1),
    device=None,
    dtype: torch.dtype = torch.float32,
) -> list[Camera]:
    """Orthographic cameras aligned with grid planes - the "known slice planes".

    Default ``axes = (2, 0, 1)`` gives one view down the short-axis stack direction
    plus two orthogonal in-plane views.  The first corresponds to the acquisition
    geometry; the other two act like the long-axis views proposal §7.3 uses to check
    that silhouette, depth and appearance stay consistent from directions the fit was
    not dominated by.
    """
    return [
        Camera.from_slice_plane(grid, axis=int(a), device=device, dtype=dtype) for a in axes
    ]


@dataclass
class FrameRecord:
    """Per-frame diagnostics accumulated during precomputation."""

    frame: int
    cv_iterations: int
    cv_time_ms: float
    cv_converged: bool
    project_time_ms: float = 0.0
    orient_time_ms: float = 0.0
    density_time_ms: float = 0.0
    residual_time_ms: float = 0.0
    refit_time_ms: float = 0.0
    n_surfels: int = 0
    e_surf_mm: float = 0.0
    e_surf_p95_mm: float = 0.0
    projection_clipped_fraction: float = 0.0
    projection_rolled_back_fraction: float = 0.0
    transport_sin_psi_min: float = 1.0
    transport_fallback_fraction: float = 0.0
    transport_rotation_mean_deg: float = 0.0
    residual_abs_mean: float = 0.0
    residual_cg_iterations: int = 0
    surface_bytes: dict[str, int] = field(default_factory=dict)
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def per_frame_update_ms(self) -> float:
        """The bracket of Eq. (8): CV + projection + orientation + residual."""
        return (
            self.cv_time_ms
            + self.project_time_ms
            + self.orient_time_ms
            + self.density_time_ms
            + self.residual_time_ms
        )

    def to_dict(self) -> dict[str, float]:
        d = {
            "frame": float(self.frame),
            "cv_iterations": float(self.cv_iterations),
            "cv_time_ms": self.cv_time_ms,
            "cv_converged": float(self.cv_converged),
            "project_time_ms": self.project_time_ms,
            "orient_time_ms": self.orient_time_ms,
            "density_time_ms": self.density_time_ms,
            "residual_time_ms": self.residual_time_ms,
            "refit_time_ms": self.refit_time_ms,
            "per_frame_update_ms": self.per_frame_update_ms,
            "n_surfels": float(self.n_surfels),
            "e_surf_mm": self.e_surf_mm,
            "e_surf_p95_mm": self.e_surf_p95_mm,
            "projection_clipped_fraction": self.projection_clipped_fraction,
            "projection_rolled_back_fraction": self.projection_rolled_back_fraction,
            "transport_sin_psi_min": self.transport_sin_psi_min,
            "transport_fallback_fraction": self.transport_fallback_fraction,
            "transport_rotation_mean_deg": self.transport_rotation_mean_deg,
            "residual_abs_mean": self.residual_abs_mean,
            "residual_cg_iterations": float(self.residual_cg_iterations),
        }
        d.update({f"surface_bytes/{k}": float(v) for k, v in self.surface_bytes.items()})
        d.update(self.extras)
        return d


@dataclass
class PrecomputedModel:
    """The stored representation :math:`\\mathcal{D}` of Eq. (6), plus diagnostics."""

    surfels: SurfelSet2D
    """:math:`\\mathcal{G}^{2D}_0` - the canonical surfel set."""

    phis: list[Tensor]
    """:math:`\\{\\Gamma_t\\}` as level sets. This is what playback reads."""

    residuals: list[Tensor]
    """:math:`\\{\\Delta a_t\\}`, each ``(N, C)``."""

    grid: Grid
    config: PipelineConfig
    frames: list[FrameRecord] = field(default_factory=list)
    lowrank: LowRankResidual | None = None
    canonical_stats: dict[str, float] = field(default_factory=dict)
    fit_stats: dict[str, float] = field(default_factory=dict)
    cameras: list[Camera] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.phis)

    def amplitude_at(self, t: int) -> Tensor:
        """:math:`a^t = a^0 + \\Delta a^t`, Eq. (25)."""
        base = self.surfels.amplitude.detach()
        if self.lowrank is not None:
            return base + self.lowrank.frame(t).unsqueeze(-1)
        return base + self.residuals[t]

    def total_precompute_ms(self) -> float:
        return sum(
            f.per_frame_update_ms + f.refit_time_ms for f in self.frames
        ) + float(self.fit_stats.get("fit/time_ms", 0.0))

    def summary(self) -> dict[str, float]:
        n = max(1, len(self.frames))
        out: dict[str, float] = {
            "n_frames": float(self.n_frames),
            "n_surfels_final": float(self.surfels.n),
            "total_precompute_ms": self.total_precompute_ms(),
            "mean_cv_iterations": sum(f.cv_iterations for f in self.frames) / n,
            "mean_cv_time_ms": sum(f.cv_time_ms for f in self.frames) / n,
            "mean_project_time_ms": sum(f.project_time_ms for f in self.frames) / n,
            "mean_orient_time_ms": sum(f.orient_time_ms for f in self.frames) / n,
            "mean_residual_time_ms": sum(f.residual_time_ms for f in self.frames) / n,
            "mean_e_surf_mm": sum(f.e_surf_mm for f in self.frames) / n,
            "max_e_surf_p95_mm": max((f.e_surf_p95_mm for f in self.frames), default=0.0),
            "mean_per_frame_update_ms": sum(f.per_frame_update_ms for f in self.frames) / n,
        }
        out.update(self.canonical_stats)
        out.update(self.fit_stats)
        return out


def _build_views(
    phi: Tensor,
    image: Tensor,
    grid: Grid,
    cameras: Sequence[Camera],
    *,
    grad_phi: Tensor | None = None,
) -> list[ViewTarget]:
    return [
        ViewTarget(
            camera=cam,
            reference=raymarch_levelset(phi, grid, cam, image=image, grad_phi=grad_phi),
        )
        for cam in cameras
    ]


@torch.no_grad()
def _update_geometry(
    surfels: SurfelSet2D,
    phi: Tensor,
    grid: Grid,
    cfg: PipelineConfig,
    grad_phi: Tensor,
    timer: StageTimer,
    record: FrameRecord,
) -> None:
    """Steps 4 of proposal §2.12: normal projection then tangent-frame update."""
    scfg = cfg.surfel

    with timer.stage("project"):
        proj = project_to_surface(
            surfels.anchor,
            phi,
            grid,
            iters=scfg.projection_iters,
            eps=scfg.projection_eps,
            max_step_mm=scfg.projection_max_step_mm,
            mode=scfg.projection_mode,
            grad_phi=grad_phi,
        )
    record.project_time_ms = timer.samples["project"][-1]
    record.e_surf_mm = float(proj.residual_mm.mean().item())
    record.e_surf_p95_mm = float(proj.residual_mm.quantile(0.95).item())
    record.projection_clipped_fraction = proj.clipped_fraction
    record.projection_rolled_back_fraction = proj.rolled_back_fraction

    with timer.stage("orient"):
        normal_prev = surfels.normal.clone()
        new_normal = surface_normals_at(
            proj.anchor, phi, grid, eps=scfg.normal_eps, grad_phi=grad_phi
        )
        if scfg.isotropic:
            # Prop. 7.5: the in-plane gauge is irrelevant for isotropic disks, so a
            # cheap re-orthogonalisation of the existing axis is sufficient.
            e1 = surfels.e1 - (surfels.e1 * new_normal).sum(-1, keepdim=True) * new_normal
            nrm = e1.norm(dim=-1, keepdim=True)
            degenerate = (nrm.squeeze(-1) < 1e-4)
            e1 = torch.where(degenerate.unsqueeze(-1), surfels.e2, e1)
            e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            e2 = torch.cross(new_normal, e1, dim=-1)
            surfels.set_geometry(anchor=proj.anchor, normal=new_normal, e1=e1, e2=e2)
            record.transport_fallback_fraction = float(
                degenerate.to(torch.float32).mean().item()
            )
        else:
            e1, e2, diag = transport_tangent_frame(
                surfels.e1,
                surfels.e2,
                new_normal,
                normal_prev=normal_prev,
                eps=scfg.normal_eps,
                degeneracy_thresh=scfg.transport_degeneracy_thresh,
                mode="gram_schmidt",
            )
            surfels.set_geometry(anchor=proj.anchor, normal=new_normal, e1=e1, e2=e2)
            s = diag.summary()
            record.transport_sin_psi_min = s["sin_psi_min"]
            record.transport_fallback_fraction = s["fallback_fraction"]
            record.transport_rotation_mean_deg = s["rotation_angle_mean_deg"]
    record.orient_time_ms = timer.samples["orient"][-1]


@torch.no_grad()
def _retune_scales(
    surfels: SurfelSet2D, phi: Tensor, grid: Grid, cfg: PipelineConfig
) -> None:
    """Re-cap disk radii against the local curvature (Prop. 8.3).

    Keeps :math:`\\tfrac12\\kappa s^2` under the configured budget as the surface bends
    through the cycle.  Only ever *shrinks* relative to the fitted scale, so it cannot
    manufacture coverage that the frame-0 fit did not justify.
    """
    scfg = cfg.surfel
    if not scfg.curvature_adaptive_scale:
        return
    kap = curvature(phi, grid.spacing).abs()
    k_at = trilinear_sample(kap, grid.world_to_voxel(surfels.anchor)).clamp_min(1e-6)
    cap = torch.sqrt(2.0 * scfg.curvature_error_budget_mm / k_at)
    new_scale = torch.minimum(surfels.scale.detach(), cap.unsqueeze(-1).expand(-1, 2))
    new_scale = new_scale.clamp(scfg.scale_min_mm, scfg.scale_max_mm)
    surfels.log_scale.data.copy_(torch.log(new_scale))


def precompute(
    images: Sequence[Tensor],
    phi0_init: Tensor,
    grid: Grid,
    cfg: PipelineConfig,
    *,
    generator: torch.Generator | None = None,
    cameras: Sequence[Camera] | None = None,
    timer: StageTimer | None = None,
    verbose: bool = False,
    phis_override: Sequence[Tensor] | None = None,
) -> PrecomputedModel:
    """Run the full precomputation for one patient.

    Parameters
    ----------
    images:
        ``T`` intensity volumes, each ``(nx, ny, nz)``, normalised to ~``[0, 1]``.
    phi0_init:
        Initial level set for frame 0 (from the ED label, via
        :func:`cvdyn2dgs.levelset.sdf.signed_distance_from_mask`, or an interior seed).
    cameras:
        Supervision views; defaults to :func:`make_supervision_cameras`.
    phis_override:
        Use these level sets instead of solving Chan-Vese. One per frame, same order as
        ``images``. This exists for the layer-1 *oracle* comparison: substituting an exact
        surface while leaving every other stage untouched is the only way to separate
        segmentation error from representation error. Chan-Vese timings are recorded as
        zero and ``cv_iterations`` as zero, so an oracle run can never be mistaken for a
        measurement of the solver.

    Returns
    -------
    :class:`PrecomputedModel`
    """
    if len(images) == 0:
        raise ValueError("need at least one frame")
    if phis_override is not None and len(phis_override) != len(images):
        raise ValueError(
            f"phis_override has {len(phis_override)} level sets for {len(images)} frames"
        )
    device = images[0].device
    timer = timer or StageTimer(device=device)
    cams = list(cameras) if cameras is not None else make_supervision_cameras(
        grid, device=device, dtype=images[0].dtype
    )

    # ---- step 1: frame-0 surface ------------------------------------------
    if phis_override is not None:
        # Oracle surface: no solve, and the solver metrics stay zero so this run cannot be
        # read as evidence about Chan-Vese convergence.
        phi0 = phis_override[0].clone()
        rec0 = FrameRecord(frame=0, cv_iterations=0, cv_time_ms=0.0, cv_converged=True)
    else:
        with timer.stage("cv"):
            cv0 = solve_frame(
                images[0],
                phi0_init,
                grid,
                cfg.chanvese,
                max_iters=int(cfg.chanvese.max_iters * cfg.chanvese.frame0_iters_scale),
            )
        rec0 = FrameRecord(
            frame=0,
            cv_iterations=cv0.iterations,
            cv_time_ms=timer.samples["cv"][-1],
            cv_converged=cv0.converged,
        )
        phi0 = cv0.phi
    grad0 = gradient_central(phi0, grid.spacing)

    # ---- step 2: canonical surfels ---------------------------------------
    surfels, can_stats = initialize_canonical_surfels(
        phi0, images[0], grid, cfg.surfel, generator=generator
    )

    # ---- step 2b: fit appearance on frame 0 ------------------------------
    views0 = _build_views(phi0, images[0], grid, cams, grad_phi=grad0)
    fit_rep = fit_appearance(
        surfels, views0, cfg.fit, cfg.loss, cfg.render, timer=timer
    )

    rec0.n_surfels = surfels.n
    mesh0 = marching_tetrahedra(phi0, grid)
    rec0.surface_bytes = {
        k: v.total_bytes
        for k, v in surface_storage(
            phi0, grid, band_mm=cfg.narrow_band_store_mm, mesh=mesh0
        ).items()
    }

    phis: list[Tensor] = [phi0]
    residuals: list[Tensor] = [torch.zeros_like(surfels.amplitude.detach())]
    records: list[FrameRecord] = [rec0]
    n_initial = surfels.n

    # ---- steps 3-6: the rest of the cycle --------------------------------
    for t in range(1, len(images)):
        rec = FrameRecord(frame=t, cv_iterations=0, cv_time_ms=0.0, cv_converged=False)

        if phis_override is not None:
            phi_t = phis_override[t].clone()
            rec.cv_iterations = 0
            rec.cv_time_ms = 0.0
            rec.cv_converged = True
        else:
            with timer.stage("cv"):
                cv = solve_frame(
                    images[t],
                    phis[-1] if cfg.chanvese.warm_start else phi0_init,
                    grid,
                    cfg.chanvese,
                )
            rec.cv_iterations = cv.iterations
            rec.cv_time_ms = timer.samples["cv"][-1]
            rec.cv_converged = cv.converged
            phi_t = cv.phi
        grad_t = gradient_central(phi_t, grid.spacing)

        _update_geometry(surfels, phi_t, grid, cfg, grad_t, timer, rec)

        with timer.stage("density"):
            if cfg.surfel.repulsion_enabled or cfg.surfel.densify_enabled or cfg.surfel.prune_enabled:
                # May change N; the residual history is realigned after the loop.
                surfels, dstats = apply_density_control(
                    surfels, phi_t, grid, cfg.surfel, n_initial=n_initial, grad_phi=grad_t
                )
                rec.extras.update(dstats)
            _retune_scales(surfels, phi_t, grid, cfg)
        rec.density_time_ms = timer.samples["density"][-1]
        rec.n_surfels = surfels.n

        # ---- residual (or full refit for the Independent-2DGS baseline) ---
        views_t = _build_views(phi_t, images[t], grid, cams, grad_phi=grad_t)

        if cfg.refit_every_frame:
            with timer.stage("refit"):
                fit_appearance(surfels, views_t, cfg.fit, cfg.loss, cfg.render, timer=None)
            rec.refit_time_ms = timer.samples["refit"][-1]
            residuals.append(torch.zeros_like(surfels.amplitude.detach()))
        else:
            with timer.stage("residual"):
                base = surfels.amplitude.detach()
                prev = residuals[-1]
                if prev.shape[0] != base.shape[0]:
                    # N changed through density control; pad or truncate the history.
                    pad = torch.zeros_like(base)
                    m = min(prev.shape[0], base.shape[0])
                    pad[:m] = prev[:m]
                    prev = pad
                rviews = [
                    ResidualView(
                        weights=render_weights(surfels, v.camera, cfg.render),
                        target=v.target_color,
                        pixel_weight=v.reference.mask_float,
                    )
                    for v in views_t
                ]
                res = solve_residual(rviews, base, cfg.residual, prev_delta=prev)
                residuals.append(res.delta)
            rec.residual_time_ms = timer.samples["residual"][-1]
            rec.residual_abs_mean = float(res.delta.abs().mean().item())
            rec.residual_cg_iterations = res.iterations
            rec.extras.update(res.summary())

        mesh_t = marching_tetrahedra(phi_t, grid)
        rec.surface_bytes = {
            k: v.total_bytes
            for k, v in surface_storage(
                phi_t, grid, band_mm=cfg.narrow_band_store_mm, mesh=mesh_t
            ).items()
        }

        phis.append(phi_t)
        records.append(rec)
        if verbose:
            print(
                f"[frame {t:3d}] cv={rec.cv_iterations:4d} it "
                f"({rec.cv_time_ms:7.1f} ms)  E_surf={rec.e_surf_mm:.4f} mm  "
                f"N={rec.n_surfels}  update={rec.per_frame_update_ms:7.1f} ms"
            )

    # ---- align the residual history to the final surfel count -------------
    # Density control (pruning/densification) can change N mid-sequence, so earlier
    # frames' residuals were solved for a different number of surfels.  Every consumer
    # of the model indexes residuals by the *final* N, so pad or truncate here rather
    # than leaving a shape mismatch to surface later as an indexing error.
    n_final = surfels.n
    n_ch = int(surfels.amplitude.shape[1])

    def _align(r: Tensor) -> Tensor:
        if r.shape[0] == n_final and r.shape[1] == n_ch:
            return r
        out = torch.zeros((n_final, n_ch), device=r.device, dtype=r.dtype)
        m = min(r.shape[0], n_final)
        out[:m, : min(r.shape[1], n_ch)] = r[:m, : min(r.shape[1], n_ch)]
        return out

    residuals = [_align(r) for r in residuals]

    # ---- optional low-rank residual compression (Eq. 9.5) ----------------
    lowrank = None
    if cfg.residual.enabled and cfg.residual.lowrank_rank is not None:
        lowrank = compress_residual(
            torch.stack([r[:, 0] for r in residuals], dim=1), cfg.residual.lowrank_rank
        )

    return PrecomputedModel(
        surfels=surfels,
        phis=phis,
        residuals=residuals,
        grid=grid,
        config=cfg,
        frames=records,
        lowrank=lowrank,
        canonical_stats={f"canonical/{k}": v for k, v in can_stats.items()},
        fit_stats=fit_rep.summary(),
        cameras=cams,
    )
