"""Shared evaluation harness for every experiment.

Design decisions that keep the comparisons honest:

**Fitting views and evaluation views are disjoint.**  The canonical fit is supervised
on the orthographic slice-plane cameras (the "known physical coordinates" of proposal
§7.1).  Quality is *measured* on perspective orbit cameras that were never fitted.
Scoring on the fitting views would measure memorisation, and for a representation whose
geometry is fixed by the level set that would be almost meaningless.

**Every method is scored against the same reference rendering of the same surface.**
The reference is produced by ray-marching the stored :math:`\\Gamma_t` (see
:mod:`cvdyn2dgs.render.raymarch`), so RQ5/RQ6 compare representations, not
segmentations.

**Segmentation scores only where ground truth exists.**  On the phantom that is every
frame; on ACDC/M&Ms-2 it is ED and ES only, and intermediate frames are restricted to
appearance and temporal metrics.

**Whole-image and ROI appearance are always both reported** (proposal §8.3), because a
black background makes whole-image PSNR meaningless on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor

from ..baselines import BaselineSpec, render_baseline
from ..core.config import PipelineConfig
from ..core.grid import Grid, trilinear_sample
from ..core.runtime import StageTimer
from ..data.phantom import Phantom4D
from ..levelset.mesh_extract import marching_tetrahedra
from ..levelset.sdf import signed_distance_from_mask
from ..metrics.photometric import appearance_report, temporal_report

if TYPE_CHECKING:  # pragma: no cover
    from ..metrics.costquality import CostQualityPoint
from ..metrics.rendering import (
    boundary_f_score,
    coverage_report,
    depth_rmse,
    normal_angular_error,
    silhouette_iou_sweep,
)
from ..metrics.segmentation import dice, mask_surface_points, surface_distance_report
from ..metrics.clinical import volume_curve_report
from ..pipeline.playback import PlaybackEngine, measure_playback
from ..pipeline.precompute import PrecomputedModel, make_supervision_cameras, precompute
from ..pipeline.storage_io import model_storage_report
from ..render.camera import Camera
from ..render.raymarch import raymarch_levelset

__all__ = [
    "EvalCameras",
    "EvaluationResult",
    "make_eval_cameras",
    "initial_levelset_from_mask",
    "run_pipeline_on_phantom",
    "evaluate_geometry",
    "evaluate_rendering",
    "evaluate_storage_and_speed",
    "evaluate_all",
    "cost_quality_comparison",
]


@dataclass
class EvalCameras:
    """Fitting and evaluation camera sets, kept explicitly separate."""

    fit: list[Camera]
    eval: list[Camera]

    def describe(self) -> dict[str, float | str]:
        return {
            "n_fit_views": float(len(self.fit)),
            "n_eval_views": float(len(self.eval)),
            "fit_projection": "orthographic slice planes",
            "eval_projection": "perspective orbit (novel views)",
        }


def make_eval_cameras(
    grid: Grid,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
    n_orbit: int = 6,
    resolution: int = 256,
    fov_deg: float = 40.0,
    radius_factor: float = 2.2,
) -> EvalCameras:
    """Build the fitting (slice-plane) and evaluation (orbit) camera sets."""
    fit = make_supervision_cameras(grid, device=device, dtype=dtype)
    centre = grid.center_world(device=device, dtype=dtype)
    radius = radius_factor * max(grid.extent_mm) * 0.5
    ev = [
        Camera.orbit(
            centre,
            radius_mm=radius,
            azimuth_deg=360.0 * i / max(1, n_orbit),
            elevation_deg=20.0 * (1 if i % 2 == 0 else -1),
            fov_deg=fov_deg,
            height=resolution,
            width=resolution,
        )
        for i in range(int(n_orbit))
    ]
    return EvalCameras(fit=fit, eval=ev)


def initial_levelset_from_mask(mask: Tensor, grid: Grid, *, band_mm: float = 12.0) -> Tensor:
    """Frame-0 initialisation from an ED label (proposal §2.12 step 1)."""
    return signed_distance_from_mask(mask, grid.spacing, max_dist_mm=band_mm)


# --------------------------------------------------------------------------- #
#  Running
# --------------------------------------------------------------------------- #
def run_pipeline_on_phantom(
    phantom: Phantom4D,
    cfg: PipelineConfig,
    *,
    cameras: EvalCameras | None = None,
    generator: torch.Generator | None = None,
    verbose: bool = False,
    oracle_surface: bool = False,
) -> tuple[PrecomputedModel, EvalCameras, StageTimer]:
    """Precompute a model on a phantom sequence, initialised from the ED label.

    ``oracle_surface`` replaces the Chan-Vese level sets with the phantom's **exact analytic**
    signed distance functions, leaving every other stage untouched.  This is the layer-1
    upper bound and it is not optional for attribution: without it, segmentation error and
    representation error are confounded, and no RQ5/RQ6 number can be assigned to either.
    The phantom is the only source where it is available exactly - real data has labels at
    ED and ES only.
    """
    device = phantom.images[0].device
    dtype = phantom.images[0].dtype
    cams = cameras or make_eval_cameras(phantom.grid, device=device, dtype=dtype)

    ed = phantom.ed_index()
    phi0 = initial_levelset_from_mask(phantom.masks[ed], phantom.grid)
    phi0 = phi0.to(device=device, dtype=dtype)

    # Play the cycle starting at ED so frame 0 is the reference phase.
    order = [(ed + t) % phantom.n_frames for t in range(phantom.n_frames)]
    images = [phantom.images[t] for t in order]

    timer = StageTimer(device=device)
    model = precompute(
        images,
        phi0,
        phantom.grid,
        cfg,
        generator=generator,
        cameras=cams.fit,
        timer=timer,
        verbose=verbose,
        phis_override=(
            [phantom.phi_gt[t].to(device=device, dtype=dtype) for t in order]
            if oracle_surface else None
        ),
    )
    model.canonical_stats["frame_order_start"] = float(ed)
    model.canonical_stats["oracle_surface"] = float(oracle_surface)
    return model, cams, timer


# --------------------------------------------------------------------------- #
#  Evaluation blocks
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_geometry(
    model: PrecomputedModel,
    *,
    gt_masks: dict[int, Tensor] | None = None,
    use_mesh_points: bool = True,
) -> dict[str, float]:
    """Dice / HD95 / ASSD against ground-truth masks, plus per-frame E_surf.

    ``gt_masks`` maps *model frame index* to a boolean mask.  On real data only ED and
    ES are present; passing only those is the correct behaviour, not a limitation to be
    worked around.
    """
    out: dict[str, float] = {}
    grid = model.grid

    e_surf = [f.e_surf_mm for f in model.frames]
    out["geom/e_surf_mean_mm"] = sum(e_surf) / max(1, len(e_surf))
    out["geom/e_surf_max_mm"] = max(e_surf) if e_surf else float("nan")
    out["geom/e_surf_p95_worst_mm"] = max(
        (f.e_surf_p95_mm for f in model.frames), default=float("nan")
    )

    if not gt_masks:
        return out

    dices: list[float] = []
    hd95s: list[float] = []
    assds: list[float] = []
    for t, gt in gt_masks.items():
        if t >= model.n_frames:
            continue
        phi = model.phis[t]
        pred = phi > 0
        dices.append(dice(pred, gt))

        if use_mesh_points:
            mesh = marching_tetrahedra(phi, grid)
            pts_pred = mesh.vertices
        else:
            pts_pred = mask_surface_points(pred, grid)
        pts_gt = mask_surface_points(gt, grid)
        rep = surface_distance_report(pts_pred, pts_gt)
        hd95s.append(rep.hd95_mm)
        assds.append(rep.assd_mm)
        out[f"geom/frame{t}/dice"] = dices[-1]
        out[f"geom/frame{t}/hd95_mm"] = rep.hd95_mm
        out[f"geom/frame{t}/assd_mm"] = rep.assd_mm

    if dices:
        out["geom/dice_mean"] = sum(dices) / len(dices)
        out["geom/hd95_mean_mm"] = sum(hd95s) / len(hd95s)
        out["geom/assd_mean_mm"] = sum(assds) / len(assds)
    return out


@torch.no_grad()
def evaluate_rendering(
    model: PrecomputedModel,
    spec: BaselineSpec,
    images: Sequence[Tensor],
    cameras: Sequence[Camera],
    *,
    frames: Sequence[int] | None = None,
    alpha_threshold: float = 0.5,
) -> dict[str, float]:
    """Silhouette / depth / normal / appearance quality on **novel** views.

    For each (frame, camera) pair the stored level set is ray-marched to produce the
    reference silhouette, depth, normal and surface intensity; the baseline's renderer
    is then scored against it.  Because all baselines share this reference, the numbers
    are directly comparable.
    """
    grid = model.grid
    frames = list(frames if frames is not None else range(model.n_frames))
    engine = PlaybackEngine(model, render_cfg=spec.config.render, projection_source="chained")

    acc: dict[str, list[float]] = {}
    rendered_seq: list[Tensor] = []
    reference_seq: list[Tensor] = []
    fixed_cam = cameras[0]

    def push(key: str, val: float) -> None:
        if val == val:  # skip NaN
            acc.setdefault(key, []).append(val)

    for t in frames:
        # Advance playback geometry to this frame (surfel-based renderers need it).
        engine.step_to(t, fixed_cam, compute_aux=False)
        phi = model.phis[t]
        mesh = marching_tetrahedra(phi, grid) if spec.renderer == "mesh" else None
        vamp = None
        if mesh is not None:
            vamp = trilinear_sample(
                images[t], grid.world_to_voxel(mesh.vertices)
            ).unsqueeze(-1)

        for ci, cam in enumerate(cameras):
            ref = raymarch_levelset(phi, grid, cam, image=images[t])
            out = render_baseline(
                spec,
                cam,
                surfels=engine.work,
                mesh=mesh,
                vertex_amplitude=vamp,
                compute_aux=True,
            )

            sweep = silhouette_iou_sweep(out.alpha, ref.hit)
            push("render/iou@0.5", sweep["iou@0.5"])
            push("render/best_iou", sweep["best_iou"])

            pred_mask = out.alpha >= alpha_threshold
            bf = boundary_f_score(pred_mask, ref.hit)
            push("render/boundary_f", bf["boundary_f"])

            joint = pred_mask & ref.hit
            # mean_depth(), not the raw accumulation: out.depth is sum_i w_i tau_i, which
            # is short of the surface distance by roughly a factor of alpha, while ref.depth
            # is a geometric distance. Comparing them directly reports an
            # opacity-proportional bias as depth error.
            for k, v in depth_rmse(out.mean_depth(), ref.depth, valid=joint).items():
                push(f"render/{k}", v)
            for k, v in normal_angular_error(out.normal, ref.normal, valid=joint).items():
                push(f"render/{k}", v)

            app = appearance_report(out.color, ref.intensity, roi=ref.hit)
            for k, v in app.to_dict().items():
                push(f"render/{k}", v)

            cov = coverage_report(out.alpha, out.n_contributing, ref.hit)
            for k, v in cov.to_dict().items():
                push(f"render/{k}", v)

            if ci == 0:
                rendered_seq.append(out.color.detach().clone())
                reference_seq.append(ref.intensity.detach().clone())

    result = {k: sum(v) / len(v) for k, v in acc.items()}
    if len(rendered_seq) >= 2:
        for k, v in temporal_report(rendered_seq, reference_seq).items():
            result[f"temporal/{k}"] = v
    return result


@torch.no_grad()
def evaluate_storage_and_speed(
    model: PrecomputedModel,
    cameras: Sequence[Camera],
    *,
    reference_volumes_ml: Sequence[float] | None = None,
    playback_loops: int = 2,
) -> dict[str, object]:
    """Eq. (36)-(38) storage, Eq. (34) playback latency and the volume curve."""
    out: dict[str, object] = {}

    store = model_storage_report(model)
    out["storage"] = store
    primary = store.get("primary", {})
    if isinstance(primary, dict):
        out["storage/compression_ratio"] = primary.get("compression_ratio", float("nan"))
        out["storage/s_ours_mb"] = primary.get("s_ours_mb", float("nan"))
        out["storage/s_full_mb"] = primary.get("s_full_mb", float("nan"))

    report = measure_playback(model, cameras, loops=playback_loops, compute_aux=False)
    out["playback"] = report.summary()

    out["clinical"] = volume_curve_report(
        model.phis,
        model.grid,
        reference_volumes_ml=reference_volumes_ml,
        smooth_eps=model.config.chanvese.eps_heaviside,
    )

    out["precompute"] = {
        "total_ms": model.total_precompute_ms(),
        "per_frame": [f.to_dict() for f in model.frames],
    }
    return out


@dataclass
class EvaluationResult:
    """Everything measured for one (baseline, dataset) pair."""

    name: str
    geometry: dict[str, float] = field(default_factory=dict)
    rendering: dict[str, float] = field(default_factory=dict)
    storage_speed: dict[str, object] = field(default_factory=dict)
    model_summary: dict[str, float] = field(default_factory=dict)
    spec: dict[str, str] = field(default_factory=dict)

    def flat(self) -> dict[str, object]:
        out: dict[str, object] = {"name": self.name}
        out.update(self.spec)
        out.update(self.geometry)
        out.update(self.rendering)
        out.update({f"model/{k}": v for k, v in self.model_summary.items()})
        for k, v in self.storage_speed.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if not isinstance(v2, (dict, list)):
                        out[f"{k}/{k2}"] = v2
            elif not isinstance(v, list):
                out[k] = v
        return out

    def headline(self) -> dict[str, object]:
        """The handful of numbers the research questions actually turn on.

        Quality and cost are in the same row deliberately, so a configuration that buys
        quality by spending more cannot be reported as an improvement.  Note that the two
        cost figures are in **different currencies**: ``precompute_ms`` is paid once and
        offline, ``p95_frame_ms`` is paid on every displayed frame and is the one the
        33.3 ms budget applies to.  See :meth:`cost_quality` for the form that keeps them
        apart and reports dominance.
        """
        pb = self.storage_speed.get("playback", {})
        pb = pb if isinstance(pb, dict) else {}
        return {
            "name": self.name,
            "dice": self.geometry.get("geom/dice_mean"),
            "hd95_mm": self.geometry.get("geom/hd95_mean_mm"),
            "e_surf_mm": self.geometry.get("geom/e_surf_mean_mm"),
            "iou": self.rendering.get("render/best_iou"),
            "boundary_f": self.rendering.get("render/boundary_f"),
            "depth_rmse_mm": self.rendering.get("render/depth_rmse_mm"),
            "normal_deg": self.rendering.get("render/normal_mean_deg"),
            "psnr_roi_db": self.rendering.get("render/psnr_roi_db"),
            "ssim_roi": self.rendering.get("render/ssim_roi"),
            "flicker": self.rendering.get("temporal/e_flicker"),
            "compression_ratio": self.storage_speed.get("storage/compression_ratio"),
            # --- cost: one-time ---
            "precompute_ms": self.model_summary.get("total_precompute_ms"),
            # --- cost: per displayed frame ---
            "p95_frame_ms": pb.get("p95_total_ms"),
            "mean_fps": pb.get("mean_fps"),
            "meets_30fps": pb.get("meets_30fps_p95"),
            "headroom_ratio": pb.get("headroom_ratio"),
            # --- which stage is responsible when the budget is missed ---
            "p95_surface_ms": pb.get("p95_surface_ms"),
            "p95_project_ms": pb.get("p95_project_ms"),
            "p95_orient_ms": pb.get("p95_orient_ms"),
            "p95_raster_ms": pb.get("p95_raster_ms"),
        }

    def cost_quality(self) -> "CostQualityPoint":
        """This result as a point in quality-versus-cost space.

        Feeding these to :func:`cvdyn2dgs.metrics.costquality.pareto_front` is what turns a
        wide table into a statement: a configuration marked *dominated* is strictly worse
        than another on every measured axis and needs no further argument, while several
        configurations on the frontier means there is a real trade-off to discuss.
        """
        from ..metrics.costquality import Cost, CostQualityPoint, QualityVector

        pb = self.storage_speed.get("playback", {})
        pb = pb if isinstance(pb, dict) else {}
        s_ours_mb = self.storage_speed.get("storage/s_ours_mb")
        return CostQualityPoint(
            name=self.name,
            quality=QualityVector(
                psnr_roi_db=self.rendering.get("render/psnr_roi_db"),
                ssim_roi=self.rendering.get("render/ssim_roi"),
                iou=self.rendering.get("render/best_iou"),
                dice=self.geometry.get("geom/dice_mean"),
                e_surf_mm=self.geometry.get("geom/e_surf_mean_mm"),
                hd95_mm=self.geometry.get("geom/hd95_mean_mm"),
                depth_rmse_mm=self.rendering.get("render/depth_rmse_mm"),
                normal_deg=self.rendering.get("render/normal_mean_deg"),
                flicker=self.rendering.get("temporal/e_flicker"),
            ),
            cost=Cost(
                precompute_ms=self.model_summary.get("total_precompute_ms"),
                p95_frame_ms=pb.get("p95_total_ms"),
                mean_frame_ms=pb.get("mean_total_ms"),
                bytes_total=(
                    None if not isinstance(s_ours_mb, (int, float))
                    else int(s_ours_mb * 1024**2)
                ),
            ),
            notes=self.spec.get("isolates", ""),
        )


def cost_quality_comparison(results: Sequence["EvaluationResult"]) -> dict[str, object]:
    """Quality against cost across baselines, with the dominance verdict.

    The frontier is the honest summary: it says which configurations are genuine choices
    without inventing an exchange rate between offline minutes, per-frame milliseconds and
    megabytes on disk.
    """
    from ..metrics.costquality import cost_quality_rows, pareto_front

    points = [r.cost_quality() for r in results]
    rows = cost_quality_rows(points)
    front = [p.name for p in pareto_front(points)]
    return {
        "rows": rows,
        "frontier": front,
        "dominated": [r["name"] for r in rows if r["pareto"] == "dominated"],
        "n_compared": len(points),
    }


def _require_runnable(spec: BaselineSpec) -> None:
    """Refuse to evaluate a baseline this runner cannot honestly realise.

    Six baselines declare an axis that :func:`run_pipeline_on_phantom` does not implement:
    the layer-1 specs carry a ``surface_source`` and the layer-2 free-geometry specs need a
    model whose geometry is optimised rather than pinned.  Running them through the standard
    surfel pipeline would silently produce ordinary ``cv-dyn2dgs`` numbers **labelled** as
    something else, which is worse than not running them at all - a reader would take the
    oracle row as an oracle result.

    ``source-oracle`` is the exception: on the phantom an exact analytic level set exists,
    so it is genuinely realisable and is wired up below.
    """
    src = spec.extra.get("surface_source")
    if src in ("mask",):
        raise NotImplementedError(
            f"baseline {spec.name!r} needs an external segmentation as its surface source. "
            f"Supply one via levelset.surface_source.MaskSequenceSource; there is nothing "
            f"to derive it from on the phantom. Reported as not measured rather than run "
            f"with the Chan-Vese surface under a misleading name."
        )
    if spec.extra.get("spacing_aware") is False:
        raise NotImplementedError(
            f"baseline {spec.name!r} varies spacing_aware, which this runner does not "
            f"thread through precompute yet. Note also that the ablation is vacuous on an "
            f"isotropic phantom - run it on anisotropic data or with a phantom whose "
            f"spacing ratio is set deliberately."
        )
    if spec.renderer == "free3dgs":
        raise NotImplementedError(
            f"baseline {spec.name!r} optimises its own geometry, so it needs "
            f"surfel.free3dgs.fit_free_3dgs rather than the surfel precompute path. "
            f"evaluate_all() would otherwise score a surface-pinned model under the "
            f"free-geometry name and invert the comparison this baseline exists for."
        )


def evaluate_all(
    spec: BaselineSpec,
    phantom: Phantom4D,
    *,
    cameras: EvalCameras | None = None,
    generator: torch.Generator | None = None,
    eval_frames: Sequence[int] | None = None,
    verbose: bool = False,
) -> EvaluationResult:
    """Run one baseline end-to-end on a phantom and evaluate everything.

    Raises :class:`NotImplementedError` for baselines whose distinguishing axis this runner
    cannot realise - see :func:`_require_runnable`.  That is deliberate: a mislabelled row
    is worse than an absent one.
    """
    _require_runnable(spec)

    oracle = spec.extra.get("surface_source") == "oracle"
    model, cams, _ = run_pipeline_on_phantom(
        phantom, spec.config, cameras=cameras, generator=generator, verbose=verbose,
        oracle_surface=oracle,
    )

    ed = phantom.ed_index()
    order = [(ed + t) % phantom.n_frames for t in range(phantom.n_frames)]
    images = [phantom.images[t] for t in order]
    gt = {t: phantom.masks[order[t]] for t in range(len(order))}
    ref_vols = [phantom.analytic_volumes_ml()[t] for t in order]

    geom = evaluate_geometry(model, gt_masks=gt)
    rend = evaluate_rendering(
        model, spec, images, cams.eval, frames=eval_frames
    )
    spd = evaluate_storage_and_speed(model, cams.eval, reference_volumes_ml=ref_vols)

    return EvaluationResult(
        name=spec.name,
        geometry=geom,
        rendering=rend,
        storage_speed=spd,
        model_summary=model.summary(),
        spec=spec.summary(),
    )
