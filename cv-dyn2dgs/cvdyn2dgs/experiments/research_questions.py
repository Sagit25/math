"""The six research questions of proposal §8.1, each as a runnable experiment.

============  ====================================================================
RQ1           Does the previous level-set seed reduce Chan-Vese iterations and time?
RQ2           Does normal projection preserve surface/rendering quality at lower cost
              than per-frame 2DGS optimisation?
RQ3           Is the canonical + surface + residual representation smaller than storing
              every frame's 2DGS?
RQ4           Does playback exceed 30 FPS including projection, orientation and
              rasterisation?
RQ5           On the same Chan-Vese surface, does 2DGS improve boundary, depth/normal,
              appearance or temporal smoothness over a mesh?
RQ6           Do 2-D disks reduce surface artefacts and view inconsistency relative to
              the same number of thin 3-D Gaussians?
============  ====================================================================

Two principles are applied throughout.

**A negative answer is a result.**  Proposal §4.3 and §10.2 say so explicitly: if
2DGS does not beat a mesh at a comparable primitive budget, "a mesh is the more
appropriate tool" is the correct conclusion.  Nothing here is arranged to avoid that
outcome, and :func:`rq5_vs_mesh` reports per-metric winners rather than a single score.

**Efficiency claims are tested, not assumed.**  Eq. (9) is a hypothesis; RQ1 and RQ2
measure both sides of it separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..baselines import get_baseline
from ..core.config import ChanVeseConfig, get_preset
from ..data.phantom import Phantom4D, PhantomConfig, make_phantom
from ..levelset.chanvese import solve_frame, solve_sequence
from ..pipeline.playback import compare_projection_modes, measure_playback
from .common import (
    EvalCameras,
    EvaluationResult,
    evaluate_all,
    initial_levelset_from_mask,
    make_eval_cameras,
    run_pipeline_on_phantom,
)

__all__ = [
    "RQResult",
    "rq1_warm_start",
    "rq2_projection_vs_independent",
    "rq3_storage",
    "rq4_playback_fps",
    "rq5_vs_mesh",
    "rq6_disk_vs_thin_ellipsoid",
    "progressive_development",
]


@dataclass
class RQResult:
    """One research question's outcome."""

    question: str
    statement: str
    verdict: str
    data: dict[str, object] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "statement": self.statement,
            "verdict": self.verdict,
            "data": self.data,
            "caveats": self.caveats,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.question}: {self.verdict}"


def _default_phantom(device=None, dtype=torch.float32, **kw) -> Phantom4D:
    cfg = PhantomConfig(**kw)
    return make_phantom(cfg, device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
#  RQ1 - warm start
# --------------------------------------------------------------------------- #
def rq1_warm_start(
    phantom: Phantom4D | None = None,
    *,
    tol_mm: float = 0.15,
) -> RQResult:
    """RQ1 / Thm. 5.5: does warm starting reduce the iteration count?

    Two measurements are made, because they answer different things:

    * **iterations and wall time** for the whole sequence with and without the warm
      start - the practical question;
    * :math:`K(e_0)` versus a high-accuracy **reference solution** per frame - the
      quantity Eq. (5.6) actually bounds, together with the initial error
      :math:`e_0` that Def. 5.1 assumes is small.

    Note the confound, and it is reported rather than hidden: disabling the warm start
    also disables the narrow-band crop (the band is defined from the warm start), so
    the wall-time gap mixes two effects while the iteration gap does not.
    """
    ph = phantom or _default_phantom(n_frames=12, shape=(64, 64, 16))
    grid = ph.grid
    ed = ph.ed_index()
    order = [(ed + t) % ph.n_frames for t in range(ph.n_frames)]
    images = [ph.images[t] for t in order]
    phi0 = initial_levelset_from_mask(ph.masks[order[0]], grid)

    base = ChanVeseConfig(max_iters=200, check_every=5)
    warm = solve_sequence(images, phi0, grid, base, track_energy=False)
    cold = solve_sequence(
        images, phi0, grid, ChanVeseConfig(**{**base.__dict__, "warm_start": False})
    )

    # K(e_0) against a converged reference, per frame (Eq. 5.6).
    ref_cfg = ChanVeseConfig(max_iters=1200, check_every=5, tol_band_change=1e-7)
    k_warm: list[int] = []
    k_cold: list[int] = []
    e0_warm: list[float] = []
    e0_cold: list[float] = []
    for t in range(1, min(4, len(images))):
        ref = solve_frame(images[t], warm.phis[t - 1], grid, ref_cfg).phi
        rw = solve_frame(
            images[t], warm.phis[t - 1], grid, base,
            reference_phi=ref, tol_to_reference=tol_mm,
        )
        rc = solve_frame(
            images[t], phi0, grid, base,
            reference_phi=ref, tol_to_reference=tol_mm,
        )
        if rw.iters_to_reference is not None:
            k_warm.append(rw.iters_to_reference)
        if rc.iters_to_reference is not None:
            k_cold.append(rc.iters_to_reference)
        if rw.initial_distance_to_reference is not None:
            e0_warm.append(rw.initial_distance_to_reference)
        if rc.initial_distance_to_reference is not None:
            e0_cold.append(rc.initial_distance_to_reference)

    warm_it = warm.total_iterations
    cold_it = cold.total_iterations
    faster = warm_it < cold_it
    data: dict[str, object] = {
        "warm": warm.summary(),
        "cold": cold.summary(),
        "iteration_ratio_cold_over_warm": cold_it / max(1, warm_it),
        "time_ratio_cold_over_warm": cold.total_time_ms / max(1e-9, warm.total_time_ms),
        "K_warm_mean": sum(k_warm) / len(k_warm) if k_warm else float("nan"),
        "K_cold_mean": sum(k_cold) / len(k_cold) if k_cold else float("nan"),
        "e0_warm_mean": sum(e0_warm) / len(e0_warm) if e0_warm else float("nan"),
        "e0_cold_mean": sum(e0_cold) / len(e0_cold) if e0_cold else float("nan"),
        "tol_mm": tol_mm,
        "mean_crop_fraction_warm": warm.summary()["mean_crop_fraction"],
    }
    return RQResult(
        question="RQ1",
        statement="Does the previous level-set seed reduce Chan-Vese iterations and time?",
        verdict=(
            f"warm start used {warm_it} total iterations vs {cold_it} cold "
            f"({'fewer' if faster else 'NOT fewer'}); "
            f"wall time ratio {data['time_ratio_cold_over_warm']:.2f}x"
        ),
        data=data,
        caveats=[
            "Disabling the warm start also disables the narrow-band crop, so the wall-time "
            "ratio mixes both effects; the iteration ratio isolates the initialisation.",
            "Thm. 5.5 is a statement about a local basin (Def. 5.3 assumes local PL and "
            "L-smoothness). It predicts an ordering, not an absolute iteration count.",
        ],
    )


# --------------------------------------------------------------------------- #
#  RQ2 - projection vs independent refit
# --------------------------------------------------------------------------- #
def rq2_projection_vs_independent(
    phantom: Phantom4D | None = None,
    *,
    cameras: EvalCameras | None = None,
    generator: torch.Generator | None = None,
) -> RQResult:
    """RQ2 / Eq. (9): is the per-frame update cheaper than a full refit, at what quality cost?"""
    ph = phantom or _default_phantom(n_frames=10, shape=(64, 64, 16))
    cams = cameras or make_eval_cameras(ph.grid, device=ph.images[0].device)

    ours = evaluate_all(get_baseline("cv-dyn2dgs"), ph, cameras=cams, generator=generator)
    indep = evaluate_all(get_baseline("independent-2dgs"), ph, cameras=cams, generator=generator)

    h_ours, h_ind = ours.headline(), indep.headline()
    t_ours = float(h_ours.get("precompute_ms") or float("nan"))
    t_ind = float(h_ind.get("precompute_ms") or float("nan"))

    def delta(key: str) -> float:
        a, b = h_ours.get(key), h_ind.get(key)
        if a is None or b is None:
            return float("nan")
        return float(a) - float(b)

    return RQResult(
        question="RQ2",
        statement="Does normal projection keep quality at lower cost than per-frame 2DGS fitting?",
        verdict=(
            f"precompute {t_ours:.0f} ms vs {t_ind:.0f} ms "
            f"({t_ind / max(t_ours, 1e-9):.2f}x); "
            f"dIoU={delta('iou'):+.4f}, dPSNR_roi={delta('psnr_roi_db'):+.2f} dB, "
            f"dSSIM_roi={delta('ssim_roi'):+.4f}"
        ),
        data={
            "ours": h_ours,
            "independent_2dgs": h_ind,
            "speedup": t_ind / max(t_ours, 1e-9),
            "quality_deltas": {
                k: delta(k)
                for k in ("iou", "boundary_f", "depth_rmse_mm", "normal_deg", "psnr_roi_db", "ssim_roi", "flicker")
            },
            "per_frame_update_ms_ours": ours.model_summary.get("mean_per_frame_update_ms"),
            "refit_ms_independent": indep.model_summary.get("mean_per_frame_update_ms"),
        },
        caveats=[
            "Independent 2DGS is the quality ceiling. The target is non-inferiority inside a "
            "pre-registered margin (proposal §8.5), not winning.",
            "Margins must be fixed from a ~5-case pilot before the main run and then left "
            "alone; this function does not choose them for you.",
        ],
    )


# --------------------------------------------------------------------------- #
#  RQ3 - storage
# --------------------------------------------------------------------------- #
def rq3_storage(
    phantom: Phantom4D | None = None,
    *,
    presets: Sequence[str] = ("v1-minimal", "v3-adaptive"),
    generator: torch.Generator | None = None,
) -> RQResult:
    """RQ3 / Eq. (36)-(38): is the compact representation actually smaller?"""
    ph = phantom or _default_phantom(n_frames=12, shape=(64, 64, 16))
    cams = make_eval_cameras(ph.grid, device=ph.images[0].device)

    out: dict[str, object] = {}
    ratios: dict[str, float] = {}
    for name in presets:
        cfg = get_preset(name)
        model, _, _ = run_pipeline_on_phantom(ph, cfg, cameras=cams, generator=generator)
        from ..pipeline.storage_io import model_storage_report, save_model
        import tempfile, os

        rep = model_storage_report(model)
        with tempfile.TemporaryDirectory() as td:
            measured = save_model(model, os.path.join(td, "model.pt"))
        by_mode = rep["by_surface_mode"]
        assert isinstance(by_mode, dict)
        out[name] = {
            "theoretical": by_mode,
            "measured_file": measured,
            "p2d_minimal": rep["p2d_minimal"],
            "p2d_explicit": rep["p2d_explicit"],
        }
        nb = by_mode.get("narrow_band_sdf", {})
        if isinstance(nb, dict):
            ratios[name] = float(nb.get("compression_ratio", float("nan")))

    best = max(ratios.values()) if ratios else float("nan")
    return RQResult(
        question="RQ3",
        statement="Is canonical 2DGS + per-frame surface + residual smaller than storing all frames?",
        verdict=(
            f"best compression ratio {best:.2f}x (narrow-band SDF surfaces); "
            f"{'a win' if best > 1.0 else 'NOT a win - storing every frame is cheaper here'}"
        ),
        data={"per_preset": out, "compression_ratios": ratios},
        caveats=[
            "Reported for all three surface storage modes, not just the cheapest. The mask "
            "mode is smallest but quantises the surface to the voxel grid.",
            "Theoretical byte counts (Eq. 37) are reported next to the measured file size; a "
            "gap is serialisation overhead, not a flaw in the accounting.",
            "Proposal §2.9 warns this can fail: if the surfaces and residuals are large the "
            "ratio drops below 1, and the break-even surface budget is reported so that is "
            "visible.",
        ],
    )


# --------------------------------------------------------------------------- #
#  RQ4 - playback speed
# --------------------------------------------------------------------------- #
def rq4_playback_fps(
    phantom: Phantom4D | None = None,
    *,
    resolutions: Sequence[int] = (256, 512),
    surfel_counts: Sequence[int] = (10_000, 20_000),
    generator: torch.Generator | None = None,
) -> RQResult:
    """RQ4 / Eq. (34): does the viewer stay under the 33.3 ms frame budget?

    Sweeps resolution and surfel count, because a single configuration says nothing
    about where the budget breaks.  The verdict uses the **95th percentile**, not the
    mean - proposal §8.3 asks for both, and a viewer that stutters is not interactive.
    """
    ph = phantom or _default_phantom(n_frames=10, shape=(64, 64, 16))
    rows: list[dict[str, object]] = []

    for n_surf in surfel_counts:
        cfg = get_preset("v3-adaptive")
        cfg.surfel.n_surfels = int(n_surf)
        model, _, _ = run_pipeline_on_phantom(ph, cfg, generator=generator)
        for res in resolutions:
            cams = make_eval_cameras(
                ph.grid, device=ph.images[0].device, n_orbit=8, resolution=int(res)
            ).eval
            rep = measure_playback(model, cams, loops=2, compute_aux=False)
            row = {"n_surfels_requested": float(n_surf), **rep.summary()}
            rows.append(row)
        seek = compare_projection_modes(model, cams[0])
        rows[-1]["seek_comparison"] = seek

    ok = [r for r in rows if r.get("meets_30fps_p95")]
    return RQResult(
        question="RQ4",
        statement="Does playback (projection + orientation + rasterisation) exceed 30 FPS?",
        verdict=(
            f"{len(ok)}/{len(rows)} configurations met the 33.3 ms p95 budget"
            + (
                f"; best {max((float(r['mean_fps']) for r in rows), default=float('nan')):.0f} FPS mean"
                if rows
                else ""
            )
        ),
        data={"configurations": rows},
        caveats=[
            "Timing is hardware dependent; every result file records the device via "
            "describe_environment(). Theory §11.3 lists the frame budget as to-be-verified "
            "on real hardware for exactly this reason.",
            "Depth/normal/distortion buffers are off during the FPS measurement, since they "
            "are needed for quality metrics and not for display.",
            "Random seeking is measured separately (chained vs canonical projection): Eq. (6) "
            "stores no per-frame anchors, so a seek must project from the canonical set.",
        ],
    )


# --------------------------------------------------------------------------- #
#  RQ5 - versus a mesh
# --------------------------------------------------------------------------- #
def rq5_vs_mesh(
    phantom: Phantom4D | None = None,
    *,
    generator: torch.Generator | None = None,
) -> RQResult:
    """RQ5: on the same surface, does 2DGS beat a strong textured mesh?"""
    ph = phantom or _default_phantom(n_frames=8, shape=(64, 64, 16))
    cams = make_eval_cameras(ph.grid, device=ph.images[0].device)

    gs = evaluate_all(get_baseline("cv-dyn2dgs"), ph, cameras=cams, generator=generator)
    mesh = evaluate_all(get_baseline("mesh-only"), ph, cameras=cams, generator=generator)

    higher_better = {
        "iou": True,
        "boundary_f": True,
        "psnr_roi_db": True,
        "ssim_roi": True,
        "depth_rmse_mm": False,
        "normal_deg": False,
        "flicker": False,
    }
    winners: dict[str, str] = {}
    h_gs, h_mesh = gs.headline(), mesh.headline()
    for k, hib in higher_better.items():
        a, b = h_gs.get(k), h_mesh.get(k)
        if a is None or b is None or a != a or b != b:
            winners[k] = "n/a"
            continue
        winners[k] = "2dgs" if ((a > b) == hib) else "mesh"

    n_gs = sum(1 for v in winners.values() if v == "2dgs")
    n_mesh = sum(1 for v in winners.values() if v == "mesh")
    return RQResult(
        question="RQ5",
        statement="Does 2DGS improve boundary / depth / normal / appearance / smoothness over a mesh?",
        verdict=(
            f"2DGS wins {n_gs} metrics, mesh wins {n_mesh}. "
            + (
                "2DGS is justified for this purpose."
                if n_gs > n_mesh
                else "On this evidence a mesh is the more appropriate tool - which proposal "
                "§4.3 accepts as a valid conclusion."
            )
        ),
        data={"winners": winners, "2dgs": h_gs, "mesh": h_mesh},
        caveats=[
            "The mesh is deliberately strong: level-set vertex normals, MRI-sampled vertex "
            "amplitudes, perspective-correct interpolation, exact z-buffer visibility.",
            "Primitive budgets are not identical - the mesh has its own vertex/face count from "
            "marching tetrahedra. Both counts are reported so the comparison can be read at "
            "matched budget.",
            "A mesh has binary coverage and no opacity, so silhouette IoU is swept over alpha "
            "thresholds instead of fixed at 0.5.",
        ],
    )


# --------------------------------------------------------------------------- #
#  RQ6 - disk versus thin ellipsoid
# --------------------------------------------------------------------------- #
def rq6_disk_vs_thin_ellipsoid(
    phantom: Phantom4D | None = None,
    *,
    thickness_ratios: Sequence[float] = (0.05, 0.1, 0.3),
    generator: torch.Generator | None = None,
) -> RQResult:
    """RQ6 / Prop. 8.4: do 2-D disks reduce depth/normal error against thin 3-D Gaussians?"""
    ph = phantom or _default_phantom(n_frames=8, shape=(64, 64, 16))
    cams = make_eval_cameras(ph.grid, device=ph.images[0].device)

    gs = evaluate_all(get_baseline("cv-dyn2dgs"), ph, cameras=cams, generator=generator)
    sweep: dict[str, object] = {}
    from ..baselines import _thin_3dgs  # noqa: PLC2701 - intentional, parameterised spec

    for rho in thickness_ratios:
        spec = _thin_3dgs(rho)
        res = evaluate_all(spec, ph, cameras=cams, generator=generator)
        sweep[f"rho={rho}"] = res.headline()

    h = gs.headline()
    best_3d_depth = min(
        (float(v["depth_rmse_mm"]) for v in sweep.values() if isinstance(v, dict) and v.get("depth_rmse_mm") == v.get("depth_rmse_mm")),
        default=float("nan"),
    )
    d2 = float(h.get("depth_rmse_mm") or float("nan"))
    return RQResult(
        question="RQ6",
        statement="Do 2-D disks reduce surface artefacts and view inconsistency vs thin 3-D Gaussians?",
        verdict=(
            f"2DGS depth RMSE {d2:.4f} mm vs best thin-3DGS {best_3d_depth:.4f} mm "
            f"({'2DGS better' if d2 < best_3d_depth else 'thin 3DGS better or equal'})"
        ),
        data={"2dgs": h, "thin3dgs_sweep": sweep},
        caveats=[
            "The thin-3DGS depth map is constant across each footprint by construction "
            "(affine projection), so a depth advantage for 2DGS is expected and is a direct "
            "test of Prop. 8.4 rather than an independent finding.",
            "Thinner ellipsoids are geometrically closer to a disk but worse conditioned; the "
            "sweep exposes that trade-off instead of picking one value.",
        ],
    )


# --------------------------------------------------------------------------- #
#  Progressive development: v1 -> v2 -> v3
# --------------------------------------------------------------------------- #
def progressive_development(
    phantom: Phantom4D | None = None,
    *,
    presets: Sequence[str] = ("v1-minimal", "v2-geometry", "v3-adaptive"),
    generator: torch.Generator | None = None,
) -> RQResult:
    """Measure what each development stage actually buys.

    This is the staged improvement of the method, evaluated rather than asserted.  Each
    preset adds one group of mechanisms (see :mod:`cvdyn2dgs.core.config`), and the table
    shows the effect on quality, cost and storage together - so a stage that improves
    quality only by spending more can be identified as such.
    """
    ph = phantom or _default_phantom(n_frames=8, shape=(64, 64, 16))
    cams = make_eval_cameras(ph.grid, device=ph.images[0].device)

    results: list[EvaluationResult] = []
    for name in presets:
        spec = get_baseline(name)
        results.append(evaluate_all(spec, ph, cameras=cams, generator=generator))

    table = [r.headline() for r in results]
    deltas: dict[str, dict[str, float]] = {}
    for i in range(1, len(table)):
        prev, cur = table[i - 1], table[i]
        d: dict[str, float] = {}
        for k in ("iou", "boundary_f", "depth_rmse_mm", "normal_deg", "psnr_roi_db", "ssim_roi", "flicker", "e_surf_mm", "precompute_ms", "compression_ratio"):
            a, b = cur.get(k), prev.get(k)
            if a is None or b is None or a != a or b != b:
                continue
            d[k] = float(a) - float(b)
        deltas[f"{presets[i-1]} -> {presets[i]}"] = d

    return RQResult(
        question="progressive-development",
        statement="What does each development stage (v1 -> v2 -> v3) actually improve?",
        verdict="; ".join(
            f"{k}: dPSNR_roi={v.get('psnr_roi_db', float('nan')):+.2f} dB, "
            f"dE_surf={v.get('e_surf_mm', float('nan')):+.4f} mm, "
            f"dtime={v.get('precompute_ms', float('nan')):+.0f} ms"
            for k, v in deltas.items()
        ),
        data={"table": table, "deltas": deltas},
        caveats=[
            "Quality, cost and storage are reported together; an improvement bought purely "
            "with extra computation is visible as such.",
            "Single-phantom results. Patient-level variation needs the paired statistics of "
            "proposal §8.5 (bootstrap CIs, Wilcoxon signed-rank), which need many cases.",
        ],
    )
