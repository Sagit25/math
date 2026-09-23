"""Experiments: theory verification first, then the research questions and ablations.

The ordering matters. ``theory_checks`` validates that the discretisation reproduces the
convergence rates the theory predicts; if it does not, the quality numbers from the
research questions rest on a broken foundation.
"""

from .ablations import ABLATIONS, AblationEntry, run_ablation, run_all_ablations, spacing_ablation
from .common import (
    EvalCameras,
    EvaluationResult,
    evaluate_all,
    evaluate_geometry,
    evaluate_rendering,
    evaluate_storage_and_speed,
    initial_levelset_from_mask,
    make_eval_cameras,
    run_pipeline_on_phantom,
)
from .research_questions import (
    RQResult,
    progressive_development,
    rq1_warm_start,
    rq2_projection_vs_independent,
    rq3_storage,
    rq4_playback_fps,
    rq5_vs_mesh,
    rq6_disk_vs_thin_ellipsoid,
)
from .theory_checks import CheckResult, fit_loglog_slope, run_all_checks

__all__ = [
    "ABLATIONS",
    "AblationEntry",
    "CheckResult",
    "EvalCameras",
    "EvaluationResult",
    "RQResult",
    "evaluate_all",
    "evaluate_geometry",
    "evaluate_rendering",
    "evaluate_storage_and_speed",
    "fit_loglog_slope",
    "initial_levelset_from_mask",
    "make_eval_cameras",
    "progressive_development",
    "run_ablation",
    "run_all_ablations",
    "run_all_checks",
    "run_pipeline_on_phantom",
    "rq1_warm_start",
    "rq2_projection_vs_independent",
    "rq3_storage",
    "rq4_playback_fps",
    "rq5_vs_mesh",
    "rq6_disk_vs_thin_ellipsoid",
    "spacing_ablation",
]
