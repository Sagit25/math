"""Ablations of proposal §8.4.

Each entry toggles exactly one mechanism against the full ``v3-adaptive`` configuration,
so the measured change is attributable.  The list follows the proposal:

* previous level-set seed (warm start) on/off
* physical voxel spacing respected or ignored
* projection iterations :math:`K_p \\in \\{1, 2, 3, 5\\}`
* closest-point versus normal-only projection
* isotropic versus anisotropic disks (Eq. 7.6-7.7)
* 2DGS versus thin 3DGS versus mesh primitives
* the individual effect of the mask / normal / depth-distortion losses
* intensity residual and temporal regularisation on/off
* full versus low-rank residual
* surface repulsion / densification on/off
* FPS versus surfel count and viewer resolution

The spacing ablation deserves a note: it is the *only* direct test of the paper's first
claimed contribution.  Setting ``spacing_aware=False`` is equivalent to pretending
:math:`h_x=h_y=h_z=1`, which on a stack with :math:`h_z/h_x \\approx 6` should visibly
distort curvature-driven regularisation along the slice direction.  If it does not, the
contribution is not doing what the theory says it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import torch

from ..baselines import BaselineSpec, get_baseline
from ..core.config import LossConfig, PipelineConfig, ResidualConfig, get_preset
from ..data.phantom import Phantom4D, PhantomConfig, make_phantom
from ..levelset.chanvese import solve_sequence
from ..metrics.segmentation import dice
from .common import (
    EvalCameras,
    EvaluationResult,
    evaluate_all,
    initial_levelset_from_mask,
    make_eval_cameras,
)

__all__ = ["AblationEntry", "ABLATIONS", "run_ablation", "run_all_ablations", "spacing_ablation"]


@dataclass
class AblationEntry:
    """One ablation: a name, a config mutation, and what it is supposed to show."""

    name: str
    mutate: Callable[[PipelineConfig], PipelineConfig]
    tests: str
    renderer: str = "2dgs"
    extra: dict[str, object] = field(default_factory=dict)

    def spec(self) -> BaselineSpec:
        cfg = self.mutate(get_preset("v3-adaptive"))
        cfg.name = self.name
        return BaselineSpec(
            name=self.name,
            description=self.tests,
            config=cfg,
            renderer=self.renderer,  # type: ignore[arg-type]
            isolates=self.tests,
            extra=dict(self.extra),
        )


def _m(fn: Callable[[PipelineConfig], None]) -> Callable[[PipelineConfig], PipelineConfig]:
    """Wrap an in-place mutation so it returns the config."""

    def wrapped(cfg: PipelineConfig) -> PipelineConfig:
        fn(cfg)
        return cfg

    return wrapped


def _set_kp(k: int) -> Callable[[PipelineConfig], PipelineConfig]:
    return _m(lambda c: setattr(c, "surfel", replace(c.surfel, projection_iters=k)))


ABLATIONS: list[AblationEntry] = [
    AblationEntry(
        "full",
        lambda c: c,
        "the unmodified v3-adaptive configuration (reference row)",
    ),
    AblationEntry(
        "no-warm-start",
        _m(lambda c: setattr(c, "chanvese", replace(c.chanvese, warm_start=False))),
        "Eq. (5.1): does seeding from the previous frame matter?",
    ),
    AblationEntry("kp=1", _set_kp(1), "Eq. (6.3) with K_p = 1"),
    AblationEntry("kp=2", _set_kp(2), "Eq. (6.3) with K_p = 2 (Prop. 6.2's recommendation)"),
    AblationEntry("kp=3", _set_kp(3), "Eq. (6.3) with K_p = 3"),
    AblationEntry("kp=5", _set_kp(5), "Eq. (6.3) with K_p = 5 - diminishing returns expected"),
    AblationEntry(
        "closest-point",
        _m(lambda c: setattr(c, "surfel", replace(c.surfel, projection_mode="closest_point"))),
        "drop the normal constraint of Eq. (6.6)",
    ),
    AblationEntry(
        "isotropic",
        _m(lambda c: setattr(c, "surfel", replace(c.surfel, isotropic=True))),
        "Prop. 7.5: isotropic disks make the in-plane gauge irrelevant",
    ),
    AblationEntry(
        "no-mask-loss",
        _m(lambda c: setattr(c, "loss", replace(c.loss, lambda_mask=0.0))),
        "the silhouette term of Eq. (33)",
    ),
    AblationEntry(
        "no-normal-loss",
        _m(lambda c: setattr(c, "loss", replace(c.loss, lambda_normal=0.0))),
        "the normal-consistency term of Eq. (33)",
    ),
    AblationEntry(
        "no-dist-loss",
        _m(lambda c: setattr(c, "loss", replace(c.loss, lambda_dist=0.0))),
        "the depth-distortion term of Eq. (33)",
    ),
    AblationEntry(
        "no-geometry-losses",
        _m(lambda c: setattr(c, "loss", LossConfig(lambda_mask=0.0, lambda_normal=0.0, lambda_dist=0.0))),
        "all three geometry terms of Eq. (33) at once",
    ),
    AblationEntry(
        "no-residual",
        _m(lambda c: setattr(c, "residual", ResidualConfig(enabled=False))),
        "Eq. (25): the appearance residual",
    ),
    AblationEntry(
        "no-temporal-reg",
        _m(lambda c: setattr(c, "residual", replace(c.residual, lambda_T=0.0))),
        "the temporal term of Eq. (9.2) - expected to raise flicker",
    ),
    AblationEntry(
        "full-rank-residual",
        _m(lambda c: setattr(c, "residual", replace(c.residual, lowrank_rank=None))),
        "Eq. (9.5): low-rank compression off",
    ),
    AblationEntry(
        "lowrank-r4",
        _m(lambda c: setattr(c, "residual", replace(c.residual, lowrank_rank=4))),
        "Eq. (9.5) with r = 4",
    ),
    AblationEntry(
        "no-repulsion",
        _m(lambda c: setattr(c, "surfel", replace(c.surfel, repulsion_enabled=False))),
        "tangential repulsion (proposal §6.4)",
    ),
    AblationEntry(
        "no-densify",
        _m(lambda c: setattr(c, "surfel", replace(c.surfel, densify_enabled=False, prune_enabled=False))),
        "bounded densification and pruning",
    ),
    AblationEntry(
        "no-density-control",
        _m(
            lambda c: setattr(
                c,
                "surfel",
                replace(c.surfel, repulsion_enabled=False, densify_enabled=False, prune_enabled=False),
            )
        ),
        "all density control at once",
    ),
    AblationEntry(
        "no-curvature-scale",
        _m(lambda c: setattr(c, "surfel", replace(c.surfel, curvature_adaptive_scale=False))),
        "Prop. 8.3: curvature-adaptive disk radii",
    ),
]


def run_ablation(
    entry: AblationEntry,
    phantom: Phantom4D,
    *,
    cameras: EvalCameras | None = None,
    generator: torch.Generator | None = None,
) -> EvaluationResult:
    """Evaluate one ablation entry."""
    return evaluate_all(entry.spec(), phantom, cameras=cameras, generator=generator)


def run_all_ablations(
    phantom: Phantom4D | None = None,
    *,
    only: Sequence[str] | None = None,
    generator: torch.Generator | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, object]]:
    """Run the ablation table and return one headline row per entry."""
    ph = phantom or make_phantom(PhantomConfig(n_frames=8, shape=(64, 64, 16)))
    cams = make_eval_cameras(ph.grid, device=ph.images[0].device)

    rows: dict[str, dict[str, object]] = {}
    for entry in ABLATIONS:
        if only is not None and entry.name not in only:
            continue
        if verbose:
            print(f"--- ablation: {entry.name} ({entry.tests})")
        try:
            res = run_ablation(entry, ph, cameras=cams, generator=generator)
            rows[entry.name] = {"tests": entry.tests, **res.headline()}
        except Exception as exc:  # noqa: BLE001 - one broken row must not kill the table
            rows[entry.name] = {"tests": entry.tests, "error": f"{type(exc).__name__}: {exc}"}
        if verbose:
            print(f"    {rows[entry.name]}")

    # Primitive comparison shares the pipeline but changes the renderer.
    for name in ("mesh-only", "thin-3dgs"):
        if only is not None and name not in only:
            continue
        if verbose:
            print(f"--- ablation: {name} (primitive comparison)")
        try:
            res = evaluate_all(get_baseline(name), ph, cameras=cams, generator=generator)
            rows[name] = {"tests": "primitive comparison", **res.headline()}
        except Exception as exc:  # noqa: BLE001
            rows[name] = {"tests": "primitive comparison", "error": f"{type(exc).__name__}: {exc}"}

    return rows


def spacing_ablation(*, anisotropy: Sequence[float] = (1.0, 3.0, 6.4)) -> dict[str, object]:
    """Test the paper's first contribution directly: does spacing-awareness matter?

    For each slice-to-in-plane spacing ratio, the same Chan-Vese solve is run with the
    spacing-aware operators of Eq. (4.14)-(4.15) and with spacing ignored, and the two
    are scored against the phantom's exact mask.  Isotropic grids should show little
    difference; strongly anisotropic ones should favour the spacing-aware scheme.

    Anything else is a finding: if the gap does not grow with anisotropy, the
    contribution is not doing what theory §4.4 claims.
    """
    out: dict[str, object] = {}
    for ratio in anisotropy:
        cfg = PhantomConfig(
            shape=(64, 64, max(8, int(48 / ratio))),
            spacing=(1.25, 1.25, 1.25 * ratio),
            n_frames=4,
            noise_sigma=0.03,
        )
        # A phantom is built per ratio: the whole point is to vary hz/hx.
        ph = make_phantom(cfg)
        grid = ph.grid
        images = ph.images
        phi0 = initial_levelset_from_mask(ph.masks[0], grid)

        from ..core.config import ChanVeseConfig

        cv = ChanVeseConfig(max_iters=150, check_every=5)
        row: dict[str, float] = {"hz_over_hx": grid.hz / grid.hx}
        for aware in (True, False):
            seq = solve_sequence(images, phi0, grid, cv, spacing_aware=aware)
            dices = [dice(p > 0, m) for p, m in zip(seq.phis, ph.masks)]
            key = "spacing_aware" if aware else "spacing_ignored"
            row[f"{key}/dice_mean"] = sum(dices) / len(dices)
            row[f"{key}/iterations"] = float(seq.total_iterations)
        row["dice_gain_from_spacing_awareness"] = (
            row["spacing_aware/dice_mean"] - row["spacing_ignored/dice_mean"]
        )
        out[f"anisotropy={ratio}"] = row
    return out
