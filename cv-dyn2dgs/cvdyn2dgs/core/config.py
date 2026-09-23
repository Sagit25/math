"""Configuration dataclasses and the v1/v2/v3 model presets.

The three presets encode the *staged development* of the method:

``v1-minimal``
    The minimum success scope of the proposal (§10.4): warm-started
    spacing-aware Chan-Vese, isotropic normal-projected surfels, a scalar
    intensity residual, no geometry regularisation, no density control.

``v2-geometry``
    Adds the 2DGS geometry losses of Eq. (33) (mask / normal-consistency /
    depth-distortion) and the anisotropic minimum-rotation tangent transport of
    Eq. (7.6)-(7.7), which Prop. 7.6 shows only matters once ``s1 != s2``.

``v3-adaptive``
    Adds the extension scope: curvature-adaptive surfel scales (bounded by the
    ``O(kappa s^2)`` planar-disk error of Prop. 8.3), tangential repulsion plus
    bounded densification/pruning against the clustering/hole failure mode of
    proposal §6.4, temporally regularised residuals and low-rank residual
    compression (Eq. 9.5, Prop. 9.3).

Every field that corresponds to a symbol in the papers carries the equation
number in its comment, so the configuration doubles as a cross-reference table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

__all__ = [
    "ChanVeseConfig",
    "SurfelConfig",
    "RenderConfig",
    "LossConfig",
    "FitConfig",
    "ResidualConfig",
    "PipelineConfig",
    "PRESETS",
    "get_preset",
]


@dataclass
class ChanVeseConfig:
    """Spacing-aware 3-D Chan-Vese (theory §4, proposal Eq. 10-16)."""

    mu: float = 0.15
    """:math:`\\mu` - weight of the length/area regularisation term, Eq. (4.1)."""

    lambda_in: float = 1.0
    """:math:`\\lambda_{\\mathrm{in}}` - inside data fidelity, Eq. (4.1)."""

    lambda_out: float = 1.0
    """:math:`\\lambda_{\\mathrm{out}}` - outside data fidelity, Eq. (4.1)."""

    eps_heaviside: float = 1.5
    """:math:`\\varepsilon` of :math:`H_\\varepsilon, \\delta_\\varepsilon`, Eq. (3.7)-(3.8).

    In mm, because :math:`\\phi` is kept close to a signed distance function.
    """

    eps_div: float = 1e-6
    """Denominator regularisation inside :math:`\\mathrm{div}_h`, Eq. (4.7).

    Prop. 6.3's remark quantifies the first-order residual this leaves behind as
    :math:`O(\\varepsilon / (\\|g\\|^2 + \\varepsilon))`.
    """

    cfl: float = 0.25
    """CFL factor for the explicit Euler step :math:`\\Delta s` of Eq. (4.16).

    The step is chosen adaptively as ``cfl * h_min / max|speed|`` rather than
    fixed, so that the ``L``-smoothness assumption behind Lemma 5.4
    (``alpha <= 1/L``) is not silently violated on high-contrast volumes.
    """

    max_iters: int = 300
    """Iteration cap for one frame's gradient descent."""

    tol_band_change: float = 1e-4
    """Stopping rule: relative change of the narrow-band level-set per iteration."""

    check_every: int = 5
    """How often to evaluate the stopping rule / recompute region means."""

    narrow_band_mm: float = 6.0
    """:math:`b` of Eq. (5.1) - half-width of the warm-start narrow band, in mm."""

    warm_start: bool = True
    """Eq. (5.1). Setting ``False`` gives the cold-start baseline of RQ1."""

    reinit_every: int = 20
    """Iterations between eikonal reinitialisations (keeps ``||grad phi|| ~ 1``,
    Prop. 3.1, which Prop. 6.2 relies on for 1-2 step projection)."""

    reinit_iters: int = 8
    """Sub-iterations of the reinitialisation PDE."""

    frame0_iters_scale: float = 4.0
    """Frame 0 has no warm start, so it gets a larger iteration budget."""


@dataclass
class SurfelConfig:
    """Canonical 2-D Gaussian surfels and their frame-to-frame update."""

    n_surfels: int = 20000
    """:math:`N` in Eq. (7.1)."""

    isotropic: bool = True
    """``s_{i,1} == s_{i,2}``.

    Prop. 7.5 proves the in-plane rotation gauge is *harmless* for isotropic
    disks, which is why v1 uses them: it removes the tangent-transport failure
    mode entirely. v2/v3 switch this off and rely on Eq. (7.6)-(7.7).
    """

    scale_init_factor: float = 0.85
    """Initial disk radius as a multiple of the mean nearest-neighbour spacing.

    Trades hole fraction against overlap; both are measured (proposal §8.3).
    """

    scale_min_mm: float = 0.05
    scale_max_mm: float = 6.0

    curvature_adaptive_scale: bool = False
    """v3. Shrink disks where curvature is high so that the planar-disk error
    :math:`O(\\kappa s^2)` of Prop. 8.3 stays below ``curvature_error_budget_mm``."""

    curvature_error_budget_mm: float = 0.20
    """Target for :math:`\\tfrac12 \\kappa s^2` (Prop. 8.3, Eq. 8.10)."""

    projection_iters: int = 2
    """:math:`K_p` of Eq. (6.3). Prop. 6.2 argues 1-2 suffices near an SDF."""

    projection_eps: float = 1e-6
    """:math:`\\varepsilon` in the projection denominator, Eq. (6.2)."""

    projection_max_step_mm: float = 8.0
    """Trust region on ``|d_i^t|``.

    Proposal §6.3 flags mis-projection across nearby surface branches as the main
    failure mode; capping the step and rejecting sign flips reduces it.
    """

    projection_mode: Literal["normal", "closest_point"] = "normal"
    """``"normal"`` is Eq. (6.3); ``"closest_point"`` is the ablation baseline
    that searches the nearest surface point without the normal constraint."""

    transport_degeneracy_thresh: float = 0.2
    """Threshold on :math:`\\|\\bar e^t_{i,1}\\| = |\\sin\\psi|` (Eq. 7.12).

    Below it, Prop. 7.7's amplification ``gamma / |sin psi|`` is unacceptable and
    we fall back to transporting the *second* axis instead.
    """

    normal_eps: float = 1e-8
    """:math:`\\varepsilon` in Eq. (7.3) / (7.7) normalisations."""

    # ---- v3 density control (proposal §6.4) -------------------------------
    repulsion_enabled: bool = False
    repulsion_steps: int = 2
    repulsion_strength: float = 0.25
    """Tangential-only repulsion: never moves anchors off :math:`\\Gamma_t`
    because the normal component is removed and projection is re-applied."""

    densify_enabled: bool = False
    densify_hole_thresh: float = 0.02
    """Trigger densification when the rendered-alpha hole fraction exceeds this."""

    densify_max_growth: float = 0.25
    """Cap total surfel growth over the sequence, so the storage comparison of
    Eq. (36)-(38) stays honest."""

    prune_enabled: bool = False
    prune_opacity_thresh: float = 0.01
    prune_overlap_thresh: float = 0.35


@dataclass
class RenderConfig:
    """Perspective-correct 2DGS rasterisation (theory §8, proposal Eq. 28-32)."""

    tile: int = 16
    """Tile edge in pixels for the binning pass."""

    max_per_tile: int = 96
    """Depth-sorted surfels kept per tile.

    This is a *truncation* of Eq. (8.6): the nearest ``max_per_tile`` surfels are
    composited and the rest discarded. Because compositing is front-to-back and
    transmittance decays, the discarded tail is almost always occluded, but the
    truncation is reported rather than hidden.
    """

    cull_cos: float = 0.05
    """:math:`c_0` of Prop. 8.1. Surfels with :math:`|n^\\top d_q| < c_0` are
    grazing-angle and their depth is ill-conditioned (condition number
    :math:`1/c_0`), so they are culled."""

    near_mm: float = 1e-3
    """Reject intersections behind the camera, :math:`\\tau^t_i(q) > 0`."""

    gaussian_cutoff: float = 3.0
    """Evaluate the Gaussian out to ``cutoff`` sigmas; beyond that alpha is ~0."""

    min_alpha: float = 1.0 / 255.0
    """Skip negligible contributions."""

    tile_chunk: int = 64
    """Tiles processed per batch; bounds peak memory at
    ``tile_chunk * max_per_tile * tile^2`` floats."""

    background: float = 0.0
    """Background intensity used when compositing (Eq. 8.6 leaves it implicit)."""

    sort_by: Literal["anchor_depth", "pixel_depth"] = "anchor_depth"
    """Compositing order. ``anchor_depth`` sorts once per tile (the standard
    splatting approximation); ``pixel_depth`` re-sorts per pixel using the exact
    :math:`\\tau^t_i(q)` of Eq. (8.2) and is slower but order-exact."""


@dataclass
class LossConfig:
    """Total loss of Eq. (33): ``L = L_app + lm*L_mask + ln*L_normal + ld*L_dist``."""

    lambda_mask: float = 0.0
    """:math:`\\lambda_m` - silhouette agreement with the projected CV mask."""

    lambda_normal: float = 0.0
    """:math:`\\lambda_n` - consistency between rendered normals and Eq. (7.3)."""

    lambda_dist: float = 0.0
    """:math:`\\lambda_d` - depth-distortion penalty."""

    appearance: Literal["l1", "l2", "huber"] = "l1"
    huber_delta: float = 0.05


@dataclass
class FitConfig:
    """Canonical (frame-0) 2DGS fitting, proposal §6.6."""

    iters: int = 600
    lr_amplitude: float = 5e-2
    lr_opacity: float = 2e-2
    lr_scale: float = 5e-3
    lr_tangent: float = 0.0
    """Left at 0 by default: the geometry is pinned to the Chan-Vese surface, so
    the fit only adjusts appearance/opacity/scale (proposal §6.6: "this loss does
    not move the boundary")."""

    optimise_anchor: bool = False
    """Kept off for the same reason. Enabling it breaks the claim that geometry
    comes from :math:`\\Gamma_t` alone."""

    grad_clip: float = 1.0
    log_every: int = 100


@dataclass
class ResidualConfig:
    """Compact appearance residual (theory §9, proposal Eq. 25-27)."""

    enabled: bool = True

    lambda_a: float = 1e-2
    """:math:`\\lambda_a` - Tikhonov term of Eq. (9.2), keeps the residual small."""

    lambda_T: float = 1e-2
    """:math:`\\lambda_T` - temporal term of Eq. (9.2), suppresses flicker.

    Prop. 9.2 needs ``lambda_a + lambda_T > 0`` for the normal equations (9.3) to
    be SPD, hence for a unique solution.
    """

    cg_iters: int = 60
    cg_tol: float = 1e-6
    """Conjugate gradient on Eq. (9.3); the system matrix is never formed."""

    lowrank_rank: int | None = None
    """``r`` of Eq. (9.5). ``None`` stores the full per-frame scalar residual.
    Prop. 9.3 gives the truncation error as the singular-value tail."""


@dataclass
class PipelineConfig:
    """Everything needed to precompute and play back one patient."""

    name: str = "v3-adaptive"
    seed: int = 0
    device: str = "auto"
    dtype: Literal["float32", "float64"] = "float32"

    chanvese: ChanVeseConfig = field(default_factory=ChanVeseConfig)
    surfel: SurfelConfig = field(default_factory=SurfelConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)

    refit_every_frame: bool = False
    """``True`` reproduces the *Independent 2DGS* baseline (proposal §8.2)."""

    store_surface_as: Literal["narrow_band_sdf", "mask", "mesh"] = "narrow_band_sdf"
    """Which representation of :math:`\\Gamma_t` is stored; Eq. (37) is reported
    for all three (proposal §8.3)."""

    narrow_band_store_mm: float = 4.0
    """Band half-width actually written to disk (can be tighter than the solver's)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replaced(self, **kw: Any) -> "PipelineConfig":
        return replace(self, **kw)


# --------------------------------------------------------------------------- #
#  Presets: the staged development of the method
# --------------------------------------------------------------------------- #
def _v1_minimal() -> PipelineConfig:
    return PipelineConfig(
        name="v1-minimal",
        surfel=SurfelConfig(
            isotropic=True,
            projection_iters=1,
            curvature_adaptive_scale=False,
            repulsion_enabled=False,
            densify_enabled=False,
            prune_enabled=False,
        ),
        loss=LossConfig(lambda_mask=0.0, lambda_normal=0.0, lambda_dist=0.0),
        residual=ResidualConfig(enabled=True, lambda_a=1e-2, lambda_T=0.0),
    )


def _v2_geometry() -> PipelineConfig:
    return PipelineConfig(
        name="v2-geometry",
        surfel=SurfelConfig(
            isotropic=False,
            projection_iters=2,
            curvature_adaptive_scale=False,
            repulsion_enabled=False,
            densify_enabled=False,
            prune_enabled=False,
        ),
        loss=LossConfig(lambda_mask=0.5, lambda_normal=0.05, lambda_dist=100.0),
        residual=ResidualConfig(enabled=True, lambda_a=1e-2, lambda_T=1e-2),
    )


def _v3_adaptive() -> PipelineConfig:
    return PipelineConfig(
        name="v3-adaptive",
        surfel=SurfelConfig(
            isotropic=False,
            projection_iters=2,
            curvature_adaptive_scale=True,
            repulsion_enabled=True,
            densify_enabled=True,
            prune_enabled=True,
        ),
        loss=LossConfig(lambda_mask=0.5, lambda_normal=0.05, lambda_dist=100.0),
        residual=ResidualConfig(enabled=True, lambda_a=5e-3, lambda_T=2e-2, lowrank_rank=8),
    )


PRESETS: dict[str, Any] = {
    "v1-minimal": _v1_minimal,
    "v2-geometry": _v2_geometry,
    "v3-adaptive": _v3_adaptive,
}


def get_preset(name: str) -> PipelineConfig:
    """Instantiate a preset by name (``v1-minimal`` / ``v2-geometry`` / ``v3-adaptive``)."""
    try:
        return PRESETS[name]()
    except KeyError as exc:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}") from exc
