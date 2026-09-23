"""The baseline family of proposal §8.2.

Every baseline is expressed as a *configuration of the same pipeline* plus a choice of
rasteriser.  That is deliberate: if the baselines were separate implementations, any
measured difference could come from engineering rather than from the idea under test.
Here the Chan-Vese solver, the supervision views, the loss, the optimiser, the tiling,
the culling thresholds and the metrics are literally shared code, so the only
differences are the ones named in each spec.

=============================  ==================================================
Baseline                        What it isolates
=============================  ==================================================
``mesh-only``                   Is a soft Gaussian silhouette worth anything over a
                                smoothly shaded, textured triangle mesh on the *same*
                                surface? (RQ5)
``thin-3dgs``                   Is the 2-D disk worth anything over a thin 3-D
                                ellipsoid with affine projection? (RQ6)
``independent-2dgs``            Quality ceiling and cost floor: refit every frame from
                                scratch. (RQ2, RQ3)
``param-copy``                  Do anchors need to move at all, or does correcting
                                appearance suffice?
``closest-point``               Does dropping the normal constraint in Eq. (6.6) help
                                or hurt?
``normal-no-residual``          How much does the appearance residual contribute?
``cv-dyn2dgs``                  The full method.
``cold-start-cv``               Does the warm start of Eq. (5.1) actually reduce
                                iterations? (RQ1)
=============================  ==================================================

The gap those eight leave open
------------------------------
All of them are configurations of *this* pipeline.  That is deliberate and good for
attribution, but it means the family contains **no comparison with any published
method**, and it never varies the surface source at all.  Both gaps are now represented:

=============================  ==================================================
Added baseline                   Layer and what it varies
=============================  ==================================================
``source-oracle``               1: exact surface instead of Chan-Vese. Mandatory -
                                without it segmentation error and representation
                                error are confounded.
``source-mask``                 1: an external segmentation (nnU-Net, CSTM) as the
                                surface source.
``source-chanvese-no-spacing``  1: spacing-aware operators turned off, so
                                contribution 1 is measured rather than asserted.
``free-3dgs-surface``           2: geometry fully optimised, seeded on the surface.
``free-3dgs-bbox_random``       2: geometry fully optimised, seeded randomly.
``gaussian-surfel-exact``       2: the Dai et al. z-scale-zero formulation.
=============================  ==================================================

Layer 4 (storage and playback) is not a renderer and therefore not a
:class:`BaselineSpec`: pre-rendered video and a raw 4-D volume have no surface and cannot
be scored on geometry.  They live in :mod:`cvdyn2dgs.metrics.viewpoint`, which refuses
geometry axes structurally.  Genuinely external methods are scored through
:mod:`cvdyn2dgs.experiments.external` against the same ray-marched reference; see
``paper/COMPARISON_TARGETS.md`` and ``external/manifest.json``.

Dyna3DGR
--------
Proposal §5.7 names Dyna3DGR as the closest related work and the most important point
of reference.  It is **not** implemented or reproduced here, and no numbers are
attributed to it.  Reproducing it would need its public code, ACDC data, and a GPU -
none of which are available in this repository's test environment.  Following the
proposal's own instruction for that situation ("if reproduction is difficult, do not
compare numbers directly; report only feasibility and structural differences"),
:data:`DYNA3DGR_COMPARISON` records the structural comparison and the published
figures *as claims from the source*, clearly labelled as not measured here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal

from .core.config import (
    PipelineConfig,
    RenderConfig,
    ResidualConfig,
    get_preset,
)
from .levelset.mesh_extract import TriangleMesh
from .render.camera import Camera
from .render.mesh_render import render_mesh
from .render.raster2dgs import RenderOutput, render_2dgs
from .render.raster3dgs import Thin3DGSConfig, render_thin_3dgs
from .surfel.model import SurfelSet2D

__all__ = [
    "RendererKind",
    "BaselineSpec",
    "BASELINES",
    "INTERNAL_BASELINES",
    "COMPARISON_LAYERS",
    "get_baseline",
    "list_baselines",
    "baselines_in_layer",
    "render_baseline",
    "DYNA3DGR_COMPARISON",
]

RendererKind = Literal["2dgs", "thin3dgs", "mesh", "free3dgs"]


@dataclass
class BaselineSpec:
    """One baseline: a pipeline configuration plus a rasteriser choice."""

    name: str
    description: str
    config: PipelineConfig
    renderer: RendererKind = "2dgs"
    isolates: str = ""
    """The single question this baseline answers, in one line."""

    caveats: str = ""
    """Known asymmetries that must be reported with the numbers."""

    extra: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "renderer": self.renderer,
            "isolates": self.isolates,
            "caveats": self.caveats,
            "preset": self.config.name,
        }


def _base() -> PipelineConfig:
    """The full method, used as the starting point for every ablation."""
    return get_preset("v3-adaptive")


# --------------------------------------------------------------------------- #
#  Baseline constructors
# --------------------------------------------------------------------------- #
def _cv_dyn2dgs() -> BaselineSpec:
    return BaselineSpec(
        name="cv-dyn2dgs",
        description="Full method: warm-started Chan-Vese, normal projection, tangent "
        "transport, geometry losses, density control, compact residual.",
        config=_base().replaced(name="cv-dyn2dgs"),
        renderer="2dgs",
        isolates="the proposed combination as a whole",
    )


def _mesh_only() -> BaselineSpec:
    cfg = _base().replaced(name="mesh-only")
    return BaselineSpec(
        name="mesh-only",
        description="Render the same Chan-Vese surface as a marching-tetrahedra mesh "
        "with level-set vertex normals and MRI-sampled vertex amplitudes.",
        config=cfg,
        renderer="mesh",
        isolates="RQ5 - whether surfels beat a strong textured mesh on the same surface",
        caveats="A mesh has binary pixel coverage, so its silhouette IoU is evaluated on "
        "a hard mask while surfel alpha is thresholded; IoU is therefore swept over "
        "thresholds rather than fixed at 0.5. The mesh also carries no opacity, so "
        "it cannot represent partial-volume boundaries at all - an honest structural "
        "difference, not a tuning artefact.",
    )


def _thin_3dgs(thickness: float = 0.1) -> BaselineSpec:
    cfg = _base().replaced(name="thin-3dgs")
    return BaselineSpec(
        name="thin-3dgs",
        description="Same anchors, frames and tangential scales, but rendered as 3-D "
        "Gaussian ellipsoids with a small normal-direction scale and affine projection.",
        config=cfg,
        renderer="thin3dgs",
        isolates="RQ6 - the primitive itself: 2-D disk vs thin 3-D ellipsoid",
        caveats="Depth is constant across a splat's footprint (affine projection), which "
        "is the deficiency under test rather than an implementation shortcut. The "
        "thickness ratio is swept in the ablation because thinner is geometrically "
        "better but worse conditioned.",
        extra={"thin3dgs": Thin3DGSConfig(thickness_ratio=thickness)},
    )


def _independent_2dgs() -> BaselineSpec:
    cfg = _base().replaced(name="independent-2dgs", refit_every_frame=True)
    cfg.residual = ResidualConfig(enabled=False)
    return BaselineSpec(
        name="independent-2dgs",
        description="Optimise a full 2DGS from scratch at every frame and store all of "
        "them. No canonical reuse, no residual.",
        config=cfg,
        renderer="2dgs",
        isolates="RQ2/RQ3 - the quality ceiling, and the time/storage cost of reaching it",
        caveats="This is the strongest quality reference. The interesting result is not "
        "beating it but landing inside a pre-registered non-inferiority margin while "
        "costing substantially less (proposal §8.5).",
    )


def _param_copy() -> BaselineSpec:
    cfg = _base().replaced(name="param-copy")
    cfg.surfel = replace(cfg.surfel, projection_iters=0, repulsion_enabled=False, densify_enabled=False)
    return BaselineSpec(
        name="param-copy",
        description="Leave anchors where frame 0 put them; only correct the appearance "
        "residual.",
        config=cfg,
        renderer="2dgs",
        isolates="whether geometric motion is needed at all, or appearance correction suffices",
        caveats="Expected to fail badly on E_surf by construction; included because it is "
        "the null hypothesis for the projection step.",
    )


def _closest_point() -> BaselineSpec:
    cfg = _base().replaced(name="closest-point")
    cfg.surfel = replace(cfg.surfel, projection_mode="closest_point")
    return BaselineSpec(
        name="closest-point",
        description="Replace the minimal-normal step of Eq. (6.6) with an unconstrained "
        "nearest-surface-point search.",
        config=cfg,
        renderer="2dgs",
        isolates="whether restricting the displacement to the normal direction matters",
        caveats="Near a true signed distance function the two coincide, so a *small* "
        "difference is the expected result and would support Prop. 6.2 rather than "
        "contradict it. The cost gap is large and is reported separately: the "
        "brute-force search here is deliberately unindexed.",
    )


def _normal_no_residual() -> BaselineSpec:
    cfg = _base().replaced(name="normal-no-residual")
    cfg.residual = ResidualConfig(enabled=False)
    return BaselineSpec(
        name="normal-no-residual",
        description="Update centres and tangent planes with Eq. (6.3)/(7.7) but store no "
        "appearance residual.",
        config=cfg,
        renderer="2dgs",
        isolates="the contribution of the residual to appearance fidelity and flicker",
    )


def _cold_start_cv() -> BaselineSpec:
    cfg = _base().replaced(name="cold-start-cv")
    cfg.chanvese = replace(cfg.chanvese, warm_start=False)
    return BaselineSpec(
        name="cold-start-cv",
        description="Restart Chan-Vese from the frame-0 initialisation at every frame "
        "instead of warm-starting from Eq. (5.1).",
        config=cfg,
        renderer="2dgs",
        isolates="RQ1 - whether the warm start reduces iterations, as Thm. 5.5 predicts",
        caveats="Also disables the narrow-band crop, since the band is derived from the "
        "warm start; the iteration count and the wall time therefore move together "
        "and are reported separately.",
    )


def _v1() -> BaselineSpec:
    return BaselineSpec(
        name="v1-minimal",
        description="Minimum success scope of proposal §10.4: isotropic disks, one "
        "projection iteration, no geometry losses, no density control.",
        config=get_preset("v1-minimal"),
        renderer="2dgs",
        isolates="the staged-development starting point",
    )


def _v2() -> BaselineSpec:
    return BaselineSpec(
        name="v2-geometry",
        description="Adds the Eq. (33) geometry losses and anisotropic minimum-rotation "
        "tangent transport.",
        config=get_preset("v2-geometry"),
        renderer="2dgs",
        isolates="the contribution of geometry regularisation and anisotropic transport",
    )


def _v3() -> BaselineSpec:
    return BaselineSpec(
        name="v3-adaptive",
        description="Adds curvature-adaptive scales, tangential repulsion, bounded "
        "densification/pruning, temporal residual regularisation and low-rank residuals.",
        config=get_preset("v3-adaptive"),
        renderer="2dgs",
        isolates="the contribution of density control and curvature-aware scaling",
    )


# --------------------------------------------------------------------------- #
#  Layer 1: the surface source, which the original baseline family never varied
# --------------------------------------------------------------------------- #
#  Every baseline above uses the Chan-Vese surface, so none of them can answer
#  "does this method depend on Chan-Vese, or is Chan-Vese a replaceable part?".
#  These three vary the source and hold the representation fixed.
def _source_oracle() -> BaselineSpec:
    cfg = _base().replaced(name="source-oracle")
    return BaselineSpec(
        name="source-oracle",
        description="Identical to cv-dyn2dgs but the level sets come from an exact "
        "surface (the phantom's analytic SDF, or ground-truth labels) instead of "
        "Chan-Vese.",
        config=cfg,
        renderer="2dgs",
        isolates="the upper bound of the representation: how much of the residual error "
        "is segmentation error rather than representation error",
        caveats="MANDATORY for attribution. Without it, every RQ5/RQ6 number mixes two "
        "error sources. On real data the oracle covers only ED and ES, so "
        "SurfaceSequence.require_full() refuses whole-sequence metrics - restrict "
        "the comparison to labelled frames rather than interpolating.",
        extra={"surface_source": "oracle"},
    )


def _source_mask() -> BaselineSpec:
    cfg = _base().replaced(name="source-mask")
    return BaselineSpec(
        name="source-mask",
        description="Level sets derived from an external segmentation mask (nnU-Net, "
        "CSTM, or any challenge submission) via signed_distance_from_mask.",
        config=cfg,
        renderer="2dgs",
        isolates="whether the method is agnostic to the surface source",
        caveats="A mask-derived SDF has a voxel-quantised zero level set, so its gradient "
        "direction is piecewise constant. sdf_fidelity() reports staircase_index for "
        "exactly this. If the projection degrades here, that is measured evidence "
        "that a genuine distance field is needed - an argument FOR the variational "
        "surface, not against it. Report the diagnostic alongside the quality drop.",
        extra={"surface_source": "mask"},
    )


def _source_cold_chanvese() -> BaselineSpec:
    cfg = _base().replaced(name="source-chanvese-no-spacing")
    cfg.chanvese = replace(cfg.chanvese)
    return BaselineSpec(
        name="source-chanvese-no-spacing",
        description="Chan-Vese with spacing_aware=False, i.e. finite differences that "
        "ignore anisotropic voxel spacing.",
        config=cfg,
        renderer="2dgs",
        isolates="contribution 1 measured rather than asserted: what the spacing-aware "
        "operators actually buy on anisotropic cine CMR",
        caveats="Only meaningful when hz differs substantially from hx, hy - which is the "
        "normal case for short-axis cine but NOT for the isotropic phantom. Run it "
        "on anisotropic data or the result is vacuous.",
        extra={"surface_source": "chan-vese", "spacing_aware": False},
    )


# --------------------------------------------------------------------------- #
#  Layer 2: geometry that is free, and the exact Gaussian-surfel primitive
# --------------------------------------------------------------------------- #
def _free_3dgs(init: str = "surface") -> BaselineSpec:
    cfg = _base().replaced(name=f"free-3dgs-{init}")
    return BaselineSpec(
        name=f"free-3dgs-{init}",
        description="3-D Gaussians whose positions, orientations and all three scales are "
        "optimised. Unlike thin-3dgs this does NOT share the surfel anchors, so it is a "
        "genuine comparison with 3DGS rather than a kernel swap.",
        config=cfg,
        renderer="free3dgs",
        isolates="whether pinning the geometry to the Chan-Vese surface is a cost or a "
        "benefit - the question the original baseline family could not ask",
        caveats="Expected to win on PSNR and lose badly on E_surf: with nothing holding "
        "them to the surface the Gaussians drift to wherever the image residual is "
        "smallest. That trade-off IS the thesis argument, so FreeFitReport.summary() "
        "refuses to emit one side without the other. The init mode changes the "
        "result and is part of the name for that reason.",
        extra={"init_mode": init},
    )


def _gaussian_surfel_exact() -> BaselineSpec:
    cfg = _base().replaced(name="gaussian-surfel-exact")
    return BaselineSpec(
        name="gaussian-surfel-exact",
        description="Thin-3DGS with the normal-direction scale driven to the numerical "
        "floor, matching the Dai et al. (SIGGRAPH 2024) formulation of a 3-D Gaussian "
        "with its z-scale set to zero.",
        config=cfg,
        renderer="thin3dgs",
        isolates="RQ6 against the published surfel formulation rather than against a "
        "thickness the ablation happened to choose",
        caveats="As the thickness goes to zero the projected conic becomes "
        "ill-conditioned; the low-pass term then dominates the footprint. Report the "
        "condition number alongside quality, or the degradation looks like a "
        "property of the primitive when it is a property of the conditioning.",
        extra={"thin3dgs": Thin3DGSConfig(thickness_ratio=1e-3)},
    )


BASELINES: dict[str, Callable[[], BaselineSpec]] = {
    "cv-dyn2dgs": _cv_dyn2dgs,
    "mesh-only": _mesh_only,
    "thin-3dgs": _thin_3dgs,
    "independent-2dgs": _independent_2dgs,
    "param-copy": _param_copy,
    "closest-point": _closest_point,
    "normal-no-residual": _normal_no_residual,
    "cold-start-cv": _cold_start_cv,
    "v1-minimal": _v1,
    "v2-geometry": _v2,
    "v3-adaptive": _v3,
    # layer 1 - surface source
    "source-oracle": _source_oracle,
    "source-mask": _source_mask,
    "source-chanvese-no-spacing": _source_cold_chanvese,
    # layer 2 - free geometry and the exact surfel primitive
    "free-3dgs-surface": lambda: _free_3dgs("surface"),
    "free-3dgs-bbox_random": lambda: _free_3dgs("bbox_random"),
    "gaussian-surfel-exact": _gaussian_surfel_exact,
}

INTERNAL_BASELINES: tuple[str, ...] = (
    "cv-dyn2dgs",
    "mesh-only",
    "thin-3dgs",
    "independent-2dgs",
    "param-copy",
    "closest-point",
    "normal-no-residual",
    "cold-start-cv",
    "v1-minimal",
    "v2-geometry",
    "v3-adaptive",
)
"""The original family: all configurations of this pipeline.

Good for attribution, useless for external validity - which is why the layer-1, layer-2
and layer-4 additions exist. See paper/COMPARISON_TARGETS.md.
"""

COMPARISON_LAYERS: dict[str, int] = {
    "source-oracle": 1,
    "source-mask": 1,
    "source-chanvese-no-spacing": 1,
    "free-3dgs-surface": 2,
    "free-3dgs-bbox_random": 2,
    "gaussian-surfel-exact": 2,
    "mesh-only": 2,
    "thin-3dgs": 2,
    "independent-2dgs": 3,
    "param-copy": 3,
    "closest-point": 3,
    "normal-no-residual": 3,
    "cold-start-cv": 3,
}
"""Which comparison layer each baseline belongs to (see paper sec. 7.3).

Baselines absent from this mapping are whole-method configurations rather than
single-axis comparisons.
"""


def list_baselines() -> list[str]:
    return sorted(BASELINES)


def baselines_in_layer(layer: int) -> list[str]:
    """Baselines that vary exactly the axis of one comparison layer (paper sec. 7.3)."""
    return sorted(k for k, v in COMPARISON_LAYERS.items() if v == layer)


def get_baseline(name: str) -> BaselineSpec:
    try:
        return BASELINES[name]()
    except KeyError as exc:
        raise KeyError(f"unknown baseline {name!r}; available: {list_baselines()}") from exc


def render_baseline(
    spec: BaselineSpec,
    camera: Camera,
    *,
    surfels: SurfelSet2D | None = None,
    mesh: TriangleMesh | None = None,
    vertex_amplitude=None,
    render_cfg: RenderConfig | None = None,
    compute_aux: bool = True,
    free_gaussians=None,
) -> RenderOutput:
    """Dispatch to the rasteriser this baseline uses.

    All four rasterisers return the same :class:`RenderOutput`, so every metric in
    :mod:`cvdyn2dgs.metrics` applies unchanged to all of them.
    """
    cfg = render_cfg or spec.config.render
    if spec.renderer == "free3dgs":
        if free_gaussians is None:
            raise ValueError(
                "renderer 'free3dgs' needs free_gaussians "
                "(cvdyn2dgs.surfel.free3dgs.FreeGaussians3D); it deliberately does not "
                "accept a SurfelSet2D, because sharing the surfel anchors is what makes "
                "thin-3dgs a kernel swap rather than a comparison"
            )
        from .surfel.free3dgs import render_free_3dgs

        return render_free_3dgs(free_gaussians, camera, cfg, compute_aux=compute_aux)
    if spec.renderer == "2dgs":
        if surfels is None:
            raise ValueError("renderer '2dgs' needs surfels")
        return render_2dgs(surfels, camera, cfg, compute_aux=compute_aux)
    if spec.renderer == "thin3dgs":
        if surfels is None:
            raise ValueError("renderer 'thin3dgs' needs surfels")
        thin = spec.extra.get("thin3dgs")
        return render_thin_3dgs(
            surfels,
            camera,
            cfg,
            thin if isinstance(thin, Thin3DGSConfig) else Thin3DGSConfig(),
            compute_aux=compute_aux,
        )
    if spec.renderer == "mesh":
        if mesh is None:
            raise ValueError("renderer 'mesh' needs a mesh")
        return render_mesh(
            mesh,
            camera,
            vertex_amplitude=vertex_amplitude,
            background=cfg.background,
            near_mm=cfg.near_mm,
        )
    raise ValueError(f"unknown renderer {spec.renderer!r}")


# --------------------------------------------------------------------------- #
#  Related work that is NOT reproduced here
# --------------------------------------------------------------------------- #
DYNA3DGR_COMPARISON: dict[str, object] = {
    "status": "not reproduced in this repository",
    "why": (
        "Requires the authors' public implementation, the ACDC dataset and GPU "
        "training. None of those are present in this environment, so no number is "
        "measured and none is attributed."
    ),
    "instruction_followed": (
        "Proposal §5.7: when reproduction is infeasible, report feasibility and "
        "structural differences only - do not compare numbers directly."
    ),
    "structural_differences": {
        "primitive": "Dyna3DGR uses volume-filling 3-D Gaussian ellipsoids; "
        "CV-Dyn2DGS uses 2-D disks constrained to the Chan-Vese surface.",
        "temporal_model": "Dyna3DGR optimises a continuous implicit neural motion "
        "field; CV-Dyn2DGS uses per-frame level sets plus a local normal projection "
        "with no learned deformation.",
        "objective": "Dyna3DGR targets cardiac motion tracking with topology and "
        "temporal consistency; CV-Dyn2DGS targets low-cost storage and real-time "
        "playback of surface appearance, and explicitly does not claim tissue "
        "correspondence.",
        "representation_scope": "Dyna3DGR renders volumetrically against the MRI; "
        "CV-Dyn2DGS renders an explicit organ surface and is not a volume renderer.",
        "supervision": "Both are per-patient self-supervised instance optimisations; "
        "neither needs a training cohort.",
    },
    "published_claim_not_verified_here": {
        "per_patient_optimisation_minutes": 11,
        "source": "figure reported in the Dyna3DGR authors' response, cited in "
        "proposal §5.7; reproduced here as a claim from the source, not a measurement",
    },
    "expected_failure_modes": {
        "dyna3dgr": "optimisation cost and model complexity",
        "cv_dyn2dgs": "tangential motion, large inter-frame displacement, topology change",
    },
}
