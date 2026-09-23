"""Where the surface comes from - the layer-1 comparison axis.

The rest of this package never asks *how* :math:`\\phi_t` was produced.  It only reads
:math:`\\phi_t` and :math:`\\nabla_h\\phi_t`.  That is a real degree of freedom, and the
proposal's baseline family does not exercise it at all: every baseline there uses the
Chan-Vese surface.  This module makes the surface source swappable so the question

    *is CV-Dyn2DGS actually dependent on Chan-Vese, or is Chan-Vese a replaceable
    part?*

can be answered by measurement instead of assertion.

Why this matters more than it looks
-----------------------------------
Chan-Vese is a 2001 method and cardiac benchmarks are dominated by deep networks, so
"your Dice is lower than nnU-Net's" is a certainty, not a risk.  The defence cannot be
*our segmentation is competitive*.  It has to be *the method does not depend on this
particular segmentation* - and that is only credible if someone swapped the source and
reported what happened.

There is a second, sharper reason.  What the surfel machinery needs is not a mask but a
**signed distance field**: Eq. (6.2)-(6.3) step along :math:`\\nabla_h\\phi_t` and
Prop. 6.2 assumes :math:`\\|\\nabla\\phi\\| \\approx 1`.  A mask converted to an SDF has
a voxel-quantised zero level set, so its gradient direction is piecewise constant and
its eikonal residual is structured rather than random.  If the projection degrades on
mask-derived fields, that is not a defeat: it is *measured evidence that a genuine
distance field is required*, which is an argument for the variational surface rather
than against it.  :func:`sdf_fidelity` exists to make that mechanism visible instead of
leaving it as a guess.

The sources
-----------
``ChanVeseSource``
    The proposed method: warm-started sequential 3-D Chan-Vese.
``MaskSequenceSource``
    Any external segmentation - nnU-Net, CSTM, a challenge submission - converted with
    :func:`cvdyn2dgs.levelset.sdf.signed_distance_from_mask`.
``OracleSource``
    A surface that is correct by construction: the phantom's analytic SDF, or
    ground-truth labels.  **This one is not optional.**  Without it, segmentation error
    and representation error are confounded and no RQ5/RQ6 number can be attributed.

Partial availability is a type, not a footnote
----------------------------------------------
Real data does not supply a ground-truth surface at every frame: ACDC labels only ED
and ES.  :class:`SurfaceSequence` therefore carries ``available_frames``, and
:meth:`SurfaceSequence.require_full` raises rather than letting a caller silently
average over frames where the oracle was interpolated.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from ..core.config import ChanVeseConfig
from ..core.grid import Grid
from .chanvese import solve_sequence
from .operators import gradient_central, gradient_norm
from .sdf import mask_from_levelset, signed_distance_from_mask

__all__ = [
    "SurfaceSequence",
    "SurfaceSource",
    "ChanVeseSource",
    "MaskSequenceSource",
    "OracleSource",
    "sdf_fidelity",
    "SURFACE_SOURCES",
    "get_surface_source",
]


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class SurfaceSequence:
    """Per-frame level sets plus where they came from and what they cost."""

    phis: list[Tensor]
    """``T`` volumes of shape ``(nx, ny, nz)``; ``phi > 0`` inside (proposal Eq. 1)."""

    source: str
    """Short key, e.g. ``"chan-vese"``, ``"nnunet-mask"``, ``"oracle"``."""

    provenance: str
    """One sentence a reader can audit: what produced these, and from what."""

    available_frames: list[int] | None = None
    """Frames the source genuinely covers.  ``None`` means all of them.

    For a ground-truth oracle on ACDC this is ``[ed, es]`` and nothing else; any metric
    averaged over other frames would be averaging over fabricated surfaces.
    """

    time_ms: float = 0.0
    """Wall time to produce the whole sequence, for the cost column."""

    diagnostics: dict[str, float] = field(default_factory=dict)
    """SDF fidelity and related measurements - see :func:`sdf_fidelity`."""

    @property
    def n_frames(self) -> int:
        return len(self.phis)

    @property
    def is_full(self) -> bool:
        return self.available_frames is None or len(self.available_frames) == self.n_frames

    def require_full(self, why: str) -> None:
        """Refuse to be used for a whole-sequence metric when coverage is partial."""
        if not self.is_full:
            have = self.available_frames or []
            raise ValueError(
                f"surface source {self.source!r} covers only frames {have} of "
                f"{self.n_frames}, so it cannot be used for {why}. Restrict the metric "
                f"to those frames, or use a source with full coverage. Averaging over "
                f"the remaining frames would average over surfaces this source did not "
                f"produce."
            )

    def masks(self) -> list[Tensor]:
        return [mask_from_levelset(p) for p in self.phis]

    def summary(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {
            "source": self.source,
            "frames": float(self.n_frames),
            "covered_frames": float(
                self.n_frames if self.available_frames is None else len(self.available_frames)
            ),
            "full_coverage": float(self.is_full),
            "time_ms": self.time_ms,
        }
        out.update(self.diagnostics)
        return out


# --------------------------------------------------------------------------- #
#  SDF fidelity - the diagnostic that explains layer-1 results
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sdf_fidelity(phi: Tensor, grid: Grid, *, band_mm: float = 4.0) -> dict[str, float]:
    """How close :math:`\\phi` is to a true signed distance function, near the surface.

    Only the band matters: that is the only region Eq. (6.2)-(6.3) and the transport
    rule ever read.  Three numbers, each measuring something different:

    ``eikonal_abs_mean`` / ``eikonal_abs_p95``
        :math:`\\big| \\|\\nabla_h\\phi\\| - 1 \\big|` in the band.  Prop. 6.2's
        least-norm-displacement argument degrades as this grows.
    ``staircase_index``
        Fraction of band voxels where the gradient direction disagrees by more than
        ``15`` degrees with the direction of a once-smoothed field.  This separates
        *quantisation* from *noise*: a mask-derived SDF has a locally flat, axis-aligned
        zero level set, so its gradient direction is piecewise constant and disagrees
        with the smoothed direction in a spatially structured way, whereas a genuine
        distance field does not.  An analytic SDF should score near zero.
    ``grad_norm_mean``
        Reported raw so a systematic scale error is visible rather than folded into an
        absolute value.

    A caller that finds a high ``staircase_index`` together with poor projection
    residuals has *explained* the failure, not merely observed it.
    """
    spacing = grid.spacing
    band = phi.abs() < float(band_mm)
    n_band = int(band.sum().item())
    if n_band == 0:
        return {
            "sdf/eikonal_abs_mean": float("nan"),
            "sdf/eikonal_abs_p95": float("nan"),
            "sdf/staircase_index": float("nan"),
            "sdf/grad_norm_mean": float("nan"),
            "sdf/band_voxels": 0.0,
        }

    grad = gradient_central(phi, spacing)
    gnorm = gradient_norm(grad)
    err = (gnorm - 1.0).abs()[band]

    # Once-smoothed field: a 6-neighbour box mean. Cheap, separable, no extra deps.
    acc = torch.zeros_like(phi)
    cnt = 0
    for axis in range(3):
        d = -3 + axis
        n = phi.shape[d]
        fwd = torch.zeros_like(phi)
        bwd = torch.zeros_like(phi)
        fwd.narrow(d, 0, n - 1).copy_(phi.narrow(d, 1, n - 1))
        fwd.narrow(d, n - 1, 1).copy_(phi.narrow(d, n - 1, 1))
        bwd.narrow(d, 1, n - 1).copy_(phi.narrow(d, 0, n - 1))
        bwd.narrow(d, 0, 1).copy_(phi.narrow(d, 0, 1))
        acc = acc + fwd + bwd
        cnt += 2
    sm = (acc + phi) / float(cnt + 1)

    g_sm = gradient_central(sm, spacing)
    n_sm = gradient_norm(g_sm).clamp_min(1e-12)
    n_raw = gnorm.clamp_min(1e-12)
    cos = (grad * g_sm).sum(dim=0) / (n_raw * n_sm)
    # 15 degrees
    misaligned = (cos[band] < 0.9659258262890683).to(phi.dtype)

    # torch.quantile refuses inputs above ~16M elements; subsample deterministically
    # rather than failing on a large band.
    flat = err.flatten().to(torch.float32)
    if flat.numel() > 1_000_000:
        stride = flat.numel() // 1_000_000 + 1
        flat = flat[::stride]
    q = torch.quantile(flat, 0.95)
    return {
        "sdf/eikonal_abs_mean": float(err.mean().item()),
        "sdf/eikonal_abs_p95": float(q.item()),
        "sdf/staircase_index": float(misaligned.mean().item()),
        "sdf/grad_norm_mean": float(gnorm[band].mean().item()),
        "sdf/band_voxels": float(n_band),
    }


# --------------------------------------------------------------------------- #
#  Sources
# --------------------------------------------------------------------------- #
class SurfaceSource(ABC):
    """Produces a level-set sequence.  Nothing downstream knows which one it is."""

    key: str = "abstract"

    @abstractmethod
    def build(
        self, images: Sequence[Tensor], grid: Grid, *, phi0_init: Tensor | None = None
    ) -> SurfaceSequence:
        """Return one level set per frame of ``images``."""

    def _finish(
        self,
        phis: list[Tensor],
        grid: Grid,
        provenance: str,
        *,
        t0: float,
        available: list[int] | None = None,
        extra: dict[str, float] | None = None,
    ) -> SurfaceSequence:
        diag = sdf_fidelity(phis[0], grid) if phis else {}
        if extra:
            diag.update(extra)
        return SurfaceSequence(
            phis=phis,
            source=self.key,
            provenance=provenance,
            available_frames=available,
            time_ms=(time.perf_counter() - t0) * 1e3,
            diagnostics=diag,
        )


class ChanVeseSource(SurfaceSource):
    """The proposed source: warm-started sequential 3-D Chan-Vese."""

    key = "chan-vese"

    def __init__(self, cfg: ChanVeseConfig | None = None, *, spacing_aware: bool = True) -> None:
        self.cfg = cfg or ChanVeseConfig()
        self.spacing_aware = bool(spacing_aware)

    def build(
        self, images: Sequence[Tensor], grid: Grid, *, phi0_init: Tensor | None = None
    ) -> SurfaceSequence:
        if phi0_init is None:
            raise ValueError("ChanVeseSource needs phi0_init (the frame-0 initialisation)")
        t0 = time.perf_counter()
        res = solve_sequence(
            images, phi0_init, grid, self.cfg, spacing_aware=self.spacing_aware
        )
        extra = {f"chanvese/{k}": v for k, v in res.summary().items()}
        return self._finish(
            res.phis,
            grid,
            provenance=(
                f"sequential 3-D Chan-Vese, warm_start={self.cfg.warm_start}, "
                f"spacing_aware={self.spacing_aware}, mu={self.cfg.mu}"
            ),
            t0=t0,
            extra=extra,
        )


class MaskSequenceSource(SurfaceSource):
    """An external segmentation, converted to a level set.

    This is the adapter for nnU-Net, CSTM, or any challenge submission: hand it a list
    of boolean volumes and it produces the same interface the Chan-Vese path produces.

    The conversion is where the interesting behaviour lives.  ``signed_distance_from_mask``
    places the interface half a voxel outside the last foreground voxel and then runs the
    reinitialisation PDE, so the zero level set inherits the voxel grid's staircase.
    ``sdf_fidelity`` measures exactly that, and the resulting ``staircase_index`` is the
    number to quote when explaining any degradation in the normal projection.
    """

    key = "mask"

    def __init__(
        self,
        masks: Sequence[Tensor],
        *,
        name: str = "external-mask",
        max_dist_mm: float | None = None,
        available_frames: Sequence[int] | None = None,
        producer: str = "unspecified external segmentation",
    ) -> None:
        if len(masks) == 0:
            raise ValueError("no masks given")
        self.masks_in = list(masks)
        self.key = name
        self.max_dist_mm = max_dist_mm
        self.available = list(available_frames) if available_frames is not None else None
        self.producer = producer

    def build(
        self, images: Sequence[Tensor], grid: Grid, *, phi0_init: Tensor | None = None
    ) -> SurfaceSequence:
        if len(self.masks_in) != len(images):
            raise ValueError(
                f"{len(self.masks_in)} masks for {len(images)} frames; the comparison "
                f"requires one surface per frame. If the source only covers some frames, "
                f"pass available_frames and supply placeholders explicitly rather than "
                f"letting the lengths disagree."
            )
        t0 = time.perf_counter()
        phis = [
            signed_distance_from_mask(m, grid.spacing, max_dist_mm=self.max_dist_mm)
            for m in self.masks_in
        ]
        return self._finish(
            phis,
            grid,
            provenance=(
                f"{self.producer} -> binary mask -> signed_distance_from_mask "
                f"(zero level set is voxel-quantised; see sdf/staircase_index)"
            ),
            t0=t0,
            available=self.available,
        )


class OracleSource(SurfaceSource):
    """A surface that is correct by construction.

    Two legitimate origins:

    * the phantom's **analytic** SDF, which is exact everywhere - this is the only
      configuration in which representation error can be isolated cleanly;
    * ground-truth labels converted to an SDF, which is exact only where labels exist.

    The second case must declare ``available_frames``.  ACDC labels ED and ES only, so an
    oracle there covers two frames out of twenty-odd, and
    :meth:`SurfaceSequence.require_full` will refuse whole-sequence metrics.  That refusal
    is the point: it is how a partially-labelled oracle is prevented from quietly becoming
    a fully-labelled one.
    """

    key = "oracle"

    def __init__(
        self,
        phis: Sequence[Tensor],
        *,
        analytic: bool,
        available_frames: Sequence[int] | None = None,
        origin: str = "unspecified",
    ) -> None:
        self.phis_in = list(phis)
        self.analytic = bool(analytic)
        self.available = list(available_frames) if available_frames is not None else None
        self.origin = origin
        self.key = "oracle-analytic" if analytic else "oracle-labels"

    @classmethod
    def from_masks(
        cls,
        masks: Sequence[Tensor],
        grid: Grid,
        *,
        available_frames: Sequence[int] | None = None,
        origin: str = "ground-truth labels",
    ) -> "OracleSource":
        phis = [signed_distance_from_mask(m, grid.spacing) for m in masks]
        return cls(phis, analytic=False, available_frames=available_frames, origin=origin)

    def build(
        self, images: Sequence[Tensor], grid: Grid, *, phi0_init: Tensor | None = None
    ) -> SurfaceSequence:
        if len(self.phis_in) != len(images):
            raise ValueError(
                f"{len(self.phis_in)} oracle level sets for {len(images)} frames"
            )
        t0 = time.perf_counter()
        note = (
            "analytic signed distance function - exact"
            if self.analytic
            else "ground-truth labels -> SDF - exact only where labels exist"
        )
        return self._finish(
            [p.clone() for p in self.phis_in],
            grid,
            provenance=f"oracle ({self.origin}): {note}",
            t0=t0,
            available=self.available,
        )


SURFACE_SOURCES = {
    "chan-vese": ChanVeseSource,
    "mask": MaskSequenceSource,
    "oracle": OracleSource,
}


def get_surface_source(key: str, *args, **kwargs) -> SurfaceSource:
    try:
        return SURFACE_SOURCES[key](*args, **kwargs)
    except KeyError as exc:
        raise KeyError(
            f"unknown surface source {key!r}; available: {sorted(SURFACE_SOURCES)}"
        ) from exc
