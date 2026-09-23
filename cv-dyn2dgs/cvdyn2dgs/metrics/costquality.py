"""Quality against cost, with the two costs kept apart.

The problem with a wide results table
-------------------------------------
Every quality and cost number this package measures already lands in one row via
:meth:`cvdyn2dgs.experiments.common.EvaluationResult.headline`.  Having both in the same
row is necessary but not sufficient: fifteen columns show that both were *measured*, not
what the *trade-off* is.  A method that buys 0.3 dB of PSNR by tripling the per-frame
latency looks, in a wide table, simply better.

Two costs, not one
------------------
This pipeline spends time in two places with completely different consequences, and adding
them or printing them side by side without a label invites the reader to compare
incomparable things:

**one-time** (:attr:`Cost.precompute_ms`)
    Chan-Vese, projection, transport, density control, residual solve.  Paid once per
    patient, offline.  A method may take minutes here and still be perfectly usable.

**per-frame** (:attr:`Cost.p95_frame_ms`)
    Surface read, projection, orientation, rasterisation.  Paid on **every** frame during
    playback, so this is the one the 33.3 ms interactivity budget applies to - and the
    budget is checked at the 95th percentile, because a viewer that stutters every tenth
    frame is not interactive.

**storage** (:attr:`Cost.bytes_total`)
    Paid once in space rather than time, and traded against both of the above.

:class:`Cost` therefore refuses to produce a single scalar.  There is no defensible
exchange rate between "three minutes of offline preprocessing" and "two milliseconds per
displayed frame", and inventing one is how a real regression gets hidden behind an
improvement somewhere else.

Dominance instead of a score
----------------------------
What *can* be said without an exchange rate is whether one configuration is
**dominated**: strictly no better on every axis and strictly worse on at least one.
:func:`pareto_front` returns the non-dominated set, which is the honest form of "these are
the options; the rest are strictly worse".  A baseline that survives on the frontier only
because it wins on one axis is visible as such, rather than being averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "FRAME_BUDGET_MS",
    "LOWER_BETTER",
    "Cost",
    "QualityVector",
    "CostQualityPoint",
    "percentile",
    "dominates",
    "shared_axis_count",
    "pareto_front",
    "cost_quality_rows",
    "stage_share",
]

FRAME_BUDGET_MS = 33.3
"""The interactivity budget of Eq. (34), checked at p95 (proposal §8.3)."""


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile.  ``q`` in ``[0, 1]``.

    Written out rather than pulled from numpy so the cost analysis, like the storage
    analysis, can be tested without PyTorch or numpy installed.
    """
    vs = sorted(float(v) for v in values if v == v)  # drop NaN
    if not vs:
        return float("nan")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    pos = q * (len(vs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (pos - lo)


@dataclass(frozen=True)
class Cost:
    """What a configuration costs, with the currencies kept separate."""

    precompute_ms: float | None = None
    """One-time, offline. ``None`` means not measured - never 0.0."""

    p95_frame_ms: float | None = None
    """Per-frame at the 95th percentile. The budget applies to this."""

    mean_frame_ms: float | None = None
    bytes_total: int | None = None

    def __post_init__(self) -> None:
        for name in ("precompute_ms", "p95_frame_ms", "mean_frame_ms"):
            v = getattr(self, name)
            if v is not None and (v < 0 or v != v):
                raise ValueError(f"{name} must be non-negative and not NaN, got {v!r}")

    @property
    def meets_budget(self) -> bool | None:
        if self.p95_frame_ms is None:
            return None
        return self.p95_frame_ms < FRAME_BUDGET_MS

    @property
    def headroom_ratio(self) -> float | None:
        """``33.3 / p95``.  ``1.4`` means 40 % margin; ``0.8`` means it fails."""
        if self.p95_frame_ms is None or self.p95_frame_ms <= 0:
            return None
        return FRAME_BUDGET_MS / self.p95_frame_ms

    def scalar(self) -> float:
        """Deliberately unavailable."""
        raise NotImplementedError(
            "Cost has no single scalar. Offline preprocessing time, per-frame playback "
            "latency and bytes on disk have no defensible exchange rate between them, and "
            "collapsing them would let a per-frame regression hide behind a faster "
            "precompute. Compare on the axis the claim is about, or use pareto_front()."
        )

    def to_dict(self) -> dict[str, float | bool | None]:
        return {
            "cost/precompute_ms": self.precompute_ms,
            "cost/precompute_s": None if self.precompute_ms is None else self.precompute_ms / 1e3,
            "cost/p95_frame_ms": self.p95_frame_ms,
            "cost/mean_frame_ms": self.mean_frame_ms,
            "cost/bytes_total": None if self.bytes_total is None else float(self.bytes_total),
            "cost/mb_total": None if self.bytes_total is None else self.bytes_total / 1024**2,
            "cost/meets_budget": self.meets_budget,
            "cost/headroom_ratio": self.headroom_ratio,
        }


@dataclass(frozen=True)
class QualityVector:
    """Quality on the axes a comparison is allowed to be judged on.

    ``None`` means *not measured* throughout, never a neutral or worst-case value.  A
    baseline that emits no depth map is not scored on depth; substituting a number would
    invent a result, and substituting the worst possible one would invent a different one.
    """

    psnr_roi_db: float | None = None
    ssim_roi: float | None = None
    iou: float | None = None
    dice: float | None = None
    e_surf_mm: float | None = None
    """Surface residual, Eq. (39). **Lower is better** - handled by :data:`LOWER_BETTER`."""

    hd95_mm: float | None = None
    depth_rmse_mm: float | None = None
    normal_deg: float | None = None
    flicker: float | None = None

    def axes(self) -> dict[str, float]:
        return {k: v for k, v in self.to_dict().items() if v is not None}

    def to_dict(self) -> dict[str, float | None]:
        return {
            "psnr_roi_db": self.psnr_roi_db,
            "ssim_roi": self.ssim_roi,
            "iou": self.iou,
            "dice": self.dice,
            "e_surf_mm": self.e_surf_mm,
            "hd95_mm": self.hd95_mm,
            "depth_rmse_mm": self.depth_rmse_mm,
            "normal_deg": self.normal_deg,
            "flicker": self.flicker,
        }


LOWER_BETTER = frozenset(
    {"e_surf_mm", "hd95_mm", "depth_rmse_mm", "normal_deg", "flicker",
     "cost/precompute_ms", "cost/p95_frame_ms", "cost/mean_frame_ms", "cost/bytes_total"}
)
"""Axes where smaller wins.  Getting one of these backwards silently inverts a
conclusion, so the set is defined once and used by the dominance test."""


@dataclass
class CostQualityPoint:
    """One configuration: what it achieved, and what it cost to achieve it."""

    name: str
    quality: QualityVector
    cost: Cost
    notes: str = ""
    extras: dict[str, float] = field(default_factory=dict)

    def comparable_axes(self) -> dict[str, float]:
        """Every measured axis, quality and cost together, keyed uniformly."""
        out = dict(self.quality.axes())
        for k, v in self.cost.to_dict().items():
            if isinstance(v, (int, float)) and v is not None and not isinstance(v, bool):
                if k in ("cost/precompute_ms", "cost/p95_frame_ms", "cost/bytes_total"):
                    out[k] = float(v)
        return out

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"name": self.name}
        out.update(self.quality.to_dict())
        out.update(self.cost.to_dict())
        if self.notes:
            out["notes"] = self.notes
        out.update(self.extras)
        return out


def _better(axis: str, a: float, b: float) -> bool:
    return a < b if axis in LOWER_BETTER else a > b


def dominates(a: CostQualityPoint, b: CostQualityPoint, *, axes: Sequence[str] | None = None) -> bool:
    """Does ``a`` dominate ``b``?

    Pareto dominance: ``a`` is at least as good on every shared axis and strictly better on
    at least one.  Axes only one of them measured are **excluded** rather than guessed, and
    the comparison is reported over the shared set - so a point cannot win by simply not
    reporting the axis it would lose on. Callers that care should check
    :func:`shared_axis_count`.
    """
    aa, bb = a.comparable_axes(), b.comparable_axes()
    keys = [k for k in (axes or aa) if k in aa and k in bb]
    if not keys:
        return False
    strictly = False
    for k in keys:
        if _better(k, bb[k], aa[k]):
            return False
        if _better(k, aa[k], bb[k]):
            strictly = True
    return strictly


def shared_axis_count(a: CostQualityPoint, b: CostQualityPoint) -> int:
    """How many axes a dominance verdict between these two actually rests on."""
    aa, bb = a.comparable_axes(), b.comparable_axes()
    return len(set(aa) & set(bb))


def pareto_front(
    points: Sequence[CostQualityPoint], *, axes: Sequence[str] | None = None
) -> list[CostQualityPoint]:
    """The non-dominated set, in input order.

    This is what can be said about quality versus cost *without* inventing an exchange
    rate.  Everything not returned is strictly worse than something that was: no better on
    any axis and worse on at least one. Everything returned is a genuine choice, and which
    one to prefer is a judgement about the application rather than a number.
    """
    out: list[CostQualityPoint] = []
    for p in points:
        if not any(dominates(q, p, axes=axes) for q in points if q is not p):
            out.append(p)
    return out


def stage_share(stage_ms: dict[str, float]) -> dict[str, float]:
    """Fraction of the frame each stage accounts for, Eq. (34).

    Shares rather than raw times, because the actionable question when the budget is
    missed is which stage to attack, and that is scale-free.
    """
    total = sum(v for v in stage_ms.values() if v == v and v >= 0)
    if total <= 0:
        return {k: float("nan") for k in stage_ms}
    return {k: v / total for k, v in stage_ms.items()}


def cost_quality_rows(
    points: Sequence[CostQualityPoint], *, axes: Sequence[str] | None = None
) -> list[dict[str, object]]:
    """Table rows with quality, both cost currencies, and the dominance verdict.

    The ``pareto`` column is the part that makes the trade-off legible: a configuration
    marked ``dominated`` is strictly worse than another row and needs no further
    discussion, while several rows on the frontier means there is a real choice to argue
    about.
    """
    front = {id(p) for p in pareto_front(points, axes=axes)}
    rows: list[dict[str, object]] = []
    for p in points:
        d = p.to_dict()
        d["pareto"] = "frontier" if id(p) in front else "dominated"
        if id(p) not in front:
            dom = next(
                (q.name for q in points if q is not p and dominates(q, p, axes=axes)), ""
            )
            d["dominated_by"] = dom
        rows.append(d)
    return rows
