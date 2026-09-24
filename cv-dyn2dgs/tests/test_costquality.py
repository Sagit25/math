"""Quality-versus-cost reasoning: percentiles, dominance, and the refusals.

Like ``tests/test_viewpoint.py`` this loads the module directly so it runs **without
PyTorch**, which is what makes it evidence rather than intention in an environment where
torch cannot be installed. ``metrics/costquality.py`` is pure arithmetic and has no torch
dependency by design.

What is being pinned down here is mostly *judgement encoded as code*:

* a percentile that interpolates, so p95 is not silently the maximum on short runs;
* the direction of every axis, since getting one backwards inverts a conclusion;
* dominance computed only over axes both configurations actually measured, so a method
  cannot win by declining to report the axis it would lose on;
* and the two refusals - no single cost scalar, no NaN costs - that stop an offline
  speed-up from covering for a per-frame regression.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


def _load():
    """Load the module by path, WITHOUT registering anything under ``cvdyn2dgs``.

    An earlier version of this helper installed stub ``cvdyn2dgs`` and
    ``cvdyn2dgs.metrics`` entries in ``sys.modules`` so that the relative-import
    machinery would be satisfied.  Those stubs had an empty ``__path__``, and because
    pytest collects the whole ``tests/`` directory in alphabetical order they were still
    in ``sys.modules`` when the other test modules were imported - so
    ``cvdyn2dgs.metrics.clinical`` became unfindable and an unrelated test file failed to
    collect.  The symptom looked like a broken editable install and was not.

    No stubs are needed: ``metrics/costquality.py`` has no runtime relative imports (its only one is under
    ``TYPE_CHECKING``), so it loads standalone under a private name that cannot collide
    with the real package.
    """
    root = Path(__file__).resolve().parent.parent
    path = root / "cvdyn2dgs" / "metrics" / "costquality.py"
    spec = importlib.util.spec_from_file_location("_cvdyn2dgs_costquality_standalone", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Registration IS required - @dataclass resolves sys.modules[cls.__module__] while
    # processing annotations - but under the private name above, which cannot shadow the
    # real package the way the old `cvdyn2dgs.metrics` stub did.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cq = _load()
P, Q, C = cq.CostQualityPoint, cq.QualityVector, cq.Cost


def test_module_needs_no_torch():
    assert "torch" not in sys.modules


# --------------------------------------------------------------------------- #
#  Percentile
# --------------------------------------------------------------------------- #
def test_percentile_endpoints():
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cq.percentile(v, 0.0) == 1.0
    assert cq.percentile(v, 1.0) == 5.0


def test_percentile_median_of_odd_and_even():
    assert cq.percentile([1, 2, 3], 0.5) == 2.0
    assert cq.percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)


def test_percentile_interpolates_rather_than_snapping_to_max():
    """On a short run a non-interpolating p95 degenerates to the maximum.

    That would make the 33.3 ms budget test effectively a worst-case test, which is a
    different and much harsher criterion than the one the protocol registered.
    """
    v = list(range(1, 11))  # 1..10
    p95 = cq.percentile(v, 0.95)
    assert p95 < 10.0
    assert p95 == pytest.approx(9.55)


def test_percentile_catches_a_tail_the_mean_hides():
    v = [10, 12, 14, 16, 18, 20, 22, 24, 26, 100]
    mean = sum(v) / len(v)
    assert cq.percentile(v, 0.95) > 3 * mean / 2


def test_percentile_drops_nan_rather_than_propagating():
    assert cq.percentile([1.0, float("nan"), 3.0], 0.5) == 2.0


def test_percentile_of_nothing_is_nan_not_zero():
    assert math.isnan(cq.percentile([], 0.5))


@pytest.mark.parametrize("q", [-0.1, 1.1])
def test_percentile_rejects_out_of_range_q(q):
    with pytest.raises(ValueError):
        cq.percentile([1, 2, 3], q)


# --------------------------------------------------------------------------- #
#  Cost: two currencies, no exchange rate
# --------------------------------------------------------------------------- #
def test_cost_refuses_to_collapse_to_one_number():
    c = C(precompute_ms=1000, p95_frame_ms=20.0, bytes_total=1000)
    with pytest.raises(NotImplementedError, match="no defensible exchange rate"):
        c.scalar()


def test_unmeasured_cost_is_none_not_zero():
    c = C(p95_frame_ms=20.0)
    assert c.precompute_ms is None
    assert c.to_dict()["cost/precompute_ms"] is None


@pytest.mark.parametrize("bad", [
    {"precompute_ms": -1.0},
    {"p95_frame_ms": float("nan")},
    {"mean_frame_ms": -0.5},
])
def test_cost_rejects_negative_and_nan(bad):
    with pytest.raises(ValueError):
        C(**bad)


def test_budget_is_checked_at_p95_against_33_3ms():
    assert C(p95_frame_ms=33.2).meets_budget is True
    assert C(p95_frame_ms=33.4).meets_budget is False
    assert C().meets_budget is None, "unmeasured must not read as a failure"


def test_headroom_distinguishes_scraping_by_from_real_margin():
    """A bare pass/fail cannot tell these apart, and they imply different conclusions."""
    tight = C(p95_frame_ms=33.0).headroom_ratio
    roomy = C(p95_frame_ms=11.1).headroom_ratio
    assert tight == pytest.approx(1.009, abs=1e-2)
    assert roomy == pytest.approx(3.0, abs=1e-2)
    assert C(p95_frame_ms=66.6).headroom_ratio == pytest.approx(0.5, abs=1e-2)


# --------------------------------------------------------------------------- #
#  Axis direction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("axis", ["e_surf_mm", "hd95_mm", "depth_rmse_mm", "normal_deg", "flicker"])
def test_error_axes_are_lower_better(axis):
    assert axis in cq.LOWER_BETTER


@pytest.mark.parametrize("axis", ["psnr_roi_db", "ssim_roi", "iou", "dice"])
def test_quality_axes_are_higher_better(axis):
    assert axis not in cq.LOWER_BETTER


@pytest.mark.parametrize("axis", ["cost/precompute_ms", "cost/p95_frame_ms", "cost/bytes_total"])
def test_cost_axes_are_lower_better(axis):
    assert axis in cq.LOWER_BETTER


def test_lower_better_direction_is_actually_applied():
    """Same PSNR, worse surface error -> dominated. Catches an inverted comparison."""
    good = P("good", Q(psnr_roi_db=30.0, e_surf_mm=0.4), C(p95_frame_ms=20.0))
    bad = P("bad", Q(psnr_roi_db=30.0, e_surf_mm=0.9), C(p95_frame_ms=20.0))
    assert cq.dominates(good, bad)
    assert not cq.dominates(bad, good)


# --------------------------------------------------------------------------- #
#  Dominance and the frontier
# --------------------------------------------------------------------------- #
def test_strictly_worse_everywhere_is_dominated():
    a = P("a", Q(psnr_roi_db=32.0), C(p95_frame_ms=18.0, precompute_ms=1000))
    b = P("b", Q(psnr_roi_db=30.0), C(p95_frame_ms=25.0, precompute_ms=2000))
    assert cq.dominates(a, b)


def test_equal_on_everything_dominates_neither_way():
    a = P("a", Q(psnr_roi_db=30.0), C(p95_frame_ms=20.0))
    b = P("b", Q(psnr_roi_db=30.0), C(p95_frame_ms=20.0))
    assert not cq.dominates(a, b)
    assert not cq.dominates(b, a)


def test_a_genuine_tradeoff_dominates_neither_way():
    """Better quality at higher cost is a choice, not an improvement."""
    fast = P("fast", Q(psnr_roi_db=29.0), C(p95_frame_ms=12.0))
    good = P("good", Q(psnr_roi_db=34.0), C(p95_frame_ms=30.0))
    assert not cq.dominates(fast, good)
    assert not cq.dominates(good, fast)
    assert len(cq.pareto_front([fast, good])) == 2


def test_frontier_excludes_only_dominated_points():
    fast = P("fast", Q(psnr_roi_db=29.0), C(p95_frame_ms=12.0))
    good = P("good", Q(psnr_roi_db=34.0), C(p95_frame_ms=30.0))
    worse = P("worse", Q(psnr_roi_db=28.0), C(p95_frame_ms=31.0))
    front = {p.name for p in cq.pareto_front([fast, good, worse])}
    assert front == {"fast", "good"}


def test_frontier_preserves_input_order():
    pts = [
        P("a", Q(psnr_roi_db=30.0), C(p95_frame_ms=20.0)),
        P("b", Q(psnr_roi_db=31.0), C(p95_frame_ms=25.0)),
    ]
    assert [p.name for p in cq.pareto_front(pts)] == ["a", "b"]


def test_a_method_cannot_win_by_not_reporting_a_losing_axis():
    """Dominance uses shared axes only, so silence must not become an advantage.

    ``partial`` omits e_surf_mm, where it would lose badly. It therefore ties on the one
    shared axis and dominance is refused in both directions - rather than ``partial``
    coming out ahead because its weakness was simply absent.
    """
    full = P("full", Q(psnr_roi_db=30.0, e_surf_mm=0.4), C(p95_frame_ms=20.0))
    partial = P("partial", Q(psnr_roi_db=30.0), C(p95_frame_ms=20.0))
    assert cq.shared_axis_count(full, partial) == 2  # psnr + p95
    assert not cq.dominates(partial, full)
    assert not cq.dominates(full, partial)


def test_no_shared_axes_means_no_verdict():
    a = P("a", Q(psnr_roi_db=30.0), C())
    b = P("b", Q(dice=0.9), C())
    assert cq.shared_axis_count(a, b) == 0
    assert not cq.dominates(a, b)
    assert not cq.dominates(b, a)


def test_rows_name_the_dominating_configuration():
    fast = P("fast", Q(psnr_roi_db=29.0), C(p95_frame_ms=12.0))
    worse = P("worse", Q(psnr_roi_db=28.0), C(p95_frame_ms=31.0))
    rows = {r["name"]: r for r in cq.cost_quality_rows([fast, worse])}
    assert rows["fast"]["pareto"] == "frontier"
    assert rows["worse"]["pareto"] == "dominated"
    assert rows["worse"]["dominated_by"] == "fast"


def test_the_expected_free_geometry_tradeoff_stays_on_the_frontier():
    """Free-geometry 3DGS should win PSNR and lose E_surf, and that must not be averaged
    away - the trade-off is the thesis argument, so both points must survive."""
    ours = P("cv-dyn2dgs", Q(psnr_roi_db=31.2, e_surf_mm=0.42), C(p95_frame_ms=21.0))
    free = P("free-3dgs", Q(psnr_roi_db=34.6, e_surf_mm=4.80), C(p95_frame_ms=26.0))
    front = {p.name for p in cq.pareto_front([ours, free])}
    assert front == {"cv-dyn2dgs", "free-3dgs"}


# --------------------------------------------------------------------------- #
#  Stage shares
# --------------------------------------------------------------------------- #
def test_stage_shares_sum_to_one():
    shares = cq.stage_share({"surface_ms": 2.1, "project_ms": 6.4,
                             "orient_ms": 1.2, "raster_ms": 11.3})
    assert sum(shares.values()) == pytest.approx(1.0)


def test_stage_share_identifies_the_dominant_stage():
    shares = cq.stage_share({"surface_ms": 1.0, "project_ms": 2.0,
                             "orient_ms": 1.0, "raster_ms": 16.0})
    assert max(shares, key=shares.get) == "raster_ms"
    assert shares["raster_ms"] > 0.7


def test_stage_share_of_all_zero_is_nan_not_a_division_error():
    shares = cq.stage_share({"a": 0.0, "b": 0.0})
    assert all(math.isnan(v) for v in shares.values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
