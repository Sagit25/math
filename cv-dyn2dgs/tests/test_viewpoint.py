"""Layer-4 storage comparison: the one part of the comparison design that runs anywhere.

Why this file loads the module by hand
--------------------------------------
``cvdyn2dgs.metrics.viewpoint`` is pure arithmetic and deliberately has **no runtime
PyTorch dependency** - ``Grid`` is imported under ``TYPE_CHECKING`` only. But
``cvdyn2dgs/metrics/__init__.py`` eagerly imports the tensor-based metric modules, so the
ordinary ``from cvdyn2dgs.metrics.viewpoint import ...`` would drag torch in through the
package initialiser.

Loading the file directly keeps that independence real and testable. That matters here for
a specific reason: this repository was written in an environment where PyTorch could not be
installed, so almost nothing else in it has ever been executed. These tests **have** run.
Keeping them torch-free is what makes them evidence rather than intention.

The substance being tested is the honest part of the storage claim. Eq. (38) compares the
proposed representation against storing every frame's surfels, but the competitor a real
viewer faces is a pre-rendered video: for a small fixed set of viewpoints it wins outright.
So the claim is only ever "small *for free-viewpoint playback*", and the qualifier has to be
a measured crossover rather than a slogan.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest


def _load_viewpoint():
    """Import ``metrics/viewpoint.py`` in isolation, without executing any package init."""
    root = Path(__file__).resolve().parent.parent
    path = root / "cvdyn2dgs" / "metrics" / "viewpoint.py"
    for name in ("cvdyn2dgs", "cvdyn2dgs.metrics"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location("cvdyn2dgs.metrics.viewpoint", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


vp = _load_viewpoint()


class FakeGrid:
    """Only ``numel`` is part of the contract; ``shape`` is read defensively."""

    numel = 128 * 128 * 32
    shape = (128, 128, 32)


def test_module_needs_no_torch():
    """The whole point: this layer is verifiable in an environment without PyTorch."""
    assert "torch" not in sys.modules


# --------------------------------------------------------------------------- #
#  Arithmetic
# --------------------------------------------------------------------------- #
def test_video_bytes_one_second_at_known_bitrate():
    # 24 frames at 24 fps is exactly one second; 2000 kbit/s -> 250000 bytes.
    b = vp.video_bytes_from_bitrate(
        bitrate_kbps=2000, n_frames=24, fps_playback=24.0, n_viewpoints=1
    )
    assert b == 250_000


def test_video_bytes_scale_linearly_in_viewpoints():
    one = vp.video_bytes_from_bitrate(
        bitrate_kbps=2000, n_frames=24, fps_playback=24.0, n_viewpoints=1
    )
    seven = vp.video_bytes_from_bitrate(
        bitrate_kbps=2000, n_frames=24, fps_playback=24.0, n_viewpoints=7
    )
    assert seven == 7 * one


@pytest.mark.parametrize("bad", [
    {"bitrate_kbps": 0, "n_frames": 24, "fps_playback": 24.0, "n_viewpoints": 1},
    {"bitrate_kbps": 2000, "n_frames": 0, "fps_playback": 24.0, "n_viewpoints": 1},
    {"bitrate_kbps": 2000, "n_frames": 24, "fps_playback": 0.0, "n_viewpoints": 1},
    {"bitrate_kbps": 2000, "n_frames": 24, "fps_playback": 24.0, "n_viewpoints": 0},
])
def test_video_bytes_rejects_nonpositive_inputs(bad):
    with pytest.raises(ValueError):
        vp.video_bytes_from_bitrate(**bad)


def test_raw_volume_bytes_is_the_product():
    assert vp.raw_volume_bytes(FakeGrid(), 24, bytes_per_voxel=2) == 128 * 128 * 32 * 24 * 2


def test_raw_volume_only_needs_numel():
    class Minimal:
        numel = 1000

    assert vp.raw_volume_bytes(Minimal(), 3, bytes_per_voxel=2) == 6000


# --------------------------------------------------------------------------- #
#  Break-even, in all three regimes
# --------------------------------------------------------------------------- #
def test_breakeven_is_ours_over_per_view():
    b = vp.viewpoint_breakeven(ours_bytes=1_000_000, video_bytes_per_viewpoint=250_000)
    assert b.breakeven_viewpoints == pytest.approx(4.0)
    assert b.breakeven_viewpoints_int == 4


def test_breakeven_below_one_is_the_strongest_claim():
    """Video losing even at a single viewpoint is the best possible outcome."""
    b = vp.viewpoint_breakeven(ours_bytes=100_000, video_bytes_per_viewpoint=250_000)
    assert b.breakeven_viewpoints < 1.0
    assert "single viewpoint" in b.verdict


def test_breakeven_above_one_states_the_limitation():
    b = vp.viewpoint_breakeven(ours_bytes=1_000_000, video_bytes_per_viewpoint=250_000)
    assert "only holds above" in b.verdict


def test_breakeven_with_free_video_is_infinite_not_a_crash():
    b = vp.viewpoint_breakeven(ours_bytes=1_000_000, video_bytes_per_viewpoint=0.0)
    assert math.isinf(b.breakeven_viewpoints)
    assert "never cheaper" in b.verdict


def test_breakeven_rejects_negative_bytes():
    with pytest.raises(ValueError):
        vp.viewpoint_breakeven(ours_bytes=-1, video_bytes_per_viewpoint=1.0)


# --------------------------------------------------------------------------- #
#  The table, and the structural advantage it is meant to show
# --------------------------------------------------------------------------- #
def test_our_storage_is_flat_and_video_grows():
    rows = vp.viewpoint_table(
        ours_bytes=1_000_000, video_bytes_per_viewpoint=250_000,
        viewpoint_counts=(1, 2, 4, 8, 16, 32),
    )
    assert len({r["ours_mb"] for r in rows}) == 1, "our storage must not vary with viewpoints"
    vids = [r["video_mb"] for r in rows]
    assert vids == sorted(vids) and vids[0] < vids[-1]


def test_winner_flips_exactly_at_the_breakeven():
    """Below 4 views video wins, above it we do. The flip is the reportable finding."""
    rows = vp.viewpoint_table(
        ours_bytes=1_000_000, video_bytes_per_viewpoint=250_000,
        viewpoint_counts=(1, 2, 4, 8),
    )
    winners = {int(r["viewpoints"]): r["winner"] for r in rows}
    assert winners[1] == "video"
    assert winners[2] == "video"
    assert winners[4] == "video", "at exact equality ours is not cheaper, so it does not win"
    assert winners[8] == "ours"


# --------------------------------------------------------------------------- #
#  Refusals - the reason this module exists rather than a pair of functions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("axis", sorted(vp.GEOMETRY_AXES))
def test_storage_only_target_refuses_every_geometry_axis(axis):
    raw = vp.RawVolumeTarget.from_grid(FakeGrid(), 24)
    with pytest.raises(ValueError, match="no surface"):
        raw.score_axis(axis)


@pytest.mark.parametrize("axis", sorted(vp.STORAGE_AXES))
def test_storage_only_target_allows_storage_axes(axis):
    vp.RawVolumeTarget.from_grid(FakeGrid(), 24).score_axis(axis)


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="unknown axis"):
        vp.RawVolumeTarget.from_grid(FakeGrid(), 24).score_axis("bogus")


def test_byte_count_without_provenance_is_impossible():
    with pytest.raises(ValueError, match="provenance is required"):
        vp.StorageOnlyTarget(key="x", name="x", total_bytes=1, provenance="   ")


def test_assumed_bitrate_requires_a_basis():
    with pytest.raises(ValueError, match="basis"):
        vp.VideoCodecTarget.assumed(
            bitrate_kbps=2000, n_frames=24, fps_playback=24.0, n_viewpoints=1,
            codec="h265", basis="  ",
        )


def test_assumed_bitrate_is_labelled_as_assumed():
    """An assumption must never be readable as a measurement in a results table."""
    t = vp.VideoCodecTarget.assumed(
        bitrate_kbps=2000, n_frames=24, fps_playback=24.0, n_viewpoints=1,
        codec="h265", basis="typical streaming rate for this resolution",
    )
    assert "ASSUMED" in t.provenance
    assert "typical streaming rate" in t.provenance
    assert t.fps is None, "an assumed byte count must not carry a measured frame rate"


def test_measured_target_records_the_command():
    t = vp.VideoCodecTarget.measured(
        total_bytes=123456, n_viewpoints=4, codec="libx265",
        command="ffmpeg -c:v libx265 -crf 28 ...", fps=60.0,
    )
    assert t.provenance.startswith("measured:")
    assert "ffmpeg" in t.provenance
    assert t.fps == 60.0


def test_unmeasured_fps_is_none_not_zero():
    """A frame rate nobody measured must not average into a mean as 0.0."""
    raw = vp.RawVolumeTarget.from_grid(FakeGrid(), 24)
    assert raw.fps is None
    assert raw.to_dict()["raw-volume/fps"] is None


def test_raw_volume_is_viewpoint_independent():
    raw = vp.RawVolumeTarget.from_grid(FakeGrid(), 24)
    assert raw.n_viewpoints is None, "one copy serves every view"


if __name__ == "__main__":  # allow running without pytest installed
    raise SystemExit(pytest.main([__file__, "-q"]))
