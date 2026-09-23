"""Viewpoint-dependent storage: the competitors Eq. (38) has to survive.

The problem with the compression claim as stated
-------------------------------------------------
:math:`\\mathrm{CR} = S_{\\mathrm{full}} / S_{\\mathrm{ours}}` (Eq. 38) compares the
proposed representation against *storing every frame's surfels*.  That is an honest
internal comparison, but it is not the comparison a clinician's budget faces.  If the
viewer only ever looks from a handful of fixed angles, the cheapest correct answer is to
pre-render those angles and encode them as video - hundreds of kilobytes, sixty frames a
second, no 3-D machinery at all.  No citation is needed to make that true.

So the claim "surfaces plus a residual are small" is only meaningful with a qualifier:
small **for free-viewpoint playback**.  The honest way to state a qualifier is to measure
where it starts to hold, which is what :func:`viewpoint_breakeven` does.  Below the
break-even viewpoint count, video wins and the thesis should say so.

The second competitor is the compressed-4DGS family, which reports whole dynamic scenes
in the single-megabyte range, and in the medical domain Cinematic Anatomy 3DGS, which
reports multi-gigabyte volumes under 70 MB at 60 FPS.  Those are different tasks -
general scenes and static volumes rather than one organ surface over time - so they are
represented here as :class:`StorageOnlyTarget` with a *stated* byte count and a note
saying where the number came from, never as a measured comparison.

No invented bitrates
--------------------
A video baseline needs a bitrate, and a bitrate is empirical.  This module refuses to
supply a default: :class:`VideoCodecTarget` requires either a byte count measured from an
actual encode, or an explicitly declared assumption together with the reason.  A guessed
bitrate would be an invented result, and inventing the competitor's number is worse than
inventing your own.

Geometry metrics are structurally unavailable
---------------------------------------------
A video has no geometry and a raw volume has no rendered surface, so neither can be
scored on :math:`E_{\\mathrm{surf}}`, Dice, depth RMSE or normal error.  Rather than
relying on the caller to remember, :class:`StorageOnlyTarget` raises from
:meth:`~StorageOnlyTarget.score_axis` for any axis it cannot legitimately be scored on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from ..core.grid import Grid

# NOTE: ``Grid`` is imported for typing only, and only ``grid.numel`` is ever read.
# This module is pure arithmetic, so keeping the import out of the runtime path lets it
# be imported and tested **without PyTorch** - which is the difference between this
# layer-4 logic being verified and merely being written. See tests/test_viewpoint.py.

__all__ = [
    "GEOMETRY_AXES",
    "STORAGE_AXES",
    "StorageOnlyTarget",
    "VideoCodecTarget",
    "RawVolumeTarget",
    "ViewpointBudget",
    "video_bytes_from_bitrate",
    "raw_volume_bytes",
    "viewpoint_breakeven",
    "viewpoint_table",
]

GEOMETRY_AXES = frozenset(
    {"dice", "hd95", "assd", "e_surf", "depth_rmse", "normal_angle", "silhouette_iou"}
)
"""Axes that require an explicit surface. A storage-only target has none."""

STORAGE_AXES = frozenset({"bytes", "fps", "compression_ratio", "viewpoints"})
"""Axes a storage-only target can legitimately be scored on."""


# --------------------------------------------------------------------------- #
#  Storage-only targets
# --------------------------------------------------------------------------- #
@dataclass
class StorageOnlyTarget:
    """A competitor that can only be compared on bytes and frame rate.

    ``provenance`` is mandatory and free-form: it must say where ``total_bytes`` came
    from.  "measured: ffmpeg -c:v libx265 -crf 28, 4 views, 24 frames" is a provenance.
    "assumed 2 Mbit/s" is also a provenance, and an honest one.  An empty string is not,
    and is rejected.
    """

    key: str
    name: str
    total_bytes: int
    provenance: str
    fps: float | None = None
    """Measured playback frame rate, or ``None`` when it was not measured.

    ``None`` is not ``0.0``: a frame rate nobody measured must not average into a mean.
    """

    n_viewpoints: int | None = None
    """Viewpoints this byte count covers. ``None`` means viewpoint-independent."""

    notes: str = ""

    def __post_init__(self) -> None:
        if not self.provenance or not self.provenance.strip():
            raise ValueError(
                f"{self.key}: provenance is required. State where total_bytes came from - "
                f"a measurement command, or the assumption and its basis. A byte count "
                f"with no provenance cannot be put in a results table."
            )
        if self.total_bytes < 0:
            raise ValueError(f"{self.key}: total_bytes must be non-negative")

    def score_axis(self, axis: str) -> None:
        """Raise if ``axis`` cannot be measured for this target."""
        if axis in GEOMETRY_AXES:
            raise ValueError(
                f"{self.key!r} has no surface, so it cannot be scored on {axis!r}. "
                f"Available axes: {sorted(STORAGE_AXES)}. Comparing it on geometry would "
                f"require inventing a surface it does not have."
            )
        if axis not in STORAGE_AXES:
            raise ValueError(f"unknown axis {axis!r}; expected one of {sorted(STORAGE_AXES)}")

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            f"{self.key}/total_bytes": float(self.total_bytes),
            f"{self.key}/total_mb": self.total_bytes / 1024**2,
            f"{self.key}/fps": self.fps,
            f"{self.key}/n_viewpoints": (
                None if self.n_viewpoints is None else float(self.n_viewpoints)
            ),
            f"{self.key}/provenance": self.provenance,
        }


def video_bytes_from_bitrate(
    *,
    bitrate_kbps: float,
    n_frames: int,
    fps_playback: float,
    n_viewpoints: int,
) -> int:
    """Bytes for ``n_viewpoints`` independently encoded clips.

    Deliberately a plain function of a *declared* bitrate rather than a model with a
    default: the caller has to write the number down, and :class:`VideoCodecTarget`
    records it in the provenance string.
    """
    if bitrate_kbps <= 0 or n_frames <= 0 or fps_playback <= 0 or n_viewpoints <= 0:
        raise ValueError("bitrate, frame count, playback fps and viewpoints must be positive")
    seconds = n_frames / float(fps_playback)
    per_view = bitrate_kbps * 1000.0 / 8.0 * seconds
    return int(math.ceil(per_view * n_viewpoints))


@dataclass
class VideoCodecTarget(StorageOnlyTarget):
    """Pre-rendered video at a fixed set of viewpoints.

    Construct via :meth:`measured` or :meth:`assumed`; the base constructor is usable but
    those two make the provenance impossible to omit.
    """

    codec: str = "unspecified"

    @classmethod
    def measured(
        cls,
        *,
        total_bytes: int,
        n_viewpoints: int,
        codec: str,
        command: str,
        fps: float | None = None,
    ) -> "VideoCodecTarget":
        """From an actual encode.  ``command`` is the encoder invocation."""
        return cls(
            key="video-codec",
            name=f"pre-rendered video ({codec})",
            total_bytes=int(total_bytes),
            provenance=f"measured: {command}",
            fps=fps,
            n_viewpoints=int(n_viewpoints),
            codec=codec,
        )

    @classmethod
    def assumed(
        cls,
        *,
        bitrate_kbps: float,
        n_frames: int,
        fps_playback: float,
        n_viewpoints: int,
        codec: str,
        basis: str,
    ) -> "VideoCodecTarget":
        """From a declared bitrate.  ``basis`` must say why that bitrate is plausible.

        The result is clearly labelled as an assumption so it can never be mistaken for a
        measurement in a table.
        """
        if not basis.strip():
            raise ValueError(
                "an assumed bitrate needs a basis. Without one this is a guess dressed "
                "as a competitor's result."
            )
        b = video_bytes_from_bitrate(
            bitrate_kbps=bitrate_kbps,
            n_frames=n_frames,
            fps_playback=fps_playback,
            n_viewpoints=n_viewpoints,
        )
        return cls(
            key="video-codec",
            name=f"pre-rendered video ({codec}, assumed)",
            total_bytes=b,
            provenance=(
                f"ASSUMED {bitrate_kbps:g} kbit/s x {n_frames} frames @ "
                f"{fps_playback:g} fps x {n_viewpoints} views; basis: {basis}"
            ),
            fps=None,
            n_viewpoints=int(n_viewpoints),
            codec=codec,
        )


@dataclass
class RawVolumeTarget(StorageOnlyTarget):
    """The clinical status quo: keep the volume, use MPR or direct volume rendering.

    Scored on bytes only.  ``fps`` stays ``None`` because this package contains no volume
    renderer, and quoting a frame rate for software that was not run would be fabrication.
    Image quality is deliberately not compared: CV-Dyn2DGS renders a surface and this
    renders a volume, so the two are not commensurable (the thesis says as much in its
    limitations).
    """

    @classmethod
    def from_grid(
        cls, grid: Grid, n_frames: int, *, bytes_per_voxel: int = 2, note: str = ""
    ) -> "RawVolumeTarget":
        total = raw_volume_bytes(grid, n_frames, bytes_per_voxel=bytes_per_voxel)
        # ``shape`` is only used to make the provenance readable, so it is read
        # defensively: the contract is ``numel``, and nothing here should force a caller
        # to supply a full Grid just to get a byte count.
        shape = getattr(grid, "shape", None)
        where = f"{tuple(shape)}" if shape is not None else f"{int(grid.numel)}"
        return cls(
            key="raw-volume",
            name="raw 4-D volume (MPR / DVR status quo)",
            total_bytes=total,
            provenance=(
                f"computed: {where} voxels x {n_frames} frames x "
                f"{bytes_per_voxel} B/voxel, uncompressed"
            ),
            fps=None,
            n_viewpoints=None,
            notes=note or (
                "viewpoint-independent: one copy serves every view, which is exactly why "
                "it is the hardest competitor to beat on flexibility and the easiest to "
                "beat on size"
            ),
        )


def raw_volume_bytes(grid: "Grid", n_frames: int, *, bytes_per_voxel: int = 2) -> int:
    """Uncompressed 4-D volume size.  ``int16`` is the usual cine CMR storage type.

    Only ``grid.numel`` is read, so any object exposing that works - which is what keeps
    this module free of a runtime PyTorch dependency.
    """
    if n_frames <= 0 or bytes_per_voxel <= 0:
        raise ValueError("n_frames and bytes_per_voxel must be positive")
    return int(grid.numel) * int(n_frames) * int(bytes_per_voxel)


# --------------------------------------------------------------------------- #
#  Break-even
# --------------------------------------------------------------------------- #
@dataclass
class ViewpointBudget:
    """Where free-viewpoint storage starts to pay for itself."""

    ours_bytes: int
    video_bytes_per_viewpoint: float
    breakeven_viewpoints: float
    """Real-valued :math:`V^{\\ast}` where video cost equals ours."""

    breakeven_viewpoints_int: int
    """Smallest integer :math:`V` at which video is no cheaper than ours."""

    raw_volume_bytes: int | None = None
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        v = self.breakeven_viewpoints
        if not math.isfinite(v):
            return "video is never cheaper at any viewpoint count"
        if v < 1.0:
            return (
                "video loses even at a single viewpoint - report this, it is the "
                "strongest possible form of the claim"
            )
        return (
            f"video is cheaper below about {v:.1f} viewpoints; the free-viewpoint claim "
            f"only holds above that"
        )

    def to_dict(self) -> dict[str, float | str | None]:
        d: dict[str, float | str | None] = {
            "viewpoint/ours_bytes": float(self.ours_bytes),
            "viewpoint/video_bytes_per_view": float(self.video_bytes_per_viewpoint),
            "viewpoint/breakeven": self.breakeven_viewpoints,
            "viewpoint/breakeven_int": float(self.breakeven_viewpoints_int),
            "viewpoint/raw_volume_bytes": (
                None if self.raw_volume_bytes is None else float(self.raw_volume_bytes)
            ),
            "viewpoint/verdict": self.verdict,
        }
        d.update({f"viewpoint/{k}": v for k, v in self.detail.items()})
        return d


def viewpoint_breakeven(
    *,
    ours_bytes: int,
    video_bytes_per_viewpoint: float,
    raw_volume_bytes_total: int | None = None,
) -> ViewpointBudget:
    """Solve :math:`V \\cdot b_{\\mathrm{video}} = S_{\\mathrm{ours}}` for :math:`V`.

    A result below 1 is the strongest outcome: video loses even for a single fixed view.
    A large result is the honest bad news, and reporting it is what makes the
    free-viewpoint qualifier a measurement rather than a slogan.
    """
    if ours_bytes < 0 or video_bytes_per_viewpoint < 0:
        raise ValueError("byte counts must be non-negative")
    if video_bytes_per_viewpoint == 0:
        v = math.inf
        v_int = 0
    else:
        v = ours_bytes / float(video_bytes_per_viewpoint)
        v_int = int(math.ceil(v))
    return ViewpointBudget(
        ours_bytes=int(ours_bytes),
        video_bytes_per_viewpoint=float(video_bytes_per_viewpoint),
        breakeven_viewpoints=v,
        breakeven_viewpoints_int=v_int,
        raw_volume_bytes=raw_volume_bytes_total,
        detail={
            "ours_mb": ours_bytes / 1024**2,
            "video_mb_per_view": video_bytes_per_viewpoint / 1024**2,
        },
    )


def viewpoint_table(
    *,
    ours_bytes: int,
    video_bytes_per_viewpoint: float,
    viewpoint_counts: Sequence[int] = (1, 2, 4, 8, 16, 32),
    raw_volume_bytes_total: int | None = None,
) -> list[dict[str, float | str]]:
    """Rows for the storage-versus-viewpoints table.

    ``ours`` is flat in the viewpoint count - that is the entire structural advantage -
    while video grows linearly and the raw volume is also flat but far larger.  Laying the
    three side by side is the clearest honest statement of where this representation is
    and is not the right choice.
    """
    rows: list[dict[str, float | str]] = []
    for v in viewpoint_counts:
        vid = float(video_bytes_per_viewpoint) * int(v)
        row: dict[str, float | str] = {
            "viewpoints": float(v),
            "ours_mb": ours_bytes / 1024**2,
            "video_mb": vid / 1024**2,
            "winner": "ours" if ours_bytes < vid else "video",
        }
        if raw_volume_bytes_total is not None:
            row["raw_volume_mb"] = raw_volume_bytes_total / 1024**2
        rows.append(row)
    return rows
