"""Score third-party methods with *this* package's metrics.

The honest bridge
-----------------
No external method is reimplemented here.  Reimplementing someone's method and then
reporting that it lost is the least trustworthy form of comparison there is: every
difference is attributable to the reimplementation.  Instead, the external code is cloned
at a pinned commit (``scripts/fetch_external.py``), run by whoever has a GPU, and asked to
dump its rendered views to a directory.  This module loads that directory and runs **the
same metrics against the same ray-marched reference** that every internal baseline is
scored against.

That last point is what makes the comparison mean anything.
:func:`cvdyn2dgs.render.raymarch.raymarch_levelset` produces the reference silhouette,
depth, normal and surface intensity from the stored :math:`\\Gamma_t`.  The 2DGS path, the
mesh path, the thin-3DGS path and now any external method are all scored against that one
reference, so a difference is a difference between *representations*, not between
evaluation protocols.

What an external method must provide, and what happens when it does not
----------------------------------------------------------------------
A directory per method::

    <dir>/meta.json                 required - see ExternalMeta
    <dir>/color/frame_0000.npy      required - (C, H, W) or (H, W)
    <dir>/alpha/frame_0000.npy      optional - (H, W) in [0, 1]
    <dir>/depth/frame_0000.npy      optional - (H, W) in mm
    <dir>/normal/frame_0000.npy     optional - (3, H, W) unit vectors

Missing channels are represented as ``None``, never as zeros.  A method that does not emit
depth is *not scored* on depth RMSE; it does not silently score infinitely badly.
:meth:`ExternalOutput.scoreable_axes` reports what can legitimately be computed, and
:meth:`ExternalOutput.require_axis` raises with an explanation otherwise.  The asymmetry is
then visible in the results table as a genuine ``n/a`` rather than a fabricated number.

Camera agreement is checked, not assumed
----------------------------------------
The most likely way an external comparison goes silently wrong is a camera-convention
mismatch: a different world-to-camera handedness, millimetres versus metres, or a
half-pixel offset in the principal point.  ``meta.json`` must therefore restate the camera
parameters the method actually used, and :func:`check_camera_agreement` compares them
against the cameras this package would have used.  A disagreement is reported loudly,
because a silent one turns the whole comparison into noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from ..render.camera import Camera
from ..render.raymarch import SurfaceReference

__all__ = [
    "ExternalMeta",
    "ExternalOutput",
    "AXIS_REQUIREMENTS",
    "check_camera_agreement",
    "write_adapter_template",
    "load_external",
    "reference_targets",
]

AXIS_REQUIREMENTS: dict[str, str] = {
    "psnr": "color",
    "ssim": "color",
    "silhouette_iou": "alpha",
    "boundary_f": "alpha",
    "depth_rmse": "depth",
    "normal_angle": "normal",
    "flicker": "color",
    "bytes": "meta.storage_bytes",
    "fps": "meta.fps",
    "precompute_time": "meta.precompute_seconds",
}
"""Metric axis -> the output it needs. Used to refuse rather than to fabricate."""


@dataclass
class ExternalMeta:
    """``meta.json`` contents.

    Every field that would appear in a results table is here, and the ones that cannot be
    derived from the dumped images are **required**, because the alternative is a table
    cell nobody can trace.
    """

    method: str
    """Manifest key, e.g. ``"at-gs"``. Must match ``external/manifest.json``."""

    commit: str
    """The commit actually run. Must match the manifest pin, or the number is not
    reproducible from this repository."""

    n_frames: int
    image_height: int
    image_width: int
    storage_bytes: int | None = None
    fps: float | None = None
    precompute_seconds: float | None = None
    device: str = "unspecified"
    command: str = ""
    """How it was run. A number without this cannot be re-obtained."""

    camera: dict | None = None
    """Camera parameters the method used, for :func:`check_camera_agreement`."""

    notes: str = ""

    @classmethod
    def from_json(cls, path: Path) -> "ExternalMeta":
        with path.open(encoding="utf-8") as fh:
            d = json.load(fh)
        missing = [
            k for k in ("method", "commit", "n_frames", "image_height", "image_width")
            if k not in d
        ]
        if missing:
            raise ValueError(f"{path}: meta.json missing required fields {missing}")
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "method": self.method,
            "commit": self.commit,
            "n_frames": float(self.n_frames),
            "storage_bytes": None if self.storage_bytes is None else float(self.storage_bytes),
            "fps": self.fps,
            "precompute_seconds": self.precompute_seconds,
            "device": self.device,
            "command": self.command,
        }


@dataclass
class ExternalOutput:
    """Rendered views dumped by a third-party method.

    ``None`` means *not provided* and is never coerced to zeros.  That distinction is the
    whole reason this class exists rather than a dict of tensors.
    """

    meta: ExternalMeta
    color: list[Tensor]
    alpha: list[Tensor] | None = None
    depth: list[Tensor] | None = None
    normal: list[Tensor] | None = None
    root: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.color)

    def available(self) -> set[str]:
        have = {"color", "meta.storage_bytes" if self.meta.storage_bytes is not None else ""}
        have.discard("")
        for name, val in (("alpha", self.alpha), ("depth", self.depth), ("normal", self.normal)):
            if val is not None:
                have.add(name)
        if self.meta.fps is not None:
            have.add("meta.fps")
        if self.meta.precompute_seconds is not None:
            have.add("meta.precompute_seconds")
        return have

    def scoreable_axes(self) -> list[str]:
        have = self.available()
        return sorted(a for a, need in AXIS_REQUIREMENTS.items() if need in have)

    def require_axis(self, axis: str) -> None:
        """Raise unless ``axis`` can be computed from what this method actually emitted."""
        if axis not in AXIS_REQUIREMENTS:
            raise ValueError(f"unknown axis {axis!r}; known: {sorted(AXIS_REQUIREMENTS)}")
        need = AXIS_REQUIREMENTS[axis]
        if need not in self.available():
            raise ValueError(
                f"{self.meta.method!r} did not provide {need!r}, so {axis!r} cannot be "
                f"computed for it. Report this cell as not measured. Substituting zeros "
                f"would invent a result, and substituting the worst possible value would "
                f"invent a different one. Scoreable axes here: {self.scoreable_axes()}"
            )

    def check_pin(self, manifest_commit: str) -> None:
        """Warn loudly if the run used a different commit from the manifest pin."""
        if self.meta.commit != manifest_commit:
            self.warnings.append(
                f"commit mismatch: ran {self.meta.commit[:12]}, manifest pins "
                f"{manifest_commit[:12]} - this number is not reproducible from the "
                f"manifest as it stands"
            )

    def summary(self) -> dict[str, float | str | None]:
        out = dict(self.meta.to_dict())
        out["scoreable_axes"] = ",".join(self.scoreable_axes())
        out["n_warnings"] = float(len(self.warnings))
        return out


def _load_series(
    root: Path, name: str, n_expected: int, *, device=None
) -> list[Tensor] | None:
    """Load ``<root>/<name>/frame_*.npy`` as tensors, or ``None`` if absent.

    ``device`` matters: these arrays are scored against a ray-marched reference that lives
    on the compute device, and ``torch.from_numpy`` always produces a CPU tensor. Leaving
    them on the CPU fails at the first comparison.
    """
    d = root / name
    if not d.is_dir():
        return None
    files = sorted(d.glob("frame_*.npy"))
    if not files:
        return None
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a hard dependency
        raise ImportError("loading external output needs numpy") from exc
    out = [torch.from_numpy(np.load(str(f))).to(device) for f in files]
    if len(out) != n_expected:
        raise ValueError(
            f"{d}: found {len(out)} frames but meta.json declares {n_expected}. "
            f"A partial dump scored as if complete would silently change every "
            f"per-frame mean."
        )
    return out


def load_external(
    root: str | Path, *, manifest_commit: str | None = None, device=None
) -> ExternalOutput:
    """Load one method's dumped views from ``root``.

    ``device`` should be the device the reference renderings are on; see
    :func:`_load_series`.
    """
    root = Path(root)
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run write_adapter_template({root!s}) to get the "
            f"expected layout and a meta.json skeleton."
        )
    meta = ExternalMeta.from_json(meta_path)
    color = _load_series(root, "color", meta.n_frames, device=device)
    if color is None:
        raise ValueError(f"{root}/color/ is required and was not found")
    out = ExternalOutput(
        meta=meta,
        color=color,
        alpha=_load_series(root, "alpha", meta.n_frames, device=device),
        depth=_load_series(root, "depth", meta.n_frames, device=device),
        normal=_load_series(root, "normal", meta.n_frames, device=device),
        root=root,
    )
    for i, c in enumerate(out.color):
        if c.dim() == 2:
            out.color[i] = c.unsqueeze(0)
    h, w = out.color[0].shape[-2:]
    if (h, w) != (meta.image_height, meta.image_width):
        out.warnings.append(
            f"image size {h}x{w} disagrees with meta.json {meta.image_height}x"
            f"{meta.image_width}"
        )
    if manifest_commit is not None:
        out.check_pin(manifest_commit)
    return out


def check_camera_agreement(
    declared: dict | None, cameras: Sequence[Camera], *, tol: float = 1e-3
) -> list[str]:
    """Compare an external method's declared camera against ours.

    Returns a list of disagreements, empty when they match.  This is the failure mode most
    likely to corrupt an external comparison while looking fine: a flipped axis or a
    metre/millimetre mix-up produces plausible images that are wrong everywhere.

    A ``None`` declaration is itself a finding - it means nothing can be verified.
    """
    if declared is None:
        return [
            "external method declared no camera parameters, so convention agreement "
            "cannot be verified; any depth or silhouette comparison is unaudited"
        ]
    problems: list[str] = []
    if not cameras:
        return ["no local cameras supplied to compare against"]
    cam = cameras[0]
    checks = (
        ("height", float(cam.height)),
        ("width", float(cam.width)),
        ("fx", float(cam.fx) if not cam.orthographic else None),
        ("fy", float(cam.fy) if not cam.orthographic else None),
    )
    for key, ours in checks:
        if ours is None or key not in declared:
            continue
        theirs = float(declared[key])
        if abs(theirs - ours) > tol * max(1.0, abs(ours)):
            problems.append(f"camera {key}: theirs={theirs:g}, ours={ours:g}")
    if "units" in declared and str(declared["units"]).lower() not in {"mm", "millimetre", "millimeter"}:
        problems.append(
            f"camera units declared as {declared['units']!r}; this package is millimetres "
            f"throughout, so depth comparisons would be off by a constant factor"
        )
    return problems


def write_adapter_template(root: str | Path, method: str = "METHOD-KEY") -> Path:
    """Write the expected directory layout and a ``meta.json`` skeleton.

    Handing a collaborator this template is cheaper than discovering after the fact that
    their dump is missing the one field that makes the numbers traceable.
    """
    root = Path(root)
    for sub in ("color", "alpha", "depth", "normal"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    skeleton = {
        "method": method,
        "commit": "<the 40-char SHA you actually ran; must match external/manifest.json>",
        "n_frames": 0,
        "image_height": 0,
        "image_width": 0,
        "storage_bytes": None,
        "fps": None,
        "precompute_seconds": None,
        "device": "<e.g. 1x A100 40GB>",
        "command": "<the exact command line>",
        "camera": {
            "units": "mm",
            "height": 0,
            "width": 0,
            "fx": 0.0,
            "fy": 0.0,
            "convention": "<world-to-camera, right-handed, +z forward?>",
        },
        "notes": "",
    }
    (root / "meta.json").write_text(
        json.dumps(skeleton, indent=2) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text(
        "# External method output\n\n"
        "Drop per-frame arrays here as `.npy`:\n\n"
        "| directory | shape | required |\n|---|---|---|\n"
        "| `color/frame_0000.npy` | `(C, H, W)` or `(H, W)` | yes |\n"
        "| `alpha/frame_0000.npy` | `(H, W)` in [0, 1] | no |\n"
        "| `depth/frame_0000.npy` | `(H, W)` in **mm** | no |\n"
        "| `normal/frame_0000.npy` | `(3, H, W)` unit vectors | no |\n\n"
        "Anything omitted is reported as not measured. It is never filled with zeros,\n"
        "because a zero depth map scores as a specific wrong answer rather than as an\n"
        "absent one.\n\n"
        "`meta.json` is required. The `commit` field must be the SHA you actually ran and\n"
        "must match the pin in `external/manifest.json`, otherwise the resulting numbers\n"
        "cannot be reproduced from this repository.\n",
        encoding="utf-8",
    )
    return root / "meta.json"


def reference_targets(refs: Sequence[SurfaceReference]) -> dict[str, list[Tensor]]:
    """Unpack ray-marched references into the channels an external dump is scored against.

    Exposed so it is obvious that external and internal methods consume *the same*
    reference object.
    """
    return {
        "mask": [r.mask_float for r in refs],
        "depth": [r.depth for r in refs],
        "normal": [r.normal for r in refs],
        "intensity": [r.intensity for r in refs],
    }
