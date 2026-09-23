"""Loaders for real cine-CMR datasets (ACDC, M&Ms-2).

Both datasets ship as NIfTI, so :mod:`nibabel` is required for this module only -
it is an optional dependency and the import is deferred with a clear message.

Dataset conventions
-------------------
**ACDC** (proposal §7.2, ref. [19]) - per patient:

* ``patientXXX_4d.nii.gz`` - the full cine stack, shape ``(X, Y, Z, T)``;
* ``patientXXX_frameNN.nii.gz`` / ``..._gt.nii.gz`` - the ED and ES frames with
  labels ``1 = RV cavity``, ``2 = LV myocardium``, ``3 = LV cavity``;
* ``Info.cfg`` - text file naming the ED and ES frame indices (1-based).

**M&Ms-2** (proposal §7.3, ref. [20]) uses the same label codes on SAX and LAX
4-chamber views, with per-patient files named by view.

Scope, stated plainly
---------------------
Only the **LV cavity** (label 3) is wired up as the target surface, matching the
"single clear structure" base scope of proposal §3.1.  Full LV/RV/myocardium needs a
multiphase level set and is listed there as extension scope; it is *not* implemented
here, and this loader will say so rather than quietly returning something else.

Ground truth exists only at ED and ES.  Intermediate frames are returned with
``mask = None``, and every metric that consumes them must therefore be an
appearance or temporal-stability metric, never a segmentation score.  The loader
enforces this by typing the masks as optional instead of filling them in.

Physical spacing is taken from the NIfTI header zooms, never assumed isotropic -
this is the input that makes the spacing-aware discretisation of theory §4.4 do
anything at all.
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from ..core.grid import Grid

__all__ = [
    "CineSequence",
    "LabelCodes",
    "load_nifti",
    "load_acdc_patient",
    "load_mnms2_patient",
    "normalize_intensity",
]


@dataclass(frozen=True)
class LabelCodes:
    """Segmentation label codes. Identical for ACDC and M&Ms-2."""

    rv_cavity: int = 1
    lv_myocardium: int = 2
    lv_cavity: int = 3


@dataclass
class CineSequence:
    """One patient's cine stack with whatever ground truth exists."""

    images: list[Tensor]
    """``T`` volumes of shape ``(nx, ny, nz)``, intensity-normalised."""

    grid: Grid
    masks: dict[int, Tensor] = field(default_factory=dict)
    """Frame index -> boolean LV-cavity mask. Present for ED and ES only."""

    ed_index: int = 0
    es_index: int = 0
    patient_id: str = ""
    source: str = ""
    affine: Tensor | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.images)

    @property
    def labelled_frames(self) -> list[int]:
        return sorted(self.masks)

    def to(self, device=None, dtype=None) -> "CineSequence":
        return CineSequence(
            images=[x.to(device=device, dtype=dtype) for x in self.images],
            grid=self.grid,
            masks={k: v.to(device=device) for k, v in self.masks.items()},
            ed_index=self.ed_index,
            es_index=self.es_index,
            patient_id=self.patient_id,
            source=self.source,
            affine=None if self.affine is None else self.affine.to(device=device, dtype=dtype),
            metadata=dict(self.metadata),
        )

    def volume_ml(self, frame: int) -> float:
        """LV cavity volume from the label, in ml. Only valid for labelled frames."""
        if frame not in self.masks:
            raise KeyError(
                f"frame {frame} has no ground-truth label; "
                f"labelled frames are {self.labelled_frames}"
            )
        return float(self.masks[frame].sum().item()) * self.grid.voxel_volume_mm3 / 1000.0

    def ef_percent(self) -> float:
        """Label-derived ejection fraction, proposal Eq. (41)."""
        v_ed = self.volume_ml(self.ed_index)
        v_es = self.volume_ml(self.es_index)
        return (v_ed - v_es) / v_ed * 100.0


def _require_nibabel():
    try:
        import nibabel  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading ACDC / M&Ms-2 needs nibabel. Install it with "
            "`pip install 'cvdyn2dgs[data]'` or `pip install nibabel`. "
            "The synthetic phantom in cvdyn2dgs.data.phantom has no such dependency."
        ) from exc
    return nibabel


def normalize_intensity(
    vol: Tensor, *, lo_pct: float = 1.0, hi_pct: float = 99.0, clamp: bool = True
) -> Tensor:
    """Percentile-normalise a volume to roughly ``[0, 1]``.

    Chan-Vese's ``mu`` weighs the curvature term against squared intensity
    differences, so it is only transferable across datasets if intensities live on a
    common scale.  Percentiles rather than min/max, because MRI has bright outliers.
    """
    flat = vol.reshape(-1).to(torch.float32)
    lo = torch.quantile(flat, lo_pct / 100.0)
    hi = torch.quantile(flat, hi_pct / 100.0)
    out = (vol - lo) / (hi - lo).clamp_min(1e-8)
    return out.clamp(0.0, 1.0) if clamp else out


def load_nifti(path: str | Path) -> tuple[Tensor, Grid, Tensor]:
    """Load a NIfTI file.

    Returns
    -------
    ``(data, grid, affine)``.  ``data`` keeps its original shape (3-D or 4-D) and
    ``grid`` describes the first three axes using the header zooms in mm.
    """
    nib = _require_nibabel()
    img = nib.load(str(path))
    arr = img.get_fdata(dtype="float32")
    data = torch.from_numpy(arr)
    zooms = img.header.get_zooms()[:3]
    grid = Grid(shape=tuple(int(s) for s in data.shape[:3]), spacing=tuple(float(z) for z in zooms))
    affine = torch.from_numpy(img.affine.astype("float32"))
    return data, grid, affine


def _parse_info_cfg(path: Path) -> dict[str, str]:
    """ACDC's ``Info.cfg`` is INI-like but has no section header."""
    text = path.read_text(encoding="utf-8")
    parser = configparser.ConfigParser()
    parser.read_string("[acdc]\n" + text)
    return {k.strip().lower(): v.strip() for k, v in parser["acdc"].items()}


def load_acdc_patient(
    patient_dir: str | Path,
    *,
    labels: LabelCodes | None = None,
    structure: str = "lv_cavity",
    normalize: bool = True,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> CineSequence:
    """Load one ACDC patient directory.

    Parameters
    ----------
    patient_dir:
        e.g. ``.../ACDC/training/patient001``.
    structure:
        Only ``"lv_cavity"`` is supported; see the module docstring on scope.

    Returns
    -------
    :class:`CineSequence` with masks at the ED and ES indices only.
    """
    labels = labels or LabelCodes()
    if structure != "lv_cavity":
        raise NotImplementedError(
            f"structure={structure!r} is extension scope (multiphase level set, "
            "proposal §3.1); only 'lv_cavity' is implemented"
        )
    root = Path(patient_dir)
    pid = root.name

    four_d = sorted(root.glob(f"{pid}_4d.nii*"))
    if not four_d:
        raise FileNotFoundError(f"no {pid}_4d.nii(.gz) in {root}")
    data, grid, affine = load_nifti(four_d[0])
    if data.dim() != 4:
        raise ValueError(f"{four_d[0]} should be 4-D (X,Y,Z,T), got {tuple(data.shape)}")

    n_t = int(data.shape[3])
    images = [data[..., t].contiguous() for t in range(n_t)]
    if normalize:
        images = [normalize_intensity(v) for v in images]
    images = [v.to(device=device, dtype=dtype) for v in images]

    meta: dict[str, str] = {}
    ed = es = 0
    info = root / "Info.cfg"
    if info.exists():
        meta = _parse_info_cfg(info)
        # ACDC frame numbers are 1-based.
        ed = int(float(meta.get("ed", 1))) - 1
        es = int(float(meta.get("es", 1))) - 1

    masks: dict[int, Tensor] = {}
    for gt_path in sorted(root.glob(f"{pid}_frame*_gt.nii*")):
        m = re.search(r"frame(\d+)_gt", gt_path.name)
        if not m:
            continue
        frame = int(m.group(1)) - 1  # to 0-based
        lab, lab_grid, _ = load_nifti(gt_path)
        if tuple(lab.shape[:3]) != tuple(grid.shape):
            raise ValueError(
                f"label {gt_path.name} shape {tuple(lab.shape[:3])} does not match "
                f"the cine grid {grid.shape}"
            )
        masks[frame] = (lab.round().to(torch.int64) == labels.lv_cavity).to(device=device)

    if masks and ed not in masks:
        ed = min(masks)
    if masks and es not in masks:
        es = max(masks)

    return CineSequence(
        images=images,
        grid=grid,
        masks=masks,
        ed_index=ed,
        es_index=es,
        patient_id=pid,
        source="ACDC",
        affine=affine,
        metadata=meta,
    )


def load_mnms2_patient(
    patient_dir: str | Path,
    *,
    view: str = "SA",
    labels: LabelCodes | None = None,
    normalize: bool = True,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> CineSequence:
    """Load one M&Ms-2 patient.

    M&Ms-2 supplies short-axis (``SA``) and long-axis 4-chamber (``LA``) cine with
    ED/ES labels.  Proposal §7.3 uses SA for the main external-domain check and LA as
    an auxiliary *cross-view* check - the LAX images are never used for fitting, only
    to test whether the silhouette, depth and appearance stay consistent from a view
    the model was not fitted to.

    File naming varies across releases, so this resolves by glob on
    ``*_{view}_CINE.nii*`` and ``*_{view}_ED.nii*`` / ``*_{view}_ES.nii*`` and
    reports clearly when nothing matches.
    """
    labels = labels or LabelCodes()
    root = Path(patient_dir)
    pid = root.name
    v = view.upper()

    cine = sorted(root.glob(f"*{v}_CINE.nii*")) or sorted(root.glob(f"*{v}.nii*"))
    if not cine:
        raise FileNotFoundError(
            f"no *{v}_CINE.nii(.gz) in {root}; M&Ms-2 naming differs between "
            "releases - pass the directory that directly contains the NIfTI files"
        )
    data, grid, affine = load_nifti(cine[0])
    if data.dim() != 4:
        raise ValueError(f"{cine[0]} should be 4-D, got {tuple(data.shape)}")

    images = [data[..., t].contiguous() for t in range(int(data.shape[3]))]
    if normalize:
        images = [normalize_intensity(x) for x in images]
    images = [x.to(device=device, dtype=dtype) for x in images]

    masks: dict[int, Tensor] = {}
    idx = {"ED": 0, "ES": 0}
    for tag in ("ED", "ES"):
        gt = sorted(root.glob(f"*{v}_{tag}_gt.nii*")) or sorted(root.glob(f"*{v}_{tag}.nii*"))
        if not gt:
            continue
        lab, _, _ = load_nifti(gt[0])
        frame = 0 if tag == "ED" else max(0, len(images) - 1)
        idx[tag] = frame
        masks[frame] = (lab.round().to(torch.int64) == labels.lv_cavity).to(device=device)

    return CineSequence(
        images=images,
        grid=grid,
        masks=masks,
        ed_index=idx["ED"],
        es_index=idx["ES"],
        patient_id=pid,
        source=f"M&Ms-2/{v}",
        affine=affine,
        metadata={"view": v},
    )
