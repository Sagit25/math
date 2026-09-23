# External comparison targets

Third-party code is **not** stored here. Each target is pinned by commit SHA in
[`manifest.json`](manifest.json) and cloned on demand into `repos/`, which is
git-ignored.

```sh
python scripts/fetch_external.py --list       # all targets and their state
python scripts/fetch_external.py --licences   # redistribution restrictions
python scripts/fetch_external.py --fetch at-gs dyna3dgr --submodules
python scripts/fetch_external.py --check      # fetched trees still match the pins
```

## Why nothing is vendored

Legal, first. Four targets may not be redistributed inside this repository:

| Target | Licence | Consequence |
|---|---|---|
| `turandai/gaussian_surfels` | **no LICENSE file** | default copyright, all rights reserved |
| `KeKsBoTer/cinematic-gaussians` | **no LICENSE file** | default copyright, all rights reserved |
| `graphdeco-inria/gaussian-splatting` | Inria/MPII research-only | redistribution restricted |
| `hbb1/2d-gaussian-splatting` | Inria/MPII research-only | redistribution restricted |

Cloning them and running them for evaluation is fine. Copying them into this tree is
not. `repos/` is git-ignored so that it cannot happen by accident, and
`fetch_external.py` prints the restriction again after every non-redistributable clone.

Practical, second. Vendoring all of them adds roughly 1.2 GB, most of it binary
assets and CUDA extension sources with their own submodules. The transfer patch for
this repository is plain text with zero binary hunks; vendoring would end that.

Reproducibility, third — and this is the real argument. A pinned SHA records exactly
which commit a number came from. A vendored copy records only that someone once
copied something.

## What is pinned

16 targets across the four comparison layers. `manifest.json` carries, per target:
commit SHA and branch, SPDX licence and a `redistributable` flag, the comparison layer,
the `refs.bib` key, how its output is consumed (`mask_sequence`, `mesh_sequence`,
`rendered_views`, `storage_only`), which adapter reads it, what it needs (`gpu`,
`cuda-ext`, `submodules`, `trained-weights`, `acdc-data`, `ffmpeg`), and a caveat
naming the asymmetry that must be reported with any number from it.

Two entries have `repo: null` because no public implementation was found (`cstm`) or
none is needed (`video-codec`, implemented in-pipeline). They are listed anyway so the
gap is visible rather than silently absent.

## What is implemented here instead

Three comparisons do **not** need any of these clones, because they fit inside the
existing pipeline — which makes them by far the cheapest to run and the easiest to keep
fair:

| Comparison | Module |
|---|---|
| Layer 1 — swap the surface source (incl. an oracle surface) | `cvdyn2dgs/levelset/surface_source.py` |
| Layer 2 — free-geometry 3DGS, and an exact Gaussian-surfel primitive | `cvdyn2dgs/surfel/free3dgs.py` |
| Layer 4 — pre-rendered video and raw-volume budgets, viewpoint break-even | `cvdyn2dgs/metrics/viewpoint.py` |

Anything genuinely external is scored through
`cvdyn2dgs/experiments/external.py`, which loads rendered views from a directory and
runs **the same metrics against the same ray-marched reference** as the internal
baselines. No third-party method is reimplemented and no number is attributed to one
that was not actually run.

## Do not clean inside `repos/`

A repo-wide tidy-up will silently invalidate the pins:

```sh
find . -name '__pycache__' -type d -exec rm -rf {} +   # ← recurses into repos/
```

Some pinned repositories commit `.pyc` files, so deleting them makes the working tree
differ from the pinned commit. `--check` catches it — this was found that way while
verifying this directory:

```
  dyna3dgr: OK  (working tree modified)
  1 problem(s): a measurement taken from these trees is not reproducible from the manifest
```

Scope such commands with `-not -path './external/repos/*'`, or recover with
`git -C external/repos/<key> checkout -- .` and re-run `--check`.

## Datasets

[`datasets.json`](datasets.json) does the same job for the data, but the situation is
different: **the cardiac datasets cannot be fetched at all.** ACDC, M&Ms, M&Ms-2 and the
STACOM motion benchmark each require individual registration and a data-use agreement. No
script may bypass that and `scripts/fetch_datasets.py` does not try — for a gated dataset
it prints exactly what to do and stops.

```sh
python scripts/fetch_datasets.py --list
python scripts/fetch_datasets.py --how acdc
python scripts/fetch_datasets.py --verify acdc --path /data/ACDC
```

What the script is actually for is the third command. It checks a manually obtained copy
against the layout `cvdyn2dgs/data/real.py` really globs for, and then checks the voxel
geometry — which is where the interesting failure lives.

### Why geometry verification matters more than file counts

Contribution 1 is the spacing-aware discretisation, and its ablation
(`spacing_aware=False`) can only show anything on **anisotropic** data. ACDC's
through-plane spacing is typically 5–10 mm against 1.4–1.8 mm in plane. Run the ablation
on a resampled copy and it comes out flat — and flat reads as *"the spacing-aware
operators don't help"* rather than *"this measurement was impossible"*. That is a
silently wrong conclusion, not a missing one.

Two defences, because one is not enough:

1. **Ratio test** — reject a copy whose median through-plane : in-plane ratio is below 2.
2. **Fingerprint match** — a mirror that resamples *in plane only* keeps a 10 mm
   through-plane spacing and therefore **passes** the ratio test. The public Hugging Face
   mirror of ACDC is exactly this case: resampled to 1×1×10 mm and centre-cropped to
   192×192. It is matched by spacing-and-shape fingerprint instead, and rejected with the
   reason. A copy where every subject shares one spacing is also rejected, since
   multi-scanner data varies.

The header parser is standard-library only, so verification runs **before** `nibabel`,
`numpy` or PyTorch are installed — i.e. it can be the first thing you do after obtaining
data rather than something you discover afterwards. It is covered by sections 15–16 of
`scripts/verify_kernels.py`, which do run.

One dataset is listed specifically so it is *not* used: MSD Task02 Heart is genuinely
open (CC-BY-SA 4.0, no registration) and still wrong — it is left-atrium segmentation
from single-phase 3-D MR, with no time dimension and no LV cavity label. Nothing about
warm starting, transport, flicker or playback can be measured on it.

## Status

**None of these comparisons has been run, and no real dataset has been obtained.** No GPU
was available where this was written. See
[`../paper/COMPARISON_TARGETS.md`](../paper/COMPARISON_TARGETS.md) for the full design and
[`../README.md`](../README.md) for what has and has not been executed.
