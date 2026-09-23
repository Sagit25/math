# CV-Dyn2DGS — PyTorch implementation

Reference implementation of

> Sukhun Yang, **CV-Dyn2DGS: Chan–Vese-Guided Dynamic 2D Gaussian Surfels for Real-Time
> Cardiac Surface Visualization** — research proposal + mathematical theory.
> Seoul National University, 2026. Advisor: 강명주.

The theory document proves its results and explicitly defers every numerical check to
implementation (§11.3, *"to be verified"*). This repository is that implementation: the
level-set machinery, the surfel geometry, the perspective-correct rasteriser, the
residual solver, the baselines, the metrics, and a test suite that checks the theory's
predicted convergence **rates** rather than only its formulas.

---

## ⚠️ Verification status — read this first

**The PyTorch code has never been executed.** It was written in an environment where
PyTorch could not be installed (no package index, no cached wheels, no GPU).

| Check | Status |
|---|---|
| Byte-compile, all 53 modules + 5 test files | ✅ passes under Python **3.9, 3.10 and 3.12** |
| Every relative import resolves to a defined name | ✅ verified (53 modules, 0 unresolved) |
| `__all__` entries and all test imports resolve | ✅ verified (0 bad entries) |
| Lint — `ruff --select F,E9,B023` | ✅ **All checks passed** (no undefined names, no late-binding closures) |
| Types — `mypy` | ✅ **0 errors in the 4 comparison modules**; 62 package-wide are type-narrowing limits, each checked |
| **Algorithmic kernels** — `scripts/verify_kernels.py` | ✅ **68/68 pass, executed under all three interpreters** |
| **Layer-4 storage logic** — `tests/test_viewpoint.py` | ✅ **34/34 pass under pytest** — runs without PyTorch by design |
| External target fetch — `scripts/fetch_external.py` | ✅ **executed**: cloned at pinned SHAs, `--check` verified, git-ignore guard confirmed |
| Line-by-line review of tensor shapes and math | ✅ done (5 real bugs found and fixed) |
| Manuscript structure — `scripts/check_paper.py` | ✅ **executed, no problems** (refs, citations, balance, table provenance) |
| **The other 4 test files** | ❌ **cannot even be collected** — `import torch` fails |
| **Running anything that needs PyTorch** | ❌ **not done** |
| **Any performance or quality number** | ❌ **none measured, none claimed** |
| Compiling the manuscript (`paper/`) | ❌ **not done** — no LaTeX distribution available |

Concretely: **1 of 5 test files runs here, and it passes.** `pytest tests/` reports
`34 tests collected, 4 errors` — the four errors are `ModuleNotFoundError: No module
named 'torch'` at import time, not failures.

`scripts/verify_kernels.py` is stdlib-only and *has* been run. It re-implements the
discrete logic where an off-by-one corrupts results silently rather than raising — the
marching-tetrahedra table, the tile-binning index arithmetic, bit packing, the ellipsoid
SDF bisection bracket, trilinear weight ordering, the CG iteration, the upwind
reinitialisation stencil, the storage algebra — and checks each against an independent
reference. It found one bug on first run (in its own sampling, not the decomposition:
the offender count matched the inclusion–exclusion prediction of 96 exactly, which
confirmed the tetrahedra were right).

Everything above the PyTorch line is still unverified. **Expect bugs on the first real
run.** There are no committed result files because no results exist; publishing invented
numbers would be worse than publishing none.

---

## Install

```bash
pip install -e .            # core: torch + numpy
pip install -e '.[data]'    # + nibabel, for ACDC / M&Ms-2
pip install -e '.[dev]'     # + pytest
```

## Bring-up order

Run these in order. Each one assumes the previous passed.

```bash
# 0) no PyTorch needed — the only part already known to pass
python scripts/verify_kernels.py

# 1) staged diagnostic: ~20 independent stages, ONE run reports every failure.
#    Stages are ordered by dependency, so fix the FIRST failure and re-run.
cvdyn2dgs smoke                 # core stages, CPU-friendly
cvdyn2dgs smoke --full          # adds precompute / playback / storage / metrics

# 2) unit tests, once smoke is green
pytest -x tests/

# 3) the theory's predicted convergence rates, measured
cvdyn2dgs theory

# 4) one phantom end to end: precompute + storage + playback FPS
cvdyn2dgs demo --preset v3-adaptive --frames 12 --shape 64 64 16

# 5) the full programme: theory checks, baselines, RQ1–RQ6, ablations
cvdyn2dgs experiments --out results --size small

# 6) a real ACDC patient (needs nibabel + the dataset)
cvdyn2dgs acdc /path/to/ACDC/training/patient001
```

Why `smoke` before `pytest`: pytest stops at the first failure, which forces a
one-bug-per-run loop. `smoke` catches per stage and prints a table, so a single run tells
you everything that is broken. It also asserts the things most likely to be wrong with an
unrun implementation — sign conventions, frame orthonormality, the adjoint identity, that
playback does not mutate the canonical set — with error messages that name the probable
cause (e.g. *"a sign flip here means the inside/outside convention is inverted"*).

On CPU use `--size tiny` and lower `fit.iters`; the defaults assume a GPU.

## Minimal API example

```python
import torch
from cvdyn2dgs.core.config import get_preset
from cvdyn2dgs.data.phantom import PhantomConfig, make_phantom
from cvdyn2dgs.experiments.common import run_pipeline_on_phantom
from cvdyn2dgs.pipeline.playback import PlaybackEngine

phantom = make_phantom(PhantomConfig(shape=(96, 96, 16), n_frames=20))
cfg = get_preset("v3-adaptive")

model, cams, timer = run_pipeline_on_phantom(phantom, cfg, verbose=True)
# model is D = {G^2D_0, {Gamma_t}, {Delta a_t}}  (Eq. 6)

engine = PlaybackEngine(model)
out, timing = engine.step_to(t=5, camera=cams.eval[0])
print(out.color.shape, timing.total_ms, timing.fps)
```

---

## How the code maps onto the papers

Sign convention everywhere: **`phi > 0` inside the heart** (proposal Eq. 1) — the
opposite of the usual "inside is negative" SDF convention.

| Module | Paper | Implements |
|---|---|---|
| `levelset/operators.py` | theory §4.4, Eq. 4.12–4.15 | **Contribution 1**: spacing-aware `grad_h`, `div_h`, curvature; `H_eps`/`delta_eps` (Eq. 3.7–3.8); energy (Eq. 4.1); region means (Eq. 4.3–4.4); gradient flow (Eq. 4.7) |
| `levelset/chanvese.py` | theory §4–§5 | Alternating minimisation, explicit Euler (Eq. 4.16), narrow-band **warm start** (Eq. 5.1) |
| `levelset/sdf.py` | Prop. 3.1, §10 | Eikonal reinitialisation; mid-time interpolation (Eq. 10.1) with the Prop. 10.1 violation diagnostic |
| `levelset/mesh_extract.py` | proposal §8.2 | Marching **tetrahedra** (16-entry table, not 256) for the mesh baseline |
| `surfel/model.py` | Def. 7.1, Eq. 7.5 | `G^2D_0`; **no normal-direction scale**; geometry as buffers, not parameters |
| `surfel/projection.py` | Eq. 6.2–6.3 | **Contribution 2**: minimal normal step + Newton-like iteration; `E_surf` (Eq. 39); Lemma 6.3 residual history |
| `surfel/transport.py` | Eq. 7.6–7.7 | **Contribution 3**: minimum-rotation tangent transport, Prop. 7.7 degeneracy fallback, Rodrigues alternative |
| `surfel/density.py` | proposal §6.4 | Tangential repulsion, bounded densification, pruning |
| `render/raster2dgs.py` | Eq. 8.1–8.6 | Perspective-correct ray–surfel intersection, tiled + differentiable; brute-force reference renderer; sparse `A_t` (Prop. 9.1) |
| `render/raster3dgs.py` | Prop. 8.4 | Thin-3DGS baseline (affine EWA projection) |
| `render/mesh_render.py` | proposal §4.3 | Z-buffered textured mesh with level-set vertex normals |
| `render/raymarch.py` | proposal §7.1 | Reference silhouette/depth/normal/intensity from the stored surface |
| `losses.py` | Eq. 33 | `L_app + λ_m L_mask + λ_n L_normal + λ_d L_dist` |
| `residual/solver.py` | Eq. 9.2–9.3 | **Contribution 4** (part): matrix-free CG on the SPD normal equations, Eq. 9.4 condition bound |
| `residual/lowrank.py` | Eq. 9.5, Prop. 9.3 | SVD truncation with the Eckart–Young error identity |
| `metrics/storage.py` | Eq. 36–38 | `S_full`, `S_ours`, `CR`, **break-even surface budget** |
| `pipeline/playback.py` | Eq. 34 | `T_surface + T_project + T_orient + T_raster`, measured per stage |
| `experiments/theory_checks.py` | all of the above | 14 numerical checks of the theory's claims |

### The three presets are a development story, not three configs

| Preset | Adds | Paper scope |
|---|---|---|
| `v1-minimal` | isotropic disks, `K_p=1`, no geometry losses, no density control | §10.4 *minimum success scope* |
| `v2-geometry` | Eq. 33 geometry losses + anisotropic transport (Eq. 7.6–7.7) | — |
| `v3-adaptive` | curvature-adaptive scales (Prop. 8.3), repulsion/densify/prune, temporal residual, low-rank (Eq. 9.5) | §10.4 *extension scope* |

`experiments/research_questions.py::progressive_development` measures what each stage
buys, reporting quality **and** cost **and** storage together — so a stage that improves
quality only by spending more is visible as such.

---

## Design decisions worth knowing about

**Geometry cannot be optimised.** Anchors, tangent frames and normals are registered as
`nn.Module` *buffers*. No loss can reach them. Proposal §6.6 says the geometry is pinned
to the Chan–Vese surface; the implementation enforces that structurally instead of
relying on a learning rate of zero. Only amplitude, opacity and scale are fitted.

**Fitting views and evaluation views are disjoint.** The canonical fit is supervised on
orthographic slice-plane cameras (the "known physical coordinates" of §7.1). Quality is
measured on perspective orbit cameras that were never fitted.

**All representations are scored against the same reference.** `render/raymarch.py`
ray-marches the stored `Gamma_t` to produce reference silhouette / depth / normal /
surface intensity. 2DGS, thin-3DGS and the mesh are all scored against it, so RQ5/RQ6
compare *representations*, not segmentations.

**The narrow band is a dense crop, not a sparse list.** On a GPU, dense arithmetic on a
small bounding box beats sparse indexing. The band is realised as a crop plus a masked
update; the region means of Eq. 4.3–4.4 are still computed over the **full** volume,
because cropping them would bias `c_out`.

**Two documented approximations in the fast rasteriser**, both measurable against
`render_2dgs_reference`:
1. per-tile ordering is by anchor depth, not per-pixel `tau`;
2. `max_per_tile` truncates to the nearest N surfels.

**Storage accounting is in bytes, with overhead counted** — index and occupancy bitmask
costs included, reported for all three surface modes, with the theoretical Eq. 37 count
next to the measured file size.

**Seeking is an open question, and the code says so.** Eq. 6 stores no per-frame anchors,
so a random seek must project the *canonical* anchors onto `Gamma_t`, whereas
precomputation projected sequentially. `playback.compare_projection_modes` measures the
gap. If it is large, honest accounting must either add per-frame anchors or restrict the
viewer to sequential playback.

---

## Known limitations

These are properties of the method or the implementation, not bugs to be fixed quietly.

* **Not a digital twin.** A time-varying anatomical visualisation, no tissue mechanics,
  electrophysiology or flow (proposal §2.11).
* **Not tissue tracking.** Normal projection takes the least-norm displacement; the
  tangential component is undetermined by design. Do not read the output as strain
  (§2.8, §4.2, §10.3).
* **Not a volume renderer.** 2DGS renders a surface. Arbitrary MRI cross-sections need a
  volumetric model (§10.3, limitation 4).
* **LV cavity only.** Full LV/RV/myocardium needs a multiphase level set; the real-data
  loader raises `NotImplementedError` rather than silently returning something else.
* **Chan–Vese assumes piecewise-constant intensity.** The phantom includes papillary
  muscles specifically to violate this (§2.5, §10.3 limitation 1).
* **Dyna3DGR is not reproduced.** No number is attributed to it. Following §5.7,
  `baselines.DYNA3DGR_COMPARISON` records the structural comparison, and the published
  ~11 min figure is labelled as a claim from the source, not a measurement.
* **Mid-time interpolation is display-only.** Prop. 10.1 / 10.2 prove the interpolated
  level set is not the true intermediate surface.
* **The phantom is a validation instrument, not a dataset.** It exists because no real
  dataset provides an exact SDF, exact normals, or variable `h` — all of which the rate
  checks need. It says nothing about robustness to real anatomy or pathology.
* **Statistics are not implemented.** Proposal §8.5 requires paired per-patient
  comparison with bootstrap CIs and Wilcoxon signed-rank tests over 20–30 cases. The
  harness produces per-case numbers; the aggregation is left to be added with real data,
  and single-phantom results must not be read as evidence about patients.
* **Non-inferiority margins are deliberately not chosen here.** §8.5 requires fixing them
  from a ~5-case pilot *before* the main run.

## Repository layout

```
cvdyn2dgs/
  core/        grid geometry, configs & presets, determinism, staged timing
  levelset/    spacing-aware operators, Chan-Vese, SDF tools, mesh extraction
  surfel/      canonical set, normal projection, tangent transport, density control
  render/      2DGS / thin-3DGS / mesh rasterisers, cameras, ray-march reference
  residual/    regularised least squares (CG), low-rank compression
  data/        synthetic 4-D phantom, ACDC & M&Ms-2 loaders
  metrics/     segmentation, rendering, photometric, clinical, storage
  pipeline/    precompute, canonical fitting, playback, serialisation
  experiments/ theory checks, RQ1-RQ6, ablations, runner
  baselines.py the proposal §8.2 baseline family
  smoke.py     25 staged diagnostics (`cvdyn2dgs smoke`)
  cli.py       command-line entry point
tests/         pytest suite (run this first)
scripts/
  verify_kernels.py  stdlib-only kernel checks — the one thing already executed
  fetch_external.py  clone the external comparison targets at pinned commits
  fetch_datasets.py  dataset provenance + layout/geometry verification (no torch needed)
  make_tables.py     generates paper/tables/*.tex from results/*.json
  check_paper.py     structural checks on the manuscript, no LaTeX needed
external/      pinned comparison targets and dataset provenance (nothing vendored)
paper/         LaTeX thesis source — see paper/README.md
```

## Datasets

The cardiac datasets **cannot be cloned**: ACDC, M&Ms, M&Ms-2 and the STACOM motion
benchmark each require individual registration and a data-use agreement.
`external/datasets.json` records provenance, licence, citation duty and expected layout
for seven datasets; `scripts/fetch_datasets.py` refuses to bypass any gate and instead
verifies a copy you obtained yourself.

```sh
python scripts/fetch_datasets.py --list          # 7 datasets, 5 need registration
python scripts/fetch_datasets.py --how acdc      # exact access steps + expected layout
python scripts/fetch_datasets.py --verify acdc --path /data/ACDC
```

`--verify` exists because of a specific trap. Contribution 1 is the spacing-aware
discretisation, and its ablation can only show anything on **anisotropic** data. On a
resampled copy the ablation comes out flat — which reads as *"the operators don't help"*
rather than *"this was unmeasurable"*. So the verifier rejects near-isotropic copies, and
separately **fingerprints known pre-processed mirrors**, because a mirror that resamples
in-plane only keeps its 10 mm through-plane spacing and passes a naive ratio test. The
public Hugging Face ACDC mirror (1×1×10 mm, cropped to 192×192) is exactly that case and
is rejected by name.

The NIfTI header parser is standard-library only, so this runs before `nibabel`, `numpy`
or PyTorch are installed — it can be the first thing you do after downloading, not
something you find out later. Sections 15–16 of `scripts/verify_kernels.py` cover it and
do run.

## Comparison targets

The eight original baselines are all configurations of *this* pipeline. That is ideal for
attribution — a measured difference cannot come from engineering — but it means the family
contained **no comparison with any published method**, and it never varied the surface
source at all. Both gaps are now addressed across four layers
([`paper/COMPARISON_TARGETS.md`](paper/COMPARISON_TARGETS.md) has the full survey with
venues, arXiv IDs and per-target caveats).

**Implemented in-pipeline** — no third-party code needed, so these are the cheapest to run
and the easiest to keep fair:

| Layer | What varies | Module | New baselines |
|---|---|---|---|
| 1 | the surface source | `levelset/surface_source.py` | `source-oracle`, `source-mask`, `source-chanvese-no-spacing` |
| 2 | geometry free vs pinned | `surfel/free3dgs.py` | `free-3dgs-surface`, `free-3dgs-bbox_random`, `gaussian-surfel-exact` |
| 4 | storage vs viewpoint count | `metrics/viewpoint.py` | pre-rendered video, raw 4-D volume |

Three design choices in there are load-bearing:

- **`source-oracle` is mandatory, not optional.** Without an exact surface, segmentation
  error and representation error are confounded and no RQ5/RQ6 number can be attributed.
- **`free-3dgs` is the mirror image of `SurfelSet2D`**: geometry is `nn.Parameter` here and
  `buffer` there, so neither can become the other by configuration. `FreeFitReport.summary()`
  refuses to emit photometric metrics without geometric ones, because free Gaussians trade
  surface fidelity for image fidelity and quoting one side inverts the conclusion.
- **Layer 4 has no default bitrate.** `VideoCodecTarget` requires a measured encode or an
  explicitly declared assumption with its basis. Guessing the competitor's number is worse
  than guessing your own. `StorageOnlyTarget.score_axis()` raises for geometry axes, so a
  video clip cannot accidentally acquire a Dice score.

**External code is pinned, never vendored.** `external/manifest.json` pins 15 targets by
commit SHA; `scripts/fetch_external.py --fetch <key>` clones them into the git-ignored
`external/repos/`. Four of them may not legally be redistributed here — `gaussian_surfels`
and `cinematic-gaussians` ship **no licence file** at all, and Inria/MPII's 3DGS and 2DGS
are research-only — and vendoring everything would add ~1.2 GB of mostly binary content.
Pinned SHAs are also simply better evidence: they record which commit a number came from.

Genuinely external results are scored by `experiments/external.py` against **the same
ray-marched reference** as every internal baseline. Channels a method did not emit are
reported as not measured, never as zeros: `ExternalOutput.require_axis()` raises rather
than letting a missing depth map score as a specific wrong answer.

> **None of the external comparisons has been run.** No GPU was available. The tables exist
> as stubs so the results have somewhere to land, and `cvdyn2dgs compare` prints the design
> with every unrun comparison marked as such.

## The manuscript

[`paper/`](paper/) holds the thesis LaTeX source (Korean body, ko.TeX). Chapters 4
(방법) and 5 (오차 분석) are complete with proofs; **chapter 8 (결과) contains no
numbers**, because nothing has been measured. The draft state is marked in the document
itself: unmeasured cells render as red `n/a`, result-dependent paragraphs as red
`[결과 대기]` boxes, and every page carries a banner.

Result tables are **generated** from `results/*.json` by `scripts/make_tables.py` — the
manuscript only `\input`s them, so no table can hold a number that no run produced. 11
of the 12 are currently stubs; the exception is the kernel-verification table, which is
filled from the checks that do run.

The manuscript has never been compiled — no LaTeX distribution was available.
`python scripts/check_paper.py` substitutes for the structural half of a compile
(references, citations, balance, table provenance) and currently reports no problems.
See [`paper/README.md`](paper/README.md) for build commands and the exact steps to turn
draft mode off once results exist.

## License

MIT.
