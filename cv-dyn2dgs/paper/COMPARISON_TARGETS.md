# Comparison targets for CV-Dyn2DGS

A literature-backed list of what this thesis could and should be compared against.
Every entry was verified against the publisher listing, arXiv, or the authors' own
repository; verification status is stated per entry.

**None of these comparisons has been performed.** See `../README.md` for why (no GPU,
no PyTorch in the authoring environment). This document is a comparison *design*, not a
results table.

## The structural problem this document addresses

All 11 baselines in `../cvdyn2dgs/baselines.py` are configurations of the same
pipeline. That is the right choice for attribution — a measured difference cannot come
from engineering — but it means **there is currently no comparison with any published
method**. That is the first thing a committee will ask about.

Claims live at four layers. Mixing layers produces unfair or meaningless comparisons,
because each layer requires holding something different fixed.

| Layer | Claim under test | Hold fixed | Vary |
|---|---|---|---|
| 1. Surface source | the Chan–Vese surface is adequate | representation, renderer, metrics | where the surface comes from |
| 2. Representation | a surface-pinned 2-D disk is the right primitive | **the surface**, supervision views, loss, metrics | what renders it |
| 3. Temporal model | canonical reuse is cheap | quality reference | what is stored/recomputed per frame |
| 4. Storage & playback | surfaces + residual are small and fast | viewpoint set, image quality | representation on disk |

Existing baselines cover layers 2 and 3 *internally*. Layer 1 is absent entirely, and
layer 4 has no external competitor.

---

## Layer 1 — Surface source

**Why this is the most urgent gap.** Chan–Vese is a 2001 method; ACDC is dominated by
deep networks. The defence is not "our Dice is also good" but "the method is agnostic to
the surface source" — and that requires swapping the source while holding everything
else fixed.

| Target | Venue | ID | Code | Role |
|---|---|---|---|---|
| [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z) — Isensee et al. | Nature Methods 18(2):203–211, 2021 | `10.1038/s41592-020-01008-z` | public | the standard strong mask baseline; mask → SDF conversion |
| [CSTM](https://arxiv.org/abs/2410.23191) — Ye, Xin, Axel, Metaxas | WACV 2025 | arXiv:2410.23191 | — | 4D whole-sequence cine segmentation; handles through-plane motion |
| [Queirós et al.](https://doi.org/10.1016/j.media.2014.06.001) | Med. Image Anal. 18(7), 2014 | `10.1016/j.media.2014.06.001` | — | classical 3D surface + anatomically constrained optical flow |
| Ground-truth mask → SDF (*oracle*) | ACDC labels | — | trivial | **must-have**: the only way to separate segmentation error from representation error |

Verification: all four confirmed. Note the source proposal cited an Avendi et al. paper
that does not exist (see the correction note in `refs.bib`); the real works are
[MedIA 2016](https://arxiv.org/abs/1512.07951) and the RV paper in Magn. Reson. Med. 2017,
and they are deep-learning + deformable-model hybrids, not Chan–Vese variants.

Any of the three outcomes is informative:
- quality improves → modularity demonstrated, Chan–Vese is a replaceable part;
- quality unchanged → the bottleneck is the representation, not the surface;
- projection degrades on mask-derived SDFs → **measured evidence that a true signed
  distance field is required**, which is a reason to keep Chan–Vese rather than a defeat.

---

## Layer 2 — Representation primitive (same surface)

| Target | Venue | ID | Code | What it isolates |
|---|---|---|---|---|
| [2DGS](https://arxiv.org/abs/2403.17888) — Huang et al. | SIGGRAPH 2024 | arXiv:2403.17888 | [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) | the primitive this thesis builds on |
| [Gaussian Surfels](https://arxiv.org/abs/2404.17774) — Dai et al. | SIGGRAPH 2024 | arXiv:2404.17774 | [turandai/gaussian_surfels](https://github.com/turandai/gaussian_surfels) | the *other* surfel formulation: a 3-D Gaussian with z-scale set to 0 — i.e. what `thin-3dgs` approximates. **Directly relevant to RQ6.** |
| [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al. | SIGGRAPH 2023 | — | public | geometry-free baseline |
| [QGS](https://arxiv.org/abs/2411.16392) — Quadratic Gaussian Splatting | 2024 | arXiv:2411.16392 | announced | curved generalisation of 2DGS; relevant to curvature-adaptive scales |
| [DG-Mesh](https://arxiv.org/abs/2404.12379) — Liu, Su, Wang | 2024 | arXiv:2404.12379 | [Isabella98Liu/DG-Mesh](https://github.com/Isabella98Liu/DG-Mesh) | mesh with temporal vertex correspondence — the mesh-side analogue of RQ5 |
| [biv-me](https://github.com/UOA-Heart-Mechanics-Research/biv-me) — Dillon, Mauger et al. | Med. Image Anal. art. 104252, 2026 | `10.1016/j.media.2026.104252` | public | **the strongest cardiac surface-only alternative**: validated, multi-cohort, production-grade |
| [Meng et al.](https://arxiv.org/abs/2209.02004) | 2022 | arXiv:2209.02004 | — | learned mesh vertex displacement from SAX+LAX |
| [CineMesh4D](https://arxiv.org/abs/2605.13994) / [Liu et al.](https://arxiv.org/abs/2607.01952) | 2026 | arXiv:2605.13994, 2607.01952 | — | 4D whole-heart mesh with differentiable contour rendering |
| [Bi-PT](https://arxiv.org/abs/2607.06923) | 2026 | arXiv:2607.06923 | — | four-chamber mesh from sparse CMR via NODE deformation |

**Important asymmetry in the current design.** `thin-3dgs` shares CV-Dyn2DGS's anchors,
so it is a *kernel swap*, not a comparison with 3DGS. A genuine 3DGS baseline optimises
its own geometry freely — that is the only way to answer *"is pinning to the surface a
cost or a benefit?"* Expect unconstrained 3DGS to win on PSNR and lose badly on
`E_surf`; that trade-off **is** the thesis argument, so it should be measured, not
assumed.

---

## Layer 3 — Temporal model

| Target | Venue | ID | Code | Notes |
|---|---|---|---|---|
| [**AT-GS**](https://arxiv.org/abs/2411.06602) — Chen, Oberson, Feldmann, Schreer, Hilsmann, Eisert | WACV 2025, pp. 742–752 | arXiv:2411.06602 | [fraunhoferhhi/AT-GS](https://github.com/fraunhoferhhi/AT-GS) | **nearest neighbour in method space — see below** |
| [**Dyna3DGR**](https://arxiv.org/abs/2507.16608) — Fu et al. | MICCAI 2025, pp. 164–174 | arXiv:2507.16608 | [windrise/Dyna3DGR](https://github.com/windrise/Dyna3DGR) | closest cardiac work; **code is public** |
| [D-2DGS](https://arxiv.org/abs/2409.14072) — Zhang et al. | ACM MM 2025 | arXiv:2409.14072 | [hustvl/Dynamic-2DGS](https://github.com/hustvl/Dynamic-2DGS) | sparse control points deform 2-D Gaussians |
| [ST-2DGS](https://arxiv.org/abs/2409.18852) — Wang et al. | 2024 | arXiv:2409.18852 | — | canonical 2DGS + deformation with depth/normal regularisers |
| [DynaSurfGS](https://arxiv.org/abs/2408.13972) | 2024 | arXiv:2408.13972 | — | planar GS + 4D neural voxels for dynamic surfaces |
| [GSTAR](https://arxiv.org/abs/2501.10283) | 2025 | arXiv:2501.10283 | — | **handles topology change** by unbinding Gaussians from the mesh — speaks directly to this thesis's stated topology limitation |
| [Deformable 3DGS](https://arxiv.org/abs/2309.13101) — Yang et al. | 2023 | arXiv:2309.13101 | public | the learned-deformation-field alternative to per-frame level sets |
| 4DGS — Wu et al. | CVPR 2024, pp. 20310–20320 | — | [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians) | the standard dynamic-GS reference |
| [Video4DGen / Dynamic Gaussian Surfels](https://arxiv.org/abs/2504.04153) | 2025 | arXiv:2504.04153 | — | time-varying warps of Gaussian surfels from a static state |
| [4DSurf](https://arxiv.org/abs/2603.28064) | 2026 | arXiv:2603.28064 | — | dynamic surface reconstruction under large deformation |
| [GPTrack](https://arxiv.org/abs/2410.20752) | 2024 | arXiv:2410.20752 | — | cardiac motion tracking, bidirectional recurrence |
| [NDM](https://arxiv.org/abs/2307.07693) / [υNDM](https://arxiv.org/abs/2411.15233) — Ye et al. | 2023 / 2024 | — | — | neural deformable models for heart-wall geometry |

### AT-GS is the citation gap that matters

Chen et al. (WACV 2025) use canonical Gaussian **surfels**, initialise each frame from
the previous one, densify adaptively, and add an explicit temporal-consistency term on
curvature maps to suppress jitter. That is the same *four* mechanisms as CV-Dyn2DGS. The
one decisive difference: AT-GS **learns** the geometry; CV-Dyn2DGS **pins** it to a
variational surface.

Two practical consequences:
1. It reports per-frame training time (38 s/frame on the datasets used), which is
   directly comparable to this thesis's precompute cost.
2. Its curvature-consistency term targets the same phenomenon as the `E_flicker` metric
   here — so `E_flicker` has an external reference point.

The proposal never named it. A thesis on "temporally consistent Gaussian surfels" that
does not cite the paper titled *Adaptive and Temporally Consistent Gaussian Surfels* is
vulnerable.

### Correction to the Dyna3DGR framing

The proposal assumed Dyna3DGR could not be reproduced partly because no implementation
was available. **It is available** at
[windrise/Dyna3DGR](https://github.com/windrise/Dyna3DGR). The only remaining obstacle is
hardware, which is a property of the authoring environment, not of the work. The §5.7
exemption ("report structural differences only if reproduction is difficult") is
therefore provisional, and `sections/02_related_work.tex` now says so.

---

## Layer 4 — Storage and playback

**This layer contains the real threat to the headline compression claim.**

### 4a. The competitor the thesis must not dodge: pre-rendered video

If a clinician views from a handful of fixed angles, H.265/AV1 of pre-rendered frames
beats every 3-D representation — hundreds of KB at 60 FPS. No citation is needed to make
this true. The correct response is to measure the crossover:

| | 1 viewpoint | 4 viewpoints | free viewpoint |
|---|---|---|---|
| pre-rendered video | tiny | ×4 | **impossible** |
| CV-Dyn2DGS | fixed | fixed | fixed |

Reporting the break-even viewpoint count turns *"we win when free viewpoints are needed"*
from an assertion into a measurement. `metrics/storage.py` already has the break-even
machinery; this needs one more axis.

### 4b. Compressed Gaussian representations

These report whole dynamic scenes in the single-MB range. Different task (general scenes,
not organ surfaces), but the claim "surfaces are small" cannot be presented without
acknowledging them.

| Target | ID | Reported |
|---|---|---|
| [Cinematic Anatomy 3DGS](https://arxiv.org/abs/2404.11285) — Niedermayr et al., VMV 2024 | arXiv:2404.11285, [code](https://github.com/KeKsBoTer/cinematic-gaussians) | multi-GB volumes → <70 MB, 60 FPS — **the closest medical storage+FPS claim** |
| [GaussianPile](https://openaccess.thecvf.com/content/CVPR2026/html/Kong_GaussianPile_A_Unified_Sparse_Gaussian_Splatting_Framework_for_Slice-based_Volumetric_CVPR_2026_paper.html) — Kong et al., CVPR 2026 | arXiv:2603.20611 | slice-based volumetric reconstruction, aggressive compression |
| [Spacetime Gaussians](https://arxiv.org/abs/2312.16812) — Li et al., CVPR 2024 | — ([code](https://github.com/oppo-us-research/SpacetimeGaussians)) | compact spacetime primitives, real-time |
| [Memory-Efficient 4DGS](https://arxiv.org/abs/2410.13613) | arXiv:2410.13613 | ~125–190× storage reduction vs 4DGS |
| [OMG4 — Optimized Minimal 4DGS](https://arxiv.org/abs/2510.03857) | arXiv:2510.03857 | >60% size reduction at equal quality |
| [Predictive 4DGS](https://arxiv.org/abs/2510.10030) | arXiv:2510.10030 | ~1 MB average, up to 90× compression |
| [ADC-GS](https://arxiv.org/abs/2505.08196) | arXiv:2505.08196 | anchor-driven; 3–8× faster rendering |
| [4D Scaffold GS](https://arxiv.org/abs/2411.17044) | arXiv:2411.17044 | scaffold-based memory reduction |
| [Rate-distortion optimised 4DGS](https://arxiv.org/abs/2507.17336) | arXiv:2507.17336 | temporal smoothness prior |
| [CC-4DGS](https://arxiv.org/abs/2609.02184) | arXiv:2609.02184 | 1–3 MB deformation storage |
| [Compactness/compression survey](https://arxiv.org/abs/2512.07197) | arXiv:2512.07197 | use for positioning rather than head-to-head |

Author lists for the 4a/4b compression family were not individually verified, so they are
referenced here by arXiv ID and title only and are **not** in `refs.bib`. Only the two
with author-verified BibTeX (Cinematic Anatomy, Spacetime Gaussians) were added.

### 4c. Clinical status quo

MPR and direct volume rendering are what is actually used today. CV-Dyn2DGS is a surface
renderer, so it does **not** compete on image content — only on storage and frame rate.
Compare those axes and state explicitly that image quality is not compared. This is
consistent with the thesis's own limitation ("not a volume renderer").

---

## Positioning only — not comparison targets

Medical Gaussian-splatting work that shares tooling but answers a different question:

| Work | ID | Why not a comparison |
|---|---|---|
| [ClipGS](https://arxiv.org/abs/2507.06647) — MICCAI 2025 | arXiv:2507.06647 | clipping planes on volumetric data |
| [XClipGS](https://arxiv.org/abs/2608.07760) | arXiv:2608.07760 | exact half-space clipping |
| [ClipGS-VR](https://arxiv.org/abs/2601.19310) | arXiv:2601.19310 | mobile VR |
| [Multi-layer GS anatomy](https://arxiv.org/abs/2410.16978) | arXiv:2410.16978 | layered CT anatomy for VR |
| [Rendering novel views of MRI](https://arxiv.org/abs/2606.26236) | arXiv:2606.26236 | **spinal** MRI stenosis grading, not cardiac |
| [X-Gaussian](https://arxiv.org/abs/2403.04116) / [R²-Gaussian](https://arxiv.org/abs/2405.20693) | — | X-ray / CT tomographic reconstruction |

---

## Comparisons that would be invalid

- **This thesis's Dice next to published ACDC deep-network Dice.** Different task: those
  methods produce masks, this one needs a signed distance field. Use the layer-1
  experiment instead.
- **PSNR against volume-rendering methods.** Surface vs volume; the numbers are not
  commensurable.
- **Dyna3DGR's published ~11 min against this thesis's precompute time.** Different
  hardware, data, and objective. Keep it labelled as a claim from the source.
- **Any per-frame metric on ACDC intermediate frames treated as ground truth.** Labels
  exist only at ED and ES; `data/real.py` returns `None` for intermediate masks to
  enforce this at the type level.

## Fairness controls (already in place — extend to new targets)

- All representations scored against the same reference: `render/raymarch.py` ray-marches
  the stored `Γ_t` for silhouette/depth/normal/intensity, so RQ5/RQ6 compare
  *representations*, not segmentations.
- Fitting views (orthographic slice planes) and evaluation views (perspective orbits) are
  disjoint.
- Each target is scored only on axes it can be scored on: a mesh has no opacity, so IoU
  is swept over thresholds rather than fixed at 0.5; a video codec has no geometry metric
  at all.
- Quality, cost and storage are always reported together, so a change that buys quality
  by spending more is visible as such.

## Priority, and the thing to do before any of it

Order: **(1) surface source incl. oracle → (4a) video-codec break-even → free-geometry
3DGS → (4c) MPR/DVR storage+FPS → one external dynamic method (AT-GS or Dyna3DGR)**.

The first three are implementable inside the existing pipeline — no third-party
reproduction needed — which makes them by far the best value.

But first: **fix the non-inferiority margins.** `independent-2dgs` is the quality
*ceiling*; the goal is not to beat it but to land inside a pre-registered margin at much
lower cost. Those margins must be chosen from a ~5-case pilot *before* the main run
(proposal §8.5). The thesis currently states they are deliberately not set yet, which is
honest — and it is the first thing to do when the experiments actually start.
