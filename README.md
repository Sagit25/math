# math

Mathematics research code.

## Projects

### [`cv-dyn2dgs/`](cv-dyn2dgs/) — CV-Dyn2DGS

PyTorch implementation of *CV-Dyn2DGS: Chan–Vese-Guided Dynamic 2D Gaussian Surfels for
Real-Time Cardiac Surface Visualization* (graduation thesis, SNU 2026).

Variational segmentation (Chan–Vese level sets) coupled to a surface-native renderer
(2-D Gaussian surfels): a canonical surfel set is built once on the first frame and
transported to later frames by normal projection onto each frame's level set, so only
the surfaces and a small appearance residual are stored.

Includes the level-set solver, the perspective-correct surfel rasteriser, the baseline
family, the metric suite, and numerical verification of the theory's predicted
convergence rates — plus the thesis LaTeX source in
[`cv-dyn2dgs/paper/`](cv-dyn2dgs/paper/).

> **Status:** written but **not yet executed** — PyTorch was unavailable in the authoring
> environment, so no performance or quality number has been measured and none is claimed.
> The manuscript's method and error-analysis chapters are complete; its results chapter is
> deliberately empty. See
> [`cv-dyn2dgs/README.md`](cv-dyn2dgs/README.md#️-verification-status--read-this-first)
> and [`cv-dyn2dgs/paper/README.md`](cv-dyn2dgs/paper/README.md) before relying on either.
