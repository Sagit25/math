"""CV-Dyn2DGS - Chan-Vese-guided dynamic 2D Gaussian surfels for cardiac surfaces.

A PyTorch implementation of the method described in

    Sukhun Yang, "CV-Dyn2DGS: Chan-Vese-Guided Dynamic 2D Gaussian Surfels for
    Real-Time Cardiac Surface Visualization" (proposal + theory), Seoul National
    University, 2026.

The package is organised so that each module maps onto a section of those
documents:

===========================  ====================================================
``cvdyn2dgs.core``           Grid geometry, configs, determinism, timing
``cvdyn2dgs.levelset``       Theory §3-§5: spacing-aware operators, Chan-Vese,
                             SDF reinitialisation, mesh extraction
``cvdyn2dgs.surfel``         Theory §6-§7: canonical surfels, normal projection,
                             tangent transport, density control
``cvdyn2dgs.render``         Theory §8: perspective-correct 2DGS rasterisation,
                             plus thin-3DGS and mesh baselines
``cvdyn2dgs.losses``         Proposal Eq. (33): appearance + mask + normal + dist
``cvdyn2dgs.residual``       Theory §9: regularised least squares, low-rank
``cvdyn2dgs.data``           Synthetic 4-D phantom and real cine-CMR loaders
``cvdyn2dgs.metrics``        Proposal §8.3: all reported metrics
``cvdyn2dgs.pipeline``       Precompute / playback / storage format
``cvdyn2dgs.baselines``      Proposal §8.2 baseline family
``cvdyn2dgs.experiments``    RQ1-RQ6, ablations, theory verification
===========================  ====================================================

Sign convention reminder: ``phi > 0`` inside the heart (proposal Eq. 1).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
