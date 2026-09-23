"""Rasterisers: perspective-correct 2DGS plus thin-3DGS and mesh baselines."""

from .camera import Camera
from .mesh_render import render_mesh
from .raster2dgs import RenderOutput, render_2dgs, render_2dgs_reference
from .raster3dgs import Thin3DGSConfig, render_thin_3dgs

__all__ = [
    "Camera",
    "RenderOutput",
    "Thin3DGSConfig",
    "render_2dgs",
    "render_2dgs_reference",
    "render_mesh",
    "render_thin_3dgs",
]
