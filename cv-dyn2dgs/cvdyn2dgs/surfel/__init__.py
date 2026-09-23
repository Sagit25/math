"""2-D Gaussian surfels: theory §6-§7."""

from .canonical import initialize_canonical_surfels, sample_mesh_surface, surface_normals_at
from .density import (
    DensityStats,
    apply_density_control,
    densify,
    knn,
    nearest_neighbour_stats,
    prune,
    tangential_repulsion,
)
from .model import StorageCount, SurfelSet2D
from .projection import ProjectionResult, e_surf, project_to_surface, surface_seed_points
from .transport import (
    TransportDiagnostics,
    initial_tangent_frame,
    minimal_rotation_matrix,
    transport_tangent_frame,
)

__all__ = [
    "DensityStats",
    "ProjectionResult",
    "StorageCount",
    "SurfelSet2D",
    "TransportDiagnostics",
    "apply_density_control",
    "densify",
    "e_surf",
    "initial_tangent_frame",
    "initialize_canonical_surfels",
    "knn",
    "minimal_rotation_matrix",
    "nearest_neighbour_stats",
    "project_to_surface",
    "prune",
    "sample_mesh_surface",
    "surface_normals_at",
    "surface_seed_points",
    "tangential_repulsion",
    "transport_tangent_frame",
]
