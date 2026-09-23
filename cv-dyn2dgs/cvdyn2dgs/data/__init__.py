"""Data sources: the synthetic 4-D phantom and real cine-CMR loaders.

The phantom is the *validation instrument* (exact SDF, exact normals, variable
spacing) used to check the theory's convergence rates; the real loaders are the
*application* path. Neither substitutes for the other.
"""

from .phantom import Phantom4D, PhantomConfig, contraction_profile, ellipsoid_sdf, make_phantom
from .real import (
    CineSequence,
    LabelCodes,
    load_acdc_patient,
    load_mnms2_patient,
    load_nifti,
    normalize_intensity,
)

__all__ = [
    "CineSequence",
    "LabelCodes",
    "Phantom4D",
    "PhantomConfig",
    "contraction_profile",
    "ellipsoid_sdf",
    "load_acdc_patient",
    "load_mnms2_patient",
    "load_nifti",
    "make_phantom",
    "normalize_intensity",
]
