"""Grid geometry, interpolation and the spacing-aware operators.

These are the foundation: if the finite differences do not carry the voxel spacing
correctly, every downstream number in the paper is wrong for a reason that would be very
hard to find later. The tests therefore check the operators against *analytic* answers
rather than against each other.
"""

from __future__ import annotations

import math

import pytest
import torch

from cvdyn2dgs.core.grid import Grid, trilinear_sample, trilinear_sample_vector
from cvdyn2dgs.levelset.operators import (
    central_diff,
    curvature,
    dirac_eps,
    divergence_backward,
    forward_diff,
    gradient_central,
    gradient_norm,
    heaviside_eps,
    region_means,
)

ANISO = Grid(shape=(24, 20, 12), spacing=(1.25, 1.25, 8.0))


def test_voxel_world_roundtrip():
    g = Grid(shape=(10, 12, 8), spacing=(0.5, 1.5, 4.0), origin=(3.0, -2.0, 11.0))
    idx = torch.rand(500, 3, dtype=torch.float64) * torch.tensor([9.0, 11.0, 7.0])
    back = g.world_to_voxel(g.voxel_to_world(idx))
    assert torch.allclose(idx, back, atol=1e-10)


def test_trilinear_is_exact_on_affine_functions():
    """Trilinear interpolation reproduces any affine function exactly."""
    g = Grid(shape=(9, 11, 7), spacing=(1.25, 2.0, 5.0))
    world = g.world_meshgrid(dtype=torch.float64)
    vol = 3.0 * world[0] - 2.0 * world[1] + 0.5 * world[2] + 7.0

    pts_vox = torch.rand(400, 3, dtype=torch.float64) * torch.tensor([8.0, 10.0, 6.0])
    got = trilinear_sample(vol, pts_vox)
    w = g.voxel_to_world(pts_vox)
    want = 3.0 * w[:, 0] - 2.0 * w[:, 1] + 0.5 * w[:, 2] + 7.0
    assert torch.allclose(got, want, atol=1e-9)


def test_trilinear_vector_matches_scalar_per_channel():
    vol = torch.randn(3, 8, 8, 8, dtype=torch.float64)
    pts = torch.rand(50, 3, dtype=torch.float64) * 7.0
    vec = trilinear_sample_vector(vol, pts)
    for c in range(3):
        assert torch.allclose(vec[:, c], trilinear_sample(vol[c], pts), atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_central_difference_is_exact_on_linear_ramps(axis):
    """A linear ramp along one axis must give exactly its slope, spacing included."""
    g = ANISO
    world = g.world_meshgrid(dtype=torch.float64)
    slope = 2.5
    vol = slope * world[axis]

    d = central_diff(vol, axis, g.spacing)
    interior = [slice(1, -1)] * 3
    assert torch.allclose(d[tuple(interior)], torch.full_like(d[tuple(interior)], slope), atol=1e-9)


def test_spacing_awareness_changes_the_answer_on_anisotropic_grids():
    """Ignoring spacing is equivalent to h = 1 and must differ when h != 1.

    This is the direct unit-level check of the paper's first contribution.
    """
    g = ANISO
    world = g.world_meshgrid(dtype=torch.float64)
    vol = world[2]  # ramp along the coarse axis

    aware = central_diff(vol, 2, g.spacing, spacing_aware=True)
    naive = central_diff(vol, 2, g.spacing, spacing_aware=False)
    inner = (slice(1, -1),) * 3
    assert math.isclose(float(aware[inner].mean()), 1.0, rel_tol=1e-9)
    assert math.isclose(float(naive[inner].mean()), g.hz, rel_tol=1e-9)


def test_gradient_of_sphere_sdf_is_unit_and_radial():
    """For phi = R - |x| (positive inside), grad phi is the inward unit normal."""
    g = Grid(shape=(41, 41, 41), spacing=(1.0, 1.0, 1.0))
    world = g.world_meshgrid(dtype=torch.float64)
    centre = g.center_world(dtype=torch.float64).view(3, 1, 1, 1)
    r = (world - centre).norm(dim=0)
    phi = 12.0 - r

    grad = gradient_central(phi, g.spacing)
    band = (phi.abs() < 4.0) & (r > 2.0)
    n = gradient_norm(grad)
    assert float((n[band] - 1.0).abs().mean()) < 2e-2

    radial = (world - centre) / r.clamp_min(1e-9)
    cos = (grad * (-radial)).sum(dim=0) / n.clamp_min(1e-9)
    assert float(cos[band].min()) > 0.99


def test_divergence_of_constant_field_vanishes():
    g = ANISO
    field = torch.ones((3,) + g.shape, dtype=torch.float64)
    div = divergence_backward(field, g.spacing)
    inner = (slice(1, -1),) * 3
    assert float(div[inner].abs().max()) < 1e-12


def test_curvature_of_sphere_matches_two_over_r():
    """div(grad phi / |grad phi|) for an inward-normal sphere SDF is -2/R.

    Sign: phi = R - |x| has an inward gradient, and the divergence of the inward unit
    normal on a sphere of radius R is -2/R.
    """
    g = Grid(shape=(61, 61, 61), spacing=(1.0, 1.0, 1.0))
    world = g.world_meshgrid(dtype=torch.float64)
    centre = g.center_world(dtype=torch.float64).view(3, 1, 1, 1)
    r = (world - centre).norm(dim=0)
    radius = 18.0
    phi = radius - r

    kap = curvature(phi, g.spacing, eps=1e-12)
    shell = (phi.abs() < 1.5) & (r > 4.0)
    measured = float(kap[shell].mean())
    assert measured == pytest.approx(-2.0 / radius, abs=0.03)


def test_heaviside_and_dirac_properties():
    """Prop. 3.2: H in (0,1) and monotone; delta integrates to 1."""
    z = torch.linspace(-500.0, 500.0, 200001, dtype=torch.float64)
    h = heaviside_eps(z, 1.0)
    assert bool((h > 0).all()) and bool((h < 1).all())
    assert bool((h[1:] >= h[:-1]).all())

    dz = float(z[1] - z[0])
    integral = float((dirac_eps(z, 1.0) * dz).sum())
    assert integral == pytest.approx(1.0, abs=2e-3)

    # delta is the derivative of H
    num = (h[2:] - h[:-2]) / (2 * dz)
    ana = dirac_eps(z[1:-1], 1.0)
    assert float((num - ana).abs().max()) < 1e-6


def test_region_means_recover_piecewise_constants():
    """Eq. (4.3)-(4.4) on a noiseless two-region image."""
    g = Grid(shape=(24, 24, 24), spacing=(1.0, 1.0, 1.0))
    world = g.world_meshgrid(dtype=torch.float64)
    centre = g.center_world(dtype=torch.float64).view(3, 1, 1, 1)
    phi = 8.0 - (world - centre).norm(dim=0)
    img = torch.where(phi > 0, torch.full_like(phi, 0.8), torch.full_like(phi, 0.2))

    c_in, c_out = region_means(img, phi, eps=0.3)
    assert float(c_in) == pytest.approx(0.8, abs=0.02)
    assert float(c_out) == pytest.approx(0.2, abs=0.02)


def test_forward_difference_respects_neumann_boundary():
    """Eq. (4.6): the outward difference across the last face is zero."""
    g = Grid(shape=(6, 6, 6), spacing=(1.0, 2.0, 3.0))
    vol = torch.arange(6 * 6 * 6, dtype=torch.float64).reshape(6, 6, 6)
    for axis in range(3):
        d = forward_diff(vol, axis, g.spacing)
        last = [slice(None)] * 3
        last[axis] = slice(5, 6)
        assert float(d[tuple(last)].abs().max()) == 0.0


def test_grid_rejects_bad_inputs():
    with pytest.raises(ValueError):
        Grid(shape=(1, 4, 4), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        Grid(shape=(4, 4, 4), spacing=(1.0, 0.0, 1.0))
