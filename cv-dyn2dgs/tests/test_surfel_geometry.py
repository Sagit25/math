"""Normal projection, tangent transport, mesh extraction and density control.

These cover the paper's core geometric machinery: Eq. (6.2)-(6.3), Eq. (7.3)-(7.7), and
the surface extraction the mesh baseline depends on.
"""

from __future__ import annotations

import math

import pytest
import torch

from cvdyn2dgs.core.config import SurfelConfig
from cvdyn2dgs.core.grid import Grid
from cvdyn2dgs.data.phantom import PhantomConfig, ellipsoid_sdf, make_phantom
from cvdyn2dgs.levelset.mesh_extract import marching_tetrahedra
from cvdyn2dgs.surfel.canonical import initialize_canonical_surfels, sample_mesh_surface
from cvdyn2dgs.surfel.density import knn, nearest_neighbour_stats, prune
from cvdyn2dgs.surfel.projection import project_to_surface, surface_seed_points
from cvdyn2dgs.surfel.transport import (
    initial_tangent_frame,
    minimal_rotation_matrix,
    transport_tangent_frame,
)


def _sphere(radius=14.0, n=49, h=1.0, dtype=torch.float32):
    g = Grid(shape=(n, n, n), spacing=(h, h, h))
    world = g.world_meshgrid(dtype=dtype)
    centre = g.center_world(dtype=dtype).view(3, 1, 1, 1)
    phi = radius - (world - centre).norm(dim=0)
    return g, phi, g.center_world(dtype=dtype), radius


# --------------------------------------------------------------------------- #
#  Projection (Eq. 6.2 - 6.3)
# --------------------------------------------------------------------------- #
def test_projection_lands_on_the_zero_level_set():
    g, phi, centre, radius = _sphere()
    torch.manual_seed(0)
    dirs = torch.randn(400, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    # Start 2.5 mm off the surface, on both sides.
    offs = torch.where(torch.rand(400, 1) > 0.5, 2.5, -2.5)
    start = centre + dirs * (radius + offs)

    res = project_to_surface(start, phi, g, iters=3, eps=0.0, max_step_mm=10.0)
    assert float(res.residual_mm.mean()) < 0.05
    # Anchors end up at the sphere radius.
    r = (res.anchor - centre).norm(dim=-1)
    assert float((r - radius).abs().mean()) < 0.1


def test_projection_residual_history_decreases():
    g, phi, centre, radius = _sphere()
    torch.manual_seed(1)
    dirs = torch.randn(200, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    start = centre + dirs * (radius + 3.0)
    res = project_to_surface(
        start, phi, g, iters=4, eps=0.0, max_step_mm=10.0, rollback_on_increase=False
    )
    hist = res.residual_history
    assert len(hist) == 5
    assert hist[1] < hist[0]
    assert hist[-1] <= hist[1]


def test_projection_trust_region_caps_the_step():
    g, phi, centre, radius = _sphere()
    start = centre.view(1, 3) + torch.tensor([[0.0, 0.0, 40.0]])
    res = project_to_surface(start, phi, g, iters=1, eps=0.0, max_step_mm=1.0)
    assert float(res.total_step_mm.max()) <= 1.0 + 1e-5
    assert res.clipped_fraction > 0.0


def test_closest_point_and_normal_projection_agree_near_an_sdf():
    """Prop. 6.2's premise: near an SDF the two projections coincide."""
    g, phi, centre, radius = _sphere()
    torch.manual_seed(2)
    dirs = torch.randn(150, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    start = centre + dirs * (radius + 1.5)

    normal = project_to_surface(start, phi, g, iters=3, eps=0.0)
    pts = surface_seed_points(phi, g, max_points=20000)
    closest = project_to_surface(start, phi, g, mode="closest_point", surface_points=pts)

    gap = (normal.anchor - closest.anchor).norm(dim=-1)
    # Agreement is limited by the voxel resolution of the seed point cloud.
    assert float(gap.mean()) < g.h_min


def test_surface_seed_points_lie_on_the_surface():
    g, phi, centre, radius = _sphere()
    pts = surface_seed_points(phi, g)
    assert pts.shape[0] > 100
    r = (pts - centre).norm(dim=-1)
    assert float((r - radius).abs().max()) < 1.2 * g.h_min


# --------------------------------------------------------------------------- #
#  Transport (Eq. 7.6 - 7.7)
# --------------------------------------------------------------------------- #
def test_initial_frame_is_right_handed_orthonormal():
    torch.manual_seed(3)
    n = torch.randn(500, 3, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)
    e1, e2 = initial_tangent_frame(n, eps=0.0)

    assert float((e1.norm(dim=-1) - 1).abs().max()) < 1e-12
    assert float((e2.norm(dim=-1) - 1).abs().max()) < 1e-12
    assert float((e1 * e2).sum(-1).abs().max()) < 1e-12
    assert float((e1 * n).sum(-1).abs().max()) < 1e-12
    assert float((e2 * n).sum(-1).abs().max()) < 1e-12
    # Right-handed: e1 x e2 = n
    assert float((torch.cross(e1, e2, dim=-1) - n).abs().max()) < 1e-12


@pytest.mark.parametrize("mode", ["gram_schmidt", "rodrigues"])
def test_transport_preserves_orthonormality(mode):
    torch.manual_seed(4)
    n_old = torch.randn(300, 3, dtype=torch.float64)
    n_old = n_old / n_old.norm(dim=-1, keepdim=True)
    e1, e2 = initial_tangent_frame(n_old, eps=0.0)
    # Small perturbation, as between adjacent cine frames.
    n_new = n_old + 0.05 * torch.randn(300, 3, dtype=torch.float64)
    n_new = n_new / n_new.norm(dim=-1, keepdim=True)

    e1n, e2n, diag = transport_tangent_frame(
        e1, e2, n_new, normal_prev=n_old, eps=0.0, mode=mode
    )
    assert float((e1n * n_new).sum(-1).abs().max()) < 1e-10
    assert float((e2n * n_new).sum(-1).abs().max()) < 1e-10
    assert float((e1n * e2n).sum(-1).abs().max()) < 1e-10
    assert float((e1n.norm(dim=-1) - 1).abs().max()) < 1e-10
    # Small normal change => small frame rotation.
    assert float(diag.rotation_angle_rad.max()) < 0.3


def test_transport_falls_back_when_degenerate():
    """Prop. 7.7: when e1 is nearly parallel to the new normal, use e2 instead."""
    n_new = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    e1 = torch.tensor([[0.0, 0.01, 0.99995]], dtype=torch.float64)
    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    e2 = torch.cross(n_new, e1, dim=-1)
    e2 = e2 / e2.norm(dim=-1, keepdim=True)

    e1n, e2n, diag = transport_tangent_frame(
        e1, e2, n_new, eps=0.0, degeneracy_thresh=0.2
    )
    assert bool(diag.fallback_mask.all())
    assert float(diag.sin_psi.item()) < 0.2
    assert float((e1n * n_new).sum(-1).abs().max()) < 1e-10
    assert float((e1n.norm(dim=-1) - 1).abs().max()) < 1e-10


def test_minimal_rotation_maps_normals():
    torch.manual_seed(5)
    a = torch.randn(200, 3, dtype=torch.float64)
    a = a / a.norm(dim=-1, keepdim=True)
    b = torch.randn(200, 3, dtype=torch.float64)
    b = b / b.norm(dim=-1, keepdim=True)
    rot = minimal_rotation_matrix(a, b)
    mapped = (rot @ a.unsqueeze(-1)).squeeze(-1)
    assert float((mapped - b).abs().max()) < 1e-9
    # Orthogonal with determinant +1
    eye = torch.eye(3, dtype=torch.float64).expand(200, 3, 3)
    assert float((rot @ rot.transpose(-1, -2) - eye).abs().max()) < 1e-9
    assert float((torch.linalg.det(rot) - 1.0).abs().max()) < 1e-9


# --------------------------------------------------------------------------- #
#  Mesh extraction
# --------------------------------------------------------------------------- #
def test_marching_tetrahedra_on_a_sphere():
    g, phi, centre, radius = _sphere(radius=14.0, n=45, h=1.0, dtype=torch.float32)
    mesh = marching_tetrahedra(phi, g)

    assert mesh.n_faces > 500
    r = (mesh.vertices - centre).norm(dim=-1)
    # Linear edge interpolation puts vertices within a fraction of a voxel.
    assert float((r - radius).abs().max()) < 0.6
    # Surface area of a sphere, within the discretisation error of a tet mesh.
    area = float(mesh.surface_area())
    assert area == pytest.approx(4 * math.pi * radius**2, rel=0.1)
    # Normals point inward (phi > 0 inside), i.e. against the radial direction.
    radial = (mesh.vertices - centre) / r.clamp_min(1e-9).unsqueeze(-1)
    assert float((mesh.normals * radial).sum(-1).mean()) < -0.95


def test_marching_tetrahedra_handles_an_empty_level_set():
    g = Grid(shape=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    phi = torch.full((8, 8, 8), -3.0)
    mesh = marching_tetrahedra(phi, g)
    assert mesh.n_faces == 0
    assert mesh.n_vertices == 0


def test_area_weighted_sampling_is_uniform_on_the_surface():
    g, phi, centre, radius = _sphere(radius=14.0, n=45)
    mesh = marching_tetrahedra(phi, g)
    gen = torch.Generator().manual_seed(7)
    pts, area = sample_mesh_surface(mesh, 4000, generator=gen)

    r = (pts - centre).norm(dim=-1)
    assert float((r - radius).abs().max()) < 0.8
    # Uniform on a sphere => the mean position is near the centre.
    assert float((pts.mean(dim=0) - centre).norm()) < 0.06 * radius
    assert float(area) == pytest.approx(4 * math.pi * radius**2, rel=0.1)


# --------------------------------------------------------------------------- #
#  Canonical initialisation and density
# --------------------------------------------------------------------------- #
def test_canonical_initialisation_is_consistent():
    ph = make_phantom(PhantomConfig(shape=(48, 48, 12), n_frames=2, noise_sigma=0.02))
    cfg = SurfelConfig(n_surfels=1500, isotropic=True)
    gen = torch.Generator().manual_seed(11)
    surfels, stats = initialize_canonical_surfels(
        ph.phi_gt[0], ph.images[0], ph.grid, cfg, generator=gen
    )

    assert surfels.n == 1500
    assert stats["init_e_surf_mm"] < 0.3
    err = surfels.frame_orthonormality_error()
    for key in ("e1_norm_err", "e2_norm_err", "e1_e2_dot", "e1_n_dot", "e2_n_dot"):
        assert float(err[key].max()) < 1e-4, key
    assert bool((surfels.scale > 0).all())
    # Isotropic request must be honoured.
    assert float((surfels.scale[:, 0] - surfels.scale[:, 1]).abs().max()) < 1e-6


def test_knn_excludes_self_and_is_sorted():
    torch.manual_seed(13)
    pts = torch.randn(200, 3)
    d, idx = knn(pts, k=3)
    assert d.shape == (200, 3)
    assert bool((idx != torch.arange(200).unsqueeze(1)).all())
    assert bool((d[:, 1:] >= d[:, :-1]).all())


def test_nearest_neighbour_stats_detect_clustering():
    uniform = torch.rand(500, 3) * 20.0
    clustered = torch.cat([torch.rand(250, 3) * 0.5, torch.rand(250, 3) * 20.0])
    su = nearest_neighbour_stats(uniform)
    sc = nearest_neighbour_stats(clustered)
    assert sc.clustering_ratio < su.clustering_ratio


def test_prune_removes_transparent_surfels():
    from cvdyn2dgs.surfel.model import SurfelSet2D

    n = 60
    normal = torch.zeros(n, 3)
    normal[:, 2] = 1.0
    e1, e2 = initial_tangent_frame(normal)
    opa = torch.full((n,), 0.5)
    opa[:10] = 1e-4
    s = SurfelSet2D(
        torch.randn(n, 3) * 30.0, e1, e2, normal,
        torch.full((n, 2), 0.5), torch.rand(n, 1), opa,
    )
    cfg = SurfelConfig(prune_opacity_thresh=0.01, prune_overlap_thresh=0.0)
    out, stats = prune(s, cfg)
    assert stats["pruned"] == 10.0
    assert out.n == n - 10


def test_ellipsoid_sdf_matches_a_sphere_exactly():
    """The phantom's exact SDF must reduce to the analytic sphere distance."""
    axes = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
    pts = torch.randn(2000, 3, dtype=torch.float64) * 8.0
    got = ellipsoid_sdf(pts, axes)
    want = 10.0 - pts.norm(dim=-1)  # positive inside
    assert float((got - want).abs().max()) < 1e-8
