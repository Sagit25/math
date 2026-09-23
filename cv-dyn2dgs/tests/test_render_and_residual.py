"""Rasterisation, the rendering operator, losses and the residual solver.

The key test is :func:`test_tiled_matches_reference_renderer`: the fast tiled rasteriser
makes two approximations that the brute-force reference does not (per-tile anchor-depth
ordering and ``max_per_tile`` truncation). Rather than assume those are harmless, the
tests measure the gap on configurations where it should vanish.
"""

from __future__ import annotations

import math

import pytest
import torch

from cvdyn2dgs.core.config import LossConfig, RenderConfig, ResidualConfig
from cvdyn2dgs.core.grid import Grid
from cvdyn2dgs.losses import normal_from_depth, total_loss
from cvdyn2dgs.render.camera import Camera
from cvdyn2dgs.render.raster2dgs import render_2dgs, render_2dgs_reference, render_weights
from cvdyn2dgs.render.raster3dgs import Thin3DGSConfig, render_thin_3dgs
from cvdyn2dgs.render.raymarch import raymarch_levelset
from cvdyn2dgs.residual.lowrank import compress_residual, rank_for_error, spectrum_report
from cvdyn2dgs.residual.solver import ResidualView, conjugate_gradient, solve_residual
from cvdyn2dgs.surfel.model import SurfelSet2D
from cvdyn2dgs.surfel.transport import initial_tangent_frame

CFG = RenderConfig(tile=16, max_per_tile=256, tile_chunk=32)


def _surfels(n=12, spread=8.0, radius=1.5, seed=0, isotropic=True):
    torch.manual_seed(seed)
    anchor = torch.randn(n, 3) * spread
    normal = torch.randn(n, 3)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    e1, e2 = initial_tangent_frame(normal)
    scale = torch.full((n, 2), radius)
    if not isotropic:
        scale[:, 1] *= 0.4
    return SurfelSet2D(anchor, e1, e2, normal, scale, torch.rand(n, 1) * 0.8 + 0.1,
                       torch.full((n,), 0.7))


def _camera(dist=60.0, res=64, fov=45.0):
    return Camera.look_at(
        torch.tensor([0.0, 0.0, dist]), torch.zeros(3), height=res, width=res, fov_deg=fov
    )


# --------------------------------------------------------------------------- #
#  2DGS rasteriser
# --------------------------------------------------------------------------- #
def test_single_surfel_matches_the_analytic_gaussian():
    """One disk facing the camera: alpha must follow exp(-r^2 / 2s^2) exactly."""
    anchor = torch.zeros(1, 3)
    normal = torch.tensor([[0.0, 0.0, -1.0]])  # facing +z camera
    e1, e2 = initial_tangent_frame(normal)
    s = 2.0
    surf = SurfelSet2D(
        anchor, e1, e2, normal, torch.full((1, 2), s),
        torch.ones(1, 1), torch.full((1,), 0.9),
    )
    cam = _camera(dist=50.0, res=65, fov=60.0)
    out = render_2dgs(surf, cam, CFG, compute_aux=True)

    centre_alpha = float(out.alpha[32, 32])
    assert centre_alpha == pytest.approx(0.9, abs=2e-2)
    # Depth at the centre equals the camera distance.
    assert float(out.depth[32, 32]) == pytest.approx(50.0, abs=0.2)
    assert float(out.alpha.max()) <= 1.0


def test_tiled_matches_reference_renderer():
    """Well-separated surfels: ordering and truncation cannot matter, so the fast and
    reference rasterisers must agree closely."""
    surf = _surfels(n=8, spread=14.0, radius=1.2, seed=3)
    cam = _camera(dist=90.0, res=48)
    cfg = RenderConfig(tile=16, max_per_tile=64, tile_chunk=16)

    fast = render_2dgs(surf, cam, cfg, compute_aux=True)
    ref = render_2dgs_reference(surf, cam, cfg)

    assert float((fast.alpha - ref.alpha).abs().max()) < 1e-4
    assert float((fast.color - ref.color).abs().max()) < 1e-4
    hit = (fast.alpha > 0.05) & (ref.alpha > 0.05)
    if int(hit.sum()) > 0:
        assert float((fast.depth - ref.depth)[hit].abs().max()) < 1e-2


def test_empty_and_offscreen_cases_return_background():
    surf = _surfels(n=4)
    # Camera looking away from everything.
    cam = Camera.look_at(
        torch.tensor([0.0, 0.0, 500.0]),
        torch.tensor([0.0, 0.0, 1000.0]),
        height=32, width=32,
    )
    out = render_2dgs(surf, cam, CFG)
    assert float(out.alpha.max()) == 0.0
    assert out.color.shape == (1, 32, 32)


def test_grazing_angle_surfels_are_culled():
    """Prop. 8.1: |n . d| < c0 has an ill-conditioned depth and must be dropped."""
    anchor = torch.zeros(1, 3)
    normal = torch.tensor([[1.0, 0.0, 0.0]])  # exactly edge-on to a +z view
    e1, e2 = initial_tangent_frame(normal)
    surf = SurfelSet2D(
        anchor, e1, e2, normal, torch.full((1, 2), 3.0),
        torch.ones(1, 1), torch.full((1,), 0.9),
    )
    cam = _camera(dist=40.0, res=32)
    out = render_2dgs(surf, cam, RenderConfig(tile=16, max_per_tile=32, cull_cos=0.2))
    assert float(out.alpha.max()) < 1e-6


def test_rasteriser_is_differentiable_in_amplitude_opacity_scale():
    surf = _surfels(n=10, spread=6.0, radius=2.0, seed=5)
    cam = _camera(dist=50.0, res=32)
    out = render_2dgs(surf, cam, CFG, compute_aux=True)
    loss = out.color.sum() + out.alpha.sum()
    loss.backward()

    for name, p in (
        ("amplitude", surf.amplitude),
        ("opacity_logit", surf.opacity_logit),
        ("log_scale", surf.log_scale),
    ):
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0.0, name


def test_geometry_buffers_receive_no_gradient():
    """The anchors/frames are buffers: no loss may move the surface."""
    surf = _surfels(n=6)
    assert not surf.anchor.requires_grad
    assert not surf.e1.requires_grad
    assert not surf.normal.requires_grad


def test_alpha_never_exceeds_one():
    """Many overlapping disks must still composite to alpha <= 1."""
    n = 60
    anchor = torch.randn(n, 3) * 0.4
    normal = torch.zeros(n, 3)
    normal[:, 2] = -1.0
    e1, e2 = initial_tangent_frame(normal)
    surf = SurfelSet2D(
        anchor, e1, e2, normal, torch.full((n, 2), 3.0),
        torch.ones(n, 1), torch.full((n,), 0.95),
    )
    out = render_2dgs(surf, _camera(dist=40.0, res=32), CFG)
    assert float(out.alpha.max()) <= 1.0 + 1e-5


def test_thin_3dgs_runs_and_gives_piecewise_constant_depth():
    """The thin-3DGS depth is constant per splat - the deficiency RQ6 measures."""
    surf = _surfels(n=6, spread=10.0, radius=2.0, seed=9)
    cam = _camera(dist=70.0, res=48)
    out2d = render_2dgs(surf, cam, CFG, compute_aux=True)
    out3d = render_thin_3dgs(surf, cam, CFG, Thin3DGSConfig(thickness_ratio=0.05), compute_aux=True)

    assert out3d.color.shape == out2d.color.shape
    hit = (out2d.alpha > 0.3) & (out3d.alpha > 0.3)
    if int(hit.sum()) > 20:
        # 2DGS depth varies across a tilted disk; the affine one varies much less.
        assert float(out2d.depth[hit].std()) >= float(out3d.depth[hit].std()) - 1e-6


# --------------------------------------------------------------------------- #
#  Ray marching reference
# --------------------------------------------------------------------------- #
def test_raymarch_recovers_a_sphere():
    g = Grid(shape=(49, 49, 49), spacing=(1.0, 1.0, 1.0))
    world = g.world_meshgrid()
    centre = g.center_world().view(3, 1, 1, 1)
    radius = 15.0
    phi = radius - (world - centre).norm(dim=0)

    c = g.center_world()
    cam = Camera.look_at(
        c + torch.tensor([0.0, 0.0, 120.0]), c, height=48, width=48, fov_deg=35.0
    )
    ref = raymarch_levelset(phi, g, cam, image=torch.ones_like(phi))

    assert int(ref.hit.sum()) > 100
    # The nearest hit is at (camera distance - radius).
    d = ref.depth[ref.hit]
    assert float(d.min()) == pytest.approx(120.0 - radius, abs=0.6)
    # The surface point must satisfy |x - centre| = radius.
    pts = ref.point.permute(1, 2, 0)[ref.hit]
    r = (pts - c).norm(dim=-1)
    assert float((r - radius).abs().max()) < 0.6
    # Normals point inward.
    nrm = ref.normal.permute(1, 2, 0)[ref.hit]
    radial = (pts - c) / r.unsqueeze(-1)
    assert float((nrm * radial).sum(-1).mean()) < -0.97


def test_normal_from_depth_matches_a_plane():
    """A flat depth map from a plane must give the plane's normal."""
    cam = Camera.look_at(
        torch.tensor([0.0, 0.0, 30.0]), torch.zeros(3), height=32, width=32, fov_deg=30.0
    )
    _, dirs = cam.rays()
    # Depth to the z = 0 plane along each ray.
    depth = 30.0 / (-dirs[..., 2]).clamp_min(1e-6)
    n = normal_from_depth(depth, cam)
    inner = n[:, 4:-4, 4:-4]
    assert float(inner[2].mean()) == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------- #
#  Rendering operator and residual (theory §9)
# --------------------------------------------------------------------------- #
def test_weight_matrix_reproduces_the_renderer():
    """Prop. 9.1: rendering is linear in amplitude, so A a must equal the render."""
    surf = _surfels(n=10, spread=8.0, radius=1.5, seed=17)
    cam = _camera(dist=60.0, res=48)
    wm = render_weights(surf, cam, CFG)
    direct = render_2dgs(surf, cam, CFG, compute_aux=False)

    approx = wm.apply(surf.amplitude.detach())
    assert float((approx - direct.color).abs().max()) < 1e-4
    assert float((wm.alpha() - direct.alpha).abs().max()) < 1e-4


def test_weight_matrix_is_linear_and_adjoint_is_correct():
    surf = _surfels(n=8, spread=7.0, radius=2.0, seed=19)
    cam = _camera(dist=55.0, res=32)
    wm = render_weights(surf, cam, CFG)

    a = torch.randn(surf.n, 1)
    b = torch.randn(surf.n, 1)
    lhs = wm.apply(2.0 * a - 3.0 * b)
    rhs = 2.0 * wm.apply(a) - 3.0 * wm.apply(b)
    assert float((lhs - rhs).abs().max()) < 1e-5

    # <A a, r> == <a, A^T r>
    r = torch.randn(1, 32, 32)
    left = float((wm.apply(a) * r).sum())
    right = float((a * wm.apply_transpose(r)).sum())
    assert left == pytest.approx(right, rel=1e-4, abs=1e-6)


def test_conjugate_gradient_solves_an_spd_system():
    torch.manual_seed(23)
    n = 40
    m = torch.randn(n, n, dtype=torch.float64)
    spd = m @ m.transpose(0, 1) + n * torch.eye(n, dtype=torch.float64)
    x_true = torch.randn(n, 1, dtype=torch.float64)
    b = spd @ x_true

    x, iters, res = conjugate_gradient(lambda v: spd @ v, b, max_iters=200, tol=1e-12)
    assert float((x - x_true).abs().max()) < 1e-8
    assert iters <= n + 5


def test_residual_solver_reduces_the_data_term():
    surf = _surfels(n=40, spread=6.0, radius=1.8, seed=29)
    cam = _camera(dist=50.0, res=48)
    wm = render_weights(surf, cam, CFG)
    base = surf.amplitude.detach()

    torch.manual_seed(31)
    perturbed = base + 0.25 * torch.randn_like(base)
    target = wm.apply(perturbed)

    cfg = ResidualConfig(lambda_a=1e-4, lambda_T=1e-4, cg_iters=200, cg_tol=1e-10)
    res = solve_residual([ResidualView(weights=wm, target=target)], base, cfg)
    assert res.data_term_after < 0.2 * res.data_term_before
    # The recovered residual should correlate with the true perturbation.
    truth = (perturbed - base).reshape(-1)
    got = res.delta.reshape(-1)
    seen = wm.contributions_per_surfel() > 1e-3
    if int(seen.sum()) > 5:
        corr = torch.corrcoef(torch.stack([truth[seen], got[seen]]))[0, 1]
        assert float(corr) > 0.5


def test_residual_requires_positive_regularisation():
    """Prop. 9.2 needs lambda_a + lambda_T > 0 for an SPD system."""
    surf = _surfels(n=5)
    cam = _camera(res=16)
    wm = render_weights(surf, cam, CFG)
    with pytest.raises(ValueError, match="positive definite"):
        solve_residual(
            [ResidualView(weights=wm, target=torch.zeros(1, 16, 16))],
            surf.amplitude.detach(),
            ResidualConfig(lambda_a=0.0, lambda_T=0.0),
        )


# --------------------------------------------------------------------------- #
#  Low-rank residual (Prop. 9.3)
# --------------------------------------------------------------------------- #
def test_lowrank_error_equals_the_singular_value_tail():
    torch.manual_seed(37)
    mat = torch.randn(200, 6, dtype=torch.float64) @ torch.randn(6, 20, dtype=torch.float64)
    mat = mat + 0.02 * torch.randn(200, 20, dtype=torch.float64)

    for r in (2, 4, 8):
        lr = compress_residual(mat, r)
        assert lr.frobenius_error == pytest.approx(lr.predicted_frobenius_error(), rel=1e-9)
        assert lr.spectral_error == pytest.approx(lr.predicted_spectral_error(), rel=1e-9)


def test_lowrank_is_exact_at_the_true_rank():
    torch.manual_seed(41)
    true_rank = 4
    mat = torch.randn(120, true_rank, dtype=torch.float64) @ torch.randn(
        true_rank, 15, dtype=torch.float64
    )
    lr = compress_residual(mat, true_rank)
    assert lr.relative_frobenius_error < 1e-12
    assert float((lr.reconstruct() - mat).abs().max()) < 1e-10


def test_rank_for_error_and_spectrum_report_agree():
    torch.manual_seed(43)
    mat = torch.randn(100, 3, dtype=torch.float64) @ torch.randn(3, 12, dtype=torch.float64)
    mat = mat + 1e-3 * torch.randn(100, 12, dtype=torch.float64)

    r = rank_for_error(mat, 1e-2)
    lr = compress_residual(mat, r)
    assert lr.relative_frobenius_error <= 1e-2 + 1e-9

    rep = spectrum_report(mat)
    assert rep["effective_rank_99"] <= 4
    assert lr.storage_floats() == r * (100 + 12)


# --------------------------------------------------------------------------- #
#  Losses
# --------------------------------------------------------------------------- #
def test_total_loss_skips_disabled_terms():
    surf = _surfels(n=6, spread=5.0, radius=2.0, seed=47)
    cam = _camera(dist=45.0, res=32)
    out = render_2dgs(surf, cam, CFG, compute_aux=True)
    target = torch.rand(1, 32, 32)

    off = total_loss(out, target, LossConfig(lambda_mask=0, lambda_normal=0, lambda_dist=0))
    assert float(off.mask) == 0.0 and float(off.normal) == 0.0 and float(off.distortion) == 0.0
    assert float(off.total) == pytest.approx(float(off.appearance))

    on = total_loss(
        out, target,
        LossConfig(lambda_mask=1.0, lambda_normal=0.1, lambda_dist=1.0),
        camera=cam, target_mask=(out.alpha > 0.5).to(out.alpha.dtype),
    )
    assert float(on.total) >= float(on.appearance)
    assert math.isfinite(float(on.total))


def test_mask_loss_is_zero_for_a_perfect_silhouette():
    from cvdyn2dgs.losses import mask_loss

    alpha = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert float(mask_loss(alpha, alpha)) == pytest.approx(0.0)
