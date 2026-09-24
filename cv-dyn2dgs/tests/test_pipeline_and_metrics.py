"""End-to-end smoke tests, storage round-trips, metrics and the phantom's ground truth.

The pipeline tests deliberately use a tiny phantom so they run in seconds on CPU. They
check that the machinery is *wired correctly* - the sign convention holds, the level set
tracks the moving surface, storage round-trips, playback does not re-solve anything - not
that the quality is good. Quality lives in :mod:`cvdyn2dgs.experiments`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from cvdyn2dgs.core.config import ChanVeseConfig, get_preset
from cvdyn2dgs.core.grid import Grid
from cvdyn2dgs.data.phantom import PhantomConfig, contraction_profile, make_phantom
from cvdyn2dgs.levelset.chanvese import solve_frame, solve_sequence
from cvdyn2dgs.levelset.operators import gradient_central, gradient_norm
from cvdyn2dgs.levelset.sdf import (
    interpolate_levelsets,
    predicted_eikonal_norm,
    reinitialize,
    signed_distance_from_mask,
)
from cvdyn2dgs.metrics.clinical import ejection_fraction, volume_curve
from cvdyn2dgs.metrics.photometric import psnr, ssim, temporal_report
from cvdyn2dgs.metrics.rendering import boundary_f_score, coverage_report, silhouette_iou
from cvdyn2dgs.metrics.segmentation import dice, surface_distance_report
from cvdyn2dgs.metrics.storage import storage_report

TINY = PhantomConfig(shape=(40, 40, 10), spacing=(1.5, 1.5, 6.0), n_frames=5, n_papillary=1)


# --------------------------------------------------------------------------- #
#  Phantom ground truth
# --------------------------------------------------------------------------- #
def test_phantom_ground_truth_is_self_consistent():
    ph = make_phantom(TINY)
    assert ph.n_frames == 5
    for phi, mask in zip(ph.phi_gt, ph.masks):
        # Sign convention: phi > 0 exactly where the mask is true.
        assert bool(((phi > 0) == mask).all())

    # Exact SDF satisfies the eikonal equation away from the grid border.
    g = gradient_norm(gradient_central(ph.phi_gt[0], ph.grid.spacing))
    band = ph.phi_gt[0].abs() < 4.0
    assert float((g[band] - 1.0).abs().mean()) < 0.1


def test_phantom_contracts_and_has_a_plausible_ef():
    ph = make_phantom(TINY)
    vols = ph.analytic_volumes_ml()
    assert vols[ph.ed_index()] > vols[ph.es_index()]
    ef = ph.ef_percent()
    assert 20.0 < ef < 85.0, f"implausible EF {ef}"
    # Voxel-counted volumes track the analytic ones.
    for a, b in zip(ph.volumes_ml(), vols):
        assert a == pytest.approx(b, rel=0.2)


def test_contraction_profile_endpoints():
    assert contraction_profile(0, 20) == pytest.approx(0.0)
    es = int(round(0.35 * 20))
    assert contraction_profile(es, 20) > 0.95
    assert contraction_profile(20, 20) == pytest.approx(0.0)


def test_papillary_muscle_breaks_intensity_homogeneity():
    """The papillary blobs are inside the mask but not at blood intensity - the
    limitation of proposal §2.5 / §10.3(1), present on purpose."""
    with_pap = make_phantom(PhantomConfig(**{**TINY.__dict__, "papillary": True, "noise_sigma": 0.0}))
    without = make_phantom(PhantomConfig(**{**TINY.__dict__, "papillary": False, "noise_sigma": 0.0}))
    inside = with_pap.masks[0]
    std_with = float(with_pap.clean[0][inside].std())
    std_without = float(without.clean[0][inside].std())
    assert std_with > std_without


# --------------------------------------------------------------------------- #
#  Level set
# --------------------------------------------------------------------------- #
def test_signed_distance_from_mask_has_the_right_sign():
    ph = make_phantom(TINY)
    phi = signed_distance_from_mask(ph.masks[0], ph.grid.spacing, max_dist_mm=10.0)
    # Interior voxels positive, exterior negative (allowing a one-voxel boundary band).
    deep_in = ph.phi_gt[0] > 2.0
    deep_out = ph.phi_gt[0] < -2.0
    assert bool((phi[deep_in] > 0).all())
    assert bool((phi[deep_out] < 0).all())


def test_reinitialisation_improves_the_eikonal_property():
    g = Grid(shape=(41, 41, 41), spacing=(1.0, 1.0, 1.0))
    world = g.world_meshgrid()
    centre = g.center_world().view(3, 1, 1, 1)
    r = (world - centre).norm(dim=0)
    # Badly scaled level set with the correct zero set.
    phi = 3.0 * (12.0 - r)

    band = (12.0 - r).abs() < 4.0
    before = float((gradient_norm(gradient_central(phi, g.spacing))[band] - 1).abs().mean())
    after_phi = reinitialize(phi, g.spacing, iters=60, dt_scale=0.3)
    after = float((gradient_norm(gradient_central(after_phi, g.spacing))[band] - 1).abs().mean())
    assert after < before
    # And the zero level set has not moved much.
    assert float(((after_phi > 0) != (phi > 0)).to(torch.float32).mean()) < 0.02


def _track_dices(cfg_phantom: PhantomConfig) -> list[float]:
    ph = make_phantom(cfg_phantom)
    cfg = ChanVeseConfig(max_iters=120, check_every=5, narrow_band_mm=8.0)
    phi0 = signed_distance_from_mask(ph.masks[0], ph.grid.spacing, max_dist_mm=12.0)
    seq = solve_sequence(ph.images, phi0, ph.grid, cfg)
    assert len(seq.phis) == ph.n_frames
    return [dice(p > 0, m) for p, m in zip(seq.phis, ph.masks)]


def test_chanvese_tracks_the_moving_surface_when_intensity_is_piecewise_constant():
    """Tracking, measured where Chan-Vese's own assumption holds.

    The original version of this test ran on the phantom WITH a papillary muscle and
    demanded Dice > 0.75. It failed at 0.48 - and the failure was not a tracking bug. Dice
    was already 0.72 on frame 0, which is initialised from the ground-truth mask itself, so
    the surface was being lost by the segmentation rather than by the frame-to-frame update.
    The cause is that the phantom includes a papillary muscle *specifically to violate* the
    piecewise-constant intensity assumption Chan-Vese rests on (proposal §2.5, §10.3): the
    region term excludes the bright muscle from the interior, and the ground-truth cavity
    mask includes it - phantom.py's own docstring for the mask builder says
    "papillary muscle counted as blood pool", which is the whole mismatch.

    Raising the threshold would have hidden that; removing the confound measures the thing
    the test is named after. The limitation itself is measured by the test below.
    """
    clean = replace(TINY, papillary=False, n_papillary=0)
    dices = _track_dices(clean)
    assert min(dices) > 0.75, f"Chan-Vese lost a piecewise-constant surface: {dices}"


def test_papillary_muscle_degrades_chanvese_as_documented():
    """The documented limitation, measured rather than asserted.

    This is deliberately *not* a quality floor. It checks the direction and that frame 0
    already shows the effect - which is what distinguishes a violated intensity assumption
    from a failure of the temporal update. If this ever passes because both configurations
    score the same, the phantom has stopped exercising the limitation and §10.3's first
    limitation is no longer supported by anything.
    """
    clean = min(_track_dices(replace(TINY, papillary=False, n_papillary=0)))
    with_muscle = min(_track_dices(replace(TINY, papillary=True, n_papillary=1)))
    assert with_muscle < clean, (
        f"the papillary muscle did not degrade segmentation "
        f"(clean {clean:.3f} vs muscle {with_muscle:.3f}); the phantom is no longer "
        f"violating the piecewise-constant assumption it exists to violate"
    )


def test_warm_start_uses_no_more_iterations_than_cold():
    """RQ1's core claim, as a unit test on a tiny case."""
    ph = make_phantom(TINY)
    phi0 = signed_distance_from_mask(ph.masks[0], ph.grid.spacing, max_dist_mm=12.0)
    base = dict(max_iters=150, check_every=5)

    warm = solve_sequence(ph.images, phi0, ph.grid, ChanVeseConfig(**base))
    cold = solve_sequence(
        ph.images, phi0, ph.grid, ChanVeseConfig(**base, warm_start=False)
    )
    assert warm.total_iterations <= cold.total_iterations


def test_interpolation_violates_the_eikonal_property_as_predicted():
    """Prop. 10.1, on the phantom's exact SDFs."""
    ph = make_phantom(PhantomConfig(shape=(40, 40, 24), spacing=(1.5, 1.5, 1.5), n_frames=6, noise_sigma=0.0))
    _, _, diag = interpolate_levelsets(
        ph.phi_gt[0], ph.phi_gt[2], 0.5, ph.grid.spacing, diagnose=True
    )
    assert diag is not None
    band = (ph.phi_gt[0].abs() < 3.0) & (ph.phi_gt[2].abs() < 3.0)
    measured = diag["eikonal_norm_measured"][band]
    predicted = diag["eikonal_norm_predicted"][band]
    # The interpolated level set is not an SDF: the norm drops below 1.
    assert float(measured.mean()) < 1.0
    assert float((measured - predicted).abs().mean()) < 0.1


def test_predicted_eikonal_norm_endpoints():
    cos_w = torch.tensor([1.0, 0.0])
    # Aligned gradients => no violation at any beta.
    assert float(predicted_eikonal_norm(cos_w, 0.5)[0]) == pytest.approx(1.0)
    # beta = 0 or 1 => no interpolation, no violation.
    assert float(predicted_eikonal_norm(cos_w, 0.0)[1]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def test_dice_and_iou_edge_cases():
    a = torch.zeros(4, 4, dtype=torch.bool)
    assert dice(a, a) == 1.0  # both empty
    b = a.clone()
    b[0, 0] = True
    assert dice(b, b) == 1.0
    assert dice(a, b) == 0.0
    assert silhouette_iou(b.to(torch.float32), b) == 1.0


def test_surface_distance_is_zero_for_identical_surfaces():
    pts = torch.randn(200, 3)
    rep = surface_distance_report(pts, pts)
    assert rep.hd95_mm == pytest.approx(0.0)
    assert rep.assd_mm == pytest.approx(0.0)


def test_surface_distance_recovers_a_known_offset():
    pts = torch.randn(300, 3) * 5.0
    shifted = pts + torch.tensor([0.0, 0.0, 2.0])
    rep = surface_distance_report(pts, shifted)
    # Every point is exactly 2 mm from its counterpart (and no closer to others,
    # generically), so both directed distances concentrate near 2.
    assert rep.assd_mm == pytest.approx(2.0, abs=0.6)


def test_psnr_and_ssim_are_perfect_for_identical_images():
    img = torch.rand(1, 32, 32)
    assert math.isinf(psnr(img, img))
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-4)


def test_psnr_degrades_with_noise():
    img = torch.rand(1, 32, 32)
    noisy = (img + 0.1 * torch.randn_like(img)).clamp(0, 1)
    assert psnr(noisy, img) < 40.0
    assert ssim(noisy, img) < 1.0


def test_flicker_penalises_a_frozen_renderer():
    """E_flicker subtracts the reference change, so a static model is not rewarded."""
    ref = [torch.full((1, 8, 8), float(t)) for t in range(5)]
    frozen = [torch.zeros(1, 8, 8) for _ in range(5)]
    perfect = [r.clone() for r in ref]

    r_frozen = temporal_report(frozen, ref)
    r_perfect = temporal_report(perfect, ref)
    assert r_perfect["e_flicker"] == pytest.approx(0.0)
    assert r_frozen["e_flicker"] > 0.5
    assert r_frozen["rendered_change"] == pytest.approx(0.0)


def test_boundary_f_score_is_one_for_identical_masks():
    m = torch.zeros(32, 32, dtype=torch.bool)
    m[8:24, 8:24] = True
    out = boundary_f_score(m, m)
    assert out["boundary_f"] == pytest.approx(1.0, abs=1e-6)


def test_coverage_report_detects_holes():
    tgt = torch.ones(16, 16, dtype=torch.bool)
    alpha = torch.ones(16, 16)
    alpha[:4] = 0.0  # a quarter of the target uncovered
    cov = coverage_report(alpha, torch.ones(16, 16, dtype=torch.int32), tgt)
    assert cov.hole_fraction == pytest.approx(0.25, abs=1e-6)


def test_volume_curve_and_ef():
    g = Grid(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
    world = g.world_meshgrid()
    centre = g.center_world().view(3, 1, 1, 1)
    r = (world - centre).norm(dim=0)
    phis = [float(radius) - r for radius in (8.0, 6.0, 8.0)]

    vc = volume_curve(phis, g)
    assert vc.ed_index in (0, 2)
    assert vc.es_index == 1
    assert vc.ef_percent > 0
    assert ejection_fraction(100.0, 40.0) == pytest.approx(60.0)


def test_storage_report_compression_and_break_even():
    rep = storage_report(
        n_frames=20,
        n_surfels=20000,
        channels=1,
        surface_bytes_per_frame=[50_000] * 20,
        p2d=10,
        p_residual=1,
    )
    # S_full = 20 * 20000 * 10 * 4 bytes
    assert rep.s_full_bytes == 20 * 20000 * 10 * 4
    assert rep.s_ours_bytes == 20000 * 10 * 4 + 20 * 50_000 + 20 * 20000 * 1 * 4
    assert rep.compression_ratio == pytest.approx(rep.s_full_bytes / rep.s_ours_bytes)
    # The break-even budget must be consistent with CR = 1.
    be = rep.break_even_surface_bytes_per_frame
    assert be > 0
    recomputed = storage_report(
        n_frames=20, n_surfels=20000, channels=1,
        surface_bytes_per_frame=[be] * 20, p2d=10, p_residual=1,
    )
    assert recomputed.compression_ratio == pytest.approx(1.0, rel=1e-6)


def test_lowrank_storage_beats_dense_residual():
    dense = storage_report(
        n_frames=30, n_surfels=20000, channels=1,
        surface_bytes_per_frame=[10_000] * 30, p2d=10, p_residual=1,
    )
    low = storage_report(
        n_frames=30, n_surfels=20000, channels=1,
        surface_bytes_per_frame=[10_000] * 30, p2d=10, p_residual=1, lowrank_rank=8,
    )
    assert low.residual_bytes < dense.residual_bytes
    assert low.compression_ratio > dense.compression_ratio


# --------------------------------------------------------------------------- #
#  Storage round-trip
# --------------------------------------------------------------------------- #
def test_narrow_band_pack_roundtrip_is_exact_in_band():
    from cvdyn2dgs.pipeline.storage_io import pack_narrow_band, unpack_narrow_band

    ph = make_phantom(TINY)
    phi = ph.phi_gt[0]
    band_mm = 4.0
    packed = pack_narrow_band(phi, band_mm)
    back = unpack_narrow_band(packed, ph.grid, dtype=phi.dtype)

    band = phi.abs() < band_mm
    assert float((back[band] - phi[band]).abs().max()) < 1e-6
    # The sign is exact everywhere, which is what masks and volumes depend on.
    assert bool(((back > 0) == (phi > 0)).all())


# --------------------------------------------------------------------------- #
#  End-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset", ["v1-minimal", "v3-adaptive"])
def test_pipeline_end_to_end_smoke(preset):
    from cvdyn2dgs.experiments.common import make_eval_cameras, run_pipeline_on_phantom
    from cvdyn2dgs.pipeline.playback import measure_playback
    from cvdyn2dgs.pipeline.storage_io import model_storage_report

    ph = make_phantom(TINY)
    cfg = get_preset(preset)
    cfg.surfel.n_surfels = 800
    cfg.fit.iters = 12
    cfg.chanvese.max_iters = 40
    cfg.residual.cg_iters = 8

    cams = make_eval_cameras(ph.grid, n_orbit=2, resolution=48)
    gen = torch.Generator().manual_seed(0)
    model, cams, _ = run_pipeline_on_phantom(ph, cfg, cameras=cams, generator=gen)

    assert model.n_frames == ph.n_frames
    assert len(model.residuals) == ph.n_frames
    assert model.surfels.n > 0
    # Anchors stay on the level set at every frame.
    assert max(f.e_surf_mm for f in model.frames) < 1.0

    rep = model_storage_report(model)
    assert rep["p2d_minimal"] < rep["p2d_explicit"]
    assert "narrow_band_sdf" in rep["by_surface_mode"]

    pb = measure_playback(model, cams.eval, loops=1, warmup=1, compute_aux=False)
    assert len(pb.per_frame) > 0
    assert all(math.isfinite(f.total_ms) for f in pb.per_frame)


def test_playback_does_not_mutate_the_canonical_surfels():
    """Playback must be repeatable: the stored canonical set is read-only."""
    from cvdyn2dgs.experiments.common import make_eval_cameras, run_pipeline_on_phantom
    from cvdyn2dgs.pipeline.playback import PlaybackEngine

    ph = make_phantom(TINY)
    cfg = get_preset("v1-minimal")
    cfg.surfel.n_surfels = 400
    cfg.fit.iters = 5
    cfg.chanvese.max_iters = 25
    cams = make_eval_cameras(ph.grid, n_orbit=1, resolution=32)
    model, cams, _ = run_pipeline_on_phantom(ph, cfg, cameras=cams)

    before = model.surfels.anchor.clone()
    engine = PlaybackEngine(model)
    for t in range(model.n_frames):
        engine.step_to(t, cams.eval[0], compute_aux=False)
    assert torch.equal(before, model.surfels.anchor)


def test_chanvese_frame_solver_leaves_the_input_untouched():
    ph = make_phantom(TINY)
    phi0 = signed_distance_from_mask(ph.masks[0], ph.grid.spacing, max_dist_mm=10.0)
    snapshot = phi0.clone()
    solve_frame(ph.images[0], phi0, ph.grid, ChanVeseConfig(max_iters=10))
    assert torch.equal(phi0, snapshot)
