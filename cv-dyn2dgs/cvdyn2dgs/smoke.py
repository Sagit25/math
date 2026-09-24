"""Staged smoke test — run this first, before anything else.

Purpose
-------
This code was written without the ability to execute it, so the first real run is
expected to surface bugs. A test suite that stops at the first failure would force a
slow one-bug-per-run loop. This module instead runs ~20 **independent** stages, catches
exceptions per stage, and prints a single table. One run tells you everything that is
broken and, because the stages are ordered by dependency, which failure is the root
cause.

Reading the output
------------------
Stages are ordered bottom-up: interpolation, then operators, then the level set, then
surfels, then rendering, then the full pipeline. A failure in an early stage explains
every later failure, so **fix the first FAIL and re-run** rather than chasing the last.

``SKIP`` means a prerequisite stage failed, so the stage was not attempted and its
status says nothing about it.

This is a wiring check, not a quality check: thresholds are loose and sizes are tiny.
Quality lives in :mod:`cvdyn2dgs.experiments`. The purely combinatorial kernels are
covered separately by ``scripts/verify_kernels.py``, which needs no PyTorch.

Usage
-----
    cvdyn2dgs smoke                # default: tiny, CPU-friendly, ~1-2 min
    cvdyn2dgs smoke --full         # adds the end-to-end pipeline stages
    python -m cvdyn2dgs.smoke      # equivalent
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Sequence

__all__ = ["StageResult", "run_smoke", "main"]


@dataclass
class StageResult:
    name: str
    status: str  # PASS | FAIL | SKIP
    ms: float = 0.0
    detail: str = ""
    error: str = ""
    requires: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


class SmokeFailure(AssertionError):
    """Raised by a stage to report a specific, diagnosable failure."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SmokeFailure(msg)


@dataclass
class Runner:
    verbose: bool = True
    results: dict[str, StageResult] = field(default_factory=dict)
    state: dict[str, object] = field(default_factory=dict)

    def stage(
        self, name: str, fn: Callable[[], str], *, requires: Sequence[str] = ()
    ) -> StageResult:
        missing = [r for r in requires if not self.results.get(r, StageResult(r, "FAIL")).ok]
        if missing:
            res = StageResult(name, "SKIP", detail=f"needs {', '.join(missing)}", requires=tuple(requires))
            self.results[name] = res
            if self.verbose:
                print(f"  SKIP  {name:<38} ({res.detail})")
            return res

        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            res = StageResult(name, "PASS", (time.perf_counter() - t0) * 1e3, detail)
        except SmokeFailure as exc:
            res = StageResult(name, "FAIL", (time.perf_counter() - t0) * 1e3, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - any error must be reported, not raised
            res = StageResult(
                name,
                "FAIL",
                (time.perf_counter() - t0) * 1e3,
                error=f"{type(exc).__name__}: {exc}",
            )
            res.detail = traceback.format_exc(limit=6)
        self.results[name] = res
        if self.verbose:
            mark = res.status
            line = f"  {mark}  {name:<38} {res.ms:8.1f} ms"
            if res.status == "PASS" and res.detail:
                line += f"  {res.detail}"
            print(line)
            if res.status == "FAIL":
                print(f"        -> {res.error}")
        return res


# --------------------------------------------------------------------------- #
#  Stages
# --------------------------------------------------------------------------- #
def run_smoke(*, full: bool = False, device: str = "auto", seed: int = 0, verbose: bool = True) -> list[StageResult]:
    """Run all smoke stages and return their results."""
    r = Runner(verbose=verbose)
    S = r.state

    # -- 0. environment ----------------------------------------------------
    def s_import() -> str:
        import torch

        from .core.runtime import resolve_device, seed_everything

        S["torch"] = torch
        S["gen"] = seed_everything(seed)
        dev = resolve_device(device)
        S["dev"] = dev
        S["dtype"] = torch.float32
        return f"torch {torch.__version__}, device {dev}"

    r.stage("0  import torch / resolve device", s_import)

    # -- 1. grid + interpolation -------------------------------------------
    def s_grid() -> str:
        import torch

        from .core.grid import Grid, trilinear_sample

        g = Grid(shape=(9, 11, 7), spacing=(1.25, 2.0, 5.0))
        world = g.world_meshgrid(dtype=torch.float64)
        vol = 3.0 * world[0] - 2.0 * world[1] + 0.5 * world[2] + 7.0
        pts = torch.rand(200, 3, dtype=torch.float64) * torch.tensor([8.0, 10.0, 6.0])
        got = trilinear_sample(vol, pts)
        w = g.voxel_to_world(pts)
        want = 3.0 * w[:, 0] - 2.0 * w[:, 1] + 0.5 * w[:, 2] + 7.0
        err = float((got - want).abs().max())
        _require(err < 1e-8, f"trilinear is not exact on an affine field: max err {err:.2e}")
        return f"affine exactness {err:.1e}"

    r.stage("1  grid + trilinear interpolation", s_grid, requires=["0  import torch / resolve device"])

    # -- 2. spacing-aware operators ----------------------------------------
    def s_ops() -> str:
        import torch

        from .core.grid import Grid
        from .levelset.operators import central_diff, curvature, gradient_central, gradient_norm

        g = Grid(shape=(24, 20, 12), spacing=(1.25, 1.25, 8.0))
        world = g.world_meshgrid(dtype=torch.float64)
        ramp = world[2]
        aware = float(central_diff(ramp, 2, g.spacing)[1:-1, 1:-1, 1:-1].mean())
        naive = float(central_diff(ramp, 2, g.spacing, spacing_aware=False)[1:-1, 1:-1, 1:-1].mean())
        _require(abs(aware - 1.0) < 1e-9, f"spacing-aware slope should be 1, got {aware}")
        _require(abs(naive - g.hz) < 1e-9, f"spacing-ignored slope should be hz={g.hz}, got {naive}")

        gs = Grid(shape=(51, 51, 51), spacing=(1.0, 1.0, 1.0))
        w2 = gs.world_meshgrid(dtype=torch.float64)
        c = gs.center_world(dtype=torch.float64).view(3, 1, 1, 1)
        rad = (w2 - c).norm(dim=0)
        radius = 16.0
        phi = radius - rad
        n = gradient_norm(gradient_central(phi, gs.spacing))
        band = (phi.abs() < 3.0) & (rad > 4.0)
        gerr = float((n[band] - 1.0).abs().mean())
        _require(gerr < 0.05, f"||grad phi|| should be ~1 for an SDF, mean error {gerr:.3f}")
        kap = float(curvature(phi, gs.spacing, eps=1e-12)[band].mean())
        _require(
            abs(kap + 2.0 / radius) < 0.05,
            f"sphere curvature should be -2/R = {-2/radius:.4f}, got {kap:.4f} "
            "(a sign flip here means the inside/outside convention is inverted)",
        )
        return f"grad err {gerr:.1e}, kappa {kap:.4f} vs {-2/radius:.4f}"

    r.stage("2  spacing-aware operators", s_ops, requires=["1  grid + trilinear interpolation"])

    # -- 3. phantom --------------------------------------------------------
    def s_phantom() -> str:
        from .data.phantom import PhantomConfig, make_phantom

        cfg = PhantomConfig(shape=(40, 40, 10), spacing=(1.5, 1.5, 6.0), n_frames=5, n_papillary=1)
        ph = make_phantom(cfg, device=S["dev"], dtype=S["dtype"], generator=S["gen"])
        S["phantom"] = ph

        for t, (phi, mask) in enumerate(zip(ph.phi_gt, ph.masks)):
            _require(
                bool(((phi > 0) == mask).all()),
                f"frame {t}: sign convention broken - (phi > 0) != mask",
            )
        ef = ph.ef_percent()
        _require(10.0 < ef < 90.0, f"implausible ground-truth EF {ef:.1f}%")
        frac = float(ph.masks[0].to(S["dtype"]).mean())
        _require(0.005 < frac < 0.6, f"cavity occupies {frac:.1%} of the volume - check fit_to_grid")
        touching = bool(ph.masks[0][0].any() or ph.masks[0][-1].any() or ph.masks[0][:, :, 0].any())
        _require(not touching, "cavity touches the volume border: the surface is not closed")
        return f"EF {ef:.1f}%, cavity {frac:.1%}, geom scale {ph.geometry_scale:.2f}"

    r.stage("3  phantom + analytic ground truth", s_phantom, requires=["0  import torch / resolve device"])

    # -- 4. SDF tools ------------------------------------------------------
    def s_sdf() -> str:
        from .levelset.operators import gradient_central, gradient_norm
        from .levelset.sdf import reinitialize, signed_distance_from_mask

        ph = S["phantom"]
        phi = signed_distance_from_mask(ph.masks[0], ph.grid.spacing, max_dist_mm=10.0)
        S["phi0_init"] = phi

        deep_in = ph.phi_gt[0] > 2.0
        deep_out = ph.phi_gt[0] < -2.0
        _require(bool((phi[deep_in] > 0).all()), "SDF-from-mask has the wrong sign inside")
        _require(bool((phi[deep_out] < 0).all()), "SDF-from-mask has the wrong sign outside")

        band = phi.abs() < 4.0
        before = float((gradient_norm(gradient_central(phi, ph.grid.spacing))[band] - 1).abs().mean())
        after = reinitialize(phi, ph.grid.spacing, iters=30)
        aerr = float((gradient_norm(gradient_central(after, ph.grid.spacing))[band] - 1).abs().mean())
        moved = float(((after > 0) != (phi > 0)).to(S["dtype"]).mean())
        _require(moved < 0.05, f"reinitialisation moved the zero set by {moved:.1%} of voxels")
        return f"eikonal {before:.3f} -> {aerr:.3f}, zero set moved {moved:.2%}"

    r.stage("4  signed distance + reinitialisation", s_sdf, requires=["3  phantom + analytic ground truth"])

    # -- 5. Chan-Vese ------------------------------------------------------
    def s_chanvese() -> str:
        from .core.config import ChanVeseConfig
        from .levelset.chanvese import solve_frame
        from .metrics.segmentation import dice

        ph = S["phantom"]
        cfg = ChanVeseConfig(max_iters=120, check_every=5, narrow_band_mm=8.0)
        res = solve_frame(ph.images[0], S["phi0_init"], ph.grid, cfg)
        d = dice(res.phi > 0, ph.masks[0])
        _require(
            d > 0.70,
            f"Chan-Vese Dice {d:.3f} on frame 0. Below 0.7 usually means the gradient-flow "
            "sign in Eq. (4.7) is inverted, or mu is far too large",
        )
        S["phi_cv0"] = res.phi
        return f"Dice {d:.3f} in {res.iterations} iters, crop {res.crop_fraction:.1%} of volume"

    r.stage("5  Chan-Vese single frame", s_chanvese, requires=["4  signed distance + reinitialisation"])

    # -- 6. mesh extraction ------------------------------------------------
    def s_mesh() -> str:

        from .core.grid import Grid
        from .levelset.mesh_extract import marching_tetrahedra

        g = Grid(shape=(45, 45, 45), spacing=(1.0, 1.0, 1.0))
        world = g.world_meshgrid()
        c = g.center_world().view(3, 1, 1, 1)
        rad = (world - c).norm(dim=0)
        radius = 14.0
        mesh = marching_tetrahedra(radius - rad, g)
        _require(mesh.n_faces > 500, f"only {mesh.n_faces} faces extracted from a sphere")

        cen = g.center_world()
        rr = (mesh.vertices - cen).norm(dim=-1)
        rerr = float((rr - radius).abs().max())
        _require(rerr < 1.0, f"mesh vertices deviate {rerr:.2f} mm from the sphere")
        area = float(mesh.surface_area())
        exact = 4 * math.pi * radius**2
        _require(abs(area / exact - 1.0) < 0.15, f"surface area {area:.0f} vs exact {exact:.0f}")
        radial = (mesh.vertices - cen) / rr.clamp_min(1e-9).unsqueeze(-1)
        dot = float((mesh.normals * radial).sum(-1).mean())
        _require(dot < -0.9, f"mesh normals point outward (mean radial dot {dot:.2f}); with "
                             "phi > 0 inside they must point inward")
        return f"{mesh.n_faces} faces, area err {abs(area/exact-1):.1%}, vertex err {rerr:.2f} mm"

    r.stage("6  marching tetrahedra", s_mesh, requires=["2  spacing-aware operators"])

    # -- 7. canonical surfels ----------------------------------------------
    def s_canonical() -> str:
        from .core.config import SurfelConfig
        from .surfel.canonical import initialize_canonical_surfels

        ph = S["phantom"]
        cfg = SurfelConfig(n_surfels=800, isotropic=True)
        surfels, stats = initialize_canonical_surfels(
            ph.phi_gt[0], ph.images[0], ph.grid, cfg, generator=S["gen"]
        )
        S["surfels"] = surfels
        _require(surfels.n == 800, f"expected 800 surfels, got {surfels.n}")
        _require(
            stats["init_e_surf_mm"] < 0.5,
            f"anchors are {stats['init_e_surf_mm']:.3f} mm off the level set after projection",
        )
        err = surfels.frame_orthonormality_error()
        worst = max(float(v.max()) for v in err.values())
        _require(worst < 1e-3, f"tangent frames are not orthonormal: worst residual {worst:.2e}")
        return f"N={surfels.n}, E_surf {stats['init_e_surf_mm']:.4f} mm, frame err {worst:.1e}"

    r.stage("7  canonical surfel initialisation", s_canonical, requires=["6  marching tetrahedra"])

    # -- 8. projection + transport -----------------------------------------
    def s_projection() -> str:
        import torch

        from .core.grid import Grid
        from .surfel.projection import project_to_surface
        from .surfel.transport import initial_tangent_frame, transport_tangent_frame

        g = Grid(shape=(49, 49, 49), spacing=(1.0, 1.0, 1.0))
        world = g.world_meshgrid()
        c = g.center_world()
        rad = (world - c.view(3, 1, 1, 1)).norm(dim=0)
        radius = 15.0
        phi = radius - rad

        dirs = torch.randn(300, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        start = c + dirs * (radius + 2.5)
        res = project_to_surface(start, phi, g, iters=3, eps=0.0, max_step_mm=10.0)
        _require(
            float(res.residual_mm.mean()) < 0.1,
            f"projection left E_surf = {float(res.residual_mm.mean()):.3f} mm",
        )
        h = res.residual_history
        _require(h[1] < h[0], f"first projection step increased the residual: {h[0]:.4f} -> {h[1]:.4f}")

        n_old = torch.randn(200, 3, dtype=torch.float64)
        n_old = n_old / n_old.norm(dim=-1, keepdim=True)
        e1, e2 = initial_tangent_frame(n_old, eps=0.0)
        n_new = n_old + 0.05 * torch.randn(200, 3, dtype=torch.float64)
        n_new = n_new / n_new.norm(dim=-1, keepdim=True)
        f1, f2, diag = transport_tangent_frame(e1, e2, n_new, normal_prev=n_old, eps=0.0)
        orth = max(
            float((f1 * n_new).sum(-1).abs().max()),
            float((f1 * f2).sum(-1).abs().max()),
            float((f1.norm(dim=-1) - 1).abs().max()),
        )
        _require(orth < 1e-10, f"transport broke orthonormality: {orth:.2e}")
        return f"E_surf {float(res.residual_mm.mean()):.5f} mm, residual {h[0]:.3f}->{h[-1]:.5f}, transport {orth:.1e}"

    r.stage("8  normal projection + tangent transport", s_projection, requires=["2  spacing-aware operators"])

    # -- 9. 2DGS rasteriser ------------------------------------------------
    def s_raster() -> str:
        import torch

        from .core.config import RenderConfig
        from .render.camera import Camera
        from .render.raster2dgs import render_2dgs
        from .surfel.model import SurfelSet2D
        from .surfel.transport import initial_tangent_frame

        normal = torch.tensor([[0.0, 0.0, -1.0]])
        e1, e2 = initial_tangent_frame(normal)
        surf = SurfelSet2D(
            torch.zeros(1, 3), e1, e2, normal, torch.full((1, 2), 2.0),
            torch.ones(1, 1), torch.full((1,), 0.9),
        )
        cam = Camera.look_at(
            torch.tensor([0.0, 0.0, 50.0]), torch.zeros(3), height=65, width=65, fov_deg=60.0
        )
        out = render_2dgs(surf, cam, RenderConfig(tile=16, max_per_tile=64), compute_aux=True)
        a = float(out.alpha[32, 32])
        _require(a > 0.5, f"a disk facing the camera rendered alpha {a:.3f} at its centre")
        _require(abs(a - 0.9) < 0.05, f"centre alpha {a:.3f}, expected ~0.9 (the opacity)")
        # out.depth is the alpha-WEIGHTED accumulation sum_i w_i tau_i, so for a single
        # disk of opacity 0.9 at 50 mm it reads 45 mm. Only mean_depth() - divided by the
        # accumulated alpha - is comparable to a geometric distance. Both are pinned here
        # because confusing them silently biases every reported depth error by the opacity,
        # which is exactly what this check caught on the first GPU run.
        d_raw = float(out.depth[32, 32])
        d = float(out.mean_depth()[32, 32])
        _require(
            abs(d - 50.0) < 0.5,
            f"expected depth {d:.2f} mm, should be the 50 mm camera distance "
            f"(raw accumulation was {d_raw:.2f} mm)",
        )
        _require(
            abs(d_raw - a * 50.0) < 0.5,
            f"raw depth {d_raw:.2f} mm should equal alpha*distance = {a * 50.0:.2f} mm; "
            f"if this fails the accumulation convention has changed and "
            f"expected_depth() must be revisited",
        )
        _require(float(out.alpha.max()) <= 1.0 + 1e-5, "alpha exceeded 1")
        return f"centre alpha {a:.3f}, expected depth {d:.2f} mm (raw {d_raw:.2f})"

    r.stage("9  2DGS rasteriser (analytic single disk)", s_raster, requires=["0  import torch / resolve device"])

    # -- 10. tiled vs reference --------------------------------------------
    def s_reference() -> str:
        import torch

        from .core.config import RenderConfig
        from .render.camera import Camera
        from .render.raster2dgs import render_2dgs, render_2dgs_reference
        from .surfel.model import SurfelSet2D
        from .surfel.transport import initial_tangent_frame

        torch.manual_seed(3)
        n = 8
        normal = torch.randn(n, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        e1, e2 = initial_tangent_frame(normal)
        surf = SurfelSet2D(
            torch.randn(n, 3) * 14.0, e1, e2, normal, torch.full((n, 2), 1.2),
            torch.rand(n, 1), torch.full((n,), 0.7),
        )
        cam = Camera.look_at(
            torch.tensor([0.0, 0.0, 90.0]), torch.zeros(3), height=48, width=48, fov_deg=45.0
        )
        cfg = RenderConfig(tile=16, max_per_tile=64, tile_chunk=16)
        fast = render_2dgs(surf, cam, cfg, compute_aux=True)
        ref = render_2dgs_reference(surf, cam, cfg)
        da = float((fast.alpha - ref.alpha).abs().max())
        dc = float((fast.color - ref.color).abs().max())
        _require(da < 1e-3, f"tiled vs reference alpha differs by {da:.2e} on separated surfels")
        _require(dc < 1e-3, f"tiled vs reference colour differs by {dc:.2e}")
        return f"alpha gap {da:.1e}, colour gap {dc:.1e}"

    r.stage("10 tiled == brute-force reference", s_reference, requires=["9  2DGS rasteriser (analytic single disk)"])

    # -- 11. autograd ------------------------------------------------------
    def s_autograd() -> str:
        import torch

        from .core.config import RenderConfig
        from .render.camera import Camera
        from .render.raster2dgs import render_2dgs
        from .surfel.model import SurfelSet2D
        from .surfel.transport import initial_tangent_frame

        torch.manual_seed(5)
        n = 10
        normal = torch.randn(n, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        e1, e2 = initial_tangent_frame(normal)
        surf = SurfelSet2D(
            torch.randn(n, 3) * 6.0, e1, e2, normal, torch.full((n, 2), 2.0),
            torch.rand(n, 1), torch.full((n,), 0.7),
        )
        cam = Camera.look_at(
            torch.tensor([0.0, 0.0, 50.0]), torch.zeros(3), height=32, width=32
        )
        out = render_2dgs(surf, cam, RenderConfig(tile=16, max_per_tile=64), compute_aux=True)
        (out.color.sum() + out.alpha.sum()).backward()

        for name, p in (("amplitude", surf.amplitude), ("opacity_logit", surf.opacity_logit),
                        ("log_scale", surf.log_scale)):
            _require(p.grad is not None, f"no gradient reached {name}")
            _require(bool(torch.isfinite(p.grad).all()), f"non-finite gradient on {name}")
            _require(float(p.grad.abs().sum()) > 0, f"zero gradient on {name}")
        for name in ("anchor", "e1", "normal"):
            _require(
                not getattr(surf, name).requires_grad,
                f"{name} is differentiable - geometry must be a buffer, not a parameter",
            )
        return "gradients finite and non-zero; geometry is not differentiable"

    r.stage("11 differentiability of the rasteriser", s_autograd, requires=["9  2DGS rasteriser (analytic single disk)"])

    # -- 12. ray-march reference -------------------------------------------
    def s_raymarch() -> str:
        import torch

        from .core.grid import Grid
        from .render.camera import Camera
        from .render.raymarch import raymarch_levelset

        g = Grid(shape=(49, 49, 49), spacing=(1.0, 1.0, 1.0))
        world = g.world_meshgrid()
        c = g.center_world()
        rad = (world - c.view(3, 1, 1, 1)).norm(dim=0)
        radius = 15.0
        phi = radius - rad
        cam = Camera.look_at(c + torch.tensor([0.0, 0.0, 120.0]), c, height=48, width=48, fov_deg=35.0)
        ref = raymarch_levelset(phi, g, cam, image=torch.ones_like(phi))

        n_hit = int(ref.hit.sum())
        _require(n_hit > 100, f"ray march hit only {n_hit} pixels on a large sphere")
        dmin = float(ref.depth[ref.hit].min())
        _require(abs(dmin - (120.0 - radius)) < 1.0,
                 f"nearest hit at {dmin:.2f} mm, expected {120.0 - radius:.2f}")
        pts = ref.point.permute(1, 2, 0)[ref.hit]
        rr = (pts - c).norm(dim=-1)
        _require(float((rr - radius).abs().max()) < 1.0, "hit points are not on the sphere")
        return f"{n_hit} hits, nearest {dmin:.2f} mm (exact {120.0-radius:.2f})"

    r.stage("12 ray-marched reference rendering", s_raymarch, requires=["2  spacing-aware operators"])

    # -- 13. rendering operator --------------------------------------------
    def s_weights() -> str:
        import torch

        from .core.config import RenderConfig
        from .render.camera import Camera
        from .render.raster2dgs import render_2dgs, render_weights
        from .surfel.model import SurfelSet2D
        from .surfel.transport import initial_tangent_frame

        torch.manual_seed(17)
        n = 10
        normal = torch.randn(n, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        e1, e2 = initial_tangent_frame(normal)
        surf = SurfelSet2D(
            torch.randn(n, 3) * 8.0, e1, e2, normal, torch.full((n, 2), 1.5),
            torch.rand(n, 1), torch.full((n,), 0.7),
        )
        cam = Camera.look_at(torch.tensor([0.0, 0.0, 60.0]), torch.zeros(3), height=48, width=48)
        cfg = RenderConfig(tile=16, max_per_tile=128)
        wm = render_weights(surf, cam, cfg)
        direct = render_2dgs(surf, cam, cfg, compute_aux=False)
        gap = float((wm.apply(surf.amplitude.detach()) - direct.color).abs().max())
        _require(gap < 1e-3, f"A a differs from the renderer by {gap:.2e} - Prop. 9.1 linearity broken")

        a = torch.randn(n, 1)
        img = torch.randn(1, 48, 48)
        left = float((wm.apply(a) * img).sum())
        right = float((a * wm.apply_transpose(img)).sum())
        _require(
            abs(left - right) <= 1e-4 * max(1.0, abs(left)),
            f"adjoint identity <A a, r> = <a, A^T r> failed: {left:.6f} vs {right:.6f}",
        )
        return f"nnz {wm.nnz}, A a gap {gap:.1e}, adjoint gap {abs(left-right):.1e}"

    r.stage("13 rendering operator A and its adjoint", s_weights, requires=["10 tiled == brute-force reference"])

    # -- 14. residual solver -----------------------------------------------
    def s_residual() -> str:
        import torch

        from .core.config import RenderConfig, ResidualConfig
        from .render.camera import Camera
        from .render.raster2dgs import render_weights
        from .residual.solver import ResidualView, solve_residual
        from .surfel.model import SurfelSet2D
        from .surfel.transport import initial_tangent_frame

        torch.manual_seed(29)
        n = 40
        normal = torch.randn(n, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        e1, e2 = initial_tangent_frame(normal)
        surf = SurfelSet2D(
            torch.randn(n, 3) * 6.0, e1, e2, normal, torch.full((n, 2), 1.8),
            torch.rand(n, 1), torch.full((n,), 0.7),
        )
        cam = Camera.look_at(torch.tensor([0.0, 0.0, 50.0]), torch.zeros(3), height=48, width=48)
        cfg = RenderConfig(tile=16, max_per_tile=128)
        wm = render_weights(surf, cam, cfg)
        base = surf.amplitude.detach()
        target = wm.apply(base + 0.25 * torch.randn_like(base))

        res = solve_residual(
            [ResidualView(weights=wm, target=target)], base,
            ResidualConfig(lambda_a=1e-4, lambda_T=1e-4, cg_iters=200, cg_tol=1e-10),
        )
        _require(
            res.data_term_after < 0.5 * res.data_term_before,
            f"CG barely reduced the data term: {res.data_term_before:.4e} -> {res.data_term_after:.4e}",
        )
        _require(res.relative_residual < 1e-4, f"CG did not converge: {res.relative_residual:.2e}")
        return f"{res.iterations} CG iters, data term -{100*(1-res.data_term_after/max(res.data_term_before,1e-30)):.0f}%"

    r.stage("14 residual solver (CG on Eq. 9.3)", s_residual, requires=["13 rendering operator A and its adjoint"])

    # -- 15. low-rank ------------------------------------------------------
    def s_lowrank() -> str:
        import torch

        from .residual.lowrank import compress_residual

        torch.manual_seed(37)
        mat = torch.randn(200, 6, dtype=torch.float64) @ torch.randn(6, 20, dtype=torch.float64)
        mat = mat + 0.02 * torch.randn(200, 20, dtype=torch.float64)
        worst = 0.0
        for rank in (2, 4, 8):
            lr = compress_residual(mat, rank)
            pf, ps = lr.predicted_frobenius_error(), lr.predicted_spectral_error()
            worst = max(
                worst,
                abs(lr.frobenius_error - pf) / max(pf, 1e-30),
                abs(lr.spectral_error - ps) / max(ps, 1e-30),
            )
        _require(worst < 1e-6, f"Eckart-Young identity violated by {worst:.2e}")
        return f"truncation error matches the singular tail to {worst:.1e}"

    r.stage("15 low-rank residual (Eckart-Young)", s_lowrank, requires=["0  import torch / resolve device"])

    # -- 16. baselines render ----------------------------------------------
    def s_baselines() -> str:
        import torch

        from .baselines import get_baseline, render_baseline
        from .core.grid import Grid
        from .levelset.mesh_extract import marching_tetrahedra
        from .render.camera import Camera
        from .surfel.model import SurfelSet2D
        from .surfel.transport import initial_tangent_frame

        g = Grid(shape=(41, 41, 41), spacing=(1.0, 1.0, 1.0))
        world = g.world_meshgrid()
        c = g.center_world()
        phi = 13.0 - (world - c.view(3, 1, 1, 1)).norm(dim=0)
        mesh = marching_tetrahedra(phi, g)
        cam = Camera.look_at(c + torch.tensor([0.0, 0.0, 90.0]), c, height=48, width=48, fov_deg=35.0)

        torch.manual_seed(9)
        n = 300
        normal = torch.randn(n, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        e1, e2 = initial_tangent_frame(normal)
        surf = SurfelSet2D(
            c + 13.0 * normal, e1, e2, -normal, torch.full((n, 2), 1.5),
            torch.rand(n, 1), torch.full((n,), 0.8),
        )

        covered = {}
        for name in ("mesh-only", "thin-3dgs", "cv-dyn2dgs"):
            spec = get_baseline(name)
            out = render_baseline(
                spec, cam, surfels=surf, mesh=mesh,
                vertex_amplitude=torch.ones(mesh.n_vertices, 1),
            )
            frac = float((out.alpha > 0.2).to(torch.float32).mean())
            _require(frac > 0.01, f"{name} covered only {frac:.2%} of the image")
            covered[name] = frac
        return ", ".join(f"{k} {v:.0%}" for k, v in covered.items())

    r.stage("16 mesh + thin-3DGS baselines render", s_baselines, requires=["6  marching tetrahedra", "9  2DGS rasteriser (analytic single disk)"])

    if not full:
        if verbose:
            print("\n  (skipping pipeline stages; pass --full to include them)")
        return list(r.results.values())

    # -- 17. full precompute -----------------------------------------------
    def s_precompute() -> str:
        from .core.config import get_preset
        from .experiments.common import make_eval_cameras, run_pipeline_on_phantom

        ph = S["phantom"]
        cfg = get_preset("v1-minimal")
        cfg.surfel.n_surfels = 500
        cfg.fit.iters = 8
        cfg.chanvese.max_iters = 35
        cfg.residual.cg_iters = 6

        # device/dtype must be passed: make_eval_cameras defaults to the CPU, and the
        # phantom lives on the resolved device, so omitting them produced CPU cameras
        # against CUDA volumes - the device mismatch seen on the first GPU run.
        cams = make_eval_cameras(
            ph.grid, n_orbit=2, resolution=48, device=S["dev"], dtype=S["dtype"]
        )
        model, cams, _ = run_pipeline_on_phantom(ph, cfg, cameras=cams, generator=S["gen"])
        S["model"] = model
        S["cams"] = cams

        _require(model.n_frames == ph.n_frames, "frame count mismatch")
        _require(len(model.residuals) == ph.n_frames, "residual count mismatch")
        worst = max(f.e_surf_mm for f in model.frames)
        _require(worst < 2.0, f"anchors drifted {worst:.3f} mm off the level set")
        for t in range(model.n_frames):
            amp = model.amplitude_at(t)
            _require(
                amp.shape[0] == model.surfels.n,
                f"amplitude_at({t}) has {amp.shape[0]} rows, surfels has {model.surfels.n}",
            )
        return f"{model.n_frames} frames, N={model.surfels.n}, worst E_surf {worst:.4f} mm"

    r.stage("17 full precompute pipeline", s_precompute, requires=["5  Chan-Vese single frame", "7  canonical surfel initialisation"])

    # -- 18. playback ------------------------------------------------------
    def s_playback() -> str:
        import torch

        from .pipeline.playback import PlaybackEngine, measure_playback

        model, cams = S["model"], S["cams"]
        before = model.surfels.anchor.clone()
        engine = PlaybackEngine(model)
        for t in range(model.n_frames):
            out, timing = engine.step_to(t, cams.eval[0], compute_aux=False)
            _require(math.isfinite(timing.total_ms), f"non-finite timing at frame {t}")
            _require(float(out.alpha.max()) > 0.05, f"frame {t} rendered an empty image")
        _require(
            torch.equal(before, model.surfels.anchor),
            "playback mutated the canonical surfels - repeat playback would drift",
        )
        rep = measure_playback(model, cams.eval, loops=1, warmup=1, compute_aux=False)
        return f"mean {rep.mean_ms:.1f} ms ({rep.mean_fps:.0f} FPS), p95 {rep.p95_ms:.1f} ms"

    r.stage("18 playback engine", s_playback, requires=["17 full precompute pipeline"])

    # -- 19. storage round-trip --------------------------------------------
    def s_storage() -> str:
        import os
        import tempfile

        from .pipeline.storage_io import load_model, model_storage_report, save_model

        model = S["model"]
        rep = model_storage_report(model)
        primary = rep["primary"]
        cr = primary["compression_ratio"]
        _require(cr > 0, f"nonsensical compression ratio {cr}")

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.pt")
            info = save_model(model, path)
            back = load_model(path)
            _require(len(back["phis"]) == model.n_frames, "frame count changed on reload")
            band = model.config.narrow_band_store_mm
            for t, (a, b) in enumerate(zip(model.phis, back["phis"])):
                b = b.to(a.device, a.dtype)
                m = a.abs() < band
                err = float((a[m] - b[m]).abs().max()) if int(m.sum()) else 0.0
                _require(err < 1e-4, f"frame {t}: band values changed by {err:.2e} on round-trip")
                _require(
                    bool(((a > 0) == (b > 0)).all()),
                    f"frame {t}: sign flipped on round-trip - masks and volumes would be wrong",
                )
        return f"CR {cr:.2f}x, file {info['file_mb']:.2f} MB, band round-trip exact"

    r.stage("19 storage save / load round-trip", s_storage, requires=["17 full precompute pipeline"])

    # -- 20. metrics -------------------------------------------------------
    def s_metrics() -> str:
        from .experiments.common import evaluate_geometry, evaluate_rendering
        from .baselines import get_baseline

        model, cams = S["model"], S["cams"]
        ph = S["phantom"]
        ed = ph.ed_index()
        order = [(ed + t) % ph.n_frames for t in range(ph.n_frames)]
        images = [ph.images[t] for t in order]
        gt = {t: ph.masks[order[t]] for t in range(len(order))}

        geom = evaluate_geometry(model, gt_masks=gt)
        _require("geom/dice_mean" in geom, "evaluate_geometry produced no Dice")
        _require(geom["geom/dice_mean"] > 0.5, f"Dice {geom['geom/dice_mean']:.3f} too low to be wired up")

        rend = evaluate_rendering(
            model, get_baseline("cv-dyn2dgs"), images, cams.eval[:1], frames=[0, 1]
        )
        for key in ("render/best_iou", "render/psnr_roi_db", "render/normal_mean_deg"):
            _require(key in rend, f"missing metric {key}")
        return (
            f"Dice {geom['geom/dice_mean']:.3f}, IoU {rend['render/best_iou']:.3f}, "
            f"PSNR_roi {rend['render/psnr_roi_db']:.1f} dB"
        )

    r.stage("20 metric suite", s_metrics, requires=["18 playback engine"])

    # -- 21. layer-1 comparison: the surface source is swappable -----------
    def s_surface_source() -> str:

        from .levelset.surface_source import (
            MaskSequenceSource,
            OracleSource,
            sdf_fidelity,
        )

        ph = S["phantom"]
        grid = ph.grid
        images = [ph.images[t] for t in range(min(3, ph.n_frames))]
        masks = [ph.masks[t] for t in range(len(images))]
        exact = [ph.phi_gt[t] for t in range(len(images))]

        mask_seq = MaskSequenceSource(masks, name="smoke-mask", producer="phantom labels").build(
            images, grid
        )
        _require(mask_seq.n_frames == len(images), "mask source produced the wrong frame count")
        _require(
            all(bool((m == (p > 0)).float().mean() > 0.95) for m, p in zip(masks, mask_seq.phis)),
            "mask -> SDF changed the inside/outside labelling by more than 5% of voxels",
        )

        # Partial coverage must be refused, not silently averaged over.
        partial = OracleSource(
            [p.clone() for p in mask_seq.phis], analytic=False, available_frames=[0],
            origin="smoke partial-coverage probe",
        ).build(images, grid)
        try:
            partial.require_full("a whole-sequence metric")
        except ValueError:
            pass
        else:
            raise AssertionError(
                "require_full() accepted a source covering 1 of "
                f"{len(images)} frames - partial oracles would silently become full ones"
            )

        fid_mask = sdf_fidelity(mask_seq.phis[0], grid)
        detail = (
            f"mask staircase {fid_mask['sdf/staircase_index']:.3f}, "
            f"eikonal {fid_mask['sdf/eikonal_abs_mean']:.3f}"
        )
        if exact:
            fid_exact = sdf_fidelity(exact[0].to(mask_seq.phis[0].dtype), grid)
            # The diagnostic exists to separate a quantised surface from a smooth one.
            # If it cannot tell them apart it is not measuring anything.
            _require(
                fid_exact["sdf/eikonal_abs_mean"] <= fid_mask["sdf/eikonal_abs_mean"] + 1e-6,
                f"analytic SDF scored worse on the eikonal residual "
                f"({fid_exact['sdf/eikonal_abs_mean']:.4f}) than a mask-derived one "
                f"({fid_mask['sdf/eikonal_abs_mean']:.4f}) - the diagnostic is inverted",
            )
            detail += f", analytic eikonal {fid_exact['sdf/eikonal_abs_mean']:.3f}"
        return detail

    r.stage(
        "21 surface source swap + SDF fidelity",
        s_surface_source,
        requires=["3  phantom + analytic ground truth",
                  "4  signed distance + reinitialisation"],
    )

    # -- 22. layer-2 comparison: geometry that is free ---------------------
    def s_free3dgs() -> str:
        import torch

        from .core.config import RenderConfig
        from .surfel.free3dgs import FreeGaussians3D, render_free_3dgs

        cams = S["cams"]
        grid = S["phantom"].grid
        model = S.get("model")
        pts = model.surfels.anchor if model is not None else None

        g = FreeGaussians3D.initialise(
            mode="surface" if pts is not None else "bbox_random",
            n_gaussians=128,
            grid=grid,
            surface_points=pts,
            channels=1,
            generator=S.get("gen"),
            device=S["dev"],
        )
        _require(g.n == 128, f"expected 128 free Gaussians, got {g.n}")
        _require(tuple(g.scale3.shape) == (128, 3),
                 "a free Gaussian must have three independent scales")

        # The structural claim: geometry IS optimisable here, unlike SurfelSet2D.
        names = {n for n, _ in g.named_parameters()}
        _require("position" in names and "quaternion" in names,
                 f"free-3DGS geometry is not parameterised: {sorted(names)}")
        if model is not None:
            sn = {n for n, _ in model.surfels.named_parameters()}
            _require(
                "anchor" not in sn,
                "SurfelSet2D exposes 'anchor' as a Parameter - the Chan-Vese surface "
                "could then be moved by a loss, which the method forbids",
            )

        # Rotation columns must stay orthonormal or the covariance is not a rotation.
        rot = torch.stack((g.e1, g.e2, g.normal), dim=-1)
        gram = rot.transpose(-1, -2) @ rot
        eye = torch.eye(3, device=gram.device, dtype=gram.dtype)
        err = float((gram - eye).abs().max())
        _require(err < 1e-4, f"rotation columns not orthonormal, max deviation {err:.2e}")

        out = render_free_3dgs(g, cams.eval[0], RenderConfig(), compute_aux=True)
        _require(bool(torch.isfinite(out.color).all()), "free-3DGS render produced non-finite colour")
        _require(float(out.alpha.max()) > 0.0, "free-3DGS rendered a completely empty image")
        return f"128 free Gaussians, alpha_max {float(out.alpha.max()):.3f}, orth err {err:.1e}"

    r.stage(
        "22 free-geometry 3DGS renders",
        s_free3dgs,
        requires=["9  2DGS rasteriser (analytic single disk)"],
    )

    # -- 23. layer-4 comparison: viewpoint break-even ----------------------
    def s_viewpoint() -> str:
        from .metrics.viewpoint import (
            RawVolumeTarget,
            VideoCodecTarget,
            viewpoint_breakeven,
            viewpoint_table,
        )

        grid = S["phantom"].grid
        n_frames = S["phantom"].n_frames

        raw = RawVolumeTarget.from_grid(grid, n_frames)
        _require(raw.total_bytes > 0, "raw volume budget came out as zero")
        _require(raw.fps is None, "raw volume reported an fps that nobody measured")

        # A geometry axis must be refused, not silently answered.
        try:
            raw.score_axis("e_surf")
        except ValueError:
            pass
        else:
            raise AssertionError(
                "a storage-only target accepted the geometry axis 'e_surf'; it has no "
                "surface, so any value would be fabricated"
            )
        raw.score_axis("bytes")  # must not raise

        # A provenance-free byte count must be impossible to construct.
        try:
            VideoCodecTarget(key="x", name="x", total_bytes=1, provenance="  ")
        except ValueError:
            pass
        else:
            raise AssertionError("a byte count with no provenance was accepted")

        vid = VideoCodecTarget.assumed(
            bitrate_kbps=2000, n_frames=n_frames, fps_playback=24.0, n_viewpoints=1,
            codec="h265", basis="smoke test placeholder, not a real encode",
        )
        _require("ASSUMED" in vid.provenance, "an assumed bitrate was not labelled as assumed")

        # A wiring check, not a measurement: a fixed placeholder keeps this stage
        # independent of stage 19 and cannot be mistaken for a storage result.
        ours = 1_000_000
        b = viewpoint_breakeven(
            ours_bytes=int(ours), video_bytes_per_viewpoint=float(vid.total_bytes),
            raw_volume_bytes_total=raw.total_bytes,
        )
        rows = viewpoint_table(
            ours_bytes=int(ours), video_bytes_per_viewpoint=float(vid.total_bytes)
        )
        _require(len(rows) > 0, "viewpoint table is empty")
        _require(
            all(float(r["ours_mb"]) == float(rows[0]["ours_mb"]) for r in rows),
            "our storage changed with the viewpoint count - it must be flat, that is the "
            "entire structural advantage being claimed",
        )
        _require(
            float(rows[-1]["video_mb"]) > float(rows[0]["video_mb"]),
            "video storage did not grow with the viewpoint count",
        )
        return f"break-even {b.breakeven_viewpoints:.2f} views; raw {raw.total_bytes/1024**2:.1f} MB"

    r.stage("23 viewpoint break-even (layer 4)", s_viewpoint,
            requires=["0  import torch / resolve device"])

    # -- 24. external adapter contract -------------------------------------
    def s_external() -> str:
        import json
        import tempfile
        from pathlib import Path as _P

        from .experiments.external import (
            AXIS_REQUIREMENTS,
            ExternalMeta,
            ExternalOutput,
            check_camera_agreement,
            write_adapter_template,
        )

        with tempfile.TemporaryDirectory() as td:
            meta_path = write_adapter_template(td, "at-gs")
            _require(meta_path.exists(), "adapter template wrote no meta.json")
            skel = json.loads(meta_path.read_text(encoding="utf-8"))
            for k in ("method", "commit", "n_frames", "camera"):
                _require(k in skel, f"template meta.json lacks {k!r}")
            for sub in ("color", "alpha", "depth", "normal"):
                _require((_P(td) / sub).is_dir(), f"template lacks the {sub}/ directory")

        import torch

        meta = ExternalMeta(
            method="at-gs", commit="0" * 40, n_frames=2, image_height=4, image_width=4,
        )
        out = ExternalOutput(meta=meta, color=[torch.zeros(1, 4, 4) for _ in range(2)])

        # Depth was not provided, so depth RMSE must be refused rather than zeroed.
        _require("psnr" in out.scoreable_axes(), "colour-only output cannot be scored on PSNR")
        _require("depth_rmse" not in out.scoreable_axes(),
                 "depth RMSE was offered for an output with no depth")
        try:
            out.require_axis("depth_rmse")
        except ValueError:
            pass
        else:
            raise AssertionError(
                "require_axis accepted an axis the method did not provide; a zero depth "
                "map scores as a specific wrong answer rather than as an absent one"
            )

        out.check_pin("1" * 40)
        _require(bool(out.warnings),
                 "a commit differing from the manifest pin raised no warning")

        # Build a camera locally rather than reading S["cams"]: that key is only set by
        # the precompute stage, so depending on it made this contract check fail with a
        # KeyError whenever an earlier stage failed - reporting a problem in the adapter
        # when the adapter was fine.
        from .render.camera import Camera

        cam = Camera.look_at(
            torch.tensor([0.0, 0.0, 50.0]), torch.zeros(3), height=16, width=16
        )
        probs = check_camera_agreement(None, [cam])
        _require(bool(probs), "an undeclared external camera was reported as agreeing")
        _require(len(AXIS_REQUIREMENTS) >= 8, "axis requirement table looks truncated")
        return f"{len(out.scoreable_axes())} scoreable axes, pin mismatch caught"

    r.stage("24 external adapter contract", s_external,
            requires=["0  import torch / resolve device"])

    return list(r.results.values())


# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cvdyn2dgs smoke",
        description="Staged smoke test. Fix the FIRST failure, then re-run.",
    )
    ap.add_argument("--full", action="store_true", help="include the end-to-end pipeline stages")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    print("CV-Dyn2DGS staged smoke test")
    print("=" * 60)
    print("Stages are ordered by dependency: fix the FIRST failure, then re-run.")
    print("SKIP means a prerequisite failed, not that the stage is broken.\n")

    results = run_smoke(full=args.full, device=args.device, seed=args.seed, verbose=not args.quiet)

    failed = [x for x in results if x.status == "FAIL"]
    skipped = [x for x in results if x.status == "SKIP"]
    passed = [x for x in results if x.ok]

    print("\n" + "=" * 60)
    print(f"{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print("\nFirst failure is the one to fix:")
        first = failed[0]
        print(f"  {first.name}\n    {first.error}")
        if first.detail:
            print("    " + first.detail.replace("\n", "\n    "))
        print("\nAll failures:")
        for x in failed:
            print(f"  - {x.name}: {x.error}")
        return 1
    if not any(x.name.startswith("20") for x in passed):
        print("\nPipeline stages were not run. Re-run with --full once these pass.")
    print("\nNext: pytest -x tests/   then   cvdyn2dgs experiments --size small")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
