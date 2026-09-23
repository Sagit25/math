"""Synthetic 4-D cardiac phantom with analytic ground truth.

Why a phantom at all
--------------------
Proposal §7 validates on ACDC and M&Ms-2, which supply ED/ES labels only - the
intermediate phases have no segmentation ground truth, and *no* real dataset
supplies a ground-truth signed distance function, exact surface normals or exact
ray depth.  But those are precisely the quantities the theory makes quantitative
predictions about:

* Prop. 7.3 - normal angular error is :math:`O(h_{\\max}^2)`;
* Lemma 6.3 - the projection residual contracts quadratically;
* Prop. 8.3 - the planar-disk error is :math:`O(\\kappa s^2)`;
* Prop. 10.1 - :math:`\\|\\nabla\\phi_\\tau\\|^2 = 1 - 2\\beta(1-\\beta)(1-\\cos\\omega)`.

Checking a convergence *rate* requires an exact reference and the ability to vary
:math:`h`.  This phantom provides both, so the theory can be verified numerically
before any claim is made about real data.  It is a validation instrument, **not** a
substitute for ACDC/M&Ms-2: it says nothing about robustness to real anatomy,
scanner variation or pathology, and the real-data path lives in
:mod:`cvdyn2dgs.data.real`.

What is modelled
----------------
* **Geometry** - the LV endocardium as a prolate spheroid whose semi-axes contract
  towards end-systole, with a small tilt and apex shift so the motion is not a pure
  scaling (a pure scaling would make tangent transport trivially exact and hide the
  drift of Prop. 11.1).  Because a rigid transform preserves distance, the signed
  distance function stays **exact** under the tilt.
* **Anisotropic sampling** - ``hz >> hx = hy`` as in short-axis cine, which is the
  entire motivation for the spacing-aware operators of theory §4.4.
* **Papillary muscles** - myocardium-intensity blobs inside the blood pool.  These
  deliberately violate Chan-Vese's piecewise-constant assumption, the limitation
  named in proposal §2.5 and §10.3(1).  They are part of the intensity model but
  *not* of the ground-truth surface, matching the ACDC convention where papillary
  muscle is counted as blood pool.
* **Appearance that actually changes over time** - the blood-pool and myocardium
  intensities pulse through the cycle.  Without this the appearance residual
  :math:`\\Delta a_t` would be identically zero and RQ3 (does the compact
  representation pay off) would be vacuous.
* **Rician noise and smooth intensity inhomogeneity** - magnitude-MRI noise
  statistics and a slowly varying multiplicative bias field.

A closed spheroid is used rather than a base-truncated cup.  A real LV is open at
the valve plane, but truncation would destroy the exact analytic SDF that is the
whole point of the phantom; base topology is exercised on real data instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from ..core.grid import Grid

__all__ = ["PhantomConfig", "Phantom4D", "ellipsoid_sdf", "make_phantom", "contraction_profile"]


# --------------------------------------------------------------------------- #
#  Exact signed distance to an ellipsoid
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ellipsoid_sdf(
    points: Tensor,
    semi_axes: Tensor,
    *,
    bisection_iters: int = 80,
) -> Tensor:
    """Exact signed distance from points to an axis-aligned ellipsoid surface.

    Sign convention matches the papers: **positive inside**, negative outside.

    The closest surface point :math:`y` to :math:`p` satisfies
    :math:`y_i = a_i^2 p_i / (a_i^2 + t)` for the Lagrange multiplier :math:`t`
    solving

    .. math:: F(t) = \\sum_i \\left(\\frac{a_i p_i}{a_i^2 + t}\\right)^2 - 1 = 0.

    :math:`F` is strictly decreasing on :math:`t > -\\min_i a_i^2`, so bisection on
    a guaranteed bracket converges unconditionally - more robust than Newton, and
    the cost is irrelevant since this runs once per frame on a small grid.

    Bracket: outside points have :math:`t \\in (0, \\sqrt{\\sum_i (a_i p_i)^2}]`
    because :math:`F(t) \\approx \\sum_i (a_i p_i)^2/t^2 - 1 < 0` beyond that;
    inside points have :math:`t \\in (-\\min_i a_i^2, 0)`.

    Parameters
    ----------
    points:
        ``(..., 3)`` positions in the ellipsoid's local frame (centre at origin).
    semi_axes:
        ``(3,)`` semi-axes :math:`a_i > 0`.

    Returns
    -------
    ``(...)`` signed distance in the same units as ``points``.
    """
    p = points.reshape(-1, 3)
    a = semi_axes.to(p.dtype).to(p.device).reshape(3)
    a2 = a * a

    scaled = (p / a) ** 2
    inside = scaled.sum(dim=-1) < 1.0

    ap2 = (a * p) ** 2  # (M, 3)
    t_hi_out = torch.sqrt(ap2.sum(dim=-1)).clamp_min(1e-12) + 1.0
    min_a2 = float(a2.min().item())

    t_lo = torch.where(inside, torch.full_like(t_hi_out, -min_a2 + 1e-9), torch.zeros_like(t_hi_out))
    t_hi = torch.where(inside, torch.zeros_like(t_hi_out), t_hi_out)

    def f_of(t: Tensor) -> Tensor:
        den = a2.view(1, 3) + t.unsqueeze(-1)
        return (ap2 / (den * den)).sum(dim=-1) - 1.0

    for _ in range(int(bisection_iters)):
        mid = 0.5 * (t_lo + t_hi)
        f_mid = f_of(mid)
        # F decreasing: F > 0 means the root is to the right of mid.
        go_right = f_mid > 0
        t_lo = torch.where(go_right, mid, t_lo)
        t_hi = torch.where(go_right, t_hi, mid)

    t = 0.5 * (t_lo + t_hi)
    y = a2.view(1, 3) * p / (a2.view(1, 3) + t.unsqueeze(-1))
    dist = (p - y).norm(dim=-1)

    # Degenerate: p at the centre. Closest surface point is along the shortest axis.
    at_centre = p.norm(dim=-1) < 1e-9
    if bool(at_centre.any()):
        dist = torch.where(at_centre, torch.full_like(dist, math.sqrt(min_a2)), dist)

    signed = torch.where(inside, dist, -dist)
    return signed.reshape(points.shape[:-1])


# --------------------------------------------------------------------------- #
#  Cycle profile
# --------------------------------------------------------------------------- #
def contraction_profile(t: int, n_frames: int, es_fraction: float = 0.35) -> float:
    """Contraction amount in ``[0, 1]``: 0 at end-diastole, 1 at end-systole.

    Rapid systolic contraction followed by slower diastolic relaxation, built from
    two raised cosines so the profile is continuous with continuous derivative at
    both ES and the wrap-around at ED.  A non-uniform profile matters: it makes the
    inter-frame displacement vary over the cycle, which is what stresses the
    small-change assumption of Def. 5.1 and the trust region of Eq. (6.3).
    """
    x = (t % n_frames) / float(n_frames)
    es = float(es_fraction)
    if x <= es:
        return 0.5 * (1.0 - math.cos(math.pi * x / es))
    return 0.5 * (1.0 + math.cos(math.pi * (x - es) / (1.0 - es)))


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PhantomConfig:
    """Parameters of the synthetic phantom. Defaults are ACDC-like."""

    shape: tuple[int, int, int] = (96, 96, 16)
    spacing: tuple[float, float, float] = (1.25, 1.25, 8.0)
    """``hz / hx = 6.4``, comparable to a real short-axis cine stack."""

    n_frames: int = 20
    es_fraction: float = 0.35

    # --- geometry (mm) ---
    a_ed_mm: float = 23.0
    """Endocardial short-axis semi-axis at ED."""
    a_es_mm: float = 15.0
    c_ed_mm: float = 38.0
    """Endocardial long-axis semi-axis at ED."""
    c_es_mm: float = 31.0
    wall_ed_mm: float = 8.0
    wall_es_mm: float = 12.0
    """Wall thickens as it contracts (approximate incompressibility)."""

    tilt_deg: float = 4.0
    """Long-axis tilt at ES; makes the motion more than a pure scaling."""
    apex_shift_mm: float = 3.0
    """Base-to-apex translation at ES."""

    # --- intensities (arbitrary units, roughly [0, 1]) ---
    blood_intensity: float = 0.85
    myo_intensity: float = 0.34
    bg_intensity: float = 0.10
    blood_pulsation: float = 0.07
    """Cycle-dependent blood-pool signal change; drives the residual."""
    myo_pulsation: float = 0.04
    texture_amplitude: float = 0.05
    """Static spatial texture on the myocardium."""

    # --- degradation ---
    noise_sigma: float = 0.04
    """Rician noise level (magnitude MRI)."""
    inhomogeneity: float = 0.15
    """Peak-to-peak amplitude of the smooth multiplicative bias field."""

    # --- confounders ---
    papillary: bool = True
    n_papillary: int = 2
    papillary_radius_mm: float = 4.0

    seed: int = 0

    fit_to_grid: bool = True
    """Shrink all lengths uniformly so the epicardium fits inside the grid.

    Without this, a small ``shape`` silently clips the spheroid at the volume border.
    That would break the closed-surface assumption the whole method rests on: the
    extracted mesh would be open, ``marching_tetrahedra`` would produce boundary
    artefacts, and the "volume inside" measurement would be wrong.  Uniform scaling
    is safe because it leaves the ejection fraction - a volume *ratio* - unchanged.
    """

    fit_margin: float = 0.88
    """Fraction of the half-extent the epicardium may occupy when ``fit_to_grid``."""

    def required_half_extent_mm(self) -> tuple[float, float, float]:
        """Half-extent the ED epicardium needs along each axis, including the pose shift."""
        a_epi = self.a_ed_mm + self.wall_ed_mm
        c_epi = self.c_ed_mm + self.wall_ed_mm
        return (a_epi, a_epi, c_epi + self.apex_shift_mm)

    def fitted(self, grid_extent_mm: tuple[float, float, float]) -> tuple["PhantomConfig", float]:
        """Return a copy scaled to fit ``grid_extent_mm``, plus the scale factor."""
        if not self.fit_to_grid:
            return self, 1.0
        need = self.required_half_extent_mm()
        avail = tuple(0.5 * e * self.fit_margin for e in grid_extent_mm)
        scale = min(a / n for a, n in zip(avail, need))
        if scale >= 1.0:
            return self, 1.0
        length_fields = (
            "a_ed_mm", "a_es_mm", "c_ed_mm", "c_es_mm",
            "wall_ed_mm", "wall_es_mm", "apex_shift_mm", "papillary_radius_mm",
        )
        kw = {f: getattr(self, f) * scale for f in length_fields}
        return replace(self, **kw), scale

    def frame_geometry(self, t: int) -> dict[str, float]:
        """Endocardial semi-axes, wall thickness and pose for frame ``t``."""
        s = contraction_profile(t, self.n_frames, self.es_fraction)
        return {
            "s": s,
            "a_mm": self.a_ed_mm + (self.a_es_mm - self.a_ed_mm) * s,
            "c_mm": self.c_ed_mm + (self.c_es_mm - self.c_ed_mm) * s,
            "wall_mm": self.wall_ed_mm + (self.wall_es_mm - self.wall_ed_mm) * s,
            "tilt_rad": math.radians(self.tilt_deg) * s,
            "apex_shift_mm": self.apex_shift_mm * s,
        }


@dataclass
class Phantom4D:
    """A generated phantom sequence with exact ground truth."""

    images: list[Tensor]
    """``n_frames`` volumes of shape ``(nx, ny, nz)``, noisy and bias-corrupted."""

    masks: list[Tensor]
    """Exact boolean endocardial masks (papillary muscle counted as blood pool)."""

    phi_gt: list[Tensor]
    """Exact signed distance functions, ``> 0`` inside."""

    clean: list[Tensor]
    """Noise-free, bias-free intensity volumes; useful for isolating the effect of
    degradation in the ablations."""

    grid: Grid
    config: PhantomConfig
    """The **effective** config, i.e. after any ``fit_to_grid`` rescaling, so the
    reported geometry always matches the voxels that were produced."""

    params: list[dict[str, float]] = field(default_factory=list)
    geometry_scale: float = 1.0
    """Factor applied by ``fit_to_grid`` (``1.0`` = the requested geometry fitted)."""

    @property
    def n_frames(self) -> int:
        return len(self.images)

    def to(self, device=None, dtype=None) -> "Phantom4D":
        cast = lambda xs, keep_bool=False: [  # noqa: E731
            x.to(device=device) if keep_bool else x.to(device=device, dtype=dtype) for x in xs
        ]
        return Phantom4D(
            images=cast(self.images),
            masks=cast(self.masks, keep_bool=True),
            phi_gt=cast(self.phi_gt),
            clean=cast(self.clean),
            grid=self.grid,
            config=self.config,
            params=self.params,
            geometry_scale=self.geometry_scale,
        )

    def volumes_ml(self) -> list[float]:
        """Per-frame LV cavity volume in millilitres (1 ml = 1000 mm^3)."""
        vox = self.grid.voxel_volume_mm3
        return [float(m.sum().item()) * vox / 1000.0 for m in self.masks]

    def analytic_volumes_ml(self) -> list[float]:
        """Exact ellipsoid volumes, independent of voxelisation.

        Comparing these with :meth:`volumes_ml` isolates discretisation error in the
        volume curve from any error introduced by the method.
        """
        out = []
        for p in self.params:
            v = (4.0 / 3.0) * math.pi * p["a_mm"] * p["a_mm"] * p["c_mm"]
            out.append(v / 1000.0)
        return out

    def ef_percent(self) -> float:
        """Ground-truth ejection fraction, proposal Eq. (41)."""
        vols = self.analytic_volumes_ml()
        v_ed, v_es = max(vols), min(vols)
        return (v_ed - v_es) / v_ed * 100.0

    def ed_index(self) -> int:
        return int(max(range(self.n_frames), key=lambda t: self.analytic_volumes_ml()[t]))

    def es_index(self) -> int:
        return int(min(range(self.n_frames), key=lambda t: self.analytic_volumes_ml()[t]))


# --------------------------------------------------------------------------- #
#  Generation
# --------------------------------------------------------------------------- #
def _rotation_x(angle: float, device, dtype) -> Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], device=device, dtype=dtype
    )


@torch.no_grad()
def make_phantom(
    cfg: PhantomConfig | None = None,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Phantom4D:
    """Generate the phantom sequence.

    Randomness (noise, papillary placement) is drawn from ``generator`` if given,
    otherwise from a fresh CPU generator seeded with ``cfg.seed``, so a phantom is
    reproducible from its config alone.
    """
    requested = cfg or PhantomConfig()
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(requested.seed))

    grid = Grid(shape=requested.shape, spacing=requested.spacing)
    # Shrink the anatomy if the grid cannot hold it (see PhantomConfig.fit_to_grid).
    cfg, geom_scale = requested.fitted(grid.extent_mm)
    centre = grid.center_world(device=device, dtype=dtype)
    world = grid.world_meshgrid(device=device, dtype=dtype)  # (3, nx, ny, nz)
    xyz = world.permute(1, 2, 3, 0)  # (nx, ny, nz, 3)

    # Smooth multiplicative bias field: a few low-frequency sinusoids.
    ext = torch.tensor(grid.extent_mm, device=device, dtype=dtype).clamp_min(1e-6)
    rel = (xyz - centre) / ext
    bias = (
        1.0
        + cfg.inhomogeneity
        * 0.5
        * (
            torch.sin(2.0 * math.pi * 0.6 * rel[..., 0] + 0.7)
            + torch.cos(2.0 * math.pi * 0.45 * rel[..., 1] - 0.3)
            + 0.5 * torch.sin(2.0 * math.pi * 0.3 * rel[..., 2] + 1.1)
        )
        / 2.5
    )

    # Static texture on the myocardium.
    texture = cfg.texture_amplitude * torch.sin(
        2.0 * math.pi * (1.7 * rel[..., 0] + 1.3 * rel[..., 1] + 0.9 * rel[..., 2])
    )

    # Papillary muscle seeds, placed once in a normalised ED frame and carried along.
    pap_local: list[Tensor] = []
    if cfg.papillary and cfg.n_papillary > 0:
        for i in range(int(cfg.n_papillary)):
            ang = 2.0 * math.pi * (i + 0.25) / cfg.n_papillary
            r = 0.55
            pap_local.append(
                torch.tensor(
                    [r * math.cos(ang), r * math.sin(ang), -0.15],
                    device=device,
                    dtype=dtype,
                )
            )

    images: list[Tensor] = []
    masks: list[Tensor] = []
    phis: list[Tensor] = []
    cleans: list[Tensor] = []
    params: list[dict[str, float]] = []

    for t in range(int(cfg.n_frames)):
        gp = cfg.frame_geometry(t)
        params.append(gp)

        rot = _rotation_x(gp["tilt_rad"], device, dtype)
        origin = centre.clone()
        origin[2] = origin[2] + gp["apex_shift_mm"]

        # World -> ellipsoid local frame (rigid: distances are preserved exactly).
        local = (xyz - origin) @ rot  # (nx, ny, nz, 3); rot^T applied on the right

        a_endo = torch.tensor(
            [gp["a_mm"], gp["a_mm"], gp["c_mm"]], device=device, dtype=dtype
        )
        a_epi = a_endo + gp["wall_mm"]

        phi = ellipsoid_sdf(local, a_endo)  # exact, > 0 inside
        mask = phi > 0
        phi_epi = ellipsoid_sdf(local, a_epi)
        myo = (phi_epi > 0) & (~mask)

        # --- intensity model ---
        cyc = 2.0 * math.pi * t / float(cfg.n_frames)
        blood = cfg.blood_intensity + cfg.blood_pulsation * math.sin(cyc)
        myo_i = cfg.myo_intensity + cfg.myo_pulsation * math.sin(cyc + 1.9)

        img = torch.full_like(phi, float(cfg.bg_intensity))
        img = torch.where(myo, torch.full_like(img, float(myo_i)) + texture, img)
        img = torch.where(mask, torch.full_like(img, float(blood)), img)

        # Papillary muscles: myocardium intensity, inside the cavity, not in the mask.
        for pl in pap_local:
            pc = pl * a_endo
            d = (local - pc).norm(dim=-1)
            pap = (d < cfg.papillary_radius_mm) & mask
            img = torch.where(pap, torch.full_like(img, float(myo_i)), img)

        clean = img.clone()
        img = img * bias

        if cfg.noise_sigma > 0:
            # Rician: magnitude of a complex signal with independent Gaussian noise.
            n1 = torch.randn(img.shape, generator=generator, dtype=torch.float32).to(
                device=img.device, dtype=img.dtype
            )
            n2 = torch.randn(img.shape, generator=generator, dtype=torch.float32).to(
                device=img.device, dtype=img.dtype
            )
            s = float(cfg.noise_sigma)
            img = torch.sqrt((img + s * n1) ** 2 + (s * n2) ** 2)

        images.append(img)
        cleans.append(clean)
        masks.append(mask)
        phis.append(phi)

    return Phantom4D(
        images=images,
        masks=masks,
        phi_gt=phis,
        clean=cleans,
        grid=grid,
        config=cfg,
        params=params,
        geometry_scale=geom_scale,
    )
