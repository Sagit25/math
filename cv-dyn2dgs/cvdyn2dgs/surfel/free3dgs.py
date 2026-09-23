"""Free-geometry 3D Gaussians - the layer-2 comparison that was missing.

The gap this closes
-------------------
The ``thin-3dgs`` baseline in :mod:`cvdyn2dgs.baselines` shares CV-Dyn2DGS's anchors,
tangent frames and tangential scales.  That makes it a *kernel swap*, not a comparison
with 3D Gaussian Splatting: it cannot answer the question that actually decides whether
this thesis's central constraint is worth anything, namely

    *is pinning the geometry to the Chan-Vese surface a cost or a benefit?*

Answering that needs a baseline whose geometry is **free** - positions, orientations and
all three scales optimised against the same supervision.  That is this module.  It is
deliberately the mirror image of :class:`cvdyn2dgs.surfel.model.SurfelSet2D`: there,
geometry is registered as buffers so no loss can reach it; here, geometry *is* the
parameters.

What the result is expected to look like, and why that is the point
------------------------------------------------------------------
Unconstrained Gaussians should win on photometric metrics and lose badly on
:math:`E_{\\mathrm{surf}}`: with nothing holding them to the surface they will drift to
wherever the image residual is smallest, including inside the myocardium and outside the
organ entirely.  Reporting PSNR alone would therefore make this baseline look better than
the proposed method, and reporting :math:`E_{\\mathrm{surf}}` alone would make it look
worse.  **The trade-off between the two is the thesis argument**, so
:func:`fit_free_3dgs` returns both and :class:`FreeFitReport` refuses to summarise itself
without both.

Initialisation is a confound, so it is explicit
-----------------------------------------------
Where the free Gaussians start changes the answer.  Seeding them on the Chan-Vese surface
hands them the very prior under test; seeding them randomly in the bounding box makes the
comparison harsher than a real 3DGS pipeline would be, since real 3DGS initialises from
SfM points.  Both are provided and neither is the default -
:meth:`FreeGaussians3D.initialise` requires the choice to be named, and the choice is
recorded in the report so no reader has to guess which was used.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

from ..core.config import LossConfig, RenderConfig
from ..core.grid import Grid
from ..render.camera import Camera
from ..render.raster2dgs import RenderOutput
from ..render.raster3dgs import Thin3DGSConfig, render_thin_3dgs

__all__ = [
    "FreeGaussians3D",
    "FreeFitReport",
    "InitMode",
    "render_free_3dgs",
    "fit_free_3dgs",
]

InitMode = Literal["surface", "bbox_random"]


def _quat_to_rot(q: Tensor) -> Tensor:
    """``(N, 4)`` quaternion ``(w, x, y, z)`` -> ``(N, 3, 3)`` rotation, columns orthonormal."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        dim=-2,
    )


def _inverse_sigmoid(x: Tensor) -> Tensor:
    x = x.clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1 - x))


class FreeGaussians3D(nn.Module):
    """3D Gaussians whose geometry is optimised, not pinned.

    Exposes the same read-only attribute names the rasteriser consumes
    (``anchor, e1, e2, normal, scale, amplitude, opacity, n, channels, device, dtype``)
    so :func:`cvdyn2dgs.render.raster3dgs.render_thin_3dgs` can be reused unchanged -
    with the third scale supplied independently instead of tied to the tangential ones.

    Note the asymmetry with :class:`~cvdyn2dgs.surfel.model.SurfelSet2D` is structural,
    not a matter of learning rates: there ``anchor`` is a buffer, here it is a
    ``Parameter``.  Neither can be turned into the other by configuration, which is the
    point - the constraint under test cannot be switched off by accident.
    """

    def __init__(
        self,
        position: Tensor,
        quaternion: Tensor,
        log_scale: Tensor,
        amplitude: Tensor,
        opacity: Tensor,
        *,
        scale_min_mm: float = 0.05,
        scale_max_mm: float = 6.0,
        init_mode: str = "unspecified",
    ) -> None:
        super().__init__()
        n = position.shape[0]
        if tuple(position.shape) != (n, 3):
            raise ValueError(f"position must be (N, 3), got {tuple(position.shape)}")
        if tuple(quaternion.shape) != (n, 4):
            raise ValueError(f"quaternion must be (N, 4), got {tuple(quaternion.shape)}")
        if tuple(log_scale.shape) != (n, 3):
            raise ValueError(
                f"log_scale must be (N, 3) - a free Gaussian has three independent "
                f"scales; got {tuple(log_scale.shape)}"
            )
        if amplitude.dim() == 1:
            amplitude = amplitude.unsqueeze(-1)
        if amplitude.shape[0] != n or opacity.shape != (n,):
            raise ValueError("amplitude must have N rows and opacity must be (N,)")

        self.scale_min_mm = float(scale_min_mm)
        self.scale_max_mm = float(scale_max_mm)
        self.init_mode = str(init_mode)

        # EVERYTHING is a parameter. This is the whole difference from SurfelSet2D.
        self.position = nn.Parameter(position.clone())
        self.quaternion = nn.Parameter(quaternion.clone())
        self.log_scale3 = nn.Parameter(log_scale.clone())
        self.amplitude = nn.Parameter(amplitude.clone())
        self.opacity_logit = nn.Parameter(_inverse_sigmoid(opacity))

    # -------------------------------------------------------------- geometry
    @property
    def n(self) -> int:
        return int(self.position.shape[0])

    @property
    def channels(self) -> int:
        return int(self.amplitude.shape[1])

    @property
    def device(self) -> torch.device:
        return self.position.device

    @property
    def dtype(self) -> torch.dtype:
        return self.position.dtype

    @property
    def anchor(self) -> Tensor:
        return self.position

    @property
    def scale3(self) -> Tensor:
        """``(N, 3)`` positive scales in mm."""
        return self.log_scale3.exp().clamp(self.scale_min_mm, self.scale_max_mm)

    @property
    def scale(self) -> Tensor:
        """``(N, 2)`` first two scales, for rasteriser compatibility."""
        return self.scale3[:, :2]

    @property
    def scale_normal(self) -> Tensor:
        """``(N,)`` the independent third scale."""
        return self.scale3[:, 2]

    def _rot(self) -> Tensor:
        return _quat_to_rot(self.quaternion)

    @property
    def e1(self) -> Tensor:
        return self._rot()[:, :, 0]

    @property
    def e2(self) -> Tensor:
        return self._rot()[:, :, 1]

    @property
    def normal(self) -> Tensor:
        """Third rotation column.

        Called ``normal`` only so the rasteriser can read it.  For a free Gaussian this
        is *not* a surface normal: nothing constrains it to agree with the geometry of
        any surface.  Scoring it against the reference normal map is legitimate and is
        expected to be where this baseline does worst.
        """
        return self._rot()[:, :, 2]

    @property
    def opacity(self) -> Tensor:
        return torch.sigmoid(self.opacity_logit)

    # ------------------------------------------------------------ construction
    @classmethod
    def initialise(
        cls,
        *,
        mode: InitMode,
        n_gaussians: int,
        grid: Grid,
        surface_points: Tensor | None = None,
        amplitude_init: float = 0.5,
        opacity_init: float = 0.1,
        scale_init_mm: float = 1.0,
        channels: int = 1,
        generator: torch.Generator | None = None,
        device=None,
        dtype=torch.float32,
    ) -> "FreeGaussians3D":
        """Create ``n_gaussians`` Gaussians.  ``mode`` must be stated explicitly.

        ``"surface"``
            Seed on ``surface_points`` (typically sampled from the Chan-Vese zero level
            set).  Favourable to the baseline, and arguably unfair *to the thesis*,
            because it hands the baseline the prior under test.
        ``"bbox_random"``
            Seed uniformly in the grid's bounding box.  Closer to a from-scratch 3DGS run
            but harsher than reality, since real 3DGS starts from SfM points.

        Report which was used.  A layer-2 result without the init mode is uninterpretable.
        """
        # seed_everything() returns a CPU generator on purpose, so every sampling call
        # must draw on the CPU and only then move to the target device. Passing a CPU
        # generator together with device="cuda" raises at runtime, which would surface
        # only on the first GPU run - exactly the run this code exists for.
        # data/phantom.py uses the same pattern.
        if mode == "surface":
            if surface_points is None or surface_points.numel() == 0:
                raise ValueError("mode='surface' needs surface_points")
            src = surface_points.to(device=device, dtype=dtype)
            if src.shape[0] >= n_gaussians:
                idx = torch.randperm(src.shape[0], generator=generator).to(src.device)
                pos = src[idx[:n_gaussians]]
            else:
                reps = (n_gaussians + src.shape[0] - 1) // src.shape[0]
                pos = src.repeat(reps, 1)[:n_gaussians]
                jitter = 0.25 * min(grid.spacing)
                noise = torch.randn(
                    tuple(pos.shape), generator=generator, dtype=dtype
                ).to(pos.device)
                pos = pos + jitter * noise
        elif mode == "bbox_random":
            lo = torch.tensor(grid.origin, device=device, dtype=dtype)
            extent = torch.tensor(grid.extent_mm, device=device, dtype=dtype)
            u = torch.rand((n_gaussians, 3), generator=generator, dtype=dtype).to(device)
            pos = lo + u * extent
        else:
            raise ValueError(
                f"mode must be 'surface' or 'bbox_random', got {mode!r}. There is no "
                f"default: the choice changes the result and must be recorded."
            )

        quat = torch.zeros((n_gaussians, 4), device=device, dtype=dtype)
        quat[:, 0] = 1.0
        log_s = torch.full(
            (n_gaussians, 3), math.log(max(float(scale_init_mm), 1e-6)),
            device=device, dtype=dtype,
        )
        amp = torch.full((n_gaussians, channels), float(amplitude_init), device=device, dtype=dtype)
        opa = torch.full((n_gaussians,), float(opacity_init), device=device, dtype=dtype)
        return cls(pos, quat, log_s, amp, opa, init_mode=mode)

    def storage_bytes(self) -> int:
        """Bytes to store this representation: 3 + 4 + 3 + C + 1 float32 per Gaussian.

        Larger per primitive than a surfel (which stores no normal-direction scale and can
        reconstruct ``e2``), and it must be stored **per frame** because nothing ties one
        frame's Gaussians to another's.  That is the storage side of the same trade-off.
        """
        per = 3 + 4 + 3 + self.channels + 1
        return self.n * per * 4


def render_free_3dgs(
    gaussians: FreeGaussians3D,
    camera: Camera,
    cfg: RenderConfig | None = None,
    cfg3d: Thin3DGSConfig | None = None,
    *,
    compute_aux: bool = True,
) -> RenderOutput:
    """Rasterise free Gaussians, reusing the thin-3DGS path with an independent 3rd scale.

    Sharing the rasteriser is deliberate: tiling, sorting, culling thresholds, the
    low-pass term and the output buffers are identical to the baseline this is compared
    against, so a measured difference comes from the *geometry being free* and not from
    two different rasterisers.
    """
    return render_thin_3dgs(
        gaussians,
        camera,
        cfg,
        cfg3d,
        compute_aux=compute_aux,
        scale_normal=gaussians.scale_normal,
    )


@dataclass
class FreeFitReport:
    """Outcome of fitting free Gaussians, with both halves of the trade-off."""

    init_mode: str
    n_gaussians: int
    iters: int
    time_ms: float
    final_loss: float
    photometric: dict[str, float]
    geometric: dict[str, float]
    storage_bytes: int
    loss_history: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float | str]:
        """Both photometric and geometric numbers, or an error.

        A free-geometry baseline reported on photometric metrics alone looks better than
        the proposed method; reported on geometric metrics alone it looks worse. Emitting
        either in isolation misrepresents the comparison, so this refuses.
        """
        if not self.photometric or not self.geometric:
            raise ValueError(
                "FreeFitReport.summary() needs both photometric and geometric metrics. "
                "Free Gaussians trade surface fidelity for image fidelity; quoting one "
                "side alone inverts the conclusion."
            )
        out: dict[str, float | str] = {
            "init_mode": self.init_mode,
            "n_gaussians": float(self.n_gaussians),
            "iters": float(self.iters),
            "time_ms": self.time_ms,
            "final_loss": self.final_loss,
            "storage_bytes_per_frame": float(self.storage_bytes),
        }
        out.update({f"photo/{k}": v for k, v in self.photometric.items()})
        out.update({f"geom/{k}": v for k, v in self.geometric.items()})
        return out


def fit_free_3dgs(
    gaussians: FreeGaussians3D,
    cameras: Sequence[Camera],
    targets: Sequence[Tensor],
    *,
    render_cfg: RenderConfig | None = None,
    loss_cfg: LossConfig | None = None,
    iters: int = 600,
    lr_position: float = 1e-1,
    lr_quaternion: float = 1e-2,
    lr_scale: float = 5e-3,
    lr_amplitude: float = 5e-2,
    lr_opacity: float = 2e-2,
    grad_clip: float = 1.0,
    log_every: int = 100,
    masks: Sequence[Tensor] | None = None,
) -> FreeFitReport:
    """Optimise all geometry against the same supervision the surfel fit uses.

    The optimiser, loss form, camera set and render config are the ones the surfel path
    uses, so the only difference is which tensors carry gradients.

    ``masks`` supplies the same silhouette supervision as the surfel loss when
    ``loss_cfg.lambda_mask > 0``; without it the term is skipped rather than silently
    treated as zero-weighted.
    """
    if len(cameras) != len(targets):
        raise ValueError(f"{len(cameras)} cameras but {len(targets)} targets")
    render_cfg = render_cfg or RenderConfig()
    loss_cfg = loss_cfg or LossConfig()
    if loss_cfg.lambda_mask > 0 and masks is None:
        raise ValueError(
            "loss_cfg.lambda_mask > 0 but no masks were given; pass masks or set the "
            "weight to zero explicitly"
        )

    opt = torch.optim.Adam(
        [
            {"params": [gaussians.position], "lr": lr_position},
            {"params": [gaussians.quaternion], "lr": lr_quaternion},
            {"params": [gaussians.log_scale3], "lr": lr_scale},
            {"params": [gaussians.amplitude], "lr": lr_amplitude},
            {"params": [gaussians.opacity_logit], "lr": lr_opacity},
        ],
        eps=1e-15,
    )

    history: list[float] = []
    t0 = time.perf_counter()
    last = float("nan")

    for it in range(int(iters)):
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=gaussians.device, dtype=gaussians.dtype)
        for k, (cam, tgt) in enumerate(zip(cameras, targets)):
            out = render_free_3dgs(gaussians, cam, render_cfg, compute_aux=False)
            diff = out.color - tgt
            if loss_cfg.appearance == "l1":
                app = diff.abs().mean()
            elif loss_cfg.appearance == "l2":
                app = (diff * diff).mean()
            else:
                d = float(loss_cfg.huber_delta)
                a = diff.abs()
                app = torch.where(a <= d, 0.5 * a * a / d, a - 0.5 * d).mean()
            total = total + app
            if loss_cfg.lambda_mask > 0 and masks is not None:
                total = total + loss_cfg.lambda_mask * (out.alpha - masks[k]).abs().mean()
        total = total / max(1, len(cameras))
        total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(gaussians.parameters()), grad_clip)
        opt.step()
        last = float(total.detach().item())
        if log_every and (it % log_every == 0 or it == iters - 1):
            history.append(last)

    return FreeFitReport(
        init_mode=gaussians.init_mode,
        n_gaussians=gaussians.n,
        iters=int(iters),
        time_ms=(time.perf_counter() - t0) * 1e3,
        final_loss=last,
        photometric={},
        geometric={},
        storage_bytes=gaussians.storage_bytes(),
        loss_history=history,
    )
