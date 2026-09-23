"""Appearance and temporal-stability metrics (proposal §8.3, Eq. 40).

Appearance
----------
PSNR / SSIM / NRMSE between the rendered projection and the observed MRI.  Proposal
§8.3 is explicit that these must be reported **both** over the whole image and
restricted to the heart ROI, because a large black background inflates PSNR to the
point of meaninglessness.  :func:`appearance_report` therefore always returns both,
and never a single headline number.

Temporal stability
------------------
Eq. (40) compares *rendered* frame-to-frame change against *observed* frame-to-frame
change:

.. math::
    E_{\\mathrm{flicker}} = \\frac{1}{T-1}\\sum_t
        \\bigl\\|(\\hat C_t - \\hat C_{t-1}) - (C^{\\mathrm{ref}}_t - C^{\\mathrm{ref}}_{t-1})\\bigr\\|_1 .

The subtraction of the reference difference is the important part: it means the
metric does not reward a model that is simply static.  A frozen renderer scores
badly here, as it should, while a renderer that reproduces the true motion scores
well even though its output changes a lot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "psnr",
    "ssim",
    "nrmse",
    "appearance_report",
    "flicker",
    "temporal_report",
]


def _masked_stats(x: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return x.reshape(-1)
    m = mask.to(torch.bool)
    if x.dim() == 3:  # (C, H, W)
        return x.permute(1, 2, 0)[m].reshape(-1)
    return x[m].reshape(-1)


def psnr(pred: Tensor, target: Tensor, *, data_range: float = 1.0, mask: Tensor | None = None) -> float:
    """Peak signal-to-noise ratio in dB."""
    a = _masked_stats(pred, mask)
    b = _masked_stats(target, mask)
    if a.numel() == 0:
        return float("nan")
    mse = float(((a - b) ** 2).mean().item())
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(float(data_range) ** 2 / mse)


def nrmse(pred: Tensor, target: Tensor, *, mask: Tensor | None = None) -> float:
    """Root-mean-square error normalised by the RMS of the target."""
    a = _masked_stats(pred, mask)
    b = _masked_stats(target, mask)
    if a.numel() == 0:
        return float("nan")
    denom = float(torch.sqrt((b * b).mean()).item())
    return float(torch.sqrt(((a - b) ** 2).mean()).item()) / max(denom, 1e-12)


def _gaussian_window(size: int, sigma: float, device, dtype) -> Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)  # (size, size)


def ssim(
    pred: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    mask: Tensor | None = None,
) -> float:
    """Structural similarity index with a Gaussian window.

    Standard formulation with :math:`k_1=0.01`, :math:`k_2=0.03`.  When ``mask`` is
    given the SSIM *map* is computed on the full image and then averaged over the
    mask - not computed on a cropped image - so that window statistics near the ROI
    border still see real neighbouring pixels rather than zero padding.
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    c, h, w = pred.shape
    dev, dt = pred.device, pred.dtype

    win = _gaussian_window(window_size, sigma, dev, dt).expand(c, 1, window_size, window_size)
    pad = window_size // 2

    def filt(x: Tensor) -> Tensor:
        return F.conv2d(x.unsqueeze(0), win, padding=pad, groups=c).squeeze(0)

    mu_x, mu_y = filt(pred), filt(target)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = filt(pred * pred) - mu_x2
    sig_y = filt(target * target) - mu_y2
    sig_xy = filt(pred * target) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    smap = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2)
    )
    smap = smap.mean(dim=0)  # average channels -> (H, W)

    if mask is None:
        return float(smap.mean().item())
    m = mask.to(torch.bool)
    if int(m.sum().item()) == 0:
        return float("nan")
    return float(smap[m].mean().item())


@dataclass
class AppearanceReport:
    """Whole-image and ROI appearance scores, kept separate on purpose."""

    psnr_full: float
    ssim_full: float
    nrmse_full: float
    psnr_roi: float
    ssim_roi: float
    nrmse_roi: float
    roi_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {
            "psnr_full_db": self.psnr_full,
            "ssim_full": self.ssim_full,
            "nrmse_full": self.nrmse_full,
            "psnr_roi_db": self.psnr_roi,
            "ssim_roi": self.ssim_roi,
            "nrmse_roi": self.nrmse_roi,
            "roi_fraction": self.roi_fraction,
        }


def appearance_report(
    rendered: Tensor,
    observed: Tensor,
    *,
    roi: Tensor | None = None,
    data_range: float = 1.0,
) -> AppearanceReport:
    """PSNR / SSIM / NRMSE, whole image and inside the ROI.

    ``rendered`` and ``observed`` are ``(C, H, W)``; ``roi`` is ``(H, W)`` boolean -
    typically the projected heart silhouette.
    """
    roi_frac = 0.0 if roi is None else float(roi.to(torch.float32).mean().item())
    return AppearanceReport(
        psnr_full=psnr(rendered, observed, data_range=data_range),
        ssim_full=ssim(rendered, observed, data_range=data_range),
        nrmse_full=nrmse(rendered, observed),
        psnr_roi=psnr(rendered, observed, data_range=data_range, mask=roi),
        ssim_roi=ssim(rendered, observed, data_range=data_range, mask=roi),
        nrmse_roi=nrmse(rendered, observed, mask=roi),
        roi_fraction=roi_frac,
    )


def flicker(
    rendered: Sequence[Tensor],
    reference: Sequence[Tensor],
    *,
    mask: Tensor | None = None,
) -> float:
    """:math:`E_{\\mathrm{flicker}}` of proposal Eq. (40).

    Parameters
    ----------
    rendered, reference:
        Sequences of ``T`` images, all the same shape, rendered / observed from the
        *same fixed view* (the metric is about temporal change, so the camera must
        not move between frames).

    Returns
    -------
    Mean over frames of the :math:`L^1` norm of the difference of differences.
    Lower is better; a temporally frozen renderer does **not** score 0 here, because
    the reference difference is subtracted.
    """
    if len(rendered) != len(reference):
        raise ValueError(f"length mismatch: {len(rendered)} vs {len(reference)}")
    if len(rendered) < 2:
        return 0.0
    total = 0.0
    for t in range(1, len(rendered)):
        d_pred = rendered[t] - rendered[t - 1]
        d_ref = reference[t] - reference[t - 1]
        diff = (d_pred - d_ref).abs()
        if mask is not None:
            m = mask.to(torch.bool)
            sel = diff.permute(1, 2, 0)[m] if diff.dim() == 3 else diff[m]
            total += float(sel.mean().item())
        else:
            total += float(diff.mean().item())
    return total / (len(rendered) - 1)


def temporal_report(
    rendered: Sequence[Tensor],
    reference: Sequence[Tensor],
    *,
    mask: Tensor | None = None,
) -> dict[str, float]:
    """Flicker plus the two raw temporal energies it is built from.

    Reporting ``rendered_change`` and ``reference_change`` next to the flicker score
    makes the degenerate solutions visible: a frozen model has
    ``rendered_change ~ 0`` while its flicker equals ``reference_change``.
    """
    e = flicker(rendered, reference, mask=mask)

    def mean_change(seq: Sequence[Tensor]) -> float:
        if len(seq) < 2:
            return 0.0
        acc = 0.0
        for t in range(1, len(seq)):
            d = (seq[t] - seq[t - 1]).abs()
            if mask is not None:
                m = mask.to(torch.bool)
                d = d.permute(1, 2, 0)[m] if d.dim() == 3 else d[m]
            acc += float(d.mean().item())
        return acc / (len(seq) - 1)

    ref_change = mean_change(reference)
    return {
        "e_flicker": e,
        "rendered_change": mean_change(rendered),
        "reference_change": ref_change,
        "flicker_relative": e / max(ref_change, 1e-12),
    }
