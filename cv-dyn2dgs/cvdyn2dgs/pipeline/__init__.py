"""End-to-end pipeline: precompute the stored representation, then play it back."""

from .fitting import FitReport, ViewTarget, fit_appearance
from .playback import (
    FrameTiming,
    PlaybackEngine,
    PlaybackReport,
    compare_projection_modes,
    measure_playback,
)
from .precompute import (
    FrameRecord,
    PrecomputedModel,
    make_supervision_cameras,
    precompute,
)
from .storage_io import (
    PackedBand,
    load_model,
    model_storage_report,
    pack_narrow_band,
    save_model,
    unpack_narrow_band,
)

__all__ = [
    "FitReport",
    "FrameRecord",
    "FrameTiming",
    "PackedBand",
    "PlaybackEngine",
    "PlaybackReport",
    "PrecomputedModel",
    "ViewTarget",
    "compare_projection_modes",
    "fit_appearance",
    "load_model",
    "make_supervision_cameras",
    "measure_playback",
    "model_storage_report",
    "pack_narrow_band",
    "precompute",
    "save_model",
    "unpack_narrow_band",
]
