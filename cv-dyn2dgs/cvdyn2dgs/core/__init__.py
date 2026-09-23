"""Grid geometry, configuration, determinism and timing."""

from .config import (
    ChanVeseConfig,
    FitConfig,
    LossConfig,
    PipelineConfig,
    PRESETS,
    RenderConfig,
    ResidualConfig,
    SurfelConfig,
    get_preset,
)
from .grid import Grid, trilinear_sample, trilinear_sample_vector
from .runtime import (
    StageTimer,
    Stopwatch,
    append_csv,
    describe_environment,
    load_json,
    resolve_device,
    save_json,
    seed_everything,
    synchronize,
)

__all__ = [
    "ChanVeseConfig",
    "FitConfig",
    "Grid",
    "LossConfig",
    "PRESETS",
    "PipelineConfig",
    "RenderConfig",
    "ResidualConfig",
    "StageTimer",
    "Stopwatch",
    "SurfelConfig",
    "append_csv",
    "describe_environment",
    "get_preset",
    "load_json",
    "resolve_device",
    "save_json",
    "seed_everything",
    "synchronize",
    "trilinear_sample",
    "trilinear_sample_vector",
]
