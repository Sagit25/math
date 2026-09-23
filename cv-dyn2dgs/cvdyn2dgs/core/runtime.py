"""Determinism, device handling, wall-clock timing and result IO.

The runtime helpers here exist mainly so that the timing numbers reported for
RQ1 (Chan-Vese convergence), RQ4 (playback FPS, proposal Eq. 34) and the
preprocessing budget (proposal Eq. 7-9) are measured consistently:

* every GPU timing block synchronises before reading the clock, otherwise CUDA
  kernel launches are asynchronous and the measurement is meaningless;
* every stage is reported separately (``T_surface``, ``T_project``,
  ``T_orient``, ``T_raster``) so that Eq. (34) can be decomposed rather than
  reported as a single opaque total.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

__all__ = [
    "seed_everything",
    "resolve_device",
    "synchronize",
    "Stopwatch",
    "StageTimer",
    "describe_environment",
    "save_json",
    "load_json",
    "append_csv",
]


# --------------------------------------------------------------------------- #
#  Determinism
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 0, *, deterministic: bool = True) -> torch.Generator:
    """Seed Python / torch RNGs and return a seeded CPU generator.

    The returned generator is threaded explicitly through the phantom sampler and
    the canonical surfel initialiser so that those stages are reproducible even
    if third-party code touches the global RNG.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Deterministic scatter-add keeps the rasteriser's backward pass stable.
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def resolve_device(spec: str | torch.device | None = None) -> torch.device:
    """Resolve ``"auto"`` / ``None`` / explicit specs to a concrete device."""
    if isinstance(spec, torch.device):
        return spec
    if spec is None or spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def synchronize(device: torch.device | None = None) -> None:
    """Block until queued device work finished (no-op on CPU)."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        with contextlib.suppress(Exception):
            torch.mps.synchronize()


# --------------------------------------------------------------------------- #
#  Timing
# --------------------------------------------------------------------------- #
class Stopwatch:
    """Context manager measuring device-synchronised wall time in milliseconds.

    >>> with Stopwatch(device) as sw:      # doctest: +SKIP
    ...     out = render(...)
    >>> sw.ms                              # doctest: +SKIP
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.ms: float = float("nan")

    def __enter__(self) -> "Stopwatch":
        synchronize(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        synchronize(self.device)
        self.ms = (time.perf_counter() - self._t0) * 1e3


@dataclass
class StageTimer:
    """Accumulates per-stage timings so Eq. (34) can be reported term by term.

    ``T_frame = T_surface + T_project + T_orient + T_raster < 33.3 ms``
    """

    device: torch.device | None = None
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[float]] = field(default_factory=dict)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        synchronize(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            dt = (time.perf_counter() - t0) * 1e3
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1
            self.samples.setdefault(name, []).append(dt)

    def mean_ms(self, name: str) -> float:
        n = self.counts.get(name, 0)
        return self.totals[name] / n if n else float("nan")

    def percentile_ms(self, name: str, q: float = 95.0) -> float:
        """Latency percentile; the proposal reports mean *and* p95 (§8.3)."""
        vals = sorted(self.samples.get(name, []))
        if not vals:
            return float("nan")
        if len(vals) == 1:
            return vals[0]
        pos = (q / 100.0) * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "total_ms": self.totals[name],
                "calls": float(self.counts[name]),
                "mean_ms": self.mean_ms(name),
                "p95_ms": self.percentile_ms(name, 95.0),
                "max_ms": max(self.samples[name]),
            }
            for name in sorted(self.totals)
        }

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()
        self.samples.clear()


# --------------------------------------------------------------------------- #
#  Environment / IO
# --------------------------------------------------------------------------- #
def describe_environment(device: torch.device | None = None) -> dict[str, Any]:
    """Record the hardware/software context of a measurement.

    Timing claims are hardware dependent (theory §11.3 lists the 33.3 ms frame
    budget as "to be verified on real hardware"), so every result file carries
    this block.
    """
    dev = resolve_device(device)
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device_type": dev.type,
        "cpu_count": os.cpu_count(),
    }
    if dev.type == "cuda":
        info["cuda"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(dev)
        props = torch.cuda.get_device_properties(dev)
        info["gpu_total_mem_gb"] = round(props.total_memory / 1024**3, 2)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
    return info


def save_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")
    return p


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def append_csv(path: str | Path, row: dict[str, Any], *, columns: list[str] | None = None) -> Path:
    """Append one row to a CSV, writing the header on first use."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or sorted(row)
    new = not p.exists()
    with p.open("a", encoding="utf-8") as fh:
        if new:
            fh.write(",".join(cols) + "\n")
        fh.write(",".join(_csv_cell(row.get(c, "")) for c in cols) + "\n")
    return p


def _csv_cell(v: Any) -> str:
    s = "" if v is None else str(_json_default(v) if isinstance(v, torch.Tensor) else v)
    return f'"{s}"' if ("," in s or '"' in s) else s


def _json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() > 1 else obj.item()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")
