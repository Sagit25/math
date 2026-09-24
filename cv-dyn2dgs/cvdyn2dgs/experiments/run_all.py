"""Run the whole experimental programme and write JSON/CSV results.

Ordering is deliberate: the theory checks run **first**.  If the discretisation does not
reproduce the predicted convergence rates there is no point measuring quality, and the
cheap checks catch sign-convention and spacing bugs that would otherwise surface as
mysteriously bad Dice scores.

Every result file records :func:`describe_environment` output, because all timing claims
are hardware dependent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from ..baselines import DYNA3DGR_COMPARISON, get_baseline, list_baselines
from ..core.runtime import append_csv, describe_environment, resolve_device, save_json, seed_everything
from ..data.phantom import Phantom4D, PhantomConfig, make_phantom
from .ablations import run_all_ablations, spacing_ablation
from .common import cost_quality_comparison, evaluate_all, make_eval_cameras
from .research_questions import (
    progressive_development,
    rq1_warm_start,
    rq2_projection_vs_independent,
    rq3_storage,
    rq4_playback_fps,
    rq5_vs_mesh,
    rq6_disk_vs_thin_ellipsoid,
)
from .theory_checks import run_all_checks

__all__ = ["main", "run_everything"]

_HEADLINE_COLUMNS = [
    "name",
    "dice",
    "hd95_mm",
    "e_surf_mm",
    "iou",
    "boundary_f",
    "depth_rmse_mm",
    "normal_deg",
    "psnr_roi_db",
    "ssim_roi",
    "flicker",
    "compression_ratio",
    "precompute_ms",
    "p95_frame_ms",
    "mean_fps",
    "meets_30fps",
]


def _phantom(size: str, device, dtype) -> Phantom4D:
    presets = {
        "tiny": PhantomConfig(shape=(48, 48, 10), n_frames=6, n_papillary=1),
        "small": PhantomConfig(shape=(64, 64, 16), n_frames=10),
        "medium": PhantomConfig(shape=(96, 96, 16), n_frames=20),
        "large": PhantomConfig(shape=(128, 128, 20), n_frames=30),
    }
    if size not in presets:
        raise ValueError(f"unknown phantom size {size!r}; choose from {sorted(presets)}")
    return make_phantom(presets[size], device=device, dtype=dtype)


def run_everything(
    out_dir: str | Path = "results",
    *,
    size: str = "small",
    device: str = "auto",
    seed: int = 0,
    baselines: Sequence[str] | None = None,
    skip: Sequence[str] = (),
    verbose: bool = True,
) -> dict[str, object]:
    """Execute the full programme and write results under ``out_dir``.

    ``skip`` accepts any of ``theory``, ``baselines``, ``rq1``..``rq6``, ``ablations``,
    ``spacing``, ``progressive``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)
    gen = seed_everything(seed)
    env = describe_environment(dev)
    save_json(out / "environment.json", env)
    if verbose:
        print(f"device={dev}  torch={env['torch']}  phantom={size}")

    ph = _phantom(size, dev, torch.float32)
    save_json(
        out / "phantom_config.json",
        {
            "config": asdict(ph.config),
            "grid": {"shape": list(ph.grid.shape), "spacing": list(ph.grid.spacing)},
            "ground_truth_ef_percent": ph.ef_percent(),
            "analytic_volumes_ml": ph.analytic_volumes_ml(),
            "ed_index": ph.ed_index(),
            "es_index": ph.es_index(),
        },
    )

    results: dict[str, object] = {"environment": env}

    # ---- 1. theory checks first ------------------------------------------
    if "theory" not in skip:
        if verbose:
            print("\n=== theory checks ===")
        checks = run_all_checks(verbose=verbose)
        results["theory_checks"] = [c.to_dict() for c in checks]
        save_json(out / "theory_checks.json", results["theory_checks"])
        n_pass = sum(1 for c in checks if c.passed)
        if verbose and n_pass < len(checks):
            print(
                f"WARNING: {len(checks) - n_pass} theory check(s) did not pass. "
                "Read them before trusting the quality numbers below."
            )

    # ---- 2. baseline table -----------------------------------------------
    if "baselines" not in skip:
        names = list(baselines) if baselines else [
            "cv-dyn2dgs",
            "independent-2dgs",
            "mesh-only",
            "thin-3dgs",
            "param-copy",
            "closest-point",
            "normal-no-residual",
            # Layer-1 oracle: the exact analytic surface. Not optional for attribution -
            # without it, segmentation error and representation error are confounded and
            # no RQ5/RQ6 number can be assigned to either. Only the phantom supplies it
            # exactly; on real data labels exist at ED and ES only.
            "source-oracle",
            # These declare an axis this runner cannot realise on a phantom, so they raise
            # and are recorded as errors rather than silently producing cv-dyn2dgs numbers
            # under another name. Listing them keeps the gap visible in baselines.json.
            "source-mask",
            "free-3dgs-surface",
            "gaussian-surfel-exact",
        ]
        if verbose:
            print(f"\n=== baselines: {names} ===")
        cams = make_eval_cameras(ph.grid, device=dev)
        rows: list[dict[str, object]] = []
        details: dict[str, object] = {}
        evaluated: dict[str, object] = {}
        for name in names:
            if verbose:
                print(f"--- {name}")
            spec = get_baseline(name)
            try:
                res = evaluate_all(spec, ph, cameras=cams, generator=gen, verbose=False)
                row = res.headline()
                details[name] = res.flat()
                evaluated[name] = res
            except NotImplementedError as exc:
                # Declared but not realisable here. Distinguished from a crash on purpose:
                # this is a known gap in the runner, not a bug in the method.
                row = {"name": name, "not_measured": str(exc)}
                details[name] = row
                evaluated[name] = None
                if verbose:
                    print(f"    NOT MEASURED: {exc}")
            except Exception as exc:  # noqa: BLE001
                row = {"name": name, "error": f"{type(exc).__name__}: {exc}"}
                details[name] = row
                evaluated[name] = None
            rows.append(row)
            append_csv(out / "baselines.csv", row, columns=_HEADLINE_COLUMNS)
            if verbose:
                print(f"    {row}")
        results["baselines"] = rows
        save_json(out / "baselines.json", {"headline": rows, "detail": details})

        # Quality against cost, with the two cost currencies kept apart and the dominance
        # verdict. Only baselines that actually produced an EvaluationResult take part; a
        # baseline that raised has nothing to place on the frontier.
        ok = [r for n, r in evaluated.items() if r is not None]
        if ok:
            cq = cost_quality_comparison(ok)
            save_json(out / "cost_quality.json", cq)
            results["cost_quality"] = cq
            if verbose:
                print(f"\n=== cost/quality frontier: {cq['frontier']} ===")
                if cq["dominated"]:
                    print(f"    dominated: {cq['dominated']}")

    # ---- 3. research questions -------------------------------------------
    rq_fns = {
        "rq1": rq1_warm_start,
        "rq2": rq2_projection_vs_independent,
        "rq3": rq3_storage,
        "rq4": rq4_playback_fps,
        "rq5": rq5_vs_mesh,
        "rq6": rq6_disk_vs_thin_ellipsoid,
    }
    rq_out: dict[str, object] = {}
    for key, fn in rq_fns.items():
        if key in skip:
            continue
        if verbose:
            print(f"\n=== {key} ===")
        try:
            r = fn(ph) if key == "rq1" else fn(ph, generator=gen)  # type: ignore[operator]
            rq_out[key] = r.to_dict()
            if verbose:
                print(f"    {r.verdict}")
                for c in r.caveats:
                    print(f"    caveat: {c}")
        except Exception as exc:  # noqa: BLE001
            rq_out[key] = {"error": f"{type(exc).__name__}: {exc}"}
            if verbose:
                print(f"    ERROR: {exc}")
    if rq_out:
        results["research_questions"] = rq_out
        save_json(out / "research_questions.json", rq_out)

    # ---- 4. progressive development ---------------------------------------
    if "progressive" not in skip:
        if verbose:
            print("\n=== progressive development (v1 -> v2 -> v3) ===")
        try:
            prog = progressive_development(ph, generator=gen)
            results["progressive"] = prog.to_dict()
            save_json(out / "progressive.json", prog.to_dict())
            if verbose:
                print(f"    {prog.verdict}")
        except Exception as exc:  # noqa: BLE001
            results["progressive"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- 5. ablations -----------------------------------------------------
    if "ablations" not in skip:
        if verbose:
            print("\n=== ablations ===")
        abl = run_all_ablations(ph, generator=gen, verbose=verbose)
        results["ablations"] = abl
        save_json(out / "ablations.json", abl)
        for name, row in abl.items():
            append_csv(out / "ablations.csv", {"name": name, **row}, columns=["name", "tests", *_HEADLINE_COLUMNS[1:]])

    if "spacing" not in skip:
        if verbose:
            print("\n=== spacing-awareness ablation (contribution 1) ===")
        sp = spacing_ablation()
        results["spacing_ablation"] = sp
        save_json(out / "spacing_ablation.json", sp)
        if verbose:
            for k, v in sp.items():
                print(f"    {k}: {v}")

    # ---- 6. related work not reproduced ----------------------------------
    results["dyna3dgr"] = DYNA3DGR_COMPARISON
    save_json(out / "dyna3dgr_structural_comparison.json", DYNA3DGR_COMPARISON)

    save_json(out / "all_results.json", results)
    if verbose:
        print(f"\nwrote results to {out.resolve()}")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cvdyn2dgs-experiments",
        description="Run the CV-Dyn2DGS experimental programme on the synthetic phantom.",
    )
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument(
        "--size", default="small", choices=["tiny", "small", "medium", "large"],
        help="phantom size preset",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--baselines", nargs="*", default=None,
        help=f"subset of {list_baselines()}",
    )
    ap.add_argument(
        "--skip", nargs="*", default=[],
        help="any of: theory baselines rq1 rq2 rq3 rq4 rq5 rq6 progressive ablations spacing",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    run_everything(
        args.out,
        size=args.size,
        device=args.device,
        seed=args.seed,
        baselines=args.baselines,
        skip=args.skip,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
