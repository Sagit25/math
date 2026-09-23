"""Command-line entry point: ``cvdyn2dgs <subcommand>``.

Subcommands
-----------
``smoke``       **run this first** — staged diagnostic, one run reports every failure
``info``        environment and available presets/baselines
``theory``      run the numerical theory checks only (fast, no rendering)
``demo``        precompute + play back one phantom, print the stage breakdown
``baselines``   evaluate a set of baselines and write a comparison table
``experiments`` the full programme (theory checks, baselines, RQ1-RQ6, ablations)
``acdc``        precompute one real ACDC patient (needs nibabel and the dataset)
``compare``     the external comparison design: four layers, and what is still missing
``datasets``    where the data comes from, and whether a copy is the *right* copy
``viewpoint``   storage vs viewpoint count - the pre-rendered-video break-even
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


from .core.config import PRESETS, get_preset
from .core.runtime import describe_environment, resolve_device, save_json, seed_everything

__all__ = ["main"]


def _cmd_smoke(args: argparse.Namespace) -> int:
    from .smoke import main as smoke_main

    argv = ["--device", str(args.device), "--seed", str(args.seed)]
    if args.full:
        argv.append("--full")
    if args.quiet:
        argv.append("--quiet")
    return smoke_main(argv)


def _cmd_info(args: argparse.Namespace) -> int:
    from .baselines import list_baselines

    dev = resolve_device(args.device)
    env = describe_environment(dev)
    print("environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print(f"\npresets:   {sorted(PRESETS)}")
    print(f"baselines: {list_baselines()}")
    print("\nsign convention: phi > 0 inside the heart (proposal Eq. 1)")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Print the comparison design, including the comparisons that have NOT been run."""
    import json

    from .baselines import INTERNAL_BASELINES, baselines_in_layer

    if args.adapter_template:
        from .experiments.external import write_adapter_template

        p = write_adapter_template(args.adapter_template, args.method)
        print(f"wrote {p} and the expected directory layout next to it")
        print("fill in meta.json - 'commit' must match external/manifest.json")
        return 0

    names = {
        1: "surface source",
        2: "representation primitive",
        3: "temporal model",
        4: "storage and playback",
    }
    manifest = Path(__file__).resolve().parent.parent / "external" / "manifest.json"
    targets: list[dict] = []
    if manifest.exists():
        with manifest.open(encoding="utf-8") as fh:
            targets = json.load(fh)["targets"]
    else:
        print(f"note: {manifest} not found; external targets unavailable\n")

    layers = [args.layer] if args.layer else [1, 2, 3, 4]
    for layer in layers:
        print(f"\nLayer {layer} - {names[layer]}")
        print("-" * 74)
        internal = baselines_in_layer(layer)
        print(f"  in-pipeline baselines ({len(internal)}):")
        for b in internal:
            print(f"    {b}")
        ext = [t for t in targets if t.get("layer") == layer]
        print(f"  external targets ({len(ext)}) - NONE OF THESE HAS BEEN RUN:")
        for t in ext:
            state = "no public code" if not t.get("repo") else "pinned"
            vend = "" if t.get("redistributable") else ", do not vendor"
            print(f"    {t['key']:<22} [{state}{vend}]")

    print(f"\n{len(INTERNAL_BASELINES)} of the baselines are configurations of this")
    print("pipeline only. See paper/COMPARISON_TARGETS.md for the full design, and")
    print("scripts/fetch_external.py --list to fetch the external code at pinned commits.")
    print("\nStatus: no comparison in this table has been measured.")
    return 0


def _cmd_datasets(args: argparse.Namespace) -> int:
    """Delegate to scripts/fetch_datasets.py, which needs no torch.

    Loaded by path rather than imported so that dataset verification keeps working in an
    environment where this package's own dependencies are not installed - which is the
    situation someone is in right before they first obtain the data.
    """
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts" / "fetch_datasets.py"
    if not script.exists():
        print(f"{script} not found", file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("_fetch_datasets", script)
    if spec is None or spec.loader is None:
        print(f"could not load {script}", file=sys.stderr)
        return 2
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    argv: list[str] = []
    if args.list:
        argv = ["--list"]
    elif args.how:
        argv = ["--how", args.how]
    elif args.verify:
        if not args.path:
            print("--verify needs --path", file=sys.stderr)
            return 2
        argv = ["--verify", args.verify, "--path", args.path]
        if args.max_patients:
            argv += ["--max-patients", str(args.max_patients)]
    else:
        argv = ["--list"]
    return int(mod.main(argv))


def _cmd_viewpoint(args: argparse.Namespace) -> int:
    """Break-even viewpoint count against pre-rendered video."""
    from .metrics.viewpoint import (
        VideoCodecTarget,
        viewpoint_breakeven,
        viewpoint_table,
    )

    if (args.video_bytes_per_view is None) == (args.assume_bitrate_kbps is None):
        print(
            "give exactly one of --video-bytes-per-view (measured) or "
            "--assume-bitrate-kbps (declared). There is no default bitrate: guessing "
            "the competitor's number would be inventing a result.",
            file=sys.stderr,
        )
        return 2

    if args.video_bytes_per_view is not None:
        per_view = float(args.video_bytes_per_view)
        target = VideoCodecTarget.measured(
            total_bytes=int(per_view),
            n_viewpoints=1,
            codec="unspecified",
            command="supplied via --video-bytes-per-view",
        )
    else:
        if not args.basis.strip():
            print(
                "--assume-bitrate-kbps requires --basis explaining why that bitrate is "
                "plausible.",
                file=sys.stderr,
            )
            return 2
        target = VideoCodecTarget.assumed(
            bitrate_kbps=args.assume_bitrate_kbps,
            n_frames=args.frames,
            fps_playback=args.playback_fps,
            n_viewpoints=1,
            codec="h265",
            basis=args.basis,
        )
        per_view = float(target.total_bytes)

    budget = viewpoint_breakeven(
        ours_bytes=int(args.ours_bytes), video_bytes_per_viewpoint=per_view
    )
    print(f"provenance: {target.provenance}")
    print(f"ours:  {args.ours_bytes / 1024**2:.3f} MB (flat in viewpoint count)")
    print(f"video: {per_view / 1024**2:.3f} MB per viewpoint")
    print(f"\nbreak-even: {budget.breakeven_viewpoints:.2f} viewpoints "
          f"(integer {budget.breakeven_viewpoints_int})")
    print(f"verdict: {budget.verdict}\n")
    print(f"{'views':>6}  {'ours MB':>9}  {'video MB':>9}  winner")
    for row in viewpoint_table(
        ours_bytes=int(args.ours_bytes), video_bytes_per_viewpoint=per_view
    ):
        print(f"{row['viewpoints']:>6.0f}  {row['ours_mb']:>9.3f}  "
              f"{row['video_mb']:>9.3f}  {row['winner']}")
    if args.out:
        save_json(Path(args.out) / "viewpoint_breakeven.json",
                  {**budget.to_dict(), **target.to_dict()})
    return 0


def _cmd_theory(args: argparse.Namespace) -> int:
    from .experiments.theory_checks import run_all_checks

    seed_everything(args.seed)
    results = run_all_checks(verbose=True)
    if args.out:
        save_json(Path(args.out) / "theory_checks.json", [r.to_dict() for r in results])
    failed = [r.name for r in results if not r.passed]
    if failed:
        print(f"\nnot passed: {failed}", file=sys.stderr)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .data.phantom import PhantomConfig, make_phantom
    from .experiments.common import run_pipeline_on_phantom
    from .pipeline.playback import measure_playback
    from .pipeline.storage_io import model_storage_report

    dev = resolve_device(args.device)
    gen = seed_everything(args.seed)
    cfg = get_preset(args.preset)
    if args.surfels:
        cfg.surfel.n_surfels = int(args.surfels)

    ph = make_phantom(
        PhantomConfig(n_frames=int(args.frames), shape=tuple(args.shape)),  # type: ignore[arg-type]
        device=dev,
        generator=gen,
    )
    print(f"phantom: {ph.n_frames} frames, grid {ph.grid}, GT EF {ph.ef_percent():.1f}%")

    model, cams, _ = run_pipeline_on_phantom(ph, cfg, generator=gen, verbose=True)

    print("\n--- precompute summary ---")
    for k, v in sorted(model.summary().items()):
        print(f"  {k}: {v}")

    print("\n--- storage (Eq. 36-38) ---")
    rep = model_storage_report(model)
    primary = rep.get("primary", {})
    if isinstance(primary, dict):
        for k in ("s_full_mb", "s_ours_mb", "compression_ratio", "surface_share", "residual_share"):
            print(f"  {k}: {primary.get(k)}")

    print("\n--- playback (Eq. 34) ---")
    pb = measure_playback(model, cams.eval, loops=2, compute_aux=False)
    for k, v in pb.summary().items():
        print(f"  {k}: {v}")

    if args.out:
        out = Path(args.out)
        save_json(out / "demo_summary.json", {
            "config": cfg.to_dict(),
            "phantom": asdict(ph.config),
            "model": model.summary(),
            "storage": rep,
            "playback": pb.summary(),
        })
        if args.save_model:
            from .pipeline.storage_io import save_model

            info = save_model(model, out / "model.pt")
            print(f"\nwrote model: {info['path']} ({info['file_mb']:.2f} MB)")
    return 0


def _cmd_baselines(args: argparse.Namespace) -> int:
    from .baselines import get_baseline, list_baselines
    from .data.phantom import PhantomConfig, make_phantom
    from .experiments.common import evaluate_all, make_eval_cameras
    from .core.runtime import append_csv

    dev = resolve_device(args.device)
    gen = seed_everything(args.seed)
    names = args.names or list_baselines()
    ph = make_phantom(
        PhantomConfig(n_frames=int(args.frames), shape=tuple(args.shape)),  # type: ignore[arg-type]
        device=dev,
        generator=gen,
    )
    cams = make_eval_cameras(ph.grid, device=dev)

    rows = []
    for name in names:
        print(f"--- {name}")
        try:
            res = evaluate_all(get_baseline(name), ph, cameras=cams, generator=gen)
            row = res.headline()
        except Exception as exc:  # noqa: BLE001
            row = {"name": name, "error": f"{type(exc).__name__}: {exc}"}
        print(f"    {row}")
        rows.append(row)
        if args.out:
            append_csv(Path(args.out) / "baselines.csv", row)
    if args.out:
        save_json(Path(args.out) / "baselines.json", rows)
    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    from .experiments.run_all import run_everything

    run_everything(
        args.out or "results",
        size=args.size,
        device=args.device,
        seed=args.seed,
        baselines=args.baselines,
        skip=args.skip,
        verbose=not args.quiet,
    )
    return 0


def _cmd_acdc(args: argparse.Namespace) -> int:
    from .data.real import load_acdc_patient
    from .experiments.common import initial_levelset_from_mask, make_eval_cameras
    from .levelset.sdf import signed_distance_from_mask  # noqa: F401 - documented alternative
    from .pipeline.precompute import precompute
    from .pipeline.storage_io import model_storage_report

    dev = resolve_device(args.device)
    gen = seed_everything(args.seed)
    cfg = get_preset(args.preset)

    seq = load_acdc_patient(args.patient_dir, device=dev)
    print(
        f"{seq.patient_id}: {seq.n_frames} frames, grid {seq.grid}, "
        f"labelled frames {seq.labelled_frames}, label EF {seq.ef_percent():.1f}%"
    )
    if seq.ed_index not in seq.masks:
        raise SystemExit(
            f"no ED label for {seq.patient_id}; cannot initialise frame 0. "
            "Proposal §2.12 step 1 needs either a label or a manual interior seed."
        )

    phi0 = initial_levelset_from_mask(seq.masks[seq.ed_index], seq.grid).to(dev)
    order = [(seq.ed_index + t) % seq.n_frames for t in range(seq.n_frames)]
    images = [seq.images[t] for t in order]

    cams = make_eval_cameras(seq.grid, device=dev)
    model = precompute(images, phi0, seq.grid, cfg, generator=gen, cameras=cams.fit, verbose=True)

    print("\n--- summary ---")
    for k, v in sorted(model.summary().items()):
        print(f"  {k}: {v}")
    print("\nNOTE: ground truth exists at ED/ES only. Intermediate frames must be judged")
    print("      by appearance and temporal metrics, never by a segmentation score.")

    if args.out:
        save_json(
            Path(args.out) / f"{seq.patient_id}_summary.json",
            {"model": model.summary(), "storage": model_storage_report(model)},
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cvdyn2dgs", description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("smoke", help="staged diagnostic - run this first")
    p.add_argument("--full", action="store_true", help="include end-to-end pipeline stages")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_smoke)

    p = sub.add_parser("info", help="environment, presets and baselines")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("theory", help="numerical theory checks")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_theory)

    p = sub.add_parser("demo", help="precompute + play back one phantom")
    p.add_argument("--preset", default="v3-adaptive", choices=sorted(PRESETS))
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--shape", type=int, nargs=3, default=[64, 64, 16])
    p.add_argument("--surfels", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--save-model", action="store_true")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("baselines", help="evaluate baselines and write a table")
    p.add_argument("names", nargs="*", default=None)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--shape", type=int, nargs=3, default=[64, 64, 16])
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_baselines)

    p = sub.add_parser("experiments", help="full experimental programme")
    p.add_argument("--out", default="results")
    p.add_argument("--size", default="small", choices=["tiny", "small", "medium", "large"])
    p.add_argument("--baselines", nargs="*", default=None)
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_experiments)

    p = sub.add_parser("acdc", help="precompute one ACDC patient (needs nibabel + data)")
    p.add_argument("patient_dir")
    p.add_argument("--preset", default="v3-adaptive", choices=sorted(PRESETS))
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_acdc)

    p = sub.add_parser(
        "compare",
        help="the external comparison design: layers, targets and what is missing",
    )
    p.add_argument("--layer", type=int, choices=[1, 2, 3, 4], default=None)
    p.add_argument(
        "--adapter-template",
        default=None,
        metavar="DIR",
        help="write the directory layout an external method should dump into",
    )
    p.add_argument("--method", default="METHOD-KEY", help="manifest key for the template")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser(
        "datasets",
        help="dataset provenance, access instructions, and layout/geometry verification",
    )
    p.add_argument("--list", action="store_true", help="all datasets and how to obtain them")
    p.add_argument("--how", metavar="KEY", help="access instructions for one dataset")
    p.add_argument("--verify", metavar="KEY", help="check a manually obtained copy")
    p.add_argument("--path", metavar="DIR", help="dataset root for --verify")
    p.add_argument("--max-patients", type=int, default=0)
    p.set_defaults(func=_cmd_datasets)

    p = sub.add_parser(
        "viewpoint",
        help="storage vs viewpoint count: the pre-rendered-video break-even",
    )
    p.add_argument("--ours-bytes", type=int, required=True)
    p.add_argument(
        "--video-bytes-per-view",
        type=float,
        default=None,
        help="measured bytes for ONE encoded viewpoint",
    )
    p.add_argument(
        "--assume-bitrate-kbps",
        type=float,
        default=None,
        help="instead of a measurement, declare a bitrate (recorded as an assumption)",
    )
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--playback-fps", type=float, default=24.0)
    p.add_argument("--basis", default="", help="why that bitrate is plausible (required)")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_viewpoint)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
