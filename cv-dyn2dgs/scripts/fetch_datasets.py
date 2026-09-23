#!/usr/bin/env python3
"""Obtain and, more importantly, *verify* the cardiac datasets.

Why this is not a downloader
----------------------------
The datasets this thesis needs cannot be cloned. ACDC, M&Ms, M&Ms-2 and the STACOM
motion benchmark each require individual registration and acceptance of a data-use
agreement. No script may bypass that and this one does not try; for gated datasets it
prints exactly what to do and stops.

What it does instead is the part that actually prevents wasted work:

1. **records provenance** - where each dataset comes from, under what terms, what must be
   cited, and which comparison it feeds;
2. **verifies a manually obtained copy** against the layout
   ``cvdyn2dgs/data/real.py`` really globs for, before any of the pipeline runs;
3. **checks the voxel geometry**, because a resampled copy silently invalidates the
   thesis's first contribution.

That third point is the one worth spelling out. Contribution 1 is the spacing-aware
discretisation, and its ablation (``spacing_aware=False``) can only show anything on
anisotropic data. ACDC's through-plane spacing is typically 5-10 mm against 1.4-1.8 mm in
plane. A convenience mirror resampled to 1x1x10 mm and centre-cropped - one exists on
Hugging Face - is perfectly good for segmentation benchmarks and useless here: the
ablation would come out flat, and flat would look like a negative result rather than a
vacuous one. This script refuses to let that happen quietly.

NIfTI headers are parsed with the standard library, so verification works before
``nibabel``, ``numpy`` or PyTorch are installed. It is meant to be the first thing run
after obtaining data.

Usage
-----
    python scripts/fetch_datasets.py --list
    python scripts/fetch_datasets.py --how acdc
    python scripts/fetch_datasets.py --verify acdc --path /data/ACDC
    python scripts/fetch_datasets.py --fetch msd_task02_heart --path /data
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "external" / "datasets.json"


# --------------------------------------------------------------------------- #
#  Manifest
# --------------------------------------------------------------------------- #
def load_manifest() -> dict:
    if not MANIFEST.exists():
        sys.exit(f"dataset manifest not found: {MANIFEST}")
    with MANIFEST.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") != 1:
        sys.exit(f"unsupported schema {data.get('schema')!r}")
    seen = set()
    for d in data["datasets"]:
        if not d.get("key"):
            sys.exit("a dataset entry has no key")
        if d["key"] in seen:
            sys.exit(f"duplicate dataset key: {d['key']}")
        seen.add(d["key"])
    return data


# --------------------------------------------------------------------------- #
#  NIfTI-1 header, standard library only
# --------------------------------------------------------------------------- #
class NiftiHeaderError(Exception):
    pass


def read_nifti_header(path: Path) -> dict:
    """Parse the 348-byte NIfTI-1 header of ``.nii`` or ``.nii.gz``.

    Returns ``dim`` (list of 8 ints), ``pixdim`` (list of 8 floats), ``datatype`` and
    ``endian``. Only the fields this verifier needs; no data is read.

    Endianness is decided from ``sizeof_hdr``, which must be 348. That is the documented
    way to detect a byte-swapped header, and getting it wrong would silently produce
    nonsense spacings - which is exactly the failure this script exists to catch.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:  # type: ignore[operator]
        raw = fh.read(348)
    if len(raw) < 348:
        raise NiftiHeaderError(f"{path.name}: shorter than a NIfTI-1 header ({len(raw)} B)")

    endian = "<"
    (sizeof_hdr,) = struct.unpack_from("<i", raw, 0)
    if sizeof_hdr != 348:
        (sizeof_hdr,) = struct.unpack_from(">i", raw, 0)
        endian = ">"
        if sizeof_hdr != 348:
            raise NiftiHeaderError(
                f"{path.name}: sizeof_hdr is {sizeof_hdr}, not 348 in either byte order - "
                f"not a NIfTI-1 file"
            )

    dim = list(struct.unpack_from(endian + "8h", raw, 40))
    datatype, _bitpix = struct.unpack_from(endian + "2h", raw, 70)
    pixdim = list(struct.unpack_from(endian + "8f", raw, 76))
    return {"dim": dim, "pixdim": pixdim, "datatype": datatype, "endian": endian}


def spacing_of(hdr: dict) -> tuple[float, float, float]:
    """``pixdim[1:4]`` - the three spatial spacings in mm."""
    p = hdr["pixdim"]
    return (abs(float(p[1])), abs(float(p[2])), abs(float(p[3])))


def n_dims(hdr: dict) -> int:
    return int(hdr["dim"][0])


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def cmd_list(datasets: list[dict]) -> int:
    access_label = {
        "registration": "REGISTRATION REQUIRED",
        "open-download": "open download",
        "application": "APPLICATION REQUIRED",
        "generated": "generated locally",
    }
    print("Datasets. None of these is redistributed with this repository.\n")
    for d in datasets:
        usable = d.get("usable_for_this_thesis", True)
        flag = "" if usable else "   [NOT USABLE HERE]"
        print(f"  {d['key']:<20} {access_label.get(d['access'], d['access']):<22}{flag}")
        print(f"  {'':<20} {d['name']}")
        if d.get("n_subjects"):
            print(f"  {'':<20} {d['n_subjects']} subjects | {d['license']}")
        print(f"  {'':<20} role: {d['role'][:96]}")
        print()
    gated = [d["key"] for d in datasets if d["access"] in ("registration", "application")]
    unusable = [d["key"] for d in datasets if not d.get("usable_for_this_thesis", True)]
    print(f"{len(datasets)} datasets: {len(gated)} need registration "
          f"({', '.join(gated)}).")
    if unusable:
        print(f"{len(unusable)} listed but NOT usable for this thesis: {', '.join(unusable)}")
        print("  (they are listed so they are not proposed as substitutes by mistake)")
    print("\nAfter obtaining one:  --verify <key> --path <dir>")
    return 0


def cmd_how(datasets: list[dict], key: str) -> int:
    d = next((x for x in datasets if x["key"] == key), None)
    if d is None:
        print(f"unknown dataset {key!r}; try --list", file=sys.stderr)
        return 2

    print(f"{d['name']}\n" + "=" * min(78, len(d["name"])))
    print(f"\nrole in this thesis:\n  {d['role']}")
    print(f"\naccess:   {d['access']}")
    print(f"url:      {d['url']}")
    if d.get("alternate_url"):
        print(f"also:     {d['alternate_url']}")
    print(f"licence:  {d['license']}")
    if d.get("cite_required"):
        b = d.get("bib")
        print("citation: REQUIRED" + (f" - refs.bib key '{b}'" if b else ""))
    if d.get("used_by"):
        print(f"loader:   {d['used_by']}")

    if d["access"] == "registration":
        print("\nThis dataset cannot be scripted. You must:")
        print(f"  1. register at {d['url']}")
        print("  2. accept the data-use agreement")
        print("  3. download and extract it yourself")
        print(f"  4. run:  python scripts/fetch_datasets.py --verify {key} --path <dir>")
        print("\nDo not commit it, and do not redistribute it.")
    elif d["access"] == "generated":
        print("\nNothing to download - generate it:")
        print("  python -c \"from cvdyn2dgs.data.phantom import make_phantom; make_phantom()\"")
        print("  or simply:  cvdyn2dgs demo")

    if d.get("expected_layout"):
        lay = d["expected_layout"]
        print("\nexpected layout:")
        if lay.get("patient_dir_glob"):
            print(f"  patient dirs:  {lay['patient_dir_glob']}")
        for f in lay.get("required_per_patient", []):
            print(f"  required:      {f}")
        for f in lay.get("optional_per_patient", []):
            print(f"  optional:      {f}")

    if d.get("expected_geometry"):
        g = d["expected_geometry"]
        if g.get("anisotropic_required"):
            print("\ngeometry:  anisotropic spacing is REQUIRED")
            print(f"  typical: {g.get('typical_spacing_mm')} mm")
            print(f"  minimum z:xy ratio checked: {g.get('min_z_to_xy_ratio')}")
        print(f"  why: {g['why']}")

    for m in d.get("known_mirrors", []):
        verdict = "usable" if m.get("usable_for_this_thesis") else "NOT USABLE"
        print(f"\nmirror ({verdict}): {m['url']}")
        print(f"  {m['reason']}")

    if d.get("caveat"):
        print(f"\ncaveat:\n  {d['caveat']}")
    return 0


# --------------------------------------------------------------------------- #
#  Verification
# --------------------------------------------------------------------------- #
def _match_any(names: list[str], pattern: str) -> list[str]:
    """``pattern`` may contain ``|`` alternatives, as the manifest uses."""
    out: list[str] = []
    for alt in pattern.split("|"):
        out += [n for n in names if fnmatch.fnmatch(n, alt)]
    return out


def verify_dataset(d: dict, root: Path, *, max_patients: int = 0) -> int:
    """Check a manually obtained copy. Returns a process exit code."""
    problems: list[str] = []
    warnings: list[str] = []

    print(f"verifying {d['key']} at {root}")
    if not root.is_dir():
        print(f"  not a directory: {root}", file=sys.stderr)
        return 2

    lay = d.get("expected_layout")
    if not lay:
        print("  no layout recorded for this dataset - nothing to check structurally")
        print(f"  (loader: {d.get('used_by') or 'none implemented'})")
        return 0

    # Locate patient directories.
    for need in lay.get("root_contains") or []:
        if not (root / need).exists():
            warnings.append(
                f"expected '{need}/' directly under the given path - if you pointed at "
                f"'{need}' itself, pass its parent instead"
            )
    pdirs = sorted(p for p in root.glob(lay["patient_dir_glob"]) if p.is_dir())
    if not pdirs and warnings:
        # Tolerate being handed the training/ directory itself.
        alt = lay["patient_dir_glob"].split("/")[-1]
        pdirs = sorted(p for p in root.glob(alt) if p.is_dir())
        if pdirs:
            warnings.append(f"matched '{alt}' directly under the given path")
    print(f"  patient directories: {len(pdirs)}")
    if not pdirs:
        problems.append(f"no directories matched {lay['patient_dir_glob']!r}")
        _report(problems, warnings)
        return 1

    expected_n = d.get("n_subjects")
    if expected_n and len(pdirs) < expected_n:
        warnings.append(
            f"manifest records {expected_n} subjects, found {len(pdirs)} - a partial "
            f"download is fine for a smoke test but not for the reported protocol"
        )

    checked = pdirs if max_patients <= 0 else pdirs[:max_patients]
    missing_required = 0
    no_labels = 0
    not_4d: list[str] = []
    spacings: list[tuple[str, tuple[float, float, float]]] = []
    in_plane: list[tuple[int, int]] = []
    header_failures: list[str] = []

    for pdir in checked:
        names = [f.name for f in pdir.iterdir() if f.is_file()]
        pid = pdir.name

        cine_files: list[str] = []
        for pat in lay.get("required_per_patient", []):
            hits = _match_any(names, pat.replace("{pid}", pid))
            if not hits:
                missing_required += 1
                if missing_required <= 3:
                    problems.append(f"{pid}: no file matching {pat.replace('{pid}', pid)!r}")
            else:
                cine_files += hits

        if not _match_any(names, "*_gt.nii*") and not _match_any(names, "*_ED*.nii*"):
            no_labels += 1

        # Geometry: read the cine header.
        for cf in cine_files[:1]:
            try:
                hdr = read_nifti_header(pdir / cf)
            except (NiftiHeaderError, OSError, EOFError, struct.error) as exc:
                header_failures.append(f"{pid}/{cf}: {exc}")
                continue
            if lay.get("cine_must_be_4d") and n_dims(hdr) != 4:
                not_4d.append(f"{pid}/{cf} is {n_dims(hdr)}-D")
            spacings.append((pid, spacing_of(hdr)))
            in_plane.append((int(hdr["dim"][1]), int(hdr["dim"][2])))

    if missing_required > 3:
        problems.append(f"... and {missing_required - 3} more patients missing required files")
    if no_labels:
        warnings.append(
            f"{no_labels}/{len(checked)} patients have no label files - ED/ES ground truth "
            f"is needed for Dice, HD95 and the oracle surface source"
        )
    for nf in not_4d[:3]:
        problems.append(f"cine is not 4-D: {nf}")
    for hf in header_failures[:3]:
        problems.append(f"unreadable NIfTI header: {hf}")

    # ---- geometry, the check that matters most --------------------------------
    geo = d.get("expected_geometry") or {}
    if spacings:
        ratios = [(pid, s, s[2] / max(min(s[0], s[1]), 1e-9)) for pid, s in spacings]
        rs = sorted(r for _, _, r in ratios)
        med = rs[len(rs) // 2]
        ex_pid, ex_sp, _ = ratios[0]
        print(f"  voxel spacing (first patient {ex_pid}): "
              f"{ex_sp[0]:.3f} x {ex_sp[1]:.3f} x {ex_sp[2]:.3f} mm")
        print(f"  through-plane : in-plane ratio, median over {len(rs)} patients: {med:.2f}")

        if geo.get("anisotropic_required"):
            need = float(geo.get("min_z_to_xy_ratio", 2.0))
            if med < need:
                problems.append(
                    f"spacing is near-isotropic (median ratio {med:.2f} < {need:g}). This "
                    f"copy has almost certainly been RESAMPLED. Contribution 1 is the "
                    f"spacing-aware discretisation, and its ablation cannot show anything "
                    f"on isotropic data - it would come out flat, and flat reads as a "
                    f"negative result rather than a vacuous one. Obtain the original."
                )
            flat = [pid for pid, _, r in ratios if r < need]
            if flat and med >= need:
                warnings.append(
                    f"{len(flat)} patient(s) are near-isotropic while the median is not; "
                    f"exclude them from the spacing ablation or report them separately"
                )
        # A single shared spacing across many subjects is not plausible for multi-scanner
        # data. Where the manifest says so explicitly this is a problem, not a warning:
        # it is the signature of a resampled copy, and the anisotropy ratio above can
        # still look healthy while the data has in fact been rewritten.
        uniq = {tuple(round(v, 3) for v in s) for _, s in spacings}
        if len(spacings) > 3 and len(uniq) == 1:
            msg = (
                f"every one of {len(spacings)} patients has identical spacing "
                f"{next(iter(uniq))}. This dataset is multi-scanner, so the spacing should "
                f"vary between subjects; a single shared value means the copy has been "
                f"resampled"
            )
            if geo.get("spacing_must_vary_across_subjects"):
                problems.append(msg)
            else:
                warnings.append(msg)

        # Named mirrors are matched by fingerprint. A mirror that keeps the through-plane
        # spacing (1x1x10 mm, say) sails through the ratio test, so listing it in the
        # manifest is only useful if it is actually detectable.
        for fp in geo.get("resampled_fingerprints") or []:
            want_sp = tuple(round(float(v), 3) for v in fp["spacing_mm"])
            hits = [pid for pid, s in spacings if tuple(round(v, 3) for v in s) == want_sp]
            shape_ok = True
            if fp.get("in_plane_shape") and in_plane:
                want_shape = tuple(int(v) for v in fp["in_plane_shape"])
                shape_ok = any(s == want_shape for s in in_plane)
            if hits and shape_ok:
                problems.append(
                    f"fingerprint match for a known pre-processed mirror "
                    f"({fp['mirror']}): spacing {want_sp} mm"
                    + (f", in-plane {tuple(fp['in_plane_shape'])}"
                       if fp.get("in_plane_shape") else "")
                    + f" on {len(hits)}/{len(spacings)} patients. {fp['reason']} "
                    f"Obtain the original from {d['url']}."
                )
    elif lay.get("cine_must_be_4d"):
        problems.append("no cine header could be read, so geometry was not verified at all")

    _report(problems, warnings)
    if problems:
        return 1
    print(f"\n  {d['key']} looks consistent with what {d.get('used_by')} expects.")
    print("  This checks structure and geometry only - not image quality or labelling.")
    return 0


def _report(problems: list[str], warnings: list[str]) -> None:
    print()
    for w in warnings:
        print(f"  warning: {w}")
    if problems:
        print(f"\n  {len(problems)} problem(s):")
        for p in problems:
            print(f"    - {p}")


# --------------------------------------------------------------------------- #
#  The one dataset that can actually be fetched
# --------------------------------------------------------------------------- #
def cmd_fetch(datasets: list[dict], key: str, dest: Path) -> int:
    d = next((x for x in datasets if x["key"] == key), None)
    if d is None:
        print(f"unknown dataset {key!r}", file=sys.stderr)
        return 2
    if d["access"] != "open-download":
        print(f"{key} is '{d['access']}' and cannot be fetched by a script.", file=sys.stderr)
        print(f"Run:  python scripts/fetch_datasets.py --how {key}", file=sys.stderr)
        return 2
    if not d.get("usable_for_this_thesis", True):
        print(f"NOTE: {key} is downloadable but NOT usable for this thesis.")
        print(f"  {d['caveat']}")
        print("\nFetching anyway, since you asked explicitly.\n")

    if shutil.which("aws") is None:
        print("The MSD archives are hosted on the AWS Open Data registry and are most")
        print("reliably fetched with the AWS CLI, which is not installed here.")
        print(f"\n  registry: {d.get('alternate_url')}")
        print(f"  homepage: {d['url']}")
        print("\n  aws s3 cp --no-sign-request \\")
        print("    s3://msd-for-monai/Task02_Heart.tar <dest>/")
        print(f"\nThen extract into {dest} and re-run with --verify if a layout is recorded.")
        return 2

    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", "--no-sign-request",
           "s3://msd-for-monai/Task02_Heart.tar", str(dest) + "/"]
    print("  " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="show all datasets and how to get them")
    ap.add_argument("--how", metavar="KEY", help="detailed access instructions for one dataset")
    ap.add_argument("--verify", metavar="KEY", help="check a manually obtained copy")
    ap.add_argument("--fetch", metavar="KEY", help="download (only works for open datasets)")
    ap.add_argument("--path", metavar="DIR", help="dataset root, for --verify / --fetch")
    ap.add_argument("--max-patients", type=int, default=0,
                    help="check only the first N patient directories (0 = all)")
    args = ap.parse_args(argv)

    datasets = load_manifest()["datasets"]

    if args.list:
        return cmd_list(datasets)
    if args.how:
        return cmd_how(datasets, args.how)
    if args.verify:
        if not args.path:
            print("--verify needs --path", file=sys.stderr)
            return 2
        d = next((x for x in datasets if x["key"] == args.verify), None)
        if d is None:
            print(f"unknown dataset {args.verify!r}; try --list", file=sys.stderr)
            return 2
        return verify_dataset(d, Path(args.path), max_patients=args.max_patients)
    if args.fetch:
        return cmd_fetch(datasets, args.fetch, Path(args.path or "."))
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
