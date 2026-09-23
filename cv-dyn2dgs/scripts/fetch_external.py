#!/usr/bin/env python3
"""Fetch the third-party comparison targets at pinned commits.

Nothing third-party is stored in this repository. Each target in
``external/manifest.json`` is pinned by commit SHA and cloned on demand into
``external/repos/<key>/``, which is git-ignored.

Why not vendored
----------------
Four of the targets may not legally be redistributed inside this repository:

* ``turandai/gaussian_surfels`` and ``KeKsBoTer/cinematic-gaussians`` publish **no
  LICENSE file**, so default copyright applies - all rights reserved;
* ``graphdeco-inria/gaussian-splatting`` and ``hbb1/2d-gaussian-splatting`` are under
  the Inria/MPII research-only licence, which restricts redistribution.

Vendoring would additionally add roughly 1.2 GB and destroy the plain-text property
of the transfer patch. Pinning by SHA is both lawful and more reproducible: the
manifest records exactly which commit any number was produced from.

Usage
-----
    python scripts/fetch_external.py --list
    python scripts/fetch_external.py --licences
    python scripts/fetch_external.py --fetch at-gs dyna3dgr
    python scripts/fetch_external.py --fetch all --layer 3
    python scripts/fetch_external.py --check

Standard library only; no torch, no network libraries beyond the ``git`` binary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "external" / "manifest.json"
DEST = ROOT / "external" / "repos"


# --------------------------------------------------------------------------- #
#  Manifest
# --------------------------------------------------------------------------- #
def load_manifest() -> dict:
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")
    with MANIFEST.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") != 1:
        sys.exit(f"unsupported manifest schema: {data.get('schema')!r}")
    seen: set[str] = set()
    for t in data["targets"]:
        key = t.get("key")
        if not key:
            sys.exit("a target has no 'key'")
        if key in seen:
            sys.exit(f"duplicate target key: {key}")
        seen.add(key)
        if t.get("repo") and not t.get("commit"):
            sys.exit(f"{key}: has a repo but no pinned commit")
        if t.get("commit") and len(t["commit"]) != 40:
            sys.exit(f"{key}: commit is not a full 40-character SHA")
    return data


def fetchable(t: dict) -> bool:
    return bool(t.get("repo")) and bool(t.get("commit"))


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def cmd_list(targets: list[dict]) -> int:
    layers = {
        1: "surface source",
        2: "representation primitive",
        3: "temporal model",
        4: "storage and playback",
    }
    for layer in sorted(layers):
        group = [t for t in targets if t["layer"] == layer]
        if not group:
            continue
        print(f"\nLayer {layer} - {layers[layer]}")
        print("-" * 78)
        for t in group:
            state = "fetchable" if fetchable(t) else "NO PUBLIC CODE"
            here = DEST / t["key"]
            if here.exists():
                state = "fetched"
            redis = "" if t.get("redistributable") else "  [do not vendor]"
            print(f"  {t['key']:<22} {state:<16} {t['license']}{redis}")
            print(f"  {'':<22} {t['name']}")
    print()
    n_fetch = sum(1 for t in targets if fetchable(t))
    print(f"{len(targets)} targets, {n_fetch} fetchable, "
          f"{sum(1 for t in targets if not t.get('redistributable'))} not redistributable")
    return 0


def cmd_licences(targets: list[dict]) -> int:
    blocked = [t for t in targets if not t.get("redistributable")]
    print("Targets that MUST NOT be copied into this repository:")
    print("=" * 78)
    for t in blocked:
        print(f"\n  {t['key']}  ({t['license']})")
        print(f"    {t['name']}")
        if t.get("caveat"):
            print(f"    {t['caveat']}")
    print("\n" + "=" * 78)
    print("Cloning and running these for evaluation is fine. Redistributing them is not.")
    print("external/repos/ is git-ignored precisely so this cannot happen by accident.")
    return 0


# --------------------------------------------------------------------------- #
#  Fetching
# --------------------------------------------------------------------------- #
def _git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True,
    )


def _url_forms(repo: str) -> list[str]:
    """Both spellings of a GitHub URL.

    Some proxies and mirrors resolve only ``.../owner/name.git`` and return 404 for
    ``.../owner/name``; others do the reverse. The manifest stores the bare form for
    readability, so try both rather than making the manifest carry the quirk.
    """
    bare = repo[:-4] if repo.endswith(".git") else repo
    return [bare + ".git", bare]


def fetch_one(t: dict, *, force: bool = False, submodules: bool = False) -> bool:
    """Clone one target at its pinned commit. Returns True on success."""
    key, repo, sha = t["key"], t["repo"], t["commit"]
    dest = DEST / key

    if dest.exists():
        if not force:
            got = _git(["rev-parse", "HEAD"], cwd=dest).stdout.strip()
            if got == sha:
                print(f"  {key}: already at pinned commit")
                return True
            print(f"  {key}: present but at {got[:12]}, pinned is {sha[:12]}"
                  f" - re-run with --force to replace")
            return False
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {key}: cloning {repo}")
    branch = t.get("branch") or "HEAD"
    ok = False
    last_err = ""

    for url in _url_forms(repo):
        # Prefer fetching only the pinned commit; not every host enables
        # uploadpack.allowReachableSHA1InWant, so fall back to a branch clone.
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        if _git(["init", "--quiet"], cwd=dest).returncode == 0:
            _git(["remote", "add", "origin", url], cwd=dest)
            r = _git(["fetch", "--depth", "1", "--quiet", "origin", sha], cwd=dest)
            if r.returncode == 0 and _git(
                ["checkout", "--quiet", "FETCH_HEAD"], cwd=dest
            ).returncode == 0:
                ok = True
                break
            last_err = r.stderr.strip()

        shutil.rmtree(dest)
        r = _git(["clone", "--quiet", "--branch", branch, url, str(dest)])
        if r.returncode == 0:
            if _git(["checkout", "--quiet", sha], cwd=dest).returncode == 0:
                ok = True
                break
            print(f"  {key}: pinned commit {sha[:12]} not reachable on {branch}")
            return False
        last_err = r.stderr.strip()

    if not ok:
        if dest.exists():
            shutil.rmtree(dest)
        tail = last_err.splitlines()[-1] if last_err else "unknown error"
        print(f"  {key}: CLONE FAILED - {tail}")
        return False

    got = _git(["rev-parse", "HEAD"], cwd=dest).stdout.strip()
    if got != sha:
        print(f"  {key}: SHA MISMATCH - got {got}, expected {sha}")
        return False

    if submodules and "submodules" in (t.get("needs") or []):
        print(f"  {key}: initialising submodules")
        r = _git(["submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=dest)
        if r.returncode != 0:
            print(f"  {key}: submodule init failed (CUDA extensions will not build)")

    print(f"  {key}: OK at {sha[:12]}")
    if not t.get("redistributable"):
        print(f"  {key}: NOT REDISTRIBUTABLE ({t['license']}) - do not commit this tree")
    return True


def cmd_fetch(targets: list[dict], keys: list[str], *, force: bool,
              submodules: bool) -> int:
    by_key = {t["key"]: t for t in targets}
    if keys == ["all"]:
        chosen = [t for t in targets if fetchable(t)]
    else:
        chosen = []
        for k in keys:
            if k not in by_key:
                print(f"unknown target {k!r}; available: {', '.join(sorted(by_key))}",
                      file=sys.stderr)
                return 2
            if not fetchable(by_key[k]):
                print(f"{k}: no public code in the manifest - nothing to fetch",
                      file=sys.stderr)
                return 2
            chosen.append(by_key[k])

    if shutil.which("git") is None:
        print("git not found on PATH", file=sys.stderr)
        return 2

    print(f"fetching {len(chosen)} target(s) into {DEST.relative_to(ROOT)}/")
    failed = [t["key"] for t in chosen if not fetch_one(t, force=force, submodules=submodules)]
    print()
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}")
        return 1
    print("all requested targets fetched at their pinned commits")
    return 0


def cmd_check(targets: list[dict]) -> int:
    """Verify every already-fetched tree still matches its pinned SHA."""
    problems = 0
    checked = 0
    for t in targets:
        dest = DEST / t["key"]
        if not dest.exists():
            continue
        checked += 1
        got = _git(["rev-parse", "HEAD"], cwd=dest).stdout.strip()
        if got != t["commit"]:
            print(f"  {t['key']}: at {got[:12]}, manifest pins {t['commit'][:12]}")
            problems += 1
        else:
            dirty = _git(["status", "--porcelain"], cwd=dest).stdout.strip()
            flag = "  (working tree modified)" if dirty else ""
            print(f"  {t['key']}: OK{flag}")
            if dirty:
                problems += 1
    if checked == 0:
        print("nothing fetched yet")
        return 0
    print()
    if problems:
        print(f"{problems} problem(s): a measurement taken from these trees is not "
              f"reproducible from the manifest")
        return 1
    print(f"{checked} tree(s) match the manifest exactly")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="show all targets and their state")
    ap.add_argument("--licences", "--licenses", action="store_true", dest="licences",
                    help="show redistribution restrictions")
    ap.add_argument("--fetch", nargs="+", metavar="KEY",
                    help="clone these targets ('all' for every fetchable one)")
    ap.add_argument("--check", action="store_true",
                    help="verify fetched trees still match the pinned SHAs")
    ap.add_argument("--layer", type=int, choices=[1, 2, 3, 4],
                    help="restrict to one comparison layer")
    ap.add_argument("--force", action="store_true", help="replace an existing tree")
    ap.add_argument("--submodules", action="store_true",
                    help="also init submodules (needed to build CUDA extensions)")
    args = ap.parse_args(argv)

    data = load_manifest()
    targets = data["targets"]
    if args.layer is not None:
        targets = [t for t in targets if t["layer"] == args.layer]

    if args.list:
        return cmd_list(targets)
    if args.licences:
        return cmd_licences(targets)
    if args.check:
        return cmd_check(targets)
    if args.fetch:
        return cmd_fetch(targets, args.fetch, force=args.force, submodules=args.submodules)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
