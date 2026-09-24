"""Make the test suite exercise *this* checkout, whatever happens to be installed.

The failure this exists for
---------------------------
On the first GPU runs ``pytest tests/`` failed at collection with

    ModuleNotFoundError: No module named 'cvdyn2dgs.metrics.clinical'

while the CLI imported that very module without trouble - ``cvdyn2dgs demo`` passed, and it
pulls in ``metrics.clinical`` transitively. So the module existed and was importable; what
differed was *which* ``cvdyn2dgs`` pytest resolved.

The first attempt at a fix inserted the repository root into ``sys.path`` and changed
nothing, for a specific reason worth recording: a modern setuptools **editable install
registers a finder in ``sys.meta_path``**, and ``sys.meta_path`` is consulted *before*
``sys.path``. An ``__editable__`` finder left behind by an earlier ``pip install -e .``
therefore intercepts every ``cvdyn2dgs`` import, and no amount of path manipulation can
outrank it. When that install points at a different checkout - or its generated mapping is
stale or partial - the top-level package resolves while a submodule does not, which is
exactly the observed symptom.

So this removes the interception rather than trying to out-prioritise it, and then verifies
the outcome instead of assuming it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "cvdyn2dgs"


def _drop_shadowing_finders() -> list[str]:
    """Remove meta-path finders that resolve this package outside the checkout.

    The test is **behavioural**, not name-based: ask each finder where it would put
    ``cvdyn2dgs`` and drop it if the answer is not this tree. Matching on ``__editable__``
    in a module name would only catch setuptools' current scheme and would miss a ``.pth``
    hook, a conda shim, or a future layout, whereas this catches anything that shadows.
    Other projects' finders are left alone.
    """
    removed: list[str] = []
    for finder in list(sys.meta_path):
        find_spec = getattr(finder, "find_spec", None)
        if find_spec is None:
            continue
        try:
            spec = find_spec("cvdyn2dgs", None, None)
        except Exception:  # noqa: BLE001 - a finder that raises here cannot be trusted
            continue
        origin = getattr(spec, "origin", None) if spec else None
        if not origin or origin == "builtin":
            continue
        try:
            resolved = Path(origin).resolve()
        except OSError:
            continue
        if ROOT not in resolved.parents:
            sys.meta_path.remove(finder)
            removed.append(f"{type(finder).__module__}.{type(finder).__name__} -> {resolved}")
    return removed


def _evict_foreign_modules() -> list[str]:
    """Forget any already-imported ``cvdyn2dgs`` module that came from outside ROOT."""
    evicted: list[str] = []
    for name in [n for n in sys.modules if n == "cvdyn2dgs" or n.startswith("cvdyn2dgs.")]:
        mod = sys.modules[name]
        f = getattr(mod, "__file__", None)
        if f and ROOT not in Path(f).resolve().parents:
            del sys.modules[name]
            evicted.append(name)
    return evicted


_removed = _drop_shadowing_finders()
_evicted = _evict_foreign_modules()

# Position 0: ahead of anything the interpreter or an installed distribution contributed.
_root = str(ROOT)
if _root in sys.path:
    sys.path.remove(_root)
sys.path.insert(0, _root)


def _verify() -> None:
    """Confirm the package resolves inside this checkout, and say so clearly if it does not.

    Failing loudly here is worth it: the alternative is a collection error several imports
    deep that names a submodule and says nothing about the actual cause.
    """
    if not PKG.is_dir():
        raise RuntimeError(f"{PKG} not found - is this a complete checkout?")

    # Only the top-level package is imported, and cvdyn2dgs/__init__.py pulls in nothing
    # heavy, so this works without PyTorch installed. Deliberately NOT importing a submodule:
    # that would execute the torch-dependent chain, and a missing third-party dependency
    # would then be misreported as a shadowing problem. An earlier version of this file made
    # exactly that mistake and blamed shadowing for an absent torch.
    import cvdyn2dgs  # noqa: PLC0415

    got = Path(cvdyn2dgs.__file__ or "").resolve().parent
    if got != PKG.resolve():
        raise RuntimeError(
            f"'cvdyn2dgs' resolved to {got}, not the checkout at {PKG}.\n"
            f"An installed copy is shadowing the source tree, so the tests would not be "
            f"testing this working directory.\n"
            f"Fix it with:  pip uninstall -y cvdyn2dgs\n"
            f"(shadowing finders removed by conftest: {_removed or 'none'}; "
            f"modules evicted: {_evicted or 'none'})"
        )


_verify()
