#!/usr/bin/env python3
"""Find tensors created without a device, the recurring cause of GPU-only failures.

Why this exists
---------------
Three device bugs have now been found in this repository, and every one of them was
invisible on the CPU and fatal on the first GPU run:

* ``FreeGaussians3D.initialise`` passed a CPU generator together with ``device="cuda"``;
* ``residual/lowrank.py`` built a ``searchsorted`` threshold on the CPU and compared it
  against a CUDA tensor;
* a third, still unidentified when this script was written, in the precompute pipeline.

Grepping for ``torch.zeros(`` and eyeballing the line does not work: the ``device=``
argument is frequently on a later line, so a line-oriented search produces both false
positives and false negatives. This script parses the AST instead, so a call spanning any
number of lines is judged correctly.

What counts as safe
-------------------
A tensor factory call is safe when any of these holds:

* it passes ``device=`` explicitly;
* it is a ``*_like`` variant, which inherits device from its argument;
* its result is immediately moved with ``.to(...)`` or ``.cuda()``;
* it is assigned to a name that is moved with ``.to(...)`` later in the same function;
* it carries a ``# device-ok: <reason>`` comment on the same line or the line above;
* the module is on the allow-list below, with a stated reason.

The ``# device-ok:`` convention is the important one. A site that is genuinely fine on the
CPU - a CPU generator draw that is moved afterwards, a value immediately reduced to a
Python float, a buffer built for serialisation - should say so *next to the code*, not in
someone's head. A blanket allow-list hides the next real bug in the same file; a per-site
justification does not.

The remaining checks are heuristics, so the report separates **definite** findings from
ones that need a human look. Being explicit about that matters: a checker that cries wolf
gets switched off, and a checker that silently forgives is worse than none.

Usage
-----
    python scripts/audit_devices.py              # report
    python scripts/audit_devices.py --strict     # non-zero exit on any definite finding

Standard library only; no torch needed.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "cvdyn2dgs"

# Factories that allocate a new tensor and therefore need a device.
FACTORIES = {
    "tensor", "zeros", "ones", "empty", "full", "arange", "linspace", "logspace",
    "eye", "rand", "randn", "randint", "randperm", "as_tensor", "from_numpy",
}
# These inherit their device from an argument.
LIKE_SUFFIX = "_like"

# Modules exempt from the check, each with a reason that must be written down.
ALLOW = {
    "cvdyn2dgs/smoke.py":
        "diagnostic stages construct small CPU fixtures deliberately and move them when "
        "the stage needs a device",
    "cvdyn2dgs/experiments/theory_checks.py":
        "convergence-rate checks run in float64 on the CPU on purpose: the rates being "
        "measured are discretisation rates, and GPU float32 would confound them",
    "cvdyn2dgs/data/phantom.py":
        "the phantom is generated on the CPU from a CPU generator and moved with .to() by "
        "Phantom4D.to()",
}
# Everything else justifies itself in place with `# device-ok: <reason>`. Prefer that: a
# module-level exemption also hides the NEXT bug in the same file.


class Visitor(ast.NodeVisitor):
    def __init__(self, path: Path, src: str) -> None:
        self.path = path
        self.src = src.split("\n")
        self.findings: list[tuple[int, str, str]] = []
        self._moved_names: set[str] = set()

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _torch_factory(node: ast.Call) -> str | None:
        """Return the factory name if this is ``torch.<factory>(...)``."""
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            if f.value.id == "torch" and (
                f.attr in FACTORIES or f.attr.endswith(LIKE_SUFFIX)
            ):
                return f.attr
        return None

    @staticmethod
    def _has_device(node: ast.Call) -> bool:
        for kw in node.keywords:
            if kw.arg in ("device", "out"):
                return True
            if kw.arg is None:  # **kwargs - cannot tell, assume handled
                return True
        return False

    def _collect_moved(self, fn: ast.AST) -> set[str]:
        """Names that get ``.to(...)`` / ``.cuda()`` applied somewhere in this function."""
        moved: set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr in ("to", "cuda"):
                    tgt = n.func.value
                    if isinstance(tgt, ast.Name):
                        moved.add(tgt.id)
                    elif isinstance(tgt, ast.Attribute):
                        moved.add(tgt.attr)
        return moved

    # ---- visiting --------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        prev = self._moved_names
        self._moved_names = self._collect_moved(node)
        self.generic_visit(node)
        self._moved_names = prev

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = self._torch_factory(node)
        if name and not name.endswith(LIKE_SUFFIX) and not self._has_device(node):
            verdict = self._classify(node)
            if verdict:
                self.findings.append((node.lineno, f"torch.{name}", verdict))
        self.generic_visit(node)

    def _justified(self, lineno: int) -> bool:
        """``# device-ok:`` on this line, or anywhere in the comment block above it.

        The whole contiguous run of comment lines is scanned, not a fixed number: a reason
        worth writing down is usually longer than one line, and a two-line lookback would
        silently stop honouring the marker as soon as the explanation grew.  Blank lines and
        preceding statements end the block.
        """
        ln = lineno - 1  # 0-based, the statement itself
        if 0 <= ln < len(self.src) and "device-ok:" in self.src[ln]:
            return True
        ln -= 1
        while ln >= 0:
            stripped = self.src[ln].strip()
            if not stripped.startswith("#"):
                return False
            if "device-ok:" in stripped:
                return True
            ln -= 1
        return False

    def _classify(self, node: ast.Call) -> str | None:
        """``None`` means benign; otherwise a short verdict string."""
        if self._justified(node.lineno):
            return None
        line = self.src[node.lineno - 1] if node.lineno - 1 < len(self.src) else ""

        # Immediately moved: torch.zeros(...).to(dev) / .cuda()
        parent_moved = getattr(node, "_immediately_moved", False)
        if parent_moved:
            return None

        # Assigned to a name that is moved later in the same function.
        tgt = getattr(node, "_assign_target", None)
        if tgt and tgt in self._moved_names:
            return "LIKELY OK (assigned to a name that is later moved with .to)"

        # A dtype-only call inside a .to() chain on the same line is usually fine.
        if ".to(" in line or ".cuda()" in line:
            return "LIKELY OK (.to on the same line)"

        return "DEFINITE (no device, not moved)"


def annotate(tree: ast.AST) -> None:
    """Mark calls that are immediately ``.to(...)``-ed, and record assignment targets."""
    for node in ast.walk(tree):
        # torch.zeros(...).to(x)  ->  Call(func=Attribute(attr='to', value=Call(...)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("to", "cuda") and isinstance(node.func.value, ast.Call):
                node.func.value._immediately_moved = True  # type: ignore[attr-defined]
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                node.value._assign_target = node.targets[0].id  # type: ignore[attr-defined]


def audit(paths: list[Path]) -> tuple[int, int]:
    definite = 0
    review = 0
    for path in paths:
        rel = str(path.relative_to(ROOT))
        if rel in ALLOW:
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, rel)
        annotate(tree)
        v = Visitor(path, src)
        v.visit(tree)
        if not v.findings:
            continue
        print(f"\n{rel}")
        for lineno, what, verdict in sorted(v.findings):
            tag = "DEFINITE" if verdict.startswith("DEFINITE") else "review  "
            snippet = src.split("\n")[lineno - 1].strip()[:84]
            print(f"  {tag} :{lineno:<5} {what:<16} {snippet}")
            if verdict.startswith("DEFINITE"):
                definite += 1
            else:
                review += 1
    return definite, review


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any DEFINITE finding remains")
    args = ap.parse_args(argv)

    if not PKG.is_dir():
        print(f"package not found: {PKG}", file=sys.stderr)
        return 2

    paths = sorted(PKG.rglob("*.py"))
    print(f"auditing {len(paths)} modules for tensors created without a device")
    print(f"({len(ALLOW)} module(s) allow-listed, each with a stated reason)")
    definite, review = audit(paths)

    print("\n" + "-" * 72)
    print(f"{definite} definite, {review} to review")
    if ALLOW:
        print("\nallow-listed:")
        for k, why in ALLOW.items():
            print(f"  {k}\n    {why}")
    if definite:
        print("\nA tensor created without a device lands on the CPU. That is harmless on a")
        print("CPU-only machine and raises on the first GPU run, which is the worst possible")
        print("time to find out. Pass device= explicitly, or move the result with .to().")
        return 1 if args.strict else 0
    print("\nNo definite findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
