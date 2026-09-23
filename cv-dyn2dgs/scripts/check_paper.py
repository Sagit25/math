#!/usr/bin/env python3
"""Structural checks on the LaTeX manuscript, without needing LaTeX.

No TeX distribution was available where the manuscript was written, so it has never
been compiled. This script catches the errors that a compile would catch first and
that are purely structural:

1. every ``\\input`` target exists;
2. every ``\\ref`` / ``\\eqref`` points at a ``\\label`` that is actually defined;
3. every ``\\cite`` key exists in ``refs.bib``;
4. no label is defined twice;
5. braces, ``\\begin``/``\\end`` and math delimiters are balanced per file;
6. no result table has been hand-edited (they must carry the generator's header);
7. the citation commands and ``\\bibliographystyle`` match the packages loaded.

It does not check typesetting. A clean run here means the document is structurally
consistent, not that it compiles.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"

RE_INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
RE_LABEL = re.compile(r"\\label\{([^}]+)\}")
RE_REF = re.compile(r"\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}")
RE_CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}")
RE_BIBKEY = re.compile(r"^\s*@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
RE_BEGIN = re.compile(r"\\begin\{([^}]+)\}")
RE_END = re.compile(r"\\end\{([^}]+)\}")

problems: list[str] = []
notes: list[str] = []


def strip_comments(text: str) -> str:
    """Remove LaTeX comments, honouring escaped percent signs."""
    out = []
    for line in text.split("\n"):
        idx = None
        for m in re.finditer(r"(?<!\\)%", line):
            idx = m.start()
            break
        out.append(line if idx is None else line[:idx])
    return "\n".join(out)


def collect(root: Path, rel: str, seen: set[str]) -> list[tuple[str, str]]:
    """Depth-first expansion of \\input, returning (path, stripped source) pairs."""
    path = root / (rel if rel.endswith(".tex") else rel + ".tex")
    key = str(path)
    if key in seen:
        return []
    seen.add(key)
    if not path.exists():
        problems.append(f"missing \\input target: {path.relative_to(PAPER.parent)}")
        return []
    src = strip_comments(path.read_text(encoding="utf-8"))
    out = [(str(path.relative_to(PAPER.parent)), src)]
    for target in RE_INPUT.findall(src):
        out += collect(root, target, seen)
    return out


def main() -> int:
    if not (PAPER / "main.tex").exists():
        print(f"no manuscript at {PAPER}/main.tex", file=sys.stderr)
        return 2

    files = collect(PAPER, "main.tex", set())
    print(f"expanded {len(files)} file(s) from main.tex")

    # ---- labels and references -------------------------------------------
    labels: dict[str, str] = {}
    for name, src in files:
        for lab in RE_LABEL.findall(src):
            if lab in labels:
                problems.append(f"duplicate \\label{{{lab}}}: {labels[lab]} and {name}")
            labels[lab] = name

    refs: dict[str, set[str]] = {}
    for name, src in files:
        for r in RE_REF.findall(src):
            for part in r.split(","):
                refs.setdefault(part.strip(), set()).add(name)

    dangling = sorted(r for r in refs if r not in labels)
    for r in dangling:
        problems.append(f"undefined reference \\ref{{{r}}} used in {', '.join(sorted(refs[r]))}")

    unused = sorted(set(labels) - set(refs))
    if unused:
        notes.append(f"{len(unused)} label(s) defined but never referenced: "
                     + ", ".join(unused[:8]) + (" ..." if len(unused) > 8 else ""))

    print(f"labels: {len(labels)}   references: {len(refs)}   dangling: {len(dangling)}")

    # ---- citations --------------------------------------------------------
    bib = PAPER / "refs.bib"
    if not bib.exists():
        problems.append("refs.bib not found")
        bibkeys: set[str] = set()
    else:
        bibkeys = set(RE_BIBKEY.findall(bib.read_text(encoding="utf-8")))

    cited: dict[str, set[str]] = {}
    for name, src in files:
        for group in RE_CITE.findall(src):
            for key in group.split(","):
                cited.setdefault(key.strip(), set()).add(name)

    missing = sorted(k for k in cited if k not in bibkeys)
    for k in missing:
        problems.append(f"\\cite{{{k}}} has no entry in refs.bib (used in "
                        f"{', '.join(sorted(cited[k]))})")
    uncited = sorted(bibkeys - set(cited))
    if uncited:
        notes.append(f"{len(uncited)} bib entry/entries never cited: " + ", ".join(uncited))

    print(f"bib entries: {len(bibkeys)}   cited: {len(cited)}   missing: {len(missing)}")

    # ---- balance ----------------------------------------------------------
    for name, src in files:
        depth = 0
        for ch in src:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    problems.append(f"{name}: unmatched closing brace")
                    break
        if depth > 0:
            problems.append(f"{name}: {depth} unclosed brace(s)")

        begins = RE_BEGIN.findall(src)
        ends = RE_END.findall(src)
        for env in set(begins) | set(ends):
            nb, ne = begins.count(env), ends.count(env)
            if nb != ne:
                problems.append(f"{name}: \\begin{{{env}}} x{nb} vs \\end{{{env}}} x{ne}")

        if src.count("$$") % 2 or (src.count("$") - 2 * src.count("$$")) % 2:
            problems.append(f"{name}: unbalanced $ math delimiters")

    # ---- generated tables must not be hand-edited -------------------------
    tdir = PAPER / "tables"
    if tdir.exists():
        stubs = 0
        for t in sorted(tdir.glob("*.tex")):
            head = t.read_text(encoding="utf-8")[:200]
            if "AUTO-GENERATED by scripts/make_tables.py" not in head:
                problems.append(f"{t.name}: missing generator header - was it edited by hand?")
            if "STUB" in head:
                stubs += 1
        print(f"tables: {len(list(tdir.glob('*.tex')))}   stubs (unmeasured): {stubs}")
        if stubs:
            notes.append(f"{stubs} result table(s) are stubs; run "
                         "`cvdyn2dgs experiments` then `scripts/make_tables.py`")
    else:
        problems.append("paper/tables/ does not exist - run scripts/make_tables.py")

    # ---- citation style vs loaded packages --------------------------------
    # plainnat/abbrvnat/unsrtnat emit \natexlab and the natbib \bibitem[..] form,
    # and \citep/\citet are natbib commands. Either needs natbib loaded, and it
    # must come before hyperref. None of this is caught by the checks above, and
    # all of it is a hard compile error.
    whole = "\n".join(src for _, src in files)
    packages = re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", whole)
    loaded = {p.strip() for group in packages for p in group.split(",")}
    natstyle = re.findall(r"\\bibliographystyle\{([^}]+)\}", whole)
    natcites = sorted(set(re.findall(r"\\(cite[pt]\*?|citeauthor|citeyear)\b", whole)))
    needs_natbib = natcites or any(s.endswith("nat") for s in natstyle)
    if needs_natbib and "natbib" not in loaded:
        why = []
        if natcites:
            why.append("uses " + ", ".join("\\" + c for c in natcites))
        if any(s.endswith("nat") for s in natstyle):
            why.append("bibliographystyle " + "/".join(natstyle))
        problems.append("natbib is required but never loaded (" + "; ".join(why) + ")")
    def load_pos(pkg: str) -> int | None:
        for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", whole):
            if pkg in [p.strip() for p in m.group(1).split(",")]:
                return m.start()
        return None

    p_nat, p_hyp = load_pos("natbib"), load_pos("hyperref")
    if p_nat is not None and p_hyp is not None and p_nat > p_hyp:
        problems.append("natbib must be loaded before hyperref")

    # ---- report ----------------------------------------------------------
    print()
    for n in notes:
        print(f"  note: {n}")
    if problems:
        print(f"\n{len(problems)} structural problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nno structural problems found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
