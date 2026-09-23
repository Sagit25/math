# CV-Dyn2DGS — thesis manuscript

LaTeX source for the thesis *CV-Dyn2DGS: Chan–Vese 표면에 정렬된 2D Gaussian Surfel을
이용한 실시간 심장 표면 시각화*. Body text is Korean (ko.TeX), matching the source
proposal.

## Status — read this first

This is a **draft**. Two facts about it are load-bearing:

1. **The manuscript has never been compiled.** No LaTeX distribution was available
   in the environment where it was written (`pdflatex`, `xelatex` and `latexmk` are
   all absent). Typesetting errors almost certainly remain. `scripts/check_paper.py`
   substitutes for the *structural* half of a compile; it cannot substitute for the
   typesetting half.
2. **No experimental result in this thesis has been measured.** The implementation in
   `../cvdyn2dgs/` is complete but has never been executed, because PyTorch could not
   be installed (no package index, no GPU). Chapter 8 (결과) therefore contains no
   numbers, and research questions RQ1–RQ6 are neither confirmed nor refuted.

What *is* finished: chapters 1–7 and 9–11, with chapter 4 (방법) and chapter 5 (오차
분석) written in full including proofs. Chapter 8 is placeholders.

Draft state is made visible rather than hidden. `preamble.tex` sets
`\draftmodetrue`, which turns on:

| macro | renders as | means |
|---|---|---|
| `\NM` | red `n/a` | a table cell that no run has filled |
| `\PH{x}` | red `<x>` | an inline value a real run must supply |
| `\pending{...}` | red box, `[결과 대기]` | a paragraph that cannot be written before results exist |
| `\caveat{...}` | grey box, `유의.` | a deliberate deviation from the proposal, or an honesty caveat |

plus a red banner on every page. Unmeasured quantities are never blank cells — a
blank could be mistaken for a result, `n/a` in red cannot.

**Do not clear `\draftmodetrue` until real results exist** (see *Filling in results*).

## Building

```sh
make            # xelatex + bibtex, 3 passes  (recommended)
make pdflatex   # pdflatex route; needs a ko.TeX font set installed
make check      # structural checks, no LaTeX needed
make clean
```

Requirements for `make`: XeLaTeX, `kotex`, `natbib`, `booktabs`, `longtable`,
`hyperref`, `tikz`, `fancyhdr`, and a system Korean font that `kotex` can find.
On TeX Live: `texlive-full` covers all of it.

Expect to iterate on the first build. Read `main.log`.

## Layout

```
main.tex                 document skeleton, title, draft notice
preamble.tex             packages, draft macros, theorem envs, notation macros
refs.bib                 30 entries, all cited
Makefile
sections/
  00_abstract.tex
  01_introduction.tex    problem, contributions, RQ1–RQ6
  02_related_work.tex
  03_notation.tex        grids, spacing, level-set and surfel notation
  04_method.tex          COMPLETE — Chan–Vese stage, surfel stage, coupling
  05_error_analysis.tex  COMPLETE — error decomposition with proofs
  06_implementation.tex  module map, deliberate deviations, execution status
  07_protocol.tex        what each experiment would measure and how
  08_results.tex         PLACEHOLDERS ONLY
  09_discussion.tex      conditional on results; mostly \pending
  10_limitations.tex
  11_conclusion.tex
  appendix_symbols.tex   symbol table
  appendix_code_map.tex  equation → module/function map
tables/                  12 files, ALL GENERATED — do not edit by hand
```

`sections/04_method.tex` and `sections/05_error_analysis.tex` are the substance of
the thesis; everything else is scaffolding around them.

## Tables are generated, not written

Every file in `tables/` is emitted by `../scripts/make_tables.py` from JSON in
`../results/`. The manuscript only `\input`s them. This is deliberate: a table that
can only be produced from a results file cannot contain a number someone made up.

```sh
make tables       # == cd .. && python3 scripts/make_tables.py results paper/tables
```

With no `results/` directory present, the generator writes *stubs*: correct table
structure, correct row and column labels, every data cell `\NM`. Current state —
11 of 12 tables are stubs. The exception is `tables/kernel_checks.tex`, which is
filled, because `scripts/verify_kernels.py` does run (pure stdlib, no torch) and
reports 41/41 checks passing.

`scripts/check_paper.py` refuses to pass if a table has lost its generator header,
which is how hand-editing gets caught.

## Comparison design

[`COMPARISON_TARGETS.md`](COMPARISON_TARGETS.md) lists every published method this thesis
could be compared against, in four layers (surface source, representation primitive,
temporal model, storage & playback), with venue, arXiv ID, code availability, and the
metric axis each target can legitimately be scored on — plus the comparisons that would be
*invalid*. None of them has been run; it is a design, not results.

Verifying that list against the literature turned up four wrong entries in `refs.bib`
(now corrected, each with a note recording what was wrong), one wrong claim in §2.2 about
what the Avendi et al. work does, and one wrong premise in §2.5 — Dyna3DGR's code *is*
public, so "reproduction is infeasible" no longer holds for anyone with a GPU.

## Bibliography provenance

Each entry in `refs.bib` is tagged in a comment:

- `[author-bibtex]` — copied from the authors' own BibTeX (project repo or publisher);
- `[verified]` — title, venue and year checked against the publisher or arXiv listing;
- `[unverified]` — taken from the source proposal and **not** confirmed.

One entry remains `[unverified]`: `park2026_rendering`, whose author list could not be
confirmed because arxiv.org was unreachable from the authoring environment.

## Filling in results

In an environment with PyTorch and a GPU:

```sh
cd ..
pip install -e .
cvdyn2dgs smoke --full            # 21 dependency-ordered stages
pytest -x tests/
cvdyn2dgs theory                  # numerical checks of the paper's theorems
cvdyn2dgs experiments --out results
python3 scripts/make_tables.py results paper/tables
python3 scripts/check_paper.py
```

Then, and only then:

1. edit `preamble.tex`: `\draftmodetrue` → `\draftmodefalse`;
2. remove the `\ifdraftmode` draft-notice block from `main.tex`;
3. replace every `\pending{...}` in `sections/08_results.tex` and
   `sections/09_discussion.tex` with prose describing what was actually measured;
4. update `sections/06_implementation.tex` §검증 상태 (`\label{sec:impl-status}`),
   whose first subsection is titled 실행되지 않았다;
5. update this file and `../README.md`.

`make check` should report `stubs (unmeasured): 0` before step 1.

## Checks that run without LaTeX

`scripts/check_paper.py` verifies: `\input` targets exist; every `\ref` resolves;
every `\cite` key is in `refs.bib`; no duplicate labels; braces, environments and
`$` are balanced per file; generated tables still carry their header; and the
citation commands match the loaded packages. Current output:

```
expanded 27 file(s) from main.tex
labels: 189   references: 143   dangling: 0
bib entries: 30   cited: 30   missing: 0
tables: 12   stubs (unmeasured): 11
no structural problems found
```

A clean run means the document is structurally consistent. It does **not** mean the
document compiles.
