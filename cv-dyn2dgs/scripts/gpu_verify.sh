#!/usr/bin/env bash
# Ordered first-run verification on a GPU box. Safe to run unattended.
#
# Why a script rather than a list of commands
# ------------------------------------------
# None of the PyTorch code in this repository has ever been executed, so the first GPU run
# is a debugging session, not a validation. Three properties matter for that:
#
#   1. DEPENDENCY ORDER. A failure in an early stage explains every later one. The stages
#      below are ordered so the FIRST failure is the one to fix.
#   2. NOTHING IS SKIPPED SILENTLY. Each stage's full log is kept, and the summary at the
#      end lists every stage with its status and duration - including ones that were not
#      attempted, and why.
#   3. THE ENVIRONMENT IS RECORDED. A timing number without the GPU model, driver, torch
#      and CUDA versions is not reproducible. Those go in the report before anything runs.
#
# On a managed batch runner (VESSL Cloud, Slurm, SageMaker) the GPU is released the moment
# the job exits, so anything not written under $OUT is lost. Mount a volume at $OUT.
#
# Usage
#   scripts/gpu_verify.sh                     # smoke + tests + theory  (fast, ~minutes)
#   scripts/gpu_verify.sh --full              # + experiments + tables  (long, GPU-hours)
#   scripts/gpu_verify.sh --size medium       # phantom size for experiments
#   OUT=/mnt/results scripts/gpu_verify.sh    # where to write (default: ./results)
#
# Exit code is 0 only if every attempted stage passed.

set -uo pipefail

OUT="${OUT:-results}"
SIZE="small"
FULL=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --size) shift; ;;
    --size=*) SIZE="${arg#*=}" ;;
    tiny|small|medium|large) SIZE="$arg" ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.." || exit 2
mkdir -p "$OUT/logs"
REPORT="$OUT/verification_report.txt"
PY="${PYTHON:-python3}"

names=(); states=(); secs=(); notes=()

say() { printf '%s\n' "$*" | tee -a "$REPORT"; }
rule() { say "$(printf '%.0s─' $(seq 1 72))"; }

# ---------------------------------------------------------------------------
# Environment first: a measurement without it is not reproducible.
# ---------------------------------------------------------------------------
: > "$REPORT"
say "CV-Dyn2DGS GPU verification"
say "started      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
say "host         $(uname -sr) $(uname -m)"
say "python       $($PY --version 2>&1)"
say "git commit   $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
say "git tree     $(git rev-parse HEAD^{tree} 2>/dev/null || echo n/a)"
say "output       $OUT"
say "mode         $([ "$FULL" -eq 1 ] && echo 'full (experiments included)' || echo 'quick')"
say "phantom size $SIZE"

if command -v nvidia-smi >/dev/null 2>&1; then
  say "gpu          $(nvidia-smi --query-gpu=name,memory.total,driver_version \
                      --format=csv,noheader 2>/dev/null | paste -sd'; ' -)"
else
  say "gpu          nvidia-smi not found - if this is a GPU box, the driver is missing"
fi

$PY - >>"$REPORT" 2>&1 <<'EOF' || true
try:
    import torch
    print(f"torch        {torch.__version__}")
    print(f"cuda build   {torch.version.cuda}")
    print(f"cuda avail   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device       {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"capability   sm_{cap[0]}{cap[1]}")
except Exception as exc:
    print(f"torch        NOT IMPORTABLE: {exc}")
EOF
rule

# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------
run_stage() {  # run_stage <name> <requires-torch:0|1> <command...>
  local name="$1"; shift
  local needs_torch="$1"; shift
  local log="$OUT/logs/$(echo "$name" | tr ' /' '__').log"

  if [ "$needs_torch" -eq 1 ] && ! $PY -c 'import torch' >/dev/null 2>&1; then
    names+=("$name"); states+=("SKIP"); secs+=("0")
    notes+=("PyTorch not importable - nothing about this stage was tested")
    say "SKIP  $name  (no PyTorch)"
    return 0
  fi

  printf 'RUN   %s ... ' "$name" | tee -a "$REPORT"
  local t0 t1 rc
  t0=$(date +%s)
  ( "$@" ) >"$log" 2>&1
  rc=$?
  t1=$(date +%s)
  local d=$((t1 - t0))

  names+=("$name"); secs+=("$d")
  if [ $rc -eq 0 ]; then
    states+=("PASS"); notes+=("")
    say "PASS (${d}s)"
  else
    states+=("FAIL"); notes+=("exit $rc - see ${log#"$OUT/"}")
    say "FAIL (${d}s, exit $rc)"
    say "      last 12 lines of $log:"
    tail -n 12 "$log" | sed 's/^/      | /' | tee -a "$REPORT" >/dev/null
    tail -n 12 "$log" | sed 's/^/      | /'
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Stages, in dependency order. Fix the FIRST failure and re-run.
# ---------------------------------------------------------------------------

# 0) No torch needed. If this fails, the checkout itself is wrong - stop and check the
#    transfer before blaming the GPU.
run_stage "kernel checks (no torch)"   0 "$PY" scripts/verify_kernels.py
if [ "${states[0]}" != "PASS" ]; then
  say ""
  say "kernel checks failed, which means the source tree is not intact."
  say "Verify the transfer before spending GPU time: expected tree"
  say "  2efe476f8bd2d2d623d4f01f35c512fd5034bc22"
  say "got"
  say "  $(git rev-parse HEAD^{tree} 2>/dev/null || echo n/a)"
  exit 1
fi

run_stage "torch-free unit tests"      0 "$PY" -m pytest -q \
    tests/test_viewpoint.py tests/test_costquality.py

# 1) The single most informative stage. 25 independent checks, dependency-ordered, one run
#    reports every failure instead of stopping at the first.
run_stage "smoke (staged diagnostic)"  1 "$PY" -m cvdyn2dgs.smoke --full

# 2) Unit tests only make sense once smoke is green.
run_stage "pytest (full suite)"        1 "$PY" -m pytest -q tests/

# 3) The theory's predicted convergence rates, measured.
run_stage "theory checks"              1 "$PY" -m cvdyn2dgs.cli theory --out "$OUT"

# 4) One phantom end to end: precompute, storage, playback timing.
run_stage "demo (end to end)"          1 "$PY" -m cvdyn2dgs.cli demo --out "$OUT"

# 5) The expensive part. Opt in with --full.
if [ "$FULL" -eq 1 ]; then
  run_stage "experiments (RQ1-RQ6)"    1 "$PY" -m cvdyn2dgs.cli experiments \
      --out "$OUT" --size "$SIZE"
  run_stage "regenerate result tables" 0 "$PY" scripts/make_tables.py "$OUT" paper/tables
  run_stage "manuscript structure"     0 "$PY" scripts/check_paper.py
else
  names+=("experiments (RQ1-RQ6)"); states+=("SKIP"); secs+=("0")
  notes+=("not requested - re-run with --full")
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
rule
say "SUMMARY"
rule
fail=0; skip=0
for i in "${!names[@]}"; do
  printf '  %-6s %-28s %5ss  %s\n' \
    "${states[$i]}" "${names[$i]}" "${secs[$i]}" "${notes[$i]}" | tee -a "$REPORT"
  [ "${states[$i]}" = "FAIL" ] && fail=$((fail + 1))
  [ "${states[$i]}" = "SKIP" ] && skip=$((skip + 1))
done
rule
say "finished     $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
say "$fail failed, $skip skipped, $(( ${#names[@]} - fail - skip )) passed"
say "logs         $OUT/logs/"
say "report       $REPORT"

if [ "$fail" -gt 0 ]; then
  say ""
  say "Fix the FIRST failure above and re-run. Stages are dependency-ordered, so a later"
  say "failure is usually a consequence of an earlier one rather than a separate bug."
  exit 1
fi
if [ "$skip" -gt 0 ]; then
  say ""
  say "Nothing failed, but $skip stage(s) were not attempted, so they are unverified."
fi
say ""
say "No result in $OUT is meaningful until 'experiments' has run (--full)."
exit 0
