#!/usr/bin/env bash
set -euo pipefail

IS_WSL=0
if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
  IS_WSL=1
fi

if command -v python.exe >/dev/null 2>&1; then
  PY=(python.exe)
elif command -v python >/dev/null 2>&1; then
  PY=(python)
elif command -v py >/dev/null 2>&1; then
  PY=(py -3)
elif command -v python3 >/dev/null 2>&1; then
  PY=(python3)
else
  echo "No Python interpreter found on PATH" >&2
  exit 127
fi

if [ "$IS_WSL" -eq 1 ] && [[ "${PY[0]}" == *python.exe ]]; then
  ROOT_DIR="$(wslpath -w "$PWD")"
  SOURCE_DIR="${ROOT_DIR}\\src"
else
  ROOT_DIR="$("${PY[@]}" - <<'PY'
from pathlib import Path
print(Path(".").resolve())
PY
)"
  ROOT_DIR="${ROOT_DIR%$'\r'}"
  SOURCE_DIR="$ROOT_DIR/src"
fi
PATHSEP="$("${PY[@]}" - <<'PY'
import os
print(os.pathsep)
PY
)"
PATHSEP="${PATHSEP%$'\r'}"
if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$PYTHONPATH$PATHSEP$ROOT_DIR$PATHSEP$SOURCE_DIR"
else
  export PYTHONPATH="$ROOT_DIR$PATHSEP$SOURCE_DIR"
fi
export DIFFUSION_BON_RESULTS_DIR="${DIFFUSION_BON_RESULTS_DIR:-results}"
if [ "$IS_WSL" -eq 1 ]; then
  export WSLENV="${WSLENV:+$WSLENV:}DIFFUSION_BON_RESULTS_DIR"
fi

"${PY[@]}" -m pytest -q
"${PY[@]}" scripts/run_with_src.py experiments/audit_then_sample.py --seeds 1 2 --states 8 --candidates 96 --bootstrap 40
"${PY[@]}" scripts/run_with_src.py experiments/controlled_sampler.py --seeds 1 2 --states 8 --candidates 96 --mc-trials 220
"${PY[@]}" scripts/run_with_src.py experiments/scorer_comparison.py --seeds 1 2 --states 8 --candidates 96 --mc-trials 220
"${PY[@]}" scripts/run_with_src.py experiments/nk_budget.py --seeds 1 2 --states 8 --max-candidates 64 --lambda-cost 0.0035
"${PY[@]}" scripts/run_with_src.py experiments/learned_diffusion_policy_lite.py --seeds 1 2 3 --train-states 12 --train-candidates 8 --eval-states 3 --candidates 32 --epochs 20 --mc-trials 80
"${PY[@]}" scripts/run_with_src.py experiments/true_action_diffusion.py --seeds 1 2 3 4 --train-states 10 --train-candidates 6 --eval-states 3 --candidates 24 --horizon 8 --epochs 12 --diffusion-steps 24 --mc-trials 30 --k-values 1 4 16 --regimes id low_diversity hidden_obstacle changed_friction action_noise
"${PY[@]}" scripts/run_with_src.py experiments/pusht_benchmark.py --seeds 1 2 3 4 --train-states 10 --train-candidates 6 --eval-episodes 3 --candidates 16 --horizon 20 --epochs 12 --diffusion-steps 24 --mc-trials 20 --k-values 1 8 16 --regimes pusht_aligned pusht_low_diversity pusht_high_temp_misaligned
"${PY[@]}" scripts/claim_audit.py --fail-on-error
