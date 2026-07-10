#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for LAM in 0.75 1.0 1.5 2.0; do
  echo "=============== lambda $LAM  $(date +%H:%M:%S) ==============="
  uv run python scripts/train_uniform.py --lam $LAM --steps 8000 || echo "ARM $LAM FAILED"
done
echo "SWEEP DONE $(date +%H:%M:%S)"
