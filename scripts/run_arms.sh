#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for OBJ in ar mlm; do
  echo "=============== $OBJ  $(date +%H:%M:%S) ==============="
  uv run python scripts/train_nano.py --objective $OBJ --steps 8000
done
echo "ALL DONE $(date +%H:%M:%S)"
