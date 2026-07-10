#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for OBJ in ar mlm; do
  echo "=============== $OBJ mixed  $(date +%H:%M:%S) ==============="
  uv run python scripts/train_nano.py --objective $OBJ --pool mixed --steps 8000
done
echo "MIXED DONE $(date +%H:%M:%S)"
