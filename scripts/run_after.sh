#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# wait for the current queue (cells C, D) to finish
while pgrep -f "train_cell.py" >/dev/null; do sleep 30; done
echo "=============== cell B, 24k steps (was cut off mid-grok at 8k)  $(date +%H:%M:%S) ==============="
uv run python scripts/train_cell.py --cell B --steps 24000 --seed 1
echo "CELL B LONG DONE $(date +%H:%M:%S)"
