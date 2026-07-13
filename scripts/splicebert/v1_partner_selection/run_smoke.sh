#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/splicebert_partner_selection/base.json}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
MAX_TRAIN="${MAX_TRAIN:-64}"
MAX_VAL="${MAX_VAL:-32}"

uv run pytest -q \
  tests/test_splicing_schema_candidates.py \
  tests/test_splicing_controls.py \
  tests/test_splicing_readout.py \
  tests/test_splicing_dataset.py

uv run python scripts/splicebert/build_dataset.py \
  --config "$CONFIG" \
  --max-examples "$MAX_EXAMPLES"

uv run python scripts/splicebert/evaluate_local_baselines.py \
  --config "$CONFIG"

uv run python scripts/splicebert/train_pair_readout.py \
  --config "$CONFIG" \
  --regime additive-frozen \
  --seed 0 \
  --max-train "$MAX_TRAIN" \
  --max-val "$MAX_VAL"

echo "Smoke pipeline completed. Do not treat its metrics as scientific results."
