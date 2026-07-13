#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/splicebert_partner_selection/base.json}"
SEEDS="${SEEDS:-0 1 2 3 4}"
REGIMES="${REGIMES:-additive-frozen additive-random-frozen interaction-frozen additive-finetuned interaction-finetuned}"

# Deliberately validation-only. Test unblinding is never performed here.
uv run python scripts/splicebert/evaluate_local_baselines.py --config "$CONFIG"

uv run python scripts/splicebert/evaluate_native_dependency.py \
  --config "$CONFIG" \
  --local-baseline results/splicebert_partner_selection/baselines/local_pair_baseline.pkl

for seed in $SEEDS; do
  for regime in $REGIMES; do
    uv run python scripts/splicebert/train_pair_readout.py \
      --config "$CONFIG" \
      --regime "$regime" \
      --seed "$seed"

    checkpoint="artifacts/splicebert_partner_selection/readouts/${regime}/seed${seed}/checkpoint.pt"
    uv run python scripts/splicebert/evaluate_behavior.py \
      --config "$CONFIG" \
      --checkpoint "$checkpoint"
    uv run python scripts/splicebert/evaluate_behavior.py \
      --config "$CONFIG" \
      --checkpoint "$checkpoint" \
      --subset-file results/splicebert_partner_selection/baselines/val_local_conflicts.jsonl
  done
done

echo "Validation matrix complete. Record the stop/go decision before tracing components."
