#!/usr/bin/env bash
# Reduced-scale full validation matrix (executor deviation, documented).
# 5 regimes x 5 seeds on MPS, train capped ~12k / val ~4k. Validation-only:
# the test split is never opened here. See return_to_planner.json for rationale.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/splicebert_partner_selection/base.json"
DEVICE="mps"
MAX_TRAIN="${MAX_TRAIN:-12000}"
MAX_VAL="${MAX_VAL:-4000}"
EVAL_MAX="${EVAL_MAX:-4000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
REGIMES="${REGIMES:-additive-frozen additive-random-frozen interaction-frozen additive-finetuned interaction-finetuned}"
CKPT_ROOT="artifacts/splicebert_partner_selection/readouts"
CONFLICTS="results/splicebert_partner_selection/baselines/val_local_conflicts.jsonl"
LOG="results/splicebert_partner_selection/reduced_matrix_progress.log"
MODEL_DIR="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
TGZ="data/models/splicebert_official_archive/models.tar.gz"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

ensure_weights() {
  if [[ ! -f "$MODEL_DIR/pytorch_model.bin" ]]; then
    log "WARNING: model weights missing; re-extracting from tarball"
    tar -xzf "$TGZ" -C data/models/splicebert_official_archive/extracted
  fi
}

log "=== reduced matrix start (train<=$MAX_TRAIN val<=$MAX_VAL eval<=$EVAL_MAX device=$DEVICE) ==="

# Native masked-LM dependency pilot (zero-shot; cheap defaults 50 examples x 8 pairs)
ensure_weights
log "native dependency pilot"
uv run python scripts/splicebert/evaluate_native_dependency.py \
  --config "$CONFIG" --device "$DEVICE" \
  --local-baseline results/splicebert_partner_selection/baselines/local_pair_baseline.pkl \
  >> "$LOG" 2>&1 && log "native dependency OK" || log "native dependency FAILED"

for seed in $SEEDS; do
  for regime in $REGIMES; do
    ensure_weights
    ckpt="$CKPT_ROOT/${regime}/seed${seed}/checkpoint.pt"
    log "TRAIN regime=$regime seed=$seed"
    start=$(date +%s)
    uv run python scripts/splicebert/train_pair_readout.py \
      --config "$CONFIG" --regime "$regime" --seed "$seed" --device "$DEVICE" \
      --max-train "$MAX_TRAIN" --max-val "$MAX_VAL" \
      >> "$LOG" 2>&1
    rc=$?
    dur=$(( $(date +%s) - start ))
    if [[ $rc -ne 0 ]]; then log "TRAIN FAILED regime=$regime seed=$seed rc=$rc (${dur}s)"; continue; fi
    log "TRAIN done regime=$regime seed=$seed (${dur}s)"

    log "EVAL full-val regime=$regime seed=$seed"
    uv run python scripts/splicebert/evaluate_behavior.py \
      --config "$CONFIG" --checkpoint "$ckpt" --device "$DEVICE" --max-examples "$EVAL_MAX" \
      >> "$LOG" 2>&1 && log "EVAL full-val OK" || log "EVAL full-val FAILED regime=$regime seed=$seed"

    log "EVAL conflict-subset regime=$regime seed=$seed"
    uv run python scripts/splicebert/evaluate_behavior.py \
      --config "$CONFIG" --checkpoint "$ckpt" --device "$DEVICE" \
      --subset-file "$CONFLICTS" --max-examples "$EVAL_MAX" \
      >> "$LOG" 2>&1 && log "EVAL conflict OK" || log "EVAL conflict FAILED regime=$regime seed=$seed"
  done
done

log "=== reduced matrix complete ==="
