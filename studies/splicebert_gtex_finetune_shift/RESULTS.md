# Results index: SpliceBERT GTEx fine-tuning shift

> **Tier: T1 — canonical.** Current, authoritative. Read this.

The fixed-MLM gate is closed, while the native-classifier behavior and its gated representation bridge
are complete. Runtime-derived checkpoints live under ignored `data/models/` paths and must carry
`transplant.json` plus `transplant.manifest.json`.

Immutable results:

| Stage | Runner | Result |
|---|---|---|
| checkpoint/transplant preflight | `scripts/splicebert/run_gtex_finetune_preflight.py` | `results/splicebert_gtex_finetune_shift/preflight_v1.json` (PASS) |
| seed-2020 mechanics smoke | unchanged `scripts/splicebert/run_sealed_test.py` | `results/splicebert_gtex_finetune_shift/seed-2020/smoke_sealed_val_v1.failed.json` (stopped at behavior gate) |
| five-seed behavior validation | `scripts/splicebert/run_gtex_coupling_shift.py` using the unchanged canonical behavior/controls | `results/splicebert_gtex_finetune_shift/five_seed_behavior_v1.json` |
| native classifier preflight v1 | `scripts/splicebert/run_gtex_native_classifier_preflight.py` at the retracted first-intronic-base coordinate | `results/splicebert_gtex_finetune_shift/native_classifier_preflight_v1.json` (FAIL; preserved instrument error) |
| corrected native classifier preflight v2 | same runner, last-exonic-base donor label frozen before rerun | `results/splicebert_gtex_finetune_shift/native_classifier_preflight_v2.json` (51/51 PASS) |
| five-seed native classifier coupling | `scripts/splicebert/run_gtex_native_classifier_coupling.py` | `results/splicebert_gtex_finetune_shift/native_classifier_coupling_v1.json` (behavior gate PASS) |
| gated layerwise representation bridge | `scripts/splicebert/run_gtex_native_representation_bridge.py` | `results/splicebert_gtex_finetune_shift/native_representation_bridge_v1.json` (representation candidate) |

The five-seed artifact includes all 400 paired per-example values for the base and each seed, gene-grouped
bootstrap intervals, 9,999 gene-level sign flips with add-one p-values, BH correction across the 15
seed-by-contrast tests, and the three family-mean contrasts. It is verification tier because the comparison
runner was finalized after the seed-2020 mechanics smoke.

Every target artifact must be added here and to `results/MANIFEST.{csv,json}` only after its status and
claim scope are decided.

The native behavior result resolves the central ambiguity in GFS4: coupling was not erased from the
released fine-tuned model. It remains visible to the supervised head, while the original MLM readout
becomes broadly incompatible. The layerwise bridge places the classifier-aligned change primarily at the
last encoder block, but this is a representation/readout result rather than a causal layer intervention.
