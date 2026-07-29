# Results index: SpliceBERT GTEx fine-tuning shift

> **Tier: T1 — canonical.** Current, authoritative. Read this.

No target result has been admitted. Runtime-derived checkpoints live under ignored `data/models/` paths and
must carry `transplant.json` plus `transplant.manifest.json`.

Planned immutable results:

| Stage | Runner | Result |
|---|---|---|
| checkpoint/transplant preflight | `scripts/splicebert/run_gtex_finetune_preflight.py` | `results/splicebert_gtex_finetune_shift/preflight_v1.json` (PASS) |
| seed-2020 mechanics smoke | unchanged `scripts/splicebert/run_sealed_test.py` | `results/splicebert_gtex_finetune_shift/seed-2020/smoke_sealed_val_v1.failed.json` (stopped at behavior gate) |
| five-seed behavior validation | `scripts/splicebert/run_gtex_coupling_shift.py` using the unchanged canonical behavior/controls | `results/splicebert_gtex_finetune_shift/five_seed_behavior_v1.json` |

The five-seed artifact includes all 400 paired per-example values for the base and each seed, gene-grouped
bootstrap intervals, 9,999 gene-level sign flips with add-one p-values, BH correction across the 15
seed-by-contrast tests, and the three family-mean contrasts. It is verification tier because the comparison
runner was finalized after the seed-2020 mechanics smoke.

Every target artifact must be added here and to `results/MANIFEST.{csv,json}` only after its status and
claim scope are decided.
