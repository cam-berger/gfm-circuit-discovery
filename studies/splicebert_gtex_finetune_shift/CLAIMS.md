# SpliceBERT GTEx fine-tuning-shift claim ledger

> **Tier: T1 — canonical.** Current, authoritative. Read this.

| ID | Current licensed claim | Evidence level | Controlling artifact | Scope / next bridge |
|---|---|---|---|---|
| GFS1 | The published family contains five raw token-classification state dictionaries whose documented training starts from `SpliceBERT.1024nt`. | source/provenance | upstream repository plus frozen hash config | Validate exact tensor contract and forward mechanics. |
| GFS2 | The derived-checkpoint instrument passed all 20 frozen preflight checks, including exact five-seed hashes/encoder coverage, tied-readout mechanics, and base positive/negative behavior controls. | established: instrument | `results/splicebert_gtex_finetune_shift/preflight_v1.json` | Instrument validity does not establish a fine-tuning effect. |
| GFS3 | A seed-2020 outcome-limited smoke showed near-zero mean coupling and could not form the requested mechanism cohort; no representational-change claim is licensed from the smoke. | exploratory smoke / stopped | `results/splicebert_gtex_finetune_shift/seed-2020/smoke_sealed_val_v1.failed.json` | Run the paired behavior-only validation family across all five seeds. |
| GFS4 | Through the derived fixed MLM readout, GTEx fine-tuning nearly eliminates the base donor–acceptor coupling effect across all five seeds: base `+0.958 [0.861,1.058]`; fine-tuned seeds `−0.029` to `+0.068`; family paired shift `−0.932 [−1.038,−0.835]`, negative 5/5, add-one/BH `q=0.0001`. The partner-specific contrasts fall on the same scale. | established: behavior/readout shift, verification tier | `results/splicebert_gtex_finetune_shift/five_seed_behavior_v1.json` | Not coupling-specific: general MLM health falls from `0.523` to `0.209–0.309`; does not establish encoder erasure or native-classifier behavior. |
| GFS5 | Frozen base-circuit transfer is not interpretable on these derived checkpoints because the prerequisite MLM coupling behavior and the `≥0.3` mechanism denominator gate fail. | closed at behavior gate | GFS3/GFS4 artifacts | A native three-class-head assay or independently validated readout would be a new study arm. |

Parameter displacement is not a causal or behavioral result. Smoke runs are instrument checks and may not
promote a claim. GFS4 is about visibility through the derived MLM readout; “fine-tuning erased coupling
from the encoder” is not licensed.
