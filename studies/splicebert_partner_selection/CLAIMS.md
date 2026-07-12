# Claims registry — splicebert_partner_selection

One row per empirical claim. **A passing unit test verifies software, never a
scientific claim** — it can advance a claim to `unit-tested`, never to `supported`.
Only a fresh, manifest-stamped artifact advances a claim to `supported`. Downgrades
and retractions are permanent rows; later analyses refine, they do not overwrite.

Status vocabulary: `open` · `implemented` · `unit-tested` · `artifact-supported` ·
`pending-fresh-run` · `retired` · `retracted`.

## Scientific claims

| ID | Claim | Status | Evidence / what would advance it |
|----|-------|--------|----------------------------------|
| C1 | On natural cue conflicts, `additive-frozen` selects the true partner far above the local PWM+length baseline (~0 on the conflict set by construction). | **artifact-supported** | `colab_results/gate_bootstrap.json`, `summary.csv`: conflict PMR 0.807 [0.802–0.812], 5/5 seeds. |
| C2 | The partner information is **linearly/additively accessible** at the site endpoints (an interaction readout does not beat the additive one). | **artifact-supported** | `summary.csv`: interaction-frozen 0.77–0.80 ≈ additive-frozen 0.80. |
| C3 | The effect is **attributable to pretraining**, not to readout capacity/training. | **artifact-supported** | `gate_bootstrap.json`: randomized-backbone control collapses to 0.056, gene-grouped margin-diff CI excludes 0 on all 5 seeds (worst CI-low 2.84). |
| C4 | A **compact, localized head circuit** is necessary AND mediates a partner-specific corruption AND is conflict-specific. | **pending-fresh-run** | Infra built, instrument- and calibration-tested; needs the 5-seed GPU tracing run (`run_tracing.py`). Advances on ≥3/5 replication with FDR-controlled necessity + mediation ≥0.30 (partner > context) + specificity. |
| C5 | The model **uses** this circuit in its native masked-LM computation (not just probe-decodable). | **open** | Out of the registered scope; needs an MLM-native readout arm. Explicitly NOT claimed by C1–C4. |
| C6 | The frozen circuit score is a **biological predictor** — adds value beyond local controls on independent splicing data. | **open** | `EXTERNAL_VALIDATION.md` phase; not started. Requires held-out external data never used for component selection. |
| C7 | The behavioral advantage survives a **receptive-field-limited CNN / composition** rival, not only the PWM+length baseline. | **open** | Rival not implemented (RRR finding 7). PWM+length is the current cheap rival; a composition/RF-CNN baseline would harden C1. |

## Statistical-machinery claims

| ID | Claim | Status | Evidence |
|----|-------|--------|----------|
| SM1 | The matched-random-null p-value + BH-FDR path is **calibrated** (uniform max-p, ~0 false discoveries on pure noise). | **artifact-supported** | `results/.../circuits/calibration_audit.json` (KS-uniform maxima, 0 FDR discoveries); `tests/test_rigor_calibration.py`. |
| SM2 | The head-intervention instrument is correct (round-trip patch identity; AtP tracks exact patching). | **unit-tested** | `tests/test_circuits_hf_bert.py` (9), `tests/test_attribution.py` (Spearman 0.876 vs exact). Advances to supported only via a manifest-backed run. |

## Retractions (permanent)

| ID | Retracted claim | Date | Reason |
|----|-----------------|------|--------|
| R1 | "additive-random-frozen control ≈ additive-frozen, so the behavioral advantage is NOT pretraining-specific (NO-GO)." | 2026-07-11 | The control was a **no-op**: `init_weights()`/`apply(_init_weights)` do nothing after `from_pretrained` on transformers ≥5, so the "random" backbone was the pretrained one (byte-identical results). Fixed via `AutoModelForMaskedLM.from_config`; the corrected control collapses to 0.056. Superseded by C3. |
