# Decisions

Append entries; do not rewrite history.

## 2026-07-10 — Initial implementation

- Chose partner-aware internal-exon acceptor–donor selection as the first real-GFM target.
- Chose official SpliceBERT as the primary checkpoint.
- Registered `additive-frozen` as the primary mechanistic condition.
- Classified bilinear readouts as diagnostic because they can synthesize interactions.
- Reserved chromosome-held-out test data for one-time confirmatory evaluation.
- Classified endpoint masking and exon-core shuffling as instrumentation, not sufficient partner-specific proof.

## 2026-07-11 — Behavioral gate: GO

- Fixed the `additive-random-frozen` control (transformers ≥5 `init_weights()` no-op → `from_config`); this converted a spurious NO-GO into a true GO. All 5 control seeds re-run.
- Ran the 25-run matrix on Colab GPU with scale caps (train 12k / val 4k / eval 4k); gates, controls, and metric unchanged.
- **Behavioral gate met (recommended CALL: GO):** additive-frozen conflict PMR 0.807 vs randomized-backbone control 0.056 (non-overlapping, 5/5 seeds); gene-grouped bootstrap margin-difference CI excludes 0 on every seed (worst CI low 2.84); conflict PMR ≥ 0.60 and replication 5/5 both met. Interaction readout adds nothing over additive → partner info is linearly accessible in frozen states.
- Froze the preregistration + method + config + control-fix script + gate analysis (hashes in `FREEZE.md`). Test split (chr19–chr22) remains sealed.
- **Human elected to STOP before circuit tracing** for review; the mechanistic stage is not yet authorized (it opens new compute and must freeze the component set first).
- Scope discipline: this stage establishes decodable, pretraining-derived partner information — not yet a localized circuit the model causally uses. That is the tracing stage's job.
