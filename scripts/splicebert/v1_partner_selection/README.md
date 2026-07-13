# v1 partner-selection pipeline — RETRACTED (R2/R3). Provenance only.

**Do not run. Do not cite the results these produced.**

This is the pipeline for the **partner-selection** task, retracted in full:

- **R2** — a *positional label leak*: `dataset.py:135` pinned the acceptor to a constant
  offset, so it sat at position 200 in **100%** of examples. A **geometry oracle reading zero
  nucleotides scores PMR 1.000**. Separately, an **oracle site detector also scores 1.000** —
  the task never required partner information at all.
- **R3** — the v2 fix (real annotated decoys) failed for a *biological* reason: in a real
  genome a strong donor at a plausible exon distance from an acceptor generally **is** its
  partner. Clean negatives barely exist, so the forced-choice task is intrinsically degenerate.
  A plain PWM scores **0.687** vs 0.392 chance.

These scripts are kept because `CLAIMS.md` and the retractions cite their outputs (under
`results/splicebert_partner_selection/`), and claim **SM1** cites `calibration_audit.json`.
The evidence that killed them is in `../audits/`.

**The live science is the coupling study.** Entry point: `../run_sealed_test.py`.
Document of record: `studies/splicebert_partner_selection/FINDINGS.md`.
