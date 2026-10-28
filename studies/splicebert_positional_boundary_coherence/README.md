# SpliceBERT positional boundary coherence

The model-free Phase-I construction ran on 2026-08-11 and stopped on an independently audited
frozen-capacity failure. No successor checkpoint or model outcome was opened.

## Purpose

The study asks one narrow mechanistic question:

> After prespecified balance of phase-correlated boundary sequence, does the learned period-3 positional
> carrier causally support SpliceBERT's preference for matched annotated coding-boundary phase?

The primary target is the period-3 attenuation bridge. A corrected directional Phase-II result can
support a rough, assay-scoped link; clearing the frozen materiality threshold is a stronger tier and
alone opens Phase III. Register relocation and route mediation are optional strengthening tests, not
prerequisites for the scoped bridge. The distinct ≈10.63 carrier is an ancillary interaction arm whose
origin and general biological meaning need not be resolved for the period-3 result to close. The intended
interpretation is a sequence-encoded coding-boundary coherence prior, not reading-frame sensing,
translation, or spliceosomal biochemistry.

## Design

[`PREREGISTRATION.md`](PREREGISTRATION.md), with Amendments A1–A5, holds the full design and numerical
rules. Every model outcome comes from the base `SpliceBERT.1024nt` checkpoint; `SpliceBERT.510nt` and the
GTEx-fine-tuned checkpoints supply provenance only. Every model run's manifest records the checkpoint
identity, and a mismatch is fatal. The behavior is the change in the masked-donor native logit margin
when the joined true acceptor is corrupted `AG→AC`. The primary contrast `Ψ` compares matched with
mismatched annotated boundary phase, after composition balance, in a chimeric 3×3 lattice of donor and
acceptor phases at a fixed 120-nt span.

Prior records supply motivation, blacklists, and precedent, within limits. `C43` gives the behavioral
motivation, sign-free sizing variance, and readout geometry, but not its target sign or magnitude or
"independent replication" language. `C44` gives a failed-instrument warning, not its development `Ω`,
route signs, or failed local states as calibration. `C36`, `C37`/`C12`, and `GFS25` give carrier and dose
precedent, not coding specificity, a `U_info` write, a general biological function, or an imported
confirmatory status. Their wording stays in the donor
[`CLAIMS.md`](../splicebert_partner_selection/CLAIMS.md) and
[`CORRECTIONS.md`](../splicebert_partner_selection/CORRECTIONS.md) (`R45`), and for `GFS25` in the
[GTEx ledger](../splicebert_gtex_finetune_shift/CLAIMS.md), and the archived exon-edge
code at `fd26cc1b` is never imported or executed.

Phase I builds fresh, gene-disjoint blocks from sequence and annotation only. It excludes every gene
exposed in the `C43`, `C44` and `GFS25` cohorts, keeps `chr8` and `chr9` sealed with no fallback to them,
and must pass composition-balance and calibration gates before any model access. Failure is
`INSTRUMENT_UNRESOLVED`, not evidence against the biology. Phase II attenuates the carriers in four
identical-sequence worlds (`Y00`, `Y10`, `Y01`, `Y11`). Its primary endpoints are `Ψ00` and
`N3 = Y00 − Y10`, its ancillary ones `N10` and `J`. If period-3 passes its capability gates but ≈10.63
fails, only `Y00/Y10` may open and P10/N10/J are `OPTIONAL_INSTRUMENT_UNRESOLVED`. Phase III is optional,
opens only after a material, composition-balanced `Ψ00` and a corrected, null-specific material `N3`, and
tests span-wide register relocation and route mediation with every candidate frozen before the
confirmation role opens.

## Decision hierarchy

| Highest passing result | Maximum licensed statement |
|---|---|
| Balanced `Ψ00` equivalent to zero | No material residual annotated-phase preference is detected after the prespecified measured composition balance in this fresh domain |
| Balanced `Ψ00` positive; capable `N3` bounded/null | The preference persists, but material dependence on the tested period-3 attenuation is not detected |
| Corrected `Ψ00` and `N3` are directionally resolved and specific, but one or both do not establish `>0.05` materiality | `P3_DIRECTIONAL_SCOPED_SUPPORT`: a rough, assay-scoped P3–boundary-preference link; no materiality or Phase-III claim |
| Balanced `Ψ00` material; corrected `N3` material and specific | `P3_NECESSITY_SUPPORTED`: period-3 materially and causally supports the composition-balanced preference under the tested attenuation; Phase III may open |
| Phase III has a material aggregate response, all three register estimates point as predicted, at least two are simultaneously resolved, and handedness/specificity pass | `P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED`: partial concordance strengthens the scoped P3–boundary link; complete cyclic relocation remains unresolved |
| P3 register relocation passes; route fails | Period-3 relocates the preference; the declared route is unresolved |
| P3 relocation plus path necessity, restoration, and separate sufficiency pass | A period-3 relative register causally implements the scoped preference through the frozen model route |
| `N10` and corrected `J` also pass, followed by joint mediation | Period-3 and ≈10.63 interact in the scoped model computation; the sign and route are reported without a biological identity claim |

No outcome licenses “SpliceBERT reads frame,” “the spliceosome checks phase,” “the model implements
translation,” “≈10.63 is nucleosomal/α-helical,” or “period-3 and ≈10.63 are one oscillator.”

## Code and artifact contract

New code goes only in `scripts/splicebert_positional_boundary_coherence/`, the flat modules
`src/gfmcd/splicing/positional_boundary_coherence.py` and `src/gfmcd/splicing/pbc_*.py`,
`results/splicebert_positional_boundary_coherence/`, and `tests/`. Nothing is imported from the archived
exon-edge program. Every result-producing stage writes an immutable JSON result with per-block endpoints,
a sibling manifest, a freeze/authorization artifact for any later stage, and an independent audit that
does not import the analysis verdict function. The compact endpoint table, manifest, and audit must make
each conclusion checkable from a clean checkout.
An audit that depends on an uncommitted raw shard must say so and cannot be called clean-checkout re-executable.

## Executable gate chain

Each row is fail-closed. A producer cannot authorize its own dependent target, and no later stage infers
a positive, negative, or bounded result from a failed instrument.

| Stage | Producer | Independent check / authorization |
|---|---|---|
| Freshness | `build_freshness_registry.py` | `audit_freshness_registry.py` |
| Real-sequence census | `build_candidate_census.py` | `audit_candidate_census.py` |
| Balanced cohort | `build_composition_balanced_cohort.py` | `audit_composition_balanced_cohort.py` |
| Pre-target protocol/code conformance | `audit_protocol_code_conformance.py` | its passing artifact and manifest are required and hash-bound by the Phase-II freezer |
| P3-first attenuation development | `calibrate_carrier_factorial.py`, then `run_carrier_factorial_preflight.py` | `audit_carrier_factorial_preflight.py`, then `freeze_carrier_factorial_target.py` |
| Attenuation confirmation | `run_carrier_factorial_target.py`, `analyze_carrier_factorial.py` | `audit_carrier_factorial.py` |
| P3 register development | `run_p3_register_preflight.py`, `analyze_p3_register_preflight.py` | `audit_p3_register_preflight.py` |
| Route discovery and freeze | `run_p3_register_candidate_screen.py`, `freeze_p3_register_candidates.py` | `audit_p3_register_candidate_freeze.py` |
| Reserve seal | `freeze_p3_register_target.py` | the seal binds the audited P3-only or optional joint authorization, candidate set, code, inputs, and every output path |
| Register confirmation | `run_p3_register_target.py`, `assemble_p3_register_target.py`, `analyze_p3_register_relocation.py` | `audit_p3_register_relocation.py` |
| Route mediation | `run_p3_register_mediation_target.py`, `assemble_p3_register_mediation_target.py`, `analyze_p3_register_mediation.py` | `audit_p3_register_mediation.py` |

## Status

Stopped. Phase I executed and stopped instrument-unresolved before any model outcome: the A5 certificate
and its 48-check independent audit show that donor phase 2 and acceptor phase 2 cannot supply the frozen
1,104-block request (`PBC1`). No complete cohort exists, no model preflight ran or opened, no target is
authorized, and no model endpoint was measured. Phases II and III are unauthorized. `PBC1` is an
instrument result only and licenses no statement about coding-boundary preference, either positional
carrier, or a biological mechanism. Any successor design must be separately preregistered; this frozen
execution path remains closed.

## Files

- [`PREREGISTRATION.md`](PREREGISTRATION.md): the frozen design. Read it with the canonical
  [`METHODOLOGY.md`](../../docs/METHODOLOGY.md), its [evidence
  ladder](../../docs/METHODOLOGY.md#evidence-ladder), and
  [`PREREG_CONFORMANCE.md`](../../docs/PREREG_CONFORMANCE.md).
- [`HYPOTHESIS_REGISTER.md`](HYPOTHESIS_REGISTER.md): the candidate biological narrative and checkpoint
  map, kept apart because it cannot promote a claim.
- [`CLAIMS.md`](CLAIMS.md), [`RESULTS.md`](RESULTS.md), [`DECISIONS.md`](DECISIONS.md): the successor
  ledger, results, and decisions, which record the construction stop as `PBC1`.
