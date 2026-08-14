# SpliceBERT reproducibility map

The shortest path from a claim to its script and machine-readable artifact for the SpliceBERT donor-circuit
study (C1–C42 + IC1), covering the full current arc. **C43 and C44 are deliberately absent from the map
below:** they were consolidated from the archived exon-edge arc (R45) and the runners that produced their
artifacts are not in this tree — each artifact's sibling `*.manifest.json` pins its runner by path and
sha256, recoverable from `codex/intron-exon-phase` at `fd26cc1b`. Neither is re-executable regardless: the
source branch gitignored the raw rows both audits recompute from. The authoritative claim ledger is
[`CLAIMS.md`](CLAIMS.md); correction/retraction history (R1–R47) is in [`CORRECTIONS.md`](CORRECTIONS.md);
the machine-checkable claim→file index is [`../../results/MANIFEST.csv`](../../results/MANIFEST.csv)
(953 rows, all paths resolve; `scripts/verify_manifest.py` passes). Repository-wide environment,
checkpoint, and data requirements are defined in [`../../REPRODUCE.md`](../../REPRODUCE.md); this file is
the study-level command map, detailed per claim family, and consistent with that top-level table. The
script catalogue/index is [`../../scripts/README.md`](../../scripts/README.md).

Artifacts are under `results/`; each active runner writes an immutable JSON plus a manifest. Paths below
are relative to the repository root and abbreviated to the family directory where unambiguous
(`SBC = results/splicebert_coupling/`, `PL = results/phase-lattice/`). Large models and source datasets
are not committed; review the corresponding manifest before asserting a rerun matches the recorded
environment.

## Environment

```bash
uv sync --extra splicebert --extra dev
uv run pytest
uv run python scripts/splicebert/<runner>.py --help
```

Use `PYTHONHASHSEED=0` for cohort construction and any deterministic shuffle/selection path;
`TOKENIZERS_PARALLELISM=false` for quiet deterministic tokenization. Historical records naming
`gfm-thrust1` are execution provenance only.

## Provenance contract

Every scientific run records: checkpoint and tokenizer; source dataset and split; inclusion/exclusion
rules; seed, sample size, and intervention parameters; exact command and code revision; preflight values
and frozen thresholds; output artifact and manifest. Do not overwrite a scientific result — write a new
artifact or intentionally supersede the old one in the index and claim ledger.

---

## Sealed behaviour and the {L3H1, L2H15} subcircuit (C1–C7)

| Claim | Canonical script | Indexed output |
|---|---|---|
| C1 raw coupling margin | `run_coupling.py` | `SBC/confirmation.json` |
| C1/C2/C3/C4 sealed confirmation + specificity | `run_sealed_test.py` | `SBC/sealed_test.json`, `SBC/sealed_val.json` (+ manifests) |
| C3 ablation scan (development) | `run_ablation_scan.py` | `SBC/development/ablation_scan.json` |
| C4 subspace (development) | `run_subspace.py` | `SBC/development/subspace.json` |
| C5 L3H1 write position + downstream readout path | `run_position_profile.py`, `run_readout_path.py`, `run_localization.py`, `run_path_decomposition.py` | `SBC/position_profile.json`, `SBC/readout_path.json` |
| C6 L2H15/L1H12 accounting | `run_l2_accounting.py`, `run_l1h12_probe.py`, `run_l1h12_route.py`, `run_l2_rescue.py` | `SBC/l2_accounting.json`, `l1h12_probe.json`, `l1h12_route.json` |
| C7 early completeness assay (uninformative → C27/C33) | `run_completeness_null.py`, `run_faithfulness.py` | `SBC/completeness_null.json`, `faithfulness.json` |

The sealed test requires the explicit unsealing flag in
[`preregistrations/PREREGISTRATION_SEALED.md`](preregistrations/PREREGISTRATION_SEALED.md). Do not rerun
it merely as a smoke test.

## Donor readout / `U_info` and behavioural signature (C16–C19)

| Claim | Canonical script | Indexed output |
|---|---|---|
| C16 low-rank `U_info` mediation | `run_donor_anchor.py` | `SBC/donor_anchor.json` |
| C17 L3H1 recovery routed through `U_info` | `run_donor_forward.py` | `SBC/donor_forward.json` |
| C18 selective downstream transporter heads | `run_donor_heads.py` | `SBC/donor_heads.json` |
| C19 behavioural signature (partner/donor-specific, strand-symmetric, phase-insensitive) | `run_donor_signature.py` | `SBC/donor_signature.json` |

## Address / payload donor graph A0→A12 (C20–C33)

| Claim | Canonical script | Indexed output |
|---|---|---|
| C20 acceptor address (positional, not identity; R20) | `run_acceptor_address.py` | `SBC/acceptor_address.json` |
| C21 targeting provenance (embeddings/layers 0–2) | `run_a0_targeting_provenance.py` | `SBC/a0_targeting_provenance.json` |
| C20;C22 token vs position within key window | `run_a1_token_position.py` | `SBC/a1_token_position.json` |
| C22 QK/V decomposition | `run_qkv.py` (+ `run_a1_token_position.py`) | `SBC/qkv.json` |
| C23 address×payload factorial | `run_a2_address_payload.py` | `SBC/a2_address_payload.json` |
| C24 payload localization (4-token acceptor window) | `run_a2b_payload_localization.py` | `SBC/a2b_payload_localization.json` |
| C25 L2H15 acts through L3H1's address channel | `run_a3_l2h15_address_payload.py` | `SBC/a3_l2h15_address_payload.json` |
| C26 `U_info` replication (disjoint genes) | `run_a5_uinfo_replication.py` | `SBC/a5_uinfo_replication.json` |
| C27 circuit necessary/sufficient but non-faithful (R21) | `run_a6_circuit_faithfulness.py` | `SBC/a6_circuit_faithfulness.json`, `a6_reconcile.json` |
| C28 generation acceptor-local + serial | `run_a7_source_localization.py` | `SBC/a7_source_localization.json` |
| C29 L0→L1→MLP2 generator backbone | `run_a8_generator_path.py` | `SBC/a8_generator_path.json` |
| C30 payload not bulk composition; semantics unresolved (R22) | `run_a9b_payload_semantics_composition.py` | `SBC/a9b_payload_semantics_composition.json`, `a9c_matched_composition_reconcile.json`, `a9c_sweep_seeded_draws.json` |
| C31 value-path wiring `{L0H7,L0H12,L0H10}→MLP0→MLP1→MLP2` | `run_a10_generator_edges.py` | `SBC/a10_generator_edges.json` |
| C32 address predominantly relative-offset | `run_a11_address_geometry.py` | `SBC/a11_address_geometry.json` |
| C33 no sparse faithfulness knee (~72/103 nodes) | `run_a12_faithfulness_size.py` | `SBC/a12_faithfulness_size.json` |

Superseded within this arc (retained in `SBC/superseded/`): `run_a4_address_geometry.py` (A4 → C32),
`run_a9_payload_semantics.py` (A9 splice-strength, withdrawn R22). `run_b2_payload_steering.py`
(`PREREGISTRATION_B2_PAYLOAD_STEERING.md`) is the C30 semantic close-attempt — red-teamed survivor=NO,
**no committed artifact / no MANIFEST row** (C30 CLOSED-BOUNDED).

## Intron-coupling arc (IC1)

| Claim | Canonical script | Indexed output |
|---|---|---|
| IC1 distal intron arrangement (sub-SESOI, no partner-specific component) | `run_ic1_distal_intron_necessity.py` (pilots `run_ic1_pilot{,2,3,4_ood}.py`) | `SBC/ic1_distal_intron_necessity_v8.json` |

IC1 is outcome-aware deterministic **verification**, not confirmatory (see
`preregistrations/PREREGISTRATION_IC1_DISTAL_INTRON_NECESSITY.md` + amendments). IC2
(`run_ic2_feasibility.py`, `run_ic2_calibration.py`) is RETIRED (design asset only).

## Period-3 carrier — structure, deletion, mediation, present-day use (C8–C10, C34–C36)

| Claim | Canonical script | Indexed output |
|---|---|---|
| C8,C11 position-table inventory + provenance | `run_phase_inventory.py`, `run_carrier_provenance.py` | `PL/phase_inventory.json`, `carrier_provenance.json` |
| C8 period-3 deletion / dose / comb-shift | `run_carrier_surgery.py`, `run_carrier_dose.py`, `run_comb_shift.py` | `PL/carrier_surgery.json`, `carrier_dose.json`, `comb_shift.json` |
| C9,C10 size-matched restoration | `run_carrier_mediation.py` | `PL/carrier_mediation.json` |
| C34 carrier locus (spatially distributed; R24 confirm gate-qualified) | `run_p3_carrier_locus.py` | `results/p3_carrier_locus.json` |
| C35 A-vs-V routing (attention ≫ values; R23/R24) | `run_p3_av_decomposition.py` | `results/p3_av_decomposition.json` |
| C36 clean nucleotide-invariant confirmation (Ψ_C=+0.141) | `run_p3_phase_surgery_confirmation.py` + cohort/preflight/health/controls/analysis/audit/freeze chain | `PL/p3_phase_surgery_confirmation_v1/confirmation_analysis.json`; `PL/p3_phase_surgery_confirmation_package_v1.json` (+ audits) |
| P3-shift-diagnostic (v2, mod3 phase) | `run_p3_shift_diagnostic.py`, `analyze_p3_shift_mod3.py` | `PL/p3_shift_diagnostic_v2.json` (+ `.mod3.json`) |

C36's full cohort/preflight machinery: `build_p3_closure_cohort.py`, `run_p3_closure_preflight.py`,
`analyze_p3_closure_preflight.py`, `build_p3_phase_surgery_cohort.py`,
`build_p3_phase_surgery_non_cds_cohort.py`, `match_p3_phase_surgery_contexts.py`,
`run_p3_phase_surgery_preflight.py`, `analyze_p3_phase_surgery_preflight.py`,
`run_p3_phase_surgery_health.py`, `run_p3_phase_surgery_controls.py`,
`analyze_p3_phase_surgery_controls.py`, `analyze_p3_phase_surgery_context_difference.py`,
`analyze_p3_phase_surgery_confirmation.py`, `audit_p3_phase_surgery_confirmation_smoke.py`,
`audit_p3_phase_surgery_confirmation_package.py`, `freeze_p3_phase_surgery_confirmation.py`. Development
and superseded cohorts live under `PL/development/` and `PL/superseded/`. The period-3 present-day-use
family (C34/C35/C36) is DONE; only training/biological origin remains (deferred). Figure 5 via
`make_fig5_p3_phase_surgery_confirmation.py` → `figures/splicebert/fig5_p3_phase_surgery_confirmation.png`.

## ≈10.63 carrier and human-only ≈10.2 attention gate (C11–C13, C37)

| Claim | Canonical script | Indexed output |
|---|---|---|
| C11 carrier geometry / sub-bin wavelength / second carrier | `run_carrier_geometry.py`, `run_carrier_subbin.py`, `run_second_carrier.py` | `PL/carrier_geometry.json`, `carrier_subbin.json`, `second_carrier.json` |
| C11,C13 gated post-softmax screen + eligibility mass | `run_postsoftmax.py`, `measure_band_mass.py` | `PL/postsoftmax.json`, `preflight_postsoftmax.json`, `band_mass.json` |
| C13 human-only QK checkpoint comparison | `run_human_carrier.py` | `PL/human_carrier.json` |
| C37 donor-scoped OV write + P10 behaviour | `run_carrier_ov_av.py`, `run_carrier_p10_behaviour.py`, `run_carrier_preflight_r.py` | `PL/carrier_ov_av.json`, `carrier_p10_behaviour_confirm.json`, `carrier_preflight_r.json` |
| C12 general-OV bounded null (terminal) | `run_carrier_ov_general.py` | `PL/carrier_ov_general.json` |

Post-softmax is interpreted with exact add-one permutation values from the report-data path
(`make_fig_report_data.py` → `PL/fig_report_data.json`, C8,C11,C13,C14), not raw zero-hit values. The
human-only conclusion is zero Level-A heads; multispecies is eight (C11/C13). C37 is folded into C12 in
the ledger (no standalone row); the general-OV read is a diagnostic, not a preregistered equivalence.

## Nucleosomal / biological-source assays (C14–C15)

| Claim | Canonical script | Indexed output |
|---|---|---|
| C14 real nucleosomal ≈10.298 nt (positive biological assay) | `run_dyad_fpr.py`, `run_dyad_real.py`, `run_dyad_wavelength.py` | `PL/dyad_fpr.json`, `dyad_real.json`, `dyad_wavelength.json` |
| C15 unanchored/splice-window sensitivity limits | `run_corpus_spectrum.py`, `run_splice_profile.py`, `run_splice_phasing.py`, `run_splice_long.py` | `PL/corpus_spectrum.json`, `splice_profile.json`, `splice_phasing.json`, `splice_long.json` |

The unanchored spectrum is a calibrated sensitivity limit, not evidence of no biological periodicity.

## Cross-example transduction (C38, C41)

| Claim | Canonical script | Indexed output |
|---|---|---|
| C38 acceptor-side payload transfer (bounded / INCONCLUSIVE at val precision) | `run_b3_cross_example_transduction.py` | `SBC/b3_cross_example_transduction_n512.json` (+ base/`_pilot`/`_smoke`) |
| C41 donor `U_info` state transfer — GENERIC READOUT AXIS (verification tier on Atlas `H_conf`; not confirmatory — R28/R32) | `run_g1_cross_example_transduction.py` | `SBC/g1_cross_example_transduction.json` (+ `_pilot`/`_decomp`/`_smoke`) |

C41 bounds only the C16/C26 final-residual delivered state via additive rank-8 `U_D` injection; it does
not bound C17/C18/C23 addressing/routing. See `preregistrations/` §G1 + `redteam/REDTEAM_SYNTHESIS_G1.json`
and `G1_REPORT.md`.

## Origin — corpus attribution and cross-model absence (C39, C40)

| Claim | Canonical script | Indexed output |
|---|---|---|
| C39 72-species vs human-only sibling P-spectrum (established) + band-deletion exon-selective damage pattern (verification level, R33) | `run_sibling_pspectrum.py`, `run_sibling_carrier_bandtransfer.py`, `run_sibling_validate.py` | `results/origin-siblings/canonical/sibling_pspectrum.json`, `sibling_carrier_bandtransfer.json`, `sibling_validate.json` |
| C40 cross-model powered null (RNA-FM/RNA-BERT) + DNABERT-2 tokenizer aliasing | `scripts/crossmodel/run_carrier_pspectrum.py`, `run_reader_and_surgery.py`, `run_reader_freqcontrol.py`, `run_dnabert2_aliasing.py` | `results/origin-crossmodel/canonical/carrier_pspectrum.json`, `reader_and_surgery.json`, `reader_freqcontrol.json`, `dnabert2_aliasing.json` |

C39/C40 are comparative corpus attribution (a frozen-checkpoint natural experiment), not training
dynamics. Figure: `figures/origin-crossmodel/crossmodel_carrier_summary.png`. Index:
`results/origin-siblings/RESULTS_INDEX.md`, `results/origin-crossmodel/README.md`.

## Cross-study positive control — Head–Context Atlas

The Atlas is a **separate study** with its own ledger and reproduction record
([`../splicebert_head_context_atlas/CLAIMS.md`](../splicebert_head_context_atlas/CLAIMS.md); scripts under
`scripts/splicebert_head_context_atlas/`, indexed in [`../../scripts/README.md`](../../scripts/README.md);
artifacts under `results/splicebert_head_context_atlas/` with `RESULTS_INDEX.md`). Its competition branch
(HCA2–HCA10) is CLOSED as not competition-specific; the one salvaged positive — **HCA9's independent
rediscovery of donor-circuit L3H1** as the donor-reconstruction mediator on a disjoint cohort — corroborates
C3/C4/C5/C17 (`run_group_mlp_closure.py`, `reanalyze_valid_donor_rows.py`). The wider Atlas (HCA11–HCA23)
is a distinct active program.

---

## Position/carrier durable index

The canonical position/carrier index is [`../../results/phase-lattice/RESULTS_INDEX.md`](../../results/phase-lattice/RESULTS_INDEX.md),
mapping position spectra, carrier deletion, mediation, attention gates, and biological-source assays to
their scripts, immutable JSON, and manifests.

## Figures

Canonical paper figures are generated from recorded artifacts (`make_fig1_summary.py`,
`make_fig2_oscillators.py`, `make_fig3_clock_circuit.py`,
`make_fig5_p3_phase_surgery_confirmation.py`, `make_fig_phase_data.py`, `make_fig_report_data.py`;
donor-circuit figure generators live in `figures/splicebert/`). A figure is explanatory; the JSON and
manifest remain the scientific record. Sources documented under `figures/splicebert/README.md`.

## Validation checklist

Before merging a change to an active workflow:

```bash
uv run pytest
uv run python scripts/verify_manifest.py
rg -n "VOID|SUPERSEDED|METHODS_AUDIT|MECHANISM.md|SEALED_TEST_RESULT" \
  scripts studies/splicebert_partner_selection results/phase-lattice figures/splicebert
```

Then verify:

1. every active result is listed in a `RESULTS_INDEX.md` or the sealed coupling result family, and in `MANIFEST.csv`;
2. every listed script exists and every retired output has no active runner;
3. Markdown links resolve;
4. exact add-one and multiple-testing conventions match [`METHODS.md`](METHODS.md);
5. the strongest prose claim does not exceed the artifact's evidence level (`CLAIMS.md` + `CORRECTIONS.md`).
