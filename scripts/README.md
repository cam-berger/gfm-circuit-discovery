# Scripts index — mechanistic discovery workflows

> **Tier: T1 — canonical.** Current, authoritative. Read this.

This is an **index of the runnable scripts**, not the scientific ledger. It maps each active script to
the claim(s) and controlling artifact it serves, grouped by result family. The authoritative
claim→evidence record is [`results/MANIFEST.csv`](../results/MANIFEST.csv) plus
[`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md)
(donor circuit, C1–C41 + IC1), [`studies/splicebert_head_context_atlas/CLAIMS.md`](../studies/splicebert_head_context_atlas/CLAIMS.md)
(Atlas, HCA1–HCA23), and [`CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md)
(R1–R40). The per-claim command map with reproduction rules is
[`studies/splicebert_partner_selection/REPRODUCIBILITY.md`](../studies/splicebert_partner_selection/REPRODUCIBILITY.md);
the repo-level entry point is [`REPRODUCE.md`](../REPRODUCE.md).

`scripts/` has four active trees, one per study plus the cross-model runners:
`splicebert/` (donor circuit + carriers + origin siblings), `splicebert_head_context_atlas/` (the Atlas
sibling study), `splicebert_gtex_finetune_shift/` (the supervised fine-tuning study), and `crossmodel/`
(cross-model origin). Superseded and VOID runners live in Git history, not alongside live workflows.
Every artifact named below exists on disk; every MANIFEST row's script/artifact/manifest/figure path
resolves (`scripts/verify_manifest.py` passes).

> **Path note (2026-07-31).** The 47 GTEx runners moved out of `scripts/splicebert/` into their own
> tree. `results/MANIFEST.csv` and all code, tests, and live documents follow the new paths. The
> per-artifact `*.manifest.json` provenance sidecars deliberately do **not**: they record the command
> as it was executed, and rewriting them would falsify the execution record. A sidecar naming
> `scripts/splicebert/run_gtex_*.py` is correct history — the runner is now the same basename under
> `scripts/splicebert_gtex_finetune_shift/`.

## Execution contract

- Run from the repository root under the `uv` environment (`uv run python scripts/<...>.py`; see
  [`REPRODUCE.md`](../REPRODUCE.md)). Historical runs used the conda env `gfm-thrust1` — provenance only.
- Read `docs/METHODOLOGY.md`, `docs/EVIDENCE_LADDER.md`, and the relevant preregistration/study record first.
- Run the preflight/smoke gate before target data; keep frozen thresholds unchanged.
- Write machine-readable output plus a manifest under the relevant `results/` family; never overwrite or
  silently revive a superseded artifact.
- The sealed chr8/9 test (`run_sealed_test.py`) is provenance, not a routine smoke test — it requires the
  unsealing flag in `preregistrations/PREREGISTRATION_SEALED.md`.
- Generate figures only from recorded artifacts.

# `scripts/splicebert/` — donor circuit, carriers, origin siblings

## Sealed behaviour and the {L3H1, L2H15} subcircuit (C1–C7)

| Script | Claim(s) | Controlling artifact |
|---|---|---|
| `run_coupling.py` | C1 | `results/splicebert_coupling/confirmation.json` |
| `run_sealed_test.py` | C1;C2;C3;C4 | `results/splicebert_coupling/sealed_test.json` (+ `sealed_val.json`) |
| `run_ablation_scan.py` | C3 (development) | `results/splicebert_coupling/development/ablation_scan.json` |
| `run_subspace.py` | C4 (development) | `results/splicebert_coupling/development/subspace.json` |
| `run_localization.py`, `run_position_profile.py`, `run_readout_path.py`, `run_path_decomposition.py` | C5 | `position_profile.json`, `readout_path.json` (+ dev `localization.json`, `path_decomposition.json`) |
| `run_l2_accounting.py`, `run_l1h12_probe.py`, `run_l1h12_route.py`, `run_l2_rescue.py` | C6 | `l2_accounting.json`, `l1h12_probe.json`, `l1h12_route.json` (+ dev `l2_rescue.json`) |
| `run_qkv.py` | C22 | `results/splicebert_coupling/qkv.json` |
| `run_completeness_null.py`, `run_faithfulness.py` | C7 (early completeness assay, superseded by C27/C33) | `completeness_null.json`, `faithfulness.json` |

## Donor readout / `U_info` and behavioural signature (C16–C19)

| Script | Claim(s) | Controlling artifact |
|---|---|---|
| `run_donor_anchor.py` | C16 | `results/splicebert_coupling/donor_anchor.json` |
| `run_donor_forward.py` | C17 | `donor_forward.json` |
| `run_donor_heads.py` | C18 | `donor_heads.json` |
| `run_donor_signature.py` | C19 | `donor_signature.json` |

## Address / payload donor graph A0→A12 (C20–C33)

| Script | Claim(s) | Controlling artifact |
|---|---|---|
| `run_acceptor_address.py` | C20 | `acceptor_address.json` |
| `run_a0_targeting_provenance.py` | C21 | `a0_targeting_provenance.json` |
| `run_a1_token_position.py` | C20;C22 | `a1_token_position.json` |
| `run_a2_address_payload.py` | C23 | `a2_address_payload.json` |
| `run_a2b_payload_localization.py` | C24 | `a2b_payload_localization.json` |
| `run_a3_l2h15_address_payload.py` | C25 | `a3_l2h15_address_payload.json` |
| `run_a4_address_geometry.py` | A4 (superseded → C32) | `superseded/a4_address_geometry.json` |
| `run_a5_uinfo_replication.py` | C26 (validates the C16 `U_info` object) | `a5_uinfo_replication.json` |
| `run_a6_circuit_faithfulness.py` | C27 (R21) | `a6_circuit_faithfulness.json`, `a6_reconcile.json` |
| `run_a7_source_localization.py` | C28 | `a7_source_localization.json` |
| `run_a8_generator_path.py` | C29 | `a8_generator_path.json` |
| `run_a9_payload_semantics.py` | A9 / A9;R22 (superseded; splice-strength withdrawn) | `superseded/a9_payload_semantics.json` |
| `run_a9b_payload_semantics_composition.py` | C30 / C30;R22 / R22 | `a9b_payload_semantics_composition.json`, `a9c_matched_composition_reconcile.json`, `a9c_sweep_seeded_draws.json` |
| `run_a10_generator_edges.py` | C31 | `a10_generator_edges.json` |
| `run_a11_address_geometry.py` | C32 | `a11_address_geometry.json` |
| `run_a12_faithfulness_size.py` | C33 | `a12_faithfulness_size.json` |

Note: `run_b2_payload_steering.py` is present but carries **no MANIFEST row / no committed artifact** — it
is the C30 semantic-identity close-attempt (`PREREGISTRATION_B2_PAYLOAD_STEERING.md`), red-teamed
survivor=NO (C30 CLOSED-BOUNDED). Design/prereg asset only.

## Intron-coupling arc (IC1)

| Script | Claim(s) | Controlling artifact |
|---|---|---|
| `run_ic1_distal_intron_necessity.py` | IC1 | `ic1_distal_intron_necessity_v8.json` (superseded `_v7`; preflight `_v8`) |
| `run_ic1_pilot.py`, `run_ic1_pilot2.py`, `run_ic1_pilot3.py`, `run_ic1_pilot4_ood.py` | IC1 (dev pilots) | `development/ic1_pilot*.json` |
| `run_ic2_feasibility.py`, `run_ic2_calibration.py` | IC2 (RETIRED; design asset) | `development/ic2_feasibility.json`, `ic2_calibration.json` |

## Period-3 carrier — structure, deletion, mediation, present-day use (C8–C10, C34–C36)

| Script | Claim(s) | Controlling artifact |
|---|---|---|
| `run_phase_inventory.py`, `run_carrier_provenance.py` | C8,C11 | `phase-lattice/phase_inventory.json`, `carrier_provenance.json` |
| `run_carrier_surgery.py`, `run_carrier_dose.py`, `run_comb_shift.py` | C8 | `carrier_surgery.json`, `carrier_dose.json`, `comb_shift.json` |
| `run_carrier_mediation.py` | C9,C10 | `phase-lattice/carrier_mediation.json` |
| `run_p3_carrier_locus.py` | C34 (locus; R24 confirm gate-qualified) | `results/p3_carrier_locus.json` |
| `run_p3_av_decomposition.py` | C35 (A-vs-V routing; R23/R24) | `results/p3_av_decomposition.json` |
| `run_p3_shift_diagnostic.py`, `analyze_p3_shift_mod3.py` | P3-shift-diagnostic (v2, mod3 phase) | `phase-lattice/p3_shift_diagnostic_v2.json` (+ `.mod3.json`) |
| `build_p3_closure_cohort.py`, `run_p3_closure_preflight.py`, `analyze_p3_closure_preflight.py` | C36 (closure cohort/preflight) | `phase-lattice/development/p3_closure_*` |
| `build_p3_phase_surgery_cohort.py`, `build_p3_phase_surgery_non_cds_cohort.py`, `match_p3_phase_surgery_contexts.py` | C36 (cohorts + matched contexts) | `phase-lattice/p3_phase_surgery_context_match_untouched_*.json`, `..._non_cds_untouched_census_*.json` |
| `run_p3_phase_surgery_preflight.py`, `analyze_p3_phase_surgery_preflight.py`, `run_p3_phase_surgery_health.py`, `run_p3_phase_surgery_controls.py`, `analyze_p3_phase_surgery_controls.py`, `analyze_p3_phase_surgery_context_difference.py` | C36 (preflight/health/controls) | `phase-lattice/development/p3_phase_surgery_*` |
| `run_p3_phase_surgery_confirmation.py`, `analyze_p3_phase_surgery_confirmation.py`, `freeze_p3_phase_surgery_confirmation.py`, `audit_p3_phase_surgery_confirmation_smoke.py`, `audit_p3_phase_surgery_confirmation_package.py` | **C36 (clean confirmation, Ψ_C=+0.141)** | `phase-lattice/p3_phase_surgery_confirmation_v1/confirmation_analysis.json`; `p3_phase_surgery_confirmation_package_v1.json` (+ `.audit.json`) |

## ≈10.63 carrier — routing, OV, behaviour + human-only ≈10.2 gate (C11–C13, C37)

| Script | Claim(s) | Controlling artifact |
|---|---|---|
| `run_carrier_geometry.py`, `run_carrier_subbin.py`, `run_second_carrier.py` | C11 (geometry/wavelength/second carrier) | `phase-lattice/carrier_geometry.json`, `carrier_subbin.json`, `second_carrier.json` |
| `run_postsoftmax.py`, `measure_band_mass.py` | C11,C13 (gated post-softmax; eligibility mass) | `phase-lattice/postsoftmax.json`, `preflight_postsoftmax.json`, `band_mass.json` |
| `run_human_carrier.py` | C13 (human-only QK comparison) | `phase-lattice/human_carrier.json` |
| `run_carrier_ov_av.py`, `run_carrier_p10_behaviour.py`, `run_carrier_preflight_r.py` | C37 (donor-scoped OV write + P10 behaviour) | `phase-lattice/carrier_ov_av.json`, `carrier_p10_behaviour_confirm.json`, `carrier_preflight_r.json` |
| `run_carrier_ov_general.py` | C12 (general-OV bounded null) | `phase-lattice/carrier_ov_general.json` |

## Nucleosomal / biological-source assays (C14–C15)

| Script | Claim(s) | Controlling artifact |
|---|---|---|
| `run_dyad_fpr.py`, `run_dyad_real.py`, `run_dyad_wavelength.py` | C14 (real nucleosomal ≈10.298 nt) | `phase-lattice/dyad_fpr.json`, `dyad_real.json`, `dyad_wavelength.json` |
| `run_corpus_spectrum.py`, `run_splice_profile.py`, `run_splice_phasing.py`, `run_splice_long.py` | C15 (sensitivity limits) | `phase-lattice/corpus_spectrum.json`, `splice_profile.json`, `splice_phasing.json`, `splice_long.json` |

## Cross-example transduction (C38, C41)

| Script | Claim(s) | Controlling artifact |
|---|---|---|
| `run_b3_cross_example_transduction.py` | C38 (acceptor-side payload; bounded/inconclusive) | `splicebert_coupling/b3_cross_example_transduction_n512.json` (+ base/`_pilot`/`_smoke`) |
| `run_g1_cross_example_transduction.py` | C41 (donor `U_info` state; GENERIC READOUT AXIS) | `splicebert_coupling/g1_cross_example_transduction.json` (+ `_pilot`/`_decomp`/`_smoke`) |

## Origin siblings — corpus attribution on the 72-species vs human-only checkpoints (C39)

| Script | Claim(s) | Controlling artifact |
|---|---|---|
| `run_sibling_pspectrum.py` | C39 (leakage-free P-spectrum) | `results/origin-siblings/canonical/sibling_pspectrum.json` |
| `run_sibling_carrier_bandtransfer.py` | C39 (band-deletion coding-specific use) | `origin-siblings/canonical/sibling_carrier_bandtransfer.json` |
| `run_sibling_validate.py` | C39 (validation) | `origin-siblings/canonical/sibling_validate.json` |

## Figures and figure/report data

| Script | Serves | Output |
|---|---|---|
| `make_fig_phase_data.py` | C8 (period-3 figure data) | `phase-lattice/fig_phase_data.json` |
| `make_fig_report_data.py` | C8,C11,C13,C14 (publication/correction summary) | `phase-lattice/fig_report_data.json` |
| `make_fig5_p3_phase_surgery_confirmation.py` | C36 | `figures/splicebert/fig5_p3_phase_surgery_confirmation.png` |
| `make_fig1_summary.py`, `make_fig2_oscillators.py`, `make_fig3_clock_circuit.py`, `make_fig4_carrier_circuit.py` | carrier/circuit paper figures (figure generators; PNGs are the MANIFEST-indexed figures) | `figures/splicebert/fig{1_summary,2_oscillators,3_clock_circuit,4_carrier_circuit}.png` |
| `figstyle.py` | shared figure styling helper (not a runner) | — |

Note: `figures/splicebert/` also holds `make_fig1_coupling.py`, `make_fig2_mechanism.py`,
`make_fig3_circuit.py`, `make_fig_carrier_heatmap.py`, `make_fig_full_circuit.py`,
`make_fig_generator_graph.py`, `make_fig_evidence_ladder.py` (donor-circuit figure generators living
beside their assets), documented in `figures/splicebert/README.md`.

## Support / analysis companions (no standalone claim row)

`run_exon_vs_intron.py` (development `exon_vs_intron.json`), `export_segment_attribution_bundle.py`
(joint-mask screening-graph builder; imported by the Atlas representative-DAG builder),
`run_pair_categories.py` (figure data
`figures/splicebert/data/pair_categories.json`), `analyse_comb_shift.py` (comb-shift analysis companion),
`calibration_audit.py`, `test_ic1_determinism.py` (determinism test).

---

# `scripts/splicebert_head_context_atlas/` — Head–Context Atlas (sibling study)

Grouped by pipeline stage. Full claim text is in
[`studies/splicebert_head_context_atlas/CLAIMS.md`](../studies/splicebert_head_context_atlas/CLAIMS.md);
durable index at `results/splicebert_head_context_atlas/RESULTS_INDEX.md`.

| Stage / script(s) | Claim(s) | Controlling artifact |
|---|---|---|
| `build_census.py` | HCA1 | `census_v1.json` |
| `behavioral_pilot.py` | HCA2/HCA3/HCA7 | `behavioral_pilot_v1.json` |
| `specificity_pilot.py` | HCA4/HCA7 | `specificity_pilot_v1.json` |
| `reanalyze_valid_donor_rows.py` | HCA8/HCA9 (L3H1 donor rediscovery) | `head_effect_donor_valid_reanalysis_v1.json` |
| `run_group_mlp_closure.py` | HCA9/HCA10, HCA10 | `group_mlp_closure_v1.json`, `group_head_rerank_valid_donor_v2.json` |
| `build_atlas_context_census.py`, `enrich_atlas_context_packets.py`, `build_atlas_matched_cohorts.py`, `validate_atlas_matched_cohorts.py` | HCA11 (wider-atlas cohort + matching) | `atlas_context_census_v1.json`, `atlas_context_enrichment_v1.json`, `atlas_context_matched_v7.*` |
| `run_atlas_model_preflight.py`, `run_atlas_discovery_screen.py`, `merge_atlas_discovery_screen.py`, `validate_atlas_discovery_screen.py` | HCA12, HCA12/HCA13 (discovery screen instrument) | `atlas_model_preflight_v1.*`, `atlas_discovery_screen_v1/`, `atlas_discovery_screen_smoke_v2_merged.*` |
| `analyze_atlas_discovery.py`, `validate_atlas_discovery_analysis.py` | HCA13, HCA18 (discovery analysis + 102-component atlas / 4 general-use candidates) | `atlas_discovery_analysis_v1.*`, `atlas_discovery_v1.summary.json` |
| `run_atlas_mechanism_preflight.py`, `validate_atlas_mechanism_preflight.py`, `build_atlas_mechanism_package.py`, `validate_atlas_mechanism_package.py` | HCA14, HCA14/HCA15, HCA18/HCA19 (mechanism handoff package) | `atlas_mechanism_preflight_v1.*`, `atlas_mechanism_package_v1.*` |
| `run_atlas_mechanism_validation.py`, `merge_atlas_mechanism_validation.py`, `analyze_atlas_mechanism_validation.py`, `validate_atlas_mechanism_validation.py`, `validate_atlas_mechanism_analysis.py` | HCA15, HCA19 (general-use candidates validate on internal-validation cohort) | `atlas_mechanism_validation_v1/`, `atlas_mechanism_validation_analysis_v1.*` |
| `run_atlas_stage2_mechanisms.py`, `merge_atlas_stage2_mechanisms.py`, `analyze_atlas_stage2_mechanisms.py`, `validate_atlas_stage2_mechanisms.py`, `validate_atlas_stage2_analysis.py`, `build_atlas_stage2_package.py`, `validate_atlas_stage2_package.py` | HCA16, HCA20 (conditional-mechanism stage + downstream MLP recovery paths; per-packet shards) | `atlas_stage2_mechanism_v1/` (health/target shards), `atlas_stage2_package_smoke_v2.*` |
| `build_atlas_representative_dags.py`, `validate_atlas_representative_dags.py` | HCA17 (representative DAG boundary) | `atlas_representative_dag_smoke_v4.*` |
| `analyze_b5_context_signal_power.py` | HCA23 (context signals closed as under-powered) | `b5_context_signal_power_v1.json` |

Atlas support / head-screen / visualizer (no MANIFEST claim row in this study's manifest):
`run_all_head_screen.py`, `merge_head_screen.py`, `validate_head_screen.py` (the Colab all-96-head
screen), `build_example_covariates.py`, `benchmark_colab.py`, `export_atlas_visualizer_evidence.py`,
`validate_atlas_visualizer_evidence.py` (HCA21/HCA22 visualizer handoff — see the Atlas study record).
The competition branch (HCA2–HCA10) is CLOSED; its `competition_branch_closure_audit_v1.json`
(HCA5/HCA7-HCA10) is a decision-audit artifact with **no runner**.

---

# `scripts/crossmodel/` — cross-model origin (C40)

| Script | Claim(s) | Controlling artifact |
|---|---|---|
| `run_carrier_pspectrum.py` | C40 (RNA-FM / RNA-BERT powered P-spectrum null) | `results/origin-crossmodel/canonical/carrier_pspectrum.json` (+ `acquisition_validation.json`) |
| `run_reader_and_surgery.py` | C40 (would-be reader heads + band deletion) | `origin-crossmodel/canonical/reader_and_surgery.json` |
| `run_reader_freqcontrol.py` | C40 (frequency-control grid) | `origin-crossmodel/canonical/reader_freqcontrol.json` |
| `run_dnabert2_aliasing.py` | C40 (BPE tokenizer-aliasing demonstration) | `origin-crossmodel/canonical/dnabert2_aliasing.json` |
| `test_carrier_spectrum.py` | unit test for the spectrum path (no claim row) | — |

---

# `scripts/verify_manifest.py`

Validates that every `results/MANIFEST.csv` row's `script`, `artifact`, `manifest`, and `figure` path
exists. Run it before merging any change to an active workflow. It currently passes with zero unresolved paths.
