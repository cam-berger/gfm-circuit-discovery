# Canonical SpliceBERT scripts

> **Tier: T1 — canonical.** Current, authoritative. Read this.

`scripts/splicebert/` contains only the active discovery, validation, reproduction, and figure paths for the current scientific record. Superseded and VOID runners are available in Git history, not alongside live workflows.

## Canonical workflow groups

### Donor–acceptor coupling

`run_sealed_test.py`, `run_position_profile.py`, `run_l2_accounting.py`, `run_readout_path.py`, `run_l1h12_probe.py`, `run_l1h12_route.py`, `run_faithfulness.py`, `run_completeness_null.py`

The sealed test is provenance, not a routine smoke test. Read the frozen preregistration before using its unsealing flag.

### Period-3 clock and position structure

`run_phase_inventory.py`, `run_carrier_geometry.py`, `run_carrier_subbin.py`, `run_second_carrier.py`, `run_carrier_provenance.py`, `run_carrier_surgery.py`, `run_carrier_dose.py`, `run_carrier_mediation.py`, `run_comb_shift.py`, `analyse_comb_shift.py`

### Attention-level carrier tests

`run_human_carrier.py`, `run_postsoftmax.py`, `measure_band_mass.py`

### Biological/source assays

`run_dyad_fpr.py`, `run_dyad_real.py`, `run_dyad_wavelength.py`, `run_corpus_spectrum.py`, `run_splice_profile.py`, `run_splice_phasing.py`, `run_splice_long.py`

### Figure and report data

`make_fig1_summary.py`, `make_fig2_oscillators.py`, `make_fig3_clock_circuit.py`, `make_fig4_carrier_circuit.py`, `make_fig_phase_data.py`, `make_fig_report_data.py`, `figstyle.py`

## Execution contract

- Run from the repository root.
- Read `docs/METHODOLOGY.md`, `docs/EVIDENCE_LADDER.md`, and the relevant study record first.
- Run the preflight before target data and keep frozen thresholds unchanged.
- Write machine-readable output and a manifest under the relevant `results/` family.
- Use masked model-health controls, size-matched nulls, and exact add-one p-values.
- Never overwrite or silently revive a superseded artifact.
- Generate figures from recorded artifacts.

The claim-to-script map and reproduction notes are in `studies/splicebert_partner_selection/REPRODUCIBILITY.md`.
