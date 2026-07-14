# SpliceBERT reproducibility map

This is the shortest path from a claim to its script and machine-readable artifact. It indexes active workflows only. Historical and VOID experiments remain in Git history.

## Environment

From the repository root:

```bash
uv sync --extra splicebert --extra dev
uv run pytest
```

Inspect a canonical runner before execution:

```bash
uv run python scripts/splicebert/<runner>.py --help
```

Runners use repository-defined data/model paths and write under `results/`. Large models and source datasets are not committed. Review the corresponding manifest before asserting that a rerun matches the recorded environment.

## Provenance contract

Every scientific run must record:

- checkpoint and tokenizer;
- source dataset and split;
- inclusion/exclusion rules;
- seed, sample size, and intervention parameters;
- exact command and code revision;
- preflight values and frozen thresholds;
- output artifact and manifest.

Do not overwrite a scientific result with a new run. Write a new artifact or intentionally supersede the old one in the index and claim ledger.

## Donor–acceptor coupling

| Purpose | Canonical script | Indexed output |
|---|---|---|
| Sealed confirmation | `run_sealed_test.py` | `results/splicebert_coupling/sealed_test.json` and manifest |
| Position of the L3H1 write | `run_position_profile.py` | `position_profile.json` |
| L2H15 → L3H1 accounting | `run_l2_accounting.py` | `l2_accounting.json` |
| Downstream path | `run_readout_path.py` | `readout_path.json` |
| L1H12 probe and route | `run_l1h12_probe.py`, `run_l1h12_route.py` | `l1h12_probe.json`, `l1h12_route.json` |
| Circuit faithfulness | `run_faithfulness.py` | `faithfulness.json` |
| Completeness null | `run_completeness_null.py` | `completeness_null.json` |

The sealed test requires the explicit unsealing flag recorded in [`PREREGISTRATION_SEALED.md`](PREREGISTRATION_SEALED.md). Do not rerun it merely as a smoke test.

## Period-3 clock and position structure

| Purpose | Canonical script | Indexed output |
|---|---|---|
| Position-table inventory | `run_phase_inventory.py` | `results/phase-lattice/phase_inventory.json` and manifest |
| Carrier geometry and orthogonality | `run_carrier_geometry.py` | `carrier_geometry.json` |
| Sub-bin wavelength | `run_carrier_subbin.py` | `carrier_subbin.json` |
| Second-carrier characterization | `run_second_carrier.py` | `second_carrier.json` |
| Training/checkpoint provenance | `run_carrier_provenance.py` | `carrier_provenance.json` |
| Period-3 behavior deletion | `run_carrier_surgery.py` | `carrier_surgery.json` and manifest |
| Dose response | `run_carrier_dose.py` | `carrier_dose.json` and manifest |
| Size-matched restoration | `run_carrier_mediation.py` | `carrier_mediation.json` and manifest |
| Comb shift/phase test | `run_comb_shift.py` | `comb_shift.json` and manifest |
| Figure data | `make_fig_phase_data.py` | `fig_phase_data.json` |

`analyse_comb_shift.py` is the analysis companion to the recorded comb-shift run. Old phase-purity, phase-function, synthetic-grid, and generic dual-reader paths are not canonical.

## ≈10.63 and ≈10.2 attention gates

| Purpose | Canonical script | Indexed output |
|---|---|---|
| Position/QK checkpoint comparison | `run_human_carrier.py` | `human_carrier.json` |
| Gated post-softmax screen | `run_postsoftmax.py` | `postsoftmax.json`, `preflight_postsoftmax.json` |
| Eligibility mass audit | `measure_band_mass.py` | `band_mass.json` |
| Carrier deletion behavior | `run_carrier_surgery.py` | `carrier_surgery.json` and manifest |
| Publication/correction summary | `make_fig_report_data.py` | `fig_report_data.json` |

The post-softmax result is interpreted with exact add-one permutation values from the report-data path, not raw zero-hit values stored by an earlier calculation. The human-only conclusion is zero Level-A heads; the multispecies conclusion is eight.

## Biological/source assays

| Purpose | Canonical script | Indexed output |
|---|---|---|
| Dyad assay false-positive calibration | `run_dyad_fpr.py` | `dyad_fpr.json` |
| Real dyad frozen-phase test | `run_dyad_real.py` | `dyad_real.json` |
| Dyad wavelength distinction | `run_dyad_wavelength.py` | `dyad_wavelength.json` |
| Unanchored sensitivity limit | `run_corpus_spectrum.py` | `corpus_spectrum.json` |
| Junction profile | `run_splice_profile.py` | `splice_profile.json` |
| Junction phasing | `run_splice_phasing.py` | `splice_phasing.json` |
| Long-window resolution/power | `run_splice_long.py` | `splice_long.json` |

The unanchored spectrum is a calibrated sensitivity limit, not evidence that no biological periodicity exists. The dyad result is the positive biological assay.

## Figures

Canonical paper figures are generated from recorded artifacts:

- `make_fig1_summary.py`
- `make_fig2_oscillators.py`
- `make_fig3_clock_circuit.py`
- `make_fig4_carrier_circuit.py`
- `make_fig_report_data.py`

Figure sources and hand-edited assets are documented under `figures/splicebert/`. A figure is explanatory; the JSON and manifest remain the scientific record.

## Validation checklist

Before merging a change to an active workflow:

```bash
uv run pytest
rg -n "VOID|SUPERSEDED|METHODS_AUDIT|MECHANISM.md|SEALED_TEST_RESULT" \
  scripts/splicebert studies/splicebert studies/splicebert_partner_selection \
  results/phase-lattice figures/splicebert
```

Then verify:

1. every active result is listed in `results/phase-lattice/RESULTS_INDEX.md` or belongs to the sealed coupling result family;
2. every listed script exists and every retired output has no active runner;
3. Markdown links resolve;
4. exact add-one and multiple-testing conventions match [`METHODS.md`](METHODS.md);
5. the strongest prose claim does not exceed the artifact's evidence level.
