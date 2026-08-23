# SpliceBERT positional circuits — current results index

This directory contains the machine-readable record for two distinct positional mechanisms in
SpliceBERT:

1. a **period-3 positional carrier** that causally supports donor–acceptor information routing; and
2. a separate **≈10.63 positional carrier** that is read through post-softmax attention. Its donor-side
   downstream use is bounded, and its training-lineage association is comparative rather than causal.

Start with:

- [`../MANIFEST.csv`](../MANIFEST.csv) / [`../MANIFEST.json`](../MANIFEST.json) for the
  **machine-readable index** — one row per tracked file under `results/`, giving its `claim_id`,
  `status`, `evidence_level`, controlling `script`, sibling `manifest`, and consuming `figure`.
  Filter the `artifact` column on `results/phase-lattice/` for this directory. Column semantics and
  the four `status` values are defined in [`../RESULTS_INDEX.md`](../RESULTS_INDEX.md);
  `scripts/verify_manifest.py` (run by `tests/test_manifest.py`) keeps it in step with the tree.
  The tables below are the human-readable view of the same tree and do not list every file;
  the manifest does.
- `figures/splicebert/EVIDENCE.md` for the evidence ladder;
- `figures/splicebert/README.md` for the publication figures;
- `studies/splicebert_partner_selection/CLAIMS.md` for the authoritative claim, correction, and
  retraction ledger.

## Current evidence

| Component | Position table | QK | Attention | OV / mediation | Behavior | Status |
|---|:--:|:--:|:--:|:--:|:--:|---|
| Period-3 positional carrier | ✓ | ✓ | ✓ | ✓ | ✓ | Causal positional support for donor–acceptor routing |
| Donor–acceptor computation | n/a | ✓ | ✓ | ✓ | ✓ | Paired-acceptor information causally affects donor prediction |
| Multi-species ≈10.63 carrier | ✓ | ✓ | ✓ | null (donor) | small suppr. | Reaches attention; OV write onto the donor readout is NULL and deletion has a small *suppressive* coupling effect — **not part of the donor circuit** (C37, donor-scoped); general OV (MLM) + origin open |
| Human-only ≈10.2 candidate | ✓ | provisional | failed | — | null | Proposed chromatin circuit closed at attention |

Evidence never promotes itself between columns. The ≈10.63 result formerly stopped at attention;
**B1 (2026-07-22, C37) now measures the OV and behavioural columns for the donor circuit** and finds it is
**not part of that circuit**: its post-softmax readers (9, replicating C11 on 1024nt) write **nothing
carrier-specific** onto the donor readout (OV value-main A-null), and deleting it has only a **small,
significant, *suppressive*** coupling effect (dC=−0.033; it slightly interferes with, rather than supports,
coupling). Its general OV content (MLM output) and biological origin remain untested.

## Results by question

### Position-table structure

| File | Role |
|---|---|
| `phase_inventory.json` / `phase_inventory.manifest.json` | Inventory of positional frequencies and candidate carriers |
| `carrier_geometry.json` | Rank-2 rotary geometry and circularity |
| `carrier_subbin.json` | Sub-bin wavelength estimate: ≈10.630 |
| `second_carrier.json` | Separation of period-3 and ≈10.63 components |
| `carrier_provenance.json` | Checkpoint chronology and cross-checkpoint comparison |

### Period-3 positional carrier and donor–acceptor circuit

| File | Role |
|---|---|
| `fig_phase_data.json` | Consolidated period-3 representation and head-readout evidence |
| `comb_shift.json` / `comb_shift.manifest.json` | Controlled phase-shift assay |
| `carrier_dose.json` / `carrier_dose.manifest.json` | Dose–response deletion of the carrier |
| `carrier_mediation.json` / `carrier_mediation.manifest.json` | Head restoration and size-matched nulls |
| `results/splicebert_coupling/` | Coupling, ablation, rescue, path, and sealed-test results |

### ≈10.63 attention circuit

| File | Role |
|---|---|
| `postsoftmax.json` | Full post-softmax scan |
| `preflight_postsoftmax.json` | Positive/negative controls, injection checks, and frozen gates |
| `band_mass.json` | Eligibility audit and attention mass in the tested band |
| `carrier_surgery.json` / `carrier_surgery.manifest.json` | Target, reciprocal, low-leak, and content-shuffle controls |
| `human_carrier.json` | Matched human-only checkpoint analysis |
| `carrier_preflight_r.json` / `.manifest.json` | **B1** Preflight R: 1024nt post-softmax Level-A reader scan → `R_1024` = 9 heads (`OV_ARM_LIVE`) |
| `carrier_ov_av.json` / `.manifest.json` | **B1 OV arm** (C37): value-main (simple `DI−DD`) over `R_1024`; **A-null** (no carrier-specific write onto the donor readout); router-negative + `M-simplemain` |
| `carrier_p10_behaviour_confirm.json` (+ `_n1200.json`) / `.manifest.json` | **B1 behaviour** (C37): coupling under ≈10.63 deletion; dC=−0.033 [−0.061,−0.005] small suppressive (deletion raises coupling); §5 risk-2 control-band qualification |

### Biological-source tests and powered refutations

| File | Role |
|---|---|
| `dyad_fpr.json` | False-positive and jitter calibration for the anchored assay |
| `dyad_real.json` | Positive detection of the nucleosomal signal |
| `dyad_wavelength.json` | Nucleosomal wavelength estimate: ≈10.298 |
| `corpus_spectrum.json` | Underpowered unanchored assay; report its sensitivity limit only |
| `splice_profile.json`, `splice_phasing.json`, `splice_long.json` | Splice-junction follow-up analyses |

### Exploratory current-use diagnostics

| File | Role |
|---|---|
| `p3_shift_diagnostic_v2.json` | Outcome-aware diagnostic of 0/+1/+2/+3 natural, intronic, exonic, and sham shifts; not an origin claim |
| `p3_shift_diagnostic_v2.mod3.json` | Derived L3H1 attention mass and peak counts under the exact absolute convention `x%3={0,1,2}` |
| `p3_phase_surgery_preflight_analysis_development_n61_20260721.json` | Development-only coupling calibration for nucleotide-invariant local P3 surgery |
| `p3_phase_surgery_controls_analysis_development_n30_v2_20260721.json` | Development-only learned-direction result against 19 same-geometry hidden-axis controls |
| `p3_phase_surgery_context_difference_development_n33_20260721.json` | Development-only matched CDS-minus-non-CDS-exon boundary calibration |
| `p3_phase_surgery_confirmation_package_v1.json` + `.audit.json` | Frozen untouched IDs, controls, thresholds, hashes, and independently reconstructed audit |
| `p3_phase_surgery_confirmation_v1/confirmation_analysis.json` | C36 clean confirmation: material phase-aligned coupling use and learned-direction specificity; coding enrichment inconclusive |
| `p3_phase_surgery_confirmation_v1/confirmation_shards.tar.gz` + `.archive.json` | Lossless archive of all 1,145 immutable target shards, with archive and per-shard manifest hashes |

The shift diagnostic generated the coding-register hypothesis but does not promote C8/C34/C35 to a
biological reading-frame interpretation. Sequence edits were retired as confounded. The active closure
uses nucleotide-invariant local surgery on the learned P3 position component; its development artifacts
are calibration only. The frozen untouched C36 result now confirms phase-aligned use of the learned P3
direction in CDS blocks; its matched non-CDS-exon contrast does not establish coding enrichment.

The positive nucleosomal detection is at the wrong wavelength for the model carrier. It therefore
refutes—rather than supports—the claim that the multi-species ≈10.63 carrier is the nucleosomal
periodicity.

## Publication data package

`fig_report_data.json` is the single input to the four publication figures. It is generated by
`scripts/splicebert/make_fig_report_data.py` and checks recomputed quantities against recorded
run artifacts before writing.

## Retired interpretations

- **“10.67”** was a Fourier-bin label, not a wavelength measurement. Use **≈10.63**.
- **“10.58 ± 0.35”** is not a measured splice-junction constant and is not a leading
  interpretation of the carrier.
- The multi-species ≈10.63 carrier is **not** the ≈10.298 nucleosomal signal.
- The human-only ≈10.2 chromatin circuit is closed at the attention gate.
- VOID and superseded result families and their runners are excluded from the active tree. Git
  history preserves their full evolution. Older figure snapshots may remain for visual provenance,
  but they are not machine-readable evidence or supported entrypoints.

## Open questions

For the ≈10.63 carrier, the status of the licensed questions after B1 (C37):

1. ~~What do the Level-A heads write through OV?~~ **Answered for the donor circuit (B1): nothing
   carrier-specific** — the value-main onto the donor readout is a null across all 9 readers. *General OV
   content (onto the MLM output) remains untested.*
2. ~~Does that write mediate a model-native behavior?~~ **Answered for coupling (B1): a small *suppressive*
   effect** — deletion raises coupling by 0.033 (sub-SESOI, CI-excludes-0); the carrier is not used to *support*
   the donor computation.
3. **Open:** which training-data statistic or architectural pressure produces the carrier? (training-time, out
   of scope for a frozen checkpoint.)
4. **Open:** does an analogous positional circuit appear in another GFM?

After B1, describe it as a **real positional attention circuit that is not part of the donor circuit
(donor-scoped OV null + small suppressive coupling effect), with general OV content and origin still
unknown**.
