# SpliceBERT donor–acceptor coupling — results index

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Machine-readable entry point: [`../MANIFEST.csv`](../MANIFEST.csv) /
[`../MANIFEST.json`](../MANIFEST.json). Filter the `artifact` column on
`results/splicebert_coupling/` to get every file in this directory with its `claim_id`, `status`,
`evidence_level`, controlling `script`, sibling `manifest`, and consuming `figure`. The column
semantics and the four `status` values are defined in [`../RESULTS_INDEX.md`](../RESULTS_INDEX.md).

Authoritative claim wording, corrections, and retractions live in
`studies/splicebert_partner_selection/CLAIMS.md` and `CORRECTIONS.md`; the running rationale for the
A0→A12 arc is in `DECISIONS_A0.md`. Nothing in this index may be read as a claim.

## Decoding the file namespace

The review flagged this namespace as opaque to a reader who has not read `DECISIONS_A0.md`
(`docs/REVIEW_2026-07-22.md` §2.2). The manifest is the fix — every file resolves to a claim — but
the prefixes decode as follows.

| Prefix | Meaning |
|---|---|
| `a0_` … `a12_` | the donor-graph experiment arc, in run order; `a2b_`, `a9b_`, `a9c_` are follow-ups to the numbered parent |
| `preflight_*` | the smoke/calibration gate for the like-named experiment; memorials and arithmetic only, target data sealed |
| `ic1_`, `ic2_` | the intron-coupling arc opened after R22; `ic1_pilot*` are its pilots, `ic2_*` are design-stage only (IC2 was never run confirmatorily) |
| `ed1_` | the exon-vs-intron **definition** arc: is donor→acceptor coupling architecture-conditioned? `ed1_cohort_pilot_*` / `ed1_matching_pilot_*` are outcome-blind feasibility and balance (annotation only, no model); `ed1_scores_*` are raw per-state donor margins; `ed1_trainpilot*` are TRAIN pilots that seal the scale constants; `ed1_architecture_boundary_pairing.json` is the val confirmatory |
| `*_vN` | a re-run under an amended protocol; the controlling artifact is the one the manifest marks `canonical`, earlier ones being `superseded` or `development` |
| `sealed_test`, `sealed_val` | the sealed behavioural test and its validation split — do not reopen as a routine smoke check |
| `manifest.json` | directory-level provenance for the exploratory `run_coupling.py` run, not a per-artifact manifest |

Everything else is named for the assay it runs (`donor_*`, `ablation_scan`, `path_decomposition`,
`readout_path`, `subspace`, …).

## Where to start

The claim → runner → artifact table is in [`../../REPRODUCE.md`](../../REPRODUCE.md) §3, and the
detailed command map in `studies/splicebert_partner_selection/REPRODUCIBILITY.md`.

## Integrity check

```bash
uv run python scripts/verify_manifest.py
```

Every file added, removed, or moved here must be reflected in `results/MANIFEST.{csv,json}` in the
same change.

## ED1 development intermediates deliberately NOT committed

`ed1_cohort_pilot_{train,val}.json` and `ed1_matching_pilot_val.json` (~38 MB) are **annotation-only**
covariate tables containing no model output. They are deterministically regenerable without a GPU and
are therefore kept out of the repository:

```bash
PYTHONHASHSEED=0 uv run python scripts/splicebert/run_ed1_cohort_pilot.py --split val
PYTHONHASHSEED=0 uv run python scripts/splicebert/run_ed1_matching_pilot.py
```

The GPU-derived raw margins (`ed1_scores_*.json`) ARE committed — they are not cheaply reproducible and
are read directly by `verify_ed1.py` and `run_ed1_sensitivity_bound.py`. Source, output and biological
input hashes for the whole arc are pinned in `ed1_provenance_audit.json`.
