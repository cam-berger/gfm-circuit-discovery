# Head–Context Atlas — results index

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Machine-readable entry point: [`../MANIFEST.csv`](../MANIFEST.csv) /
[`../MANIFEST.json`](../MANIFEST.json). Filter the `artifact` column on
`results/splicebert_head_context_atlas/` to get every file here with its `claim_id` (`HCA*`),
`status`, `evidence_level`, controlling `script`, sibling `manifest`, and consuming `figure`. Column
semantics and the four `status` values are defined in [`../RESULTS_INDEX.md`](../RESULTS_INDEX.md).

Authoritative claim wording lives in `studies/splicebert_head_context_atlas/CLAIMS.md`; the decision
log is `DECISIONS.md` in the same directory. Nothing in this index may be read as a claim. The Atlas
is infrastructure and the substrate for cross-context work, not a standalone headline
(`docs/REVIEW_2026-07-22.md` §9).

## Decoding the file namespace

| Element | Meaning |
|---|---|
| `atlas_context_census_*`, `atlas_context_enrichment_*`, `atlas_context_matched_*` | annotation-only cohort construction and matched-cohort audits (HCA11) |
| `atlas_model_preflight_*` | instrument preflight on the model (HCA12) |
| `atlas_discovery_screen_*`, `atlas_discovery_*` | the 102-component discovery screen and its merged matrices (HCA12, HCA18) |
| `atlas_discovery_analysis_*` | the frozen discovery-analysis instrument and its outputs (HCA13, HCA18) |
| `atlas_mechanism_package_*`, `atlas_mechanism_preflight_*`, `atlas_mechanism_validation_*` | whole-component/group validation on the gene-disjoint internal-validation cohort (HCA14, HCA15, HCA19) |
| `atlas_stage2_*` | the conditional QK/V/OV and downstream-path stage (HCA16, HCA20) |
| `atlas_representative_dag_*` | representative DAG bundles for the visualizer (HCA17) |
| `*_smoke_vN`, `*_smoke_*` | smoke/plumbing runs; memorials and mechanics only, outcome-sealed |
| `*.audit.json` | an independent reconstruction of the like-named result |
| `*.manifest.json` | per-artifact provenance (checkpoint, data hashes, split, seed, revision) |
| `*.summary.json`, `*.jsonl`, `*.jsonl.gz` | compact per-run summaries and matrices |
| `_vN` | protocol/amendment revision; the manifest marks which one is `canonical` |

Bulk `target_shards/`, `health_shards/`, packet matrices, and Colab distribution archives are
gitignored as reproducible intermediates or external run bundles; only the compact artifacts needed
to audit the recorded claims are version-controlled. The manifest therefore indexes the committed
tree, not the full runtime output.

## Where to start

The stage → entry point → record table is in [`../../REPRODUCE.md`](../../REPRODUCE.md) §3.

## Integrity check

```bash
uv run python scripts/verify_manifest.py
```

Every file added, removed, or moved here must be reflected in `results/MANIFEST.{csv,json}` in the
same change.
