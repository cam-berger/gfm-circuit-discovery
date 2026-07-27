# Atlas data catalog

> **Tier: T1 — canonical for source admission.**

No source enters the target census until its release, assembly, license/access terms, checksums,
coordinate convention, evidence semantics, and cluster/split audit are recorded in a manifest.

| Element family | Site classes | Preferred evidence | Adapter state | Target-run state |
|---|---|---|---|---|
| Core promoter | `promoter.tata`, `promoter.inr`, `promoter.bre`, `promoter.dpe` | Curated promoter/TSS records plus direct CAGE or promoter assay support; motif calls labeled derived/computational | Normalized BED ingestion available; source-specific adapter pending | blocked |
| Splicing | `splice.donor`, `splice.acceptor` | Curated transcript annotation | GENCODE GTF derivation implemented | blocked |
| Intron regulation | `splice.branchpoint`, `splice.polypyrimidine_tract` | Experimentally supported branch-point/tract resources preferred | source-specific adapter pending | blocked |
| 3′ end processing | `polya.signal`, `polya.cleavage_region` | Curated poly(A) sites with cleavage evidence | source-specific adapter pending | blocked |
| TF binding | `tfbs.<factor>` | ChIP/exo/footprinting-supported sites; motif-only calls remain computational | normalized BED ingestion available | blocked |
| Translation initiation | `translation.start_codon`, `translation.kozak` | Curated coding transcript annotation; ribosome evidence as an optional higher tier | GENCODE GTF derivation implemented | blocked |
| Translation termination | `translation.stop_codon` | Curated coding transcript annotation | GENCODE GTF derivation implemented | blocked |

## Locally available development assets

- GRCh38/hg38 sequence and GENCODE v50 GTF are present outside the repository and remain external.
- EPDnew human release 006 and FANTOM5 files exist in a legacy worktree. They are not silently
  inherited; a new source manifest and coordinate audit are required before atlas admission.

Large genomes, raw annotations, processed collections, candidate pools, and score shards remain
ignored. Only compact summaries, manifests, audits, schemas, code, and claim-controlling artifacts
belong in Git.

## Normalized BED contract

Source-specific adapters emit tab-separated rows with:

`chrom`, `start`, `end`, `feature_id`, `score`, `strand`, `site_class`,
`homology_cluster`, followed by optional `key=value` fields.

Coordinates are zero-based half-open on the named assembly. `feature_id` is globally stable within
the pinned release. `homology_cluster` is the gene/sequence-family unit used before split assignment.
The builder receives source ID, release, and evidence type explicitly and writes them into every
element. A candidate-control pool uses the simpler JSONL `SiteReference` contract
(`example_id`, `start`, `end`, `site_class`, optional `element_id`) and is produced without model
outcomes.

## Evidence labels

- `experimentally_supported`: the source record itself carries direct experimental support.
- `curated`: accepted curated annotation without a direct assay claim for the exact element.
- `derived_from_curated_annotation`: deterministic boundary/window derived from curated annotation.
- `computational_prediction`: motif or model call only.

The atlas must stratify or restrict by evidence type; it must never merge these labels into a single
“annotated” biological claim.
