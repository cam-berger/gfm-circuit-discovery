# Atlas data catalog

> **Tier: T1 — canonical for source admission.**

No source enters the target census until its release, assembly, license/access terms, checksums,
coordinate convention, evidence semantics, and cluster/split audit are recorded in a manifest.

| Element family | Site classes | Preferred evidence | Adapter state | Target-run state |
|---|---|---|---|---|
| Core promoter | `promoter.epd_tss_core` | EPDnew CAGE-supported TSS records | EPDnew v006 adapter implemented; five-nt TSS core | context-margin development-qualified but not confirmation-replicated |
| Splicing | `splice.donor`, `splice.acceptor` | Curated transcript annotation | GENCODE GTF derivation implemented | context-margin positive-control unresolved; not target-scored |
| Intron regulation | `splice.branchpoint`, `splice.polypyrimidine_tract` | Experimental lariat-seq for branch points; curated derivation for tracts | Mercer exact round-trip liftover and GENCODE tract adapters implemented | both context-margin effects confirmation-replicated |
| 3′ end processing | `polya.signal`, `polya.cleavage_region` | Multi-protocol 3′-end sequencing atlas | PolyASite 2.0 adapter implemented | both context-margin effects confirmation-replicated |
| TF binding | `tfbs.ctcf_peak_core`, `tfbs.rest_peak_core`, `tfbs.cebpb_peak_core` | Multi-experiment ENCODE ChIP clusters | UCSC/ENCODE cluster adapter implemented; derived five-nt peak center | all three anchor-context effects confirmation-replicated; motifs unresolved |
| Enhancer | `enhancer.encode_dels_core` | ENCODE Registry dELS biochemical annotation | ENCODE v4 cCRE adapter implemented; derived five-nt cCRE center | anchor-context effect confirmation-replicated; motif/base unresolved |
| Translation initiation | `translation.start_codon`, `translation.kozak` | Curated coding transcript annotation; ribosome evidence as an optional higher tier | GENCODE GTF derivation implemented | start-codon context-margin effect confirmation-replicated; Kozak supply-quarantined |
| Translation termination | `translation.stop_codon` | Curated coding transcript annotation | GENCODE GTF derivation implemented | context-margin development gate failed; confirmation sealed |

## Locally available development assets

- GRCh38/hg38 sequence and GENCODE v50 GTF are present outside the repository and remain external.
- EPDnew human release 006, PolyASite 2.0, UCSC ENCODE TF clusters, ENCODE Registry v4 cCREs,
  Mercer high-confidence branch points, and both UCSC liftover chains are pinned in
  `SOURCE_MANIFEST_V2.json`. Raw files remain ignored.

Large genomes, raw annotations, processed collections, candidate pools, and score shards remain
ignored. Only compact summaries, manifests, audits, schemas, code, and claim-controlling artifacts
belong in Git.

## Normalized BED contract

Source-specific adapters emit tab-separated rows with:

`chrom`, `start`, `end`, `feature_id`, `score`, `strand`, `site_class`,
`homology_cluster`, followed by optional `key=value` fields. Multisource v2 replaces source-local
clusters with a shared locus-block identity after normalization.

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
