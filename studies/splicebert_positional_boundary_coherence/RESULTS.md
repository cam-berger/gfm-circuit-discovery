# Results index — SpliceBERT positional boundary coherence

> Model-free Phase I stopped on an independently audited necessary-capacity
> failure. No model outcome was produced.

The freshness and census records establish the inputs to the stop, not a target result. The full
2,343,604,295-byte candidate census is an ignored, deterministically regenerable intermediate; its exact
byte hash (`9be356ac…614`) and producing revision (`addd7be9`) are bound by the retained producer manifest
and independent audit.

| Run / result | Question | Status | Artifact | Manifest | Claim IDs | Notes |
|---|---|---|---|---|---|---|
| Freshness registry v1 | Were all frozen historical source genes excluded? | `AUDIT_PASS` | [`freshness_registry_v1.audit.json`](../../results/splicebert_positional_boundary_coherence/development/freshness_registry_v1.audit.json) | [`manifest`](../../results/splicebert_positional_boundary_coherence/development/freshness_registry_v1.audit.manifest.json) | — | 5,998 unique version-normalized genes; model-free. |
| Candidate census v1 | Can the real-sequence census be independently reconstructed from pinned hg38/GENCODE/MaxEnt inputs? | `AUDIT_PASS` | [`candidate_census_v1.audit.json`](../../results/splicebert_positional_boundary_coherence/development/candidate_census_v1.audit.json) | [`manifest`](../../results/splicebert_positional_boundary_coherence/development/candidate_census_v1.audit.manifest.json) | — | 6,861 donor, 7,827 acceptor, and 50,773 scaffold candidates. Passing this audit did not authorize a model target. |
| Composition-balanced cohort v1 | Can the frozen 120/600/384-block, seven-source, globally gene-disjoint design be supplied? | `AUDIT_PASS_INSTRUMENT_UNRESOLVED` | [`composition_balanced_cohort_v1.json`](../../results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.json); [`audit`](../../results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.audit.json) | [`producer`](../../results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.manifest.json); [`audit`](../../results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.audit.manifest.json) | `PBC1` | Donor phase 2 supplies 809/1,104 genes (deficit 295); acceptor phase 2 supplies 854/1,104 (deficit 250). Stopped before block selection, balance/classifier calibration, checkpoint loading, or outcomes. |

Artifacts are never overwritten. Phase II and Phase III remain sealed. Every later figure or claim row must
link to the exact indexed artifact, manifest, and independent audit that support it.
