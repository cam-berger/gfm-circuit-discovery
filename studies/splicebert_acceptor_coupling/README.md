# SpliceBERT acceptor coupling

The [donor study](../splicebert_partner_selection/README.md) masks a donor and corrupts its intron-mate
acceptor, and its account of that behavior is asymmetric. This study reads the same intron from the other
end, to test whether the asymmetry belongs to the model or to the assay. It masks the `G` of the
acceptor's `AG` (`next_acceptor_pos - 1`), corrupts the partner donor `[d, d+2)`, and reads the same
native margin, `logit(G) − max(A, C, T)`, on `SpliceBERT.1024nt`. AC1–AC9 use canonical `GT-AG` internal
short introns on both strands and validation data.

The corruption instrument does not mirror. The acceptor-side operator is alt-invariant within ±25% (C42),
but 81.0% of validation donors have `seq[d-1] == "G"`, so donor-side alts such as `CC`, `TA`, and `AA`
plant new splice motifs. Only `AC` (two substitutions) and `AT` (one) are motif-clean, and comparisons
between the two directions must match substitution counts (R44).

## Findings

Masked-acceptor dependence on the partner donor reaches a bounded behavior gate, and in a bounded
post-outcome screen on the same validation cohort, destroying either donor base changes the endpoint
sub-additively (AC1, AC2; both verification tier). The exploratory 96-head paired screen is
instrument-unresolved and supplies candidates only (AC4). On a gene-disjoint target within validation chromosomes 11/16/19 (not chromosome-held-out), `L1H12`
mean ablation and `L4H11`/`L4H9`/`L4H7` noising show bounded, intervention-specific effects, reused
`L3H1` and a distinct route are inconclusive, and `L2H15` failed its health check; no reverse write,
delivery, mediation, or circuit is established (AC5–AC8). The frozen handoff to write and delivery tests stopped
before target science, because its validation rows violate the frozen consumer's schema (AC9). On one
validation reserve, natural `GC-AG` donors are associated with weaker masked-acceptor coupling and lower
`L3H1` donor attention than matched `GT-AG` donors (AC10, AC11; bounded verification). The comparison is
quasi-observational and does not establish intrinsic class specificity, a causal component, write,
delivery, splicing efficiency, or biology.

## Status

Stopped. Masked-acceptor dependence and natural donor-class heterogeneity are measured, but causal
construction of the acceptor margin remains unresolved, and the mechanism program ended at a terminal
instrument failure. These records must not be composed into a synthetic successful circuit, and nothing
here is a claim about splicing, spliceosome biochemistry, or transcript abundance.

## Files

- [`CLAIMS.md`](CLAIMS.md): the `AC*` ledger. Its [evidence
  catalog](CLAIMS.md#reader-facing-evidence-catalog) maps legacy aliases to descriptive IDs.
- [`CORRECTIONS.md`](CORRECTIONS.md): `ACR*` corrections. [`DECISIONS.md`](DECISIONS.md) and
  [`DECISIONS_R4_EDGE.md`](DECISIONS_R4_EDGE.md): chronology and the retired R4 edge design.
- [`preregistrations/`](preregistrations/): frozen protocols and the retired R4 draft.
- [`attestations/`](../../results/splicebert_acceptor_coupling/attestations/): attestations for the
  archived result packages.
- `scripts/splicebert_acceptor_coupling/` and `src/gfmcd/splicing/reverse_coupling.py`: runners and the
  reusable readout.

Read the [evidence ladder](../../docs/METHODOLOGY.md#evidence-ladder) and
[`docs/NOMENCLATURE.md`](../../docs/NOMENCLATURE.md) before writing prose here.
