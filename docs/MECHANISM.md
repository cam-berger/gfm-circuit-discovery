# SpliceBERT donor mechanism and its limits

Claims, with their evidence levels and scope, are in
[`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md);
corrections are in
[`studies/splicebert_partner_selection/CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md).
Terms follow [`docs/NOMENCLATURE.md`](NOMENCLATURE.md).

## Frozen mechanism statement

> A structured, top-ranked causal core combines a predominantly relative positional address with an
> acceptor-local payload, then delivers the resulting information to the donor readout within a
> distributed computation under the tested interventions.

## Relation to the current paper

The NeurIPS draft builds its discovery workflow on this frozen donor-side mechanism. The draft's
positional-scaffold/content-gating hypothesis is not part of the mechanism: C36 establishes phase-aligned
use of learned period-3 geometry, C43 a separate register-shaped behavior in one fixed-span chimeric
lattice, and C44 no mediation result, because its instrument failed authorization before the target
cohort was exposed.

## Checkpoints

The donor mechanism and the period-3 carrier are measured on `SpliceBERT.1024nt` (final, multi-species).
The ≈10.63 carrier's attention circuit (C11) is measured on `SpliceBERT.510nt` (stage-1, multi-species),
because establishing it needed a matched contrast with its human-only twin `SpliceBERT-human.510nt`,
which differs only in training data. Its donor-circuit OV, behavior and B1 reader-screen tests ran on
`SpliceBERT.1024nt`. The two multi-species checkpoints are one training lineage, but a result from one is not
a result about the other.

## The mechanism

In the frozen short-intron assay, SpliceBERT reconstructs the masked G at a splice donor. Corrupting the
donor's paired acceptor lowers the donor margin (the logit of G minus the largest of A, C and T), while
matched non-partner and other-donor controls have much smaller effects.

```text
ADDRESS   period-3 positional code across the donor–acceptor span
            → position geometry → L2H15 / MLP2 → L3H1 attention
PAYLOAD   acceptor-local state → L0H7/H12/H10 → MLP0 → MLP1 → MLP2
WRITE     L3H1 combines address and payload → low-rank donor-side state U_info
DELIVERY  U_info → L4H0 / L4H2 / L5H11 → masked-donor MLM margin
```

L3H1 selects the acceptor mostly by learned positional geometry rather than partner identity in its
attention weights (C22), with a substantial relative-offset signal plus absolute-position and edge
modulation (C32). The payload originates in the four-token acceptor window and is built by a serial,
MLP-centered path (C29, C31). L3H1's contribution needs address and payload jointly (C23); downstream
heads carry its low-rank donor-side write to the readout.

The learned period-3 code acts as a distributed relative-positional routing coordinate. Its effect is
spread across the span (C34), and within the tested intervention it works through L3H1's attention
pattern rather than carrier-dependent changes in L3H1's values (C35). Shifting the period-3 direction
within CDS blocks, with nucleotides unchanged, cleanly confirms phase-aligned use above same-geometry
hidden-axis controls (`Ψ_C=+0.141 [0.110,0.172]` logit units; C36). Matched coding-context enrichment
remains unresolved.

## What we mean by circuit

The named components are causally implicated by necessity, mediation and path tests. They form a
structured causal core within distributed computation under the tested node ranking and interventions;
we do not claim they are the smallest sufficient set. The hand-built circuit is necessary but has
faithfulness `F≈0.005` when the complement is scrubbed, and the nine-component core's clean outputs are
sufficient when injected, but the core cannot regenerate them alone (C27). A held-out size curve reaches
90% faithfulness at an interpolated 64.2 nodes, first exceeds it at the sampled point 72/103 (so the
crossing lies in (64, 72]), and has no knee (C33). A static ranking only bounds the smallest faithful set
from above, so minimality remains unclaimed.

## Limitations and open questions

### 1. Payload semantic identity

This is the most important unresolved part of the mechanism. The payload is acceptor-local and is not
explained by the preregistered GC, PPT, k-mer and positional composition covariates. Absolute local
MaxEnt strength was unstable across seeded draws and not significant under two-way gene-dyadic
inference, so the payload must not be called splice-site strength (withdrawn under R22). It may be
conditional competitiveness, a learned site-quality variable or another local latent; none is
established (C30).

### 2. Cross-context transduction

Most donor-graph evidence comes from canonical short introns and within-example clean/corrupt patching,
so it establishes the mechanism on the tested distribution, not interventional generality. A state
injected from one gene into a gene-disjoint, context-mismatched target moves its margin, largely and
dose-monotonically. On the donor side, a gene-generic readout axis carries all of that transfer, and the
example-specific component is causally inert for the donor margin in the linear regime (C41, a
verification-level result, not confirmatory; R28, R32). On the acceptor side, content specificity is not
separable at achievable validation precision: a norm-matched scramble still recovers about a third (C38).

C41 bounds only the C16/C26 final-residual state under additive injection; example-specific
information in the addressing and routing components (C17/C18/C23), in earlier layers or in non-additive
form remains open. A clean acceptor-side call would need sealed test data (C38).

### 3. The separate ≈10.63 carrier

The ≈10.63 carrier is distinct from period 3. Within the donor circuit it stops at post-softmax
attention. Its readers' values are sub-SESOI on the donor coupling margin (not a powered equivalence;
R31), deleting it has a small suppressive effect, and a diagnostic puts its general OV content on the MLM
output at a bounded null. The ledger treats the general-OV and behavioral branches as terminal (C12,
C37). B1, the
preregistered test of whether it writes onto the donor-side state `U_info`, was never computed. That test
was scoped to layer-0 reader heads, and no layer-0 head on this checkpoint passed the reader screen
(9 scanned, 0 passed), so the question is unresolved, not merely untested
(R36 in the [correction ledger](../studies/splicebert_partner_selection/CORRECTIONS.md)).

### 4. Carrier training lineage

A frozen checkpoint cannot say why a carrier is present, but published checkpoints give training-lineage
evidence without retraining: 510-nt stage-1 siblings that differ only in corpus, and two single-nucleotide
transformer MLMs trained on ORF-free ncRNA.

| Comparison | Period-3 carrier | ≈10.63 carrier | Source |
|---|---|---|---|
| 72-species stage-1 sibling | strong (band power 435× median) | strong (369×) | C39 |
| Human-only stage-1 sibling | undetected (6.2×, below a powered noise floor) | undetected (2.4×, under the tested floor) | C39 |
| RNA-FM and RNA-BERT position embeddings | absent under a powered null (≈1× median; a ≥5% carrier would be detected) | absent | C40 |
| 60 Mb of human genome, model-free spectrum | — | undetected at the preregistered period (dinucleotide channels Z≈0.93–1.00, n.s.; ≥5% floor) | C40 |

In the 72-species sibling, deleting the period-3 band reproduces C8's exon-selective damage on a second
checkpoint, at verification level (masked-base damage +0.170 nats in exons versus +0.032 in introns, a
difference-in-differences of +0.138 against at most +0.005 for energy-matched control bands;
intron-channel health 87.6%; R33). The comparisons show a comparative association with multi-species
pretraining but do not separate species diversity from corpus size, composition, sampling or
optimization; they establish no training-time mechanism and no training-time or coding-specific origin.
Codon periodicity is a plausible corpus correlate of period 3, not an established coding-specific or
reading-frame mechanism, and origin dynamics would need checkpoint series or controlled corpus or
objective ablations.

The lineage association does not reopen the nucleosomal reading. The ≈10.298-nt signal in mouse ES-cell
chromatin (Voong 2016, mm9) sits 0.33 nt from the ≈10.630 carrier, about seven times the 0.045-nt spread
of either estimate, so it is a powered refutation of the nucleosome bridge (C14, R17).

### 5. Reciprocal masked-acceptor construction

The reverse experiment masks the acceptor `G` and intervenes on its paired donor (records and numbers in
the [acceptor-coupling ledger](../studies/splicebert_acceptor_coupling/CLAIMS.md)). Its validation gate
records a large pair-specific response (verification tier), but the component account is deliberately
partial. On a held-out validation cohort (500 rows, 304 genes, gene-disjoint within validation
chromosomes), `L1H12` is supported under token-mean ablation and `L4H11`, `L4H9` and `L4H7` under
activation noising; `L3H1` reuse is inconclusive and `L2H15` fails the frozen health check. The hand-off
to a mechanism analysis ended in a terminal instrument failure before any target was scored, which is
neither a behavior null nor grounds to repair and retry. Natural `GC-AG` donors show weaker
masked-acceptor behavior and lower `L3H1` donor attention than matched `GT-AG` donors (bounded
verification records). None of this establishes a mirror image of the donor graph, an acceptor-side
address, payload, write, delivery, mediation, sufficiency or complete circuit, or a biological mechanism.

## Scope

The core behavior is established for one SpliceBERT checkpoint, canonical internal short introns, both
strands and the recorded gene splits. It is a model mechanism, not a direct claim about spliceosome
biochemistry or transcript abundance: no external biological assay (RNA-seq, measured junction usage,
PSI) enters the study. C38 and C41 bound cross-example generality; wider biological and cross-model
generalization remains open. The acceptor-payload arc (A9/A9b, C30) ran entirely on the validation
split. Coupling and its patching and restoration effects are raw MLM logit differences (logit units);
"nats" denotes masked-base cross-entropy damage only.
