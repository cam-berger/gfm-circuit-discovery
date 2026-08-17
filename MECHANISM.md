# Current SpliceBERT mechanism and its limits

> **Tier: T1 — canonical.** Current, authoritative. Read this.

This is the one-file reviewer narrative. Exact claims are controlled by
[`studies/splicebert_partner_selection/CLAIMS.md`](studies/splicebert_partner_selection/CLAIMS.md), and
correction history by
[`studies/splicebert_partner_selection/CORRECTIONS.md`](studies/splicebert_partner_selection/CORRECTIONS.md).

## Frozen mechanism statement

> A structured, top-ranked causal core binds a predominantly relative positional address to an
> acceptor-local payload, then delivers the resulting information to the donor readout within an
> irreducibly distributed computation.

## Checkpoints

Two checkpoints appear in this document and must not be conflated. The donor mechanism and the period-3
carrier described below are measured on `SpliceBERT.1024nt` (final, multi-species). The separate ≈10.63
carrier in *Key limitations* §4 is measured on `SpliceBERT.510nt` (stage-1, multi-species), because
establishing it required a matched contrast against its human-only twin `SpliceBERT-human.510nt`, which
differs only in training data. These are two checkpoints of one training lineage; a result from one is
not a result about the other. Term-by-term conventions are in
[`docs/NOMENCLATURE.md`](docs/NOMENCLATURE.md).

## Computation

SpliceBERT is asked to reconstruct the masked G at a splice donor. Corrupting that donor's paired
acceptor lowers the native donor margin, whereas matched non-partner and other-donor controls have much
smaller effects. The supported internal account is:

```text
period-3 positional coordinate across the donor–acceptor span
                    │
                    ▼
position geometry → L2H15 / MLP2 → L3H1 attention address
                                         ×
acceptor-local state → L0H7/H12/H10 → MLP0 → MLP1 → MLP2 payload
                                         │
                                         ▼
                              L3H1 donor-side write
                                         │
                                         ▼
                         low-rank U_info → L4H0/L4H2/L5H11
                                         │
                                         ▼
                                masked-donor MLM margin
```

The address is predominantly relative-offset based with a modest absolute-position component. The
payload is generated locally at the acceptor by a serial MLP-centered path. L3H1 combines the two and
writes a compact donor-side representation; downstream heads transport that representation to the final
readout. Period-3 restoration experiments localize the carrier's effect across the span and show that,
within the tested intervention, recovery runs through L3H1's attention pattern rather than
carrier-dependent changes in its own values.

## What “circuit” does and does not mean

The named components are causally implicated in this behavior through necessity, mediation, and path
tests. They form a **structured causal core within an irreducibly distributed computation**. They are not
claimed to be the smallest sufficient set.

That distinction is empirical, not rhetorical. The small hand-built circuit is necessary but has
faithfulness `F≈0.005` when the complement is scrubbed; the clean circuit outputs are sufficient when
injected, but the sparse circuit cannot regenerate those outputs alone. A held-out size curve reaches
90% faithfulness only near 72/103 nodes (grid-resolution; the sampled sizes bracket the 90% crossing to
(64, 72]) and has no knee. Static ranking gives an upper bound on the
smallest faithful set, so **minimality remains unclaimed**.

## Strongest supported interpretation

- **Behavior:** a donor prediction depends on its paired acceptor under the frozen short-intron assay.
- **Address:** L3H1 selection is driven mostly by learned positional geometry, not partner identity in
  the attention weights.
- **Payload location:** the source state is local to the four-token acceptor window.
- **Generation:** a serial L0→MLP0→MLP1→MLP2 path, led by L0H7/H12/H10, builds the value-path payload.
- **Binding/delivery:** a non-forced reconstruct–inject factorial requires attention address and value
  payload jointly; L3H1 writes a stable low-rank donor-side object that downstream heads transport.
- **Carrier role:** the learned period-3 code acts as a distributed relative-positional routing
  coordinate supporting L3H1 attention and donor coupling. Nucleotide-invariant local surgery cleanly
  confirms phase-aligned use in CDS blocks (`Ψ_C=+0.141 [0.110,0.172]` raw logit units), above same-geometry hidden-axis
  controls; matched coding-context enrichment remains unresolved.

## Key limitations and open questions

### 1. Payload semantic identity

This is the most important unresolved part of the headline mechanism. The payload is acceptor-local and
is not explained by the preregistered GC/PPT/k-mer/positional composition covariates. Absolute local
MaxEnt strength, however, was unstable across independently seeded draws and not significant under
two-way gene-dyadic inference. Therefore the payload must **not** be called splice-site strength. Its
meaning may be conditional competitiveness, a learned site-quality variable, or another local latent,
but none is established.

### 2. Cross-context transduction

Most donor-graph causal evidence comes from canonical short introns and intra-example clean/corrupt
patching. That establishes the mechanism on the tested distribution, not interventional generality.

The cross-example transduction test this section used to call for **has now been run, on both sides,
and the answer is not the one it anticipated.** Deriving a state on one gene and injecting it into a
gene-disjoint, context-mismatched target does move the target margin, largely and dose-monotonically —
but on the donor side the transfer is carried entirely by a **gene-generic readout axis**, and the
example-specific component is causally inert for the donor margin in the linear regime (**C41**, at
verification tier per R28/R32). On the acceptor side, content-specificity is **not separable at
achievable validation precision**: a norm-matched scramble still recovers about a third, and powering up
did not shrink the interval because the variance is gene-heterogeneity-dominated (**C38**).

So the open question has changed shape. It is no longer "can the state be transplanted at all" — it can —
but **whether anything example-specific is transported, and if so where it lives**. C41 bounds only the
C16/C26 final-residual state under additive injection; it says nothing about the addressing and routing
components (C17/C18/C23), which were never intervened on, nor about earlier-layer or non-additive
example-specificity. A clean acceptor-side call would need the sealed chr8/9 cohort, which has not been
opened.

### 3. Carrier origin

The period-3 carrier's present-day causal use is cleanly confirmed (C36). A single frozen checkpoint
cannot say why training built it, but two published-checkpoint natural experiments now give
*comparative* origin evidence without retraining (C39, C40):

- **Corpus, not human sequence.** In the sibling stage-1 checkpoints — same architecture, stage, and
  510-nt length, differing only in corpus — the period-3 carrier is strong in the 72-species model
  (band power 435× median) and undetected in the human-only model (6.2×, below a powered noise floor).
  This supports comparative attribution to *multi-species* pretraining. It does not isolate species
  diversity from corpus size, composition, sampling, or optimization history.
  In the sibling where it exists, band-deletion reproduces C8's exon-selective damage pattern (exon vs intron
  masked-base damage +0.170 vs +0.032 nats, DiD +0.138 against an energy-matched-control DiD max +0.005;
  intron-channel health 87.6%), reproducing C8 on a second checkpoint.
- **Not universal across the tested models.** In two single-nucleotide transformer MLMs trained on ORF-free ncRNA
  (RNA-FM, RNA-BERT), the period-3 carrier is absent from the position-embedding matrix under a powered
  null (band power ≈1× median control; a ≥5% carrier would have been detected). Codon periodicity is a
  plausible corpus correlate, not an established coding-specific or reading-frame mechanism.

Origin *dynamics* and a causal species-diversity interpretation require checkpoint series or controlled
corpus/objective ablations. The licensed claim is comparative association with multi-species
pretraining, not a training-time mechanism or coding-specific origin.

### 4. Separate ≈10.63 carrier

The ≈10.63 carrier is distinct from period 3 and, within the donor circuit, stops at post-softmax
attention (its general OV content and behavioral function remain open, C12/C37). **One donor-side arm is
open and is NOT evaluable as preregistered (R36):** B1's `U_info`-projected write test (`S_ov`, `Ω`) was
never computed, and every per-head OV branch is scoped to **L0** readers — the red-team's fix for upstream
position-propagation confounding — but no L0 head reaches Level A on this checkpoint (9 scanned, 0 passed).
Whether the carrier writes a carrier-specific component onto the donor readout is therefore **unresolved**
(protocol Endpoint B), not merely untested. Its **origin is now
corpus-attributed** by the same sibling contrast (C39): the ≈10.63 band is strong in the 72-species
stage-1 checkpoint (369× median) and undetected in the human-only stage-1 checkpoint (2.4×, powered
null) — so the "multispecies" label is now measured, not asserted. This is corroborated on the input
side: the model-free corpus spectrum finds no detectable ≈10.63 periodicity in 60 Mb of human genome at
the preregistered period (all dinucleotide channels Z≈0.93–1.00, n.s., with a ≥5% detection floor,
`corpus_spectrum.json`), and the carrier is likewise absent from the RNA-FM/RNA-BERT position
embeddings (C40). The carrier is therefore a feature multi-species training manufactured, not a
reflection of a human-sequence statistic.

"Corpus-attributed origin" does not reopen the nucleosomal reading: the ≈10.298-nt nucleosomal signal
sits 0.33 nt from the ≈10.630 carrier — a gap ~7× the largest uncertainty on either side (two-grid
sub-bin spread 0.045 nt; nucleosomal estimator spread 0.045 nt) — so the record holds that as a powered
refutation of the nucleosome bridge rather than support for it (C14, R17;
[`docs/NOMENCLATURE.md`](docs/NOMENCLATURE.md)). The carrier's OV/behavioral function inside the model
stays open; only its *corpus* origin is now licensed.

### 5. Reverse masked-acceptor construction

The reciprocal endpoint—masking the acceptor `G` and corrupting its paired donor—has a consolidated
but incomplete record (`AC1`–`AC11`). For canonical `GT-AG` introns, the validation gate records a
large pair-specific response; R3c supports bounded `L1H12` mean-ablation and `L4H11/H9/H7` noising
patterns, while `L3H1` reuse remains inconclusive and `L2H15` health-ineligible. A prospectively frozen
reader→post-`W_O` write→delivery program passed its R1c validation instrument but stopped at an
immutable R1c→R5 package-contract failure before target. It supplies no target behavior, write,
mediation, or delivery result.

A separate natural donor-class comparison shows weaker reverse coupling (`rho=0.2571`, upper bound
`0.5402`) and lower L3H1 donor attention (ratio `0.3755`, upper bound `0.5809`) for `GC-AG` than matched
`GT-AG` donors on 67 validation pairs. Those are bounded behavior and attention-geometry modifiers,
not a causal account. A later development-only score-margin diagnostic revises the early-head shorthand
to nominal `L0H14`/`L0H5`/`L0H3` candidates, leaves `L0H4`/`L1H6`/`L1H13` unresolved, and finds an
opposite-direction `L2H15` diagnostic. Its separate arm-conditioned interventions, unadjusted adaptive
family, incomplete clustering, and QK ambiguity prevent class-specific or claim promotion (`ACR23`);
the `MLP1` hypothesis likewise remains post-outcome.
Therefore the donor-side graph above must not be reflected across the intron: causal construction of
the masked-acceptor margin remains unresolved.

## Scope

The core behavior is established for one SpliceBERT checkpoint, canonical internal short introns, both
strands, and the recorded gene splits. It is a model mechanism, not a direct claim about spliceosome
biochemistry or transcript abundance. Cross-example generality is bounded by C38/C41 as described
above; wider biological and cross-model generalization remains open.

**Units.** Coupling and its patching/restoration effects are raw MLM logit-differences (logit units); "nats"
denotes masked-base cross-entropy damage only. See [`docs/NOMENCLATURE.md`](docs/NOMENCLATURE.md).

**Two scope facts.** The acceptor-payload arc (A9/A9b, C30) ran entirely on the **validation** split and never
opened the sealed chr8/9 test. **No external biological assay** enters the study at any point — no RNA-seq, no
measured junction usage, no PSI; every result is model-internal.
