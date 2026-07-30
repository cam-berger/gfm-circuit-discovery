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
  confirms phase-aligned use in CDS blocks (`Ψ_C=+0.141 [0.110,0.172]`), above same-geometry hidden-axis
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
The Head–Context Atlas broadens observational and component-level validation across clustered,
multi-pair, cassette, and matched-control contexts, but it does not yet show that the same internal state
can be transplanted across examples or contexts to predictably move behavior.

A decisive follow-up should perform **cross-example transduction**: derive a source state or direction
on one gene/context, inject it into a gene-disjoint target while matching site type, distance, strength,
coding status, and corruption magnitude, and require a dose-ordered target-margin change above
matched-random and within-example controls. This connects the Atlas to the donor circuit without
promoting attention or attribution to causality.

### 3. Carrier origin

The period-3 carrier's present-day causal use is cleanly confirmed (C36). A single frozen checkpoint
cannot say why training built it, but two published-checkpoint natural experiments now give
*comparative* origin evidence without retraining (C39, C40):

- **Corpus, not human sequence.** In the sibling stage-1 checkpoints — same architecture, stage, and
  510-nt length, differing only in corpus — the period-3 carrier is strong in the 72-species model
  (band power 435× median) and undetected in the human-only model (6.2×, below a powered noise floor).
  The human-only absence is the surprising part: codon structure exists in human coding sequence, yet
  training on human alone did not build the period-3 code. It is a product of *multi-species* pre-training.
  In the sibling where it exists, band-deletion reproduces C8's exon-selective damage pattern (exon vs intron
  masked-base damage +0.170 vs +0.032 nats, DiD +0.138 against an energy-matched-control DiD max +0.005;
  intron-channel health 87.6%), reproducing C8 on a second checkpoint.
- **Codon-linked, not universal.** In two single-nucleotide transformer MLMs trained on ORF-free ncRNA
  (RNA-FM, RNA-BERT), the period-3 carrier is absent from the position-embedding matrix under a powered
  null (band power ≈1× median control; a ≥5% carrier would have been detected). A codon-scale positional
  code does not form when the training corpus lacks reading frames — consistent with a codon origin.

Origin *dynamics* (the trajectory over training) still require checkpoint series or controlled retraining
that public artifacts do not provide; the licensed claim is comparative attribution, not a training-time
mechanism.

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

## Scope

The core behavior is established for one SpliceBERT checkpoint, canonical internal short introns, both
strands, and the recorded gene splits. It is a model mechanism, not a direct claim about spliceosome
biochemistry or transcript abundance. Generalization requires the transduction and cross-model tests
above.

**Units.** Coupling and its patching/restoration effects are raw MLM logit-differences (logit units); "nats"
denotes masked-base cross-entropy damage only. See [`docs/NOMENCLATURE.md`](docs/NOMENCLATURE.md).

**Two scope facts.** The acceptor-payload arc (A9/A9b, C30) ran entirely on the **validation** split and never
opened the sealed chr8/9 test. **No external biological assay** enters the study at any point — no RNA-seq, no
measured junction usage, no PSI; every result is model-internal.
