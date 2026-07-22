# Current SpliceBERT mechanism and its limits

This is the one-file reviewer narrative. Exact claims are controlled by
[`studies/splicebert_partner_selection/CLAIMS.md`](studies/splicebert_partner_selection/CLAIMS.md), and
correction history by
[`studies/splicebert_partner_selection/CORRECTIONS.md`](studies/splicebert_partner_selection/CORRECTIONS.md).

## Frozen mechanism statement

> A structured, top-ranked causal core binds a predominantly relative positional address to an
> acceptor-local payload, then delivers the resulting information to the donor readout within an
> irreducibly distributed computation.

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
tests. They form a **structured causal core**. They are not claimed to be the smallest sufficient set.

That distinction is empirical, not rhetorical. The small hand-built circuit is necessary but has
faithfulness `F≈0.005` when the complement is scrubbed; the clean circuit outputs are sufficient when
injected, but the sparse circuit cannot regenerate those outputs alone. A held-out size curve reaches
90% faithfulness only near 72/103 nodes and has no knee. Static ranking gives an upper bound on the
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

The period-3 carrier's present-day causal use is now cleanly confirmed, but a frozen checkpoint cannot
identify why training created it. The matched CDS-minus-non-CDS-exon contrast was inconclusive, so even
coding enrichment is not licensed by C36. Origin requires training-time evidence: checkpoint
trajectories, controlled retraining, objective/corpus ablations, or a cross-model comparison.

### 4. Separate ≈10.63 carrier

The multispecies ≈10.63 feature is distinct from period 3 and currently stops at post-softmax attention.
Its OV content, behavioral function, and biological origin remain open.

## Scope

The core behavior is established for one SpliceBERT checkpoint, canonical internal short introns, both
strands, and the recorded gene splits. It is a model mechanism, not a direct claim about spliceosome
biochemistry or transcript abundance. Generalization requires the transduction and cross-model tests
above.
