# SpliceBERT findings of record

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

This is the concise scientific narrative for the donor-circuit study. Exact live wording is controlled by
[`CLAIMS.md`](CLAIMS.md); full superseded narratives and R1–R24 are preserved in
[`CORRECTIONS.md`](CORRECTIONS.md). The repository-level [`../../MECHANISM.md`](../../MECHANISM.md) is the
one-file reviewer account.

## Status at a glance

| Finding | Strongest evidence | Current interpretation |
|---|---|---|
| Donor–acceptor coupling | sealed behavior + specificity | A masked donor score depends on its own paired acceptor in the canonical short-intron domain. |
| Structured donor core | QK → attention → OV/mediation → behavior | Address/payload binding and low-rank delivery form a causal core, not a minimal or sparse-faithful whole circuit. |
| Period-3 carrier | position → routing → mediation → clean confirmatory behavior | A distributed relative-positional routing coordinate is causally used in a phase-aligned manner; coding enrichment remains unresolved. |
| Multispecies ≈10.63 carrier | position → QK → post-softmax attention | A distinct positional attention circuit; OV, behavior, and origin remain open. |
| Human-only ≈10.2 proposal | position/QK; fails attention | The proposed chromatin-reader circuit is closed at the attention gate. |
| Nucleosomal ≈10.298 signal | sequence-side biological assay | A strong biological signal exists but is distinct from the ≈10.63 carrier. |

## 1. Behavior and scope

Masking the G of a donor `GT` and corrupting its paired acceptor `AG→CC` changes the native donor margin
by **+0.872 [0.771, 0.969]** on the sealed chr8/9 split (n=400). Matched non-partner AG and other-donor
controls change it by only **+0.007/+0.005**. All frozen gates passed.

This licenses partner-specific nonlocal coupling for canonical internal short introns (60–150 nt), both
strands, one checkpoint, and the recorded splits. It does not establish literal transcript-level partner
selection, splicing in cells, typical long-intron generality, or cross-context transduction.

## 2. Current causal account

The supported computation is:

1. **Address:** learned position geometry supplies a predominantly relative donor→acceptor address with
   modest absolute-position modulation. L2H15/MLP2 amplify and route this address into L3H1 attention.
2. **Payload generation:** L0H7/H12/H10 feed MLP0→MLP1→MLP2 in a serial acceptor-local value path.
3. **Binding:** a non-forced reconstruct–inject factorial requires the attention address and acceptor
   values jointly for L3H1's coupling contribution.
4. **Delivery:** L3H1 writes a compact donor-side `U_info` object; L4H0/L4H2/L5H11 transport it to the
   final masked-donor readout.

`{L3H1, L2H15}` removes **81.6%** of coupling while retaining **99.2%** masked-MLM health. Rank-8
`U_info` patching recovers about **0.92** versus matched random subspaces ≤**0.05**, and an independently
derived `U_info` generalizes across disjoint genes.

## 3. Sufficiency is not minimality

The identified core is necessary and its clean outputs are sufficient when injected, but it does not
reconstruct the behavior from a scrubbed complement: `F(C)=0.005`. The non-embedding compute spine has
post-hoc necessity `N(C without embed)=0.989`, while injected clean outputs give
`D(C without embed)=0.986`.

A held-out backward-ranked size curve reaches 50/80/90/95% faithfulness at approximately
34/56/72/80 of 103 nodes and has no knee. The named mechanism is therefore a **structured top-ranked
core embedded in an irreducibly distributed computation**. Static ranking only upper-bounds the smallest
faithful set; no minimal-circuit claim is made.

## 4. Payload location is established; semantics are not

The pre-L3H1 payload converges in the four-token acceptor window and is generated serially through the
L0→MLP2 backbone. It is not concentrated in the donor-derived `U_info` basis, implying representation
conversion across L3H1.

A9 originally associated the payload with local MaxEnt strength on one held-out sample. A9b and the
cross-sample audit showed:

- preregistered bulk/frequency composition has no predictive power for the transferred payload under
  adequate common support;
- the original strength association reproduces on its exact sample but varies across independently
  seeded draws;
- every composition-adjusted two-way gene-dyadic CI includes zero.

Therefore the payload is **acceptor-local and not bulk composition**, but its semantic identity is open.
“Encodes splice-site strength” is withdrawn (R22).

## 5. Period-3 carrier: distributed routing coordinate

Deleting the rank-2 period-3 position band collapses coupling **0.946→0.027**. Matched single-head
restoration identifies L2H15/L3H1 as mediators (**0.363/0.348**, null max **0.042**).

C34/C35 refine the role:

- restoring the carrier only at donor/acceptor windows recovers roughly 2–4%; broad restoration across
  the span is required (C34);
- on a period-3-deleted base, restoring L3H1's intact-carrier attention pattern recovers
  **+0.257/+0.284** on discovery/confirm, while restoring its values gives **−0.011/−0.005** (C35).

The licensed interpretation is a distributed relative-positional routing coordinate acting mainly
through L3H1 attention within this intervention. C34's confirm split is gate-qualified because M1 missed
its frozen threshold (5.5%>5%); C35 does not exclude upstream carrier effects on payload construction.
Training origin is not identifiable from the frozen checkpoint.

C36 supplies the clean, nucleotide-invariant confirmation that C34 lacked. Moving only the learned P3
position component inside annotation-safe blocks—without changing any nucleotide, codon, motif, MaxEnt
score, coordinate, or sequence length—produces a material modulo-3 recovery contrast in CDS blocks:
**Ψ_C=+0.141 [0.110,0.172] nats** across 943 untouched genes. In-register shifts `{3,6,9}` preserve
coupling at about **0.757**, whereas adjacent off-register shifts reduce it to about **0.614**; the
identical upstream-intron surgery changes coupling by only about **0.002**. The learned P3 direction
exceeds all 19 signed-permutation controls that preserve its spatial waveform, Gram matrix, norms,
Fourier support, and intervention dose (paired excess **+0.136 [0.051,0.222]**; exact rank p=0.05).

The matched coding boundary did **not** resolve: CDS-minus-non-CDS-exon **+0.006
[−0.080,+0.093]**, n=52 pairs. Therefore C36 establishes phase-aligned use of the learned P3 direction
in CDS blocks, not coding specificity, reading-frame sensing, or training origin.

## 6. Other positional/biological results

The separate multispecies **≈10.63** carrier reaches controlled post-softmax attention in eight heads;
its OV content, behavior, and biological origin remain unknown. The human-only ≈10.2 proposal yields zero
corrected Level-A heads and stops at attention. A dyad-anchored biological assay finds a real nucleosomal
signal near **10.298 nt**, sufficiently resolved to distinguish it from ≈10.63.

## 7. First missing bridges

1. **Cross-example/context transduction:** derive a state/direction on discovery genes and cause a
   controlled, dose-ordered effect in gene-disjoint targets under matched controls.
2. **Payload semantics/site-choice steering:** identify a causal local direction first, then validate its
   biological label with independently seeded draws and dyadic inference.
3. **Period-3 origin:** use training trajectories, controlled retraining, corpus/objective ablations, or
   cross-model evidence.
4. **≈10.63 downstream use:** qualify OV and behavior before speculating about origin.

No candidate advances merely because it is visible in a probe, spectrum, attention map, or attribution
graph. Attribution ranks; intervention decides.
