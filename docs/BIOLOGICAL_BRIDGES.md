# Biological bridges and their evidence levels

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Three features of the SpliceBERT account invite a biological reading: the donor–acceptor circuit, the
period-3 positional carrier, and the separate ≈10.63 carrier. They are **not** at the same evidence
level, and the point of this document is to keep them apart.

- one bridge is **supported** and currently under-stated in the repository's prose (donor–acceptor ↔
  splice-site pairing);
- one is a **candidate origin that is unproven**, with the tempting over-reading already excluded
  (period-3 ↔ codon 3-periodicity);
- one is a **refuted** bridge whose true correlate is unknown, and that refutation is itself a result
  (≈10.63 ↔ nucleosomal/helical periodicity).

Presenting all three at their correct, different levels is the honest position, and a stronger one than
the field's habit of attaching a biological story to every legible feature. A genomic foundation model
can perfectly well contain a component with a clear biological bridge, a component whose origin is
merely plausible, and a component whose most attractive biological explanation has been killed by a
powered test.

Wording of every claim cited below is controlled by
[`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md);
correction history is in
[`studies/splicebert_partner_selection/CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md);
promotion rules are in [`EVIDENCE_LADDER.md`](EVIDENCE_LADDER.md). Statements labelled **Background
biology** are textbook-level context supplied to orient the reader. They are not findings of this
project, and nothing in this project's evidence depends on them.

**Checkpoints.** Bridges 1 and 2 concern **SpliceBERT.1024nt** (final, multi-species lineage). Bridge 3
concerns the Level-A attention circuit measured in **SpliceBERT.510nt** (stage-1, multi-species), whose
matched contrast is the human-only 510nt twin. Do not read results across the two as if they were one
measurement.

---

## Bridge 1 — L3H1 and splice-site pairing (intron definition)

**Evidence level reached: behaviour, plus a directed causal core inside the model. Supported.**

### What is established

A masked donor `GT` is scored differently when *its own* paired acceptor is corrupted: the native
masked-donor margin changes by **+0.872 [0.771, 0.969]** (C1), while matched non-partner AG and
other-donor controls move it by **+0.007** and **+0.005** (C2). The delivered signal is partner-specific
(~37×) and donor-specific (~74×), and is strand-symmetric (C19). The cohort is canonical internal short
introns of **60–150 nt** on a sealed chr8/9 split, n=400
([`studies/splicebert_partner_selection/FINDINGS.md`](../studies/splicebert_partner_selection/FINDINGS.md)).

The internal account is causal, not correlational. `{L3H1, L2H15}` ablation removes **81.6%** of the
coupling while retaining **99.2%** masked-MLM health (C3); L3H1's acceptor read is conjunctive, with
key+value **+0.663** and interaction **+0.531 [0.477, 0.583]** (C4); a non-forced reconstruct–inject
factorial shows the coupling contribution requires the positional address and the acceptor value payload
*jointly* (C23); L3H1 writes a low-rank donor-side object `U_info` whose rank-8 patching recovers about
**0.92** against matched random subspaces ≤**0.05** (C16), routed onward by L4H0/L4H2/L5H11 (C18); and an
independently derived `U_info` generalizes across disjoint genes (rank-1 overlap **0.970**, held-out
rank-8 sufficiency **+0.875**, C26).

So the model does not merely co-represent the two sites. It computes, at the donor, a quantity that
depends on which specific acceptor is its intron mate, and it does so through an identified, directed,
causally tested pathway.

### Why this is the intron-definition regime

**Background biology.** Vertebrate spliceosome assembly is generally described in two regimes. Under
*intron definition*, the 5′ splice site and the downstream 3′ splice site region are recognized as a
pair across the intron, so recognition of one site is coupled to recognition of its partner across the
intervening sequence. Under *exon definition*, the pairing runs across the exon instead, and this is
the regime usually associated with the long introns typical of mammalian genes. Intron definition is
associated with short introns. The individual sites are recognized by base-pairing and protein contacts
— the 5′ splice site by U1 snRNA complementarity, the 3′ end through branch point, polypyrimidine tract,
and AG recognition — and intron length is a known constraint on which regime operates.

The behaviour measured here is exactly the cross-intron dependency that intron definition names, in
exactly the intron-length band where intron definition is the expected regime. That correspondence is
what makes this the project's most defensible biological bridge: the model, trained only on sequence,
has learned an internal operation with the same *functional shape* as a documented recognition regime,
and the operation has been traced to specific components.

The repository currently under-states this. `MECHANISM.md` and `FINDINGS.md` present the behaviour as
"partner-specific nonlocal coupling", which is accurate but does not tell a biologist which phenomenon
it resembles. The licensed upgrade is to *name* the regime while keeping every existing boundary.

### What is not established

- **Not spliceosome biochemistry.** This is a model computation on a masked-language-model readout, not
  a claim about snRNP recruitment, transcript-level splice-site selection, or splicing in cells
  (`MECHANISM.md`, "Scope").
- **The determinants have not been dissected.** Coupling strength has *not* been regressed on 5′/3′
  splice-site complementarity, motif strength, or intron length. Absolute local MaxEnt strength as a
  payload label is explicitly withdrawn (C30 / R22): it reproduced on its own sample but was unstable
  across independently seeded draws and non-significant under two-way gene-dyadic inference. Any future
  determinant analysis must carry that same multi-draw and dyadic burden.
- **The intron-length dependence is untested.** The cohort is 60–150 nt and there is no matched
  long-intron arm, so the study cannot say whether coupling weakens where exon definition would be
  expected. This is the single most informative biological test available, and it has not been run.
- **The payload's semantic identity is open** (C30). The payload is acceptor-local and not explained by
  the preregistered bulk-composition covariates, but what local variable it encodes is unresolved.
- **Distributed, not sparse.** The identified components are a structured causal core within an
  irreducibly distributed computation: scrubbed faithfulness is `F(C)=0.005` (C27) and about **72/103**
  nodes are needed for 90% faithfulness with no knee (C33). "L3H1 is the pairing head" would be wrong.
- **Generality is untested interventionally.** Nearly all causal evidence is intra-example clean/corrupt
  patching; cross-example/cross-context transduction has not been demonstrated (`MECHANISM.md` §2).

**First missing bridge:** a matched intron-length series (and a determinant analysis under R22-grade
inference) that tests whether the coupling behaves like intron definition rather than merely resembling
it, together with cross-example transduction to show the state is transferable at all.

---

## Bridge 2 — the period-3 carrier and codon 3-periodicity

**Evidence level reached: behaviour, for the carrier's present-day *role*. Origin: corpus-attributed
(C39/C40, R25) — multi-species and codon-linked; training *dynamics* still deferred.**

Use this three-part formulation exactly, in this order:

> **Candidate origin = codon-composition periodicity in the coding corpus. Proven role =
> relative-positional routing. Origin unproven, pending training-time evidence.**

### Proven role: relative-positional routing

Deleting the rank-2 learned period-3 position band collapses donor–acceptor coupling **0.946 → 0.027**
(C8). The address L3H1 uses is **predominantly relative-offset based with a modest absolute-position
component** (C32). The carrier's contribution is **spatially distributed across the donor–acceptor
span**, not site-local: restoring it only at the donor/acceptor windows recovers about **2–4%** (C34).
Within carrier patching, recovery is mediated mainly by L3H1's **attention pattern** (**+0.284** on
confirm) rather than by carrier-dependent changes in its **values** (**−0.005** on confirm) (C35).
Nucleotide-invariant local surgery — moving only the learned P3 position component, changing no
nucleotide, codon, motif, MaxEnt score, coordinate, or length — confirms phase-aligned causal use in CDS
blocks: **Ψ_C = +0.141 [0.110, 0.172] nats**, n=943, with the learned direction exceeding all 19
same-geometry controls (paired excess **+0.136 [0.051, 0.222]**) (C36).

The licensed reading of the *role* is therefore positional: a distributed relative-positional routing
coordinate that the model reads across the span and uses to point attention.

Two boundaries travel with these numbers. C34's confirm split is **gate-qualified** — the preregistered
M1 gate failed at 5.5% > 5%, making it verification-level rather than a clean confirmatory replication.
C35's value-null holds **within that intervention** and does not exclude upstream carrier effects on
payload construction (R24).

### Candidate origin, and why it stays a candidate

**Background biology.** Protein-coding sequence exhibits a well-known 3-base periodicity in nucleotide
composition, arising from unequal base usage across the three codon positions. This period-3 spectral
peak is the basis of classical Fourier and spectral open-reading-frame finding. A model trained on
genomic sequence containing coding regions is therefore exposed to a strong period-3 compositional
regularity.

That makes codon-composition periodicity a **plausible candidate origin** for a learned period-3
positional code. It is not evidence of one. A frozen checkpoint can show what a feature is *used for
now*; it cannot show what *created* it.

### The over-reach is already excluded

The tempting upgrade — "the model has learned reading frame" — is ruled out by existing evidence. The
delivered donor-side signal is **phase-independent**: flat across both intron length mod 3 and
annotated splice phase (C19). Whatever the carrier's compositional origin, its present-day function in
this circuit is positional addressing, not reading-frame encoding.

### What must not be claimed

- **No coding enrichment.** C36's matched CDS-minus-non-CDS-exon contrast is **+0.006 [−0.080, +0.093]**,
  n=52 pairs — **inconclusive**, neither a null nor a positive. The repository correctly does not claim
  coding enrichment, and this document does not either. C36 establishes phase-aligned *use* of the
  learned P3 direction in CDS blocks; it does not establish coding specificity, reading-frame sensing,
  or training origin.
- **No origin claim.** The origin study (P3.4) was designed, red-teamed before any compute, and found
  **not identifiable on a frozen checkpoint**: frozen-model measurements speak to current use rather
  than training origin, the proposed frame contrast reduced to codon *composition* rather than reading
  frame, and the equivalence bar was unreachable at the available compute budget. It was deferred, not
  run
  ([`studies/splicebert_partner_selection/redteam/REDTEAM_SYNTHESIS_P3_4.md`](../studies/splicebert_partner_selection/redteam/REDTEAM_SYNTHESIS_P3_4.md);
  the design of record is
  [`PREREGISTRATION_P3_4_CARRIER_ORIGIN.md`](../studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_P3_4_CARRIER_ORIGIN.md),
  explicitly **not frozen**).

**First missing bridge:** training *dynamics* — checkpoint trajectories, controlled retraining, or a
carrier-penalized objective. The *corpus* origin is already attributed by the frozen-checkpoint sibling and
cross-model natural experiments (C39/C40, R25); what a single frozen checkpoint cannot give is the training
*trajectory*.

---

## Bridge 3 — the ≈10.63 carrier and nucleosomal/helical periodicity: refuted

**Evidence level reached: routing/attention for the carrier itself; a powered refutation of the obvious
biological bridge; the true correlate is unknown.**

### The result

The obvious bridge is excluded. The model's separate multi-species carrier sits at **10.630** and the
real dyad-anchored nucleosomal sequence signal sits at **10.298** — the two are **0.33 nt apart**, the
carrier is fitted to ±0.005, and the nucleosomal signal is strong (held-out signed **z = +249**, depth
4.87%), so this is a **powered refutation, not an underpowered null** (C14; figure
`figures/splicebert/fig4_carrier_circuit.CAPTION.md`). The separately proposed human-only ≈10.2
chromatin-reader circuit is closed one level earlier: **zero** corrected Level-A heads, so it does not
reach the attention gate. That refutes the proposed reader set, not every possible chromatin
representation (C13).

This is the correct way to state it, and the two wavelengths must never appear in prose without the
"0.33 nt apart, powered" qualifier in the same sentence, because the bare pair reads as the opposite of
the finding.

**Background biology.** Nucleosome-bound DNA shows a ~10.2–10.5 bp periodicity in dinucleotide
preference, reflecting the helical repeat of DNA as it bends around the histone octamer. This is the
periodicity a reader would expect a genomic model's ~10-nt feature to correspond to. That expectation is
what was tested, and it failed.

### What the carrier is, and is not

The ≈10.63 carrier is real and it is read: eight heads route it through the softmax under full controls
(C11). Its downstream arms have now been *measured*, and both returned boundaries rather than positives:
its readers are sub-SESOI on the donor coupling margin, and deleting it moves coupling only −0.033
[−0.061, −0.005] nats (C37→C12). Its **general** OV content and its behavioural function remain
**unresolved** and are not claimed; its biological origin is corpus-attributed (C39/C40, R25). Whether it
writes onto the donor readout `U_info` was never computed and is **open** (**R31**). Local splice-junction phasing cannot arbitrate the question: a
~51-nt junction window cannot resolve 10.298 from 10.630, because the two filters are nearly collinear
over that support — a **resolution** limit, not a sample-size limit, so more data would not help
(C15;
[`studies/splicebert_partner_selection/METHODS.md`](../studies/splicebert_partner_selection/METHODS.md)).

### Publication statement

> The obvious bridge is excluded. The true correlate is unknown. No biological claim is made.

### The open question is a hook, not a hole

What kind of ≈10.63-nt periodicity is **neither** codon **nor** nucleosomal, and why would eight
attention heads read it? That question is well-posed precisely because the two leading hypotheses were
tested and one was killed with a calibrated assay. It is future work, not a gap in the account: the
account already says, at the right level, that a real positional attention circuit exists whose purpose
is unknown.

**First missing bridge:** push the carrier past attention — OV content and a qualified behavioural test
— before any further origin speculation. Origin work is downstream of the computational-use test, not a
substitute for it.

---

## Summary

| Bridge | Evidence level reached | Licensed statement | First missing bridge |
|---|---|---|---|
| **1. L3H1 / donor–acceptor ↔ splice-site pairing, intron definition** | Behaviour + directed causal core (C1–C5, C16–C18, C23, C26; bounded by C27/C33) | A masked donor's score depends causally on its *specific* paired acceptor across a 60–150 nt intron, implemented by an identified structured causal core within an irreducibly distributed computation. This is cross-intron communication of the kind intron definition describes; it is a model computation, not a spliceosome claim. | A matched intron-length series and a determinant analysis (5′/3′SS complementarity, motif strength) under R22-grade multi-draw + dyadic inference; plus cross-example transduction. |
| **2. Period-3 carrier ↔ codon 3-periodicity** | Role: behaviour (C8, C32, C34, C35, C36). Origin: corpus-attributed (C39/C40, R25). | Corpus origin attributed = codon-linked, multi-species (absent in the human-only sibling and in ORF-free ncRNA models, C39/C40); proven role = relative-positional routing. Phase-independence of the delivered signal (C19) already excludes a reading-frame interpretation. No coding enrichment is claimed — C36's matched contrast was **+0.006 [−0.080, +0.093]**, inconclusive. | Training *dynamics* only: checkpoint trajectories, controlled retraining, or a carrier-penalized objective. The single-frozen-checkpoint P3.4 ΔL approach is not identifiable (deferred); the *corpus* origin is settled by the sibling/cross-model natural experiments (C39/C40). |
| **3. ≈10.63 carrier ↔ nucleosome / helical periodicity** | Carrier: routing/attention (C11). Bridge: powered refutation (C14); the human-only ≈10.2 variant closed at attention (C13). Function: open (C12; C37 donor-scoped OV null). Corpus origin: attributed (C39/C40, R25). | The obvious bridge is excluded — the real nucleosomal signal at 10.298 and the model carrier at 10.630 are 0.33 nt apart under a powered assay. The true biological correlate is unknown. No biological claim is made. | General OV content and a qualified behavioural test for the carrier remain open; the *corpus* origin is already attributed (C39/C40), a biological correlate is not. Splice-window phasing cannot arbitrate (resolution, not power — C15). |
