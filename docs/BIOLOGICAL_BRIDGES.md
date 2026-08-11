# Biological bridges and their evidence levels

> **Tier: T1 — canonical.** Current, authoritative. Read this.

This document separates model mechanism from biological interpretation. Exact model claims are
controlled by [`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md),
and corrections by
[`studies/splicebert_partner_selection/CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md).
Background biology supplies hypotheses and analogies; it is not evidence about what SpliceBERT
implements.

**Checkpoints.** The donor mechanism and period-3 carrier are measured on `SpliceBERT.1024nt`. The
separate ≈10.63 carrier is established most directly in `SpliceBERT.510nt` by comparison with its
human-only stage-1 sibling. Results from these checkpoints are not interchangeable.

---

## Bridge 1 — donor–acceptor coupling and intron definition

**Evidence level reached: model behavior plus a directed causal core. Biological status: coherent
functional analogy only.**

### What the model evidence establishes

For canonical internal 60–150 nt introns, corrupting the paired downstream acceptor changes the masked
donor margin by **+0.872 [0.771, 0.969]** (C1), while matched non-partner AG and other-donor controls
change it by **+0.007/+0.005** (C2). C3–C5, C16–C18, C23, C26, and C31 support a directed causal account:
a predominantly positional address and acceptor-local payload meet at L3H1, which writes a low-rank
donor-side state that downstream components deliver to the MLM readout. C27 and C33 show that this is a
structured core inside a distributed computation, not a complete sparse circuit.

Cross-example interventions now refine the generality claim. C38 shows strong, dose-monotone,
position-specific transfer of a matched cross-gene acceptor-side payload, but content specificity is
unresolved because a matched scramble retains substantial recovery. C41 shows that the C16/C26
**final donor residual** transfers as a gene-generic readout axis; its example-specific orthogonal
component is inert for the donor margin in the tested additive linear regime. C41 is a strong
**verification-tier** result, not a confirmatory result, because required frozen gates were changed or
absent (R28/R32).

### Why intron definition is an analogy, not a result

Intron definition describes coordinated recognition across an intron, whereas exon definition
describes coordination across an exon. SpliceBERT's short-intron, cross-intron dependence has the same
geometric and functional shape as intron definition. That makes intron definition a biologically
coherent interpretation to test; it does **not** show that the model implements the biological regime.

ED1 was the preregistered architecture-boundary assay intended to discriminate that stronger reading.
It failed its frozen instrument gate: the planted-confound false-positive rate was **28.5%** against a
predeclared **≤10%** target. The artifact therefore records **`GATES_FAILED -- no interpretation`**.
The primary cross-exon switch estimand was **+0.014 [−0.013,+0.040]** in raw logit units, below the
**0.106** SESOI and with an interval spanning zero. A post-run sensitivity bound is useful for designing
a replacement assay, but it cannot relabel the frozen failed-gate result.

The licensed biological sentence is:

> SpliceBERT computes partner-specific cross-intron communication whose functional geometry resembles
> intron definition; ED1 did not pass the instrument gates required to claim architecture-conditioned
> intron/exon-definition behavior.

The study does not establish spliceosome biochemistry, splicing in cells, long-intron generality, or
the semantic identity of the acceptor-local payload. A redesigned ED1 assay with a valid
magnitude-sensitive confound gate, followed by independent biological validation, is the first missing
bridge.

---

## Bridge 2 — period-3 carrier

Only two claims are licensed:

1. **Current role:** the learned rank-2 period-3 direction is a distributed relative-positional routing
   coordinate that causally supports donor–acceptor coupling.
2. **Comparative origin:** the carrier is associated with the multi-species pretraining corpus; its
   training dynamics and biological semantics are unresolved.

Deleting the direction changes coupling **0.946 → 0.027** (C8). C32, C34, and C35 localize its use to
mostly relative-offset geometry distributed across the donor–acceptor span and mediated primarily by
L3H1's attention pattern within the tested intervention. Nucleotide-invariant local surgery confirms
phase-aligned causal use in CDS blocks: **Ψ_C = +0.141 [0.110, 0.172] raw logit units**, n=943, and the
learned direction exceeds all 19 same-geometry controls (C36).

The comparative checkpoint evidence attributes the carrier to multi-species pretraining (C39/C40): it
is strong in the 72-species stage-1 sibling, absent below a powered detection floor in the matched
human-only sibling, and absent from the tested ORF-free ncRNA transformer MLMs. This comparison does
**not** isolate species diversity from corpus size, composition, sampling, or optimization history, so
it is comparative attribution rather than a controlled causal corpus ablation.

Coding-specific and reading-frame interpretations are not supported. C36's matched
CDS-minus-non-CDS-exon contrast is **+0.006 [−0.080,+0.093]**, n=52 pairs, and C19 finds no dependence of
the delivered signal on intron length modulo 3 or annotated splice phase. Codon periodicity is a
plausible corpus correlate, but the evidence does not establish coding enrichment, reading-frame
semantics, or why training constructed the coordinate. Those questions require checkpoint trajectories
or controlled corpus/objective ablations.

---

## Bridge 3 — the ≈10.63 carrier and nucleosomal periodicity

**Evidence level reached: controlled post-softmax attention for the carrier; powered refutation of the
proposed nucleosomal bridge.**

The ≈10.63 carrier is real and read by eight heads (C11), but its general biological function is not
known. Its donor-circuit value-write and behavior arms returned boundaries rather than positives
(C12/C37), and the preregistered `U_info` write endpoint was not evaluable as scoped (R31/R36).

The attractive nucleosomal interpretation is excluded. A real dyad-anchored nucleosomal signal occurs
near **10.298 nt**, whereas the model carrier is **10.630 nt**: the **0.33 nt** gap is roughly seven times
the largest estimated uncertainty on either side (C14). The human-only ≈10.2 reader proposal also fails
at the controlled attention gate (C13). Local splice-window phasing cannot resolve the two wavelengths
over its short support, which is a resolution limit rather than a biological null (C15).

The licensed statement is:

> The ≈10.63 positional attention carrier is not the measured ≈10.298 nucleosomal periodicity. Its
> biological correlate and general downstream function remain unknown.

---

## Bridge 4 — 3′-boundary organisation and U2AF1-centred site recognition

**Evidence level reached: none. This bridge is a preregistered *motivation*, not a result.**

It is recorded here before it is tested, because that is the point at which a bridge is most likely to
be promoted by accident, and because a protein name next to a model component is this project's most
promotable construction.

U2AF1-centred 3′-splice-site recognition has experimentally resolved sequence organisation around the
invariant `AG`: U2AF1 variants alter 3′-site preferences in patients, cells, and biochemical assays
([Ilagan 2015](https://doi.org/10.1101/gr.181016.114)), and structural and affinity measurements resolve
the −3 and +1 contacts ([Yoshida 2020](https://doi.org/10.1038/s41467-020-18559-6)). BG1 borrows **only
the positions** from that literature — biological −3 and +1 — and freezes T-over-A and G-over-A contrasts
on a synthetic panel that holds the acceptor `AG` and every positioned `AG`/`GT`/`GC` fixed. BG2 asks
whether the same edited-window value changes travel the already-established L3H1 route.

Nothing about that design can identify a protein. The panel is synthetic, the readout is a masked-donor
logit margin, and the frozen model has no notion of binding, complex assembly, or catalysis. What a
passing BG1/BG2 would license is a statement about **where in the sequence** the model's acceptor-local
payload is sensitive, and **whether that sensitivity uses the known route** — a functional and positional
parallel, at the model level, to the organisation the U2AF1 literature describes.

The wording licensed **if and only if** BG1 and BG2 both pass and both clear an independent audit is
frozen verbatim in the protocol (§7). Until then the licensed statement is the empty one:

> No result in this repository relates SpliceBERT to U2AF1. A frozen protocol exists that would test a
> positional parallel; it has not been run.

See `docs/NOMENCLATURE.md` standing phrase 3 for the denial that must accompany every use of the protein
name, and the `U2AF1` vocabulary row for the constructions that are forbidden outright.

---

## Summary

| Bridge | Licensed statement | Boundary |
|---|---|---|
| Donor–acceptor coupling ↔ intron definition | Partner-specific cross-intron communication has the functional geometry of intron definition. | ED1 failed its planted-confound gate; no intron/exon-definition implementation claim is licensed. |
| Period-3 carrier | Current role is distributed relative-positional routing; comparative origin is associated with multi-species pretraining. | Coding specificity, reading-frame semantics, training dynamics, and a clean species-diversity causal contrast are not established. |
| ≈10.63 carrier ↔ nucleosome | The carrier reaches attention, but the measured ≈10.298 nucleosomal signal is distinct. | General function and biological correlate remain open. |
| 3′-boundary organisation ↔ U2AF1 recognition | **None — nothing is licensed.** BG1/BG2 are frozen and not run. | A positional parallel motivated by published biology; it can never become a protein identification, and it is not a result at all until BG1/BG2 run and pass an independent audit. |
