# Pre-registration — the causal signature of U_info (donor-graph 1e, core)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_donor_signature.py`. Definitions import from
`gfmcd.splicing.circuit`. Builds on **C16–C18**: the donor coupling is carried by a low-rank
object `U_info` (C16) that L3H1 delivers (C17) and L4/L5 transporter heads relay (C18). 1a–1c
mapped *where* the information flows; 1e asks *what it encodes* — the prompt's requirement that
an edge never be characterized by AG→CC corruption alone.

Because C16 established (C-Suff) that `U_info` carries the coupling, the **coupling's signature
is U_info's signature**: stratifying and re-corrupting the coupling characterizes the object.

## 0. Scope of the core, and what is deferred

**Testable now (annotation-free):** partner-specificity, donor-specificity, intron-length mod 3,
strand. **Cited, established:** period-3 dependence (C8: deleting the clock drives coupling
0.946→0.027). **Deferred to the period-3 bridge (need CDS annotation, built and validated there,
then applied back):** coding-vs-noncoding (the dataset is protein-coding-only, so coding context
is currently a constant) and **annotated splice phase 0/1/2**.

**Amendment (2026-07-15).** The CDS/splice-phase bridge (`gfmcd.splicing.phase`, validated:
90% of val internal exons are clean CDS, phases 0/1/2 well-distributed, annotated-phase ≠
intron-mod-3 at 32% agreement) was built ahead of the period-3 thread, so the two deferred
conditions are now computed here after all: **coding vs noncoding** (exon is / is not a clean
CDS — 90%/10% in val) and **annotated splice phase 0/1/2**. Both are reported **descriptively**
(coupling stratified by each, with CIs); they add no pass/fail gate. This completes the
seven-condition signature the prompt specified.

## 1. Target — imported unchanged

Readout / corruption / coupling / eligibility as C16 (val, both strands, gap [60,150],
unconditional set). Baseline `coup(x) = margin(clean) − margin(partner AG→CC)` per example
(`CouplingTask.coupling`); gene-grouped bootstrap CIs (`gboot`).

## 2. Conditions

- **Partner vs non-partner acceptor.** Baseline coupling (destroy `next_acceptor_pos` AG) vs
  **non-partner** coupling (destroy a distance-matched non-partner AG, `matched_decoy_acceptor`),
  on the examples that have such a decoy.
- **Donor vs non-site G.** Donor coupling vs the coupling measured at a **masked non-site G**
  (≥30 nt from both sites): does destroying the partner acceptor move a non-donor G's margin?
- **Intron length mod 3.** Stratify baseline coupling by `intron_length_down % 3 ∈ {0,1,2}`.
- **Strand.** Stratify baseline coupling by `strand ∈ {+,−}`.

## 3. FROZEN gates and their interpretation

| Test | PASS threshold | Interpretation / falsifier |
|---|---|---|
| **S-Partner** | partner-coupling CI-low **>** non-partner-coupling CI-high | partner-specific → **paired-intron consistency**. Falsified if non-partner ≈ partner → generic splice-site evidence. |
| **S-Donor** | donor-coupling CI-low **>** non-site-G-coupling CI-high | donor-specific → **donor-identity confidence**. Falsified if the non-site G moves as much → the corruption shifts G globally, not a donor readout. |
| **S-Strand** (sanity) | +-strand and −-strand coupling CIs **overlap** | strand-symmetric, as a real biological effect must be. A large asymmetry is a strand-handling artifact, not signal. |
| **Intron-mod-3** (descriptive) | report coupling per mod-3 class + CIs | strong monotone/So dependence on intron mod 3 would indicate **raw cross-site phase alignment**; a flat profile argues against it (leaving annotated-phase, built later, to test biological boundary phase). |

## 4. Firewall placement

Characterizes the *content* of the established `U_info` mediation — an interpretation of the
Behavior-level result, not a new level. It does not, by itself, reach Biology.

## 5. Post-hoc reflection (REQUIRED)

State per-gate pass/fail; assemble the signature into the most defensible reading of what
`U_info` encodes (generic vs paired-intron consistency vs donor-identity vs phase-sensitive);
and the intent check — does the annotation-free core already resolve U_info's identity, or is
the deferred phase/coding evidence load-bearing for the interpretation. Name what is deferred.
