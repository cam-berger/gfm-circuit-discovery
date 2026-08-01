# The nine-page paper this repository backs

> **Tier: T1 — canonical.** Current, authoritative. Read this.

This file answers one question: *if the record here were written up at roughly nine pages, what would be
in it, and which artifact backs each part?* It is a map, not a manuscript. It restates **no numbers** —
every figure lives in a claim row so that it cannot drift here. Cite the claim ID and let
[`CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md) carry the value.

Anything not listed below is either a bounded side-thread, a preserved negative, or provenance. It stays
in the repository because the record is append-only; it does not belong in the paper.

## Thesis

A masked-language model of pre-mRNA predicts a splice donor partly by reading **which** acceptor is its
intron mate. That behaviour is traced end to end: a predominantly *relative* positional address, carried
by a learned period-3 code read across the donor–acceptor span, is bound at `L3H1` to an
*acceptor-local* payload built by a serial `L0 → MLP0 → MLP1 → MLP2` path, written as a low-rank
donor-side object and transported to the readout.

Three self-imposed limits are as much of the contribution as the mechanism:

1. the payload's **semantic identity is unresolved**, and the obvious label was withdrawn (R22);
2. there is **no sparse knee** — the named components are a top-ranked *structured core inside an
   irreducibly distributed computation*, never a minimal circuit (C27, C33);
3. the delivered donor state transfers across genes as a **generic readout axis**, with its
   example-specific component causally inert (C41).

## Section map

| § | Pages | Content | Claims | Figure | Controlling artifacts |
|---|---|---|---|---|---|
| 1 | 1 | **Method contract.** Attribution ranks; intervention decides. The evidence ladder as a table; prereg → smoke → confirmatory → claim; memorials that can fail and sit on the measurement path; planted-positive and permuted-label calibration; held-out gene splits; gene-dyadic inference. Must cite [`PREREG_CONFORMANCE.md`](PREREG_CONFORMANCE.md) in the main text. | — | `fig_evidence_ladder` | [`METHODOLOGY.md`](METHODOLOGY.md), [`EVIDENCE_LADDER.md`](EVIDENCE_LADDER.md) |
| 2 | 1 | **The behaviour.** Corrupting a donor's paired acceptor moves the native masked-donor margin; matched non-partner and other-donor controls do not. Strand-symmetric; insensitive to intron length mod 3 and annotated phase. Regime: canonical short introns, intron-definition geometry. | C1, C2, C19 | `fig1_coupling` | `sealed_test.json`, `donor_signature.json` |
| 3 | 1 | **Necessity and anatomy.** `{L3H1, L2H15}` removes most of the coupling at retained masked-MLM health — the health control is what answers "you broke the model." The acceptor read is conjunctive; L3H1 reads at the acceptor and writes into the donor neighbourhood. | C3, C4, C5, C18 | `fig_full_circuit` | `sealed_test.json`, `position_profile.json`, `readout_path.json` |
| 4 | 1 | **The address.** L3H1's selection is driven by learned positional geometry, not partner identity — carry **R20** (the identity reading is retracted) explicitly. The address is predominantly relative offset with modest absolute modulation. | C22, C32, C25 | `fig2_mechanism` | `a1_token_position.json`, `a11_address_geometry.json` |
| 5 | 1 | **The payload.** Generation converges locally at the acceptor on a serial MLP-backed path; the resolved wiring is `L0H7/H12/H10 → MLP0 → MLP1 → MLP2`. The payload is *not* in the donor-side basis, so L3H1 converts representations. | C28, C29, C31, C24 | `fig_generator_graph` | `a7_source_localization.json`, `a8_generator_path.json`, `a10_generator_edges.json` |
| 6 | 1 | **Binding and delivery.** A non-forced reconstruct–inject factorial requires the attention address and the acceptor value payload *jointly*. L3H1 writes a stable low-rank donor-side direction that downstream heads transport; it replicates on disjoint genes. | C23, C16, C17, C26 | `fig3_circuit` | `a2_address_payload.json`, `donor_anchor.json`, `donor_forward.json`, `a5_uinfo_replication.json` |
| 7 | ½ | **What "circuit" means here.** Necessity and injected sufficiency hold while scrubbed-complement faithfulness collapses; the held-out size curve has no knee. State together with **R21** — the whole-cover memorials are embed-clamp-dominated and architecture-forced, and the genuine on-path anchor is `value_linearity`. | C27, C33 | *(missing — see below)* | `a6_circuit_faithfulness.json`, `a12_faithfulness_size.json` |
| 8 | 1¼ | **The carrier.** Period-3 deletion collapses coupling; single-head restoration beats a matched null; the locus is distributed across the span, not site-local; recovery runs through L3H1's attention pattern rather than carrier-dependent values; nucleotide-invariant local surgery gives the clean phase-aligned confirmation. Carry **R23** (`Δ_bind` is architecture-forced — reported, not interpreted) and **R24** (C34's confirm split missed its preregistered gate; C35's value-null is within-intervention only). Origin in three sentences: the 72-species vs human-only sibling contrast, the ncRNA absence, and the limit that this is comparative attribution, not training dynamics. Drop the sibling band-deletion arm (**R33**). | C8, C9, C34, C35, C36, C39, C40 | `fig5_p3_phase_surgery_confirmation`, `crossmodel_carrier_summary` | `carrier_surgery.json`, `carrier_mediation.json`, `p3_carrier_locus.json`, `p3_av_decomposition.json`, `p3_phase_surgery_confirmation_v1/`, `origin-siblings/`, `origin-crossmodel/` |
| 9 | 1¼ | **Limits, negatives, corrections.** The payload semantics are unresolved and the bulk-composition confound is refuted, not the naming. Cross-example transfer is a generic readout axis at verification tier. One paragraph absorbs the whole ≈10.63 second-carrier thread, including the powered refutation of the nucleosome reading — always naming the organism the comparator was measured in and never quoting a "±0.005" fit precision (**R37**). One clause each for the distal-intron null and the definition-switch null. A corrections box naming R20–R24 and pointing at [`CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md). | C30, C41, C11–C15, C37, IC1, ED1 | `fig_carrier_heatmap` | `a9b_payload_semantics_composition.json`, `g1_cross_example_transduction.json`, `postsoftmax.json`, `dyad_wavelength.json`, `ic1_distal_intron_necessity_v8.json`, `ed1_architecture_boundary_pairing.json` |

Two scope facts belong in §9 and are easy to lose: **only the C1 seal was ever spent** — everything from
A9 onward ran on validation — and **no external biological assay enters the study at any point**. No
RNA-seq, no measured junction usage, no PSI. Every result is model-internal.

## One sentence each, not a section

- The Head–Context Atlas independently rediscovered `L3H1` as the donor-reconstruction mediator on a
  disjoint cohort, and validates four general-use components in both donor and acceptor roles
  (HCA9, HCA19).
- Both positional carriers survive supervised GTEx fine-tuning and remain causally used for local
  donor/acceptor margins, though not for paired-site coupling (GFS23, GFS25).
- The released GTEx classifiers still show partner-specific coupling through their native head (GFS8),
  which pre-empts the "it is an artifact of the MLM readout" objection.
- A sparse-dictionary decomposition of the fine-tuning delta was attempted in five variants and did not
  resolve (GFS21, GFS26–GFS34). Say so plainly; it is a preserved negative, not a gap in the write-up.

## Out of the paper, kept in the repository

C6, C7, C10, C21, C38, the Head–Context Atlas competition branch (HCA1–HCA10) and its instrument rows
(HCA12–HCA17, HCA21–HCA22), and the GTEx arc beyond the four sentences above. Each is either superseded
by a stronger row, an instrument check, or a closed branch. The never-delete rule keeps them in the
ledgers; the paper does not carry them.

## What is missing before submission

These are known gaps, not oversights. They are recorded here so they are not rediscovered late.

1. **No figure plots the faithfulness size curve** (`a12_faithfulness_size.json`) — the no-knee result
   is the paper's most quotable finding and currently has no figure. §7 needs one.
2. **No figure plots the cross-example transfer decomposition** (`g1_cross_example_transduction.json`).
3. **`fig1_coupling.provenance.json` and `fig2_mechanism.provenance.json` record their sources under a
   different checkout with a null commit.** Both generators read only committed artifacts, so
   regenerating is cheap — do it before submission or two of three core figures ship with unverifiable
   provenance.
4. **Reproducing a claim by its documented command can overwrite the artifact being compared against.**
   56 runners default `--out` to a registered artifact path (R35). Always pass an explicit `--out` to a
   new path; see [`REPRODUCE.md`](../REPRODUCE.md) §2.2.
