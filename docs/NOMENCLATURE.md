# Nomenclature: the project's controlled vocabulary

Several of this project's terms have been retired, withdrawn, or bounded, and several near-identical
numbers refer to entirely different objects at entirely different evidence levels. This file is the
single controlled vocabulary. It **records** decisions taken elsewhere; it does not make new ones.
Where this file and a claim ledger differ, the ledger wins.

Authoritative detail lives in
[`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md),
[`studies/splicebert_partner_selection/CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md),
[`studies/splicebert_acceptor_coupling/CLAIMS.md`](../studies/splicebert_acceptor_coupling/CLAIMS.md),
[`studies/splicebert_acceptor_coupling/CORRECTIONS.md`](../studies/splicebert_acceptor_coupling/CORRECTIONS.md),
and [`results/phase-lattice/RESULTS_INDEX.md`](../results/phase-lattice/RESULTS_INDEX.md).

**Statuses.** `LIVE` — usable in current prose, under the stated conditions. `RETIRED` — the term was
never a measurement of what it appeared to name; do not use it in live prose. `WITHDRAWN` — the claim
the term encoded was retracted by an R-number. `CLOSED` — the hypothesis was tested and failed at a
named gate; the term stays, but only as a closed result.

---

## Reader-facing evidence identifiers

New evidence records use a descriptive, three-part identifier:

```text
<study>/<topic>/<plain-language-slug>
```

For example, `acceptor-coupling/components/l1h12-mean-ablation` says which study owns the record,
which part of the evidence ladder it concerns, and what was measured. Each segment is lowercase
kebab-case. The identifier names one immutable evidence record; a successor experiment gets a new
identifier rather than silently changing the old record's scope.

Short codes such as `AC5`, `C37`, `R3c`, and `M4d` remain where frozen protocols, artifacts, manifests,
and historical prose already use them. They are **legacy aliases**, not the preferred reader-facing
names. Run labels (`R3c v2.1`), claim aliases (`AC5`), and correction aliases (`ACR16`) name different
objects and must not be substituted for one another. A current ledger should lead with the descriptive
identifier or title and carry the short code only as a compatibility alias. This additive convention
preserves immutable citations while stopping new public vocabulary from depending on local numbering.
The acceptor study's current mappings are maintained in its
[`CLAIMS.md`](../studies/splicebert_acceptor_coupling/CLAIMS.md#reader-facing-evidence-catalog).

---

## The vocabulary

| Term | Status | Sentence that must accompany it | Authoritative detail |
|---|---|---|---|
| **≈10.63 carrier** (multi-species) | LIVE — model-internal, attention level; comparative corpus association | The multi-species ≈10.63-nt positional carrier is read through post-softmax attention by eight heads; its general OV content and behavioral function remain open (C12; C37 is a donor-scoped OV null) and no biological correlate is established. C39/C40 associate it with the tested multi-species training lineage without isolating species diversity, corpus size, composition, sampling, optimization, or causal origin. | C11, C12, C37, C39, C40, R25; `results/phase-lattice/RESULTS_INDEX.md` |
| **10.298** (nucleosomal) | LIVE — biological, sequence-side | ≈10.298 nt is a real dyad-anchored nucleosomal periodicity measured **in DNA, not in the model**, and specifically **in mouse ES-cell chromatin** (Voong 2016 chemical-cleavage dyad map, mm9). It sits 0.33 nt from the model's ≈10.630 carrier — a gap **~7× the largest uncertainty on either side** (carrier two-grid sub-bin spread 0.045 nt; nucleosomal estimator spread 0.045 nt) — so the nucleosomal explanation of the carrier is a **powered refutation**, not an open possibility. Do **not** quote a "±0.005" carrier fit precision: that figure was a bin-unit synthetic-control error misread as a measurement uncertainty (R37). The refutation does **not** depend on the mouse point estimate transferring to human at better than 0.33 nt: 10.630 lies outside the 10.2–10.5 bp DNA-helical range recorded from the literature in the row below. **No human dyad-anchored measurement exists in this record** — always name the organism. | C14, C15, R17, R30, R37; `studies/splicebert_partner_selection/CLAIMS.md`; `data/NUCLEOSOME_MAP.md` |
| **"10.67"** (also 10.6667, 10.6875) | RETIRED | "10.67" was the label of a discrete Fourier bin (512/48 on the stage-1 grid, 1026/96 = 10.6875 on the final grid), not a wavelength measurement; the grid labels disagree while the sub-bin measurement converges on ≈10.630. Use **≈10.63 carrier**. | `results/phase-lattice/RESULTS_INDEX.md` retired list; R16/R37 in `studies/splicebert_partner_selection/CORRECTIONS.md` |
| **"10.58 ± 0.35"** | RETIRED | "10.58 ± 0.35" is not a measured splice-junction constant and is not an interpretation of the carrier. Omit it; use ≈10.63 carrier. | `results/phase-lattice/RESULTS_INDEX.md` retired list |
| **Human-only ≈10.2 component** | CLOSED at the attention gate | The proposed human-only ≈10.2 chromatin-reader circuit has zero corrected Level-A heads and is closed at the attention gate; that refutes the proposed reader set, not every possible chromatin representation. | C13 |
| **10.2–10.5 bp helical periodicity** | LIVE — distinct referent, disambiguate; **the range itself is under an owner question** | "≈10.2–10.5 bp" is the generic DNA-helical periodicity range from the literature and is **not** the human-only ≈10.2 candidate component of SpliceBERT's position table; never let the two share an unqualified "10.2". **Open (B4):** Rhodes and Klug (*Nature* 292:378–380, 1981) report 10.6 ± 0.1 bp for random-sequence DNA, which would place 10.630 *inside* the range and remove the fallback leg of R30 — the powered ≈10.298 refutation is unaffected. Until B4 is ruled on, do not lean on "10.630 lies outside the DNA-helical range" as if it were settled. | `docs/REVIEW_2026-07-22.md` §7; `docs/REVIEW_OPEN_QUESTIONS.md` B4 |
| **"splice-site strength"** as the payload's semantic identity | WITHDRAWN (R22) | The payload is acceptor-local and is not explained by the preregistered bulk-composition covariates, but its **semantic identity remains unresolved**; the "splice-site strength" naming was withdrawn under R22. Say "an acceptor-local payload". | R22; C30 (bounded) |
| **splice-site strength** as an external MaxEnt covariate | LIVE — a different use, keep | MaxEntScan splice-site strength remains a legitimate external sequence covariate for matching and stratification; R22 withdrew it as the payload's semantic identity, not as a measurable sequence property. | `DECISIONS_A0.md` D-A0-3; `PREREGISTRATION_A0` |
| **circuit** | LIVE — always qualified at first use | A **structured causal core within distributed computation under the tested interventions**: causally necessary and sufficient when clean outputs are injected, but not minimal — the static-ranking curve crosses 90% at 64.2 nodes by interpolation, first exceeds it at the sampled 72/103 point, and has no knee. | `docs/METHODOLOGY.md`; C27, C33 |
| **frozen handoff schema mismatch** (masked-acceptor program) | CLOSED — terminal instrument failure, no target result | The validation instrument passed, but its authentic six-margin rows fail the frozen consumer's exact three-field contract before any target behavior or mechanism stage. Say **instrument unresolved before target science**; never call it a behavior null, failed mechanism, or pending repair. | `acceptor-coupling/instrument/frozen-handoff-schema-mismatch`; ACR19 |
| **natural-`GC` modifier** | LIVE — bounded model-internal verification | Natural `GC-AG` donor class is associated with weaker masked-acceptor coupling and lower `L3H1` donor attention than matched `GT-AG` on one validation reserve. The comparison is quasi-observational and does not establish intrinsic class specificity, a causal component, write, delivery, splicing efficiency, or biology. | `acceptor-coupling/natural-gc/behavior-modifier`; `acceptor-coupling/natural-gc/l3h1-attention-modifier`; ACR20–ACR21 |
| **natural-`GC` QK score diagnostic** | DEVELOPMENT ONLY — no claim alias | The archived exact donor-versus-other pre-softmax score transform refines adaptive candidates at nominal, unadjusted resolution. Different arm-conditioned batch-mean interventions and incomplete inference/provenance block class-specific head, query-versus-key, write, route, mediation, or mechanism language. | `acceptor-coupling/natural-gc/qk-score-diagnostic`; ACR22–ACR23 |
| **period-3 carrier** | LIVE — role established; comparative corpus association, training dynamics open | The period-3 carrier contributes across the donor–acceptor span through L3H1's attention pattern. C39/C40 associate its presence with the tested multi-species training lineage and its absence with the tested human-only sibling and ORF-free ncRNA models; they do not isolate causal origin or coding specificity. | C8, C34, C35, C36, R24, C39, C40, R25 |
| **coding-boundary phase** (also "annotated phase", `p`, the `0/1/2` of C43/C44) | LIVE — disambiguate from the period-3 carrier at first use | **Coding-boundary phase is a property of the annotation, not of the model's position table.** It is the codon reading-frame phase (0/1/2) of a coding exon boundary in GENCODE. It is **not** the period-3 carrier's phase, which is a positional routing coordinate learned by the model — the two are different objects at different evidence levels, and C44 is precisely the test of whether the carrier implements a coding-boundary-phase preference. That test returned `INSTRUMENT_UNRESOLVED`, so **no sentence may present the period-3 carrier as the mechanism of C43**, and equally none may say period-3 *fails* to implement it. Never write an unqualified "phase" in prose that also discusses the carrier. The successor reached only a model-free capacity stop (`PBC1`), so its carrier test did not run and the licensed wording is unchanged. | C43, C44, PBC1, R45; `studies/splicebert_exon_edge_compatibility/README.md`; `studies/splicebert_positional_boundary_coherence/README.md` |
| **"authentic-module" context** (C43's lattice) | RETIRED — say **chimeric** | The C43 constructs join a donor module from one gene to an acceptor module from another; each module is authentic hg38 sequence but the joined construct exists in no genome. The source study wrote "synthetic-module" while the result was a development candidate and "authentic-module" once it became supported; that substitution was corrected on consolidation (R45) and must not be reintroduced from Git history. The retired phrase nonetheless survives verbatim in the carried frozen protocol and in the `scope` fields of the carried artifacts, which are immutable and correctly untouched — read those occurrences as "chimeric". Relatedly, "120-nt" names the **donor→acceptor span**, not the length of either module. | C43, R45 |
| **U2AF1** | PRE-REGISTERED MOTIVATION ONLY — never a component label | U2AF1 appears in this project **only as the published biology that motivated a frozen positional contrast** (−3 T-over-A, +1 G-over-A). It names no model component, no head, no direction, and no payload. Forbidden constructions, in live prose and in figures: "SpliceBERT's U2AF1 head/feature/module", "the model implements U2AF1", "U2AF1-like representation", "the model recognises the 3′ site". The only licensed form is a **functional and positional parallel**, stated with its denial in the same sentence — see standing phrase 3. | `preregistrations/PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md` §1; `docs/BIOLOGICAL_BRIDGES.md` Bridge 4 |
| **3′-boundary grammar** (also "boundary grammar", the `BG1`/`BG2` prefix) | DESIGN LABEL — **instrument-unresolved**; BG1/BG2 targets not run | "Boundary grammar" is the **name of a preregistered contrast**, not of a finding. It denotes the planned response to single-base substitutions at biological −3 and +1 on a synthetic panel that preserves the acceptor `AG` and every positioned `AG`/`GT`/`GC`. The development preflight failed four frozen capability/resolution families, so no sentence may say SpliceBERT *has* a boundary grammar, *reads* one, or that the acceptor-local payload *is* one. "Grammar" carries no linguistic or spliceosomal content. | `preregistrations/PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md`; `docs/ROADMAP.md` Track 2 |
| **checkpoint attribution** | LIVE — mandatory disclosure | The donor circuit and the period-3 carrier are established on **SpliceBERT.1024nt**; the ≈10.63 carrier's attention circuit is measured on the stage-1 multi-species **SpliceBERT.510nt** against its human-only twin — state which checkpoint a result comes from, and never present the two as one measurement. | `MECHANISM.md` "Checkpoints"; `docs/NOMENCLATURE.md` |
| **carrier register** (`r`) | LIVE — a third object, and **three different operators wear the name** | **Register is an intervention coordinate, not an observation** — not the annotated coding-boundary phase `p` (a GENCODE property) and not the carrier's own learned phase (a position-table property). Name the operator, because they are not interchangeable and one of them failed: (i) **C36's local hidden-axis shift** — the learned direction moved by 3/6/9 nt within a block, nucleotide-invariant; (ii) **C44's local carrier surgery**, whose frozen 3×3×3×3 package tested relocation of C43's ridge under `δ + r = 0` and returned `INSTRUMENT_UNRESOLVED` — that instrument is attempted-and-unresolved and **may not be reopened or relaxed**; (iii) the successor study's **span-wide analytic ramp** `φ_r(t) = −(2πr/3)(t−d)/(a−d−1)`, predicting the ridge `δ = (p_A−p_D) mod 3 = −r mod 3`, which is a *different, wider* operator that has never been run. Same relation, different intervention: a successor result would not be a replication of (ii), and (ii)'s failure is not evidence about (iii). In all three the sign is fixed before data and is **never relabelled from an observed ridge**. | C36, C44, R45; `studies/splicebert_positional_boundary_coherence/PREREGISTRATION.md` §9.2–9.3 |
| **unqualified "phase"** in prose spanning annotation and carrier | FORBIDDEN in outcome prose | Three distinct objects now share the word: annotated coding-boundary phase (`p`), the period-3 carrier's learned phase, and the register `r`. Any sentence that could be read against more than one must name which. This extends the C43/C44 rule above to the successor study, which freezes the prohibition in its own protocol. | C43, C44, R45; `studies/splicebert_positional_boundary_coherence/PREREGISTRATION.md` §1.3 |
| **`P3_*` outcome-tier labels** (`P3_DIRECTIONAL_SCOPED_SUPPORT`, `P3_NECESSITY_SUPPORTED`, `P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED`, `P3_REGISTER_PARTIAL_HANDEDNESS_FALSIFIED` / `_AMBIGUOUS`, `OPTIONAL_INSTRUMENT_UNRESOLVED`) | RESERVED — defined, unearned | These name **rungs of a frozen decision hierarchy that no run has reached**. They are not statuses any current result holds and must not appear in live prose, a figure, or the paper as if they described an outcome. Each is separately defined, and they are deliberately not interchangeable: directional-scoped support explicitly withholds materiality, and partial concordance explicitly withholds complete cyclic relocation and all route-mediation language. Quote a tier only from the study's decision-hierarchy table, never from memory. | `studies/splicebert_positional_boundary_coherence/README.md` §"Decision hierarchy" |
| **sequence-encoded coding-boundary coherence prior** | RESERVED — the ceiling, not a finding | The **maximum** biological interpretation the successor study's design can license, even on a complete positive. It is a statement about a masked-language model's prior over joined boundaries. It never becomes "SpliceBERT reads frame", "the spliceosome checks phase", "the model implements translation", or a claim about U2AF or any other factor. Today it labels nothing. | `studies/splicebert_positional_boundary_coherence/PREREGISTRATION.md` §1.3 |
| **"donor" in a DNABERT-2 / `DBP*` context** | AMBIGUOUS — always disambiguate | The word means **two unrelated things** in this repository. Everywhere in the SpliceBERT record it is a **splice donor site** (the 5′ exon–intron boundary). In the DNABERT-2 promoter study it names the **source example whose activations are transplanted** into a corrupt run — a patching term with no splicing content, in phrases like "donor matrix", "same-class donor", "mismatched-donor control", `negative_class_donors`. Prefer **"activation source"** / **"source example"** in new prose; where committed artifact names and immutable result fields already say "donor", read them in the patching sense. A sentence containing "donor" and a DNABERT-2 result must make the sense explicit. | `studies/dnabert_promoter/DECISIONS.md` D11; `studies/dnabert_promoter/CLAIMS.md` DBP8 |
| **"compact" / "the circuit" / "sparse" for the `DBP*` study** | RETIRED — say **structured causal core within a distributed, MLP-dominated computation** | The DNABERT-2 arc's prose called `L0 MLP + L0H8` a "compact confirmed component set". Its own held-out faithfulness curve refutes that: the five-node set reaches standalone faithfulness `0.194`, there is no knee, and `1.000` arrives only at size 39 once all three early MLPs are included. Conditional delivery (`0.882`) had been read as if it were faithfulness. | DBP7, DBPR4 |
| **"transferable" / "sequence-specific delivery" for the `DBP*` transplant** | WITHDRAWN — say **class-level transfer, no example specificity** | Matched-source restoration does not beat unrelated *same-class* restoration (`0.036`, p `0.252`; `0.053`, p `0.091`) — the contrast that would show example specificity is null on both components. Only the class contrast is large. A quoted "Holm-adjusted p = 2.8539e-22" for the class contrast appears in **no artifact** and must not be requoted. | DBP8, DBPR5 |
| **"the Layer-2 endpoint" as a localization result** | BOUNDED — say **strongest tested endpoint, `0.0019` single-seed margin** | Validation balanced accuracy is `0.8050` / `0.8031` / `0.8014` / `0.8014` at DNABERT-2 encoder layers 2 / 3 / 1 / 4. The data do not separate the early layers, so Layer 2 is a working choice. Relatedly, an unqualified DNABERT-2 **layer index** is ambiguous in the results tree: `causal/layer3_*` is a superseded replicate at encoder Layer 3 against a different probe. | DBP4, DBPR2, DBPR7 |

---

## Standing phrases

Three phrases carry a caveat that a skimming reader would otherwise miss. Use them verbatim.

1. **At first use of "circuit" in any top-level document:** *structured causal core within distributed
   computation under the tested interventions*. The named circuit is not established as minimal or
   autonomous.
   This now holds in **both** models: the SpliceBERT donor arc reached it at C33 (no knee in the held-out
   size curve) and the DNABERT-2 promoter study reached it independently at DBP7 (five-node set at
   standalone faithfulness `0.194`, 1.0 only at size 39). Neither licenses the other — the agreement is
   in the *shape of the limit*, not in any shared number.
2. **Whenever ≈10.63 and ≈10.298 appear together:** the "0.33 nt apart / powered refutation" qualifier
   must be in the **same sentence**, and **≈10.298 must be named as the mouse ES-cell measurement it is**
   (Voong 2016, mm9). A reader must never be able to skim the two numbers and conclude that the model
   found nucleosome positioning, and must never infer that a human dyad map exists in this record — none
   does (R30). Express the power as the gap **relative to the measured spreads** (0.045 nt on either
   side, so ~7×); **do not quote a "±0.005" fit precision** — that figure was a bin-unit
   synthetic-control error misread as a measurement uncertainty (R37). The refutation does not rest on
   cross-species transfer at 0.33 nt precision: 10.630 lies outside the 10.2–10.5 bp DNA-helical range
   recorded from the literature in the entry below.
3. **Whenever U2AF1 is named:** the denial must sit in the **same sentence** as the parallel — *"a
   functional and positional parallel to U2AF1-centred 3′-site recognition; it does not imply that
   SpliceBERT implements U2AF1, binds RNA, or performs spliceosomal biochemistry."* A protein name
   beside a model component is the single most promotable construction in this project: a reader who
   skims "−3/+1" and "U2AF1" in adjacent clauses will remember a protein in the network. The parallel is
   a **motivation for a positional contrast**, never an identification. The BG1 development instrument
   failed, so no result is licensed and BG2 remains sealed.

## Units convention

The donor–acceptor **coupling** and the patching/restoration effects on it (`C`, `ΔC`, `Ψ_C`, the +0.87
sealed effect, the `R_xfer`/`R_C` recoveries) are **raw MLM logit-differences** — the native masked-donor
margin `logit(G) − max(A,C,T)` — i.e. **logit units**, not nats. The label **"nats"** is correct only for
(a) **masked-base cross-entropy damage** (e.g. exon-vs-intron masked-base damage +0.170 vs +0.032 nats), a
genuine per-base NLL, and (b) verbatim quotation of a **frozen preregistered threshold**. Where a ledger row
or frozen record labels a coupling/margin quantity "nats", read it as the logit-difference it measures; those
records are not rewritten (immutability), but **live prose should say "logit units" for coupling and reserve
"nats" for cross-entropy damage**.

## Retirement records are not to be scrubbed

Retired and withdrawn terms **must remain** where they are recorded as retired: in
`studies/splicebert_partner_selection/CORRECTIONS.md`, in the "Retired interpretations" section of
`results/phase-lattice/RESULTS_INDEX.md`, and in any preregistration or decision log that documents the
retraction. Removing a retired term from its own retirement record would destroy the correction
history, which is part of the evidence. This file governs **live prose only**: prose that asserts
something in the present tense.

## Open nomenclature questions

Items that cannot be resolved without an owner ruling — because resolving them would choose between
two recorded numbers or two recorded wordings — are collected in
[`docs/REVIEW_OPEN_QUESTIONS.md`](REVIEW_OPEN_QUESTIONS.md).
