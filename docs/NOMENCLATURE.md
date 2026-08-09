# Nomenclature: the project's controlled vocabulary

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Several of this project's terms have been retired, withdrawn, or bounded, and several near-identical
numbers refer to entirely different objects at entirely different evidence levels. This file is the
single controlled vocabulary. It **records** decisions taken elsewhere; it does not make new ones.
Where this file and a claim ledger differ, the ledger wins.

Authoritative detail lives in
[`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md),
[`studies/splicebert_partner_selection/CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md),
and [`results/phase-lattice/RESULTS_INDEX.md`](../results/phase-lattice/RESULTS_INDEX.md).

**Statuses.** `LIVE` — usable in current prose, under the stated conditions. `RETIRED` — the term was
never a measurement of what it appeared to name; do not use it in live prose. `WITHDRAWN` — the claim
the term encoded was retracted by an R-number. `CLOSED` — the hypothesis was tested and failed at a
named gate; the term stays, but only as a closed result.

---

## The vocabulary

| Term | Status | Sentence that must accompany it | Authoritative detail |
|---|---|---|---|
| **≈10.63 carrier** (multi-species) | LIVE — model-internal, attention level only; corpus origin attributed | The multi-species ≈10.63-nt positional carrier is read through post-softmax attention by eight heads; its general OV content and behavioral function remain open (C12; C37 is a donor-scoped OV null) and no biological correlate is established, but its **corpus origin is attributed** (multi-species training; C39/C40). | C11, C12, C37, C39, C40, R25; `results/phase-lattice/RESULTS_INDEX.md` |
| **10.298** (nucleosomal) | LIVE — biological, sequence-side | ≈10.298 nt is a real dyad-anchored nucleosomal periodicity measured **in DNA, not in the model**, and specifically **in mouse ES-cell chromatin** (Voong 2016 chemical-cleavage dyad map, mm9). It sits 0.33 nt from the model's ≈10.630 carrier — a gap **~7× the largest uncertainty on either side** (carrier two-grid sub-bin spread 0.045 nt; nucleosomal estimator spread 0.045 nt) — so the nucleosomal explanation of the carrier is a **powered refutation**, not an open possibility. Do **not** quote a "±0.005" carrier fit precision: that figure was a bin-unit synthetic-control error misread as a measurement uncertainty (R37). The refutation does **not** depend on the mouse point estimate transferring to human at better than 0.33 nt: 10.630 lies outside the 10.2–10.5 bp DNA-helical range recorded from the literature in the row below. **No human dyad-anchored measurement exists in this record** — always name the organism. | C14, C15, R17, R30, R37; `studies/splicebert_partner_selection/CLAIMS.md`; `data/NUCLEOSOME_MAP.md` |
| **"10.67"** (also 10.6667, 10.6875) | RETIRED | "10.67" was the label of a discrete Fourier bin (512/48 on the stage-1 grid, 1026/96 = 10.6875 on the final grid), not a wavelength measurement; the grid labels disagree while the sub-bin measurement converges on ≈10.630. Use **≈10.63 carrier**. | `results/phase-lattice/RESULTS_INDEX.md` retired list; `figures/splicebert/fig2_oscillators.CAPTION.md` panel D |
| **"10.58 ± 0.35"** | RETIRED | "10.58 ± 0.35" is not a measured splice-junction constant and is not an interpretation of the carrier. Omit it; use ≈10.63 carrier. | `results/phase-lattice/RESULTS_INDEX.md` retired list |
| **Human-only ≈10.2 component** | CLOSED at the attention gate | The proposed human-only ≈10.2 chromatin-reader circuit has zero corrected Level-A heads and is closed at the attention gate; that refutes the proposed reader set, not every possible chromatin representation. | C13 |
| **10.2–10.5 bp helical periodicity** | LIVE — distinct referent, disambiguate | "≈10.2–10.5 bp" is the generic DNA-helical periodicity range from the literature and is **not** the human-only ≈10.2 candidate component of SpliceBERT's position table; never let the two share an unqualified "10.2". | `docs/REVIEW_2026-07-22.md` §7 |
| **"splice-site strength"** as the payload's semantic identity | WITHDRAWN (R22) | The payload is acceptor-local and is not explained by the preregistered bulk-composition covariates, but its **semantic identity remains unresolved**; the "splice-site strength" naming was withdrawn under R22. Say "an acceptor-local payload". | R22; C30 (bounded) |
| **splice-site strength** as an external MaxEnt covariate | LIVE — a different use, keep | MaxEntScan splice-site strength remains a legitimate external sequence covariate for matching and stratification; R22 withdrew it as the payload's semantic identity, not as a measurable sequence property. | `DECISIONS_A0.md` D-A0-3; `PREREGISTRATION_A0` |
| **circuit** | LIVE — always qualified at first use | A **structured causal core within an irreducibly distributed computation**: causally necessary and sufficient when its clean outputs are injected, but not minimal — 90% faithfulness needs about 72 of 103 measured nodes and the size curve has no knee. | `docs/METHODOLOGY.md`; C27, C33 |
| **period-3 carrier** | LIVE — role established; corpus origin attributed, training dynamics open | The period-3 carrier acts as a distributed relative-positional routing coordinate read across the donor–acceptor span, through L3H1's attention pattern; its **corpus origin is attributed** (C39/C40: multi-species and codon-linked — absent in the human-only sibling and in ORF-free ncRNA models), while training *dynamics* remain deferred. | C8, C34, C35, R24, C39, C40, R25 |
| **coding-boundary phase** (also "annotated phase", `p`, the `0/1/2` of C43/C44) | LIVE — disambiguate from the period-3 carrier at first use | **Coding-boundary phase is a property of the annotation, not of the model's position table.** It is the codon reading-frame phase (0/1/2) of a coding exon boundary in GENCODE. It is **not** the period-3 carrier's phase, which is a positional routing coordinate learned by the model — the two are different objects at different evidence levels, and C44 is precisely the test of whether the carrier implements a coding-boundary-phase preference. That test returned `INSTRUMENT_UNRESOLVED`, so **no sentence may present the period-3 carrier as the mechanism of C43**, and equally none may say period-3 *fails* to implement it. Never write an unqualified "phase" in prose that also discusses the carrier. | C43, C44, R45; `studies/splicebert_exon_edge_compatibility/README.md` |
| **"authentic-module" context** (C43's lattice) | RETIRED — say **chimeric** | The C43 constructs join a donor module from one gene to an acceptor module from another; each module is authentic hg38 sequence but the joined construct exists in no genome. The source study wrote "synthetic-module" while the result was a development candidate and "authentic-module" once it became supported; that substitution was corrected on consolidation (R45) and must not be reintroduced from Git history. The retired phrase nonetheless survives verbatim in the carried frozen protocol and in the `scope` fields of the carried artifacts, which are immutable and correctly untouched — read those occurrences as "chimeric". Relatedly, "120-nt" names the **donor→acceptor span**, not the length of either module. | C43, R45 |
| **checkpoint attribution** | LIVE — mandatory disclosure | The donor circuit and the period-3 carrier are established on **SpliceBERT.1024nt**; the ≈10.63 carrier's attention circuit is measured on the stage-1 multi-species **SpliceBERT.510nt** against its human-only twin — state which checkpoint a result comes from, and never present the two as one measurement. | `figures/splicebert/fig1_summary.CAPTION.md` "A note on checkpoints" |

---

## Standing phrases

Two phrases carry a caveat that a skimming reader would otherwise miss. Use them verbatim.

1. **At first use of "circuit" in any top-level document:** *structured causal core within an
   irreducibly distributed computation*. The named circuit is not sparse and minimality is unclaimed.
2. **Whenever ≈10.63 and ≈10.298 appear together:** the "0.33 nt apart / powered refutation" qualifier
   must be in the **same sentence**, and **≈10.298 must be named as the mouse ES-cell measurement it is**
   (Voong 2016, mm9). A reader must never be able to skim the two numbers and conclude that the model
   found nucleosome positioning, and must never infer that a human dyad map exists in this record — none
   does (R30). Express the power as the gap **relative to the measured spreads** (0.045 nt on either
   side, so ~7×); **do not quote a "±0.005" fit precision** — that figure was a bin-unit
   synthetic-control error misread as a measurement uncertainty (R37). The refutation does not rest on
   cross-species transfer at 0.33 nt precision: 10.630 lies outside the 10.2–10.5 bp DNA-helical range
   recorded from the literature in the entry below.

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
`results/phase-lattice/RESULTS_INDEX.md`, in the "Retired language" section of
`figures/splicebert/README.md`, and in any preregistration or decision log that documents the
retraction. Removing a retired term from its own retirement record would destroy the correction
history, which is part of the evidence. This file governs **live prose only**: prose that asserts
something in the present tense.

## Open nomenclature questions

Items that cannot be resolved without an owner ruling — because resolving them would choose between
two recorded numbers or two recorded wordings — are collected in
[`docs/REVIEW_OPEN_QUESTIONS.md`](REVIEW_OPEN_QUESTIONS.md).
