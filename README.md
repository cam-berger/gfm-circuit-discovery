# Mechanistic Discovery in Genomic Foundation Models

> **Tier: T1 — canonical.** Current, authoritative. Read this.

This repository develops a disciplined way to discover **how genomic foundation models (GFMs)
implement biological computations**. The objective is not to catalogue salient activations; it is to
move from a model-native behavior to a causally supported, explicitly bounded mechanism.

> **Repository state (2026-07-28):** the scientific record includes the SpliceBERT donor circuit
> (A0–A12), positional carriers, Head–Context Atlas, cross-example transduction, and cross-model
> carrier origin. The active transfer study is the DNABERT-2 Biological Reconstruction Atlas.

**SpliceBERT is the first worked example, not the boundary of the project.** The active transfer
program is the DNABERT-2 Biological Reconstruction Atlas: a behavior-first census across annotated
sequence elements. Caduceus remains a prepared, deferred architecture-contrast design.

## Current mechanism, in one sentence

> A structured, top-ranked causal core binds a predominantly relative positional address to an
> acceptor-local payload, then delivers the resulting information to the donor readout within an
> irreducibly distributed computation.

Here, **circuit** means a causally supported collection of components responsible for a scoped behavior —
throughout this repository, a **structured causal core within an irreducibly distributed computation**.
It does not imply that the collection is the smallest possible circuit. In fact, SpliceBERT's 90%
faithfulness requires about 72 of 103 measured nodes; the named circuit is the structured core of that
distributed computation, not a minimal graph. The controlled vocabulary for this and the other
load-bearing terms — including which retired numbers must not reappear in live prose — is
[`docs/NOMENCLATURE.md`](docs/NOMENCLATURE.md).

The key limitation is equally important: the payload is acceptor-local and is not explained by the
preregistered bulk-composition variables, but its **semantic identity remains unresolved**. The earlier
“splice-site strength” label was withdrawn after it failed independently seeded draws and two-way
gene-dyadic inference (R22).

## Current scientific record

| Result | Evidence reached | Licensed interpretation | First missing bridge |
|---|---|---|---|
| Native donor–acceptor coupling | Behavior + specificity | A masked donor score depends on its own paired acceptor under canonical short-intron controls. | Wider biological and sequence-context transduction. |
| Donor graph A0→A12 | QK → attention → OV/mediation → behavior | An acceptor-local generator feeds L3H1, which binds payload to a mostly relative address and delivers a low-rank donor-side signal through a structured core. | Payload semantics; cross-context causal transfer. |
| Period-3 carrier C8/C34/C35/C39 | Position → routing → mediation → behavior; origin corpus-attributed | A distributed relative-positional routing coordinate shapes L3H1's attention across the donor–acceptor span. Origin: built by multi-species pretraining, codon-linked (absent in human-only sibling and in ncRNA models, C39/C40). | Training *dynamics* (trajectory); upstream effects on payload construction. |
| Multispecies ≈10.63 carrier C12/C37/C39 | Position → QK → attention; origin corpus-attributed | Eight heads read a distinct positional carrier after softmax. Origin: multi-species pretraining — strong in the 72-species stage-1 sibling, absent in the human-only sibling, and **not detectable** at ≈10.63 in 60 Mb of human genome above that unanchored assay's 5–10% modulation floor (a sensitivity limit, not an absence). The nucleosomal comparator that excludes the chromatin reading, ≈10.298 nt, is measured in **mouse** ES-cell chromatin (Voong 2016, mm9) — 0.33 nt from the carrier — a gap ~7× the largest uncertainty on either side (0.045 nt each), so this remains a powered refutation (R30, R37). Do not quote a "±0.005" carrier fit precision (R37). | General OV/behavioral function still open. |
| Human-only ≈10.2 proposal | Position/QK; fails at attention | The proposed chromatin-reader circuit is closed at the attention gate. | No higher claim is licensed. |
| Head–Context Atlas | Matched discovery → group validation → conditional mechanism tests | Four general-use component candidates validate; context-specific signals remain exploratory because they missed the frozen precision gate. | Better-powered context cohorts and cross-example transduction. |

Evidence levels are not interchangeable. A feature in a position table is not automatically used by
attention; attention is not automatically causal; a causal model mechanism is not automatically a
biological mechanism.

**Two checkpoints appear in this table.** The coupling behavior, the donor graph, the period-3 carrier,
and the Atlas are measured on `SpliceBERT.1024nt` (final, multi-species). The ≈10.63 and human-only ≈10.2
rows come from the stage-1 pair `SpliceBERT.510nt` and `SpliceBERT-human.510nt`, which differ only in
training data — establishing the ≈10.63 attention circuit required that matched contrast. A result from
one checkpoint is not a result about the other; see [`MECHANISM.md`](MECHANISM.md).

## Method

Every study follows the same broad sequence:

1. Define the biological computation and a model-native behavior.
2. Preflight the assay with positive, negative, sensitivity, resolution, and invariant gates.
3. Use probes, attention, attribution, spectra, or subspaces only to generate candidates.
4. Test necessity with matched-size or structured null interventions.
5. Test mediation, direction, and information using patching, rescue, transplant, QK/OV decomposition,
   or edge/path interventions.
6. Replicate across appropriate examples, draws, splits, contexts, checkpoints, or models and stop at
   the strongest interpretation licensed by the evidence.

Attribution ranks; intervention decides. See the [methodology](docs/METHODOLOGY.md),
[evidence ladder](docs/EVIDENCE_LADDER.md), and [mechanism account](MECHANISM.md).

## Study map

| Study | Status | Purpose |
|---|---|---|
| [`studies/splicebert/`](studies/splicebert/README.md) | worked-example overview | Concise entry point to the SpliceBERT evidence ladder. |
| [`studies/splicebert_partner_selection/`](studies/splicebert_partner_selection/README.md) | completed core record; open limitations | Donor graph, period-3 carrier, compact claims, corrections, preregistrations, and reproduction map. |
| [`studies/splicebert_head_context_atlas/`](studies/splicebert_head_context_atlas/README.md) | discovery and internal validation complete | Multi-context head/MLP reuse atlas; context-specific follow-ups remain exploratory. |
| [`studies/splicebert_segment_attribution/`](studies/splicebert_segment_attribution/README.md) | exploratory instrument | Joint-mask attribution DAGs used as candidate generators, not causal claims. |
| [`studies/dnabert2_reconstruction_atlas/`](studies/dnabert2_reconstruction_atlas/FINDINGS.md) | baseline census bounded; promoter circuit traced to a late core, then shown to be positionally rather than content addressed (D17) | Systematic DNABERT-2 behavior census and causal reverse engineering of a promoter-fine-tuned model. |
| [`studies/caduceus_partner_transfer/`](studies/caduceus_partner_transfer/README.md) | design hardened; deferred, not run | Planned cross-architecture behavioral transfer to an RC-equivariant bidirectional Mamba MLM. |
| [`studies/codon_phase_lattice/`](studies/codon_phase_lattice/README.md) | historical exploratory protocol | Superseded phase-shift framing retained for provenance; not an active result family. |

The reusable study template is under
[`studies/TEMPLATE_REAL_GFM_STUDY/`](studies/TEMPLATE_REAL_GFM_STUDY/README.md).

## Repository map

- `docs/` — transferable method, evidence rules, transfer guide, roadmap, and cleanup record.
- `scripts/splicebert/` — active SpliceBERT experiment, audit, and figure scripts.
- `scripts/splicebert_head_context_atlas/` — Atlas census, screening, mechanism, merge, and audit workflows.
- `scripts/crossmodel/` — cross-model carrier runners (C39/C40). These require a separate environment;
  see [`REPRODUCE.md`](REPRODUCE.md).
- `scripts/dnabert2_atlas/` — annotated-collection assembly, validation, and phase-gated atlas runs.
- `src/gfmcd/adapters/` — model access and hook interfaces.
- `src/gfmcd/reconstruction_atlas/` — BPE-aware masking, controls, source-region interventions,
  statistics, gates, and immutable atlas artifact contracts.
- `src/gfmcd/circuits/` — reusable circuit-analysis primitives.
- `src/gfmcd/crossmodel/` — cross-model carrier-origin assays (RNA-FM / RNA-BERT / DNABERT-2; C39/C40).
- `src/gfmcd/splicing/` — SpliceBERT-specific assays, competition logic, and cohort utilities.
- `results/splicebert_coupling/`, `results/phase-lattice/`, and
  `results/splicebert_head_context_atlas/` — immutable results, compact matrices, manifests, and audits.
- `results/origin-crossmodel/` and `results/origin-siblings/` — the carrier-origin natural experiments
  (C39/C40). These two use a `canonical/` subdirectory; the other result trees keep canonical artifacts
  at the subtree root. See [`results/RESULTS_INDEX.md`](results/RESULTS_INDEX.md).
- `figures/splicebert/` — generated figures and declared provenance.
- `tests/` — the memorial and analysis-logic suite. Many memorials are data-gated and skip without a
  local genome; see [`REPRODUCE.md`](REPRODUCE.md).
- `configs/` — versioned study configurations, including DNABERT-2 site registries and frozen
  phase configs.
- `data/` — documentation only (`NUCLEOSOME_MAP.md`). The model, FASTA, and GTF are **not committed**
  and must be supplied; see [`REPRODUCE.md`](REPRODUCE.md).
- Runtime handoffs, logs, and scratch files are ignored. Historical DNABERT-2 programs are referenced
  by commit from the atlas archive index rather than copied into the active tree.

Large Atlas shards and reproducible runtime intermediates are intentionally excluded from Git; compact
results, manifests, audits, code, tests, and Colab notebooks are version-controlled.

Historical GFM-nano, reading-frame, repeat-copy, and retracted v1 partner-selection implementations are
preserved in Git history rather than restored to the active tree.

## Document tiers

Over 160 markdown documents accumulated across the study's history, and several of them cover
overlapping ground. Nothing is deleted or merged: the superseded narratives are part of the record. Instead
every document carries a tier banner immediately under its title, so a reader can tell at a glance whether
it is current guidance or a retained record.

| Tier | Meaning | Where it is used |
|---|---|---|
| **T1 — canonical** | Current, authoritative. Read this. | This file, `CLAUDE.md`, `AGENTS.md`, `MECHANISM.md`, `REPRODUCE.md`, the `docs/` method set, each study's `CLAIMS.md`, each `results/**/RESULTS_INDEX.md`, and the figure READMEs. |
| **T2 — provenance** | Supporting record: preregistrations, decision logs, correction ledgers, audits. | `CORRECTIONS.md`, `DECISIONS*.md`, `preregistrations/`, `redteam/`, `RESULTS_VERIFICATION.md`, review and audit documents, study-level narratives and run reports. |
| **T3 — historical** | Superseded or point-in-time; retained for the record. Not current guidance. | Documents whose own text marks them superseded or archived, dated one-off plans and handoffs, and captions for figures that are no longer figures of record. |

The tier is navigational only. It records where a document sits relative to the current record; it never
adjusts what that document says. Where a T2 or T3 document disagrees with a T1 one, the claim and
correction ledgers decide — `studies/splicebert_partner_selection/CLAIMS.md` for live wording,
[`CORRECTIONS.md`](studies/splicebert_partner_selection/CORRECTIONS.md) for why it changed.

The canonical set is small on purpose:
[`MECHANISM.md`](MECHANISM.md) · [`REPRODUCE.md`](REPRODUCE.md) ·
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) · [`docs/EVIDENCE_LADDER.md`](docs/EVIDENCE_LADDER.md) ·
[`docs/NOMENCLATURE.md`](docs/NOMENCLATURE.md) · [`docs/BIOLOGICAL_BRIDGES.md`](docs/BIOLOGICAL_BRIDGES.md) ·
[`docs/PREREG_CONFORMANCE.md`](docs/PREREG_CONFORMANCE.md) ·
[`docs/REFERENCES.md`](docs/REFERENCES.md) · [`docs/TRANSFER_GUIDE.md`](docs/TRANSFER_GUIDE.md) ·
[`docs/ROADMAP.md`](docs/ROADMAP.md) · [`results/RESULTS_INDEX.md`](results/RESULTS_INDEX.md) ·
the per-study `CLAIMS.md` files.

## Start here

1. Read the [methodology](docs/METHODOLOGY.md) and [evidence ladder](docs/EVIDENCE_LADDER.md).
2. Read the one-file [current mechanism and limitations](MECHANISM.md).
3. Use the [live claim ledger](studies/splicebert_partner_selection/CLAIMS.md); consult
   [corrections](studies/splicebert_partner_selection/CORRECTIONS.md) before quoting a result.
4. Review the [Head–Context Atlas](studies/splicebert_head_context_atlas/README.md) for the current
   cross-context branch.
5. Read the [transfer guide](docs/TRANSFER_GUIDE.md) and the
   [DNABERT-2 Atlas](studies/dnabert2_reconstruction_atlas/README.md) for the active second-GFM program.
6. Follow [`REPRODUCE.md`](REPRODUCE.md) for the canonical environment and claim→script→artifact map.

For development:

```bash
uv sync --extra splicebert --extra dev
uv run pytest
```

Run experiments from the repository root. Result-producing scripts must write an immutable result and a
manifest; no new run may silently overwrite the scientific record.
