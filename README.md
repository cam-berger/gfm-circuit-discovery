# Mechanistic Discovery in Genomic Foundation Models
## Current mechanism

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

## Scientific record

| Result | Evidence reached | Licensed interpretation | First missing bridge |
|---|---|---|---|
| Native donor–acceptor coupling | Behavior + specificity | A masked donor score depends on its own paired acceptor under canonical short-intron controls. | Wider biological and sequence-context transduction. |
| Donor graph A0→A12 | QK → attention → OV/mediation → behavior | An acceptor-local generator feeds L3H1, which binds payload to a mostly relative address and delivers a low-rank donor-side signal through a structured core. | Payload semantics; cross-context causal transfer. |
| Period-3 carrier C8/C34/C35/C39 | Position → routing → mediation → behavior; origin corpus-attributed | A distributed relative-positional routing coordinate shapes L3H1's attention across the donor–acceptor span. Origin: built by multi-species pretraining, codon-linked (absent in human-only sibling and in ncRNA models, C39/C40). | Training *dynamics* (trajectory); upstream effects on payload construction. |
| Coding-boundary phase C43/C44 | Behavior (replicated); mediation unresolved | Masked-donor partner coupling prefers **annotated** donor–acceptor codon reading-frame matches (`Ψ=+0.3901`, fresh outcome-blind replication) — but only inside a frozen **chimeric** crossed lattice at one span, with the resolution gate passing at its Monte-Carlo noise floor, and "only when matched" is **not** licensed. Whether the period-3 carrier implements it is **unresolved** (C44): four of the frozen authorization gates failed, the 384-block target was never exposed, and neither a positive, a null, nor an inverted conclusion is licensed. | A capable mechanism instrument. The local carrier/A-V/Q-K family is attempted-and-unresolved, not eliminated. |
| Multispecies ≈10.63 carrier C12/C37/C39 | Position → QK → attention; origin corpus-attributed | Eight heads read a distinct positional carrier after softmax. Origin: multi-species pretraining — strong in the 72-species stage-1 sibling, absent in the human-only sibling, and **not detectable** at ≈10.63 in 60 Mb of human genome above that unanchored assay's 5–10% modulation floor (a sensitivity limit, not an absence). The nucleosomal comparator that excludes the chromatin reading, ≈10.298 nt, is measured in **mouse** ES-cell chromatin (Voong 2016, mm9) — 0.33 nt from the carrier — a gap ~7× the largest uncertainty on either side (0.045 nt each), so this remains a powered refutation (R30, R37). Do not quote a "±0.005" carrier fit precision (R37). | General OV/behavioral function still open. |
| Human-only ≈10.2 proposal | Position/QK; fails at attention | The proposed chromatin-reader circuit is closed at the attention gate. | No higher claim is licensed. |
| Head–Context Atlas | Matched discovery → group validation → conditional mechanism tests | Four general-use component candidates validate; context-specific signals remain exploratory because they missed the frozen precision gate. C38/C41 separately establish strong but bounded cross-example transfer. | Better-powered context cohorts and transfer-specificity tests. |

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
| [`studies/splicebert_partner_selection/`](studies/splicebert_partner_selection/README.md) | **core record**; open limitations | The donor circuit and both positional carriers: claims, corrections, preregistrations, and the reproduction map. |
| [`studies/splicebert_head_context_atlas/`](studies/splicebert_head_context_atlas/README.md) | discovery and internal validation complete | Multi-context head/MLP reuse atlas. Supplies the gene-disjoint cohort behind C41 and an independent rediscovery of L3H1; context-specific follow-ups remain exploratory. |
| [`studies/splicebert_gtex_finetune_shift/`](studies/splicebert_gtex_finetune_shift/README.md) | native causal mechanism mapped; sparse-difference instrument bounded | What happens to the same encoder under supervised fine-tuning: late attention routes into a compact final-MLP code, and inherited positional-carrier use. |

The reusable study template is under
[`studies/TEMPLATE_REAL_GFM_STUDY/`](studies/TEMPLATE_REAL_GFM_STUDY/README.md).

Five earlier study directories were removed from the working tree in the 2026-07-31 cleanup and remain
in Git history: the DNABERT-2 reconstruction atlas, the Caduceus transfer design (hardened but never
run), the superseded codon phase-lattice protocol, the joint-segment attribution instrument, and the
`splicebert/` overview stub whose content is now in [`MECHANISM.md`](MECHANISM.md). None of them
controls a live claim in the ledgers above.

A sixth, the **exon-edge compatibility arc**, was **consolidated rather than merged** on 2026-08-04: its
eleven `E*` rows became two rows in the donor ledger (**C43**, **C44**) and its ~100 MB of runners, modules,
tests, and cohorts stayed on branch `codex/intron-exon-phase` at commit **`fd26cc1b`**. Unlike the five
above it *does* control live claims, so
[`studies/splicebert_exon_edge_compatibility/`](studies/splicebert_exon_edge_compatibility/README.md)
survives holding only an archive record and the two frozen protocols behind those claims. Read
[`CORRECTIONS.md` R45](studies/splicebert_partner_selection/CORRECTIONS.md) before citing anything from
that arc: it reproduces the nine dropped rows and records three overstatements in the original
study's prose that must not be reintroduced from Git history.

## Repository map

- `docs/` — transferable method, evidence rules, nomenclature, transfer guide, and roadmap.
- `scripts/splicebert/` — active SpliceBERT experiment, audit, and figure scripts.
- `scripts/splicebert_head_context_atlas/` — Atlas census, screening, mechanism, merge, and audit workflows.
- `scripts/crossmodel/` — cross-model carrier runners (C39/C40). These require a separate environment;
  see [`REPRODUCE.md`](REPRODUCE.md).
- `src/gfmcd/adapters/` — model access and hook interfaces.
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
- `configs/` — versioned study configurations and frozen phase configs.
- `data/` — documentation only (`NUCLEOSOME_MAP.md`). The model, FASTA, and GTF are **not committed**
  and must be supplied; see [`REPRODUCE.md`](REPRODUCE.md).
- Runtime handoffs, logs, and scratch files are ignored.

Repository code is distributed under the [MIT License](LICENSE). External data and model dependencies,
including their separate terms, are summarized in [third-party assets](docs/THIRD_PARTY_ASSETS.md).


## Start here

1. Read the [methodology](docs/METHODOLOGY.md) and [evidence ladder](docs/EVIDENCE_LADDER.md).
2. Read the one-file [current mechanism and limitations](MECHANISM.md).
3. Use the [live claim ledger](studies/splicebert_partner_selection/CLAIMS.md); consult
   [corrections](studies/splicebert_partner_selection/CORRECTIONS.md) before quoting a result.
4. Review the [Head–Context Atlas](studies/splicebert_head_context_atlas/README.md) for the current
   cross-context branch.
5. Review the [GTEx fine-tuning-shift findings](studies/splicebert_gtex_finetune_shift/FINDINGS.md) for
   the supervised task representation and its current sparse/biological boundaries.
6. Read the [transfer guide](docs/TRANSFER_GUIDE.md) for how to port the method to another GFM.
7. Follow [`REPRODUCE.md`](REPRODUCE.md) for the canonical environment and claim→script→artifact map.

For development:

```bash
uv sync --extra splicebert --extra biology --extra dev
uv run --extra splicebert --extra biology --extra dev pytest
```

Run experiments from the repository root. Result-producing scripts must write an immutable result and a
manifest; no new run may silently overwrite the scientific record.
