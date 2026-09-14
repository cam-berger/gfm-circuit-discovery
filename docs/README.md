# Research documentation

Start with the [repository overview](../README.md) for the scientific question and current scope.
This directory explains how experiments are designed, interpreted, and audited. Exact scientific
claims remain controlled by each study's claim ledger, corrections, and immutable result artifacts.

## Reading paths

| Purpose | Start here |
|---|---|
| Understand the current evidence | [Evidence audit](../RESULTS.md), [mechanism and limits](../MECHANISM.md), [study index](../studies/README.md) |
| Reproduce or inspect a result | [Reproduction guide](../REPRODUCE.md), [result index](../results/RESULTS_INDEX.md), [script index](../scripts/README.md) |
| Design a new experiment | [Methodology](METHODOLOGY.md), [evidence ladder](EVIDENCE_LADDER.md), [transfer guide](TRANSFER_GUIDE.md) |
| Check terminology and historical qualifications | [Nomenclature](NOMENCLATURE.md), [preregistration conformance](PREREG_CONFORMANCE.md), the relevant study's corrections |

## Methods and research standards

- [Methodology](METHODOLOGY.md) — the seven-stage workflow, causal estimands, intervention design,
  controls, and limits of circuit claims.
- [Evidence ladder](EVIDENCE_LADDER.md) — the experiments required to move from observation to causal
  use, mediation, directed circuitry, and biological interpretation.
- [Nomenclature](NOMENCLATURE.md) — checkpoint names, units, positional carriers, payload terminology,
  and retired interpretations.
- [Preregistration conformance](PREREG_CONFORMANCE.md) — documented differences between frozen
  protocols and executed code. Consult this before describing a historical gate as enforced.
- [Biological bridges](BIOLOGICAL_BRIDGES.md) — the additional evidence needed to connect a model
  mechanism to training history or biological function.
- [Transfer guide](TRANSFER_GUIDE.md) — adapter, assay, and study requirements for another genomic
  foundation model.

## Supporting resources

- [Roadmap](ROADMAP.md) — open questions, deferred work, and stopped programs.
- [References](REFERENCES.md) — external literature and its verification status.
- [Third-party assets](THIRD_PARTY_ASSETS.md) — external checkpoints, genomes, annotations, and license
  terms.
- [Repository audit procedure](REPO_SCRUB_PROCEDURE.md) — how to maintain a readable repository while
  preserving the research record.
- [Method coverage survey](MECH_INTERP_GAP_ANALYSIS.md) — advisory inventory and proposed extensions;
  consult current claims and decisions before acting on its recommendations.

## Historical reviews

These records preserve the context of earlier audits. Current study ledgers and corrections determine
which findings and decisions still apply.

| Record | Retained purpose |
|---|---|
| [Repository review, July 2026](REVIEW_2026-07-22.md) | Section 4 defines the result-manifest contract used by the verifier and tests. |
| [DNABERT-2 integration review](REVIEW_2026-08-11_DNABERT_PR40.md) | Background for the promoter study's claim and correction ledgers. |
| [Review questions and dispositions](REVIEW_OPEN_QUESTIONS.md) | Historical remediation decisions and unresolved questions; includes the CI dependency-policy rationale. |

Frozen protocols live with their [owning studies](../studies/README.md). They retain the wording and
paths used at execution; later corrections belong in the live record rather than in those protocols.
