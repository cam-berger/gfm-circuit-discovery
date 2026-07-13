# Documentation index

**[INDEX]** — the map of this folder. Every document below has a label, a purpose, and a status.
Nothing lives in `docs/` without one; anything that lost its purpose is in `docs/archive/` with a note
saying what replaced it.

**Start here:** [`STATUS.md`](STATUS.md) — where the program stands today, across all three tracks,
including what has been retracted.

---

## The six labels

| Label | Answers | Rule |
|---|---|---|
| **INDEX** | *where do I find things?* | This file. |
| **BACKGROUND** | *why does any of this work?* | Written for a reader new to the field. Timeless — no results. |
| **METHOD** | *how do I do it?* | Procedure and API. Reusable across models. No claims. |
| **RECORD** | *what happened, and how do I check it?* | Every number carries a status. Retractions stay in, with what killed them. |
| **ARGUMENT** | *why should anyone care?* | Outward-facing prose. Every claim must trace to a RECORD doc. |
| **PLAN** | *what next?* | Forward-looking only. When a plan is executed, it moves to `archive/` and its results move to a RECORD doc. |

The chain a reviewer should be able to walk, for any claim in the repo:

```
ARGUMENT  (the claim)
   ↓ cites
RECORD    RESULTS.md — the number, with a status tag
   ↓ cites
RECORD    DECISIONS.md — why we measured it that way, and what we got wrong first
   ↓ cites
RECORD    PREREGISTRATION.md — what we predicted before we looked
   ↓ names
          the script that regenerates it
```

If a claim cannot be walked back to a script, it is not a claim. That is what this folder is for.

---

## Reading orders

**If you are reviewing this work** (2 hours, in order):
1. [`STATUS.md`](STATUS.md) — the current state, and the honest list of what is broken.
2. [`REC.md`](REC.md) — the argument, and the critique of the proposal it replaced.
3. [`RESULTS.md`](RESULTS.md) — every number, with a status. Read §10, the retractions, first.
4. [`DECISIONS.md`](DECISIONS.md) — D1–D35. The chain of thought, including six retracted claims.
5. [`RELAY_TRACE.md`](RELAY_TRACE.md) — the deepest single trace, and the retraction that reshaped it.

**If you are joining the project** (a day):
1. [`FOUNDATIONS_circuit_discovery.md`](FOUNDATIONS_circuit_discovery.md) — the mechanistic-interpretability background.
2. [`BIOLOGY_genomic_computations.md`](BIOLOGY_genomic_computations.md) — the genomics, for an ML audience.
3. [`CIRCUIT_DISCOVERY_HOW_TO.md`](CIRCUIT_DISCOVERY_HOW_TO.md) — the five-pillar procedure.
4. [`LAB.md`](LAB.md) — the `gfmcd.lab` API, then open `notebooks/study_reading_frame_counter.ipynb`.
5. [`EXPERIMENTS.md`](EXPERIMENTS.md) — the plain-language account of how we got here.

**If you are starting the next GFM study** (the "spring from" path):
1. [`STATUS.md`](STATUS.md) — in particular, what the SpliceBERT retractions cost us.
2. [`REAL_GFM_CIRCUIT_DISCOVERY_PROTOCOL.md`](REAL_GFM_CIRCUIT_DISCOVERY_PROTOCOL.md) — the seven questions, the gate ladder, the claim ladder, and the stop rules. **This is the reusable asset.**
3. `studies/TEMPLATE_REAL_GFM_STUDY/` — copy it.
4. [`TODO.md`](TODO.md) — the reproducibility debt you will inherit.

---

## The documents

### Orientation

| Document | Label | Purpose | Status |
|---|---|---|---|
| [`STATUS.md`](STATUS.md) | INDEX | Where the program stands today: three tracks, what is settled, what is retracted, what is broken. The single source of truth for counts and status. | Current — 2026-07-12 |

### Background — why the methods work

| Document | Label | Purpose | Status |
|---|---|---|---|
| [`FOUNDATIONS_circuit_discovery.md`](FOUNDATIONS_circuit_discovery.md) | BACKGROUND | Mechanistic interpretability from first principles for a first-year grad student: circuits, superposition, patching, attribution, induction heads, grokking, shortcut learning. | Current |
| [`BIOLOGY_genomic_computations.md`](BIOLOGY_genomic_computations.md) | BACKGROUND | The genomics for an ML audience: what reading frame, repeats, splicing, NMD, CTCF orientation *are*, and what the shortcut vs. the circuit looks like for each. | Current |

### Method — how to run a study

| Document | Label | Purpose | Status |
|---|---|---|---|
| [`CIRCUIT_DISCOVERY_HOW_TO.md`](CIRCUIT_DISCOVERY_HOW_TO.md) | METHOD | The five-pillar procedure, step by step, on a model you control. Discovery → testing → ablation → baselines → conflict. | Current |
| [`REAL_GFM_CIRCUIT_DISCOVERY_PROTOCOL.md`](REAL_GFM_CIRCUIT_DISCOVERY_PROTOCOL.md) | METHOD | The protocol for a *pretrained* GFM: seven questions, ten gates, the claim ladder, the stop rules, and the checklist for adding a new study. | Current — the asset the next GFM springs from |
| [`LAB.md`](LAB.md) | METHOD | `gfmcd.lab` API reference and the notebook workflow. | Current |

### Record — what happened, and how to check it

| Document | Label | Purpose | Status |
|---|---|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | RECORD | Predictions frozen *before* measurement: the five confirmatory tests, the necessity 2×2 verdict, the Phase-C relay hypotheses. | Partly frozen — see the status box in the file |
| [`DECISIONS.md`](DECISIONS.md) | RECORD | D1–D35. Every design decision, the evidence that forced it, what was given up — and every claim we raised and retracted. The chain of thought. | Current |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | RECORD | E1–E8, in plain language. What we did and what it showed, readable without the jargon. | Current — covers the nano tracks |
| [`RESULTS.md`](RESULTS.md) | RECORD | Every measured number, with a status tag (SETTLED / PENDING / WITHDRAWN / RETRACTED). §10 is the retraction table. | Current |
| [`RELAY_TRACE.md`](RELAY_TRACE.md) | RECORD | The Phase-C deliverable: the operational definition of counter / relay / bag, and the retraction of four of five "relays". | Current |

### Argument — the outward-facing case

| Document | Label | Purpose | Status |
|---|---|---|---|
| [`REC.md`](REC.md) | ARGUMENT | The critique of the original proposal and the program that replaced it. The reasoning, with the literature. | Current (v2) — v1 in `archive/` |
| [`Proposal_Sufficiency_Circuit_Discovery.md`](Proposal_Sufficiency_Circuit_Discovery.md) | ARGUMENT | The PI-facing proposal: novelty, hypotheses, task breakdown, contributions. | Current — predates the SpliceBERT retractions; see `STATUS.md` before quoting §4 |

### Plan — what is next

| Document | Label | Purpose | Status |
|---|---|---|---|
| [`PLAN.md`](PLAN.md) | PLAN | The phase-and-gate ladder across all three tracks. A phase that fails its gate stops the ladder. | Current |
| [`TODO.md`](TODO.md) | PLAN | The reproducibility backlog: what stops a clean-checkout reproduction, ordered by how much it blocks. | Current |

### Explainers

| Document | Label | Purpose | Status |
|---|---|---|---|
| [`explainers/circuit_explainer.html`](explainers/circuit_explainer.html) | EXPLAINER | Standalone one-pager: what a circuit is and how we find one. For a non-specialist. | Current |
| [`explainers/mechinterp_primer.html`](explainers/mechinterp_primer.html) | EXPLAINER | Standalone primer on mechanistic interpretability. For a non-specialist. | Current |

### Archive

[`archive/`](archive/) — superseded documents, kept rather than deleted, each with a banner saying what
replaced it and why. See [`archive/README.md`](archive/README.md). Nothing in `archive/` may be cited as
evidence.

---

## Where the other records live

`docs/` holds the program-level record. Two tracks keep their own, and this is deliberate — they are
worked instances, not narrative:

| Location | Holds | Label |
|---|---|---|
| `studies/splicebert_partner_selection/` | The SpliceBERT study record: `CLAIMS.md` (the claims registry, **the current one**), `DECISIONS.md`, `PREREGISTRATION.md`, `FREEZE.md`. | RECORD |
| `studies/TEMPLATE_REAL_GFM_STUDY/` | The scaffold to copy for the next GFM. | METHOD |
| `branches/codex-repeat-induction/` | The repeat-copy track's findings and figures: `FINDING.md`, `COPY_CEILING.md`, `COPY_TRACE.md`, `NATURAL_REPEATS.md`. | RECORD |
| `results/` | The measured numbers as JSON/CSV, with run manifests. Markdown summarises these; it does not replace them. | ARTIFACT |
| `scripts/README.md` | Which script produces which committed number. | METHOD |

**The rule:** a markdown summary is a *view* of an artifact under `results/`, never an alternative
source of truth. If they disagree, the artifact wins and the markdown is wrong.
