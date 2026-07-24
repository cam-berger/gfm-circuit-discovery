# Is our discovery method formalized as a transferable top-down process?

> **Tier: T2 — provenance.** Analysis/characterization note, 2026-07-24 (frame 0a2ba5cc). Reads the
> canonical T1 docs ([`METHODOLOGY.md`](METHODOLOGY.md), [`EVIDENCE_LADDER.md`](EVIDENCE_LADDER.md),
> [`TRANSFER_GUIDE.md`](TRANSFER_GUIDE.md)) and the atlas [`RESEARCH_SPINE.md`](../studies/splicebert_head_context_atlas/RESEARCH_SPINE.md).
> Positions the method against the current mechanistic-interpretability literature. It changes no claim,
> value, or status; where it proposes anything, that lives in "Open method questions" for the owner.

## Short answer

**Yes — the process is already formalized, and it has already begun transferring.** Three canonical (T1)
documents define a complete top-down pipeline that is model-agnostic by construction:

- `METHODOLOGY.md` — the **seven-stage workflow** and eight core principles;
- `EVIDENCE_LADDER.md` — the **five-column promotion ladder** (P → QK → Attention → OV/mediation →
  Behavior), read in causal-abstraction terms;
- `TRANSFER_GUIDE.md` — the **Phase 0–5 port sequence**, the **adapter contract**, and the
  cross-model comparison axes.

A second study, `studies/caduceus_partner_transfer/`, is a live instance of the port sequence applied to
a different architecture (Caduceus) — so "one complete process we can take to other GFMs" is not aspiration;
it is the documented operating model, with SpliceBERT as the worked example.

**One correction to the framing in the request.** The process is **behavior-first, not atlas-first.** The
head–context atlas is not Stage 1; it is the instrument used at **Stage 4 (exploratory localization)**,
*after* a biological computation is specified (Stage 1), the model's behavior on it is established (Stage 2),
and the measurement instrument is preflighted (Stage 3). This ordering is the single most load-bearing
design choice in the method, and the 2026 literature is now full of expensive demonstrations of what happens
when it is violated (see §3). Starting from the atlas and working toward head interventions is the right
*localization→intervention* arc **within** Stages 4–6 — but it must sit downstream of a qualified behavior,
or the whole edifice screens for structure that predicts nothing.

---

## 1. The formalized process, in one view

| Stage (METHODOLOGY §) | Question | Primary tools | Licenses | Ladder column |
|---|---|---|---|---|
| **1. Specify the computation** | What must be detected/integrated, and which native output moves? | Written computational statement + rival explanations | A behavioral assay, not a circuit hypothesis | — |
| **2. Establish behavior** | Does the model actually do it? | Native logits/loss, paired biological counterfactuals, stratified baselines | The behavior exists and survives simple baselines — else **stop** | — |
| **3. Preflight the instrument** | Could the assay find the effect, and reject a fake one? | Positive/negative controls, injection/recovery, invariants, thresholds, stopping rule | The instrument is trustworthy | — |
| **4. Generate & localize candidates** | Where might it live? | **Head–context atlas**, probes, spectra/Fourier, subspaces, attention/QK, attribution, (ACDC/EAP cross-check — recommended, not yet run) | A **candidate set** — ranks, does not decide | P, QK |
| **5. Necessity & specificity** | Is a component required, specifically? | Ablation, projection removal, band deletion + matched nulls | Necessity under this intervention | Attention |
| **6. Mediation, information, direction** | Does it mediate, carry what, in which direction? | Activation/path patching, rescue, subspace transplant, QK/OV decomposition | A directed, information-specific path | OV / mediation |
| **7. Replicate, delimit, interpret** | Does it generalize, and where does it stop? | Held-out/sealed replication, checkpoint/species/model contrasts, cross-example transduction | Scope + the first missing bridge | Behavior |

The **atlas** (`RESEARCH_SPINE.md`) is the Stage-4 instrument, and it is itself disciplined: its unit is not
a head name but an evidence-graded relation `component × biological context × model behavior × causal
mechanism`; it keeps biological and model labels in separate namespaces; and it states explicitly that
"screening ranks candidates; it does not name functions" and that "a head becomes mechanistically
interpretable only after exact group intervention plus QK/OV/MLP/path validation." That is Stage 4 handing
off to Stages 5–6 — the "atlas → interventions on specific heads" arc in the request, correctly bounded.

The eight **core principles** are the invariants that make the pipeline transferable rather than
SpliceBERT-specific: *start with biology then verify behavior; preflight before interpretation; attribution
ranks, intervention decides; necessity/mediation/direction/specificity are different claims; controls must be
capable of failure; evidence never auto-promotes; preserve negatives and retractions; stop at the first
missing bridge.*

---

## 2. Where this sits in the literature

The method is a **top-down, behavior-anchored causal-abstraction pipeline**. Each stage has a named lineage
in the mechanistic-interpretability (MI) literature; the contribution is not a new primitive but the
**assembled, disciplined ordering** and its port contract.

**Causal abstraction is the backbone.** Treating a circuit as a proposed alignment between a high-level
causal model and network variables, tested by intervention, is the Geiger et al. (2025, *JMLR*) framing that
`METHODOLOGY.md` and the five-column ladder already cite. The ladder's OV/mediation column is exactly an
interchange/injection intervention in that vocabulary — the first level at which a graded faithfulness number
can even be computed.

**Attribution-ranks / intervention-decides is the field's hard-won consensus.** A recent survey of attention-
head intervention frames the move "from visualization to intervention" as a paradigm shift from observing
correlations to causally validating hypotheses, and re-derives the specialization-vs-redundancy tension
(many heads are removable; the important ones are not always the interpretable ones). Our principle 3 and the
Stage 4→5 boundary encode this directly: the atlas is observational and only ranks.

**Faithfulness is graded and size-swept, not binary.** The MIB benchmark (Mueller et al., 2025) scores circuit
localization by an area-under-the-faithfulness-vs-size curve (CPR/CMD). Our `EVIDENCE_LADDER.md` uses that
vocabulary and, crucially, reports the honest negative it implies for SpliceBERT: the donor ladder is complete
across all five columns *and* the sparse-circuit faithfulness is low with **no sparse knee** — a full row of ✓
does not assert a sparse faithful circuit. That is the ladder doing its job as a guard, not a scoreboard.

**Automated circuit discovery is the acknowledged, not-yet-run cross-check.** ACDC (Conmy et al., 2023) and
edge-attribution patching (Syed et al., 2024) automate subgraph identification; `METHODOLOGY.md` §4 lists an
ACDC/EAP pass as *recommended and explicitly not yet performed*, and states that no ledger claim rests on one.
Running it would blunt the standard "you found the circuit you went looking for" objection. This is the
clearest, cheapest method upgrade available and it is already named in the canonical doc.

**The top-down vs bottom-up axis.** The field distinguishes *top-down* (map pre-defined interpretable
computations to components) from *bottom-up* (SAE/dictionary-learning feature discovery "without a priori
theories"). Our pipeline is deliberately top-down and behavior-anchored. The bottom-up stack (SAEs,
transcoders, attribution graphs, Gemma Scope-style feature suites) is complementary, not competing: it is a
richer Stage-4 localization basis. Nothing in the method forbids swapping SAE features in as candidate
variables — they would still have to survive Stages 5–6 to be licensed. The adapter contract's
"projection/subspace editing" hook is the natural insertion point.

---

## 3. Why behavior-first ordering is the load-bearing choice (2026 evidence)

The strongest recent case *for* our ordering comes from biological-FM interpretability failures published in
early 2026 — and they are failures of exactly the atlas-first shortcut:

- A systematic single-cell FM interpretability study (scGPT, Geneformer; 37 analyses, 153 tests) found that
  attention encodes structured biology (PPIs early, regulation late) **but adds zero incremental value for
  perturbation prediction**: trivial gene-level baselines beat attention/correlation edges, and **causal
  ablation of the regulatory-aligned heads produced no degradation**. The heads most aligned with ground-truth
  biology were the *most dispensable*.
- The companion write-up sharpens it: ablating the ground-truth-aligned heads yields **zero** degradation
  while random ablation degrades — evidence that there may be **no clean circuit to find**, and that
  attention alignment is not causal importance.

Read against our pipeline, those studies are what happens when you enter Stage 4 (atlas/attention screening)
without Stages 2–3 having *causally* qualified the behavior and the instrument. The atlas surfaces structure;
structure that has not passed a necessity intervention against a matched null is not a mechanism. Our method
already encodes the fix — Stage 2 requires the behavior survive simple baselines, Stage 3 requires the
instrument reject fakes, Stage 5 requires matched-size nulls, and principle 5 ("controls must be capable of
failure") is precisely the test those single-cell papers ran and reported honestly. The donor–acceptor result
passes it (C3: ablating `{L3H1, L2H15}` removes 81.6% of coupling at 99.2% MLM health, against nulls); the
single-cell "regulatory heads" do not. Same test, opposite verdict — which is the method working.

**Implication for the DNABERT2 port:** the highest-value early deliverable is not an atlas. It is a Stage-2
behavioral qualification with a **negative-capable** design — a native or defensible zero-shot behavior, a
paired biological counterfactual, and a simple-baseline horse-race — followed by a Stage-3 preflight. Only
then does the atlas earn its compute. `TRANSFER_GUIDE.md` Phase 1 already says "stop if behavior is absent or
wholly explained by a simpler baseline"; the single-cell papers are the cautionary exhibit for why that
stopping rule is not optional.

---

## 4. Honest gaps in the formalization

The process is formalized; these are the seams a reviewer would probe, all already visible in the record:

1. **No automated cross-check has been run.** ACDC/EAP is recommended in `METHODOLOGY.md` §4 but not executed;
   until one is, "we didn't just find the circuit we went looking for" rests on hand-built ranking (A12) plus
   the size-curve, not an independent automated search.
2. **The atlas is a SpliceBERT instrument, not yet a generic Stage-4 module.** `RESEARCH_SPINE.md` and
   `ONTOLOGY.md` encode splice-site context tags. Porting the *atlas method* (not its tags) to DNABERT2 needs
   a fresh context ontology for whatever computation DNABERT2 demonstrably performs — the adapter contract
   covers the mechanics, but the biological ontology is per-study by design.
3. **Bottom-up localization is allowed but unexercised.** No SAE/transcoder basis has been used as a Stage-4
   candidate generator here; that is an available upgrade, not a limitation of the framework.
4. **Cross-model comparison is defined but only partially instantiated.** The Caduceus study exercises Phases
   0–1; the comparison axes in `TRANSFER_GUIDE.md` (addressing strategy, redundancy, causal concentration)
   have a full worked template only for SpliceBERT.

## Open method questions (for the owner)

- Run an ACDC or EAP pass on the donor graph before the DNABERT2 port, so the transfer inherits an
  automated-discovery cross-check as a standard stage rather than a recommendation? (Cheapest credibility gain.)
- Add an explicit **Stage 4b — bottom-up localization (SAE/transcoder)** slot to `METHODOLOGY.md`, so the
  behavior-first pipeline can consume feature-level candidates without changing the license logic?
- Promote a one-page **"Stage-2 negative-capable qualification" checklist** into `TRANSFER_GUIDE.md` Phase 1,
  citing the 2026 single-cell FM failures as the rationale, so the DNABERT2 port cannot skip to the atlas?

*Literature characterized here is cited as leads; primary sources are in `docs/REFERENCES.md` for the
in-pipeline anchors (Geiger, Mueller, Conmy, Syed, Zhang & Nanda, Heimersheim & Nanda). The 2026 single-cell
FM interpretability results are external to this repo and should be read in the primary before any is entered
as a citation in a claim ledger.*

## Verified external citations (2026-07-24 pass)

Each DOI below was resolved to its primary record and read for the companion note
[`LITERATURE_BIOFM_INTERP_2026-07-24.md`](LITERATURE_BIOFM_INTERP_2026-07-24.md); none is retracted. These are
method evidence, external to the repo, and are **not** ledger citations until the owner enters them.

- **Kendiukhov 2026** — *Systematic Evaluation of Single-Cell Foundation Model Interpretability Reveals
  Attention Captures Co-Expression Rather Than Unique Regulatory Signal.* arXiv:2602.17532
  (`10.48550/arXiv.2602.17532`), single-author preprint, not peer-reviewed. Exact figures: gene-level
  baselines AUROC **0.81–0.88** vs attention/correlation edges **0.70**; pairwise edge scores add **zero**
  predictive contribution; **causal ablation of regulatory heads produces no degradation**; generalises
  K562→RPE1; CSSI improves GRN recovery up to **1.85×**. The load-bearing exhibit for behavior-first ordering.
- **Kadem & Zheng 2026** — *Interpreting Transformers Through Attention Head Intervention.* arXiv:2601.04398
  (`10.48550/arXiv.2601.04398`), preprint. Survey: visualization→intervention paradigm shift;
  specialisation-vs-redundancy tension.
- **Consens et al. 2025** — *Interpreting Attention Mechanisms in Genomic Transformer Models: A Framework for
  Biological Insights.* bioRxiv `10.1101/2025.06.26.661544`, preprint. Genomic attention-to-biology framework;
  the direction our method requires be causally validated.
- **Rood, Hupalowska & Regev 2024** — *Toward a foundation model of causal cell and tissue biology with a
  Perturbation Cell and Tissue Atlas.* *Cell*, `10.1016/j.cell.2024.07.035`, peer-reviewed. Sets the
  causal-foundation-model standard against which the single-cell audit reads as a gap.
