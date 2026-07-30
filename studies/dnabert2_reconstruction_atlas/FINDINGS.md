# DNABERT-2 atlas — findings of record

> **Tier: T2 — provenance.** The reviewer entry point. Exact live wording is controlled by
> [`CLAIMS.md`](CLAIMS.md); numerical detail is in [`RESULTS.md`](RESULTS.md); rationale, failed
> controls, and corrections are in [`DECISIONS.md`](DECISIONS.md).

**Read this first.** Sixteen claims (D1–D12, D14–D17), 34 infrastructure facts, and 997 result rows
decompose into **six findings**. This file is the whole scientific story in two pages.

---

## The result in one paragraph

A DNABERT-2 promoter classifier reaches a strong, externally replicated decision (+1.89 logit margin
on GENCODE TSSs excluded from its training source), and that decision is carried by a **sparse late
component core** — two attention heads writing into `[CLS]`, on a distributed MLP scaffold, through a
rank-4 subspace. The sequence that behaviourally matters is **concentrated within ±30 nt of the TSS**,
and it is **not carried by nucleotide arrangement**. But the components that carry the decision **do
not attend to where that information is**: they sit at the positional prior their architecture forces,
and comparing base against fine-tuned checkpoints shows they sat there *before* fine-tuning. The one
head fine-tuning moved turned out to address a **fixed window position, not TSS sequence content**.
**No head in this model has been shown to recognise a promoter.** Whatever performs the recognition
must live in the readout — which this study did not examine.

---

## Status at a glance

| # | Finding | Strongest evidence | Interpretation |
|---|---|---|---|
| **1** | The causal core is **positionally addressed, not content-addressed** | Attention + behaviour, two checkpoints, three independent lines | The most novel result, and it is a negative: attention allocation does not track information density, and the one head that changed addresses a window offset. |
| **2** | A **sparse late core** carries the decision on a distributed MLP scaffold | OV/mediation, replicated on sealed splits against matched nulls | The mechanism spine: `L11H2`/`L11H5` → `L11MLP` rank-4 subspace → classifier readout. Not a complete or sparse-faithful circuit. |
| **3** | Behaviourally relevant sequence is **TSS-proximal and not arrangement-borne** | Behaviour, replicated; two refutations | A polarity replicates; the bounded domain, the distal edge, and arrangement as carrier are all refuted. Composition remains open. |
| **4** | The base model shows **no excess reconstruction** of annotated elements | Behaviour, development gate, 13 classes | A powered negative for 5 of 13 classes; 8 underpowered. Confirmation never opened. |
| **5** | The base model **does** show a proximal-context effect for **9 classes** | Behaviour, replicated on sealed splits | A real context effect, but not a motif, a mechanism, or biological function. Promoter is not among the nine. |
| **6** | The fine-tuned classifier **replicates behaviourally; its context mechanism does not** | Behaviour, replicated + one failure | Licenses the classifier as an object of study. Licenses no context-specific mechanism. |

---

## 1. The causal core is positionally addressed, not content-addressed

*Claims D15, D16, D17. This is the study's contribution.*

DNABERT-2 has **symmetric ALiBi and no position embeddings**, and the classifier reads `[CLS]` at
window index 0 — TSS-relative **−249**. Because the widest band is also the one nearest `[CLS]`, the
ALiBi-only prior places **49–89% of every head's `[CLS]` attention mass** in `far_upstream` regardless
of content. Raw attention profiles are therefore uninterpretable; everything below is
*measured minus prior*.

**Information is concentrated near the TSS.** At identical dose, width and contiguity, masking
`−9:+31` costs **+0.3490** more target-specific margin than masking far-upstream windows
(95% CI [+0.1153, +0.5892], Holm `p=0.0058`, n=186) — a **4.13×** per-token density ratio.

**The carriers do not look there.** `L11H2`/`L11H5` sit essentially *at* their positional prior
(excess `far_up` +0.046/+0.032, offset by `special` −0.047/−0.031). The one candidate head that
genuinely selects the TSS neighbourhood, `L9H7`, contributes **~19× less**. Head reach does not
predict causal effect (four tests, nothing qualifies).

**Fine-tuning did not change this.** Across all 144 heads the `[CLS]` geometry correlates at
**0.951** between base and fine-tuned checkpoints, and the carriers were already at the prior *before*
fine-tuning (+0.0256/+0.0141). Exactly one head moved appreciably onto sequence: `L9H7`.

**And that head addresses a position, not a motif.** Three independent lines: the pseudo-TSS control
arm already in the record shows **88–92%** of `L9H7`'s band attention is present in same-locus windows
containing **no TSS**; under circular rotation (content preserved exactly) its attention does **not**
follow the content — prior-corrected `F` excess goes +0.412 → **−0.137…−0.164**, *below* prior, while
positional excess holds at **+0.387…+0.443**; and destroying the band's content in place removes only
**7.2%** of the attention to it.

**Bounds.** `[CLS]`-row attention only — no OV, no mediation, no behaviour for the attention results.
Attention mass is not information transfer. `E_reg` is **regional, not positional** (it cannot separate
position from resident composition) and its magnitude is skew-inflated: median +0.1464 sits *below*
the 0.25 SESOI and the per-locus sign test fails (104/186, `p=0.1234`). "Relay" is an inference from
architecture, not a measurement. Two third-party checkpoints, n=1, lineage assumed.

## 2. A sparse late core carries the decision, on a distributed MLP scaffold

*Claims D5, D6, D7, D10, D11.*

A development-frozen **eight-head** `[CLS]` set replicates jointly in both patch directions
(confirmation necessity **+0.732**, sufficiency **+1.138**) and beats all 63 layer-profile-matched
random sets. Within it, **`L11H2`/`L11H5`** is a layer-specific pair beating all 45 alternative
layer-11 pairs. The pair depends on correct **value-payload alignment** (+0.0615 confirmation) but
shows **no** target-specific key-address dependence (+0.0084, below SESOI). `L11MLP` is the top
accuracy-relevant node in both splits, and a fixed **rank-4** subspace of its 768-dimensional write
captures **84.5%** of the class-margin effect on the confirmation split.

**Bounds.** Faithfulness is the honest limit: the eight heads alone retain **13.7%** of the contrast;
adding all MLP writes reaches **61.3%**. This is a causal priority map on an MLP-maintained
representation, **not** a complete or sparse-faithful circuit. No individual node passes the full
replicated accuracy gate. D11 is post-hoc on reused cohorts.

## 3. Behaviourally relevant sequence is TSS-proximal and not arrangement-borne

*Claims D8, D9, D12, D14.*

Equal-dose masking nominates **+11:+21 nt** (replicated, +0.2056 confirmation), and a
downstream-vs-upstream **polarity** replicates: masking `+1:+31` costs **+0.3262** more than masking
`−39:−9` (Holm `p=0.0096`, n=149). Scrambling bases within the implicated tiles while holding mono-
*and* dinucleotide counts **exactly** fixed leaves the polarity intact — arrangement retains only
**12.9%** (`mask − reorder` +0.4172, Holm `p=0.0066`).

**Two refutations, stated as such.** The *bounded* downstream domain and its distal edge near +31 are
**refuted** (D12 E1/E1b both fail); D8's tile stands alone as a single-tile effect, not a domain.
Arrangement is **refuted** as the carrier.

**Bounds.** **Composition is not established** as the carrier — D14's P2 was declared underpowered
*before* the run (MDE 0.448). No motif, no biological annotation replicated on confirmation. Native
sequence-order necessity failed its development gate (D9) and is bounded to one tile. D14's n=83
three-arm intersection is supply-selected and enriched 1.47× over D12's confirmed effect.

## 4. The base model shows no excess reconstruction of annotated elements

*Claims D1, D2.* Across **13 classes** from GENCODE, EPDnew, PolyASite, ENCODE and Mercer lariat-seq,
**no class** beat every empirical control on the development split. All classes beat order-0/1/2
nucleotide baselines, but empirical advantages ranged −0.150 to +0.020 nats per covered nucleotide.
Five classes met the powered floor; **eight remain underpowered**. Confirmation stayed sealed, so this
licenses **no** biological null — it stops internal localization for those classes.

## 5. The base model does show a proximal-context effect for nine classes

*Claim D3.* Nine classes replicate a positive target-preserving proximal-context effect on the
geometry-qualified alternative margin (confirmation **+0.0125 to +0.0479** nats/nt; every value clears
its SESOI with a positive lower bound and passes Holm). This is a **model-behaviour** effect against a
distal null: it establishes no motif, mechanism, or biological function, and proximity-sensitive
encoding remains part of the licensed effect. **Promoter is not one of the nine** — it qualified in
development and failed to replicate.

## 6. The fine-tuned classifier replicates behaviourally; its context mechanism does not

*Claim D4.* The `prom_300_all` classifier separates GENCODE MANE TSSs from composition-matched
same-locus pseudo-TSSs by **+1.889** logits on confirmation (exact GUE exclusion, Holm `p=0.0002`).
The target-preserving **local-context** difference-in-differences does **not** replicate (+0.2750, Holm
`p=0.0569`). This licenses the classifier as a behaviourally sound object of study; it licenses no
context-specific mechanism, and the failure is why the context-mechanism search stopped.

---

## What this study does not establish

The full list is in [`README.md`](README.md#what-this-study-does-not-establish). The three that matter
most to a reviewer:

1. **No biological promoter element or motif.** Nothing qualified against CAGE, ENCODE cCRE,
   CpG-island, or EPD motif annotation on confirmation.
2. **No source-to-component mediation.** The regional value swap into the causal pair was null at
   **all 30 tiles** (max +0.00124). Nothing links an implicated sequence region to a component.
3. **The readout is uncharacterised** — OV matrices, MLPs, the classification head. Given finding 1,
   this is where promoter recognition must largely live, and it is the largest open gap.

## Corrections of record

Two claims were **retracted or bounded by later work in this study**, and both are preserved rather
than rewritten:

- **D16's "TSS reader"** naming is retracted by D17 (finding 1). The relocation measurement stands;
  the semantic name does not.
- **D17's own description of a defect in its decision rule was itself wrong** — it named raw `F`,
  which the frozen rule never thresholds. The real defect (raw `P`, 27% of which is ALiBi prior) is
  stated in `CLAIMS.md`, and the false version is corrected in place with the reasoning shown.

## Verify

```bash
uv run --extra splicebert --extra dev python scripts/verify_manifest.py
```

997 result rows each resolve to exactly one claim, status, script and provenance manifest.
`uv run --extra splicebert --extra dev pytest -q` → 459 passed, 46 skipped. Frozen protocols are in
[`preregistrations/`](preregistrations/README.md); their known departures are self-reported there and
in `DECISIONS.md` — this arc has **not** been through the independent conformance audit that covers
the SpliceBERT arc.
