# Open / bounded claim triage — what to close, what to explore

> **Tier: T2 — provenance/handoff.** Dated execution/handoff record; not current canonical guidance.

**Tier:** T2 (provenance / decision aid). Date 2026-07-22. Inputs: `studies/splicebert_partner_selection/CLAIMS.md`
(C1–C36, IC1), `studies/splicebert_head_context_atlas/CLAIMS.md` (HCA1–HCA22), and the preregistration
disposition index. This is a decision aid; wording changes are applied in the controlling ledgers, not here.

## Summary

The donor-circuit line (C-series) is in good shape: 30 of its claims are `established`, and its residual
open/bounded items are few and well-defined. The loose ends are concentrated in two places:

1. **The competition branch of the Atlas is a resolved null but is still administratively open** — a
   cluster of `exploratory` claims and two un-retired protocols hang off a premise (HCA4) that failed.
   This is bookkeeping debt, not science, and it is the single biggest source of "loose ends."
2. **Three genuinely-open science questions** remain, only one of which is a clean, self-contained next
   rung: the ≈10.63 carrier (C12). The other two (payload semantics C30, cross-example transduction) are
   larger programs.

Everything else that is `open`/`deferred` is *correctly* deferred because it needs a second checkpoint or
model — do not spend effort there before the transfer milestone.

---

## Bucket A — CLOSE OUT NOW (evidence is terminal; this is bookkeeping)

These need a wording/status change in the ledger, not new experiments.

| Item | Current status | Action | Why it's terminal |
|---|---|---|---|
| **C7** | superseded/bounded | Mark **closed**; redirect all completeness questions to C27/C33. | Completeness was unquantifiable by the broad-ablation null; C27 (F=0.005) and C33 (no knee) *quantified* it. C7 is a stub pointing forward — retire it as a live row. |
| **C10** | bounded | Mark **closed (terminal bound)**. | "Larger-head restorations don't strengthen C9 because matched nulls are inadequate" is a permanent power statement, not a pending question. No experiment closes it further. |
| **C15** | bounded | Mark **closed (terminal sensitivity bound)**. | Splice-junction phasing can't resolve 10.298 vs 10.63 at its coherence length — a stated resolution limit. Already load-bearing as a tripwire ("D30 is a sensitivity limit, not a null"). Nothing to run. |
| **C34 / C35** | gate-qualified (R23/R24) | Mark the **P3 present-day-use question closed**, citing C36. | C36 is the clean nucleotide-invariant confirmation (Ψ_C=+0.141) that C34's gate-qualified confirm lacked. The only remaining P3 question is *origin*, which is deferred (Bucket C). Stamp the current-use family done. |
| **HCA4** | null | Mark **closed (powered null)**. | Competition specificity is refuted: \|M_comp\|=0.088 vs matched decoy \|M_decoy\|=0.135. A clean negative result. |
| **HCA5** | withdrawn / invalid estimand | Confirm **closed**. | Rank-concentration ≠ faithfulness; pooled acceptor rows used the wrong coordinate. Replaced by HCA10. |
| **HCA6** | superseded (not licensed) | Confirm **closed**. | Confirmatory competition run was never licensed because its gate (HCA4) failed. Credits correctly not spent. |
| **HCA2, HCA3, HCA7, HCA8, HCA9, HCA10** | exploratory | **Retire as a closed line** ("competition branch, closed at behavioral specificity"). | These are the interior of a branch whose specificity premise failed. HCA9 is worth *keeping* explicitly because it rediscovered donor-circuit L3H1 as a positive control — fold that one useful nugget into the donor-circuit record and close the rest. |
| **IC2** (parked) | v5.3 design complete, run not opened | **Retire** (do not run). | It is the confirmatory acceptor-competition run for the branch HCA4 refuted. Running it would confirm a behavior we've shown isn't competition-specific. A parked run for a dead premise is a loose end — cut it. |
| **IC2A** (deferred draft) | draft only | **Retire** (do not pursue). | Same branch; draft only, no outcome claimable. |
| **IC1** | outcome-aware deterministic verification | Leave as-is (already correctly labeled non-confirmatory). | Not a loose end; it is honestly scoped. No action. |

**Net effect of Bucket A:** the competition sub-study collapses from ~10 dangling rows + 2 parked protocols
into one closed-line entry plus one salvaged positive control (HCA9→donor circuit). The C-series sheds three
stub/terminal-bound rows. Nothing is deleted — closed rows stay in the ledger with a terminal status, per the
"never delete a correction" rule.

---

## Bucket B — GENUINELY OPEN, worth doing before moving on (ranked by leverage / effort)

### B1. ≈10.63 carrier: push past attention (C12) — DO THIS FIRST
- **State:** established at post-softmax attention in 8 heads (C11); OV content, behavioral function, and
  biological origin are `open` (C12). This is the one open claim that is a clean, self-contained next rung on
  an *already-built* ladder.
- **Not yet done:** OV/mediation decomposition of the 8 reader heads, then a qualified behavioral test
  (does deleting/patching the carrier move any model-native output?).
- **Why first:** lowest effort, highest ratio of "closes a stated open claim" to "new machinery." The
  instrument (post-softmax scan, matched controls, add-one permutation) already exists. Two outcomes are both
  publishable: it writes something downstream (promotes C12 up the ladder) or it is a read-only positional
  code with no downstream effect (a clean bounded result, which is itself interesting).
- **Guard:** keep the origin question (10.63 biological correlate) *out* of this — that is Bucket C
  (training-time evidence). B1 is only the OV→behavior rungs.

### B2. Payload semantics (C30) — the headline limitation
- **State:** bounded (R22). Payload is acceptor-local, not bulk composition; "splice-site strength"
  withdrawn. Semantic identity unresolved. This is *the* stated main limitation of the whole donor circuit.
- **Not yet done at the right design:** the site-choice steering program (ROADMAP Track 2) — learn candidate
  directions on paired site-choice contrasts (not absolute-strength labels), show a direction causally and
  dose-responsively reallocates between matched sites, *then* ask whether an external biological variable
  predicts the dose response.
- **Why second:** higher effort and higher risk than B1, but it is the difference between "we traced how the
  signal moves" and "we know what the signal is." A reviewer will flag its absence but the paper can ship with
  it scoped out (it currently is). Do B2 if you want to close the headline gap rather than defer it.
- **Guard:** do not let "splice-site strength" back in without external validation. Existence → specificity →
  semantics, in that order.

### B3. Cross-example / cross-context transduction (the generalization gap)
- **State:** not a claim yet; flagged as the first missing bridge in MECHANISM and ROADMAP Track 1. Almost all
  donor-graph causal evidence is intra-example clean/corrupt patching on canonical short introns.
- **Not yet done:** derive `U_info` (or a site-choice direction) on discovery genes, inject into gene-disjoint
  matched targets, require dose-ordered target-margin movement above matched-random and within-example
  controls. The Atlas matched cohorts (HCA11/HCA19) are the ready substrate; this is what connects the
  validated Atlas components to the donor circuit.
- **Why third by effort but arguably first by review-risk:** it is the single most likely rejection reason for
  Result 1 (see the review doc). If the goal is "publication-ready," a *modest* version of B3 de-risks the
  paper more than anything else. If the goal is "close the ledger's stated opens," B1 is cleaner.

### B4. The L3H1/carrier → L2MLP bridge (HCA20 residual) — narrow, designed, not run
- **State:** HCA20 says L2MLP's partner-conditioned B relation is material but its upstream attention route is
  **instrument-limited** (the frozen same-B validated attention set is empty); the A-estimand QK/V is
  non-identifiable at population scale. HCA20 explicitly calls for "a new frozen cross-estimand intervention
  to test an L3H1/carrier→L2MLP bridge."
- **Why noted but not urgent:** this is a specific, already-scoped follow-up, but it is inside the Atlas
  mechanism layer and depends on a new frozen design. Do it only if the Atlas becomes a headline; otherwise
  it rides along with B3 (both are about connecting Atlas components to the donor circuit).

### B5. The 20 context-associated Atlas signals (HCA18) — hypotheses, need power
- **State:** 20 matched context signals preserved but *none* validation-ready — every context estimate failed
  the frozen precision gate (MDE80≤0.05; observed 0.090–0.299). Correctly not called validated.
- **Not yet done:** a fresh, better-powered context design (more matched pairs per context, or a coarser
  context taxonomy) before any exact mechanism validation.
- **Why lowest priority:** these are leads, not loose ends. Do not erase them; do not promote them. Revisit
  only if a context-specificity story becomes a target. A power calculation up front (what N clears MDE80≤0.05)
  would tell you whether this is even feasible before committing.

---

## Bucket C — CORRECTLY DEFERRED (do NOT work these before the transfer milestone)

These are `open`/`deferred` for a principled reason — a frozen checkpoint cannot answer them. Attempting them
now would be wasted effort.

| Item | Why it must wait |
|---|---|
| **Period-3 carrier origin** (P3_4 protocol, not frozen) | "Why did training create it" needs training-time evidence: checkpoint trajectories, controlled retraining, corpus/objective ablation, or cross-model comparison. C36 closed present-day *use*, not origin. |
| **≈10.63 carrier biological origin** (part of C12) | Same class: origin is a training/corpus question. B1 addresses only its OV→behavior rungs; its biological correlate stays deferred. |
| **Caduceus transfer** | Deferred by design; the CUDA/Colab package is dry-run validated but the model experiment is not run. This is the *next study*, not a loose end of this one. |

**Guidance:** bundle all three under the "second real GFM / training-time evidence" milestone. They are the
reason to move on, not things to finish before moving on.

---

## Bucket D — DONE INFRASTRUCTURE (not open; consider de-cluttering the ledger)

HCA11–HCA17, HCA21, HCA22 are all `supported` **instrument / deployment / visualization-handoff** claims —
they certify that measurement and export plumbing works, not scientific findings. They are not loose ends and
need no work. But nine "supported instrument" rows in a claim ledger dilute the signal. Consider moving them to
an `INSTRUMENT_LOG.md` (T2) and leaving the science claims (HCA1, HCA18–HCA20) in the Atlas `CLAIMS.md`. This is
optional hygiene, not a scientific decision.

---

## The short answer

- **Get rid of / close now (bookkeeping, ~half a day):** C7, C10, C15 (terminal); the entire competition
  branch — HCA2/3/4/5/6/7/8/9/10 collapsed to one closed-line entry, salvaging HCA9 as a donor-circuit
  positive control; and **retire the parked IC2 and deferred IC2A** protocols (dead premise). Stamp the P3
  present-day-use family (C34/C35/C36) closed.
- **Explore before moving on (pick by goal):** **B1 (≈10.63 OV→behavior, C12)** if the goal is to close a
  stated open claim cleanly; **B3 (transduction)** if the goal is a publication-ready donor-circuit paper;
  **B2 (payload semantics)** if you want to close the headline limitation. B4/B5 ride along with the Atlas.
- **Do not touch yet:** all origin questions (P3 and 10.63) and Caduceus — they belong to the transfer
  milestone and need evidence a frozen checkpoint cannot provide.
