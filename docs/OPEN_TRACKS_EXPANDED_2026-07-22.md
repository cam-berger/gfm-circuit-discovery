# Open tracks, expanded — mechanistic-interpretability method + biological grounding

**Tier:** T2 (decision aid / provenance). Date 2026-07-22. Companion to `CLAIM_TRIAGE_2026-07-22.md`.
This expands the five open tracks (B1–B5) with (a) the exact mech-interp intervention each needs and how it
extends the instrument you already have, and (b) the biological phenomenon each connects to. External
references were verified against public records during writing; confirm exact scope against primary sources
before citing in a manuscript.

Two framing tools recur, so state them once:

- **The QK/OV factorization** (Elhage et al., "A Mathematical Framework for Transformer Circuits",
  *Transformer Circuits Thread*, 2021). A head's behavior splits into two independent circuits: the
  **QK circuit** `W_QK = W_Q W_Kᵀ`, which decides *where* the head attends (its attention pattern), and the
  **OV circuit** `W_OV = W_V W_O`, which decides *what it writes* into the residual stream once it attends.
  Your whole evidence ladder is essentially this split made empirical: "reaches attention" = the QK/softmax
  rungs; "OV/mediation" = the OV rung. This is why "stops at attention" (C11/C12) is a precise, principled
  boundary, not a vague one — you have characterized the carrier's QK behavior and have simply not yet
  measured its OV.
- **Necessity vs. sufficiency are not symmetric** (Heimersheim & Nanda, "How to use and interpret activation
  patching", arXiv:2404.15255, 2024; Zhang & Nanda, ICLR 2024). Noising a component (corrupt→clean removed)
  tests necessity; denoising (clean→corrupt restored) tests sufficiency. They can disagree, and the metric +
  corruption choice changes the answer. You already respect this (your reconstruct–inject factorial, your
  size-matched nulls); the tracks below should keep both arms.

---

## B1 — Push the ≈10.63 carrier from attention to OV → behavior (closes C12)

### The claim and the exact gap
C11 establishes the ≈10.63 carrier at **post-softmax attention** in 8 heads (`L2H12, L3H3, L3H5, L3H8,
L3H9, L3H11, L3H13, L4H13`), with carrier deletion dropping coherence to 40–44% while reciprocal / low-leak
/ content-shuffle controls retain 89–107%. C12 says the **OV content, behavioral function, and origin** are
unknown. In QK/OV terms: you have the carrier's QK story cold and its OV story blank.

### The mech-interp method (extends instruments you already have)
1. **OV decomposition of the 8 reader heads.** For each carrier-reading head, ask what it *writes* when it
   attends along the carrier. Decompose `W_OV` in the carrier's rank-2 plane and project the head's output
   contribution onto the final-residual readout — the same machinery you used to build `U_info` for the
   donor circuit (C16), but anchored on the carrier plane instead of the clean−corrupt donor direction.
   Output: does any carrier-reader write a low-rank, structured object, or does it write noise?
2. **Carrier-conditioned mediation / dose–response.** You already have the carrier-deletion and dose
   instruments (`carrier_dose`, `carrier_surgery`) from the period-3 work. Re-point them at the ≈10.63
   plane: delete/attenuate the carrier and measure a *model-native* output (MLM margin at various masked
   positions), with the same matched controls (reciprocal band, low-leak, content-shuffle) that made C11
   credible. This is the necessity arm.
3. **A qualified behavioral test.** The hard part is that, unlike the donor circuit, there is no
   pre-specified behavior for this carrier — so preregister *which* native output you expect it to move and
   why, before looking. If none moves under a powered test, that is the sufficiency-negative result.

### Why it is the cleanest next rung
Every instrument (post-softmax scan, rank-2 plane geometry, add-one permutation with the 1/401 floor,
matched deletion controls) already exists and is validated. You are moving one column right on a ladder you
built, not building a new ladder. Both outcomes are publishable: a downstream OV write **promotes C12** to
OV/mediation (a second causal positional mechanism alongside period-3); a powered *no-downstream-effect*
result is a clean bounded finding — a positional feature the model represents and routes but does not use,
which is itself an interesting negative and a natural contrast to the period-3 clock that *is* used.

### The biology (kept honest)
Do **not** attach the carrier to a biological phenomenon in B1. The obvious candidate — the ~10.2–10.5 bp
DNA helical / nucleosomal periodicity — is already **refuted** by C14 (the real nucleosomal signal sits at
≈10.298 nt, 0.33 nt from the model's ≈10.63, and the model is fit to ±0.005: a powered distinction, not a
match). So B1's biology is deliberately empty: its scope is OV→behavior *inside the model*. The biological
origin of a 10.63-nt period the model manufactured that is neither codon (period-3) nor nucleosomal
(≈10.298) is a genuine open question, but it belongs to the training-time program (Bucket C), because a
frozen checkpoint cannot answer "why did training build this."

### Effort / risk
Lowest effort of the five; low risk. The main design cost is the preregistered behavioral hypothesis in
step 3.

---

## B2 — Payload semantics (closes the headline limitation C30)

### The claim and why it is the headline gap
The donor circuit's payload is **acceptor-local** (C24) and **not** explained by preregistered bulk
composition — GC / polypyrimidine-tract / k-mer / positional covariates (C30). The earlier "splice-site
strength" label was **withdrawn (R22)** because absolute local MaxEnt strength was unstable across
independently seeded draws and not significant under two-way gene-dyadic inference. So you have proven what
the payload *is not*, and localized *where* it lives, but its positive semantic identity is open. This is
the single biggest "what is the model actually computing" gap in the paper.

### The biology — what "splice-site strength" means and why it was the natural (wrong) label
The 3′ (acceptor) splice site in metazoans is a ~40-nt composite signal: the branch point, the
polypyrimidine tract, and the terminal `AG`, recognized by U2 snRNP and its associated factors. **MaxEntScan**
(Yeo & Burge, "Maximum entropy modeling of short sequence motifs with applications to RNA splicing signals",
*J. Comput. Biol.* 11(2–3):377–394, 2004) scores how well a candidate site matches the maximum-entropy model
of real donor/acceptor motifs — the field-standard proxy for "splice-site strength." It was the natural
hypothesis for an acceptor-local, partner-specific payload. R22 is exactly right to withdraw it: an
acceptor-local latent that *correlates* with MaxEnt is not the same as the model *representing* MaxEnt, and
the sample instability shows the correlation is not even robust. The candidates that remain — conditional
competitiveness, a learned site-quality variable, or another local latent — are all biologically plausible
and all currently unnamed.

### The mech-interp method (a new design, not a re-run — this is why C30 is still open)
The right instrument is **causal steering with paired contrasts**, not another correlational strength probe.
This is the activation-steering / concept-direction paradigm (Turner et al., "Steering Language Models With
Activation Engineering" / ActAdd, arXiv:2308.10248, 2023; Rimsky/Panickssery et al., "Steering Llama 2 via
Contrastive Activation Addition" / CAA, arXiv:2312.06681, ACL 2024; Zou et al., "Representation Engineering:
A Top-Down Approach to AI Transparency", arXiv:2310.01405, 2023), where a candidate concept direction is
estimated as a **difference-in-means between contrastive prompt
sets** (the difference-in-mean-probe direction of Marks & Tegmark, "The Geometry of Truth", arXiv:2310.06824,
2023) and then *added* to the residual stream to test whether it causally moves behavior.

Adapted to splicing (this is ROADMAP Track 2, and it has not been run at this design):
1. **Learn candidate directions on paired site-choice contrasts**, not absolute-strength labels — e.g.
   acceptor A-preferred vs. acceptor B-preferred for a shared donor, matched on distance/composition. The
   difference-in-means direction over these pairs is the candidate payload axis.
2. **Existence first:** inject the direction (dose-scaled) at the acceptor-local site and require a
   dose-ordered change in the donor's native margin, above matched-random-direction and scrambled-source
   controls, *without* generic MLM damage (a health channel you already gate on).
3. **Specificity next:** is the causal effect partner-specific (moves the intended paired site, not a decoy)?
4. **Semantics last:** only after existence + specificity, ask whether an *external* biological variable
   (MaxEnt on the paired site, competitor MaxEnt, relative strength, usage frequency) predicts the dose
   response. Label the direction no more strongly than that external validation supports.

### The critical caveat (why this is higher-risk than B1)
Steering directions are **not automatically identifiable**: a direction that moves behavior is not
necessarily *the* concept direction, and recent work formalizes exactly this non-identifiability of steering
vectors. This is the same trap as R20 (L3H1's attention preference looked like partner *identity* but was
position geometry). So B2 must keep the existence→specificity→semantics firewall and never promote a
behavior-moving direction to a named biological variable without the external predictor in step 4. Done
loosely, it re-earns a retraction; done in this order, it closes the headline gap.

### Effort / risk
Higher effort and the highest interpretive risk of the tracks, because "name the payload" is precisely where
the field's methods are weakest. The paper can ship with C30 scoped out (it currently is), so B2 is
"close the main limitation" work, not "make it publishable" work.

---

## B3 — Cross-example / cross-context transduction (the generalization gap)

### The claim status and why it dominates review-risk
This is not yet a claim; it is the **first missing bridge** named in MECHANISM and ROADMAP Track 1. The
problem is structural to the method: nearly all donor-graph causal evidence is **intra-example
clean/corrupt patching** on canonical short introns. Activation patching, by construction, licenses a
statement about the prompt distribution it is run on — it demonstrates mediation *on that example pair*, not
that the internal state is a transferable, interventionally general object. A reviewer familiar with the
patching-best-practices literature (Zhang & Nanda 2024; Heimersheim & Nanda 2024) will ask this immediately,
which is why it is the single most likely rejection reason for Result 1.

### The mech-interp method
Promote `U_info` from a *within-example mediator* to a *transferable direction* via a transplant design:
1. **Derive** `U_info` (or a site-choice direction) on **discovery genes only**.
2. **Inject** it into **gene-disjoint target** examples, matched on site type, distance, native
   margin/strength, coding status, sequence composition, and intervention norm.
3. **Require** dose-ordered target-margin movement above (a) within-example, (b) matched-random-direction,
   (c) scrambled-source, and (d) non-target-position controls.
4. **Use two-way / dyadic inference** whenever sources or targets are reused (as you already do for the
   donor signature and payload semantics), so the effect is not driven by a few reused items.

Conceptually this is the same operation as B2's steering, but the *payload* being transplanted is the
already-validated `U_info` object (C16/C26), not a newly-learned semantic direction — which makes B3
lower-interpretive-risk than B2 even though it is more infrastructure. It is also directly supported by the
**causal-abstraction** view (Geiger et al., arXiv:2301.04709, 2023 / JMLR 2025): a genuine causal variable
should support interchange interventions *across* inputs, not just within a matched pair. B3 is the
cross-input interchange test that the intra-example patching has so far skipped.

### The biology
The transplant cohorts are exactly the Atlas matched cohorts (HCA11/HCA19) — single-pair, clustered,
multi-pair, cassette/complex, matched-control — so B3 is what finally *connects the validated Atlas
components to the donor circuit*. Biologically, success would show the model carries a portable
"this donor's partner is intact" signal that generalizes across gene contexts, consistent with a general
intron/exon-definition competence rather than a per-gene lookup. Keep the scope honest: it remains a model
mechanism, not a claim about spliceosome biochemistry.

### Effort / risk
Highest infrastructure of the "worth doing" set, but **lowest interpretive risk** because it transplants an
already-characterized object. If the goal is a publication-ready donor-circuit paper, a *modest* B3 (one or
two context classes) de-risks it more than anything else here.

---

## B4 — The L3H1 / carrier → L2MLP bridge (HCA20 residual)

### The claim status
HCA20 established that the three validated general A-heads (L2H15, L3H1, L1H12) have specificity-controlled
downstream MLP recovery paths, and that L2MLP's partner-conditioned B relation is material (donor
+0.344 [0.309,0.379]; acceptor +0.320 [0.286,0.354]) — **but its upstream attention route is
instrument-limited** because the frozen same-B validated attention set is empty, and the A-estimand QK/V is
**non-identifiable at population scale**. HCA20 itself calls for "a new frozen cross-estimand intervention to
test an L3H1/carrier → L2MLP bridge."

### The mech-interp method
This is a targeted **path-patching** design (edge/path intervention distinguishing a direct route from
redundant ones — the ACDC/edge-attribution family, Conmy et al., NeurIPS 2023): freeze a new cross-estimand
intervention that patches the L3H1 (address) / period-3 carrier output specifically into L2MLP's input and
measures the partner-conditioned B effect, with the A-estimand held clean. It is narrow and already scoped;
the blocker is that it needs a *new frozen design* because the existing validated attention set can't
identify the route.

### Why it is noted but not urgent
It lives inside the Atlas mechanism layer and only matters if the Atlas becomes a headline. Otherwise it
rides along with B3 — both are about wiring Atlas components into the donor circuit. Biologically it would
connect the positional addressing machinery (period-3 clock + L3H1 address) to the payload-writing MLP,
tightening the "address × payload" story (C23) at the Atlas scale.

### Effort / risk
Medium; well-scoped but needs a new preregistered intervention. Do it only alongside a decision to promote
the Atlas.

---

## B5 — The 20 context-associated Atlas signals (HCA18)

### The claim status
The discovery atlas preserved **20 matched context-associated signals** (17 material-suppress, 3
material-support after BH and cross-half stability), concentrated in acceptor clusters (11), donor clusters
(6), and donor-side cassette exons (3). **None is validation-ready**: every context estimate failed the
frozen precision gate (`MDE80 ≤ 0.05`; observed 0.090–0.299). They are correctly held as hypotheses, not
validated specificity.

### What is actually needed (and why it is lowest priority)
These are **leads, not loose ends** — nothing is broken, and the discipline (not promoting them) is working.
What they need is **power**, not a new method: a fresh, better-powered context design (more matched pairs per
context, or a coarser context taxonomy that pools related contexts) so a context estimate can clear
`MDE80 ≤ 0.05`. The single most useful cheap step is a **power calculation up front** — compute the N of
matched pairs per context required to reach MDE80 ≤ 0.05 given the observed effect spread (0.090–0.299), and
decide from that whether a better-powered design is even feasible before committing compute. If the required
N is unreachable within the 1-kb-window cohort (recall only ~10% of competition targets fit a 1024-nt
window, HCA1), that is itself the answer.

### The biology
The context taxonomy (single-pair / clustered / multi-pair / cassette) maps onto real alternative-splicing
architecture — cassette exons, alternative 5′/3′ sites, and their well-documented dependence on flanking
**intron length** and exon/intron definition mode. The three cassette-exon donor-side signals are the most
biologically loaded, since cassette exons are the canonical alternative-splicing unit. But none is licensed
until powered.

### Effort / risk
Lowest priority. Do the power calc; revisit only if a context-specificity story becomes a target. Do not
erase the 20 signals; do not promote them.

---

## The biological through-line (why the short-intron scope is a strength, not a limitation)

One point ties B1–B5 together and is currently under-stated in the repo. The donor circuit is scoped to
**canonical internal short introns (60–150 nt)**. In splicing biology, intron length is the primary switch
between the two splice-site recognition modes: **intron definition** (the spliceosome assembles across the
intron, pairing the two splice sites directly) dominates for short central introns, and **exon definition**
(assembly across the exon) takes over above roughly a 200–250 nt intron-length threshold (Fox-Walsh et al.,
2005, and the length-constraint literature it anchors). SpliceBERT's short-intron donor–acceptor coupling
therefore sits squarely in the **intron-definition regime** — the biologically correct frame for a mechanism
where a donor is scored by its *directly paired* acceptor across the intron. That is the tightest, most
defensible biological bridge in the whole project, and every track above either strengthens it (B3: does the
paired-partner signal generalize?) or sharpens what the paired signal *is* (B2: what acceptor-local quantity;
B1: an orthogonal positional carrier that is *not* part of it). Stating the intron-definition framing
explicitly turns the "short-intron scope" caveat into a positive claim about which biological mechanism the
model has learned to implement.

## References (verify against primary sources before manuscript use)

- Elhage et al. **A Mathematical Framework for Transformer Circuits.** *Transformer Circuits Thread*, 2021.
  (QK/OV factorization; induction heads follow-up: Olsson et al., 2022.)
- Heimersheim & Nanda. **How to use and interpret activation patching.** arXiv:2404.15255, 2024.
- Zhang & Nanda. **Towards Best Practices of Activation Patching in Language Models.** ICLR 2024.
- Conmy et al. **Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC).** NeurIPS 2023.
- Geiger et al. **Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability.**
  arXiv:2301.04709, 2023 / *JMLR* 26, 2025.
- Turner et al. **Steering Language Models With Activation Engineering (ActAdd).** arXiv:2308.10248, 2023.
  Rimsky/Panickssery et al. **Steering Llama 2 via Contrastive Activation Addition (CAA).** arXiv:2312.06681,
  ACL 2024. Zou et al. **Representation Engineering: A Top-Down Approach to AI Transparency.** arXiv:2310.01405,
  2023. Marks & Tegmark. **The Geometry of Truth**
  (difference-in-mean probes). arXiv:2310.06824, 2023. (Also the steering-vector *non-identifiability*
  caution in the 2025–2026 follow-up literature — e.g. "On the Non-Identifiability of Steering Vectors in
  LLMs".)
- Yeo & Burge. **Maximum entropy modeling of short sequence motifs with applications to RNA splicing
  signals (MaxEntScan).** *J. Comput. Biol.* 2004 (verify volume/pages and PMID against the primary record).
- Fox-Walsh et al., 2005 (intron- vs exon-definition length threshold ≈200–250 nt); see also Berget 1995;
  Sterner et al. 1996; De Conti et al. 2013 for the definition-mode framework.
