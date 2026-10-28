# Roadmap

What is next, what is deferred, and which owner decisions are still open. This page ranks work; it is
not evidence for any claim. Claims are in the study ledgers (for the donor arc,
[`CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md)) and retractions in
[`CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md). Re-read both before acting on
anything here.

## Where things stand

The aim is a transferable, evidence-gated practice for causal mechanistic discovery in genomic foundation
models (GFMs), with cross-model comparisons to tell recurring computational motifs apart from solutions
tied to one architecture, objective, corpus or checkpoint.

In SpliceBERT, donor–acceptor coupling has behaviour-level causal evidence, an A0–A12 account in terms of
address, payload and delivery, a period-3 routing carrier (C34, C35), and an audited Head–Context Atlas.
The structured core is real but not sparse-faithful: the faithfulness curve has no mid-size knee. The
separate multispecies ≈10.63 carrier stops at post-softmax attention.

## Main next questions

1. Does the donor-circuit information state carry over to new genes and contexts?
   ([Track 1](#track-1-cross-context-transduction))
2. What acceptor-local quantity does the payload carry, and can it steer which site a donor favours?
   The first preregistered attempt stopped at its preflight.
   ([Track 2](#track-2-payload-semantics-and-site-choice-steering))
3. Would an adaptive search find a smaller faithful circuit than the tested ranking? The live draft is
   `PREREGISTRATION_A13_ADAPTIVE_MINIMALITY.md` (node level), which has not run. The acceptor study's
   edge-level `PREREGISTRATION_R4_EDGE_LEVEL_CAUSAL_CORE.md` was never frozen and is retired; its
   exploratory run used a signed objective that rewarded over-recovery, and no R4 result may be used.
   ([Method coverage](#method-coverage))
4. Does the DNABERT-2 result hold with multiple seeds and committed provenance?
   ([Second model](#second-model-dnabert-2))

## Track 1: cross-context transduction

Does the donor-circuit state generalize across genes and contexts, or does patching only work within a
matched clean/corrupt example? The approach is to derive `U_info`, or a candidate site-choice direction,
on discovery genes only, then transplant it into gene-disjoint targets from Atlas-tagged cohorts on
GENCODE v50 comprehensive, matched for site type, distance, native margin/strength, coding status,
sequence composition and intervention norm. The effect must be dose-ordered and beat within-example,
matched-random-direction, scrambled-source and non-target-position controls, with two-way (dyadic)
inference whenever sources or targets are reused.

The first frozen attempt,
[`PREREGISTRATION_G1_CROSS_EXAMPLE_TRANSDUCTION.md`](../studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_G1_CROSS_EXAMPLE_TRANSDUCTION.md),
found that the final donor residual transfers as a generic readout axis (C41, verification level). On
the acceptor side, transfer is strong but content specificity is unresolved (C38). Earlier-layer,
non-additive and wider-context specificity remain open. The track ends when an independently derived
direction predictably moves the intended native output in a new gene or context under qualified health
and specificity controls, or when a null or instrument-limited result is reported at a calibrated
detection floor.

## Track 2: payload semantics and site-choice steering

What acceptor-local latent is carried, and can it steer which site a donor favours? MaxEnt, motif
scores, attention, attribution and SVD directions are candidate generators only. Directions should be
learned on discovery genes from paired site-choice contrasts, not absolute strength labels alone, and tested in
order for existence, partner specificity, and semantics (external variables predict the dose response),
with matched competitor and decoy edits, gene-disjoint splits, multi-draw stability and dyadic inference.
The track ends when a direction causes predictable, dose-responsive reallocation between matched sites
without generic MLM damage, under a semantic label no stronger than the external validation supports.

The semantic question itself is closed-bounded. R22 withdrew absolute external strength (MaxEnt) as the
payload's identity, because that estimand proved sample-unstable and non-significant under dyadic
inference; the payload is acceptor-local and not explained by bulk composition (C30). The steering
protocol `PREREGISTRATION_B2_PAYLOAD_STEERING.md` did not survive red-team review.

The first preregistered attempt since then was BG1/BG2, frozen on 2026-08-10 in
[`PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md`](../studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md).
It used an eight-cell, AG-preserving panel of synthetic within-gene edits at biological −3 and +1 (BG1,
behaviour), followed conditionally by gene-disjoint L3H1 acceptor-window value recovery (BG2, route). On
2026-08-11 the train-split preflight returned `PASS=false`: planted sensitivity, route dose sensitivity,
route resolution and target resolution all missed their frozen thresholds. Neither BG1 validation nor BG2
was opened, the exit is `INSTRUMENT_UNRESOLVED`, and no boundary-grammar claim exists. Development-only
directional estimates are candidate generators and cannot authorize BG2. Even a pass would not have named
the payload or reopened C30, which can move only through an R-number. The rationale and three open
questions are in [`DECISIONS_A0.md`](../studies/splicebert_partner_selection/DECISIONS_A0.md) under
"BG1/BG2 — the acceptor boundary contrast".

## Other open problems

Two further gaps are known. Adaptive minimality is open: the no-knee curve explains why a small injected
core can be sufficient without being a self-contained generator. The period-3 carrier's origin is also
open. Its present-day use is causal and C39/C40 associate it with multi-species pretraining, but coding
specificity and training dynamics are unresolved, and the sibling comparison cannot separate species
diversity from corpus size, composition, sampling or optimization history.

Two problems were set aside on 2026-07-25 and are not active tracks.

The ≈10.63 carrier's write onto the donor readout is open and cannot be evaluated as preregistered (R36).
The B1 v2 OV endpoint (`S_ov`, the `U_info`-projected `suff_recovery` and the `Ω` subspace overlap, each
against a covariance-matched null with a `PP-ovplant` gate) was never computed. Every per-head branch
required an L0 reader, because a deeper reader's value-restore absorbs upstream positional propagation
(red-team must-fix #7), and no L0 head passed the reader screen (9 scanned, 0 passed; `R_1024` is
L2/L3/L4 only). A carrier-specific write that is behaviourally redundant for coupling is therefore
neither excluded nor excludable here. Closing it needs either an owner-authorized amendment admitting
deep readers, which re-admits the confound, or a reader set with a qualifying L0 head. Do not just run
`S_ov` on `R_1024`; that computes the confounded quantity the design excluded on purpose. C12 is
terminal for the general-OV and behavioural branches: the readers' values are sub-SESOI on the donor
coupling margin, deletion has a small suppressive coupling effect with equivalence underpowered, and a
diagnostic gives a bounded null on the general MLM output (C12, C37). Its training-condition
association is measured (C39): strong in the 72-species sibling, undetected in the human-only sibling, and
not detectable in the human genome's dinucleotide spectrum above that assay's 5–10% modulation floor,
which is a sensitivity limit and not an established absence (R16, C15).

No human dyad-anchored nucleosome measurement exists in this record (R30). The nucleosomal comparator,
≈10.298 nt, sits 0.33 nt from the ≈10.63 carrier, a powered refutation, but it was measured in mouse
ES-cell chromatin (Voong 2016, mm9), like every dyad script in the tree. The cross-species step is an
assumption, so name the organism (`docs/NOMENCLATURE.md`, standing phrase SP2). A fallback argument, that
10.630 lies outside the 10.2–10.5 bp DNA-helical range, is contested
([B4](#b4-the-dna-helical-range-behind-r30)); until B4 is ruled on, rely on the 0.33-nt comparison alone.
Closing the gap needs an external human dyad map on hg38, not model compute. A human MNase set is
obtainable, but MNase's AT-cleavage bias confounds a WW-periodicity readout and needs its own bias
control; no comparable human chemical-cleavage map was found. The gap is a disclosed limitation, and the
refutation does not depend on closing it.

## Exon-edge compatibility (archived 2026-08-04)

This arc asked whether coupling tracks coding-boundary phase across a joined boundary. It reached a
replicated behaviour (C43) and a stopped mechanism attempt (C44); its other nine claim rows were archived
rather than carried. The bulk is on `codex/intron-exon-phase` at `fd26cc1b`, which exists only in the
owner's local clone and was never pushed, and R45 governs what may be said about it. Open, in priority order:

1. Does C43 hold outside the chimeric lattice? Every construct joins a donor module from one gene to an
   acceptor module from another at one fixed 120-nt span. A native-context assay, or the same lattice at
   two more separations, is the cheapest test.
2. Is it stop-avoidance or coding-boundary phase? Mismatched joins carry ~0.61 more in-frame stop codons.
   Diagnostics argue against stop load as the driver (`ρ = −0.093`; high-load blocks show a lower `Ψ`),
   but nothing yet manipulates stop load independently.
3. A capable mechanism instrument. C44 closes nothing: four authorization gates failed and the target
   was never exposed. A successor needs a newly preregistered instrument; the failed path may not be
   reopened or relaxed. The reserved 384-block target is the C43 replication cohort, so it carries a known
   behavioural covariate.
4. E4's lead. A post-layer-2 acceptor plane decodes coding-boundary phase at 0.894 held-out (0.973
   exact-distance-matched), but its specificity gate failed (16/99 nulls at or above target). A successor
   needs an exchangeable null with mixed strata and fresh outcomes. The audit is re-runnable from that commit
   (1.96 MB, no genome, no checkpoint).

The arc closes when C43 reproduces outside a fixed-geometry chimera or is recorded as a construct-scoped
behaviour. No mechanism claim can be made before item 3. The sanctioned successor for items 2 and 3 is
[positional boundary coherence](../studies/splicebert_positional_boundary_coherence/README.md). Its Phase I
stopped instrument-unresolved on 2026-08-11, before model access: the frozen 120/600/384-block design
needs 1,104 unique genes in every source slot, and only 809 donor-phase-2 and 854 acceptor-phase-2 genes
exist (capacity certificate `PBC1`). That is not evidence for or against the boundary preference, and
every later phase is unreached and sealed.

## Second model: DNABERT-2

The transfer milestone is met on its stated terms, which count a qualified null as success. The
[DNABERT-2 promoter study](../studies/dnabert_promoter/README.md) (`DBP1`–`DBP10`, landed 2026-08-11)
reached a frozen representation (`DBP3`) and a test-confirmed five-node conditional causal core
(`DBP6`), then three limits: no knee in the held-out faithfulness curve (`DBP7`), class-level-only
transplant transfer (`DBP8`), and a causal null on the TATA bridge at n=71 (`DBP10`). A second model,
in a bounded single-seed study, also shows a structured causal core inside distributed computation.

Three items in its [`DECISIONS.md`](../studies/dnabert_promoter/DECISIONS.md) stand between this and more
than a first pass. First, commit the cleaning provenance: the splits are leakage-controlled (326 rows
removed, disjointness asserted), but `shared/clean/cleaning_report.json` and the cleaned hashes are not
committed, and exact-hash matching gives no near-duplicate screen at 70 nt (`DBP2`). Second, run more
seeds: everything is seed 42, and the endpoint margin is `0.0019` (`DBP4`). Third, close historical
provenance by rerunning the assays whose frozen artifacts predate manifests, then rerun the figure
package (non-blocking).

The planned successor is a matched frozen-versus-fine-tuned comparison. The unmatched notebook-only
extension once labelled DBP11–DBP17 was archived under DBPR12. A successor must freeze a protocol, hold
task, cohort and readout fixed across states, run multiple seeds and calibration controls, and write
scripts and manifests before any comparison is interpreted.

Two earlier attempts left the tree in the 2026-07-31 cleanup and remain in Git history. The DNABERT-2
Biological Reconstruction Atlas (no artifact, cohort or claim shared with `DBP*`) qualified its
preflight and ran a 13-class census, but no class beat every empirical control (five adequately powered,
eight underpowered), so it stopped before localization. Caduceus-PS has a red-teamed v2 design and a
dry-run Colab package but was never run; it needs CUDA. The only other cross-model work with live claims
is the carrier-sibling pair C39/C40 (`scripts/crossmodel/`, `results/origin-siblings/`,
`results/origin-crossmodel/`). Read [`TRANSFER_GUIDE.md`](TRANSFER_GUIDE.md) before reviving any of them.

## Method coverage

This section comes from an advisory gap analysis by Jeyashree Krishnan and Ajay (branch `jeyashree-ajay`,
Algoverse CSAP cohort 2026), ported on 2026-08-04. The port corrected five module paths and rewrote one
recommendation (R46).

| Method | Where |
|---|---|
| Head and MLP ablation; unit interchange | `src/gfmcd/circuits/hf_bert.py` (`head_ablation_scan`, `patch_heads`), `src/gfmcd/splicing/mlp_units.py` |
| Positional Fourier spectrum | `src/gfmcd/splicing/positional_spectrum.py` |
| Low-rank subspace and cross-example transplant (`U_info`, rank 8; C38, C41) | `src/gfmcd/splicing/subspace.py` `patch_component` |
| QK address geometry | `src/gfmcd/splicing/address.py` |
| Post-LN reconstruct-inject, MLM health | `src/gfmcd/splicing/circuit.py` (`CouplingTask`) |
| Residual path patching | `src/gfmcd/circuits/hf_bert_residual.py` |
| Phase surgery (C36, Ψ_C) | `src/gfmcd/splicing/p3_phase_surgery.py` |
| Carrier dose-response | `scripts/splicebert/run_carrier_dose.py` |
| OV decomposition | `src/gfmcd/splicing/payload.py` |
| Rescue and node-removal curves (C9, C33) | `run_l2_rescue.py`, `run_ablation_scan.py`, `run_a12_faithfulness_size.py` |
| Biological counterfactuals (AG→CC, GT→CC) | `src/gfmcd/splicing/coupling.py` |
| Matched nulls, planted positives | every major assay; `test_circuit_memorials.py` |

The analysis judged the intervention, mediation, spectrum and power-gating work strong. Its recommended
additions, in order (class A would close an active open problem; class B is not yet blocking):

| Order | Method | Class | Effort | Unlocks |
|---|---|---|---|---|
| 1 | Integrated gradients (Sundararajan et al. 2017), beside `src/gfmcd/circuits/attribution.py` | A2 | 1 day | Track 2 candidates |
| 2 | Logit lens, once on 10–20 examples | A3 | 1 day | ≈10.63 write without an L0 reader; layer order |
| 3 | EAP or lightweight ACDC (Conmy et al. 2023) on 50–100 pairs | A1 | 3–5 days | adaptive minimality |
| 4 | Steering-vector variant of `patch_component` | B3 | 1 day | null-content baseline for C38/C41 |
| 5 | Causal scrubbing (Chan et al. 2022) of the full A0–A12 hypothesis | B2 | 1 week | journal-version faithfulness |
| 6 | SAEs on MLP0 acceptor activations ("Towards Monosemanticity") | B1 | 2 weeks | payload semantics, after a larger cohort |

Three notes. `attribution.py` already provides `head_attribution_scan` / `AttributionEffect` and was
restored in `ab670640` because a frozen preregistration cites it; add new variants beside it and never
overwrite it. On the 103-node graph (6 layers of 16 heads, plus MLPs) with 1024-token inputs, ACDC needs
`O(nodes²)` patching passes, which is feasible on a GPU with `CouplingTask.margin` as the metric. Causal
scrubbing needs its hypothesis graph preregistered, so it comes after Tracks 1 and 2. SAEs would overfit
the current ~500–2000-example cohort.

Out of scope: influence functions (the training data and optimizer states are unavailable, so origin is
studied comparatively through C39/C40); probing classifiers (readouts are native model outputs by
policy, and probes are not intervention evidence; see [`METHODOLOGY.md`](METHODOLOGY.md)); anomaly
detection and representation engineering (no misalignment hypothesis applies); and KL-divergence
faithful attribution (APA/Nanda 2024), which would need a new preregistration to replace the frozen
faithfulness metric. None of these methods is evidence the paper depends on (see
[`RESULTS.md`](RESULTS.md)). A method that becomes part of a claim must pass the usual preflight,
intervention, null-comparison, immutable-artifact and ledger workflow first.

## Open decisions

These came out of the 2026-07-22 remediation program and later reviews. None was applied automatically,
because each would choose between two recorded numbers or wordings, or change a recorded statement.
Resolved items, including the full C1 lists, are in Git history as `docs/REVIEW_OPEN_QUESTIONS.md` at
commit `712c6db`.

### B4: the DNA-helical range behind R30

Raised 2026-08-10. `docs/NOMENCLATURE.md` records 10.2–10.5 bp as "the generic DNA-helical periodicity
range from the literature". The positional-boundary-coherence `HYPOTHESIS_REGISTER.md` (lines 181–186,
sources at 580–584) cites Rhodes and Klug, *Nature* 292:378–380 (1981): 10.6 ± 0.1 bp for random-sequence
DNA. If that belongs in the range, 10.630 is not outside it.

R30's refutation of the nucleosomal reading has two legs. The primary leg is the 0.33-nt gap to the
≈10.298 mouse ES-cell dyad measurement, ~7× the measured spreads on either side (0.045 nt each); it does
not use the range. The fallback leg says the refutation "does not rest on cross-species transfer at 0.33
nt precision" because 10.630 falls outside 10.2–10.5 bp. It appears in `docs/NOMENCLATURE.md` (the
≈10.298 row and standing phrase SP2) and in the nucleosome paragraph above. No claim changes today: C14,
C15 and R30 rest on the primary leg, and C39/C40 are independent of both.

Decision required: confirm 10.2–10.5 bp with its provenance, widen it to include the random-sequence
estimate, or split it into sequence-dependent and random-sequence referents; then restate or drop the
fallback in both places.

### B3: "10.67-nt" in the nucleosome map

Recorded 2026-07-22. `data/NUCLEOSOME_MAP.md:36` called the nucleosomal signal "10.67-nt", a retired
label for the wrong object (the nucleosomal signal is ≈10.298). Since commit `4c4f330` (2026-07-24) the
line reads "≈10.298-nt nucleosomal signal", described there as a nomenclature ruling. Decision required:
confirm and close.

### BLOCKER 1: tier banners on live ledgers

Recorded 2026-07-22. A doc-tiering pass inserted a tier banner into 35 read-only files, including 27
frozen preregistrations, three `CLAIMS.md`, `CORRECTIONS.md`, `DECISIONS_A0.md` and
`RESULTS_VERIFICATION.md`. All 35 were reverted. Decision required: whether live ledgers should carry a
tier banner deliberately (frozen preregistrations should not). The tier system was retired at the
owner's direction on 2026-08-14 (R47), which may settle this.

### C1: leftovers of the reverted `results/` reorganization

Recorded 2026-07-22. A reorganization moved 197 development and superseded files into
`results/<subdir>/development/`; it was reverted and dropped. Four questions were recorded against the
moved state: whether to create a symmetric `canonical/` layout, whether moved manifests get an appended
relocation note, whether bare artifact basenames in ledgers and preregistrations need requalifying, and
where runner preflight files belong. `results/*/development/` directories exist today, so the questions
may still apply.

## Repository conventions

Keep the record on `master`, the default branch, so the repository link shows the current project. Keep
`CLAIMS.md` compact and `CORRECTIONS.md` complete. Use [`REPRODUCE.md`](REPRODUCE.md) as the single entry
point for environment and reproduction. Preserve immutable artifacts and manifests, and keep large
reproducible shards outside Git.
