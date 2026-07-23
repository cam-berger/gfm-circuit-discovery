# Pre-compute design red-team — estimand C (6 lenses) — 2026-07-18

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Adversarial design red-team run **before any compute**, per the house pattern (IC1 5-lens, P3 4-lens,
Caduceus 5-lens). Input: `DESIGN_DRAFT.md` (the formalized estimand-C design). Six independent lenses
(identifiability, statistical, confound, calibration, architecture-forced, cohort/precedent) + a
synthesis. Full machine records: session workflow `wf_0f86167f-44b` journal.

## Verdict: **NO_GO_REDESIGN** (overall FATAL)

Three lenses returned **fatal** (identifiability, cohort_precedent, statistical); three **fixable**
(confound, calibration, architecture_forced). Four convergent, code-grounded reasons the confirmatory
single-head C atlas as drafted must not be run for Algoverse credits:

1. **The behavior C attributes to is likely bounded-null.** `E_ih` is definitionally a conditional
   functional of the behavioral `M_comp` (|E_ih| ≤ |M_comp|; SESOI-C was *defined* as a fraction of
   `M_comp`). Both project priors put `M_comp` at bounded-null: **IC1** ΔC_specific = +0.006, CI
   [−0.011,+0.024] **includes 0**; **IC2** β_transport SD 0.774, ±0.05 equivalence declared structurally
   unreachable at ~1,985 genes. If `M_comp` is bounded-null, `E_ih` is bounded-null a fortiori.
2. **Provably underpowered by 2–3 orders of magnitude.** `E_ih` is a *third-order* contrast (a difference
   of `M_comp`, itself a difference of margins) — strictly higher variance than IC2's β_transport, on a
   **smaller** sealed cohort. IC2 census: 142 competing loci genome-wide, **0 usable at ≥80 nt** (windows
   overlap). The 4 seeded draws re-index one cache → **zero** added independent n.
3. **Single-head existence contradicts C27/C33.** The analogous coupling computation is irreducibly
   distributed (F=0.005, no knee, 90% needs 72/103 nodes, mean-ablating even L3H1 leaves MLM health
   intact). If competition is processed like coupling, single-head `E_ih ≈ 0` for *every* head by
   redundancy → a null omnibus is the **predicted, uninterpretable** outcome ("distributed", not "absent").
4. **Wrong cohort relationship + collider.** `partner_data.attested` is **exon-mate** (alt-5′ of the same
   exon), while `donor_margin`/the C1–C35 circuit use the **intron-mate** partner — the gate would certify
   behavior X and license claims about behavior Y. And there is **no "unselected" pool** for C: every
   example is conditioned on the collider "a competitor exists AND is in-window", inducing a Berkson
   strength↔distance dependence the behavioral gate can be *passed by*.

## Decision (Cam, 2026-07-18): **fix-in-place, still targeting the Colab confirmatory run**

Cam elected to apply the must-fixes and keep the paid all-96-head Colab confirmation on the table, rather
than adopt the recommended demotion to a cheap-MPS behavioral program. The credit-protection the red-team
demanded is preserved as a **hard pre-compute power/cohort gate baked into the Colab runner**: it runs the
cheap cohort census + MDE projection first and STOPs to `INSTRUMENT-LIMITED` before the bulk credit spend
if the sealed cohort cannot power the estimand.

## Must-fixes and how each is applied (MF1–MF11)

| # | Finding (lenses) | Fix applied in the frozen design |
|---|---|---|
| MF1 | `E_ih` bounded by likely-null `M_comp` (identifiability, cohort, statistical — FATAL) | Behavioral competition gate promoted to a **hard standalone MPS preflight on discovery** run BEFORE any Colab job; bounded-null `M_comp` is a **terminal STOP → INSTRUMENT-LIMITED** (coheres with IC1/IC2), pre-registered, not a footnote. |
| MF2 | Third-order contrast underpowered at sealed n (statistical, identifiability — FATAL) | **Pre-compute power gate** (`build_census.py`): realized usable sealed n per arm + projected per-head MDE (IC1 SD 0.162 / IC2 SD 0.774, ×96 multiplicity). `MDE ≥ SESOI-C ⇒ not licensable`, declared at design time. Gate wired into the Colab runner as a hard STOP. |
| MF3 | Single-head existence contradicts C33-distributed (statistical — FATAL) | Confirmatory estimand re-scoped from single-head existence to a **group / importance-ranked-set / faithfulness-size** statistic that can detect a *distributed* signal; a per-head null is pre-registered as **expected, not evidence of absence**. Single-head effects → descriptive census (A) only. |
| MF4 | Exon-mate vs intron-mate cohort (cohort — FATAL) | New **intron-mate** cohort builder (`src/gfmcd/splicing/competition.py`) from transcript-order junctions (donor of exon k → acceptor of exon k+1); `attested` is NOT reused. Both site types (alt-5′SS donor, alt-3′SS acceptor). Verified by logic smoke. |
| MF5 | No "unselected" pool / collider selection (confound — FATAL-adj) | Drop "unselected"; common support reframed as **between-cohort**; matched-single-site arm matched on **selection variables** (competitor-annotation propensity, in-window probability) + confounds; report induced strength×distance correlation + a collider-bias sensitivity analysis. |
| MF6 | Matched decoy = generic weaker motif ⇒ R22 strength confound (confound, calibration, cohort — HIGH) | Decoy must be a **real, non-partner, MaxEnt-strength-matched** site anchored to the **competitor** (score5/score3 caliper), not `other_gt/other_ag`. Can-fail `decoy_strength_matched` memorial. |
| MF7 | mean-ablation injects a between-condition mean shift (architecture, statistical — HIGH) | HC1 primary intervention → **scrub-to-corrupt** (patch head output to its competitor-DESTROYED value; on-distribution, condition-aware; reuses A6 machinery / `head_patch_scan`). mean → discovery robustness only. New on-path add-one memorial for the patch path. |
| MF8 | HC2 A-vs-V uninformative-by-architecture for a partner-presence contrast (architecture — HIGH; cf C23 interaction-dominated) | Do **not** freeze HC2 OV-stability until a **QK-main sensitivity floor** is shown on the gate-B coupling contrast; if the competition contrast is interaction-dominated (as C4/C23), demote HC2 to **descriptive route_shift** and drop the OV-stability decision. Re-derive which head reads at the **competitor** (do not import L3H1 from the coupling circuit). |
| MF9 | Calibration keyed to the ground-truth label / wrong permutation (calibration — HIGH) | Planted-positive conditioned on a **readable motif** at the competitor window (not the label), sized so the through-model swing = SESOI-C, and required to fire on decoy-destroy too; permutation matched to the statistic (permute competitor-status across matched (target, destroyed-real-motif) pairs, perturbation type held fixed); add a **competitor-strength-only** plant that must stay null after adjustment. |
| MF10 | Draws collapse to one tuple; "all-4" funnel (statistical — HIGH) | **Pool** draws (draw-aware bootstrap) + across-draw **stability-SD trigger** (IC1 style), not "significant in all 4"; report how many clusters actually have ≥2 candidate competitors. |
| MF11 | Tandem/NAGNAG + partner_in_window distance bias (cohort — MED/HIGH); SLOT positive control (cohort — MED) | Frozen **min_separation** (excludes NAGNAG/tandem), mutual-exclusivity required; report window-inclusion rate per distance stratum; add **IC2's SLOT effect (τ_D≈0.53) analog** as a required preflight positive control so a null is *evidenced*, not defaulted. |

## Accepted residual risks (proceed without further design iteration)
- max-T Westfall–Young permutation is FWER-correct under arbitrary head correlation (subset-pivotality
  holds for label permutation); the §9-Q4 worry is a red herring — the only cost is power (already
  surfaced by MF2/MF3).
- Post-LN QK-patch softmax-nonlinearity affects *wording* (main-effect additivity is exact at head output
  O, approximate at the margin), not validity.
- Census A (descriptive prevalence) proceeds as long as cross-site-type comparisons are flagged
  distance-truncation-confounded and no confirmatory claim rides on frequency.

## The honest expected outcome
Per the synthesis: the most probable result of the corrected program is a **cheap, publishable behavioral
null that extends the IC arc** — obtained at MPS cost. The Colab confirmatory run launches only if the
pre-compute power gate AND the behavioral competition gate AND the SLOT positive control all pass.
