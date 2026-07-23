# SpliceBERT decision record

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

This is a consolidated chronology of scientific pivots. It records why the active tree changed; it is not a second claim ledger.

## 2026-07-10 to 2026-07-11 — initial partner-selection framing

The project began as a frozen-representation forced-choice task intended to distinguish true splice partners from decoys. Early performance appeared to support the task.

Decision: **withdraw the framing and implementation.** The true site occupied a fixed position, a geometry-only oracle could solve the task, and site detection was sufficient. A v2 using annotated decoys still reduced to choosing the stronger major site over weaker minor sites. A “random backbone” control was also a no-op.

Consequence: forced-choice partner-selection code and reports are historical only. The active question became whether the model's native score at one splice site depends on its paired site.

## 2026-07-12 — pivot to native cross-site coupling

The replacement design masked a donor nucleotide, corrupted the paired acceptor with real bases, and compared motif- and distance-matched controls. This produced a native behavioral quantity that could be tested without a trained probe.

Decision: use unconditional coupling for behavioral magnitude and reserve effect-conditioned examples for recovery fractions. This replaced the inflated 1.359 held-out mean with the unconditional 0.973 validation estimate and later 0.872 sealed estimate.

Decision: include both strands after a transcript-order bug was found. The corrected eligible population roughly doubled and the selected heads remained L3H1 and L2H15.

## 2026-07-12 — localization became a subcircuit claim

L3H1 ranked first in the exhaustive head scan, but held-out factorials showed it also had a partner-independent role. L2H15 was a second causal contributor missed by denoising alone.

Decision: describe `C = {L3H1, L2H15}` as a **dominant subcircuit**, not a single dedicated head and not a share of summed causal mass. Marginal effects overlap in serial and redundant systems.

Decision: use masked probes for model health. The earlier unmasked metric measured input copying and could not fail.

Decision: enumerate or de-duplicate finite random-null universes and use exact add-one p-values. The previous duplicated null overstated significance.

## 2026-07-12 — mechanism interventions were narrowed

Key/value patching supported a conjunctive acceptor read. Whole-output restoration closed the L2H15 → L3H1 edge; position profiling located L3H1's output at the donor neighborhood; downstream patching showed a distributed readout chain.

Decision: retract key/value **swap** evidence because swapping complete pairs is an algebraic identity.

Decision: retract the later **content-addressed relocation** interpretation because moving vectors that already contain absolute position information makes the result architecturally forced. The sealed numerical result remains provenance, but it does not license a pointer or lookup claim.

Decision: report completeness as inconclusive. Large complement ablations damage general language-model behavior, and size-matched random sets are similarly destructive.

## 2026-07-12 — sealed replication

Four preflight blockers were fixed before unblinding: strand eligibility, algebraically invariant binding intervention, partner leakage into a non-partner pool, and an unmatched/distance-confounded specificity comparator. The circuit was reselected on corrected data and did not change.

Decision: freeze code hashes, circuit, controls, and 13 pass/fail gates in [`preregistrations/PREREGISTRATION_SEALED.md`](preregistrations/PREREGISTRATION_SEALED.md), then open chr8/chr9 once. All 13 gates passed.

Decision: preserve the preregistration unchanged. Post-unblinding corrections belong in this record and [`CLAIMS.md`](CLAIMS.md).

## 2026-07-12 to 2026-07-13 — period-3 interpretation corrected

An initial “42-head codon-phase lattice” was built on phase purity. Audit showed the statistic was maximized by single-offset heads and therefore measured sharpness/locality.

Decision: retract the broad lattice. Require repeated comb structure, deletion, protected controls, behavior, and restoration.

The replacement result established a rank-2 period-3 position code: deletion collapsed attention phase mass and donor–acceptor coupling, while matched single-head restorations identified L2H15 and L3H1 as mediators.

Decision: treat the period-3 clock as part of the donor–acceptor mechanistic story, not a detached collection of phase plots.

## 2026-07-13 — second-carrier hypotheses disciplined

A separate ≈10.63 positional carrier prompted α-helical, DNA-helical, nucleosomal, and generic-ruler hypotheses. Several early assays were either unpowered or built on unverified controls.

Decision: an anomalous control freezes interpretation. A band may be called energy-matched or low-leak only after those properties are measured; it may not be called inert by assertion.

Decision: calculate entropy headroom, coherence length, wavelength resolution, and expected detection floor before training or scanning. The synthetic/source exploration taught useful design lessons but is not an active result family.

Decision: use landmark-anchored linear matched filters for weak biological periodicity. Unanchored power is quadratic in effect depth and was calibrated to be blind at the relevant scale.

A powered dyad assay found a strong ≈10.298 nucleosomal signal, including frozen-phase replication and offset specificity. It did not match the multispecies ≈10.630 carrier.

Decision: close the nucleosomal-source interpretation for the multispecies carrier and leave its origin open. Preserve the positive biological finding as a distinct result.

## 2026-07-13 — the evidence-level firewall held

QK-logit scans suggested human-only ≈10.2 readers and multispecies ≈10.63 readers. Earlier post-softmax estimators produced impossible or unbounded quantities.

Decision: stop and build a gated post-softmax instrument with known period-3 positive/negative controls, injection recovery, per-head permutation nulls, bounded coherence, band-mass eligibility, deletion audits, reciprocal bands, and content-shuffle checks.

The validated instrument found eight multispecies Level-A heads and no corrected human-only Level-A head. Exact add-one correction removed a raw zero-hit artifact; band-mass measurement showed that most preregistered human-only readers were never eligible.

Decision: fire the preregistered stopping rule. Close the human-only chromatin-circuit proposal at QK, do not run species-matched rescue experiments, and advance only the multispecies carrier to potential OV/behavior work.

## 2026-07-14 — active-tree consolidation

The repository contained overlapping reports, superseded scripts, unindexed result JSON, VOID families, and older figures that made historical branches look active.

Decision: make the active tree an executable record of the current claims:

- keep seven authoritative study documents;
- keep only scripts that generate indexed discovery, validation, or reproduction artifacts;
- remove unindexed/VOID result families from the active tree;
- preserve frozen preregistration and permanent retractions;
- rely on Git history for obsolete experiment implementations.

This is a scope correction, not deletion of scientific provenance. The authoritative record keeps the conclusions, limitations, failures, and reasons for each pivot while removing competing entry points.

## 2026-07-22 — reviewer roadmap B1–B5 formalized and closed (design red-team)

The reviewer/ROADMAP branches B1–B5 were each written up as a full preregistration and put through a pre-compute adversarial design red-team (five independent lenses per branch) BEFORE any compute was spent. The point was to catch identifiability and power failures at the design stage, where they are cheap, rather than after an expensive MPS run.

Decision: record B1, B2, and B3 as **fixable-then-runnable, but not this-session survivors.** B1's behavioural coupling arm (ΔC under deletion re-pointed to the ≈10.63 plane) is clean and identifiable, but its OV-write arm is non-identifiable as written — the `dO_h` contrast conflates the carrier's already-established QK/routing role with the OV column (an R23-class ladder column-jump); it needs the P3.3 A-vs-V value isolation plus a coupling-equivalence sizing pilot. B2's semantics arm — the only arm that would close C30 — is non-identifiable on a frozen checkpoint at MPS budget (it re-runs the R22 multi-draw/dyadic wall), so C30 is scoped out and only a geometry-controlled existence + partner-specificity positive search remains runnable. B3 carries two convergent, fixable fatals (an object/space contradiction in the injected object and a U_info backdoor); it becomes identifiable and MPS-feasible after a bounded v2 redesign and a discovery pilot.

Decision: **defer B4.** The L3H1/carrier→L2MLP bridge is non-identifiable as a direct edge on a frozen post-LN checkpoint (verified across three convergent lenses): the only implementable carrier deletion is global at the position-embedding source, and post-LN renormalization leaves no additive carrier component to subtract at a mid-layer edge, so the intended "edge distinct from global C8" does not exist. This is an identifiability blocker, not a power or cohort-location one; a real cross-estimand bridge would need a training-time / subspace-freeze design.

Decision: **B5 (context-signal power) is the one runnable survivor** and is being executed separately in the Head–Context-Atlas study, not here. B4 and B5 are Atlas-study preregs and are tracked in `../splicebert_head_context_atlas/`.

Operational note: the SpliceBERT.1024nt weights and the coupling/atlas data are absent from this worktree, so any future B1/B2/B3 execution must be launched from the main checkout. The B1–B4 drafts are retained as designs-of-record; the per-branch dispositions and verbatim must-fixes live in each prereg's closure section and in the matching `redteam/REDTEAM_SYNTHESIS_B*.md`.

## 2026-07-22 — B1 advanced to v2 (split-freeze) after the OV-estimand redesign was re-audited

B1's OV arm was redesigned per the first red-team's must-fix #1 (replace the routing/OV-conflated `ΔO_h` with the C35/P3.3 **value main effect**) and put through a focused 3-lens pre-compute re-audit (`redteam/REDTEAM_SYNTHESIS_B1_V2.md`). The re-audit caught a real internal-consistency bug and produced a **split-freeze**:

- **The catch (M-1a):** the first v2 cut defined the value-main as the *symmetric* two-row main `½[(DI−DD)+(II−ID)]`, which equals the simple `DI−DD` **plus ½·Δ_bind** — re-importing half the architecture-forced R23 interaction (on the C35 router it gives **+0.073, not the −0.005 null**). Corrected to C35's actual **simple single-row `DI−DD`** (attention pinned deleted; `A_D·(V_I−V_D)`, zero iff values are carrier-independent). The ill-posed readout-agnostic "L1" (head-output value-main is generically nonzero via position-embedding propagation) was demoted to a reported diagnostic; the licensed OV endpoint is the **donor-readout-projected** value-main (`U_info`/coupling), covariance-matched nulls (M-2), router-negative measured at the readout scale with `SESOI_v` pre-declared ≥0.02 (M-1b).
- **Behavioural arm → FROZEN_RUNNABLE.** `ΔC(D10)` coupling equivalence (pre-committed by Cam: expect a NULL) is a paired within-example contrast on a near-inert deletion (`carrier_dose`: 0.946→0.970, `|point|≈0.024`). The re-audit **corrected v1's power pessimism**: `HW<0.026` at n=256 needs only **ρ>0.42**, not ρ≥0.98 — so the equivalence is cleanly reachable, gated on the §7 sizing pilot, with the dual-branch (equivalence / re-select the inert control band) pre-committed.
- **OV arm → design-frozen pending its `--smoke` gate on the main checkout** (M-simplemain, the readout-scale M-vmain router-negative, PP-ovplant, covariance-matched nulls, Preflight-R non-empty). Modal OV outcome is a bounded donor-scoped A-null or Endpoint B (`R_1024` may be empty — the ≈10.63 readers are pre-softmax on 1024nt) — both legitimate, pre-committed. The OV claim is **donor-circuit-scoped** (agreed with Cam): the carrier's general OV content (MLM output) and origin stay open.

Design-of-record v1: `preregistrations/PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR.md`; frozen-candidate: `…_V2.md`; re-audit: `redteam/REDTEAM_SYNTHESIS_B1_V2.md`. Execution (both arms) launches from the main checkout.

## 2026-07-22 — B1 EXECUTED: the ≈10.63 carrier is not part of the donor circuit (C37, bounds C12)

Both B1 v2 arms were run on the SpliceBERT.1024nt checkpoint (drivers `run_carrier_preflight_r.py`, `run_carrier_ov_av.py`, `run_carrier_p10_behaviour.py`; artifacts under `results/phase-lattice/`; full write-up in the prereg's "Execution results" section and CLAIMS **C37**).

- **Preflight R → `OV_ARM_LIVE`, R_1024 = 9** post-softmax ≈10.63 readers — C11's eight 510nt readers **replicated on 1024nt** + L2H6. **This revises T1's pessimism** (the "disjoint, pre-softmax only" was true of the *QK* readers, not the post-softmax set); the OV arm was identifiable, not Endpoint B.
- **OV arm → A-null.** All 9 readers' simple value-main `Mv=DI−DD` within ±SESOI_v=0.02; router-negative (period-3 L3H1) nulls at −0.0099 and **`M-simplemain`=0.0 exact** — the M-1a fix (simple, not symmetric, main) held empirically; `Δ_bind`=+0.20 quarantined (R23). The carrier writes nothing carrier-specific onto the donor readout.
- **Behaviour → small suppressive, not a clean equivalence.** dC(D10) stabilised across n (−0.015→−0.034→**−0.0333 [−0.0606,−0.0053]** at n=2500/1085 genes); the CI **excludes 0** — deleting the carrier **raises** coupling by 0.033 (sub-SESOI, *small-but-detectable*), replicating the dose arm (+0.024). Not used to *support* coupling; given the OV null, not mediated by the readers' values.
- **§5 risk-2 discharged:** band 94–100 is inert for MLM but **not** for coupling (deletion raises it 0.033) — its `carrier_surgery`/`dose` "inert control" role is qualified for coupling (does not touch C8; period-3's effect is ~30× larger and supportive).

**Two design projections the run corrected (recorded for honesty, no R-number — these were design *projections*, not claims):** (1) T1 was too pessimistic on OV identifiability; (2) the behavioural power was mis-projected by *both* red-teams — v1 said ρ≥0.98, the re-audit "corrected" to ρ>0.42 but borrowed C8's MLM per-example SD (~0.02) instead of the coupling-margin SD (~0.87, which the v1 red-team's must-fix #5 had flagged). The two-sided ±0.05 equivalence is therefore expensive and unreached; reported one-sided at achieved precision. The behavioural pilot did its job — the empirical sizing corrected both projections.

**Provenance/hygiene note:** run in this worktree with `data/models` **symlinked** from the main checkout (gitignored; removed at cleanup). The n=256 pilot JSON was overwritten by n=1200 in the first chain (its numbers are preserved in this entry and the run logs); n=1200 (`_n1200.json`) and n=2500 (`_confirm.json`) are retained as immutable artifacts. C37 bounds C12 (donor-scoped); general OV (MLM) and origin remain open.
## 2026-07-22 — Bookkeeping close-out (no new experiments)

A pure bookkeeping pass to close administrative loose ends; nothing was run. Four actions:

- **HCA9 L3H1 positive control folded in.** The Head–Context Atlas competition branch closed at behavioral
  specificity (`../splicebert_head_context_atlas/`: `CLAIMS.md` consolidation banner, `DECISIONS.md`
  2026-07-22 — the perturbation is **not** competition-specific, HCA4/HCA10). Its one salvageable result —
  the branch independently **rediscovered donor-circuit L3H1** as the donor-reconstruction mediator on a
  disjoint donor-target cohort (donor B +0.114 [0.098,0.130]; exact 400-gene L3H1 patch +0.0857 [0.0492,0.1280]
  vs behavioral B +0.1015) — is recorded here as a **cross-study positive control** corroborating C3/C4/C5/C17,
  with a pointer added to the `CLAIMS.md` interpretation guardrails.

- **Three terminal C-rows closed** (`CLAIMS.md` status vocabulary gains `closed`): **C7** (the early
  broad-ablation completeness assay — a forward-pointing stub superseded by C27/C33), and **C10** and **C15**
  (permanent matched-null power and calibrated-sensitivity bounds). Nothing runs to close these further; they
  are marked terminal so they are not read as open loose ends. Claim text and artifacts are unchanged.

- **Period-3 present-day-use family (C34/C35/C36) stamped DONE.** C36's clean nucleotide-invariant
  confirmation supplies the clean phase-aligned-use evidence that C34's gate-qualified confirm (R24) lacked;
  C35 fixes the routing locus. Present-day causal **use** is established; only **origin** remains, deferred
  (P3.4, training-time evidence). Recorded as a guardrail stamp in `CLAIMS.md` and reflected in `CLAUDE.md`.

- **IC2 and IC2a retired.** IC2 is the confirmatory acceptor-competition run for exactly the competition-
  specific premise the Atlas refuted (HCA4/HCA10); running it would confirm a behavior already shown not to be
  competition-specific, so it moves from **parked → retired** (revivable design asset only). IC2a (the deferred
  proximal-module decomposition that fed the same arc) moves from **not-pursued → retired**. Both preregs keep
  their frozen bodies; only their status banners and the `preregistrations/README.md` dispositions change. No
  outcome may be claimed from either.

These are lifecycle closures, not retractions — no new R-number is warranted; the empirical record is unchanged.
