# B2 PAYLOAD SEMANTICS (close C30) — 5-lens design red-team synthesis (2026-07-22)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Reviews:** `../preregistrations/PREREGISTRATION_B2_PAYLOAD_STEERING.md` v1 (DRAFT for pre-compute red-team).
Five independent adversarial lenses, run BEFORE any confirmatory compute. **Outcome: NOT freezable as
written.** The drafter's disposition (FIXABLE_THEN_RUNNABLE, "existence+specificity core runnable now") is
**overstated**: two convergent fatals (one fixable engineering, one identifiability redesign) plus a
specificity **precision wall** verified against the repo's own pilots mean that (a) the SEMANTICS arm — the
only arm that actually **closes** C30 — is non-identifiable / equivalence-unreachable at MPS budget
(the drafter already concedes this), and (b) the EXISTENCE+SPECIFICITY "minimal core" is either
near-tautological (EXISTENCE) or position-confounded and underpowered (SPECIFICITY). B2 can, after a
**redesign** (not footnotes), be run as a one-sided EXISTENCE + partner-specificity **positive search**
with preflight discovery-MDE gates — but its modal outcome is **tighten C30 (Endpoint B), not close it**,
and it is **NOT a this-session survivor**.

| Lens | Verdict | Headline |
|---|---|---|
| Identifiability | fixable (fatal inside) | **u_choice = diff-in-means between two DIFFERENT-position acceptors → a position/distance axis (R20 re-entry), uncontrolled at every rung** (code-verified) |
| Statistics | fixable | E_partner DiD **unbudgeted & underpowered** (~900 genes at DiD SD 0.774); EXISTENCE power uses wrong SD + no max-T penalty; semantics multiplicity uncorrected |
| Confound completeness | fixable (fatal inside) | The ONLY two backdoor-blockers (E_partner, E_decoy) are **null/wrong-signed at pilot**; E_rel is near-forced; label confound = R20 |
| Calibration ↔ intent | fixable (fatal inside) | **Planted-positive travels A9's L3H1 VALUE path, NOT B2's net-new residual-add hook** (code-verified) — the interpreted channel has no sensitivity gate |
| Feasibility / power / MPS | fixable (NO resource blocker) | Existence-core buildable & affordable; **powered arm (EXISTENCE) is non-diagnostic, diagnostic arm (E_partner) underpowered**; semantics equivalence off the table |

---

## The convergent FATALs (independently verified against code + data)

### FATAL-1 — u_choice is a position/distance axis, not a "choice/payload" axis (R20 re-entry at direction construction; Identifiability + Confound lenses)
- **Design (§1 L60-64):** `u_choice = mean_win(r_pref) − mean_win(r_less)`, where preferred = higher native
  `B_i(p)` acceptor, less-preferred = lower-`B` acceptor, **within a shared-donor alt-3'SS set**.
- **Verified in `src/gfmcd/splicing/competition.py`:** the two competing acceptors are DISTINCT 3'SS receiving
  one shared donor (L18-22), from different transcripts, mutually exclusive by construction. `CompetitionExample`
  carries **`target_partner_distance` and `competitor_partner_distance` as SEPARATE fields (L164-165)** precisely
  because they differ; the window randomization moves the whole window, never the two sites relative to each
  other. So preferred vs less-preferred sites systematically differ in **donor-distance and absolute position**.
- **Why fatal:** **R20** (CORRECTIONS) established that L3H1's acceptor "preference" (higher `B`) is **DOMINATED by
  learned position-embedding geometry**, not partner identity — SpliceBERT has absolute learned PEs. A
  difference-in-means between two different-position windows therefore has the **position/distance offset baked
  into the direction**; §5's mitigation only residualizes position out of the scalar *label*, which cannot remove
  the offset from a diff-in-means between two different-position centroids. The named control family
  (random-orthogonal, equal-var-PC, `U_info`, `U_read`, unembedding, site-identity; §4 L170-176) contains **NO
  absolute-position/distance axis** — beating a position-agnostic random direction does not exclude a
  position-geometry confound. This is fatal for SEMANTICS naming **and compromises the Endpoint-B fallback**
  ("a causal, partner-specific choice direction exists"), because the direction is not separated from geometry at
  **any** rung. Internal contradiction to resolve: §1 L62 labels on raw native `B`; §5 L196-197 says residualize
  position/composition BEFORE labelling — these define different objects.
- **Fix (redesign, not footnote):** caliper-match preferred vs less-preferred **within each contrast** on
  `target_partner_distance` AND absolute window position (and exon-length/CDS/GC) so the centroids differ in
  preference but not geometry; **add an explicit absolute-position + donor-distance direction to the control
  family** so a geometry axis that reproduces the effect is caught; add an on-path memorial that `u_choice` is
  ~orthogonal to the absolute-position subspace / survives a held-out position stratum. Demote SEMANTICS naming to
  deferred; report Endpoint B only if the effect survives with the geometry axis controlled.

### FATAL-2 — Calibration travels the wrong channel (P3.4 "dead-PP" class; Calibration + Identifiability + Feasibility lenses)
- **Design:** the planted-positive (§4 L161-165) and the on-path sensitivity memorials **M-α0** and **M-dose**
  (§3 L147,149) all reuse A9's `interchange(synth=(Vplant,d0,β))` branch.
- **Verified in `scripts/splicebert/run_a9_payload_semantics.py` L218-227 + `src/gfmcd/splicing/payload.py`
  L40-70:** that branch builds `dr = β·coef·d0`, delivers it via **`deliver(Wv,dr) = dr @ W_Vᵀ` (L3H1 VALUE
  projection)**, forms `od = head_output(base, Ap, dV)` with the attention row **`Ap` FROZEN**, and
  `inject_capture`/`patch_heads` patches **ONLY L3H1's donor head output at the donor token** (residual never
  re-propagated).
- **Why fatal:** B2.1 EXISTENCE's actual channel is the **NET-NEW general residual add `h_l(j) += α·u_choice` at
  an arbitrary (window, layer) followed by a full forward** — explicitly listed as net-new item #3 (§2 L129-131,
  verbatim "A9 only injects through the L3H1 value path"). So the sensitivity control validates A9's value-path
  head-patch plumbing, **not** the channel B2.1 interprets; §4's claim that "the entire B2.1→B2.3 pipeline must
  recover Vplant" is **internally contradictory** with §2 item 3 (the value-path synth branch cannot exercise a
  residual-add hook). A miswired or off-manifold general hook **passes calibration and fails silently** — this is
  exactly the P3.4 M5 hole (the `--smoke` gate does not dry-run the real planted-positive on the interpreted
  channel).
- **Fix (engineering, doable on MPS):** build the planted-positive to inject a known scalar-scaled direction
  **through B2's own residual-add hook** and re-run the full forward, requiring end-to-end recovery of the planted
  scalar; run **M-α0 (exact zero-dose identity) and M-dose (monotonicity) ON that hook**; add the hook's PP to
  `--smoke` before freeze. Keep the A9 value-path synth only as a secondary anchor.

---

## The load-bearing MAJOR (raised, in some form, by all five lenses)

### SPECIFICITY is not identifiable at the signal level AND underpowered at budget → the "existence+specificity core runnable now" disposition is unsupported
After E_rel is stripped as forced (below), concept-specificity rests **entirely** on E_partner and E_decoy, and
both fail at the pilot signal level the recon itself establishes:
- **E_partner** (partner-specificity, a difference-in-differences). Behavioural analogue `M_comp_minus_sham`
  (alt3ss) = **+0.0146 CI [−0.051, +0.088] INCONCLUSIVE** (n=140), and the **sham alone = +0.032 ≈ M_comp +0.034**
  — i.e. ~**95%** of the raw signed competition effect is reproduced by a distance-matched non-site perturbation
  (`behavioral_pilot_v1.json`). The partner-specific signed signal is **indistinguishable from a generic
  perturbation** in the native cohort. As a DiD it is **never power-costed**; at the ic2 DiD SD **0.774**
  (`ic2_calibration.json` part3, verified) reaching CI-HW < SESOI 0.05 needs `≈(1.96·0.774/0.05)² ≈ 900`
  gene-clustered sites — **materially more than the ~400 the prereg budgets**.
- **E_decoy** (spatial specificity). Analogue `specificity_absMcomp_minus_absMdecoy` (alt3ss) = **−0.0736 CI
  [−0.195, +0.013]**, i.e. the **DECOY fires LARGER than the intended site** (abs_M_decoy 0.161 > abs_M_comp 0.087)
  — **wrong sign** for "intended ΔR > decoy ΔR" — and **425/725 windows lack a usable decoy** (`specificity_pilot_v1.json`).
  Its precursor already returned an INSTRUMENT-LIMITED verdict.
- **E_rel is near-forced (R7/R21 class), not an independent discriminator.** `E_rel = ΔR_ij = ΔB_i − ΔB_j` is
  offered as a confound-specificity condition, but the injection is applied at window `j` ONLY; `B_i`/`B_j` are read
  at different windows (`coupling.py` `_margin`/`acceptor_margin`), so `B_i` moves only by weak cross-window
  leakage and `E_rel ≈ −ΔB_j` whenever EXISTENCE fires. A single-window injection **cannot** raise every margin, so
  a generic acceptor-confidence direction produces the identical "redistribution." **E_rel must be LABELED FORCED
  and dropped from the specificity AND.**
- **Power is on the wrong endpoint.** EXISTENCE steers `B_j` on the **material** axis (pilot `B_partner` alt3ss
  +0.244) — a generic "raise-coupling"/readout direction passes it; it is **near-tautological** and non-diagnostic.
  The **diagnostic** partner-specific redistribution sits on the **weak bounded-null** `M_comp` axis (BOTH +0.006
  [−0.026,+0.040]; SD 0.306–0.317) and is underpowered. The EXISTENCE power projection also uses the **wrong SD**
  (SD(M_comp)=0.31 substituted for the unmeasured SD(ΔB_j); raw `B_j` SD is 0.52–0.64) and **omits the max-T family
  penalty**; the on-manifold (M-health/M-ood-surviving) dose ceiling may sit **at or below SESOI_B=0.05**, making
  "EXISTENCE positive" unreachable by effect size, not just noise.

**Consequence:** a passed EXISTENCE would establish only local linear **steerability of `B`** (near-given), with
SPECIFICITY unable to name the direction as partner-specific/choice at the ~400-site budget, and the base signed
behaviour **net of a matched sham not even established to exist**. Steering toward an un-established,
confound-dominated behaviour re-earns an R22-class retraction if done loosely.

---

## Additional issues (fixable)
- **SEMANTICS multiplicity uncorrected (Statistics).** The naming claim fires "for some X ∈ {V2b MaxEnt,
  competitor-gap, usage}" (§6 L228-230) with per-X FPR ≤ 0.05 (§4 L170-172) — an OR over three tests inflates the
  naming FWER to ~0.14. Apply a family (Bonferroni / shared-resample max-T) correction and pre-register α at the
  family level.
- **Preference label is a near-coin-flip (Calibration/Confound).** Labels are `sign(B_i−B_j)`; the pilot signed gap
  (M_comp +0.006 BOTH / +0.034 alt3ss, SD ≈0.26–0.32) gives per-example SNR ≈0.1 ⇒ ~half the "preferred" labels are
  noise, and M-label's "sign > chance" can PASS while `u_choice` is mostly geometry/noise. Add a **label-signal
  floor** (restrict to sets whose native `B_i−B_j` CI-excludes-0) and a quantitative M-label bar.
- **SESOI_B=0.05 is set ABOVE the phenomenon it detects (Calibration).** Pilot signed M_comp (0.006–0.034) and
  M_comp_minus_sham (inconclusive) are **below** 0.05; a steer clearing 0.05 is larger than the entire natural
  signed choice behaviour (over-drive, estimand drift). Calibrate SESOI to the phenomenon scale (~0.02–0.03) or
  reframe EXISTENCE against the random+sham family rather than a fixed bar.
- **X3 "usage" can only support a positive, never a null (Calibration).** `n_tx_target_junction`/`target_is_mane`
  (`competition.py` L172-175) is coarse ordinal; its dyadic association is the same high-variance object IC2 showed
  is equivalence-unreachable. Keep it strictly inside the one-sided-positive framing.

## Minors
- After removing forced anchors, re-anchor **M-dose** to the residual-add hook so ≥2 failable on-path memorials
  (M-α0 identity + M-dose monotonicity) live on the **interpreted** channel, not the A9 value path.
- **Operational trap (verified):** `data/models` **does not exist in this worktree**; the SpliceBERT.1024nt
  checkpoint is present only at the main checkout root
  (`/Users/cberg/Research/projects/gfm-circuit-discovery/data/models/...`). Scripts default to the relative path —
  run from the main checkout / symlink `data/`, and pin `--model` to an absolute path in the manifest.

## Feasibility & MPS (lens 5) — no resource blocker; the blockers are power-allocation, calibration-channel, identifiability
- **BUILDABLE / affordable:** every reused primitive exists and composes on disk (`payload.py`
  deliver/head_output/inject_margin_donor; `run_a9` interchange+synth; `competition.py`
  build/enumerate + usage fields; `competition_seq.build_window`; `coupling.py` destroy/perturb;
  `circuit.py` CouplingTask/margin/gboot_ratio; `subspace.top_k_subspace`), cohort supply is ample
  (`census_v1.json`: 11,275 val / 32,054 train usable, median cluster 3), GTF/hg38/checkpoint all present locally. A
  minimal EXISTENCE run is affordable on MPS.
- **But** the powered arm (EXISTENCE) is non-diagnostic; the diagnostic arm (E_partner DiD) needs ~900
  gene-clustered sites (never costed); E_decoy is dead/wrong-signed & decoy-sparse; SEMANTICS equivalence is
  unreachable (β_transport SD 0.774 → ±0.05 off the table; C8 CI-HW 0.0198 wall). The drafter already concedes the
  semantics-arm equivalence honestly (§5/§6 Endpoint B, §9 "likely Endpoint B") — that concession is correct and
  should be retained.

---

## Independent verification (my checks, not the reviewers')
- **FATAL-1:** `competition.py` L164-165 confirmed — `target_partner_distance` and `competitor_partner_distance`
  are separate fields; the two acceptors differ in position/distance by construction. R20 (position-geometry
  domination of L3H1 "preference") confirmed in CORRECTIONS. ✓
- **FATAL-2:** `run_a9_payload_semantics.py` L218-227 confirmed — synth branch builds `dr=β·coef·d0`,
  `deliver(Wv,dr)=dr@W_Vᵀ` (value path), `head_output` with frozen `Ap`, `inject_capture` patches L3H1 donor only;
  `payload.py` L40-70 confirms value-path plumbing. B2's hook (§2 L129-131) is a different, un-calibrated channel. ✓
- **Pilot data:** `behavioral_pilot_v1.json` — alt3ss M_comp +0.0339 (small-but-detectable), M_comp_minus_sham
  +0.0146 [−0.051,+0.088] (inconclusive), sham +0.0324, SD(M_comp) 0.306/0.317; BOTH M_comp +0.006 (bounded-null).
  `specificity_pilot_v1.json` — alt3ss intended-vs-decoy −0.0736 [−0.195,+0.013] (inconclusive, decoy 0.161 >
  intended 0.087), n_lacked_decoy 425. `ic2_calibration.json` — β_transport_matched_sd 0.7738. All ✓.
- **Operational trap:** `ls data/models` in worktree → No such file or directory; present at main checkout. ✓

---

## RECOMMENDATION (to Cam)

**Disposition: FIXABLE_THEN_RUNNABLE, but survivor = FALSE — do NOT execute this session, and do NOT market B2 as
"close C30."** The design red-team did its job: as written B2 cannot close C30 (the SEMANTICS naming arm is
non-identifiable / equivalence-unreachable at MPS budget — the drafter already concedes this), its calibration
does not travel the channel it interprets (FATAL-2), and its core direction is a position/distance axis
uncontrolled at every rung including the Endpoint-B fallback (FATAL-1). The two "real" specificity discriminators
are null/wrong-signed at pilot and the diagnostic one is underpowered (~900 genes), so the "existence+specificity
core runnable now" claim is unsupported: EXISTENCE alone is near-tautological steerability of `B`.

**What is legitimately salvageable, after must-fixes, as a future one-sided positive SEARCH (not confirmatory
closure):** rebuild the planted-positive on B2's own residual-add hook (FATAL-2, engineering); **caliper-match
preferred/less-preferred pairs on distance + absolute position and add a position/distance direction to the
control family** (FATAL-1, redesign of the estimator); label E_rel FORCED and rest specificity on a **powered**
E_partner (~900 genes) with a **preflight discovery-MDE gate** (declare inconclusive before spend if projected
CI-HW > SESOI, per the P3.4 remedy) plus a **hard preflight gate that partner-specific signed behaviour net of a
matched sham EXISTS on a fresh discovery draw** (currently inconclusive); measure SD(ΔB_j) directly and re-project
EXISTENCE power with the max-T penalty; family-correct the three SEMANTICS X's; **scope SEMANTICS out / defer** to
a training-time or external-assay design. Even fully fixed, the modal outcome is **Endpoint B (tighten C30), not
Endpoint A (close it)** — and the preflight behaviour-existence + geometry-controlled gates, given the pilots
(M_comp_minus_sham ≈ 0; position-confounded diff-in-means), are **likely to halt the run before confirmatory
spend**. That is the honest, valuable result: **C30's semantic identity is not closable by frozen-checkpoint
steering at MPS budget** — the naming claim needs training-time / external-assay evidence, and the paper should
**ship with C30 scoped out** rather than over-claim a steering closure. I recommend recording B2 as a hardened
design-of-record (FIXABLE, not this-session), scoping semantics out, and prioritizing the depth-first spine's next
bridge over spending confirmatory compute on an existence-only near-tautology.
