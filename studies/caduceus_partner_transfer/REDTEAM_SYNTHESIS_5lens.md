# Red-team synthesis — Caduceus T1 behavioral-transfer prereg DRAFT

Adjudication of the 5-lens design red-team. Status: 1/5 returned. My call per issue: ACCEPT / MODIFY / REJECT + the resulting design change. This becomes the changelog presented to Cam.

## Lens C — estimand↔intent & SESOI (agentId ac6e4bf6113056c98) — verdict: FIXABLE

**C1 [HIGH] SESOI on the wrong estimand.** ρ≥τ floor sits on RAW C̄; the intent-bearing quantities are the SPECIFICITY contrasts (C̄−C̄_nonpartner, C̄−C̄_nondonor), gated only by CI-excludes-0. A large *generic* coupling + a tiny significant partner sliver both pass → false "QUALIFIES."
→ **ACCEPT.** Change: define ρ_partner=(C̄−C̄_nonpartner)/A and ρ_donor=(C̄−C̄_nondonor)/A; move the meaningfulness floor onto **ρ_partner** (primary). CI-excludes-0 stays as the existence bar on each contrast. This is the single most important fix so far — the floor must guard the intent-bearing quantity.

**C2 [HIGH] τ circularity / underspecified derivation.** τ set on same-model discovery, function unstated; naive derivations reduce to "beat the discovery point estimate" (reproducibility tautology) or are redundant with CI-excludes-0.
→ **ACCEPT.** Change: seal the τ-derivation FORMULA before unsealing discovery ρ. Leading option (B): meaningfulness = real ρ_partner exceeds the **permuted-partner null's ρ_partner upper CI** (beat the model's OWN no-partner-identity floor) — non-circular, no ρ_SpliceBERT needed. Consider also a modest absolute cross-model floor as a fraction of ρ_SpliceBERT ONLY if C3 resolves ρ as cross-model-valid. Decide in synthesis.

**C3 [MED] ρ=C̄/A_donor normalizes a LONG-RANGE effect by a LOCAL anchor.** A_donor = local donor-readout sharpness (per-architecture); legit within-model normalizer, NOT a valid cross-model effect size — but transfer's purpose is cross-model.
→ **ACCEPT (converges with my own flag + expected instrument lens).** Change: add a channel-matched long-range anchor **A_lr = margin_donor(clean) − margin_donor(downstream-intron-scrambled)**, sharing the exact downstream→donor pathway; report ρ_lr alongside. This ALSO doubles as the long-range positive control (certifies the SSM downstream→upstream channel). Scope ρ as within-model unless A_lr licenses cross-model. Big win: one construct fixes the anchor-validity AND the instrument-range-limit worry.

**C4 [LOW] verdict label "QUALIFIES — behaviour transfers" quotable as method-portability.**
→ **ACCEPT.** Rename → "QUALIFIES-FOR-PHASE-3 (behaviour-only; mechanism unproven)."

**C5 [LOW] ρ CI must be PAIRED gene-cluster bootstrap** (resample genes once, recompute BOTH means on that resample, ratio per replicate).
→ **ACCEPT.** State it; applies to ρ_partner/ρ_donor/ρ_lr too.

Strongest point (agent): specificity ARCHITECTURE is right (B-3 exclusion + distance/composition matching + donor-specificity); the flaw is the meaningfulness apparatus was built around C̄ not the specificity contrasts.

## Lens B — statistics & power (a55fdba7e78c72fe7) — verdict: FIXABLE

**B1 [HIGH] No joint-power/MDE certification on independent GENE count → a NULL is uninterpretable.** gap∈[60,150] is a severe filter; gene-clustering collapses exons; the strand-subgroup AND-gate (§6e) halves gene-n. Pipeline certified only to detect the LARGE planted A_donor, never the anticipated coupling ρ through the 4-arm conjunction. So "DOES NOT QUALIFY / honest architecture null" (the flagship deliverable) conflates true non-transfer with conjunctive under-power.
→ **ACCEPT — top statistical fix.** Change: on discovery, (1) report realized independent GENE count + top-gene exon share (discovery & confirmatory separately); (2) per-arm MDE at 80–90% power on gene-n; (3) JOINT power of the conjunction by gene-bootstrap simulation at anticipated ρ; (4) license a null as "informative" only if joint power ≥ ~0.8, else widen cohort (relax gap as a LABELED robustness arm / add chromosomes) or demote strand-subgroup AND-gate to a reported diagnostic.

**B2 [HIGH] τ sampling instability (converges with C2).** If τ≈ρ_discovery, regression-to-the-mean fails ~50% under a real stable effect — the R22 failure mode. Held-out ≠ sampling stability.
→ **ACCEPT.** Change: prefer a fixed a-priori fraction (τ data-independent); if data-derived, compute ρ across MULTIPLE independently-seeded discovery draws and set τ below the across-seed LOWER envelope; log the seeded spread in the freeze. Combine with C2 (permuted-null floor). **τ must never be the discovery point estimate.**

**B3 [MED] ρ ratio CI: use Fieller/delta-method, not naive percentile** (heavy-tailed denominator → skewed/unstable); resample genes carrying the (C, A_donor) PAIR jointly.
→ **ACCEPT.** Fieller's theorem for the ratio CI (returns unbounded interval exactly when denominator not clear of 0 = honest signal); report CV of A_donor; paired gene-cluster resample. Keep C̄/SE standardized effect for the power calc.

**B4 [MED] "ρ CI-excludes-0" is ALGEBRAICALLY EQUIVALENT to "C̄ CI-excludes-0"** (A_donor>0 forced by memorial ⇒ sign(ρ)=sign(C̄)) — the REFERENCE §6 tautology anti-pattern; I put it in §6(d).
→ **ACCEPT.** Drop "ρ CI-excludes-0" from the decision; existence = C̄ CI-excludes-0 (gate a); keep only the ρ≥τ FLOOR; report ρ CI as a diagnostic.

**B5 [LOW] FWER: ANDing 4 arms is an intersection-union test — already conservative.** No Bonferroni, no omnibus (the claim IS a conjunction). Only cost = joint power (→B1).
→ **ACCEPT.** Keep conjunctive/IUT uncorrected; STATE in the prereg it's an IUT (FWER-conservative) so no one "corrects" it later. (NB: differs from IC2's omnibus posture — correct here because conjunctive, not disjunctive.)

**B6 [MED] Dyadic dependence hinges on control construction.** As ported, `other_ag` is drawn WITHIN each exon's window (same gene) ⇒ one-way gene clustering DEFENSIBLE (unlike A9). But draft wording "non-partner REAL AG" is ambiguous — a GLOBAL cross-gene matched pool would reuse genes as partner+control ⇒ two-way dyadic dependence a one-way bootstrap under-covers (A9/R22 trap).
→ **ACCEPT.** PIN in the frozen prereg: matched non-partner AG + non-donor G drawn WITHIN the example's own window (paired, same gene). If any cross-gene pool is used, upgrade that arm to a two-way dyadic bootstrap. (Verify what coupling.py actually does.)

**B7 [MED] Permuted-partner FPR must permute at GENE/cluster level** (block permutation preserving within-gene structure), else anti-conservative.
→ **ACCEPT.** Gene-cluster block permutation; calibrate threshold to that clustered null with an out-of-sample split.

**B8 [LOW] Heavy tails + only 2 confirmatory chromosomes** → percentile miscoverage; can't chromosome-cluster with 2 super-clusters; paralog homology leakage.
→ **ACCEPT.** BCa/studentized bootstrap; report top-gene leave-one-out influence; prefer ≥3–4 confirmatory chromosomes; note paralog homology as scope caveat.

Strongest point (agent): scale-invariance strategy is correct & hard-won; A_donor plant travels the same channel; conjunctive IUT is the right FWER posture. Missed: joint-power certification on realized gene-n → the null is not yet licensed.

## Lens A — identifiability (ac81d7f721bb406fa) — PENDING
## Lens D — calibration (ab210900a4be6d8fe) — verdict: FIXABLE
(Verified load-bearing: Caduceus-PS IS MLM-trained — 15% mask, 80% [MASK], single-nt char tokenizer, AutoModelForMaskedLM. Masked-donor readout premise HOLDS.)

**D1 [HIGH] Long-range positive control MISSING — converges with C3 + my flag (now 3 lenses).** A_donor shares the READOUT channel but NOT the >=60nt PROPAGATION leg. A null coupling with A_donor passing = indistinguishable from an undetected long-range instrument limit, yet §6 routes that into "honest architecture null" — the headline deliverable is the one verdict calibration can't license.
→ **ACCEPT — the dominant must-fix.** Change: add a **long-range planted-positive that MUST be detected before any null is reportable**:
  - PRIMARY (Caduceus-native, same channel, SIGN-AGNOSTIC): **gross-downstream-disruption floor** — scramble the whole downstream acceptor region (AG + branch point + PPT / large block); require **|Δ margin_donor| > 0, gene-clustered CI-excludes-0**. Certifies the adapter can move the donor readout from that distance AT ALL. **Wire its FAILURE to a new INSTRUMENT-LIMIT branch in §6** (currently missing) so a dead long-range channel can't masquerade as biology.
  - COMPLEMENTARY (assay-logic oracle): run the identical downstream-corruption assay through the shared `coupling.py` on **SpliceBERT** (known +0.87) and require recovery — end-to-end validation of the long-range assay logic (caveat: exercises SpliceBERT's adapter, not Caduceus's SSM plumbing).
  UNIFY with C3: **A_lr = margin_donor(clean) − margin_donor(downstream-scrambled)** serves BOTH as this instrument-limit gate (|A_lr| CI-excludes-0) AND as the channel-matched long-range normalizer for ρ_lr.

**D2 [HIGH] Permuted-partner FPR null not distance/composition-matched → calibrates the WRONG contrast.** §4.2 reassigns "a random non-partner acceptor" (composite: breaks identity AND distance); under B-4 distance-decay random non-partners are FARTHER → null looks "more null" → FPR too lenient. But estimand 2 IS distance-matched.
→ **ACCEPT.** Change: build the permuted null with the SAME matched reassignment as estimand 2 — permute PARTNER IDENTITY holding distance + local composition fixed (break only the tested factor).

**D3 [MED] Missing MLM-health / mask-fill memorial.** `margin>0 at one donor` is too weak; sealed SpliceBERT had B3 (health ≥90% on masked probes); dropped here. (`coupling.py::mlm_health` already exists — port it.)
→ **ACCEPT.** Add failable on-path memorial: (a) [MASK] id exists + is the one used; (b) mask-fill top-1 accuracy on N random real ACGT positions >> 25% (preset floor); (c) objective==MLM from config. Into §8 preflight + --smoke.

**D4 [MED] Permuted-null gene-reuse/dyadic clustering unspecified + calibrate/test split not pinned to discovery** (merges with B6/B7).
→ **ACCEPT.** Pin clustering; two-way dyadic if cross-gene; pin the FPR calibrate/test split to the DISCOVERY cohort, never the sealed confirmatory.

**D5 [LOW] rc_plumbing_consistency honest & worth keeping, but PS symmetrization can MASK a coordinate off-by-one** (output stays ~equal despite wrong RC index) → doesn't independently guarantee readout-position correctness.
→ **ACCEPT.** Keep it, but add a **coordinate-SENSITIVE (non-RC-symmetric) memorial**: `margin_donor` high AT the annotated donor index, low when shifted ±1 nt on known canonical donors — catches the off-by-one RC symmetry hides.

**D6 [LOW] both_strands_present count>0 misses partial strand bias.**
→ **ACCEPT.** Strengthen to fraction band: minus-strand share ∈ [0.3, 0.7].

Strongest point (agent): inherits the sealed assay's local controls + matched nulls with discipline; scrupulously honest about RC memorial. Missed: a positive control for the long-range PROPAGATION that IS the novelty.

## Lens A — identifiability (ac81d7f721bb406fa) — verdict: FIXABLE
(Read the actual selectors: `run_sealed_test.py:58-66` A3 null = min|p−partner|, distance-only, NO side constraint; `circuit.py:530-541` A2 = another REAL donor GT distance-matched; `circuit.py:544-551` binding decoy = downstream-only, tol=40; `coupling.py:19-23` docs partnership ≈ strength+distance. RC math verified.)

**A1 [HIGH] OVERCLAIM in my draft: the local-composition / acceptor-STRENGTH confound is NOT blocked.** §1/§5 claim "distance- AND local-sequence-matched" but every ported selector distance-matches only ("motif-matched"="an AG present"). Partner is a stronger/more-canonical acceptor than a random distance-matched AG ⇒ positive partner-specificity is strength-confounded.
→ **ACCEPT — genuine draft error.** Resolution (proportionate for a Phase-1 transfer gate): (i) **DELETE the false "local-sequence-matched / composition blocked" language**; (ii) make the PRIMARY estimand-2 **distance-matched, IDENTICAL to SpliceBERT's A3** (fair transfer comparison — do NOT hold Caduceus to a stricter bar than the source result), and **state the acceptor-strength confound as an INHERITED limitation shared with the SpliceBERT +0.87**; (iii) add a **strength-caliper SECONDARY robustness arm** using MaxEnt `score3` (maxentpy already installed for A9/A9b: |Δscore3|≤caliper), with a **pre-committed non-identifiable reading**: matched-contrast≈0 under tight strength-matching ⇒ "non-identifiable given partnership≈strength collinearity," NOT "no transfer" (REFERENCE §5 discipline).

**A2 [MED] "Ports verbatim" is false for the control-selection POLICY** (lives in the runner, ≥2 different pickers exist). A new runner makes the matching algorithm an UNFROZEN free parameter.
→ **ACCEPT.** FREEZE the exact selector as code in the prereg (which function, distance reference to-donor vs to-partner, side constraint, caliper/tol). Don't state the estimand in prose the frozen code doesn't implement.

**A3 [MED] Common-support / Berkson selection on the matched subset.** Paired diff computed only on exons WITH a valid matched non-partner AG (pairs ⊊ X even at tol=40); exons lacking a decoy may be the isolated, largest-coupling partners ⇒ conditioning on "control exists" biases C̄−C̄_nonpartner. Marginal overlap check ≠ sufficient.
→ **ACCEPT.** PER-EXAMPLE support check (caliper satisfiable per exon); report retention fraction; pre-declare a dropped-vs-retained C̄ balance check; if they differ, report Berkson bias.

**A4 [MED] Estimand-3 mis-specified in my draft.** I wrote "distance-matched masked NON-DONOR G"; the ported A2 is another REAL donor GT (`other_gt`) at matched separation. An arbitrary non-donor G has a floored baseline ⇒ drop ~0 trivially ⇒ passes for the wrong reason.
→ **ACCEPT — draft error.** Estimand-3 control = **distance-matched OTHER REAL donor GT (`other_gt`), per SpliceBERT A2.**

**A5 [MED] Permuted-partner FPR not distance-matched** (converges D2).
→ **ACCEPT** (merge with D2).

**A6 [LOW] A3 null doesn't enforce SIDE.** min|p−partner| allows an UPSTREAM control; a bidirectional Mamba's forward/reverse passes need not propagate symmetrically ⇒ an upstream control at same |Δ| is not a valid distance control ⇒ confounds signed direction with partnership.
→ **ACCEPT.** Freeze A3 null to **same-side (downstream)** with a SIGNED-distance caliper (match the binding-decoy p>d+20 constraint). Important for directional-state SSM.

**A7 [LOW] RC demotion CORRECT (independent verification).** PS: f(RC(x))_{p,b}=f(x)_{L−1−p,comp(b)} identically ⇒ forward-vs-RC contrast =0 by construction; strand-specificity was never a SpliceBERT estimand (both strands fed in transcript orientation) ⇒ nothing lost. Residual risk: a mis-IMPLEMENTED memorial that reads logit(G) at mirrored index instead of logit(**C**=comp) at L−1−p would FALSE-fail.
→ **ACCEPT.** Write the exact readout transform (complemented base at mirrored index) in the memorial spec.

Strongest point (agent): fix the acceptor-strength confound (claimed, not implemented) before freeze — either strength-caliper + pre-committed non-identifiable reading, or honestly scope as distance-only with the strength confound stated. Missed: "ports verbatim" created false confidence; the strongest control is claimed, not implemented, and partnership is near-collinear with strength ⇒ a null could be uninterpretable (architecture null vs definitionally-near-zero contrast) without a pre-committed reading.

## Lens E — instrument/SSM (a66447aa66761142d) — verdict: FIXABLE
(Verified against Caduceus HF repo config/tokenizer/modeling code + paper + mamba-ssm runtime. feasibility: mask_token=YES, single_nt_vocab=YES, cpu_mps_fallback=**NO**.)

**E1 [HIGH] No long-range control → null branch UNIDENTIFIED (4-lens unanimous: E+D+C+my flag).** Leftward flow rides ONLY the reverse Mamba pass (BiMambaWrapper: mamba_rev on flip(1), flipped back, ADDED to forward); finite selective state (d_state=16, expand=2) + the "add" merge ⇒ range/decay suppression is a real failure mode. A_donor is local, passes even if the downstream→upstream channel is dead ⇒ "architecture null" is not identified against "instrument-range limit" (both = A_donor PASS + coupling CI-incl-0).
→ **ACCEPT — headline fix.** Adopt E's **distance-resolved transmission curve** (subsumes C3's A_lr + D1's floor): fix the masked donor-G readout; apply a strong generic real-base block perturbation (scramble/N a 10–20 nt window) at sliding downstream offset d∈{~10,30,60,100,150,200} nt; measure |Δmargin_donor(d)|, gene-clustered CI. (i) VISIBILITY: block adjacent → Δ large. (ii) CERTIFICATION: at 60–150 nt, Δ CI-excludes-0 and non-negligible vs near-distance ⇒ reverse pass carries downstream content to the required range. Surgical primary variant: non-specific 10 nt scramble AT the acceptor locus (exercises the exact channel, minus specificity). **Wire to §6:** coupling-null + transmission DEAD@60–150 ⇒ **INSTRUMENT-RANGE LIMIT** (re-run/redesign, no biological claim); coupling-null + transmission HEALTHY ⇒ **honest architecture/objective null**. Relabel A_donor "local readout+recognition sensitivity only."

**E2 [HIGH] Coordinate off-by-one: Caduceus has NO prefix token — trailing [SEP], prefix_tokens=0.** `build_inputs_with_special_tokens` has CLS commented out, returns `token_ids_0 + [SEP]` ⇒ nucleotide_to_token(p)=p, NOT p+1. SpliceBertAdapter HARD-RAISES on prefix<1 ⇒ naive port crashes or gets "fixed" to prefix=1 ⇒ **every masked readout reads the neighbouring nucleotide** → silently wrong margins. My draft's "fixed prefix"/"same coordinate contract as SpliceBERT" INVITES this.
→ **ACCEPT — critical.** New adapter: **prefix_tokens=0**, assert one TRAILING [SEP] (id 1). Memorial = **empirical known-position roundtrip** on a REAL batched/padded construct (mask a known canonical donor-G, confirm argmax/margin at the intended base), NOT arithmetic. Reword prereg: "fixed SUFFIX [SEP], prefix_tokens=0."

**E3 [HIGH] Left-padding silently corrupts positions under batching.** tokenizer_config `padding_side='left'` (SpliceBERT pads right). Reusing nucleotide_to_token(p)=p+prefix is wrong for every left-padded row (pad shifts indices). Any mixed-length batch breaks; correct only by accident at equal-length or bs=1.
→ **ACCEPT — critical.** Adapter **forces padding_side='right'** (or asserts equal length / bs=1). Memorial verifies a known readout position under a **deliberately mixed-length padded batch**.

**E4 [MED] NO reliable local CPU/MPS fallback; blocker is KERNELS not FLOPs.** `modeling_caduceus.py` UNCONDITIONALLY `from mamba_ssm...import Mamba`; config fused_add_norm=True+rms_norm=True (Triton RMSNorm → CUDA-only) + use_fast_path=True; causal-conv1d CPU known-broken; MPS unsupported. CPU workaround (disable fused_add_norm/rms_norm) SILENTLY changes normalization ⇒ logits no longer the model's.
→ **ACCEPT — corrects my "7.7M ⇒ CPU trivial" misframing.** **Colab (Linux+CUDA) is PRIMARY compute, not a fallback** (precedent: additive-frozen handoff). §8 feasibility preflight = hard gate. No MPS forcing. CPU only with logit-for-logit CUDA validation.

**E5 [LOW] RC memorial construction/tolerance** — RC-equivariance EXACT by param tying (RCPSEmbedding + RCPSLMHead fwd+F.linear(flip(x_rc),weight[complement_map])), but only over NUCLEOTIDE positions ([SEP]/pad not palindromic; fp32+scan noise).
→ **ACCEPT.** Build RC at NUCLEOTIDE level, re-tokenize (re-append [SEP]), compare only nt positions, small tol; keep labeled plumbing. (3rd independent confirmation the RC demotion is correct.)

**E6 [LOW] Vocab-id mapping.** Chars after 7 reserved specials: A=7,C=8,G=9,T=10,N=11; specials/pad 0–6,12–15. Wrong ordering ⇒ plausible-looking garbage margin.
→ **ACCEPT.** Resolve A/C/G/T ids via `convert_tokens_to_ids` at init; margin_readout_decodable asserts ids + margin_donor(clean)>0.

Strongest point (agent): RC demotion exactly right (mechanism confirmed). Missed by the design: conflates the MLM READOUT channel (A_donor certifies) with the long-range INFO-FLOW channel (reverse Mamba pass only) — identical in attention, distinct in an SSM, "the whole ballgame"; + 2 false SpliceBERT tokenizer assumptions (prefix≥1, right-pad — both FALSE).

---

# ===== MASTER INTEGRATED CHANGE LIST (all 5 lenses; feeds v2 prereg) =====
Verdicts: **fixable ×5, fatal ×0.** Core design sound; fixes harden it.

**HIGH — must-fix before freeze**
1. **Long-range transmission-curve control** (E1/D1/C3) + wire to a NEW §6 INSTRUMENT-LIMIT branch; relabel A_donor local-only. *[This makes the flagship null interpretable.]*
2. **Adapter coordinate correctness** (E2 prefix=0/trailing[SEP]; E3 right-pad; E6 dynamic A/C/G/T ids) + **empirical known-position roundtrip memorial under a mixed-length padded batch**. *[Phase-2 gate; would've silently broken everything.]*
3. **Effect-size floor on ρ_partner=(C̄−C̄_nonpartner)/A** not raw C̄ (C1); existence = CI-excludes-0 per contrast.
4. **τ sealed as a non-circular, sampling-stable FORMULA** (C2/B2): a-priori fraction and/or permuted-null floor, checked across seeded discovery draws; never the discovery point estimate.
5. **Joint-power certification** on realized independent GENE-n licenses the null (B1): report gene-n (disc+conf), per-arm MDE, conjunction joint power ≥~0.8 by gene-bootstrap; else widen cohort / demote strand-subgroup gate.
6. **Acceptor-STRENGTH confound honesty** (A1): delete false "composition-matched"; primary estimand-2 = distance-matched (== SpliceBERT A3, fair transfer, strength confound stated as INHERITED); secondary MaxEnt-score3 caliper arm w/ pre-committed non-identifiable reading.

**MED**
7. Drop tautological "ρ CI-excludes-0" gate (B4); existence = C̄ CI-excludes-0.
8. PERMUTED-partner FPR null: distance-matched + same-side + composition-held (D2/A5/A6); GENE-cluster block permutation (B7/D4); calibrate/test split pinned to DISCOVERY (D4).
9. Fieller/delta ratio CI + PAIRED gene-cluster bootstrap (B3/C5); report A_donor CV.
10. MLM-health memorial (D3): [MASK] id used; mask-fill top-1 ≫25% on random ACGT; objective==MLM (port `coupling.py::mlm_health`).
11. Estimand-3 control = other REAL donor GT `other_gt` per A2 (A4) — fix mislabel.
12. FREEZE exact control selector as code (A2); non-partner pool WITHIN-window/same-gene (B6/D4); same-side (downstream) signed-distance caliper (A6).
13. Per-example common-support/Berkson check (A3): retention fraction + dropped-vs-retained C̄ balance.
14. Compute: **Colab CUDA primary** (E4); §8 feasibility hard gate; no MPS; CPU only w/ logit-for-logit validation.

**LOW**
15. Coordinate-SENSITIVE memorial: margin_donor high AT donor index, low at ±1 nt (D5) — catches off-by-one RC symmetry hides.
16. RC memorial at nucleotide level, complemented base at mirrored index, small tol, nt-positions-only (A7/E5/D5).
17. both_strands_present → minus-strand share ∈ [0.3,0.7] (D6).
18. Rename verdict label → "QUALIFIES-FOR-PHASE-3 (behaviour-only; mechanism unproven)" (C4).
19. State the conjunction is an IUT (FWER-conservative; no Bonferroni/omnibus) (B5).
20. Prefer ≥3–4 confirmatory chromosomes (chromosome jackknife); BCa; paralog-homology scope caveat (B8).

**Cross-cutting verified facts:** RC demotion CORRECT (A7/D5/E5, 3 lenses). Caduceus IS MLM (15% mask, 80% [MASK]), single-nt vocab decodable. RC-equivariance exact by param tying. NO CPU/MPS fallback (kernels).

**Genuine forks for Cam:** (a) τ philosophy — permuted-null floor (within-model, recommended) vs fraction-of-ρ_SpliceBERT (cross-model; anchor-validity disputed by C3) vs existence-only; (b) strength confound — distance-matched primary + caliper-secondary (recommended) vs strength-primary; (c) confirm Colab-CUDA compute; (d) PH companion — LEAN DROP (strand-specificity out of scope for PS by architecture, never a SpliceBERT estimand).

## Provisional integrated changes (update as lenses land)
1. Effect-size floor → **ρ_partner** (C1); existence stays CI-excludes-0 per contrast.
2. **Seal τ as a FORMULA**, non-circular + sampling-stable: a-priori fraction and/or permuted-partner-null floor, checked across seeded draws; never the discovery point estimate (C2+B2).
3. Add **A_lr long-range anchor/control** (downstream-intron-scramble); scope ρ within-model unless A_lr licenses cross-model (C3; watch E).
4. **Certify JOINT POWER on realized gene-n** before any null is called informative; report gene counts; consider ≥3–4 confirmatory chromosomes; strand-subgroup gate → diagnostic if it kills power (B1,B8).
5. Drop tautological **ρ CI-excludes-0** gate; existence = C̄ CI-excludes-0 (B4).
6. **Fieller** ratio CI + **paired** gene-cluster bootstrap; C̄/SE for power (B3,C5).
7. **Pin controls WITHIN-window** (paired, same gene); dyadic bootstrap only if cross-gene pool (B6).
8. **Gene-cluster block permutation** for the partner FPR null (B7).
9. Rename verdict label → "QUALIFIES-FOR-PHASE-3 (behaviour-only)" (C4).
10. State conjunction is an **IUT** (no Bonferroni/omnibus) (B5).
