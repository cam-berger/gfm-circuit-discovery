# Preregistration — direct-edge P3 relational implementation and mediation

> **Tier: T2 — provenance. Frozen before any model output from this direct-edge P3
> intervention family.**

**Status: FROZEN 2026-08-03.** This protocol was written only after the independent fresh
behavioral audit returned `PASS` and
`MATERIAL_PREFERENTIAL_POSITIVE`. It does not amend the stopped middle-exon/skipping
protocol, reopen the old sealed crossed-lattice confirmation role, or modify C36's historical
present-day-use claim.

## 1. Question and claim boundary

The confirmed behavior is a direct-edge preference: at `SpliceBERT.1024nt`, masked-donor
coupling is materially larger when authentic donor and acceptor coding-boundary phases match
than when they differ in the frozen 120-nt crossed lattice. The mismatch-equivalence family did
not pass, so **only when coherent** is already forbidden.

This successor asks whether the learned period-3 position component causally implements that
preference as a relative carrier-register computation, and whether the effect travels through
L3H1's donor-query/acceptor-key routing rather than only through carrier-dependent values.

For biological boundary phases `p_D,p_A in {0,1,2}`, define

`delta = (p_A - p_D) mod 3`.

For carrier-register labels `q_D,q_A in {-1,0,+1}`, define

`r = (q_A - q_D) mod 3`.

The frozen high-level hypothesis is that useful coupling follows

`delta + r = 0 mod 3`.

This is a checkpoint-scoped model hypothesis. Coding-junction phase compatibility is an
arithmetic consequence of ligating coding exon boundaries, not a known splice-site-recognition
rule. A positive result does not show that the spliceosome checks frame, that SpliceBERT observes
translation or the ligated junction, that the carrier is coding-specific, or that the route is
the model's only route.

## 2. Prerequisite and exposure roles

Target exposure is authorized only by the immutable fresh-replication chain:

- `crossed_direct_edge_replication_audit.json` has `status=PASS`,
  `independent_audit_passed=true`, every target-independent gate true, and final class
  `MATERIAL_PREFERENTIAL_POSITIVE` or `ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN`;
- the cohort, construction audit, freeze package, attempts, native result, cell result,
  analysis, all manifests, and the final result audit match their frozen hashes;
- the six-file official `SpliceBERT.1024nt` checkpoint inventory and runtime are exact.

The already exposed 120-block **development** role from
`crossed_direct_edge_cohort_c1_c4_compact_v3.json` supplies implementation smoke,
calibration, and projected target power. Its genes are excluded from the fresh replication by
construction. Six development blocks are assigned to plumbing smoke and the other 114 to
development preflight using only SHA-256 of the block ID under frozen salt
`direct-edge-p3-smoke-v1`. Smoke may expose tensor shapes, mechanical errors, forward
determinism, and same-state round trips, but no `G`, `S_r`, `Omega`, or model-effect verdict.

All 384 blocks of `crossed_direct_edge_replication_cohort_v1.json` are the one-shot target.
No target block is selected, excluded, weighted, or ranked by its already observed `Psi_b`, a
sequence covariate, stop load, carrier response, activation, or any other outcome. A nested
24-block specificity subset (12 per strand, chromosome-round-robin after SHA-256 ordering under
salt `direct-edge-p3-specificity-v1`) is frozen before target execution for the expensive
signed-axis and matched-head nulls. Its identity is not chosen using behavioral values.

Development may change implementation only before the target package is frozen. After the exact
target code and package are frozen, the 114-block preflight is rerun with that byte-identical
code. Any failure stops target exposure; thresholds, signs, support, sample size, and null
families may not be changed.

## 3. Model-native readout and fixed coordinates

- Sequence length: 1022 nt; tokenizer length: 1024 including `[CLS]` and `[SEP]`;
  nucleotide-to-token offset: `+1`.
- Donor nucleotide: 392; donor query token: 393.
- True acceptor boundary: 512; corruption: `AG[510:512] -> AC`.
- Readout: `f = logit(G) - mean(logit(A),logit(C),logit(T))` at the masked donor.
- Coupling: `C = f(acceptor intact) - f(the same sequence with AG->AC)`.
- The previously validated `AG->AT` arm remains a bound behavioral sensitivity input; it is not
  silently replaced by `CC`, and `CouplingTask` is forbidden because its margin and corruption
  differ from this assay.
- Intact and AC states always receive the identical carrier intervention. Captures are tagged by
  block, biological cell, corruption state, carrier state, model layer/head, and sequence hash.
  An intact cache may never be patched into an AC input or vice versa.

## 4. Nucleotide-invariant carrier-register intervention

Extract `C3` independently from the learned position table by mean-centering, retaining rFFT bins
339--345 of the 1026-position table, and real inversion. Tokens, nucleotide sequence, splice
motifs, and every non-P3 position component remain unchanged.

The two equal-length boundary supports are defined in nucleotide coordinates:

- donor support `[32,424)` -> tokens `[33,425)`;
- acceptor support `[448,840)` -> tokens `[449,841)`;
- authentic bridge `[424,448)` and all positions outside the two supports are untouched.

Let signed source displacement be `s = source_index - target_index`. The carrier-register label
is frozen as

`q = -s mod 3`, represented by `{-1,0,+1}`.

This convention is outcome-independent: replacing `C3[x]` by `C3[x+s]` changes the effective
annotated state from `p` to `p-s=p+q`; equality of the effective donor and acceptor states gives
`delta+r=0`. An independent symbolic and index-level audit must verify this truth table.

For center `b in {3,6,9}` and orientation in `{source_plus,source_minus}`:

- `source_plus`: `s=+(b-q)`, implemented with amount `b-q`;
- `source_minus`: `s=-(b+q)`, implemented with amount `b+q`.

Every amount is in `[2,10]`; no analyzed cell is a zero-shift no-op. The complete
`q_D x q_A` grid travels together for every biological `p_D x p_A` cell, center, orientation,
block, and corruption state. Equal exposure over absolute biological phases and absolute carrier
registers cancels arbitrary **additive** donor-local and acceptor-local response tables. It does
not exclude every nonlinear separable local interaction; the bridge, health, hidden-axis, and
localized Q/K tests remain load-bearing for that reason.

## 5. Carrier estimands

Within each seven-source block, average equally over absolute `p_D,p_A,q_D,q_A`, centers, and
orientations to form

`G(delta,r) = E[C | (p_A-p_D)=delta, (q_A-q_D)=r]`.

For `r=0,1,2`, define

`S_r = G((-r) mod 3,r) - mean_{delta != (-r)} G(delta,r)`

and the primary estimand

`Omega = mean(S_0,S_1,S_2)`.

`Omega` alone is insufficient: all three `S_r` must be positive under simultaneous inference.
The opposite-handed ridge `delta=r` is frozen as a diagnostic and cannot replace the declared
handedness after outcomes.

Common-mode preservation is tested separately for each `q` by restricting to `q_D=q_A=q` and
computing the biological equality-diagonal excess. It means preservation of the preferred
diagonal, not exact equivalence of effect magnitudes. Center-specific, orientation-specific,
strand-specific, and donor-phase-specific estimates are reported.

The anchor geometry for mediation and Q/K work is fixed before development outcomes at
`b=6`, averaged over both orientations. It is not chosen from the six geometry results.

## 6. Development preflight and target authorization

All gates are evaluated with the final frozen code on the 114 development blocks. A failure is
`INSTRUMENT_UNRESOLVED` for the affected successor stage and cannot be repaired with target data.

### 6.1 Construction and mechanical gates

1. Both source cohorts and their controlling independent construction audits pass; role and gene
   disjointness reconstruct exactly from pinned GENCODE v50 and hg38 inputs.
2. A NumPy implementation and the Torch implementation of `C3` agree within `1e-5`; selected-bin
   and rejected-bin energy checks identify bins 339--345 rather than merely round-tripping the same
   tensor.
3. For every center/orientation/register, the audited signed displacement satisfies
   `q == -s mod 3`; the preferred ridge follows symbolically from effective phases.
4. Supports are disjoint and exact; bridge, special tokens, token IDs, nucleotide text, motifs,
   non-P3 position components, and positions outside support are unchanged exactly.
5. All 81 biological-by-carrier cells occur once per geometry. The complete design gives identical
   mean total squared intervention energy for `r=0,1,2` to relative tolerance `1e-8`; learned and
   signed-axis families use the same geometry.
6. The phase-aligned `q_D=q_A=0` family reproduces the no-surgery development `Psi` with a paired
   95% interval wholly inside `[-0.05,+0.05]`. This is failable because the retained P3 band is not
   an exact single Fourier bin.
7. Zero-delta projection patches and same-state full-head replacement reproduce their host forward
   within `1e-6`; q=+/-1 changes donor-query and acceptor-key head slices by nonzero RMS. Forced
   algebraic regroupings are labelled recorders, not positive controls.

### 6.2 Capability, health, sensitivity, and resolution gates

1. No-surgery development `Psi` has lower 95% bound above `+0.05`; phase-blind pooled AC coupling
   has lower bound above `+0.02`.
2. Under every interpreted carrier register, pooled coupling remains above `+0.02`; donor and
   acceptor intact-site margins remain above the native-source floors bound by the fresh
   replication.
3. On frozen non-site probes, true-base log-probability change has lower 95% bound above `-0.10`
   nat and accuracy retention is at least `0.80`. Logits, projections, attentions, and derived
   scores are finite; attention rows sum to one within `1e-5`.
4. Repeated forwards differ by at most `1e-8`; exact request counts, batching, token length,
   prefix, device, and state/spec alignment pass.
5. Development carrier `Omega` has lower 95% bound above zero, all three simultaneous `S_r` lower
   bounds exceed zero, all three common-mode point estimates are positive, and the target exceeds
   all 99 balanced S3 acceptor-label controls (seeds `2026080601..2026080699`; exact add-one
   `p<=0.01`) with target-minus-mean-control lower bound above zero.
6. Projected target `MDE80` is at most `0.05` for carrier `Omega`, total L3H1 mediation, routing
   mediation, value mediation, and Q/K `Omega`. Projection uses development block residual
   variance and the fixed target `N=384`; no effect magnitude enters sample-size choice.
7. The anchor A/V and Q/K preflight channels have positive point estimates in the declared
   direction and all capture/patch/decoy state tags pass. These are channel-alive gates, not target
   evidence.

## 7. Inference and null families

The independent unit is the seven-source block. Point estimates equal-weight blocks. Use 20,000
whole-block bootstrap draws within chromosome-by-strand strata, seed `20260806`. `Omega` gets a
two-sided percentile interval. The `S_r`, common-mode, center, orientation, strand, phase,
mediation, rescue, and specificity endpoint families use max-absolute-studentized simultaneous
95% intervals over each family. `MDE80=(1.96+0.8416212335729143)*SD` of centered stratified
bootstrap means.

Null families are never pooled:

1. **99 target-label controls.** Within every seed, all six S3 acceptor-phase relabelings occur
   exactly 64 times across the 384 target blocks. Values and intervention rows do not change.
2. **31 signed hidden-axis controls.** Seeds `2026080701..2026080731` orthogonally signed-permute
   the hidden axis of `C3`, preserving its waveform, spatial Gram matrix, row norms, support, and
   dose. They run at the anchor geometry on the frozen 24-block target subset. Native P3 remains
   present, so this tests learned-direction alignment specificity, not replacement by an
   alternative carrier. Learned `Omega` must exceed all 31 (`p<=1/32`) and its paired excess over
   their mean must have lower 95% bound above zero.
3. **31 matched heads.** The family contains all 15 non-target layer-3 heads, L2H15 as the known
   carrier predecessor, and three SHA-256-selected heads from each other layer after excluding
   those 16 and L3H1. It uses the same head width, tokens, states, carrier anchor, and 24 blocks.
   L3H1 must exceed all controls (`p<=1/32`) with paired mean excess lower bound above zero for an
   L3H1-specific Q/K claim.
4. **Decoy key window.** The true acceptor key window is nucleotides 509--512 / tokens 510--513.
   The bridge-decoy window is nucleotides 434--437 / tokens 435--438. The identical four-token key
   delta is added at the decoy rather than the true acceptor; there is no K/V swap.

## 8. L3H1 A/V conditional mediation

Run the anchor carrier grid and capture L3H1 head 1 query, key, and value projections separately
for intact and AC states. `R` is the phase-aligned `q_D=q_A=0` anchor capture for the same sequence
state and orientation; `T` is the requested `q_D,q_A` capture. The host is always the target
carrier-surgery forward. Four cells are named explicitly:

- `A_R V_R`: replace L3H1 Q,K,V by same-state reference captures;
- `A_T V_R`: target Q,K with reference V;
- `A_R V_T`: reference Q,K with target V;
- `A_T V_T`: target Q,K,V, equal to the raw target host within tolerance.

Each cell yields its own `G`, `S_r`, and `Omega`. Define

- total L3H1-mediated carrier effect:
  `D_total = Omega(A_T V_T) - Omega(A_R V_R)`;
- routing mediation: `D_route = Omega(A_T V_R) - Omega(A_R V_R)`;
- value mediation: `D_value = Omega(A_R V_T) - Omega(A_R V_R)`.

The interaction/regrouping is reported but is architecture-conditioned and never called binding.
`D_route` and `D_total` must each have lower 95% bound above `+0.05` and all three per-r routing
contrasts must have simultaneous lower bounds above zero. A **routing-specific** claim additionally
requires the simultaneous `D_value` interval wholly inside `[-0.05,+0.05]` with its own
`MDE80<=0.05`. A non-equivalent value contribution produces a mixed A/V classification, not a
routing-only claim.

## 9. L3H1 local Q/K relational test

At the anchor carrier geometry, use the `q=0,0` same-state host and hold all values and all
nonpatched projections fixed. Capture donor-query changes from donor-only register interventions
and acceptor-key changes from acceptor-only interventions. Add the head-sliced donor Q delta only
at token 393 and the acceptor K delta only at tokens 510--513. This constructs the complete
`q_D x q_A` Q/K factorial without patching a clean state into a corrupted state.

The primary Q/K readout is the same `Omega_QK`; all three `S_r,QK` and all three common-mode
diagonal scores are required positive. Differential registers must cyclically relocate utility to
`delta=-r`; a general loss is insufficient.

For each `q in {-1,+1}`, define two common-mode rescues on the native biological diagonal:

- add matched K after Q-only: `H(q,q,delta=0)-H(q,0,delta=0)`;
- add matched Q after K-only: `H(q,q,delta=0)-H(0,q,delta=0)`.

All four simultaneous lower bounds must exceed zero and their mean lower bound must exceed
`+0.02`. Applying the same K delta at the bridge decoy instead of the true acceptor must lose at
least `0.02` in the paired true-minus-decoy contrast. L3H1 must also pass the separate matched-head
family in Section 7.

## 10. Target decision table

All target results require every construction, capability, health, numerical, provenance,
chronology, and independent-audit gate.

| Outcome | Final mechanism classification | Licensed statement |
|---|---|---|
| Carrier `Omega` lower bound `>+0.05`; all `S_r` simultaneous lower bounds `>0`; common-mode, geometry, label-null, signed-axis, health, and resolution gates pass; A/V `D_total` and `D_route` pass; `D_value` is equivalent; Q/K cyclic relocation, four rescues, decoy, and matched-head gates pass | `P3_RELATIONAL_ROUTING_SPECIFIC` | At this checkpoint and in this assay, the learned P3 coordinate causally implements a relative-register preference through an L3H1 Q/K routing route; value changes at L3H1 are bounded below the 0.05 scale. |
| Same carrier and Q/K gates; `D_total` and `D_route` pass; value equivalence fails with an interpretable value contribution | `P3_RELATIONAL_MIXED_AV` | A relative-register P3 route through L3H1 is supported, but routing-only wording is forbidden. |
| Carrier gates pass but one or more localized mediation/QK gates fail with capable instruments | `P3_RELATIONAL_CARRIER_ROUTE_UNRESOLVED` | The P3 intervention causally moves the phase-preference ridge, but the declared L3H1 local route is not established. |
| All carrier capability gates pass and the entire `Omega` interval lies in `[-0.05,+0.05]` | `P3_CARRIER_BOUNDED_NULL` | This exact local learned-P3 surgery does not mediate a material relative-register effect in the frozen checkpoint/domain. |
| Any required control or preflight fails | `INSTRUMENT_UNRESOLVED` | No positive, null, or inverted mechanism conclusion. |
| Any other pattern | `INCONCLUSIVE` | No directional mechanism conclusion. |

A material opposite-handed result is reported as a preregistered falsification of the declared
orientation, never relabeled positive. No outcome can restore the word **only**, because the
behavioral mismatch-equivalence endpoint already failed.

## 11. Capability-conditioned negative scope

A powered negative closes only the declared `SpliceBERT.1024nt` checkpoint, masked-donor
direct-partner behavior, 120-nt authentic crossed geometry, local learned-P3 carrier surgery,
L3H1 A/V replacement, and local Q/K intervention family. It is not a biological no-go and does not
exclude a weak, distributed, shortcut-based, earlier-layer, nonlinear, context-limited, or
alternative-head approximation.

“Insufficiently embedded” is not a default rescue. It becomes evidence only in a separately
preregistered checkpoint trajectory, sibling-corpus contrast, or targeted-training experiment in
which both the behavior and the same causal route strengthen as the relevant biological variation
increases. Existing comparative carrier-origin evidence motivates that experiment but does not
substitute for it here.

## 12. Biological interpretation guardrail

Splice-site recognition is known to depend on splice motifs, branchpoint/polypyrimidine signals,
exon/intron definition, and regulatory sequence. Phase can nevertheless be statistically encoded
near coding boundaries, and coding sequence often carries a period-3 compositional signal. Because
SpliceBERT was trained on unspliced sequence with a masked-language objective and never directly
observed the ligated translated product, the strongest biological reading is a sequence-encoded
prior for coding-junction frame coherence—not a spliceosomal frame-check mechanism.
