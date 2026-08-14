# Corrections — SpliceBERT acceptor coupling

Permanent record of what was wrong, bounded, or mis-specified. Nothing here is edited away; wording that
was retracted stays retracted.

**`ACR*` numbers are local to this study and do not continue the donor study's own correction sequence.**
They were originally written with an `AR1-` prefix and renamed for two reasons: that form was
confusable with the donor study's `PREREGISTRATION_AR1_CORRUPTION_OPERATOR_SENSITIVITY` and its `DONA`
arm, and its dash-joined numerals parsed as a stale correction-range assertion under
`tests/test_ledger_consistency.py::test_no_document_understates_the_correction_range`. The rename is
cosmetic; no correction's content changed.

## ACR1 — R1's power gate is pointed at the effect instead of at the nulls, and returns `INSTRUMENT_UNRESOLVED` on a sound run (2026-08-05)

**Outcome in one line: a specification defect in a frozen decision rule, found by executing it, and
deliberately NOT repaired post-hoc.**

`PREREGISTRATION_R1_REVERSE_COUPLING_GATE.md` §6.3 reads: *"Observed `MDE80` on either paired difference
`> SESOI` → `INSTRUMENT_UNRESOLVED`."* On the settled run that clause fires:

| quantity | interval | `MDE80` |
|---|---|---|
| paired difference vs readout control | `[+0.9628, +1.1113]` | `0.106` |
| paired difference vs partner control | `[+0.9334, +1.0816]` | `0.105` |
| control — decoy-acceptor readout | `[-0.0327, -0.0063]` | `0.019` |
| control — decoy-`GT` corruption | `[-0.0005, +0.0270]` | `0.029` |

**Why it is a defect.** A power gate exists so that a null is not an underpowered null. The
discrimination that needs power here is *"is this control null, or is it SESOI-sized?"* — and on the
control arms the instrument resolves that with room to spare (`0.019` and `0.029` against a `0.05`
band). The paired differences are not a null being tested; they are a heavy-tailed effect of size `1.0`
(`coupling.py` records the forward effect as mean `0.436` / median `0.051`), whose bootstrap SD scales
with the effect itself. Demanding `MDE80 ≤ 0.05` there asks the study to resolve a `0.05` shift inside a
`1.0` heavy-tailed effect — precision the question never required and that no achievable `n` on this
cohort would buy.

**Why it was not repaired.** The same preregistration had already been corrected once post-hoc (§7.1:
the `GO` conjunct requiring each control's CI inside `±SESOI`, removed after pilot 2 breached it by
`0.0009`). A **second** post-hoc correction, again in the direction that enables `GO`, would make the
document worthless as a preregistration regardless of how sound each individual argument was. The rule
was executed as frozen and the verdict is reported as it fell.

**What the run establishes despite the verdict.** All eight memorials PASS, including the arithmetic
instrument anchor (forward `Y_D = +0.9436` against AR1's val `+0.9909`, tolerance `±0.15`), zero planted
motifs across 970 constructs, planted-positive, and a permuted-label FPR. `AC1`–`AC3` carry the
measurements with the verdict stated in the same row rather than in a footnote.

**Disposition.** Closing this requires a **new** preregistration (`R1b`) that gates `MDE80` on the
control arms, frozen before it runs, executed on a fresh val draw. It is not an amendment: R1 is frozen
and hash-verifiable, and §6.3 stays as written.

**One thing this cost nothing.** The §7.1 correction turned out to be **immaterial on the settled
cohort**: the decoy-`GT` control landed at `+0.0129 [-0.0005, +0.0270]`, inside `±SESOI`, so the drafted
rule and the frozen rule return the same verdict. The artifact records
`verdict_under_drafted_rule = INSTRUMENT_UNRESOLVED` alongside `verdict`. The disclosure in §7.1 stands
regardless — the rule was still touched after its governing number was seen, and a correction that
happens not to matter is not a correction that was licensed.

## ACR4 — AC2's mechanism clause was not identified by its own panel, and is refuted once the missing cell is run (2026-08-05)

**Published wording, now retracted:** *"The reverse-direction operator's effect is governed by whether
`seq[d+1]` is rewritten, **and by nothing else**… every alt that rewrites it lands at `ρ ≈ 0.99`."*

**The panel could not have identified that.** Across all nine alts in `OPERATORS`, two coincidences held
without exception and neither was noticed:

1. `z[1] == "T"` **iff** `edit_size == 1`. "Keeps the `T`" and "changes one base" were the same set.
2. Every alt that rewrote the `T` **also destroyed the `G`** (all nine have `z[0] != "G"`). "Rewrites the
   `T`" and "destroys both bases" were also the same set.

So the panel contained **two of three reachable cells** — *destroy both* (`ρ≈0.99`) and *destroy `G`
only* (`ρ≈0.67`) — and three different mechanisms predicted identically. The missing cell is *destroy the
`T` while keeping the `G`*, which requires `z[0]=="G"`: `GA` (admissible where `seq[d+2] != "G"`, 37.0%
of val donors) and `GG` (where `seq[d+2] != "T"`, 84.0%). `GC` is excluded because it **is** the
non-canonical U2 donor — 0% admissible under R44's own rule, which is a definition, not a power failure.

**The missing cell, run (R2, `r2_donor_base_decomposition.json`, R1b's own draw-0 cohort):**

| alt | destroys | `edit_size` | n | ρ vs `AC` |
|---|---|---|---|---|
| `AC` | `G` and `T` | 2 | 974 | `1.000` (reference, forced) |
| `AT` | `G` only | 1 | 974 | `0.672 [0.648, 0.699]` |
| `GA` | `T` only | 1 | 360 | `0.857 [0.821, 0.892]` |
| `GG` | `T` only | 1 | 820 | `0.834 [0.807, 0.860]` |

**What survives.** Edit size is **not** the explanation: `AT`, `GA` and `GG` are all one substitution and
split `0.672` against `~0.845`, non-overlapping. The confound noted at (1) is broken in AC2's favour.

**What is refuted.** "Rewriting `seq[d+1]` lands at `ρ ≈ 0.99`" is **false**. `GA` and `GG` rewrite the
`T` and land at `~0.845`. Reaching `1.000` requires destroying **both** bases, not the `T`.

**The corrected reading, which is stronger than the wrong one.** The reverse coupling responds to
destruction of **either** base of the donor `GT`, **strongly sub-additively**: `G` alone `0.672`, `T`
alone `~0.845`, both `1.000` — the singles sum to `1.52`, so the operator saturates. And the **`T`
contributes more than the `G`** (`0.845` vs `0.672`), which is not the ordering the motif's canonical
name suggests. AC2 is rewritten to this and its status is unchanged at established, because the
dose-response itself replicated three times; only its *mechanism attribution* was wrong.

**Residual confound, stated rather than buried.** `GA`/`GG` are the only admissible `T`-only alts and both
**retain the `G`**; a "destroy the `T` and also the `G`" cell is by definition the both-bases arm, so
"keeps `G`" is not separable from "destroys `T` alone" on this operator family. `GA` and `GG` also sit on
different admissibility sub-cohorts (37% vs 84%) and are reported separately for that reason; their
agreement (`0.857` vs `0.834`) is the only cross-check available.

**Where this came from.** The confound was found by an adversarial design review, not by the runs. Both
R1 and R1b passed all their memorials with the defective wording in place, because **no memorial tested
whether the panel identified the claim the panel was quoted for.** That is the same shape as `ACR3`: a
quantity no gate was watching.

## ACR3 — AC3's "19%" was one draw; the selection effect's direction holds but its magnitude does not (2026-08-05)

**Published wording, now corrected:** *"On the `CC`-admissible stratum — 17.7% of sites — `Y_A = +0.8294`
against `+1.0200` on the full cohort: **19% larger** once the R44 admissibility filter is escaped."*

R1b's draw 0 recomputed the same contrast on a fresh cohort and returned `0.9372` on the stratum against
`0.9763` on the full cohort — a **4.0%** shortfall, not 19%.

| run | stratum `Y_A` | full-cohort `Y_A` | shortfall | stratum share |
|---|---|---|---|---|
| R1 | `0.8294` | `1.0200` | **18.7%** | 17.7% |
| R1b draw 0 | `0.9372` | `0.9763` | **4.0%** | 19.6% |

**The direction reproduces; the magnitude does not.** `AC3` is rewritten to claim the direction and to
quote the range `4–19%`, and its status drops from `established` to `bounded`.

**Why this happened, and it is not a surprise in hindsight.** The stratum contrast was carried as a
**memorial**, not as an endpoint. Memorials in this study are checked for pass/fail, never powered: the
stratum is ~19% of the cohort (n≈172–190), it was never given a CI, and no draw-stability requirement
touched it — §5.9 gates the *verdict class*, and this quantity does not enter the verdict. A number that
no gate protects is a number that can drift, and this one drifted by a factor of four and a half.

**The general lesson, which is the reusable part.** R1b added draw stability precisely because R1's
partner control had flipped significance between draws — and that instinct was right, but it was applied
only to the decision layer. Every quantity a claim row quotes needs the same treatment or an explicit
"single-draw, unpowered" label. On R1b's own three draws the partner control turned out extremely stable
(`0.0147 / 0.0156 / 0.0144`, SD `0.0006`) — so **pilot 2's `+0.0284` was the outlier, and the instability
that motivated §5.9 was itself a small-sample artefact.** The memorial that drifted was a different one,
and it drifted because nothing was watching it.

**Disposition.** A stable estimate needs the stratum contrast run as its own powered endpoint with a CI
and multiple draws. **Not pursued** — it bounds a limitation of a superseded secondary arm (`DONA`) and
changes nothing about `AC1` or `AC2`. Recorded as a bounded row, not a live loose end.

## ACR2 — pilot 1's other-acceptor control was mis-constructed and its number must not be quoted (2026-08-05)

Pilot 1 built its site-specificity control from `real_acc[0]` — the first *real annotated* acceptor in
the window — with no distance matching. An audit over 1,200 val exons put those sites at a median offset
of `+157` from the donor against the partner's median gap of `95`, with **36.9% upstream of the donor**.
Its reported `+0.0434 [+0.0023, +0.0947]` therefore mixes partner identity with a large and
signed distance mismatch.

Superseded by the C2 idiom imported from `run_sealed_test.py`, which builds both controls from
**spurious distance-matched motifs** rather than real annotated sites (`matched_decoy_acceptor` for the
readout arm, `matched_decoy_donor` for the corruption arm). **`+0.0434` must not be quoted.** It appears
in `DECISIONS.md` as provenance only, labelled there.

## ACR5 — R3's preregistered noising second estimand was not run; AC4 is mean-ablation only (2026-08-11)

**Frozen requirement.** `PREREGISTRATION_R3_PAIRED_COMPONENT_SCAN.md` §5.4 requires the detected set
plus rank 1 in each direction to receive a noising second estimand. A component whose classification
disagrees across the two interventions must be reported as estimand-dependent.

**Execution.** `r3_paired_component_scan.json` contains the paired mean-ablation scan only. No R3
noising artifact exists. The artifact and its manifest say noising was "deferred"; deferral is not
execution, and no amendment removed the frozen requirement.

**Correction.** AC4's four-way partition and rank-1 identifications are **mean-ablation results only**.
They do not establish that any `SHARED`, `FORWARD_SPECIFIC`, or `REVERSE_SPECIFIC` label is stable across
intervention estimands. Nothing from R3 licenses a claim that ablation and noising agree, a mediated
reverse mechanism, or component equivalence across directions. The mean-ablation measurements, Holm
partition, and verification-tier status remain as recorded.

**Disposition.** Do not edit the frozen R3 protocol or backfill its immutable artifact. R4 was designed
to compare both intervention idioms, but its protocol was never frozen and is now retired; its exploratory execution did
not produce a licensed result (`DECISIONS_R4_EDGE.md`, `D10`–`D11`). A future frozen successor must run
both estimands or narrow its claim before observing the outcome.

## ACR6 — AC4 must not inferentially order ranks 2 and below (2026-08-11)

**Published wording, now corrected:** AC4 called `L2H15` "rank 2 in both" and `L1H12` "third."

**Frozen rule.** `PREREGISTRATION_R3_PAIRED_COMPONENT_SCAN.md` §5.3 authorises identification of rank 1
in each direction. It requires ranks 2+ to be reported only as a detected set and explicitly says the
arm is not powered to order them. The original AC4 row repeated that boundary in its final column while
violating it in the claim sentence.

**Correction.** `L2H15` has large mean-ablation effects in both directions (`0.473` / `0.505`), and
`L1H12` also has positive effects (`0.211` / `0.259`). Those values may be reported; "rank 2," "third,"
and any inferential ordering below `L3H1` may not. `L3H1` remains the identified rank-1 head in both
directions.

**Disposition.** AC4's wording now follows the frozen decision rule. No estimate, partition label,
artifact, or evidence tier changes.

## ACR7 — AC3's selection direction reverses on R1b draw 2 (2026-08-11)

**Published wording, now retracted:** after `ACR3`, AC3 said that the `CC`-admissible stratum had a
smaller reverse effect on every examined draw, with a `4–19%` shortfall.

That correction had inspected only R1 and R1b draw 0. The committed R1b artifact carries the same
memorial for all three draws:

| run | stratum `Y_A` | full-cohort `Y_A` | signed shortfall |
|---|---:|---:|---:|
| R1 | 0.829410 | 1.020039 | +18.69% |
| R1b draw 0 | 0.937206 | 0.976302 | +4.00% |
| R1b draw 1 | 1.063371 | 1.098718 | +3.22% |
| R1b draw 2 | **1.053406** | **1.004754** | **−4.84%** |

The direction therefore does **not** reproduce. The stratum contrast was never a powered endpoint, has
no gene-clustered interval, and did not enter the verdict. A pass/fail memorial that merely reports the
values cannot establish its sign.

**Correction.** AC3 is retracted. The four values may be reported as descriptive evidence that the
operator-admissibility filter changes cohort composition, with an observed signed range of about
`−4.8%` to `+18.7%`; no stable direction is licensed. This supersedes `ACR3`'s “direction holds”
disposition without altering the immutable artifacts.

## ACR8 — R1/R1b are gates, and their permutation memorial is not gene-preserving (2026-08-11)

**Two distinct defects were being hidden by “established: behaviour; all memorials pass.”**

First, the frozen R1 protocol says explicitly: *“This is a GATE, not a claim,”* that the result cannot
reach established status, and that `GO` only authorises a mechanistic battery. R1b supersedes the power
rule but inherits the task and verification scope. Its validation cohort is chr11/16/19. The
`+0.9682` forward quantity in R1b is a same-cohort arithmetic anchor reproducing the earlier validation
effect; it is not measured on C1's sealed chr8/9 cohort.

Second, both runners describe their null as partner/decoy labels “permuted within gene,” but implement
one independent sign flip per **row**:

```python
flip = np.array([1 if rng.random() < 0.5 else -1 for _ in range(len(own))])
permuted = diff_partner * flip
```

R1b has 974 rows but 645 genes, so rows from the same gene can receive different labels. The subsequent
gene bootstrap does not repair the randomisation unit. The artifact's `permuted_label_fpr` memorial is
therefore not the frozen cluster-preserving null, and “all nine memorials pass” is a self-reported
artifact field rather than a conformance conclusion.

**Correction.** AC1 is a bounded validation-gate observation, not an established or confirmatory claim.
Its large effect and recorded controls remain reportable with both R1b's `GO` and
`verdict_under_R1_rule=INSTRUMENT_UNRESOLVED`, but protocol conformance and familywise FPR are unresolved.
A successor must assign one exchangeable sign per gene and pass those signs through the actual decision
path. The sealed split remains unopened.

## ACR9 — R3's calibration path cannot license AC4's partition (2026-08-11)

`ACR5` recorded the absent noising estimand and `ACR6` removed rank ordering below rank 1. A source audit
found four additional protocol departures that reach the decision itself:

1. The frozen specificity memorial requires the pseudo-coupling dispersion to be smaller in **both**
   directions. The runner checks only `frac_D` against `drop_control_D`.
2. The permuted-direction FPR uses `(base_rp - base_rp) * flip + scalar_noise`: an exact zero vector plus
   a constant. It never permutes the observed component-by-direction effects and never asks whether
   **any** component clears the Holm family.
3. The planted positive is likewise a zero vector plus `3×SESOI`; it bypasses component construction,
   Holm correction, health exclusion, and the four-way classifier it is supposed to calibrate.
4. `SHARED` is assigned whenever an **unadjusted** percentile interval lies inside `±SESOI`. Holm is
   used only for the nonzero direction-specific branch. The nominal 73-member set is therefore not a
   familywise equivalence partition under the protocol's stated “Holm-corrected Δ” rule.

The frozen decision rule is unambiguous: any memorial failure yields `INCONCLUSIVE` and licenses no
component statement. A memorial incapable of failing on the target failure mode is not a passing
calibration.

**Correction.** The nominal `73 SHARED / 17 UNRESOLVED / 4 FORWARD_SPECIFIC / 2 REVERSE_SPECIFIC`
partition and both rank identifications are retracted as inferential claims. The per-head mean-ablation
point estimates remain an exploratory candidate screen: they nominate `L3H1`, `L2H15`, and `L1H12`,
among others, but do not establish equivalence, direction specificity, rank, mediation, or wiring.
`L5H11` and `L4H2` had near-zero/negative reverse point estimates under this one screen; “contribute
nothing” is not licensed. This supersedes the dispositions in `ACR5` and `ACR6`; no frozen protocol or
immutable artifact is edited.

## ACR10 — committed acceptor artifacts do not pin a reconstructible code/checkpoint state (2026-08-11)

The R1, R1b, R2, and R3 manifests record `git_dirty: true`. Their model input points to the checkpoint
**directory**, which the file-only manifest hasher records as `missing: true`, rather than hashing the
exact `pytorch_model.bin` (and configuration/tokenizer files where relevant). The genomic inputs and
output hashes are present, so artifact integrity can be checked; the precise executable code and model
bytes cannot be reconstructed independently from those manifests.

The AC4 result row in `results/MANIFEST.csv` was additionally labelled `mediation`, contradicting both
the artifact and ledger. It is an instrument-unresolved causal candidate screen. R4's reference smoke
was labelled `calibration`; it is an instrument artifact from a draft exploratory runner.

**Correction.** The manifest index now uses the repository evidence vocabulary (`Behaviour` for R1/R1b/R2,
`Instrument` for R3 and R4), while the immutable run manifests remain untouched. Future acceptor runs
must start from a clean commit, hash the exact checkpoint files and output, and record a versioned
artifact path. No existing result is promoted by this metadata repair.

## ACR11 — R3's reverse health control scored a different masking policy (2026-08-11)

R3's reverse endpoint masks the `G` of the intron-mate acceptor while a component is intervened on.
Its reverse health helper instead encoded each random probe as the sole masked position. The acceptor
was therefore visible during the health pass, and the model forward pass was not the one whose
competence the gate purported to protect. A high score could coexist with damage specific to the
double-mask policy used when a non-site probe is added to the masked-acceptor behavior.

**Required correction.** The implementation in this tree still uses the old single-mask health policy;
no code repair is claimed here. A future successor must keep the acceptor `G` masked while adding one
genuinely masked non-site probe, exclude both splice sites from probe selection, and add a behavioral
test that inspects the encoded mask count and positions. It must then rerun health under every
decision-bearing intervention. None of this can retroactively validate R3: its immutable health
measurements used the old policy, adding another reason AC4's partition remains instrument-unresolved.

## ACR12 — AC2's corrected missing cell is a post-outcome screen, not an established result (2026-08-11)

`ACR4` correctly refuted the earlier claim that rewriting only `seq[d+1]` restores the full two-base
effect. Its disposition nevertheless left AC2 “established,” citing replication of the older operator
panel. The decisive `GA`/`GG` cells were designed after that panel's outcome and were run once on R1b
draw 0. The older `AT`/`CT`/`TT` repetitions do not independently replicate the new comparison between
`G`-only and `T`-only interventions.

**Correction.** AC2 is **bounded post-outcome causal operator evidence**. The observed cells license the
descriptive conclusion that this validation cohort responds to either donor base, sub-additively, with
larger `GA`/`GG` than `AT`; they do not license a stable nucleotide-role or biological-mechanism claim.
This supersedes only `ACR4`'s status disposition. Its numerical correction and the immutable R2
artifact remain unchanged.
