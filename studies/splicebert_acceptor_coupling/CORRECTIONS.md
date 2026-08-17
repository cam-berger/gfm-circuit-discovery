# Corrections — SpliceBERT acceptor coupling

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

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
path. The later R1c masked-acceptor target outcomes remain unscored; chr8/9 itself was already used by
donor `C1` (`ACR18`).

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

**Correction.** `reverse_mlm_health` now keeps the acceptor `G` masked and adds one genuinely masked
non-site probe, with both splice sites excluded from probe selection. A behavioral unit test inspects
the actual encoded mask count and positions rather than searching source text. This code repair does
not retroactively validate R3: its immutable health measurements used the old policy, adding another
reason AC4's partition remains instrument-unresolved. A successor must rerun health under every
decision-bearing intervention.

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

## ACR13 — corrected R1c/R3b instruments are unrun, and their split, cohort, and Sri-provenance boundaries must remain explicit (2026-08-11)

The first documentation pass described R1c and R3b as implemented successors without consistently
separating **instrument readiness** from **new model evidence**. It also left three boundaries too easy
to collapse: R1c's chromosome seal versus R3b's within-validation gene buckets, historical R3's exact
cohort identity versus the current R3b census, and current gene-disjointness versus the anonymous Sri
GUE sample.

**R1c correction.** R1c is a draft, behavior-only confirmation instrument. Its production preflight is
fixed to 400 outcome-blind examples on validation chr11/16/19; only a future frozen target may use 400
examples on sealed chr8/9. The runner now binds the exact whole-gene endpoint-scale-normalized family, structured
plant, conservative null declaration, health and construction gates, runtime contract, protocol, and
instrument fingerprint. A draft or partially pinned preflight cannot authorize target. **No R1c model
preflight or target has run, no artifact controls a result, and no `AC*` claim is created.** The live
annotated exon-mate acceptor is decision-bearing; the historical near-floor spurious-`AG` readout is
retained only as a descriptive continuity arm.

**R3b correction.** R3b is a draft focused head+MLP closeout instrument. Before current-cohort
selection it must reproduce historical R3 at exactly **974 rows / 645 genes** with ordered example-ID
SHA-256 `161dff038d747dbaa7b91a96d24b34d45f55de5482b5140410641354e46d2a8e`; count or hash drift aborts,
and every represented gene is excluded. Both pseudo controls are now required to fall within a 40-nt
distance caliper. They are described truthfully as **non-partner sequence motifs**, not guaranteed
unannotated non-sites; annotated overlap is recorded rather than used as an outcome-dependent filter.

An outcome-blind temporary construction check was previously summarized as proving bucket capacity.
No immutable census artifact or manifest binds that check to exact GTF/FASTA and transformation bytes,
so its exact counts and the phrase “model-free cohort readiness” are withdrawn. The runner implements
the exhaustive census and requires each 500-row prefix to contain at least 30 genes; the first conformant
preflight must establish and bind whether those gates pass. No adapter or model outcome was read, no
model preflight or target has run, and no head or MLP conclusion is licensed. Both planned buckets use
validation chr11/16/19. Bucket 1 is a held-out **gene bucket**, not the R1c target cohort; R3b
does not score the R1c masked-acceptor target outcomes on chr8/9.

**Sri provenance correction.** `L5H8`, `L3H5`, `L0H13`, and `L0H14` enter R3b only as transferred
exploratory hypotheses from commit `6a136390`. The historical branch retains no gene-bound identities
for the adaptively reused 200 GUE rows, so overlap with R1c/R3b is unknowable. Excluding historical R3
genes and separating current buckets establishes neither independent replication of Sri nor
gene-disjointness from its anonymous sample.

**Disposition.** This is a pre-run documentation and instrument correction. It does not rewrite any
immutable R1/R1b/R2/R3 artifact and does not soften any earlier failure or retraction: AC1 remains a
bounded validation gate, AC2 remains a bounded post-outcome screen, AC3 remains retracted, and AC4's
partition remains instrument-unresolved. Promotion requires a newly frozen protocol, both exact pins,
a fresh conformant preflight, and the separately authorized target appropriate to that program.

## ACR14 — R1c development v2 failed an invalid exon-equivalence gate and an effect-bearing calibration plant (2026-08-15)

R1c's first complete 400-row development execution is preserved as an **instrument failure**, not a
behavior result. The draft/unpinned validation artifact is
`tmp/r1c_preflight_fullgtf_dev_v1.json`, SHA-256
`13fe7c5f0599a600d3fad01979c59ba2fc2e17e0c614e7055372c419755583f9` (sibling manifest SHA-256
`5f5cd65a45a0de34b77446785e9b18a38d74c8fce63b282586e23c41623eceb7`). It used the full GENCODE v50
GTF, 400 validation rows / 321 genes, stayed on chr11/16/19, and returned `INCONCLUSIVE`. Because the
protocol was still draft, neither reviewed seal pin was installed, the worktree was dirty, and this
was the instrument-development cohort, these numbers are historical diagnostics only and create no
`AC*` row.

The diagnostic means were paired-donor effect at the downstream acceptor `+1.0722`, matched
non-partner-donor effect at that acceptor approximately zero (`-0.00004`), and paired-donor effect at
the upstream exon-mate acceptor `+0.3180`. The old v2 design incorrectly required that last, live
annotated readout to be equivalent. That gate failed because the exon-mate response was materially
positive; it is not evidence that the paired-donor endpoint failed. The old forward masked-donor
anchor also ran, but it is irrelevant to the acceptor-only question and is removed from the successor.

The production-path plant also failed `0/200`. The old outer trials sign-flipped effect-bearing raw
six-row outcomes, and then added independent constants directly to already-derived rows. That path was
not translation-invariant and did not preserve an underlying readout-by-edit factorial. It therefore
did not test the intended R1c declaration with a coherent synthetic 2x2.

**Pre-freeze correction.** R1c v3 is acceptor-only. It measures the four raw cells `Y_PD`, `Y_PN`,
`Y_XD`, and the previously unmeasured `Y_XN`; derives the sole primary family
`[Y_PD, Y_PN, S_P, I]`; treats `Y_XD`, `Y_XN`, and `S_X` as descriptive; and licenses a core paired-donor
path without requiring `I`, plus a nested downstream-preference path that requires positive `I`.
Outer trials now center each raw cell endpoint, share one sign across all four cells within a gene,
and derive the family after flipping. The plant was fixed **before `Y_XN` was scored** at raw-cell
delta `[0.40, 0, 0, 0]`, which maps exactly to primary `[0.40, 0, 0.40, 0.40]`.

Interaction interpretation is additionally restricted, before scoring, to outcome-blind dyads with
`abs(exon_length - intron_length_down) <= 20 nt`. The corrected exhaustive scan found 6,064 rows on the
authorized validation chromosomes, 5,695 comparator-complete rows, and 1,567 dyad-eligible rows / 832
genes, with the preserved first 400 containing 321 genes. The production runner scans the
already-authorized chromosomes before filtering, preserves the v2 cohort ordering so the redesign does
not silently redraw validation after three cells were observed, and binds every census hash.

**Post-redesign development execution.** The draft v3 path subsequently ran once on those 400
validation rows and returned `PREFLIGHT_GO`; the immutable development artifact is
`tmp/r1c_preflight_fullgtf_acceptor2x2_dev_v3.json`, SHA-256
`b491664ab715cf57e603e25526a19b5560566f045ab9a8c3369209f5f2077a29` (manifest SHA-256
`e711912b914b7c085e5fc93cf83242ba56847e817cc58c6305a975326d2e5349`). The sole family gave
`Y_PD=+1.1947 [+1.0497,+1.3397]`, `Y_PN=-0.0033 [-0.0193,+0.0127]`,
`S_P=+1.1980 [+1.0531,+1.3428]`, and the distance-matched interaction
`I=+0.9213 [+0.7488,+1.0938]`. The missing fourth cell was `Y_XN=+0.0122`; the descriptive
exon-mate specificity was `S_X=+0.2767`. All declared gates passed. Centered whole-gene calibration
produced 0/200 conservative null positives, 200/200 core-plant successes, and 194/200 stronger-plant
successes; the corresponding exact one-sided bounds passed.

This remains development evidence: the protocol and runner were draft, both seal pins were absent,
and the worktree was dirty. It creates no `AC*` row and cannot bind a target. The R1c target outcomes remain unscored;
the target stays hard-blocked until review, freeze, both exact pins, and a fresh passing post-freeze
preflight.

## ACR15 — R1c v3 mixed a zero-null p-value with SESOI-boundary intervals; v4 is the unrun successor (2026-08-15)

`ACR14` accurately preserves the bytes, hashes, cohort, point estimates, simultaneous intervals, and
then-emitted `PREFLIGHT_GO` verdict of the immutable v3 development artifact. Its interpretation needs
one further correction. The v3 declaration joined interval classifications at the `+/-0.05` SESOI to
a raw randomisation p-value computed at zero. Those are different nulls. The small zero-null p-value
therefore could not serve as the required test of a material-positive boundary or as a two-sided
equivalence test for `Y_PN`. The artifact passed the classifier that existed at execution time, but it
is not a valid target-binding preflight and its verdict is not behavior evidence.

**Stored-row diagnostic reanalysis.** Without rescoring the model or changing the v3 artifact, the
retained sufficient rows were replayed through explicit whole-gene boundary tests: one-sided tests at
`+0.05` for `Y_PD`, `S_P`, and `I`; TOST at `[-0.05,+0.05]` for `Y_PN`; path-level
intersection-union p-values; Holm across the core and stronger paths; and the strict simultaneous-CI
conjunction. At 2,000 shared sign draws, every constituent edge had add-one p-value `1/2001`, and both
Holm-adjusted path p-values were `2/2001`. The replayed strict intervals were
`Y_PD [1.0422,1.3472]`, `Y_PN [-0.0194,0.0128]`, `S_P [1.0458,1.3501]`, and
`I [0.7418,1.1008]`; both paths were declared by the corrected classifier. This is a diagnostic
reanalysis of already-observed development rows, not a fresh model run, immutable preflight artifact,
independent replication, or claim.

**Calibration audit.** The first 200-trial exact-path replay was deliberately preserved when its core
boundary surface produced `5/200` declarations: the one-sided 95% Clopper--Pearson upper bound was
`0.05184`, narrowly missing the `0.05` engineering gate. The stronger path's worst surface was `4/200`
(`0.04518`); the plants succeeded `200/200` for core and `196/200` for stronger. Extending the same
deterministic namespaces to 1,000 trials, rather than changing a seed or scientific
threshold, gave worst-surface counts `10/1000` for core (upper bound `0.01690`) and `9/1000` for
stronger (upper bound `0.01565`), with plant success `1000/1000` and `982/1000` (lower bounds
`0.9970` and `0.9734`). The 200-trial failure was an unstable audit window; it is not erased by the
expanded replay.

After the v4 replay and artifact-binding hardening was complete, the same retained v3 primitive rows
were passed once through the finalized **4,000-inner-draw / 1,000-outer-trial** code path, without
loading the model or writing a result artifact. The worst boundary counts were `12/1000` for core
(upper bound `0.01937`) and `18/1000` for stronger (upper bound `0.02658`); plant successes were
`998/1000` and `983/1000` (lower bounds `0.99372` and `0.97461`). The compact trial-record digest was
`ed620e46dc07250912678c16740de4204efabe0ef3791c486aef1ac3a1dbc2b9`. This verifies the finalized
statistics implementation on already-observed residuals only. It is still not a fresh v4 model
preflight, a bindable artifact, independent evidence, or a claim.

**v4 successor and disposition.** R1c v4 now makes the boundary/TOST/IUT/Holm classifier and strict-CI
concordance the sole production decision, uses 4,000 shared add-one sign draws per classifier, and
fixes the expanded 1,000-trial outer audit at no more than 38 boundary declarations and at least 916 plant
successes per path. It remains **DRAFT, unrun on the model, and unpinned**.
There is no fresh v4 model preflight, target artifact, or `AC*` claim, and the v3 artifact cannot be
grandfathered into the v4 seal. The R1c target outcomes remain unscored (`ACR18`). The current component successor is the
acceptor-only R3c draft; its disclosed two-row developer check was mechanics-only and is not evidence.
At the point recorded by ACR15, R5a/R5b/R5c were prospective method drafts with no runner, preflight,
target artifact, or result. Their later executable-but-model-free status is recorded in ACR17; it adds
no model evidence.

## ACR16 — R3c v1 could revive a preflight-health failure at target; the failure now forms a monotone ceiling (2026-08-15)

R3c v1 produced a development-only bucket-0 preflight on 500 validation rows / 284 genes, artifact
SHA-256 `b4d9172b911588f7ba83e7441564851b0111763e3c0ffea5931403dc27d3f913`. Its
`L2H15_mean_component` health check scored `31/64`, below the registered one-half floor, and the
corresponding v1 planted-power path was suppressed. The target authorization logic nevertheless
recomputed health on bucket 1 and could have restored that path if the later health check passed. That
would have let a held-out outcome revive a path that the bindable preflight had not authorized.

**Correction.** The v1 package is preserved as development history, but it cannot authorize a target
or support `L2H15`. No bucket-1 outcome was opened under v1. The prospective v2/v2.1 rule carries the
reviewed v1 failure forward and defines effective target authorization as preflight health **and**
current health: target can remove a path but cannot add one. The completed v2.1 target therefore records
`L2H15_mean_component` as **`HEALTH_FAILED`** even though its current target health check passed `34/64`
and its scientific endpoint conjunction would otherwise pass. Those endpoint values are diagnostics,
not support (`AC8`). Diagnosing and revisiting `L2H15` requires a new prospectively frozen successor;
the failure cannot be cleared by reinterpretation of this target.

## ACR17 — R3c v2 inspected a model path before checking the mandatory target authorization input; v2.1 is the clean successor (2026-08-15)

The first durable v2 preflight is preserved at
`r3c_acceptor_head_preflight_v2.json`,
SHA-256 `bc5f0797201a85fa2584a644212454aeb6fb7f4f1fd5c33497f9a552a981c031`, with
its `r3c_acceptor_head_preflight_v2.manifest.json`,
SHA-256 `6f77285cef03ccfa33c49478fa7b8c580628e0f67f9cfb844ad460c6a70cc398`.
It passed its scientific, calibration, health-ceiling, and live-hook capability gates. Once its literal
pin was activated, however, a target invocation missing `--preflight-artifact` inspected the supplied
model path before rejecting the missing authorization input. No bucket-1 outcome was opened under v2.
Because the corrected early guard changes runner bytes, the v2 artifact cannot be grandfathered into a
v2.1 authorization.

**Correction and completed successor.** V2.1 checks the mandatory target argument before inspecting any
caller-supplied model, GTF, FASTA, output, cohort, or preflight path, and required a fresh clean
preflight. The immutable successor package is:

- [v2.1 preflight](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.json),
  SHA-256 `c0c9306fe894ce73bfd981a14c85e2aec6660da33f01d9716cee4327b3ff2d3e`, and
  [manifest](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.manifest.json),
  SHA-256 `4a53f96b365c7bed227c8e0c995cee65a022620994e8898bab988b354fce41bb`;
- [r3c_acceptor_head_target_v2_1.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json),
  SHA-256 `ebd4bfd8c10cc0ccb218b3f1ceca634605903d8afbb5617de312087100627cc6`, and
  [r3c_acceptor_head_target_v2_1.manifest.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.manifest.json),
  SHA-256 `dc644901224227f03e683f7416587dccf3eedd700052b211089242b9b8d69cbe`;
- [r3c_acceptor_head_target_v2_1.audit.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json),
  SHA-256 `0363d02dced48d0d1a8ba56f1862be4a0cbb0b82eb508f9aeb87b8c61b05edb3`, reporting
  `AUDIT_PASS_MIXED_SCIENTIFIC_OUTCOME` after two exact replays without model rescoring, target-cohort
  reopening, or chr8/9 access.

The manifests record clean executions at commits `bb22dff7e2f3ec4e3de73a8d45ad1740793b66f4`
(preflight) and `8dde21ce043d7d9e4835e7ccc5851d13a3af4a36` (target). The target used bucket 1 on
validation chr11/16/19, n=500 / 304 genes, and returned exactly four supported component patterns:
`L1H12_mean_component` and the `L4H11`, `L4H9`, and `L4H7` noising components (`AC5`, `AC6`).
`L3H1_reused_component` and `distinct_acceptor_route_candidate` remain
**`INCONCLUSIVE_METHOD_DISAGREEMENT`** (`AC7`); `L2H15_mean_component` remains **`HEALTH_FAILED`**
(`AC8`). Neither predecessor artifact is deleted or rewritten; this entry controls their authorization
interpretation.

R3c constructs one masked-acceptor endpoint and no masked-donor endpoint. It never scores the R1c
masked-acceptor target outcomes on chr8/9 and
licenses no address, payload, write, distinct route, delivery, mediation, complete circuit, or biological
mechanism. The frozen R5 chain stops at entry because R5a requires supported `L3H1` reuse and correctly
rejects this target. A next experiment must be either a new prospective, gene-disjoint clarification of
that registered path or a separately frozen acceptor-route program centered on the supported
`L4H11`/`L4H9`/`L4H7` patterns. The latter must not be called delivery without restoration and mediation
evidence.

## ACR18 — chr8/9 was already spent by the donor arc; only the R1c masked-acceptor outcomes remained held out (2026-08-17)

Earlier versions of this ledger said that sealed chr8/9 was globally unopened or unspent. That is
false. The canonical donor-side result `C1` uses
[`sealed_test.json`](../../results/splicebert_coupling/sealed_test.json) on the chr8/9 test split.
Historical inverse-acceptor development also materialized chr8/9 sequence before its chromosome filter
was repaired. Neither fact measures the R1c masked-acceptor target endpoint, but both prevent this study
from describing the chromosomes themselves as virgin, unopened, or independently sealed.

The valid narrower statement is outcome-specific: **the canonical R1c→R5 program did not open, score,
or spend its masked-acceptor target outcomes on chr8/9.** Its only scored successor cohort was the R1c
validation preflight on chr11/16/19. The R1c artifact itself discloses the historical sequence
materialization and records `sealed_split_opened=false` only for that run. This is an
endpoint-outcome-held-out boundary, not a claim that no prior program read those chromosomes.

This correction changes no numerical result. It narrows every current seal statement in AC1–AC9 and
the study summary to the endpoint actually protected. It also prevents the stopped R1c target from
being advertised as an independent chromosome cohort when the opposite-direction donor behavior had
already used the same chromosomes.

## ACR19 — the frozen R1c package passes, but its frozen R5 handoff rejects the authentic six-margin rows (2026-08-17)

The prospectively frozen R1c v4 validation preflight ran once from clean commit `9008ba3`, on strict
MPS with `PYTORCH_ENABLE_MPS_FALLBACK` absent, using 400 validation rows / 321 genes from chr11/16/19.
That run did not materialize chr8/9 and did not access the R1c target outcomes (`ACR18`). The immutable artifact
[`r1c_reverse_behavior_preflight_v4.json`](../../results/splicebert_acceptor_coupling/r1c_reverse_behavior_preflight_v4.json)
has SHA-256 `1b27c4132c71a576c435b7b71f32831af9a44fbca6325931f35716ae4133cbdd` and size
`74,352,692` bytes; its [manifest](../../results/splicebert_acceptor_coupling/r1c_reverse_behavior_preflight_v4.manifest.json)
has SHA-256 `7412f3159e82f6ab0bbaa95e2796f41aa038eef4233efa7f09ec65867e39c45c` and size
`11,370` bytes. R1c's public verifier returned attestation SHA-256
`04be60c2239c98eefb01521f7a8a94a68f4d2cc4db5db8f1dddb5622f7378ce1`; an independent
statistics-only audit replayed the row algebra, 4,000-draw sign tests, simultaneous intervals,
10,000-draw raw and trimmed whole-gene bootstrap, and all 1,000 calibration trials. All 13 gates pass,
so the package's `PREFLIGHT_GO` is genuine instrument-validation evidence.

The frozen cross-stage contract is nevertheless not executable on this authentic package. Every R1c
`/3` row records six margin levels:

```text
P_clean, P_under_D, P_under_N, X_clean, X_under_D, X_under_N
```

R1c's own cohort identity deliberately projects those rows to the three partner/common-leg `P_*`
fields. The bound R5 helper `_r1c_ordered_partner_margin_levels`, however, rejects any row whose
`margin_levels` key set is not **exactly** those three fields before it performs the projection. The
real six-field R1c package therefore raises `R5CapstoneCohortError: R1c partner margin levels differ`.
The memorial
`test_reviewed_r1c_preflight_is_authentic_but_frozen_r5_handoff_rejects_its_six_margin_rows`
first authenticates the immutable package with R1c's own verifier and then reproduces that exact
failure through the frozen R5 builder.

This is an instrument/package-contract failure, not a scientific negative. The apparent code repair
would be small—accept the six-field record and project the three `P_*` fields—but making it after the
validation outcome is known changes a source file bound into the frozen joint-stack fingerprint
`b89efd7c0d83d579aa7e0a2d36f8059c96913242a1c943ec64d87fe7590b00bc`. Rewriting the artifact or
rerunning the preflight would likewise violate the one-run/no-post-outcome-redesign rule. No repair,
rerun, or pin installation is made.

**Disposition.** AC9 closes the canonical R1c→R5 adjudication as terminal
`INSTRUMENT_UNRESOLVED`. `FROZEN_PREFLIGHT_ARTIFACT_SHA256` in the R1c runner remains unset; all
shared R1c preflight/target package pins remain unset; no R1c target receipt exists; no R5a, R5b, or
R5c scored artifact exists; and the R1c masked-acceptor target outcomes remain unscored. The validation estimates do not become a
behavior claim, and no null or positive inference about L3H1 reading/writing or L4H11+H9 delivery is
licensed.

## ACR20 — the canonical coupling cohorts are `GT-AG`-only; natural `GC-AG` is a separate modifier scope (2026-08-17)

The historical `circuit.eligible` predicate requires the native donor motif to equal `GT`. Raw coupling
construction admits natural `GC-AG` introns, but every historical caller that passed through this
predicate discarded them before scoring. AC1–AC9 and the canonical donor-side coupling arc are therefore
scoped to canonical `GT-AG` introns even where earlier prose did not say so explicitly.

M1–M6 widen that scope deliberately. To preserve the byte-bound R1c/R5 implementation, the consolidated
tree does not change `circuit.eligible`; it exposes the same split, paired-acceptor, motif, and gap rules
through the outcome-blind `donor_class.eligible_donor_motifs` helper. Every donor-class runner opts into
`GC` explicitly. Tests pin the historical GT-only predicate and the widened natural-donor predicate
separately.

This correction changes no historical cohort or number. It prevents AC10–AC11 from being read as a
replication on the same biological stratum as AC1–AC9, and prevents the canonical `GT-AG` mechanism from
silently generalizing to the minor donor class.

## ACR21 — M6 numerically replicated both endpoints, but its one-run lifecycle and audit surface block confirmatory promotion (2026-08-17)

M6 reserved 67 matched validation pairs / 56 treatment genes and declared exactly one run. The first run
occurred at `2026-08-17T12:26:22Z` from clean commit
`42954a67947ed7d2748c57816661b917cf5d78e8`, under protocol SHA-256
`da326e4da94b5daa916f59f7268001e290ea00ab5ef2eb8da255c4dd477bf46a`. Its artifact SHA-256 is
`80150013b084cb659c7e3a0e1739573fd1a12e0c95c8cc433067ec9929d4051a`; its manifest SHA-256 is
`4e470908b93885133a817a12eee2b074c8008f3d0b3aafbfb4ac596ae27216df`, and that manifest records
`git_dirty=false`.

After those outcomes were visible, the protocol's Tier banner was moved, changing its hash to
`8d91b570…8cd3`, and the already-spent reserve was run again despite the literal “there is no second
one” rule. The second artifact (`726fafee…c32`) and manifest (`ba7a8d14…b06`) were committed on the
source branch; the latter records `git_dirty=true`. The scientific values are identical across runs,
but the files are not bit-identical because the serialized protocol hash differs. The consolidated
tree restores the first clean package as the controlling artifact; the prohibited second package stays
reachable through merge parent `b79570d` and is excluded from the live manifest.

Five further limits prevent clean confirmatory language. The runner requires a nonempty
`--protocol-sha256` but does not compare it with the protocol file. The alleged ordered treatment-ID
hash sorts its input and is order-insensitive; comparator IDs and pair mapping are not sealed. The
bootstrap samples the union of arm genes and can break one-to-one matched pairs when the two members
have different genes. No row values or bootstrap draws are stored, so the interval is not independently
replayable from the compact artifact. Finally, the manifest records the model directory as
`missing:true`, not a byte-bound checkpoint, and M6 had invariants rather than the repository's full
positive/negative/sensitivity/resolution preflight.

**Disposition.** The large numerical replication remains useful: AC10 and AC11 are bounded held-out
verification of a natural donor-class behavior and attention modifier. They are not established or
confirmatory evidence, and they license no component necessity, write, mediation, route, or biology.

## ACR22 — M1–M5 are adaptive development candidates, not evidence for a completed modifier mechanism (2026-08-17)

M1–M5 were useful exploration on the train split, but their outcomes do not control a live claim.
Most numerical outputs remained in scratch storage, most available manifests were dirty and did not
byte-bind the checkpoint, and several decisive multi-head/audit outputs had no manifest. The 96-head
selection used a cohort with strong donor-transcript-support imbalance and its pair-level bootstrap and
permutation did not cluster repeated genes. Its mean-ablation intervention also averaged over batch and
sequence, making results depend on batch composition; the offline label permutation did not recompute
that intervention.

The strongest honest development observations are an early-L0/L1 candidate family, a post-outcome
`MLP1` candidate, and evidence that these early candidates modulate L3H1 attention geometry. M3b did
not compute a confidence interval or equivalence test for forward-minus-unmatched transport, so “all
six transport-null” is not licensed. M4's address/payload swaps and M4c's attention-level ablations use
the same adaptive cohort and lack the prospective matched nulls needed for promotion. None measures the
canonical post-`W_O` acceptor-query write or L4H11/H9 delivery contemplated by R5.

The source branch is preserved as a merge parent, and the reusable donor-class matching, estimators, and
development runners remain available for audit. Its scratch outcomes are not imported as canonical
evidence. AC10–AC11 are the only live claims from this modifier arc.
