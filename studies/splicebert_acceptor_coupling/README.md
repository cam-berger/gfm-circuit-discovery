# SpliceBERT acceptor coupling — paired-boundary dependence read from the other end

> **Tier: T1 — canonical.** Current, authoritative. Read this.

## Why this study exists

`studies/splicebert_partner_selection` masks the **donor** and corrupts the intron-mate **acceptor**,
in every confirmatory and mechanistic result it contains. The A0→A12 arc decomposed that behaviour into
a positional address, an **acceptor-local** payload, and delivery to a **donor-side** readout. That
account is stated asymmetrically throughout — and until this study, **nothing in the record tested
whether the asymmetry is a property of the model or of the assay.**

This study reads the same intron from the other end:

| | mask | corrupt | quantity |
|---|---|---|---|
| donor study (C1) | `donor_pos` | partner acceptor `[p-2, p)` | `Y_D` |
| **here** | `next_acceptor_pos - 1` | partner donor `[d, d+2)` | `Y_A` |

Both readouts mask a `G` — the donor's is the `G` of its `GT`, the acceptor's the `G` of its `AG` — so
the native masked-LM margin `logit(G) − max(A, C, T)` is used unchanged. The reversal is the
substitution of one position list for another, and `tests/test_reverse_coupling_memorials.py` asserts
at source level that `CouplingTask.encode`/`margin` contain nothing donor-specific.

## Two masked-acceptor programs — do not conflate them

The historical
[`sri-algoverse-acceptor-interpretability`](https://github.com/cam-berger/gfm-circuit-discovery/tree/6a136390ce3f03fb1b3ad2b9c180c055b163c0a4/acceptor-interpretability)
branch described itself as the reciprocal acceptor circuit, but it did **not** reverse the donor study's
paired-site intervention. It and the present study share an acceptor readout and otherwise ask different
questions:

| | Historical acceptor-context pilot | Present paired-donor study |
|---|---|---|
| Question | Does sequence immediately upstream of an acceptor support its reconstruction? | Does an acceptor reconstruction depend on that intron's paired donor? |
| Readout | Mask both bases of `AG` simultaneously and sum the `A`- and `G`-margins as `m_acc`. | Mask only the acceptor `G`, exactly mirroring C1's single-`G` margin as `Y_A`. |
| Intervention | Corrupt fixed or heuristically located branch-point/PPT-labelled windows. | Replace the paired donor `GT` with a motif-clean alternative. |
| Cohort | The same 200 GUE `splice_reconstructed` acceptor-test examples throughout Phases 0–5; Phase 0 also uses 200 label-2 controls, and the decoy comparison uses only examples with a downstream `AG`. | Historical R1/R1b/R3 use 974 paired validation exons / 645 genes; R3c/R3d and the R1c validation instrument use separate outcome-blind cohorts. The frozen R5 stages never scored a cohort. |
| Status | Branch-local, exploratory, no live claim. | `AC1`–`AC8` preserve bounded, retracted, inconclusive, and health-failed canonical-`GT` evidence; `AC9` is the terminal R1c→R5 handoff failure; `AC10`–`AC11` add bounded natural-`GC` behavior and attention modifiers. |

The Sri pilot therefore cannot be cited as prior evidence for reciprocal donor→acceptor coupling, and
`AC1`–`AC8` neither validate nor refute its branch-point/PPT hypotheses.

Within the present paired-donor study, the **canonical construction arc** and the **natural donor-class
modifier arc** must also stay separate. AC1–AC9 ask what can be established for canonical `GT-AG`
introns and whether a reader→write→delivery account can be authorized. AC10–AC11 compare natural
`GC-AG` introns with matched `GT-AG` introns. The latter changes biological stratum and reaches only
behavior and attention geometry; M4d only records development diagnostics over adaptive QK candidates,
under separate arm-conditioned interventions, and none is the missing
causal mechanism (`ACR20`–`ACR23`).

### Corrections to the historical pilot

| Earlier reading | Correction from the branch's later phases and code audit | Surviving status |
|---|---|---|
| Natural context and the true site beating a downstream `AG` establish an “AG-scanning” mechanism. | The two assays mask different positions while leaving the other `AG` visible; the downstream decoy also varies in distance and local context. It is not a matched scanning comparison. | Exploratory discrimination between those constructs, not an identified scanning algorithm. |
| Fixed-window “PPT” disruption (`+1.245`) and “BP” disruption (`+0.212`) identify annotated biological elements. | GUE supplies neither per-sequence branch-point nor PPT annotation. The windows were fixed approximations. Phase 3's hand-built `yUnAy` locator reduced the putative BP effect to `+0.090 [−0.007,+0.186]`, but also changed the BP edit from 23 nt to 5 nt and the PPT edit from 15 nt to about 27 nt; the apparent coordinate refinement is dose-confounded. Phases 4–5 then return to the original fixed windows. | The branch reports a large effect for one fixed upstream window, but its retained files cannot verify the estimate; there is no stable standalone BP result or coordinate refinement. |
| The tract is “close to sufficient,” its effect is identity-driven, and BP×PPT forms an AND gate. | Corruption is a necessity-style manipulation, not a sufficiency test. In-place shuffling shows only that bulk composition is insufficient; it does not isolate motif identity from order, exact position, k-mers, or distribution shift. The marginal intervals for PPT-corrupted and random-context conditions overlap, but no paired equality or equivalence test was run. The four cells exist, yet no interaction estimand, interval, or decision rule was computed; the runner's `GATE IS PLAUSIBLE` flag ignores the PPT-only cell. | At most, a hypothesis that the joint masked-`AG` score may depend on base arrangement in a fixed upstream window. A defined interaction test and restoration/transplant sufficiency assay remain unrun. |
| Five heads are tract-content readers. | Same-sample attention ranking over all 96 heads nominated candidates; Phase 4 then made 42 head×region classifications from **84 uncorrected clean–corrupt and clean–shuffle intervals**. Effects include `~10⁻⁵` (`L0H14`). The six inherited comparator heads lack documented selection provenance and are not the current canonical donor core. | Attention-based candidates only; no output-use claim and no component-wise bridge to `AC4`. |
| Phase 5 tested all five candidates, found they jointly explain about 2%, and therefore established a distributed or indirect mechanism. | The runner ablates **four** candidates (`L5H8`, `L4H12`, `L3H5`, `L0H13`) plus the negative control `L0H2`; it never intervenes on `L0H14`. No tested candidate shows positive necessity, and the four-candidate joint interval includes zero. The only joint interval excluding zero is `clean − ablated = −0.030 [−0.060,−0.002]` for those four plus the control: ablation raises `m_acc`, the wrong sign for the proposed positive tract-support role. Only `L5H8` and `L4H12` receive restoration and Q/V tests, reported null at the masked-`AG` query rows. | No identified mediator. The branch reports no measurable mediation through those two heads' direct query-row outputs; other heads, positions, MLPs, indirect routing, and wider mechanisms remain untested. |

The provenance boundary is decisive. The entire Sri program entered as one commit (`6a136390`); its
result JSON/CSV files are gitignored and absent, and it has no manifests, checkpoint or dataset revision
pins, preregistration, claim ledger, raw rows, or executable memorials. The synthetic pilot it says the
GUE results replicate has no source or artifact in that commit. Corruptions seed from Python's
process-randomized `hash(sequence)` and are not actually balanced ACGU draws because `G` is forbidden
after `A`. All phases adaptively reuse the same 200 acceptor examples, intervals resample rows rather
than genes, and the attention family is uncorrected. The six PNGs preserve reported summaries, not
recomputable evidence. None of the branch's numerical effects is a live claim; they are retained only
as historical provenance and hypotheses. Because the retained branch does not bind the 200 GUE rows to
gene identifiers, overlap with the corrected R1c/R3c cohorts is unknowable. A future result may be
gene-disjoint from historical R3 and from its own preflight while still lacking demonstrated
gene-independence from Sri's anonymous GUE sample.

### What followed

1. **The promotion follow-ups for the local-context question remain unrun:** rerun with immutable artifacts and deterministic
   constructs; use source-bound branchpoints, an explicitly operational pyrimidine-rich interval, and
   gene-disjoint discovery/evaluation cohorts; predefine and test the factorial interaction and later
   test tract sufficiency in a genuinely independent transplant; then search heads,
   MLPs, and non-query positions from the native readout with matched-size nulls and family correction.
   The new draft implementation lives in
   `splicebert_acceptor_context`: it supplies the
   single-G primary endpoint, equal-dose 2×2, exact transition-matched controls, double-mask bridge, and
   restoration accounting. Its outcome-blind cohort builder is implemented, but source-bound cohort
   artifacts and a target run do not yet exist.
2. **The present study answers the separate paired-boundary question.** R1/R1b replace the paired donor
   and add site- and partner-specificity controls; R2 corrects the donor-operator interpretation; R3 scans
   all 96 heads under paired mean ablation. R3c v2.1 then tests seven fixed acceptor-only paths on a
   gene-disjoint held-out validation bucket: `L1H12` mean ablation and `L4H11`/`L4H9`/`L4H7` noising are
   supported, `L3H1` reuse and the distinct route are inconclusive by method disagreement, and `L2H15`
   is health-failed. This supersedes the Sri branch only as evidence for reciprocal paired-site
   dependence—not as evidence about branch points or the PPT.
3. **The attempted edge follow-up remains exploratory.** R4 was designed to add MLP writes, noising,
   and edge-level comparison, but its protocol was never frozen and is now retired. Its signed objective rewarded
   over-recovery (`D10`) and its uncertainty-aware edge partition was never implemented (`D11`), so it
   created no `AC*` result.
4. **A separate forward biological-semantic pilot remains non-ledger evidence.** `DECISIONS.md` reports
   a forward masked-donor experiment (`AG→CC`, motivated by C30), not Sri's joint masked-`AG` endpoint or
   AC1's reverse readout. The literature window cannot be identified in the 60–150 nt cohort. Measured
   only at spans ≥250 nt, a matched `CTAAC` restoration gives `+0.00850 [+0.00670,+0.01031]` at
   −49…−19. The same exploratory arc reports that inserting 400 nt reduces coupling by about 82%, versus
   a 13–28% position/edge effect, so this is a much weaker coupling regime. The motif effect is one draw,
   about sixfold below the `0.05` SESOI, and has no frozen preregistration or committed runner/artifact;
   the PPT-dominance hypothesis was not retested.

## The one thing to know before designing anything here

**The instrument does not mirror.** The acceptor-side corruption operator is validated alt-invariant
within ±25% (**C42**). The donor-side operator is **not**, and cannot be:

- R44 measured `seq[d-1] == "G"` at **81.0%** of validation donors (the 5'SS consensus). `CC` therefore
  plants a `GC` donor one base upstream, `TA` plants a **canonical `GT`** (95.5% on the atlas
  condition-B cohort), and `AA` plants an `AG` (62.0%). `AC` and `AT` are universally motif-clean;
  `AC` is the clean two-substitution operator and `AT` the clean one-substitution operator (`D8`).
- `GT → AC` is **two** substitutions; `GT → AT` is **one** and leaves the `T` at `d+1` standing.

C42 does not transfer — its own row scopes it to the acceptor operator. Every runner here uses
`MOTIF_CLEAN_DONOR_ALT` and re-verifies admissibility per construct with `planted_donor_motifs`, never
from a tuple. Any comparison between the two directions must match substitution counts;
`reverse_coupling.matched_operators` exists so that mismatch cannot be introduced silently.

## Prior operator evidence and the archived-edge boundary

One earlier donor-operator sensitivity result is relevant to this study:

| source | where | what it found | why it does not settle the question |
|---|---|---|---|
| `AR1` arm `DONA` | `splicebert_partner_selection`, `ar1_corruption_operator_sensitivity.json` | corrupt donor, read masked partner acceptor; verdict **`NOT_INVARIANT`**; every alt with `z[1] == "T"` at ρ ≈ 0.62 | declared "secondary and descriptive"; **no specificity control**; `n = 190` / 171 genes drawn entirely from the **19%** of donors lacking the consensus `G` |

The archived exon-edge compatibility prose is not a second reciprocal-acceptor result and is not folded
into this study. `R45` superseded that arc and carried only `C43` and `C44`: `C43` is a bounded
**masked-donor** phase-preference result in a chimeric crossed lattice, while `C44` says its proposed
period-3 implementation is instrument-unresolved. Neither supplies a native masked-acceptor operator
estimate, component selection rule, or symmetry premise here.

## What is here

R1, R1b, R2, R3, R3c, R3d, and the R1c v4 validation preflight use only train or validation data.
The canonical R1c→R5 program did not open or score its masked-acceptor target outcomes on chr8/9;
the chromosomes themselves had already been used by the opposite-direction donor result `C1`
(`ACR18`). R3c's bucket 1 is held out by gene within validation chr11/16/19, not a chromosome-sealed confirmation. R3d supplied the sole prospective `L3H1` Path-A
component prior. The frozen R1c v4 validation preflight then passed all 13 gates and independent replay,
but the exact frozen R5 handoff rejected its authentic six-margin rows before any package pin or sealed
target outcome was opened (`ACR19`). The canonical R1c→R5 program is therefore closed as terminal
`INSTRUMENT_UNRESOLVED` (`AC9`); R5a/R5b/R5c have no scored artifact. R4 remains a retired,
never-frozen draft with a historical exploratory execution; the BP arc is non-ledger evidence. The
M1–M5 natural donor-class runners are development-only candidate generators. The imported M4d
score-margin package diagnostically revises M4c's broad head labels but remains non-claim development
QK evidence; its post-selection intervals, clustering, separate arm-conditioned interventions, and provenance limits are
recorded in `ACR23`. M6 used a separate validation reserve and produced bounded AC10–AC11; its first
clean package is retained, while its one-run lifecycle and audit limitations prevent confirmatory
promotion (`ACR21`).

| arm | question | outcome |
|---|---|---|
| **R1** ([prereg](preregistrations/PREREGISTRATION_R1_REVERSE_COUPLING_GATE.md), frozen) | is there a reverse coupling, and is it specific? | `INSTRUMENT_UNRESOLVED` — from a power gate aimed at the effect rather than at the nulls. **Superseded; retained as provenance.** See `ACR1`. |
| **R1b** ([prereg](preregistrations/PREREGISTRATION_R1b_REVERSE_COUPLING_GATE.md), frozen) | the same, with `MDE80` gated on the control arms + three-draw verdict stability | bounded validation gate: artifact says **`GO`** on all three draws, but the frozen scope forbids claim promotion and the permutation memorial is row-wise rather than gene-wise (`ACR8`) → `AC1` |
| **R1c v4** ([frozen protocol](preregistrations/DRAFT_R1C_SEALED_REVERSE_BEHAVIOR_CONFIRMATION.md)) | does the masked downstream acceptor depend selectively on its paired donor, and on matched exon/intron dyads is that selectivity stronger than at the upstream exon-mate acceptor? | its fresh 400-row validation preflight passed all 13 gates and exact replay, but the frozen R5 handoff rejects the authentic six-field rows; the R1c target outcomes stayed unscored → terminal `INSTRUMENT_UNRESOLVED` (`AC9`, `ACR19`) |
| **R2** | how does reverse coupling respond when each donor base is destroyed? | post-outcome operator screen: either base changes the endpoint, sub-additively; the `T`-only interventions are larger → bounded `AC2`, correcting `ACR4` |
| **R3** ([prereg](preregistrations/PREREGISTRATION_R3_PAIRED_COMPONENT_SCAN.md), frozen) | which of the 96 heads are **shared** between the two directions and which are **direction-specific**? | instrument unresolved: raw mean-ablation values nominate candidates, but the nominal 73/17/4/2 partition and rank identifications are retracted because the calibration path did not implement the frozen familywise decision (`ACR9`) → open `AC4` |
| **R4** ([retired draft](preregistrations/PREREGISTRATION_R4_EDGE_LEVEL_CAUSAL_CORE.md)) | historical question: do the two directions share their **edges**, or only their **components**? | retired, never-frozen design; its exploratory execution over-rewarded recovery (`D10`) and did not implement the proposed edge partition (`D11`); do not execute; no `AC*` result |
| **R3c v2.1** ([frozen protocol](preregistrations/DRAFT_R3C_ACCEPTOR_HEAD_CONFIRMATION.md)) | do fixed donor-reader and later-head candidates contribute selectively to the paired-donor-induced masked-acceptor margin under mean ablation and same-example noising? | clean [bucket-0 preflight](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.json) ([manifest](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.manifest.json)) and held-out-within-validation [r3c_acceptor_head_target_v2_1.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json) ([r3c_acceptor_head_target_v2_1.manifest.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.manifest.json); [r3c_acceptor_head_target_v2_1.audit.json](../../results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.audit.json)); supported only `L1H12` mean and `L4H11`/`L4H9`/`L4H7` noising component patterns → `AC5`/`AC6`; `L3H1` reuse and distinct route inconclusive, `L2H15` health-failed → `AC7`/`AC8` |
| **R3d Path A** ([frozen protocol](preregistrations/DRAFT_R3D_ACCEPTOR_SPECIFICITY_ADJUDICATION.md)) | does a fresh registry-disjoint train target support the prespecified `L3H1` paired-versus-pseudo component conjunction? | supported on 3,909 target genes; used only as the sole component prior for the frozen R1c→R5 design. Path B was excluded from entry and is not promoted here. |
| **R5a/R5b/R5c** ([frozen preflight-only protocols](preregistrations/)) | conditional on R1c behavior and R3d Path A, does L3H1 read the donor, write a post-`W_O` acceptor-query-local state, and do L4H11+H9 deliver it? | not run: the mandatory exact R1c→R5 package handoff failed before any scored R5 stage; no preflight, target, or result exists (`AC9`) |
| **M1–M5 natural donor-class development** | why is reverse coupling weaker for natural `GC-AG` than matched `GT-AG`, and which early components might modulate L3H1 geometry? | adaptive train-split candidate generation only. M4d's exact log-odds transform nominates `L0H14`/`L0H5`/`L0H3` only at the nominal, unadjusted score-margin level, leaves three earlier labels unresolved, and gives `L2H15` an opposite-direction diagnostic. No live claim, query-versus-key attribution, transport equivalence, post-`W_O` write, mediation, or delivery (`ACR22`, `ACR23`). |
| **M6 natural donor-class verification** ([original frozen protocol](preregistrations/PREREGISTRATION_M6_VALIDATION_CONFIRMATION.md)) | do the behavior ratio and L3H1 attention ratio reproduce on the donor-class-development-held-out validation reserve? | numerically yes on 67 pairs / 56 treatment genes, but bounded to verification by lifecycle, checkpoint, cohort-hash, bootstrap, and replay limits → `AC10`/`AC11`, `ACR21` |

### Current successor status and evidence boundary

- **R1c v4 passed its validation instrument and then stopped at the frozen handoff.** The immutable
  [preflight](../../results/splicebert_acceptor_coupling/r1c_reverse_behavior_preflight_v4.json)
  ([manifest](../../results/splicebert_acceptor_coupling/r1c_reverse_behavior_preflight_v4.manifest.json))
  used 400 rows / 321 genes on chr11/16/19, returned `PREFLIGHT_GO`, and passed independent replay of
  all 13 gates. The artifact/manifest SHA-256 pair is `1b27c4132c71a576c435b7b71f32831af9a44fbca6325931f35716ae4133cbdd` /
  `7412f3159e82f6ab0bbaa95e2796f41aa038eef4233efa7f09ec65867e39c45c`.
  The bound R5 handoff rejects its authentic six-margin rows because it requires an exact three-key
  `P_*` set before projection. No post-outcome parser change, rerun, or pin installation is authorised.
  The target stayed sealed and AC9 is terminal `INSTRUMENT_UNRESOLVED`.
- **R3c v2.1 completed its registered within-validation target.** It hard-asserts the historical R3 exclusion identity
  at 974 rows / 645 genes and ordered example-ID SHA-256
  `161dff038d747dbaa7b91a96d24b34d45f55de5482b5140410641354e46d2a8e`, then excludes every represented
  gene. Its fixed acceptor-only registry contains ten heads, no MLPs and no donor readout; it derives a
  62-row family under within-example token-mean ablation and same-example corrupt-to-clean noising,
  with seven Holm-corrected candidate paths and independent live-hook controls. Both fixed 500-row
  buckets remain on validation chr11/16/19: bucket 0 is development/preflight and bucket 1 is the
  held-out within-validation target. The clean v2.1 preflight and target are linked in the table above;
  target n=500 / 304 genes, baseline paired-donor effect `+0.9749`, pseudo-donor effect `+0.0232`, and
  clean masked-acceptor margin `+3.5515`. Only `L1H12_mean_component` and the
  `L4H11`/`L4H9`/`L4H7` noising component paths are `SUPPORTED`. `L3H1_reused_component` and
  `distinct_acceptor_route_candidate` are `INCONCLUSIVE_METHOD_DISAGREEMENT`; `L2H15_mean_component`
  is `HEALTH_FAILED` under the monotone v1 ceiling. R3c masks one acceptor `G`, constructs no donor
  readout, and does **not** score the R1c masked-acceptor target outcomes on chr8/9. Its strongest licensed status is **bounded,
  intervention-specific component patterns; no reuse, distinct-route, delivery, mediation, or
  mechanism claim**.
- **R5a/R5b/R5c remain model-free methods, not model evidence.** They specify paired-donor reader tests,
  a post-`W_O` acceptor-query-local write, and conditional L4H11+H9 delivery with matched controls. The
  scored stages never ran because the predecessor package could not cross the frozen handoff. Their
  unset pins and absent outputs are part of the terminal record, not unfinished authorization work.

The earlier R3b head+MLP design remains a never-frozen, unrun record. Its reciprocal donor endpoint is
not part of the current acceptor-only forward path.

R3 was intended to answer "same or different" as a **single paired estimand** rather than two screens:
both readouts are measured on the same exons in the same process, so `Δ(c) = frac_A(c) − frac_D(c)`
has cohort, distance, composition and gene structure cancelling inside every contrast. Its primary
estimand is **mean-ablation**, because the repository's two idioms disagree 45× on exactly the class of
component at issue (forward `L2H15`: `+0.0115` noising vs `+0.5146` ablation) — a noising-primary sweep
returns a false null for the addresser. **MLP writes were out of scope in R3**: its runner had no MLP
intervention, while `HCA18` reports MLP writes dominating its acceptor-cluster signals. The abandoned
R3b draft proposed an MLP primitive; current R3c intentionally narrows to a fixed attention-head
registry. R3c v2.1 now supplies the bounded AC5/AC6 component patterns and the explicit AC7/AC8
non-support records. R3's raw head effects remain useful for
candidate generation, but its permutation null, planted positive, one-direction specificity check,
unadjusted equivalence rule, missing noising estimand, and mismatched reverse health policy make the
frozen decision `INCONCLUSIVE` (`ACR9`, `ACR11`).
R4 is the attempted edge-level continuation. Even heads nominally classified `SHARED` under historical
R3 mean ablation can
have different wiring; that label compares relative necessity, not recruitment or computation. The
design was intended to add **MLP writes** and R3's missing **noising second estimand** on
one cohort. It did not close either gap: the protocol was never frozen and is now retired, the completed exploratory
`tau_HI` sweep used a signed objective that rewarded over-recovery (`D10`), and the runner never
implemented the preregistered uncertainty-aware edge partition (`D11`). Its source artifact is
uncommitted, and no `AC*` row was created. Engineering record:
[`DECISIONS_R4_EDGE.md`](DECISIONS_R4_EDGE.md).

### Closed causal-adjudication path

The final program was frozen as one ordered chain:

1. **Prior:** R3d Path A alone supplied the prospective `L3H1` component premise. R3d Path B,
   `L2H15`, H7, the optional low-rank basis, and the natural donor-class early-head candidates were secondary only.
2. **Validation instrument:** R1c v4 passed its fresh validation preflight and exact replay.
3. **Frozen package handoff:** the authentic R1c package failed the mandatory R5 schema contract.
4. **R1c target behavior:** not tested; its masked-acceptor target outcomes on chr8/9 remained unscored
   (`ACR18`).
5. **Reader, post-`W_O` write, and delivery:** not tested; R5a/R5b/R5c stopped with no scored artifact.

The chain ends at step 3. The repository records one terminal outcome—AC9
`INSTRUMENT_UNRESOLVED`—rather than repairing the parser, rerunning the preflight, substituting a
secondary component, or turning validation estimates into a scientific result. The natural-GC arc
is recorded separately as bounded AC10–AC11 modifier evidence; M4d remains a development correction,
and no GC/GT, early-head, or MLP result is inserted into this closed program.

- [`CLAIMS.md`](CLAIMS.md) — the claim ledger (`AC*`). Separate from the donor study's `C*` ledger on
  purpose: acceptor reconstruction is a different behaviour from donor reconstruction, and giving it
  its own ledger keeps C1–C42 from silently acquiring reverse-direction rows.
- [`DECISIONS.md`](DECISIONS.md) — chronology, including both pilots and their numbers, and the
  2026-08-08 BP arc (what it costs to leave the short-intron band; creates no claim row).
- [`DECISIONS_R4_EDGE.md`](DECISIONS_R4_EDGE.md) — `D1`–`D11`, the edge-level R4 design record,
  including the two negative results about the instrument (`D10` signed objective, `D11`
  unimplemented partition).
- [`CORRECTIONS.md`](CORRECTIONS.md) — `ACR*`. Never delete a correction; record, don't rewrite.
- [`preregistrations/`](preregistrations/) — `R1`, `R1b`, `R3`, `R3c v2.1`, R3d, the
  R1c/R5 source stack, and the original M6 protocol are frozen records; R3b remains an unrun draft and `R4` a **RETIRED DRAFT / NEVER
  FROZEN** record. AC9 closes the R1c/R5 chain before target, so the remaining `None` pins are terminal
  evidence rather than an invitation to execute it.
- `scripts/splicebert_acceptor_coupling/` — runners. `src/gfmcd/splicing/reverse_coupling.py` — the
  reusable readout. `results/splicebert_acceptor_coupling/` — artifacts, each with a manifest.

## Scope, standing

AC1–AC11 use one checkpoint lineage (`SpliceBERT.1024nt`), internal short introns, and both strands.
AC1–AC9 are restricted to canonical `GT-AG`; AC10–AC11 are the explicit natural `GC-AG` versus
matched-`GT-AG` exception (`ACR20`).
AC1 is a bounded behavior gate, AC2 a bounded post-outcome operator screen, AC3 is retracted, AC4 is
instrument-unresolved candidate generation, AC5/AC6 are bounded held-out component patterns, AC7/AC8
preserve inconclusive or health-failed paths, and AC9 is a terminal package-handoff instrument failure
before the R1c target cohort. AC10/AC11 are bounded behavior/attention modifier evidence and do not
establish causal construction. All are model-internal, and **no external biological assay enters
those records.** The
later non-ledger forward BP pilot uses longer spans and a literature-derived motif/window as an
intervention stimulus; that is a biological prior, not external validation. Nothing here is a claim
about splicing, spliceosome biochemistry, or transcript abundance. Evidence never promotes across levels; read
[`../../docs/EVIDENCE_LADDER.md`](../../docs/EVIDENCE_LADDER.md) and
[`../../docs/NOMENCLATURE.md`](../../docs/NOMENCLATURE.md) before writing prose here.
