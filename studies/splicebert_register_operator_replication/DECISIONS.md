# Design decisions — CC/max crossed-register study

## Owner decision at Stop Point 1

**Owner approved 120 nt only and the sequence-selected 342-block clean fallback,
explicitly as outcome-aware operator verification, with the 0.05 resolution gate
unchanged.** The owner's messages were “120 only, any other decisions I have to
make” and, after the cohort options and power risk were reported, “342”.
This accepts a substantial risk of an instrument-unresolved
outcome: its historical AC/linear variance gives MDE80=0.05426. No CC/max variance
or result exists yet. The preferred fresh design cannot supply a valid cohort under
the preserved exclusions.

Cohort and span decisions are complete; no further design choice is pending.
At approval, no protocol had been frozen, no model loaded for this
study, no CC outcome computed, and no claim/manuscript/commit/push made. The owner's
choice of CC/max as primary was made before outcomes and is fixed. The new
[draft protocol](DRAFT_PREREGISTRATION.md) preserves AC and the linear readout as
bridges and AT for the historical sensitivity conditions.

## Fresh-cohort census

The [construction certificate](planning/construction_v1.json) and its
[manifest](planning/construction_v1.manifest.json) bind data, source scripts, exclusion
registries, per-phase/stratum counts and the capacity proof. Two model-free memorials
passed: the independently transcribed C43 eligibility calculation reproduces 219,775
coding edges before filters and 19,437 eligible edges / 4,816 genes after the original
blacklist; the reconstructed PBC gene census matches all role/phase gene hashes and
the 8,790-gene union. PBC selected no blocks or reserve, but the requested exclusion
of its **entire census** is honored.

All rows below apply the CC cleanliness predicate and exclude chr8/9/Y:

| Exclusion policy | Coding genes | Phase 0 / 1 / 2 gene supply per role | Optimistic balanced-block ceiling |
|---|---:|---:|---:|
| Literal requested minimum: C43 + failed 120 development blocks + PBC census | 2,394 | 1,769 / 1,501 / 1,339 | 378 |
| Also keep the predecessor's 120-block confirmation reserve sealed | 1,655 | 1,307 / 1,081 / 934 | 250 |
| Preserve all original C43 prior exclusions | **105** | **80 / 47 / 16** | **0** |

These are generous **six-coding-source ceilings before** searching for a distinct
seventh bridge source or testing seams. They are not constructed cohorts. The first
two policies relax inherited exclusions and are sensitivity calculations, not
authorization to reuse those genes. The minimum-only policy also fails to protect
the predecessor's sealed confirmation reserve and is not recommended.

Under the preserved design, PBC exclusion removes 2,363 of 2,468 otherwise available
clean coding genes, leaving 343 eligible edges in 105 genes. No plus-strand
chromosome stratum can supply two distinct sources for every phase; minus-strand
capacity is at most one block, itself below the two-block bootstrap stratum minimum.
No bridge search can repair this necessary-condition failure. Fresh capacity is zero
for the specified geometry/stratification, not a biological impossibility claim.

## Frozen-cohort hygiene and fallback

Two independent sequence reconstructions agree. All 3,456 historical cell sequence
hashes and 2,304 native-window hashes match hg38. The independent check is preserved
in [independent_hygiene_v1.json](planning/independent_hygiene_v1.json) with a
[manifest](planning/independent_hygiene_v1.manifest.json).

- CC is unsafe at 19/1,152 target acceptor modules: 57/3,456 cells in 18 blocks.
- Native acceptors in donor-role source windows add 16 unsafe modules in 16 further
  blocks. A clean target **and** clean six-source native control set leaves 350
  blocks, 172 plus-strand and 178 minus-strand.
- The target-only 366 and strict 350 subsets each have a singleton chromosome ×
  strand stratum, incompatible with the historical bootstrap precondition.
- Dropping that singleton, then hash-trimming to equal strands while preserving at
  least two blocks per retained stratum yields **342 blocks / 2,394 source genes**,
  171 per strand, 32 strata and 57 occurrences of each S3 permutation per control.
  Selection used only sequences and IDs before the subset variance calculation.

The exact [selection receipt](planning/fallback342_selection_v1.json) and
[manifest](planning/fallback342_selection_v1.manifest.json) record every inclusion,
exclusion and hash preimage. None of C43's original files is altered.

The alternative is to retain all original 384 blocks and quantify the planted-motif
exposure prospectively, with the clean subsets as prespecified sensitivity analyses.
That keeps N and the original gene/sample structure but is **not motif-clean** and
would require an explicit protocol variant replacing the zero-plant gate. It cannot
be described as an independent fresh replication or silently substituted for the
clean proposal. No such override is assumed here.

Keep the original bridge AG→AC negative control. The primary CC/max pseudo-phase
family uses identical two-base CC interventions and is the matched-size structured
null for label alignment. The AC bridge is a separate control, not evidence of
location-specificity of CC perturbations. Adding a clean CC bridge would require
more exclusions and is not necessary for this scoped behavioral question.

## Power and the cost of cleanliness

The [power projections](planning/power_v1.json) have a
[manifest](planning/power_v1.manifest.json) and a reusable
[planning script](../../scripts/splicebert_register_operator_replication/plan_power.py).
They use only within-stratum centered historical residuals; no target sign or
magnitude determines selection. C43's recorded finite-bootstrap MDE80 is 0.04976964;
the exact empirical variance of the same stratified bootstrap gives 0.05008128.
That distinction does not revise C43's fixed-seed PASS.

Illustrative projections from the analytic empirical variance:

| N | Same SD | 10% larger SD | 20% larger SD |
|---:|---:|---:|---:|
| 384 | 0.05008 | 0.05509 | 0.06010 |
| 480 | 0.04479 | 0.04927 | 0.05375 |
| 576 | 0.04089 | 0.04498 | 0.04907 |

The preferred fresh design has none of this capacity. Even the relaxed 250-block
upper bound projects MDE≈0.0621 before additional bridge losses.

For the actual fixed 342 IDs, independently recomputed
[subset variance](planning/fallback342_power_v1.json) and
[manifest](planning/fallback342_power_v1.manifest.json) give:

- exact empirical-bootstrap MDE80 **0.05393105**;
- original 20,000-draw/20260803 MDE80 **0.05425863**;
- CC/max SD would need to be about **7.8% lower** than that historical variance
  proxy to pass under square-root scaling.

This is a risk disclosure, not a prediction of CC's result or permission to weaken
the gate. MDE80 estimates scalar detection against zero; it is not 80% power for
all five directions, the pseudo test, mismatch equivalence, or a lower bound above
the 0.05 materiality threshold. Actual resolution is measured once after scoring;
a failure is reported as instrument unresolved without replacing the cohort.

## Additional spans

The owner selected **120 nt only** at Stop Point 1. The following comparison is
retained as design rationale; additional spans are outside the approved scope.
Candidate additions 144 and 168 nt
keep the same modulo-three class. They require new bridge lengths (48 and 72 nt
versus 24 nt), shifted acceptor/probe geometry, common-source support and seam
checks. The preserved 32+64-nt intronic flanks rule out 90/96-nt spans with a
nonempty bridge. No additional-span construction or model run has been performed.

At N=384, each extra complete span costs approximately another 24–25 minutes of
historical forward time, plus construction/audit work; practical cost is roughly
linear in spans. Measuring the same blocks at three spans does not create three
times as many independent blocks or improve the primary 120-nt resolution.

If all three scalar span effects required simultaneous 95% inference, a conservative
Bonferroni planning calculation raises the MDE factor about 15.5% and the sample
requirement about 33.4%. Correlated max-statistic inference may cost less, but needs
its own frozen family. The default instead keeps one primary span; extras would be
secondary, reported together without selecting a favorable span. No claim of
generality across separations is available from a 120-nt-only rerun.

## Wall-clock budget after approval

The recorded C43 MPS run needed 1,319.78 s for 127,872 unique crossed-cell masked
requests and 74.27 s for 6,912 native requests, at batch size 16. Full original
fidelity replay is therefore **23.23 minutes at the historical rate**. The new
design retaining AC/AT and the AC bridge needs 342 cell requests plus 24 native
requests per block: **140,544 requests / 24.24 minutes at N=384**, or about **21.6
minutes at N=342**. Recording both margins adds no forwards.

The current machine is an Apple M5 Pro with 64 GiB; the historical GPU model was
not recorded, so these are historical-rate estimates, not a benchmark of the new
runner. Checkpoint/data hashes and the recorded Python/NumPy/Torch/Transformers/
pyfaidx versions match the available assets. No new forward benchmark was run.

Budget **4–6 hours after approval** as an engineering estimate: about 2–4 hours for
production transcription, conformance and smoke preparation; 20–45 minutes for
statistical calibration; roughly 45–50 minutes historical model time for full
fidelity plus the new one-span run; and 30–60 minutes for result replay/audit and
paper-ready reporting. Provision 1–2 hours for model stages including startup and
QA. A failed fidelity/calibration gate stops execution and invalidates this delivery
estimate; the schedule does not override a failed gate.

## Planning checks before execution

- Focused planning/motif tests: **25 passed**.
- Full locked environment suite with GTF/FASTA/FAI set: **1,222 passed, 12 skipped**.
- Manifest registry verification: passed, 1,369 existing rows.
- Planning manifests: input/output hashes verified. The independent selected-power
  code reproduces the all-384 historical MDE float exactly and matches the reviewer’s
  342-block calculation. Exact binomial acceptance constants were independently checked.
- Repo-discipline checker: no findings on the new study; one existing dependency
  failure in `scripts/dnabert_promoter/build_promoter_cohort.py` (`datasets` absent)
  plus legacy advisory findings. No unrelated file was changed to silence it.
- R49's model-free raw-row reproduction was checked read-only: 25/25 fields bit
  identical. This is not a new model fidelity replay.

At this planning checkpoint, not run: new construction/model runner, full old-model AC fidelity, stochastic
calibration, native CC controls, CC target, any additional span, or final result
audit. The strongest new evidence is **construction infeasibility for the requested
fresh cohort and an audited sequence-only fallback with a quantified power risk**.
It licenses no new biological or model-behavior claim.

Larger census tables, reconstruction scripts, and independent review notes remain
in the user-specified external side-session notes folder. The compact design receipts
here preserve their hashes; they are planning records, not a claim that the future
execution package is already portable or frozen.

## Frozen execution and calibration stop

The owner-approved design was frozen in [protocol_freeze_v1.json](protocol_freeze_v1.json),
SHA-256 `82bc168e2246dff82519215203e02747f0209c289fa00c3e538613d95573d854`.
The controlling prose is now [PREREGISTRATION.md](PREREGISTRATION.md) and
[STATISTICAL_CALIBRATION.md](STATISTICAL_CALIBRATION.md); the draft files preserve
design history and do not supersede the freeze.

The execution construction is
[execution_construction_v3.json.gz](planning/execution_construction_v3.json.gz), with
its independent [PASS audit](planning/execution_construction_v3.audit.json).
Preparation versions v1/v2 remain immutable. Before freezing, v3 corrected the
stop-diagnostic arithmetic to canonical integer counts divided once by six, so
equal stop excesses have equal ranks and median-group membership. The selected
median is 2/3, with 114 blocks above and 228 at or below it. The historical block
stop fields were preserved. Exact development source snapshots needed to retrieve
earlier manifest inputs remain in the external side-session notes under
`cc-execution-inputs/development-source-snapshots/`.

Smoke passed 167 tests and 18 deterministic memorials. Full stochastic calibration
then returned **INSTRUMENT_UNRESOLVED**: five-direction family noncoverage was
**83/1,000**, above the frozen limit of **71**. The full positive-decision false
positive count was 0/1,000, and sensitivity was 255/256 at Ψ=0.05 and 252/256 at
Ψ=0.15; these passes cannot override the failed coverage gate. See the complete
[result account](RESULTS.md). The
[independent audit](../../results/splicebert_register_operator_replication/analysis_v1.audit.json)
reproduced all 1,000 consumed null raw-payload hashes and reductions, and all 6,000
interval families (maximum absolute discrepancy 5.413×10⁻¹⁶). It returns PASS for
the stop audit, `INSTRUMENT_UNRESOLVED` for the run, and `authorized_for_cc=false`.

No model was loaded. Historical forward fidelity, native CC, crossed CC and target
analysis did not run. Actual stage authorization was checked and rejects all three
model stages on this failed calibration. The stable full repository suite passed
1,364 tests with 12 skipped, excluding the two later-written audit test files, which
subsequently passed 62 focused tests. No v1 threshold, amplitude, seed, cohort or output
was changed after the failure.

## Candidate statistical repair — a new design choice, not executed

This entry records the decision state at the v1 stop. The later authorization and
development evidence are recorded in the next entry; the failed v1 record is unchanged.

The failure is not permission to bypass calibration or report a CC estimate.
The current task preserves C43's statistical procedure; repairing that procedure
would be an additional methodological change beyond the approved operator/cohort
change. No such amendment has been frozen or run.

One analytically motivated candidate is a finite-stratum residual correction.
For an endpoint vector x in stratum h of size n_h, form
`y_hi = mean_h(x) + sqrt(n_h/(n_h−1)) * (x_hi − mean_h(x))`, then apply the same
whole-block resampling indices to y. Apply the same scalar factor to every endpoint
from a block, retaining their covariance and equal-block weighting. The cohort
mean is unchanged. Conditional bootstrap covariance becomes the unbiased
within-stratum sample covariance contribution, rather than its factor
`(n_h−1)/n_h` underestimate. For equal population covariance across this cohort's
32 strata, the uncorrected expected variance ratio is `(342−32)/342 = 0.906433`.

This correction is **a candidate, not a validated repair**. The five-direction
failure includes 51/500 additive and 32/500 heterogeneous null realizations; the
additive fixture has quantization-scale cyclic contrasts. Correcting the known
variance bias alone is not evidence that all six interval families will calibrate.
Any candidate must first survive an independent design audit and fail-capable
deterministic tests, then be frozen and checked on a new, declared validation
stream. The consumed v1 streams may diagnose a repair but cannot validate it as
fresh evidence. No searching successive seeds, changing the acceptance threshold,
weakening planted amplitudes, or treating the first passing attempt as confirmation.

A repaired protocol would keep **120 nt, the same 342 IDs, CC/max as primary, AC/AT
and both margins, the 99 controls, and MDE80≤0.05**. Historical AC/linear forward
fidelity would still be checked against the original 384-block artifact using its
original inference; a repaired new-CC analysis must be named separately. Increased
uncertainty could make the already marginal resolution gate harder to pass. A new
operator result would remain outcome-aware verification rather than fresh biological
replication. The failed v1 result remains part of the record regardless of any repair.

## Authorized A1 repair and consumed-data development

After being told that a repair changes the inherited inference procedure, the owner
answered **“authorized”**. This authorizes the separately versioned repair with the
same 342 blocks, 120 nt, CC/max primary operator and MDE80≤0.05 threshold. The exact
single candidate and advancement rule are in [AMENDMENT_A1.md](AMENDMENT_A1.md).
It neither changes v1 nor authorizes seed searches or further methodological retries.

Independent design review accepted the finite-stratum correction before the complete
consumed-data replay. The [development record](planning/inference_repair_v2.development.json)
then passed all unchanged calibration requirements: family noncoverage counts were
52, 65, 59, 50, 21 and 27 out of 1,000 in the frozen family order; full positive-decision
false positives were 0/1,000; planted Ψ=0.05 detection was 254/256 and planted Ψ=0.15
material classification was 250/256. Every observed block endpoint, point estimate
and pseudo-control realization remained exactly unchanged across all 1,768 cases.
No fresh calibration root or model outcome was used in that replay.

These are development results from already-consumed simulations. They permit
advancing to the one declared fresh validation after source freeze and smoke; they
do not validate the repaired instrument as fresh evidence. The full historical
384-block AC/linear fidelity gate retains its original inference method.

The independently reviewed candidate was adopted without further alteration by
[protocol_freeze_v2.json](protocol_freeze_v2.json), SHA-256
`f8d443cd1cad29af8ccb47a3c3b16278db8ed738ed0117d1df31ab86284cfd0d`.
That receipt freezes the unchanged A1 text (whose original draft banner preserves
its preparation state), the full source/test and preserved-v1 evidence closure,
and new output paths. The [design audit](planning/inference_repair_v2.design_audit.json)
passed exact unequal-stratum enumeration and scope/version checks. Post-freeze
[smoke](../../results/splicebert_register_operator_replication/smoke_v2.json) passed
480 tests and all 18 statistical memorials.

The single [fresh calibration](../../results/splicebert_register_operator_replication/statistical_calibration_v2.json)
then passed: family noncoverage was **36, 47, 56, 42, 23 and 18/1,000** in the frozen
family order; full positive-decision false positives were **0/1,000**; sensitivity
was **252/256** at Ψ=0.05 and **253/256** at Ψ=0.15. There were no execution errors.
The [independent calibration audit](../../results/splicebert_register_operator_replication/statistical_calibration_v2.audit.json)
recounted all 1,768 realizations, regenerated all 1,000 null raw payloads with exact
hashes and reductions, and independently reconstructed all 1,000 complete null
inferences and 6,000 interval families. Maximum numerical discrepancy was
8.882×10⁻¹⁵. It explicitly authorized historical fidelity. This is a PASS of the
declared synthetic calibration, not proof of exact nominal coverage or any model
behavior. C45/R50 and the failed v1 remain unchanged.

## Historical fidelity and native preflight before crossed scoring

The complete 384-block historical AC/linear forward replay passed its independent
[fidelity audit](../../results/splicebert_register_operator_replication/historical_fidelity_v2.audit.json):
2,268,153 raw checks had zero failures; stored margins and protected log probabilities
matched exactly, and 12,618 original-method inference fields were bitwise identical.
This gate was completed before any native or crossed CC model forward.

The [native CC run](../../results/splicebert_register_operator_replication/native_v2.json.gz)
then passed its corrected preflight: equal-block coupling was 0.333075 with 95% CI
[0.311516, 0.355360], above the required lower bound of 0.05. Repeated forwards
matched exactly. These are native controls, not the crossed target effect; the final
independent raw-measurement audit must still verify them. The fixed crossed run
was then started with 116,964 requests. No threshold, seed, cohort or operator was
changed after either control result.

## Completed v2 run: resolution-only failure retained

The crossed stage completed all 116,964 requests. Both independent audits passed,
reconstructing the native preflight and every target endpoint. The final immutable
[result](../../results/splicebert_register_operator_replication/result_v2.json)
records `INSTRUMENT_UNRESOLVED`: MDE80=0.064708 exceeds the fixed 0.05 limit. All
other nine gates pass; none is unrun. Measured Ψ=+0.429719 [0.384891, 0.475976] is
retained with every bridge, control and stop diagnostic. No favorable pattern is
used to waive resolution. Draft C46/R51 and [RESULTS.md](RESULTS.md) record the
complete outcome, preserving C45/R50 and the original C43/C44 rows verbatim.
No additional experiment or methodological retry was initiated, and the manuscript
and Git index were not edited by this execution.


## Result consolidation and artifact registration

The completed v2 result is registered retrospectively in the repository manifest.
The compact export contains the original estimate, all ten gate outcomes, all six
prespecified scalar arms, source hashes, and the retained v1 stop. A separate
342-row table supports scalar reanalysis. The full execution prose is preserved
in `RESULTS_DETAIL.md`; the result page is now concise. No frozen artifact or
claim row changed, and the MDE80 limit remains 0.05. Raw forwards and calibration
streams stay locally preserved and ignored, with their original manifests.
See R52 and the result inventory for the exact storage and availability boundary.
