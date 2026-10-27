# CC/max crossed-register verification: audited v2 result

**V2 completed; the frozen classification is `INSTRUMENT_UNRESOLVED`.** The measured
CC/max contrast is **Ψ=+0.429719, 95% CI [+0.384891, +0.475976]** logit units.
**Resolution is the sole failed capability gate: MDE80=0.064708>0.05.**
The other nine gates pass, including the independent audit; no gate is unrun.
These estimates are retained observations from a failed instrument qualification,
not a licensed positive, bounded-null, inverted, or equivalent target conclusion.

This is the owner-approved **120-nt-only**, **342-block** outcome-aware operator
verification: 2,394 globally distinct source genes, 171 blocks per strand, and 32
chromosome × strand strata. CC/max was fixed before CC outcomes. The cohort is a
sequence-selected motif-clean subset of C43, whose AC outcomes were already known.
It is not a fresh outcome-blind biological replication. The
[A1 amendment](AMENDMENT_A1.md) applies the reviewed finite-stratum residual correction
to the same 20,000 whole-block bootstrap draws (seed 20260803). No threshold, seed,
block, readout, or corruption was changed after this result.

Draft [C46](CLAIMS.md) and [R51](../splicebert_partner_selection/CORRECTIONS.md#r51)
record this version. C43/C44 and the failed v1 C45/R50 remain unchanged. The complete
v1 report is preserved verbatim [below](#preserved-v1-report).

## Primary endpoints and structured controls

| CC/max endpoint | Observed value |
|---|---:|
| Ψ, ordinary 95% CI | +0.429719 [+0.384891, +0.475976] |
| M0, matched mean | +0.376054 |
| M+, first cyclic mismatch mean | -0.068210 |
| M−, second cyclic mismatch mean | -0.039119 |
| Largest of 99 balanced pseudo-phase controls | +0.047619 |
| Exact one-sided add-one p | 0.01 |
| Target minus mean pseudo-control, mean / one-sided lower95 | +0.430412 / +0.392682 |

The target exceeds every structured control. The p-value is the **0.01 floor of
this 99-control test**, not a measure of separation. The five simultaneous
95% directional lower bounds are:

| Direction | Mean | Simultaneous lower95 |
|---|---:|---:|
| `m0_minus_plus` | +0.444264 | +0.380245 |
| `m0_minus_minus` | +0.415174 | +0.355741 |
| `donor_phase_0_excess` | +0.447839 | +0.363748 |
| `donor_phase_1_excess` | +0.375644 | +0.289480 |
| `donor_phase_2_excess` | +0.465673 | +0.379586 |

Both mismatch-equivalence intervals extend outside ±0.05: M+ is
-0.068210 [-0.101424, -0.034995] and M− is -0.039119 [-0.069099, -0.009139].
Thus **“only when matched” is not licensed** even apart from the resolution failure.
Positive directional bounds and control superiority do not override a failed
capability gate.

## All ten capability gates

| Gate | Recorded check | Verdict |
|---|---|---|
| 1. Construction and prerequisites | Construction, smoke, fresh calibration, and full historical fidelity all audited PASS | PASS |
| 2. Native positive control | CC/max +0.333075 [+0.311516, +0.355360]; lower95>0.05 | PASS |
| 3. Crossed capability | Phase-blind CC/max +0.089575 [+0.074954, +0.104014]; lower95>0.02 | PASS |
| 4. Bridge negative control | Bridge-site AG→AC/max +0.015715 [-0.007279, +0.036435]; CI within ±0.05 | PASS |
| 5. Resolution | Centered-stratified MDE80=0.064708; required ≤0.05 | **FAIL** |
| 6. Historical alternative corruption | Phase-blind AC−AT/linear +0.005362 [+0.002175, +0.008548]; CI within ±0.05 | PASS |
| 7. Protected MLM health | True-base log-probability change -0.004285 [-0.005561, -0.003021] nat; lower95>−0.10. Accuracy retention 0.997258≥0.80 | PASS |
| 8. Site health | Intact donor/acceptor max-margin lower95 +2.915465 / +1.559053, each above native floor 0 | PASS |
| 9. Numerics | Complete finite 1,024-token requests, correct masks, exactly matching repeated forwards | PASS |
| 10. Independent audit | Raw measurements, inference, gates, controls, diagnostics and provenance independently reconstructed | PASS |

Protected accuracy counts are 36,368 construct-correct versus
36,468 source-correct across 73,872 aligned probe pairs.
The bridge-site linear counterpart is +0.013721 [-0.006375, +0.032108].
Intact donor/acceptor linear margins are +3.479654 [+3.364835, +3.592960] and
+2.347362 [+2.246667, +2.448955]; their native floors are also zero.

The corrected analytic MDE80 is 0.064213. The explicitly
**uncorrected** analytic diagnostic is 0.061178;
neither replaces the frozen finite-draw gate. No additional bootstrap seed or
larger cohort was tried. The failure is a resolution failure, not a null result.

## AC bridge and readout comparisons

All arms below use the same selected 342 blocks and corrected v2 uncertainty.
They must not be substituted for the original 384-block C43 estimate.

| Corruption / readout | Ψ and ordinary 95% CI | M0 | M+ | M− |
|---|---|---:|---:|---:|
| `cc_max` | +0.429719 [+0.384891, +0.475976] | +0.376054 | -0.068210 | -0.039119 |
| `cc_linear` | +0.414338 [+0.372480, +0.457324] | +0.364443 | -0.064866 | -0.034925 |
| `ac_max` | +0.408252 [+0.365822, +0.451663] | +0.358383 | -0.075351 | -0.024387 |
| `ac_linear` | +0.392826 [+0.353274, +0.433231] | +0.346959 | -0.072396 | -0.019338 |
| `at_max` | +0.406033 [+0.363505, +0.449782] | +0.350891 | -0.076738 | -0.033545 |
| `at_linear` | +0.391285 [+0.351187, +0.432104] | +0.340570 | -0.073638 | -0.027792 |

| Paired Ψ contrast | Mean and four-family simultaneous 95% CI |
|---|---|
| `psi_cc_minus_ac_max` | +0.021467 [+0.012611, +0.030323] |
| `psi_cc_minus_ac_linear` | +0.021512 [+0.013212, +0.029812] |
| `psi_max_minus_linear_cc` | +0.015380 [+0.008294, +0.022467] |
| `psi_max_minus_linear_ac` | +0.015426 [+0.008506, +0.022345] |

The historical Ψ(AC−AT)/linear sensitivity interval is
+0.001541 [-0.005029, +0.008112], wholly within ±0.05.
The diagnostic phase-blind AC−AT/max interval is
+0.006012 [+0.002585, +0.009439]. Operator/readout differences
are reported, not used to select a preferred primary. Competing-base identities
and tie counts are retained in the [producer analysis](../../results/splicebert_register_operator_replication/analysis_v2.json).
The bridge-site control remains AG→AC, as frozen; it is not a CC-matched sham or
proof of CC spatial specificity.

## In-frame stop-codon diagnostic and sensitivity

Mean annotated-frame junction stop counts, donor phase by acceptor phase:

| Donor phase | Acceptor 0 | Acceptor 1 | Acceptor 2 |
|---|---:|---:|---:|
| 0 | 0.002924 | 0.573099 | 0.751462 |
| 1 | 0.637427 | 0.014620 | 0.567251 |
| 2 | 0.608187 | 0.640351 | 0.017544 |

Using the frozen canonical stop-excess median **2/3**, Spearman
ρ(stop excess, block Ψ)=-0.136205; descriptive two-sided
p=0.011688. Above-median blocks have mean
Ψ=+0.340640 (n=114); blocks at or below the median have
Ψ=+0.474258 (n=228). These descriptive summaries do not
separate annotated phase from stop composition or identify a mechanism.

Leave-one-block estimates range from +0.423627 to
+0.432182; leave-one-chromosome estimates range from
+0.403495 to +0.441427.
Excluding the plus strand gives +0.397609; excluding the
minus strand gives +0.461828. These are sensitivity ranges,
not confidence intervals, and they do not waive the resolution gate.

## Calibration, fidelity, and independent verification

The authorized single A1 repair first passed its independent design audit and
one replay of the consumed v1 simulations as development evidence. The
[v2 freeze](protocol_freeze_v2.json), SHA-256
`f8d443cd1cad29af8ccb47a3c3b16278db8ed738ed0117d1df31ab86284cfd0d`,
preceded post-freeze smoke and the single fresh validation. Fresh null noncoverage
counts were **36, 47, 56, 42, 23 and 18/1,000** for primary Ψ, directions, mismatches,
paired bridges, historical linear Ψ(AC−AT), and historical phase-blind AC−AT.
Each passed the unchanged limit of 71. Full positive-decision false positives were
**0/1,000** (limit 38); planted 0.05 detection was **252/256** and planted 0.15
material classification was **253/256** (each required ≥216). This passes the
specified synthetic calibration; it does not prove exact nominal coverage generally.

The independent calibration audit regenerated all 1,000 null raw payload hashes,
reduced them exactly, and reconstructed all 6,000 interval families (maximum
absolute numerical difference 8.882×10⁻¹⁵). Before any CC forward, full historical
384-block AC/linear replay then passed **2,268,153 raw checks with zero failures**:
stored margins and protected log probabilities matched exactly, and **12,618
original-method inference fields were bitwise identical**. The historical inference
was not amended to A1.

The independent new-measurement audit passed **2,128,816 checks with zero failures**
across native and crossed data. All recorded margins and repeat logits matched
exactly; the largest independent log-probability recomputation discrepancy was
4.825×10⁻⁷ nat, within the frozen 10⁻⁴ tolerance. The final result auditor re-reduced
**125,172 raw requests** and compared **21,346 numeric inference fields**,
with maximum absolute discrepancy **2.665e-15**.
It independently confirmed all ten gates and the resolution-only failure. These
audits loaded no model and generated no further forwards.

## Proposed paper wording and L131 disposition

The manuscript remains unchanged. A supported replacement for the L131 paragraph is:

> We re-evaluated fixed-span crossed donor–acceptor chimeras using AG→CC corruption
> and the primary max-margin donor readout. Across 342 motif-clean seven-source
> blocks (2,394 globally distinct genes) at a 120-nt span, the measured matching
> contrast was Ψ=+0.430, 95% CI [0.385, 0.476] logit units. This was outcome-aware
> verification on a subset of the earlier cohort. Although all five directional
> lower bounds were positive and the estimate exceeded all 99 balanced pseudo-phase
> controls (exact add-one p=0.01), the preregistered resolution gate failed
> (MDE80=0.0647>0.05). The frozen verdict is therefore instrument-unresolved, and
> this run does not establish a confirmed CC/max register preference.

The following positional-results paragraph should not use this run to assert that
coupling “is stronger for register-matched chimeras.” Retain the independently
supported learned-component phase-sensitivity statement, and describe the crossed
CC/max evidence as unresolved under its frozen qualification rule. Likewise remove
“and favors matched annotated boundary registers” from the conclusion if the paper
is to rely throughout on the primary CC/max operator. C43 remains its historical
AC/linear finding in the ledger; it is not silently relabeled CC/max. Neither this
run nor its stop diagnostic establishes a period-3 implementation, mediation,
phase independent of stop composition, or generality beyond the single 120-nt span.

## Evidence and verification record

- [Final v2 record](../../results/splicebert_register_operator_replication/result_v2.json)
  and [manifest](../../results/splicebert_register_operator_replication/result_v2.manifest.json):
  SHA-256 `ba3433ab09bbeb2425e8d6b077c25b2ad997ba20f608c360bf367115a8ad8087`.
- [Independent result audit](../../results/splicebert_register_operator_replication/analysis_v2.audit.json)
  and [manifest](../../results/splicebert_register_operator_replication/analysis_v2.audit.manifest.json):
  SHA-256 `b8e4cb7d13b911f67c9fa316364ed94e6538d56a056ca16c9417a093858d5dd5`.
- [Independent raw-measurement audit](../../results/splicebert_register_operator_replication/measurement_v2.audit.json)
  and [manifest](../../results/splicebert_register_operator_replication/measurement_v2.audit.manifest.json).
- [Crossed raw requests](../../results/splicebert_register_operator_replication/cells_v2.json.gz),
  [native requests](../../results/splicebert_register_operator_replication/native_v2.json.gz),
  and [historical replay](../../results/splicebert_register_operator_replication/historical_fidelity_v2.json.gz)
  each have sibling manifests, exclusive output paths, and uncompressed-content hashes.
- [Fresh calibration](../../results/splicebert_register_operator_replication/statistical_calibration_v2.json),
  [calibration audit](../../results/splicebert_register_operator_replication/statistical_calibration_v2.audit.json),
  [historical fidelity audit](../../results/splicebert_register_operator_replication/historical_fidelity_v2.audit.json),
  and [smoke receipt](../../results/splicebert_register_operator_replication/smoke_v2.json).

The producer analysis retains its immutable `AWAITING_INDEPENDENT_AUDIT` state;
the later audit and final record supply the completed verdict. Execution commands,
UTC attempt/completion times, hardware, package versions, checkpoint/data pins,
parameters and source hashes are in the manifests. Timestamps are recorded provenance,
not cryptographic proof of wall-clock chronology. Full calibration realization
streams remain local and excluded from Git; compact evidence is still uncommitted
draft material, not part of the tracked-result registry or a published claim.

The full repository suite passed **1,598 tests, with 12 skipped**. The final frozen
study smoke passed **480 tests plus 18 statistical memorials**. The shared-genome
full-suite command was `uv run --locked --extra splicebert --extra biology --extra dev pytest -q`
with `GFMCD_GTF`, `GFMCD_FASTA`, and `GFMCD_FASTA_FAI` set. After the documentation
updates, **56 ledger, manifest, hash-pin, and repository-hygiene tests passed**.
All **106 frozen files**, **11 v2 artifact manifests**, and **205 local links**
verified; C43/C44 and both C45 rows remained byte-identical, the v1 prose remained
verbatim, and the manuscript hash was unchanged. The existing tracked registry
verified all 1,369 rows; the new uncommitted evidence is not yet in that registry.
The repository-discipline checker retained its pre-existing missing-`datasets`
failure in `scripts/dnabert_promoter/build_promoter_cohort.py:9` and 67 legacy
advisory findings, with no findings for this study. No unrelated dependency change
was made. No additional spans, fresh biological cohort, carrier interventions,
methodological retry, manuscript edits, staging, commits, or pushes were performed.

<a id="preserved-v1-report"></a>

---

The following report is the preserved v1 snapshot. Its “not run” statements refer
to v1; the separately authorized and audited v2 execution is recorded above.

# CC/max crossed-register verification: calibration stop

**Run status: independently verified `INSTRUMENT_UNRESOLVED`, before any model
forward.** The owner selected 120 nt only and
the sequence-selected 342-block fallback. The primary operator remained AG→CC
with the max-margin donor readout throughout. The 0.05 resolution threshold was
not relaxed.

## Recorded result

The [frozen protocol](PREREGISTRATION.md) and
[calibration specification](STATISTICAL_CALIBRATION.md) were bound by
[protocol_freeze_v1.json](protocol_freeze_v1.json), SHA-256
`82bc168e2246dff82519215203e02747f0209c289fa00c3e538613d95573d854`,
before stochastic calibration. The
[calibration result](../../results/splicebert_register_operator_replication/statistical_calibration_v1.json)
records 256 development realizations, 1,000 null realizations, and 256 paired
planted-signal realizations at each of two amplitudes. It finished without an
execution error in 276.09 seconds. No model was loaded.

The five-direction simultaneous confidence family failed its coverage gate:
**83/1,000** intervals missed at least one known endpoint, versus a frozen maximum
of **71**. The observed family noncoverage rate is 8.3%, with exact binomial 95% CI
**[6.665%, 10.185%]**. Passing sensitivity or false-positive checks does not override
this failure.

| Calibration requirement | Observed | Frozen rule | Result |
|---|---:|---:|---|
| Full positive-decision false positives | 0/1,000 | ≤38 | PASS |
| Primary Ψ interval noncoverage | 62/1,000 | ≤71 | PASS |
| Five-direction family noncoverage | **83/1,000** | ≤71 | **FAIL** |
| Two-mismatch family noncoverage | 70/1,000 | ≤71 | PASS |
| Four paired operator/readout contrasts noncoverage | 70/1,000 | ≤71 | PASS |
| Historical linear Ψ(AC−AT) noncoverage | 29/1,000 | ≤71 | PASS |
| Historical linear phase-blind AC−AT noncoverage | 33/1,000 | ≤71 | PASS |
| Detect planted Ψ=0.05 with lower bound >0 | 255/256 | ≥216 | PASS |
| Full material-positive decision for planted Ψ=0.15 | 252/256 | ≥216 | PASS |

The one-sided 95% upper bound on the full positive-decision false-positive rate
is 0.2991%. The one-sided 95% lower bounds for the two sensitivity rates are
98.1604% and 96.4603%, respectively. These describe the synthetic instrument,
not SpliceBERT behavior. In particular, the 0/1,000 false-positive count does not
establish correct confidence-interval coverage.

The [independent audit](../../results/splicebert_register_operator_replication/analysis_v1.audit.json)
recounted every realization, regenerated all **1,000 consumed null raw payloads**
with matching hashes, independently reduced their recorded logits, and replayed
all **6,000 interval families**. Reduced values matched exactly; the largest
absolute interval discrepancy was **5.413×10⁻¹⁶**. No alternative seeds or inference
settings were evaluated. Audit PASS verifies the failed calibration and resulting
stop; it does not authorize CC execution.

The failure separates into **51/500 additive** and **32/500 heterogeneous** nulls.
In the additive half, 16 failures involved only the two quantization-scale cyclic
contrasts, 34 involved only the three ordinary-noise donor-phase contrasts, and one
involved both groups. Thus the failure cannot be dismissed as solely a rounding
artifact. The finite-stratum bootstrap variance underestimate under the iid
synthetic fixture is a plausible contributor, but its correction has not been
shown to repair this family coverage.

## Execution and gate inventory

The independently reconstructed construction passed for all 384 historical blocks
and for the selected 342 clean blocks. The post-freeze smoke run passed 167 tests
and all 18 deterministic statistical memorials. The stable full repository suite
then passed **1,364 tests, with 12 skipped**; the two subsequently authored audit
test files passed **62 focused tests** separately. After the draft C45/R50 edits,
**65 ledger, provenance, and repository-hygiene tests passed**. Their first pass
caught three stale R1–R49 range summaries; these now correctly say R1–R50.

| Stage | Status |
|---|---|
| Independent construction audit | PASS |
| Source-bound protocol freeze | FROZEN |
| On-path smoke checks | PASS |
| Full stochastic statistical calibration | **FAIL: directional coverage** |
| Full 384-block historical AC/linear forward fidelity | NOT RUN |
| Native CC positive controls and floors | NOT RUN |
| 342-block crossed CC/AC/AT model scoring | NOT RUN |
| Target inference and ten-gate classification | NOT RUN |
| Independent verification of the calibration stop | PASS; stop retained |

Target gate 1 cannot pass because statistical calibration failed and historical
forward fidelity was not run. Target gates 2–9 are unmeasured: native positive
control, crossed capability, AC bridge, actual CC resolution, AC/AT robustness,
protected health, site health, and model numerics. Gate 10, an independent audit of
target model evidence, has no target evidence to examine. An audit of the stop
does not turn these unrun gates into passes.

The actual stage-authorization function was checked against the failed calibration:
it rejects fidelity, native, and cells stages before loading a model. No subsequent
attempt or target output exists.

## Requested scientific endpoints

There is **no new measured Ψ or CI, M0/M+/M−, directional bound, pseudo-control
maximum, add-one p-value, AC bridge, actual CC MDE, or stop-load/Ψ association**.
The corresponding fields in the final stopped-run record must be null rather than
zero. Synthetic calibration estimates cannot fill any of these fields.

Sequence-only stop counts were computed before outcomes. The selected construction's
mean stop-count matrix (donor phase by acceptor phase) is:

| Donor phase | Acceptor 0 | Acceptor 1 | Acceptor 2 |
|---|---:|---:|---:|
| 0 | 0.002924 | 0.573099 | 0.751462 |
| 1 | 0.637427 | 0.014620 | 0.567251 |
| 2 | 0.608187 | 0.640351 | 0.017544 |

Canonical block stop excess has median 2/3, with 114 blocks above and 228 at or
below it. Without model outputs, no correlation or high/low coupling contrast is
available. This sequence diagnostic does not separate annotated phase from stop
composition.

## Interpretation and paper disposition

The strongest new conclusion is an **instrument failure in the frozen synthetic
coverage test**. It is neither a positive result nor a bounded null for CC/max
register preference, and it does not revise C43's historical numerical result.
The cohort would have been outcome-aware operator verification, not a fresh
outcome-blind replication. No additional span or period-3 mechanism was tested.

There is no valid CC/max replacement for the paper's L131 numerical paragraph from
this run. The manuscript has not been edited. If the attempted rerun needs to be
reported, the supported wording is:

> A preregistered AG→CC/max-margin verification on 342 motif-clean seven-source
> blocks at a 120-nt span stopped before model scoring because the five-direction
> confidence-interval calibration missed known synthetic truths in 83/1,000 null
> realizations, exceeding the frozen limit of 71. The rerun therefore provides no
> new register-preference estimate under the primary operator.

Any repair must preserve this failed version and establish a new, reviewed protocol
before further execution. Passing a different procedure would not erase the v1
failure. The already opened calibration streams cannot be presented as fresh
validation of a repair.

## Reproduction and retained evidence

The [calibration manifest](../../results/splicebert_register_operator_replication/statistical_calibration_v1.manifest.json)
binds the exact input/source files and the full realization stream. The local stream
`results/splicebert_register_operator_replication/statistical_calibration_v1.records.jsonl.gz`
has SHA-256
`2c941d6d7c2e7a2bf186c2d0223bba5258001baac226bb8188645c1f473e6ead`.
It is retained locally and excluded from Git because it is about 611 MiB. Compact
results, manifests, and audits are the reviewable records; no files are committed
or pushed by this task.

Frozen execution commands are recorded in the manifests. Output files are exclusive:
do not rerun a producer against an existing v1 output. The full-suite command was
`uv run --locked --extra splicebert --extra biology --extra dev pytest -q`, with
`GFMCD_GTF`, `GFMCD_FASTA`, and `GFMCD_FASTA_FAI` pointing to the shared genome files,
and with the two new independent-audit test files excluded while they were being
written. A prior concurrently edited suite run had one stale authorization-fixture
failure; the final stable suite and frozen smoke run passed. No failed execution
or calibration output was overwritten.

The compact [final record](../../results/splicebert_register_operator_replication/result_v1.json)
has SHA-256 `1cd6df321679fb649bce5d49562a7a44b9c637c33d71e25ed7fd01e317b1dfc3`
and a [manifest](../../results/splicebert_register_operator_replication/result_v1.manifest.json).
It records all unmeasured model endpoints as null and explicitly denies CC execution.
The audit's chronology limitation is retained: the calibration attempt JSON has no
UTC creation field or separate manifest. Its bytes are bound by the final calibration
manifest; filesystem mtime only corroborates start order. The freeze time and final
manifest time are recorded, but an exact start timestamp is not cryptographically
recorded in the attempt itself.

The existing tracked-result registry passed (1,369 rows). The repository-discipline
checker retained the pre-existing missing-`datasets` failure in
`scripts/dnabert_promoter/build_promoter_cohort.py:9` and legacy advisory findings;
no unrelated dependency change was made. The new study's uncommitted compact
artifacts are not yet part of that tracked-result registry. Draft C45 and R50 are
review material, not a claim that these artifacts have been committed or published.
