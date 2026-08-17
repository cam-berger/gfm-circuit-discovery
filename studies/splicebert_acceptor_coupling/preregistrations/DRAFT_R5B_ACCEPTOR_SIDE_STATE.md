# DRAFT R5b — cross-fitted acceptor-side residual state `U_A`

> **Status: DRAFT — MODEL-FREE AUDIT RUNNABLE; MODEL RUNS HARD-BLOCKED.** The fail-closed runner is
> implemented at `scripts/splicebert_acceptor_coupling/run_r5b_acceptor_side_state.py`, but every
> literal protocol/instrument/cohort/entry/preflight pin is still `None`. Consequently both
> `--stage preflight` and `--stage target` stop before inspecting any supplied model, data, cohort, or
> artifact path. No model preflight, target artifact, scientific result, R3c held-out execution, or
> chr8/9 access is authorized by this draft.

## 1. Question and claim boundary

Conditional on R5a establishing a causal paired-donor read through `L3H1`, does a stable,
cross-fitted, fold-indexed subspace family `U_A` of the clean-versus-paired-donor-corrupt difference
at the **final masked-acceptor residual** carry a material portion that is both sufficient and necessary for the
native single-G acceptor margin?

R5b can establish a low-rank final-residual mediator for a material portion of the tested model
behavior. Without a separately passed completeness/faithfulness gate it cannot establish that the
subspace contains the whole paired-donor effect. It cannot show
that `L3H1` writes that state, identify a downstream delivery route, establish example-specific
content, name the state's semantics, or establish a biological mechanism. Those first two questions
belong to R5c; semantics and biological causation are outside R5a--R5c.

Every outcome is the masked-acceptor margin. Masked-donor endpoints are forbidden. All residual
interventions are scored through the checkpoint's complete MLM head; the map from final residual to
margin is **not called linear**.

## 2. Fixed-sequence entry and stopping rule

The fixed sequence is **R3c -> R5a -> R5b -> R5c**. R5b may open only if:

1. R3c and R5a were frozen before their target outcomes;
2. every R5a instrument/control gate passed; and
3. R5a returned `L3H1_DONOR_LOCAL_READER_SUPPORTED` under its complete corrected families.

The nested `L2H15_ADDRESS_REUSE` result is not required to define `U_A`, but its status travels with
the final interpretation. If R5b has no rank passing the rule in Section 7, R5c stays closed as a
valid scientific stop; a plumbing/health/control-capability anomaly also closes R5c but is
`INSTRUMENT_UNRESOLVED`. No rank, energy threshold, or alternate representation may be chosen after
seeing either outcome.

All R5a--R5c methods and fingerprints are sealed together before the first model score on the
mechanism-evaluation reservation, as required by R5a Section 2. R5b may execute only after the frozen
R5a gate passes, but its protocol, runner, rank/null construction, overlap SESOI, calibration registry,
and hashes may not change in response to R5a evaluation outcomes.

The common seal is the canonical `gfmcd.r5.stack_binding/1`: normalized all-three-runner bytes, exact
all-three-protocol bytes, the R3c dynamic entry runner, the complete repository-local Python helper
closure, executable registries, schema vocabulary, and stable package runtime. R5b recomputes this
payload, requires the exact same `r5_stack_fingerprint_sha256` embedded in R5a and its entry contract,
and serializes both payload and digest for R5c. Runner-local fingerprints remain additional seals;
three unrelated local fingerprints are not a joint prospective freeze.

## 3. Cohort and folds

R5b imports the shared v2 R5 mechanism census and reservation builder from
`gfmcd.splicing.acceptor_mechanism_cohort`; it does not maintain a second selection implementation.
Its literal cohort,
construction, exclusion, checkpoint, GTF, FASTA, protocol, runner, helper-core, and environment hashes
must match the R5a target-bindable preflight. It never materializes chr8/9.
The exclusion registry is not an independent CLI choice: R5b accepts only the fresh
`gfmcd.r5.exclusion_registry/1` emitted and manifest-bound by the authenticated R5a target. Its
self-digest and embedded R3c artifact/manifest hashes must reproduce the exact upstream R3c package;
an alternate path is rejected even if its gene lists happen to match.
The R3c package must use corrected successor schema
`gfmcd.r3c.acceptor_head_confirmation/2` and instrument revision
`r3c-v2.1-monotone-health-early-preflight-authorization`. Its ordered seven-path authorization must equal the
bucket-0-health/current-health intersection, L3H1 must be effectively authorized, and the synthetic
calibration must power all seven health-enabled paths. The reviewed v1 failure keeps
`L2H15_mean_component` suppressed even if current target health passes; `/1` is rejected before R5b
path or model access.
The outcome-blind census is now bound as a design expectation: GENCODE-v50 **comprehensive** GTF
SHA-256 `83fba3e9...13a453` (the v50-basic file is forbidden), local hg38 FASTA SHA-256
`5be01555...af5e51`, 6,064 input rows, 1,233 excluded genes, 366 eligible rows/272 eligible genes,
142/130 available genes in buckets 0/1, and the first 128 genes in each bucket. Exact full hashes live
in the shared module and enter the runner fingerprint. At draft status the independent cohort-input and
reconstructed-audit authorization pins remain literal `None`; the expected identities are not
wildcards and cannot themselves authorize a run.

The mechanism-evaluation reservation contains exactly 128 one-row-per-gene records. Order its genes by
`(sha256("r5b-u-a-crossfit-v1|" + gene_id).hexdigest(), gene_id)` and assign alternating ordered genes
to folds 0 and 1, beginning with fold 0. Each fold therefore contains exactly 64 rows/genes and must
contain both strands. A gene occurs
in exactly one fold. For a row
in fold `f`, every basis and every rank-matched learned quantity used on that row is fit only on genes
outside `f`. There is no in-sample causal endpoint. The two direction-specific held-out results are
combined by concatenating their primitive evaluation rows, not by averaging two already-normalized
ratios.

## 4. State definition and projector algebra

For each example capture, at the single masked acceptor token:

- `h+`: final transformer residual on the native clean donor;
- `h-`: final residual after the paired donor `GT -> AC` corruption; and
- `delta_h = h+ - h-`.

For each derivation fold, average `delta_h` within gene and fit one **uncentered** SVD through maximum
rank 32. Candidate ranks are exactly

`K = {1, 2, 4, 8, 16, 32}`.

The basis at rank `k` is the first `k` right singular vectors. `P_{f,k}=U_{f,k}^T U_{f,k}` must be
symmetric, idempotent, and rank `k` within frozen numerical tolerances. The cumulative singular-value
energy curve and its 90% energy rank are diagnostic only. They are serialized under
`diagnostic_energy_*` fields and **never select the scientific rank**.

Every derivation, adjacent-control, whole-gene refit, and maximum-rank runtime fit must have numerical
rank at least 32 under the scale-invariant rule
`sigma_32 / sigma_1 > 1e-10`. Equality or a smaller ratio is rank deficient and stops the instrument.
The minimum required singular-value ratio is serialized. The optimized Gram-SVD refitter uses a direct
SVD fallback in the machine-precision ambiguity zone so squaring the condition number cannot make a
rank-deficient refit appear rank 32.

The compact artifact serializes the ordered 128 row fold IDs and their SHA-256, and for each fold the
SHA-256 of the fitted maximum-rank basis and every rank-specific projector. Array digests bind a
canonical little-endian float64 shape-and-value representation. Residual differences and activation
captures are never serialized. To make the fixed-sequence R5c intervention executable without
refitting or confusing the two reservations, each R5b model stage additionally writes a fresh,
manifest-bound, uncommitted sibling NPZ containing the two rank-32 fold bases, the 32 nested random
bases by fold, the exact 128 row IDs, and their fold IDs. The preflight sibling preserves bucket 0
before bucket 1 is opened. A digest is provenance, not a substitute for causal endpoints. Evaluation
aborts if recomputation changes any fold, basis, projector, row, or sibling-bundle digest.

On each held-out row, use only `P_{f,k}` from the other genes:

- sufficiency cell: `h_S(k) = h- + P_{f,k}(h+ - h-)`;
- necessity cell: `h_N(k) = h+ - P_{f,k}(h+ - h-)`;
- `S_k = m(h_S(k)) - m(h-)`;
- `N_k = m(h+) - m(h_N(k))`.

Necessity and sufficiency remain separate raw row-level endpoints. A minimum, average, product, or
ratio of `S_k` and `N_k` is not a state score.

Repeat the same construction using the matched pseudo-donor corruption to obtain `S_k^P`, `N_k^P`,
and the partner-specific contrasts `S_k-S_k^P`, `N_k-N_k^P`. The pseudo edit never enters basis
fitting. A final-residual patch at a neighboring token would have an architecture-forced no-effect on
the position-wise BERT MLM decoder and is therefore not used as empirical evidence. The failable
positional control instead projects the same rank/energy at the masked acceptor using an SVD basis
derived from the **mean final residual over the exact four-token
adjacent window** in each state. Thus its derivation row is
`mean_j h+_j - mean_j h-_j` over those four fixed positions; it is not a single adaptively chosen token.

## 5. Exact full-MLM readout controls

For captured final residual `h`, the checkpoint call `ad.model.cls(h)` must reproduce the model's
native full-vocabulary logits at the masked acceptor, and `margin_hard(ad.model.cls(h), bid)` must
reproduce the cached native margin. Both maximum absolute errors must be below the literal runner
tolerance `1e-5` (still nonbinding until the joint freeze pins the runner fingerprint).

As a second plumbing check, apply the checkpoint's full MLM transform
`cls.predictions.transform(h)` and then use the exact decoder weights/bias and A/C/G/T token IDs with
`linear_mlm_base_logits`. This must reproduce the four selected logits and the hard-G margin at the
same tolerance. This helper covers only the decoder **after** the transform. It cannot replace the
first control and does not make the full final-residual-to-margin map linear.

An independent full-residual hook control replaces the entire masked-token `h-` with the same-row
captured `h+` before the complete MLM head. It must reproduce the native clean full-vocabulary logits
and hard margin within the same tolerance; replacing `h-` by itself must reproduce the corrupt logits.
This dense clean-residual replay is deliberately independent of the fitted `U_A` basis and validates
the intervention/readout hook even if every candidate rank is scientifically null. It is an
architecture/plumbing capability identity, not evidence for `U_A` and not a completeness result.

Every intervention outcome in Sections 4--8 is recomputed through `ad.model.cls`, including its dense
transform, activation, layer normalization, decoder, and bias. Projector algebra is linear in residual
space; the behavioral readout is not assumed linear.

## 6. Rank- and energy-matched controls

For each held-out fold and draw, construct one deterministic Haar-random rank-32 basis with seed bound
by `(fold, draw, protocol hash)`. Rank `k` uses exactly its first `k` rows. Thus the 32 controls at
each rank are **nested Haar prefixes across the complete rank grid**, rather than six independently
resampled null families. Apply each prefix projector to the same `delta_h`, then
rescale its projected vector **per row** to match the L2 norm of `P_{f,k} delta_h`. The random control
therefore matches rank, target token, intervention direction, and injected/removed energy. Draws with
a zero random projection against a nonzero target are invalid and stop the instrument; they are not
zero-filled or redrawn.

The adjacent-residual basis control is also rank `k` and per-row energy matched. Rank families are
never pooled: rank 1, 2, 4, 8, 16, and 32 are six different intervention sizes.

## 7. Frozen outcome-based rank selection

The R5b selector is exactly `select_crossfit_outcome_rank`. It receives the complete rank grid and,
for each `k`:

1. the raw mean held-out partner sufficiency `S_k` obtained by concatenating all cross-fit evaluation
   rows;
2. a literal Boolean `random_overlap_gate[k]`; and
3. a literal Boolean `random_control_gate[k]`.

It chooses the smallest `k` satisfying all three conditions:

`S_k >= 0.90 * max_{j in K} S_j`,

`random_overlap_gate[k] == true`, and

`random_control_gate[k] == true`.

If `max_j S_j <= 0`, a map is incomplete/non-Boolean, or no rank qualifies, the result is
`NO_U_A_RANK` and no `U_A` claim is licensed. The selected outcome remains valid because every rank
and every selection-bearing contrast is included in the prespecified familywise inference below.

### 7.1 Random-overlap gate, fully specified

For each `k`, compute cross-fold subspace overlap

`O_k = mean squared cosine of principal angles(U_fold0,k, U_fold1,k)`.

The absolute overlap SESOI is frozen as `tau_overlap=0.20`. Refit uncertainty is measured with 4,000
whole-gene bootstrap draws: independently resample the 64 derivation genes within each half, rebuild
the gene-mean `delta_h` matrices, refit the complete uncentered rank-32 SVDs, and recompute all six
`O_k`. A draw with deficient energy/rank is an instrument failure, not discarded. Construct one
simultaneous 95% lower interval across all six ranks from the maximum refit deviation. A point overlap,
singular-value energy curve, or fixed-basis row bootstrap cannot replace this refit interval.

Generate two disjoint deterministic pools, each containing 999 pairs of independent Haar-random
rank-32 bases in the same hidden width. The nuisance-calibration pool uses namespace
`r5b-overlap-haar-calibration-v1`; the test-null pool uses
`r5b-overlap-haar-test-v1`. Within either pool, rank `k` uses the first `k` rows of both maximal bases.
Fit each rank's mean and **sample** standard deviation on the 999 calibration overlaps only, then
freeze those 12 parameters. Use `fixed_calibration_overlap_null` to standardize both the observed
`O_k` values and all 999 independent test-null overlaps with those fixed calibration parameters.
The observed statistic and test-null pool never enter nuisance estimation; the test pool is not
pooled with, recycled into, or leave-one-out standardized against itself.

On each test-null draw take the maximum fixed-calibration standardized overlap across all six ranks.
The complete nested six-rank maximum family uses that independent 999-draw distribution for its
familywise critical value and exact add-one Monte Carlo inference at `alpha=0.025`.
`random_overlap_gate[k]` is true only if (i) the observed standardized `O_k` exceeds the common
familywise `97.5%` max-null critical value, (ii) its exact add-one max-family p-value is `<=0.025`,
and (iii) its simultaneous refit lower bound is strictly above `tau_overlap`. Touching `0.20` is
unresolved. The first two conditions reject chance overlap; the third requires materially stable
absolute overlap and is not inferred from null significance.

The artifact binds both namespaces, all ordered draw seeds, separate pool SHA-256 values, the 12
frozen rank-wise means/SDs, and the test-maximum SHA-256. Calibration and test pool hashes must differ.
Unequal array hashes alone do not establish disjointness. Each draw carries the canonical identity
`(namespace, seed, draw_index)`; the artifact binds the ordered identities and identity-set hashes for
both pools, and the fixed-calibration helper rejects duplicates, reordered reuse, partial overlap, or
any identity intersection across pools.
Neither pool contains model activations, but their full draw arrays need not be committed when these
compact records and deterministic construction reproduce them exactly.

### 7.2 Random causal-control gate, fully specified

For each rank `k`, define one complete gene-clustered causal family containing, in fixed order:

- target `S_partner`, `N_partner`, `S_pseudo`, `N_pseudo`, `S_specific`, `N_specific` (6 rows);
- adjacent-basis `S_partner`, `N_partner` (2 rows);
- 32 random rank/energy controls x `{S_partner,N_partner}` (64 rows); and
- target-minus-each-random-control x `{S_partner,N_partner}` (64 rows).

Thus each rank family has exactly 136 rows. It uses 4,000 whole-gene bootstrap/randomisation draws,
SESOI `0.05`, exact add-one p-values, and `alpha=0.025/6`; this Bonferroni allocation covers the six
different intervention sizes. `random_control_gate[k]` is true only when the target partner `S` and
`N` and both specificity contrasts are `MATERIAL_POSITIVE`, both pseudo endpoints and both adjacent-
basis endpoints are `EQUIVALENT_WITHIN_SESOI`, and **all 64** target-minus-random `S/N` contrasts
have simultaneous lower bounds above zero. Otherwise it is false. No average random baseline, pooled
rank family, unadjusted percentile, or best random draw enters the gate.

Operationally, the Boolean is one intersection-union path: arbitrary-boundary one-sided tests decide
the strict alternative `effect > +0.05`, TOST decides the strict interior alternative
`-0.05 < effect < +0.05`, and zero-boundary one-sided tests decide target-minus-random superiority.
Its p-value is the maximum of all constituent p-values and must pass the rank family's
`alpha=0.025/6`; the complete-family simultaneous intervals must independently satisfy lower bound
`>+0.05` for materiality, lower bound `>-0.05` and upper bound `<+0.05` for equivalence, or lower
bound `>0` for target-minus-random superiority. Equality with or touching any required boundary is
`UNRESOLVED`. The six rank allocations provide the full scanned-rank correction. A zero-null max
p-value cannot stand in for the materiality or TOST constituents.

The shared boundary-test core forms its observed statistic from per-row boundary-centered values and
uses its serialized conservative machine-tolerance tie rule. This prevents 128 mathematical copies of
`0.05` averaging to `0.05000000000000001` and being misclassified as a strict alternative. R5b imports
that rule and its `comparison_tolerance`; it has no divergent local boundary-test implementation.

The overlap family's `0.025` plus the six causal families' combined `0.025` exhaust R5b's stagewise
`alpha=0.05`. R5b receives that alpha only through the strict R5a-primary fixed-sequence gate; it is
not added to R5a's error budget as an independently open family. The target-minus-random boundary is
deliberately **zero**, not the behavioral SESOI: those contrasts ask whether the targeted projector
beats every matched null, while the separate target `S/N` endpoints must clear the `0.05` SESOI.

## 8. Inference, plants, and strict preflight

The exchangeable unit is the gene; raw rows are never independently resampled. Each rank-specific
family is passed once to the production simultaneous classifier. The overlap family and six causal
families together exhaust the selection-bearing scan. Ratios and the diagnostic energy curve are
reported only as descriptive.

Calibration binds three separate objects and always reruns production code.

1. **Causal boundary registry.** At every rank, cycle each material constituent at `+0.05`, each
   TOST constituent separately at `-0.05` and `+0.05`, and each target-minus-random superiority
   constituent at `0`, with every other path conjunct on a coherent easy alternative. Build each case
   from centered raw margin primitive cells, then regenerate pseudo/specific rows, all 32 nested-null
   gaps, boundary tests, intervals, the Boolean gate, and the exact selector. Shifting a derived row,
   Boolean, p-value, or selected rank is forbidden.
2. **Overlap/refit registry.** Generate raw paired 64-gene-by-width `delta_h` matrices by multiplying
   dense orthonormal gene-score matrices by fixed rank-32 bases. The three geometries have squared
   prefix overlap 0, exactly `tau_overlap=0.20`, and 0.80 for **all six prefixes**. Singular values are
   prespecified and separated so every prefix is identifiable under refit. Run both SVD fits, all six
   prefix overlaps, the 4,000-draw whole-gene refit, and the disjoint 999/999 nested-Haar pools through
   the production overlap classifier. No plant is applied directly to `O_k`, a singular vector, its
   interval, or `random_overlap_gate`. The zero and touching-boundary geometries must not pass; every
   prefix of the strong geometry must pass.
3. **Selector registry.** Exercise each rank as the unique smallest eligible rank, curves exactly at
   and immediately on both sides of `S_k=0.90*max_j S_j`, non-positive maxima, incomplete maps, and
   literal non-Boolean gates. This is a deterministic binding test of selection logic, not a power
   simulation.

Every literal causal boundary case executes in the binding suite. There are exactly 76 decision edges
per rank and six ranks, hence 456 rank-by-edge cases; the fixed cycle executes each once before any
repeat. With engineering tolerance `epsilon_cal=0.01`, the runner computes and
pins before evaluation the smallest `N_cal` whose exact one-sided 95% Clopper--Pearson upper bound at
`floor(N_cal*alpha_family)` is at most `alpha_family+epsilon_cal`, and whose lower bound at
`floor(0.95*N_cal)` successes is at least `0.90`. The complete equal-cycle boundary upper bound must be
at most `alpha_family+epsilon_cal`; the strong-alternative lower bound must be at least `0.90`. If the
computed trial count is infeasible, R5b remains DRAFT rather than reducing it.

For the current registry and `alpha_family=0.025/6`, the prospective formula yields `N_cal=456`; this
number is part of the model-free audit and future fingerprint, not chosen from an observed calibration
result. The reported binomial bounds summarize the complete equal-cycle engineering audit rather than
claiming that one observation estimates each individual boundary surface's error rate.

This is an engineering audit for gross implementation inflation, not evidence that scientific FWER is
at most nominal alpha. Formal nominal control remains the unchanged nested-Haar max test and the six
production arbitrary-boundary/IUT families under their assumptions. The exact case registry,
algebra-signature map, raw-matrix and raw-margin generators, formula, `N_cal`, seeds, and hashes are
sealed with all R5 methods before evaluation. Synthetic plants are not model-path evidence; the
candidate-independent dense residual replay in Section 5 establishes hook capability even if `U_A`
is null.

Additional conjunctive preflight gates are:

1. exact bound identities and the R5a/R3c entry chain;
2. exactly 128 one-row-per-gene rows, two 64-gene folds, both strands in each fold, gene-disjoint
   cross-fit fits, and complete cohort/fold hashes;
3. shared-cohort motif hygiene plus exact single-masked-token construction for native, pseudo, target,
   adjacent-basis, and random arms; because every projector intervention occurs after the transformer
   at one final-residual row, non-target-token survival is architecture-fixed and is not mislabeled as
   an empirical MLM-health result;
4. full-MLM residual-to-logit/margin reproduction, decoder-after-transform cross-check, and the
   independent dense `h- <- h+` full-residual replay;
5. projector symmetry/idempotence/rank, replace-component no-op, and exact fold coverage once per row;
6. all rank/energy controls generated as bound nested Haar prefixes without redraw, zero-fill, or
   rank drift;
7. complete disjoint 999-draw Haar calibration and 999-draw Haar test-null pools, plus
   32-draw-per-rank causal controls;
8. the complete least-favourable boundary/refit/selector suite, engineering calibration gates, and
   every endpoint-specific arbitrary-boundary/simultaneous-CI gate; and
9. no opened evaluation outcomes before a clean preflight and all sealed R5a--R5c method/hash pins;
10. native paired-donor acceptor behavior and pseudo behavior are measured under the frozen
    material/equivalence rules as model-continuity outcomes.

Items 1--9 are instrument gates; failure is `INSTRUMENT_UNRESOLVED`. With those healthy, a resolved
failure of item 10 is `ENTRY_BEHAVIOR_NOT_REPLICATED` or `ENTRY_BEHAVIOR_NONSPECIFIC`, a valid scoped
stop. A partial rank curve is diagnostic only and cannot be used to select a rank.

## 9. Decision and licensed wording

`ACCEPTOR_SIDE_STATE_SUPPORTED` requires:

1. a non-null selected rank from Section 7;
2. both full-MLM reproduction controls and every instrument gate;
3. at the selected rank, material-positive held-out partner `S` and `N`;
4. equivalent pseudo `S` and `N`, material-positive partner-minus-pseudo `S` and `N`;
5. the familywise random-overlap and causal-control gates; and
6. equivalent adjacent-basis `S` and `N`.

If no rank passes every conjunct, the confirmatory verdict is the single prespecified scientific stop
`NO_U_A_RANK`. The artifact retains the complete sufficiency, necessity, pseudo, specificity,
adjacent-basis, random-control, and overlap decisions at every rank, so a paper may accurately
describe which requirements were resolved or unresolved. Those component patterns are not promoted
post hoc to competing categorical verdicts. Boundary-touching intervals remain unresolved within
that stop. `INSTRUMENT_UNRESOLVED` is reserved for failed plumbing, health, invariants, hash binding,
degenerate refits, or classifier capability. A resolved sub-SESOI component bounds only this
final-residual projector intervention.

The strongest possible prose is: “Two gene-disjoint, cross-fitted rank-k projectors with materially
stable overlap are sufficient and necessary for a material portion of the paired-donor-dependent
acceptor-margin component under matched rank/energy controls.” The notation `U_A` denotes this
fold-indexed projector family, not a proven unique population basis. It may not say that `L3H1`
writes `U_A` (R5c), that `U_A` is example-specific, what it represents biologically, or that the
model performs a cellular splicing mechanism.

## 10. Required artifacts before freeze

The implemented compact artifact stores primitive margins, residual/projector diagnostics, ordered fold
IDs and their hash, basis/projector digests, fold and gene memberships, every rank/control family in
exact order, selection maps and literal Boolean gates,
both overlap-pool identities/hashes, frozen rank-wise calibration means/SDs, test-max digest,
compact gene sufficient statistics, exact hashes, command, environment, and immutable manifests.
Full activation captures are not written. Maximum-rank and selected-rank bases exist only in fresh
uncommitted sibling bundles so R5c can execute the frozen intervention. Every such bundle is an
explicit manifest output with byte size, file SHA-256, and shape-aware per-array digests; it is not
scientific evidence by itself.

If and only if the bucket-1 target supports a selected rank, R5b writes **two distinct selected-rank
runtime bundles**, one for bucket 0 and one for bucket 1. Each contains exactly
`selected_rank` (redundant scalar), `selected_bases_by_fold[2,k,D]`,
`random_bases_by_draw_fold[32,2,k,D]`, `row_fold_ids[128]`, and
`ordered_row_ids[128]`. The compact `r5c_runtime_contract_by_bucket` authenticates each bundle's
reservation/construction/gene hashes, checkpoint, selected rank, cross-fit map, file bytes, and the
four execution-array digests. Reusing one 128-row contract for both buckets is forbidden. The
top-level `r5_entry_contract` separately states whether all instrument gates passed and the target is
claim-eligible; R5c remains closed unless it names `ACCEPTOR_SIDE_STATE_SUPPORTED`, acceptor-G-only,
the bucket-1 cohort and construction, the selected rank, and that bucket's exact cross-fit map.

The target verifier first authenticates the preflight artifact by a literal SHA-256 and its sibling
manifest, then replays every observed causal family and the rank selector from the one-row-per-gene
sufficient statistics before it can open bucket 1. Both model stages require a clean Git identity,
source/checkpoint snapshots that remain unchanged through execution, and fresh output paths.
The execution manifest enumerates exact bytes for every file in the joint Python closure and all
three protocols, while the prospective stack fingerprint normalizes only literal freeze-pin RHS
values.
Before model loading, the target also compares that preflight with the current local and joint method
bindings, exact R3c→R5a artifact/manifest chain, authenticated exclusion registry, comprehensive
source identity, full two-bucket census and bucket-0 reservation, checkpoint bundle, device/package
runtime identity, acceptor-only endpoint contract, and every production constant. Any difference is
`INSTRUMENT_UNRESOLVED`; a byte-valid preflight from a different method or runtime cannot open bucket 1.
