# FROZEN R5b v2 — pre-unseal local-writer derivation and optional basis freeze

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

> **Status: FROZEN FOR PREFLIGHT ONLY (2026-08-17); ALL SCORED STAGES HARD-BLOCKED.**
> The v2 runner is
> `scripts/splicebert_acceptor_coupling/run_r5b_acceptor_side_state.py`.
> Its reviewed protocol, normalized instrument, cohort, checkpoint, historical-R3, and joint-stack
> identities are pinned in source. The output-derived R5a-preflight, derivation, validation, and
> outbound-package pins remain literal `None`. Consequently
> `--stage derivation`, `--stage validation`, and `--stage freeze` stop before
> resolving any caller-supplied artifact, cohort, genome, checkpoint, model, or
> output path until their exact predecessor packages are reviewed and pinned. Only `--stage audit`
> is currently authorized, and that stage loads
> neither model nor genome and opens no chr8/chr9 data.

## 1. Question and estimand

R5b develops one pre-unseal writer instrument:

> Does the output-projected contribution of L3H1 at the masked-acceptor query
> carry a causally useful local state before the proposed L4/L5 delivery route?

The exact writer site is

`L3H1.output_projected@acceptor_query`.

It is L3H1's additive, bias-free contribution through
`layer[3].attention.output.dense` at the masked acceptor-G query. It is after
`W_O` and before L4/L5. It is not a pre-`W_O` head context, the full L3 attention
block, the incoming or outgoing whole residual, or the final masked-acceptor
residual.

Dense necessity and sufficiency at this site are the basis-independent primary
R5b result. An optional low-rank `U_A` is path-local compression evidence only.
No low-rank outcome may veto, weaken, redefine, or become a prerequisite for the
dense result.

The retired final-residual `U_A` proposal is not an estimand, intervention site,
runtime array, or decision variable in R5b v2.

## 2. Pretarget entry and fixed sequence

R5b is upstream of every sealed biological target. It consumes no R1c target,
R5a target, R5c target, or chr8/chr9 outcome.

Before either residual bucket can be reserved, the runner authenticates:

1. the exact R3d Path-A artifact and sibling manifest, with Path B forbidden as
   a gate;
2. the R3c package only to reconstruct the prior-exposure exclusion registry;
3. the exact R5a **method-preflight** artifact and sibling manifest under
   `gfmcd.r5a.acceptor_address_binding/2`; and
4. the common checkpoint and joint R5-stack identities.

The R5a record must have `stage=preflight`, `status=PREFLIGHT_PASS`,
`scientific_entry_authorized=false`,
`r5a_opened_chr8_chr9=false`, and `observed_scientific_inference=null`. Its only
behavioral endpoint is `masked_acceptor_G_margin_only`; both donor-readout flags
are false. Its nested `r5_entry_contract` must have `claim_eligible=false`. Its
entry order is exactly:

1. `authenticated_R3d_Path_A_prior`;
2. `authenticated_R1c_validation_preflight_nonentry`;
3. `R5a_instrument_preflight`.

The legacy-named `target_gene_ids_sha256` field in that upstream contract binds
the sorted-unique genes of the R1c validation-preflight handoff. It is not an R5a
or R5b target cohort and cannot authorize scientific entry. L2H15 is not an R5b
entry gate.

The R5b sequence is:

1. bucket-0 paired no-op health and primary dense family;
2. optional bucket-0 cross-fit rank scan and candidate selection;
3. exact completed bucket-0 package authentication;
4. bucket-1 paired no-op health and primary dense family;
5. optional bucket-1 evaluation of exactly one frozen bucket-0 candidate;
6. one mechanical union refit, only after successful validation; and
7. one terminal three-way basis disposition.

Scientific failures are retained. Instrument anomalies stop interpretation.
The dense result never depends on steps 2, 5, or 6.

## 3. Residual-validation cohorts and irreversible method receipts

R5b uses only the two shared outcome-blind residual-validation reservations on
chr11/chr16/chr19. It does not construct or open chr8/chr9 rows.

- bucket 0: 128 records, exactly one row per independent gene, candidate
  derivation;
- bucket 1: 128 different records, exactly one row per independent gene,
  frozen-candidate validation;
- bucket 0 union bucket 1: one rank-fixed refit after validation is locked.

Both buckets preserve both strands and exclude the authenticated union of
historical R3, exact R3c cohorts, the R3d direct-exposure registry, and the R3d
target. This is prior-exposure registry disjointness, not a claim of global gene
freshness.

R5b has no biological-target CLI stage and does not call the shared biological
target ledger. It owns a separate create-only method-attempt lifecycle:

- derivation: create a bucket-0 receipt after upstream authentication and before
  cohort, model, or output inspection;
- validation: authenticate the exact completed bucket-0 package, then create a
  bucket-1 receipt before cohort, model, or output inspection;
- completion: only after the immutable result and clean sibling manifest exist,
  bind their exact byte identities and official status in a create-only
  completion record.

An existing or partial receipt spends that bucket. Failure, code repackaging, or
method revision does not reset it. Freeze authenticates the exact derivation and
validation artifact, manifest, and completion pins before building a disposition.

After dense scoring succeeds, a prespecified optional-basis failure does not
discard that dense primary record. Insufficient numerical rank, a projector-rank
mismatch, or failure of a frozen rank/energy-matched null invariant is recorded
as `gfmcd.r5b.optional_basis_instrument_failure/1`; the stage publishes and
completes with `BASIS_INSTRUMENT_UNRESOLVED`. A validation stage reached after an
unresolved derivation scores and publishes its basis-independent dense family but
does not fit, apply, or refit a basis. All other scorer, hook, checkpoint,
authentication, shape, or raw-replay failures abort and leave the receipt spent.

## 4. Exact post-`W_O` writer and internal scorer

Let `c_h[b,q,D_h]` be L3H1's context at the acceptor query and let
`W_O[:,h]` be L3H1's column block in the L3 attention output projection. The
writer is

`w[b,q,D] = c_h[b,q,D_h] @ W_O[:,h].T`.

The shared dense bias is excluded because it cannot be assigned to one head.
The runner authenticates the `W_O` block plus clean, paired-corrupt, and
pseudo-corrupt head contexts and output-projected contributions. Pre-`W_O`
contexts are serialized only so the post-`W_O` contribution can be replayed;
they are never treated as the intervention variable.

The provider also serializes `gfmcd.r5b.post_wo_numeric_provenance/1`: the exact
checkpoint operator dtype for the head context, `W_O` block, and contribution;
the `torch.nn.functional.linear` replay operator; relative tolerance zero; and
one frozen absolute tolerance of `1e-5`. Replay casts JSON values back to that
checkpoint dtype before applying the operator. The same tolerance is used at
capture and offline replay. Stored clean-minus-corrupt and clean-minus-pseudo
deltas must additionally equal their serialized contribution differences in the
checkpoint dtype, so the operator tolerance does not weaken delta tamper checks.

The runner itself loads the authenticated checkpoint, captures L3 head outputs,
projects L3H1 through the shared `project_head_context_to_residual` primitive,
and applies residual-width deltas through the shared
`add_attention_output_delta` primitive. There is no provider-artifact CLI and no
public path that converts a self-attested provider artifact into a scientific
result.

For a clean-to-corrupt replacement at one query, the hook adds

`delta_w = w_clean - w_corrupt`

to the actual output of L3 `attention.output.dense`. Every other token, every
other head contribution, the shared bias, the incoming residual, all later
blocks, and the complete MLM head remain native. The dense projection hook must
execute exactly once per forward. All behavioral outcomes are read only after
the remaining checkpoint has executed.

Every scored artifact carries and replays:

- the exact residual-width `W_O` contribution arrays and their SHA-256s;
- the fixed normalized G-versus-mean(A,C,T) decoder control axis;
- all raw four-cell margin primitives;
- exact genes and row order;
- the dense and reached nested decision records; and
- `direct_model_scoring=true` and
  `external_provider_artifact_accepted=false`.

The source, checkpoint, upstream packages, protocol, instrument, and joint stack
are rehashed after scoring and before publication.

## 5. Basis-independent primary dense family

For the paired-donor leg, let `m+` be the native clean masked-acceptor-G margin,
`m-` the native paired-donor-corrupt margin, `m(-;w+)` the corrupt forward after
adding the clean-minus-corrupt writer at the acceptor query, and `m(+;w-)` the
clean forward after subtracting it. The row endpoints are

- `S_writer = m(-;w+) - m-`;
- `N_writer = m+ - m(+;w-)`.

The same four-cell construction is run for the matched pseudo-donor edit.
Partner specificity is paired minus pseudo, separately for `S` and `N`. Every
primitive margin cell is serialized and replayed through the shared dense
estimand constructor.

Primary dense support requires both the ordinary mean and the 10%-trimmed mean:

- paired `S` and `N`: simultaneous lower bounds strictly above `+0.05` logits;
- pseudo `S` and `N`: simultaneous intervals strictly inside
  `[-0.05,+0.05]`; and
- paired-minus-pseudo `S` and `N`: simultaneous lower bounds strictly above
  `+0.05`.

Necessity and sufficiency remain separate. They are never multiplied, averaged,
divided, or converted to a recovery ratio.

### 5.1 Query placement

The same row-specific acceptor-query writer difference is added at the four
prespecified acceptor offsets `{-2,-1,+1,+2}`. The corresponding clean arm
subtracts the same difference at that offset. All offset-by-`{S,N}` endpoints
must be equivalent inside `[-0.05,+0.05]` under both estimators. This tests local
placement; it is not a final-residual control.

### 5.2 Content diagnostics

Two directions are always reported but do not gate dense existence:

- the unit bucket-wide mean paired writer difference; and
- the fixed normalized checkpoint G-versus-mean(A,C,T) decoder axis.

Large effects on either direction remain visible and prohibit stronger
example-specificity or nontrivial-content wording.

## 6. Paired no-op health

The exact post-`W_O` hook is exercised with a zero delta on clean and
paired-corrupt forwards. For each no-op margin difference, the ordinary and
10%-trimmed estimators must have simultaneous lower bounds strictly above
`-0.01` logits. Health is evaluated before scientific interpretation.

Failure is `INSTRUMENT_UNRESOLVED_NOOP_HEALTH`; it is not evidence against the
writer hypothesis.

## 7. Optional nested `U_A`

`U_A` is a low-rank component of the upstream L3H1 writer difference
`w_clean - w_paired_corrupt`. It is never a final-residual state.

Ranks are fixed at `{1,2,4,8,16,32}`. SVDs are uncentered. A required singular
direction must satisfy the strict scale-relative rule
`sigma_k / sigma_1 > 1e-10`; equality or a smaller value is an instrument
failure.

### 7.1 Bucket-0 derivation

Genes are ordered by

`(sha256("r5b-v2-local-writer-crossfit|" + gene_id), gene_id)`

and alternated into two exact 64-gene folds. Each row is scored only with a basis
fit on the opposite fold. A whole-bucket rank-32 fit is produced in parallel but
cannot score bucket-0 outcomes; after selection it is the only candidate basis
that may be applied to bucket 1.

At every rank the runner preserves paired, pseudo, specificity, adjacent,
gene-generic, and readout-axis primitives. It creates 32 deterministic Haar null
bases of the same rank. Every null projection is rescaled row by row to exactly
match the L2 energy of the candidate projection. No redraw, zero fill, rank
pooling, or averaged null comparator is permitted. Projection arrays and target,
pseudo, and random basis identities are authenticated.

The smallest rank is selected only when:

1. ordinary-mean paired `S` is at least 90% of its maximum over ranks;
2. trimmed-mean paired `S` is at least 90% of its own maximum;
3. the complete paired/pseudo/specific/placement family passes;
4. the two cross-fit bases have mean-squared principal-cosine overlap above
   `0.20`; and
5. candidate-minus-each-random `S` and `N` pass for all 32 draws.

No qualifying rank yields no optional candidate and leaves dense primary
evidence unchanged.

### 7.2 Bucket-1 validation

Bucket 1 evaluates exactly one selected rank and the exact whole-bucket-0 basis.
It may not scan another rank, refit the applied basis, or use a bucket-1 basis
for intervention. The 32 deterministic rank/energy null controls are regenerated
from the frozen namespace and authenticated by exact IDs.

Only for stability, an outcome-independent bucket-1 SVD is fit at the already
selected rank. Whole-gene bootstrap refits compare it with the fixed bucket-0
basis using mean-squared principal cosine. The 95% lower bound must exceed
`0.20`. That diagnostic basis is never applied to a margin endpoint.

### 7.3 One union refit

Only after validation status is
`NESTED_CANDIDATE_VALIDATED_FOR_REFIT` may the runner concatenate bucket 0 and
bucket 1 in authenticated row order and perform one SVD at the already selected
rank. There is no rank scan or outcome test at this step.

If derivation or validation has `BASIS_INSTRUMENT_UNRESOLVED`, freeze emits the
same terminal disposition without attempting this union refit.

The union refit freezes the basis, 32 deterministic rank-matched random bases,
the normalized gene-generic axis, and the normalized readout axis. The entire
refit is recomputed by the offline disposition validator. R5c may load these
arrays only for a validated disposition. R5c refitting, rank selection, basis
mutation, or control regeneration is forbidden.

## 8. Inference and multiplicity

The independent unit is the gene, with exactly one row per gene. One shared
whole-gene bootstrap resample is used across every endpoint in a reached family.
Every gate must pass both:

1. the ordinary mean; and
2. the 10%-trimmed mean.

Inference is studentized cluster-bootstrap inversion, not an exact permutation
or sign-flip test. Maximum signed bootstrap-`t` statistics provide add-one Monte
Carlo adjusted p-values across all decision edges; maximum absolute
bootstrap-`t` statistics provide simultaneous intervals. Strict boundary
inequalities are required and `exact_test_claimed=false` is recorded.

Production scored stages use exactly 10,000 inference draws and 10,000 bucket-1
overlap draws. The dense family uses `alpha=0.05`. Bucket-0 nested rank families
use `0.05/6` per rank because rank is scanned; bucket 1 evaluates only the fixed
candidate and uses `0.05`.

The model-free audit may use fewer draws solely to exercise algebra, registry,
and classifier branches. It licenses no scientific inference.

## 9. Instrument and publication gates

Before a scored stage is published, every reached condition must hold:

1. literal protocol, instrument, cohort, checkpoint, upstream-package, and joint
   stack pins;
2. exact R3d and R5a artifact **and manifest** authentication;
3. exact cohort-input and reconstructed cohort-audit identities;
4. exact checkpoint, GTF, FASTA, FASTA index, dependency locks, and transitive
   local Python closure;
5. strict MPS runtime policy for derivation and validation;
6. exactly 128 one-row-per-gene records in the correct residual bucket;
7. one masked acceptor-G endpoint and no donor readout;
8. exact post-`W_O` L3H1 projection, shared-bias exclusion, and hook call count;
9. paired no-op health;
10. raw-cell replay, complete registries, and both robust estimators;
11. rank, projector, energy, fold, row-order, and random-ID invariants when the
    optional path is reached, or an exact prespecified optional-basis failure
    record with dense primary explicitly unaffected;
12. exact production draw counts; and
13. post-scoring input rehashing.

Scored artifacts and sibling manifests are create-only. Manifests must record a
clean worktree and bind exact artifact bytes. Runtime NPZ files are execution
inputs, not evidence, and must not be committed.

## 10. Terminal disposition and R5c handoff

After exact derivation and validation packages authenticate, R5b emits exactly
one `gfmcd.r5b.local_writer_basis_disposition/2` status:

- `VALIDATED_BASIS`;
- `NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS`; or
- `BASIS_INSTRUMENT_UNRESOLVED`.

Every disposition has `claim_eligible=false`, `residual_validation_only=true`,
`sealed_chr8_chr9_opened=false`,
`masked_donor_endpoint_constructed=false`, `dense_primary_unaffected=true`, and
`optional_c2_only=true`.

The exact path-local C2 consequences are:

- validated: `EXECUTE_OPTIONAL_C2_WITH_AUTHENTICATED_FROZEN_LOCAL_WRITER_BASIS`;
- no validated basis:
  `SKIP_OPTIONAL_C2_NO_VALIDATED_BASIS_DENSE_PRIMARY_UNAFFECTED`;
- instrument unresolved:
  `SKIP_OPTIONAL_C2_BASIS_INSTRUMENT_UNRESOLVED_DENSE_PRIMARY_UNAFFECTED`.

Only `VALIDATED_BASIS` may carry an NPZ and
`gfmcd.r5b.frozen_local_writer_basis/2` runtime contract. That contract binds:

- exact basis and 32 random-basis IDs;
- every runtime array's name, shape, dtype, and SHA-256;
- selected rank and residual width;
- NPZ name, size, and SHA-256;
- the post-`W_O` writer contract;
- checkpoint, cohort audit, and joint-stack fingerprint;
- exact R5a-preflight, derivation, and validation artifact **and manifest**
  identities; and
- target-time refit, rank selection, and mutation prohibitions.

For either nonvalidated status, contract and bundle are null and no NPZ path is
accepted. The dense C1 path remains available regardless of optional C2 status.

The public offline validator requires the exact disposition artifact/manifest
and exact linked derivation/validation artifact/manifest pairs. It returns the
authenticated status, exact path-local C2 disposition, optional runtime
contract, and `dense_primary_unaffected=true`.

## 11. Licensed language

All R5b records are method-development evidence. The strongest R5b-only wording
is:

> “On the prespecified residual-validation cohorts, interventions on L3H1's
> post-`W_O` acceptor-query contribution exhibited the recorded dense writer
> pattern. The optional rank-k compression candidate was [validated / not
> validated / instrument-unresolved] and was frozen only as a prespecified input
> to R5c.”

R5b alone may not say that the biological mechanism is established, that L4/L5
delivers the state, that `U_A` is necessary for the dense effect, that the state
is example-specific unless both content diagnostics permit it, what the state
means biologically, or that the model implements cellular splicing. Those
promotions require the separately sealed R5c capstone and its full controls.
