# Head–Context Atlas mechanism-validation amendment A2 — conditional mechanism stage

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Status: **FROZEN BEFORE THE FULL DISCOVERY MATRIX, WHOLE-VALIDATION OUTCOMES, OR ANY STAGE-TWO
MECHANISM OUTCOME WAS READ** (2026-07-19).

This amendment closes the remaining QK/OV/path ambiguities in `MECHANISM_VALIDATION_PLAN_V1.md`. It is
conditional: it applies only to relations that pass the untouched whole-effect stage under Amendment A1.

## A2.1 Stage-two package and gatekeeping

After whole-effect analysis and its independent audit, a deterministic builder creates one new package
containing only `validated_candidate_ids`. It may read the audited whole-effect classifications and the
already-frozen discovery package; it may not read any QK/V/path output. The package freezes candidate
endpoints, controls, eligible downstream MLP layers, upstream attention groups, representative packets,
and the stage-two family before another model forward.

No whole-effect survivor means no stage-two run. A failed or inconclusive whole effect cannot be rescued
by an attractive QK/V plot, group effect, path recovery, or DAG.

## A2.2 B attention candidates — population QK versus V/OV

`B` has a legitimate same-packet source: the identical masked packet with its annotated partner motif
destroyed. Patch every active (non-padding) model-token coordinate, including required special tokens;
report this explicitly. This preserves the exact QKV→whole-head bridge instead of silently leaving a
special-token route unpatched. Within that coordinate set:

- `QK`: replace the candidate head's query and key projections with destroyed-partner source states;
- `V/OV`: replace its value projection (the fixed output matrix then maps this through OV);
- `QKV`: replace all three projections as a plumbing bridge to the exact whole-head output scrub.

Run the same grid for the frozen importance-matched head control. Native→native identities must be
≤`1e-4`, QKV→whole target and health logits ≤`1e-4`, and finite/coordinate/health gates must pass. QK and
V/OV are separately interpretable main effects. Their algebraic interaction is not a binding test and is
never reported as one (R23). QKV is a bridge, not a mechanism endpoint.

Per untouched whole endpoint, issue a label only from simultaneous intervals:

- `partner_sensitive_routing`: candidate QK clears the material floor in the validated whole-effect
  direction and V/OV is familywise bounded inside ±0.05;
- `partner_sensitive_value_write`: V/OV clears the floor and QK is bounded;
- `mixed_partner_sensitive`: both clear the floor in the validated direction;
- otherwise `mechanism_unresolved`.

The importance control remains visible; it does not redefine candidate QK/V post hoc.

## A2.3 A attention candidates — no population router/writer label

Zeroing a head output has no natural QK or V source. A matched control packet is a different genomic
sequence, so replacing all-token projections from it is a large cross-sequence state transplant—not a
decomposition of A. For `A` candidates:

- do not run population QK/V inference and do not assign routing/writing labels;
- the already-frozen representative discovery case/control pairs may receive separately labeled
  `matched_control_state_interchange` QK and V maps for qualitative localization and DAG annotation;
- those representative values are not effect fractions, do not enter a p-value family, and cannot
  validate a mechanism.

This narrows the original plan's A-interchange language rather than letting a convenient but
non-identifiable population contrast masquerade as decomposition.

## A2.4 Downstream MLP recovery for validated attention candidates

Avoid selecting one favorable MLP. Enumerate every MLP residual write from the candidate's attention
layer through L5. For each layer `j`:

1. apply the candidate whole intervention (A zero or B destroyed-source scrub);
2. in that same forward, restore MLP `Lj`'s native residual write;
3. define signed recovery as `margin(candidate+restore_j) − margin(candidate_whole)`;
4. repeat under the frozen importance-control head intervention;
5. report candidate recovery, control recovery, and their paired difference.

This is a controlled recovery assay, not an additive mediation percentage. A path label requires a
simultaneous candidate-minus-control recovery interval that clears +0.05 nats in at least one
prespecified downstream layer and a validated, non-negligible whole-effect denominator.

## A2.5 Validated MLP candidates and upstream attention dependence

For an MLP candidate, the upstream attention group is the exact set of validated attention candidates in
the same `{general or contrast} × estimand` group and strictly no later than the MLP layer. Apply that
head group jointly (zero for A, destroyed-source scrub for B), then restore the candidate MLP's native
write. Compare recovery with restoration of its frozen importance-matched MLP control under the same
head intervention.

If this prespecified upstream set is empty, report `upstream_path_instrument_limited`; do not import a
head by rank after validation. Attention and MLP effects are overlapping causal cuts and are never added.

## A2.6 Within-endpoint simultaneous family

For each validated whole endpoint, the prespecified mechanism vector contains all applicable candidate
and control QK/V main effects, exact-group effect, enumerated downstream recovery contrasts, and MLP
upstream-path contrasts. QKV identity bridges and representative A state interchanges are gates or
descriptive diagnostics, not family members.

Use the same population estimator as the whole endpoint (one-way gene clustered for general-role
endpoints; inverse-reuse weighted unique-dyad two-endpoint sandwich for context endpoints). Construct the
joint covariance over the mechanism vector, project only numerical negative eigenvalues to zero, and use
a deterministic 200,000-draw Gaussian max-|t| critical value (`seed=0`) for simultaneous intervals at
FWER `alpha=0.05`. Preserve marginal intervals and achieved familywise MDEs.

No categorical path/routing/writing label is issued when the relevant simultaneous interval is
inconclusive or the familywise MDE exceeds the 0.05-nat material floor.

## A2.7 Evidence-graded representative DAGs and export

Only after stage-two analysis and independent audit, generate DAGs for the discovery representatives
already frozen before validation access. Keep model depth on one axis and source nucleotide positions at
the input boundary. Edge width encodes signed contribution magnitude; thresholding may hide small edges
but must report the hidden remainder mass.

Every edge carries one of: `screening`, `exact_component`, `QK`, `V_OV`, `MLP_path`,
`negative_control`, or `remainder`. Highlighting requires the corresponding audited population result;
activation×Jacobian screening edges remain visibly provisional. Export to `gfm-visualizer` only the
validated relation, simultaneous intervals, frozen representatives, exact interventions, evidence labels,
and artifact hashes. Never export a discovery-only head-function label.
