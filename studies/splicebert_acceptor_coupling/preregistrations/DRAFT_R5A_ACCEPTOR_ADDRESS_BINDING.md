# DRAFT R5a v3 — masked-acceptor address binding on the exact R1c rows

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

> **Status: DRAFT; zero genome or model scoring authorized.** The literal method,
> checkpoint, R1c-package, R5a-preflight, and target-receipt pins are incomplete. This
> document does not unseal chr8/chr9 or authorize a target attempt.

## 1. Question and evidence boundary

R5a asks whether, while SpliceBERT predicts the single masked acceptor G, L3H1 uses a
position-sensitive address to read the paired upstream donor window, and whether its
address and donor-local values jointly change the native masked-acceptor G margin.

Every decision endpoint is a change in the raw masked-acceptor G logit margin. The
runner constructs no masked-donor endpoint. Pre-softmax QK logits and post-softmax
attention probabilities remain signed diagnostics in different units; neither is
tested against the behavior SESOI or substituted for behavior.

A target pass supports a scoped L3H1 paired-specific reader under these exact
interventions. It does not establish a stable representation, payload semantics,
write site, delivery path, mediation, sufficiency, faithfulness, minimality, or a
biological mechanism.

## 2. Prospective order and the two R1c handoffs

The order is fixed before any R5a target data are opened:

1. Authenticate the exact R3d target package and replay Path A as `SUPPORTED`. Path A
   is prior component evidence. R3d Path B never gates R5a. The runner performs this
   R3d/R3c authentication before opening either R1c package and reuses the resulting
   authenticated exposure chain rather than loading it again during chain assembly.
2. Authenticate the exact two-file R1c **validation-preflight** package. It supplies
   exactly 400 ordered chr11/chr16/chr19 rows and the common `P_clean`, `P_under_D`,
   and `P_under_N` legs for same-row instrument replay.
3. Run the R5a validation preflight and freeze its artifact and manifest. This stage
   validates hooks, surgeries, health, calibration, row order, and common-leg replay.
   It performs no observed scientific classification, has
   `scientific_entry_authorized=false`, and cannot license the paper headline.
4. Separately unseal and complete the prospective R1c target on chr8/chr9.
5. Authenticate the public `gfmcd.r1c.verified_target_handoff/1` attestation and its
   exact artifact, manifest, attempt, completion, and predecessor-preflight files. A
   confirmed R1c behavior verdict opens R5a target entry. A prespecified valid R1c
   behavior null is a **scientific stop** before an R5a receipt or intervention;
   `INCONCLUSIVE` is an instrument failure.
6. Authenticate the frozen R5a preflight, reserve the R5a biological target, then and
   only then open the output path, GTF/FASTA, chr8/chr9 rows, checkpoint, or model.
7. After the receipt, load R1c's exact target common legs and replay them on the same
   400 rows. R1c recorded them with `eager=False` (SDPA); R5a requires `eager=True` for
   hooks. The frozen maximum absolute per-leg tolerance is 1e-4. Any larger SDPA/eager
   mismatch is `INSTRUMENT_UNRESOLVED`, never a scientific null.
8. Run the reached R5a core gates in order: G1, G2, then G3. Run the L2H15 secondary
   family only after a supported core and only if its own prospective health and
   surgery gates pass.

The exact R3c preflight package remains provenance only and has
`scientific_entry_authorized=false`. Neither R3c nor historical R3c health can rescue
R1c, gate R5a entry, or support the R5a headline.

## 3. Exact same-row cohorts

R5a never constructs a new scientific cohort.

- Preflight materializes all 400 R1c validation rows in authenticated order.
- Target materializes all 400 R1c sealed target rows in authenticated order.
- No row is selected, dropped, substituted, or reordered by an R5a outcome.
- Repeated genes are allowed; inference first aggregates rows within gene and requires
  at least 30 independent genes.
- The target handoff binds cohort, ordered-row, ordered-gene, sorted-unique-gene, and
  construction digests.
- Paired and distance-matched pseudo donor windows are the scientific source windows.
  No adjacent window is a scientific arm or a materialization requirement.

Both donor edits use the same `GT -> AC` operator. The paired edit acts at the
annotated donor; the pseudo edit acts at R1c's outcome-blind matched pseudo donor. Each
source window is the exact four tokens at offsets `[-1, 0, +1, +2]`.

## 4. Components, query, controls, and units

The primary reader is L3H1. L0H2 is the size-matched negative head. L2H15 is a
separately prespecified secondary/upstream-reader candidate; it is never a replacement
primary and never gates the L3H1 headline.

Every surgery fixes the acceptor query token and operates only on the specified
position/word component, four QK logits, complete query attention row, or four-token V
window. Whole-run hidden-state replacement is not a decision intervention.

Behavior effects use masked-acceptor G margin-logit units with SESOI 0.05. QK
preferences use pre-softmax logit-difference units. Attention preferences use
post-softmax probability-difference units. QK and attention receive no behavior SESOI,
equivalence decision, or zero-null gate; report their mean, prespecified 10%-trimmed
mean, and positive fraction as diagnostics only.

## 5. G1 — query-fixed position × word surgery with fixed V

For L3H1 and L0H2, exchange the paired and pseudo four-token windows before embedding
LayerNorm in a 2×2 factorial:

| Cell | Word component | Position component |
|---|---|---|
| full | native | native |
| word_shift | paired↔pseudo | native |
| position_shift | native | paired↔pseudo |
| joint_shift | paired↔pseudo | paired↔pseudo |

Swaps must be disjoint, exact, outside-preserving, and involutive. Capture the clean
numerical acceptor-query Q vector for each tested head before any swap. Every 2×2 cell
uses that byte-identical clean Q with the cell's shifted K tensor and one identical
clean V tensor. Inject the resulting query-head context into the clean recipient
through the real attention output projection and full MLM decoder.

The contextual Q that the shifted cell would natively produce is recorded only as a
descriptive change and never enters a decision row. Capability records the clean-query
digest for every head and cell and requires zero fixed-query assignment error. The
full/native cell must also reconstruct its captured native attention row and margin
within the frozen hook tolerances. Thus G1 licenses a position-sensitive source-window
reader, not a compound intervention in which donor-window shifts also change Q.

G1 requires both position simple effects to be material-positive for L3H1, equivalent
for L0H2, and material-positive in the L3H1 target-minus-control contrasts. Word simple
effects and the interaction are replayed but do not define the position result. QK and
attention versions of these quantities are diagnostic bridges only.

## 6. G2 — exact QK surgery with fixed clean V

Capture the complete clean query-head logit row. Replace exactly four logits, indexed
by the paired or pseudo window, with the corresponding logits from the matched donor-
corrupt run. Every unselected logit remains bit-identical. Softmax the edited row,
combine it with the same clean V tensor in every leg, inject the query-head context,
and measure only the masked-acceptor margin.

Primary frozen arms are L3H1×paired, L3H1×pseudo, and L0H2×paired. G2 requires
material-positive L3H1 paired and paired-specific effects, equivalent L3H1 pseudo and
L0H2 controls, and material-positive target-minus-control contrasts against both.

The secondary frozen arms are L2H15×paired, L2H15×pseudo, and L0H2×paired. They use the
same exact-four-logit/fixed-clean-V instrument and inference rules. A supported result
is reported as an additional upstream reader role. A null, unresolved result, or its
own health failure leaves a supported L3H1 core unchanged.

## 7. G3 — fixed-V attention × donor-local value reconstruction

For each paired and pseudo corruption leg, reconstruct four L3H1 acceptor-query cells:

| Cell | Attention row A | Value tensor V |
|---|---|---|
| baseline | corrupt | corrupt |
| address_only | clean | corrupt |
| value_only | corrupt | clean only in the selected four-token window |
| joint | clean | clean only in the selected four-token window |

Values outside the selected window remain recipient values. Inject every cell through
the real L3H1 query-head path and full MLM decoder. For `joint_lift`,
`address_effect_at_value_plus`, and `value_effect_at_address_plus`, G3 requires
material-positive paired and paired-specific effects, equivalent pseudo controls, and
material-positive paired-minus-pseudo target-minus-control contrasts. The A×V
interaction is replayed but has no required sign.

## 8. Inference, multiplicity, and stops

The independent unit is a gene. Duplicate rows aggregate within gene. Every reached
core gate receives the full stagewise alpha 0.05 in the fixed sequence; alpha is not
fragmented across unreached gates. Within a reached family, shared gene-bootstrap
resamples and a simultaneous max-|t| critical value cover all required rows,
directions, and estimators.

The ordinary gene mean and prespecified 10%-trimmed gene mean must both pass. Use a
studentized nonparametric gene bootstrap with at least 10,000 draws, seed 0, shared
resample indices, bootstrap-t inversion, and add-one Monte Carlo p-values. The
bootstrap and its p-values are **not exact**.

Material-positive means the simultaneous lower bound is strictly above +0.05.
Equivalent-within-SESOI means the interval is strictly inside [-0.05,+0.05] and both
boundary tests pass. Boundary touching fails. Active pseudo or L0H2 controls are
scientific failures, not generic instrument failures.

A reached core scientific failure stops all later core interventions. A core
instrument failure also stops later core interventions. L2H15 is then `NOT_TESTED`.
Once the core succeeds, the secondary family is reported separately and cannot revise
the L3H1 status.

## 9. Preflight and prospective health

Preflight requires exact checkpoint, GTF, FASTA, FASTA index, clean Git, joint-stack,
and runtime identities; device exactly `mps`; and no MPS fallback environment variable.
It checks full `model.cls` replay, attention output-projection replay, whole-head no-op
replay, exact surgery assignment/outside-preservation invariants, fixed-V invariants,
classifier plants/boundaries, and exact R1c same-row common-leg replay.

The validation replay freezes the SDPA-to-eager 1e-4 tolerance before target unsealing.
The target repeats that exact three-leg, 400-row comparison before scientific reader
interpretation. Both stages record per-leg and maximum errors. The fixed-V A×V family
also records exact `A@V` reconstruction maximum error and requires it below 1e-5.

Health is paired no-op noninferiority at deterministic non-site probes, not an absolute
accuracy floor. Compare each query-local head-zero stress to its exact same-hook no-op;
the one-sided 95% gene-bootstrap lower bound for paired accuracy difference must exceed
-0.15.

Core health registers L3H1 and L0H2. Secondary health independently registers L2H15.
The old R3c L2 health result is not reused. Secondary health failure closes only the
L2H15 role path; it is neither R5a-preflight failure nor L3H1 target failure.

## 10. Immutable target ledger and provenance

The R5a target identity binds stage, frozen checkpoint digest, R1c sorted-unique-gene
digest, exact ordered-row digest, R1c capstone-construction digest, and the
`masked_acceptor_G_margin_only` endpoint policy. The create-only receipt precedes
output resolution, cohort/genome access, checkpoint inspection, model loading, and
intervention scoring. Any later failure spends that identity.

The target manifest binds the R5a result; the R5a attempt receipt; frozen R5a preflight
artifact and manifest; all six exact R1c target/predecessor files; R3d artifact,
manifest, registry, registry manifest, attempt, and completion; R3c artifact and
manifest; checkpoint files; GTF, FASTA, and index; recursive local Python closure; all
R5 protocols; clean Git revision; and runtime policy. After result and manifest are
published immutably, a create-only completion binds their identities and the official
disposition.

Artifacts store primitive cells, decision matrices, exact row/gene order, family-row
registries, sufficient-statistic digests, capabilities, execution order, and the
separate diagnostic units. Figures may use recorded artifacts only.

## 11. Frozen reporting template

If the R1c target gate and all three R5a core gates pass:

> On the authenticated R3d Path-A premise and the sealed R1c masked-acceptor behavior
> gate, query-fixed position×word, exact QK/fixed-V, and attention×value/fixed-V
> interventions support L3H1 as a paired-specific, position-sensitive upstream-donor
> reader for the masked-acceptor margin. Mean and 10%-trimmed mean clear prespecified
> target-minus-control requirements. This is a scoped causal reader result, not yet a
> write, delivery, mediation, sufficiency, faithfulness, or biological-mechanism claim.

Append exactly one secondary sentence: either L2H15 supports an additional upstream
reader role, does not support that role, is scientifically unresolved, or was not
tested because its own prospective instrument/health gate failed. Never use L2H15 to
qualify or retract the L3H1 headline.

If any core gate fails, report the exact scientific stop or instrument stop and leave
all downstream core claims unmade.
