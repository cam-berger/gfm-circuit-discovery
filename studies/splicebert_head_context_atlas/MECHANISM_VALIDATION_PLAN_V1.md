# Head–Context Atlas mechanism-validation plan v1

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Status: **GENERIC RULES FROZEN BEFORE THE FULL DISCOVERY MATRIX WAS READ** (2026-07-19).
Candidate identities, matched controls, and representative discovery packets will be instantiated by a
deterministic package builder only after the independently audited discovery shortlist exists. That
candidate-specific package must then be frozen before any internal-validation packet or outcome is read.

## Purpose and claim boundary

The 102-component screen answers *where to look*. This stage tests what a shortlisted component actually
does. It may license a bounded label such as `partner-sensitive routing`, `partner-sensitive value/write`,
`mixed`, or `MLP-mediated path`; it may not turn a rank, attention map, or attribution edge into a
biological function.

The mechanism unit is:

`component × estimand × target role/context × exact intervention × untouched validation result`.

Competition remains absent from the organizing family. The ten atlas contrasts and two target roles are
the only v1 contexts.

## Inputs and sealing

The package builder may read only:

- the complete, independently audited **discovery** analysis and shortlist;
- the complete discovery target/health matrices and matched-pair table;
- the frozen analysis plan/amendment and this document.

It must not open, hash, count, or sample internal-validation packet, sequence, pair, or outcome files.
The package records `validation_loaded=false` and only the fixed filenames/rules that the later runner
will use. Any candidate/control/example decision made after validation access is post hoc.

Execution and simultaneous-interval details are frozen in
`MECHANISM_VALIDATION_AMENDMENT_A1.md`; it was written under the same outcome-blind boundary.
Conditional QK/OV/path and DAG rules are sharpened in `MECHANISM_VALIDATION_AMENDMENT_A2.md`, also
before full discovery or validation outcomes.

## Candidate-specific package

For every selected general profile or context association, freeze:

1. candidate ID, component, A/B estimand, role/contrast, discovery effect and uncertainty;
2. a same-type importance-matched component control: same layer when an attention alternative exists,
   otherwise nearest absolute discovery effect, excluding every selected component; deterministic hash
   breaks ties;
3. exact group membership—general candidates grouped by estimand, context candidates by
   contrast × estimand; groups are patched jointly and single-component effects are never summed;
4. representative discovery material:
   - general: one donor and one acceptor packet closest to that role's recorded component mean;
   - context: the matched case/control pair whose component difference is closest to the full matched
     mean; the control packet itself is the negative-context example;
5. every input hash, selection distance, tie-break, intervention source, and expected output family.

Representative packets explain a typical measured case. They do not enter inference and are never
selected for visual drama.

## Exact interventions

### Whole component and exact groups

- **A:** zero the selected head output(s) or MLP residual write(s) in the native masked-site forward.
- **B:** replace selected head output(s) or MLP write(s) with the same packet's partner-destroyed-source
  value in the native forward.
- Apply the same operation jointly to each frozen group. Report its exact effect and uncertainty; do not
  compare it with a sum of single effects.
- Source-to-source patches for every component and group must reproduce the native logits within `1e-4`;
  all-component source patches must reproduce the source forward within `1e-3`.

### QK versus V/OV for attention heads

For a **B** candidate, the primary source is the same packet with its paired splice motif destroyed:

- `B_QK`: patch that head's query and key projections at all real-token positions from the destroyed
  source, leaving native values;
- `B_V`: patch its value projections from the destroyed source, leaving native query/key routing;
- `B_QKV`: patch all three as a plumbing bridge to the full head-output scrub.

These are separately interpretable main effects. Their algebraic interaction is not a binding test and
is never claimed (R23). Native→native Q/K/V identity, QKV→head-output agreement, finite logits, and
masked-health controls are required.

An **A** zero-output effect has no natural QK counterfactual. For A-only candidates, QK/V uses the frozen
matched-control packet as a *state-interchange diagnostic*: case QK or V is replaced with its matched
control source. This can localize where case-specific information resides, but it does not decompose the
zero-output A effect and cannot be compared numerically as an A fraction. The output schema must call it
`matched_control_state_interchange`, not `A_QK`/`A_V`.

An attention mechanism label requires untouched-validation support:

- `partner_sensitive_routing`: B_QK is material and B_V is bounded within ±0.05 nats;
- `partner_sensitive_value_write`: B_V is material and B_QK is bounded;
- `mixed_partner_sensitive`: both are material;
- otherwise `mechanism_unresolved`.

No categorical label is issued from A-only interchange. Importance-matched controls are reported beside
all QK/V results; intervention-size diagnostics remain visible and are not post-treatment covariates.

### MLP and path validation

For an MLP candidate, patch its exact residual write under A or B as above. Then test upstream dependence
on the frozen shortlisted attention group and downstream propagation with write-level clamps only. A path
claim requires candidate-specific recovery beyond the importance-matched MLP/head-group control and a
non-negligible whole-component denominator. Layer-state or embedding clamps cannot serve as completeness
memorials (R21), and attention/MLP effects are never added.

For attention candidates, downstream MLP recovery is secondary: scrub the head, restore one downstream
MLP's clean write, and compare with an importance-matched downstream-write restoration. This is a
controlled recovery assay, not an additive mediation fraction.

## Untouched internal-validation inference

- General endpoints use every unique internal-validation packet once, roles separate, with one-way
  gene-clustered inference.
- Context endpoints use every internal-validation pair for the frozen contrast, inverse control-reuse
  weights, and the same unique-dyad two-endpoint gene sandwich as discovery.
- Repeat only the candidate/component/estimand/context endpoints frozen in the package.
- Holm-FWER `alpha=0.05` covers all whole-component atlas validation endpoints. Within each whole-effect
  endpoint that validates, a max-|t| family covers its QK, V, exact-group, and prespecified MLP/path
  endpoints. Mechanism labels require simultaneous intervals.
- The material floor remains `tau=0.05` nats. `bounded_small` requires the entire simultaneous interval
  inside ±tau; otherwise a non-significant result is `inconclusive`.
- General/context validation can fail or remain underpowered without being renamed. Thin arms report
  achieved MDE and cannot license an equivalence claim merely because discovery was large.

## Attribution DAGs and visualization

After validation, export for each surviving relation:

- the frozen median-effect discovery case and its matched negative-context control;
- a model-native single-G mask trace, with an optional separately labeled joint two-nucleotide core mask;
- activation×Jacobian component screening edges with remainder nodes;
- explicit evidence labels on every edge: `screening`, `exact_component`, `QK`, `V/OV`, `MLP_path`, or
  `negative_control`.

Only paths supported by the exact validation artifact are highlighted as validated. The surrounding DAG
remains a measured candidate map, not a complete feature-level attribution graph. `gfm-visualizer`
receives only validation-pinned profiles, simultaneous intervals, representative packets, intervention
results, and these evidence-graded DAGs; discovery ranks stay in this repository.

## Stop rules

- Any source-identity, QKV bridge, coordinate, grid, health, split, provenance, or repeated-dyad failure:
  stop and diagnose before inference.
- Whole effect fails untouched validation: retire the discovery relation; do not run a mechanistic story.
- Whole effect validates but QK/V/path is inconclusive: retain the causal component relation with
  `mechanism_unresolved`.
- No shortlist candidates: publish the powered atlas null/instrument limits; do not force a mechanism.
