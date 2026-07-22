# Head–Context Atlas mechanism-validation amendment A1

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Status: **FROZEN BEFORE THE FULL DISCOVERY MATRIX OR ANY INTERNAL-VALIDATION MODEL OUTCOME WAS READ**
(2026-07-19).

This amendment resolves execution and inference details that
`MECHANISM_VALIDATION_PLAN_V1.md` left implicit. It changes no candidate identities, shortlist caps,
biological contexts, causal estimands, material floor, or validation data.

## A1.1 Physical split references and access order

`atlas_internal_validation` is the row-level split label; the physically separated cohort files are
named `.validation.{pairs,packets,sequences}.jsonl`. Candidate packages seal these real filenames as
repository-relative references so the same package is portable to Colab. The package builder does not
open, hash, count, or sample them. The validation executor must verify the package and its independent
audit first, and only then resolve or touch these files.

The executor independently rechecks zero gene and sequence overlap with discovery, absence of
chr8/9/11/16/19, and source-manifest hashes. This internal gene-hash validation is not chr8/9 confirmation;
chr8/9 remains sealed.

## A1.2 Exact whole-effect execution

- Masked health runs first on a deterministic role-balanced 256-packet panel. A large generic effect is
  still an annotation, not an exclusion; baseline health, finite outputs, coordinate checks, and source
  identities are hard gates.
- Target execution touches only packets needed by at least one frozen endpoint. General candidates use
  the whole validation packet union, context candidates use the corresponding matched-pair union, and
  the independent L3H1/B control uses donor targets.
- Candidate singletons, their importance controls, and exact groups are explicit endpoints. Identical
  physical interventions shared by overlapping endpoints are computed once and emitted under each
  frozen semantic alias. Groups containing heads and MLP writes are intervened on simultaneously.
- Packet shards are immutable under a hash of package, package audit, checkpoint, cohort, endpoint grid,
  phase, precision, and seeds. A resume accepts shards only under that exact contract.

## A1.3 Whole candidate family and simultaneous intervals

The primary whole-component family contains only shortlisted **candidate relations**:

- each general candidate contributes two endpoints, donor and acceptor;
- each context candidate contributes its one frozen matched contrast.

Importance controls, exact groups, and the L3H1/B positive control do not consume primary-family alpha.
They are prespecified diagnostics/mechanism endpoints and cannot rescue a candidate that fails its own
whole-effect endpoint.

Detection uses two-sided Holm-adjusted p-values across this whole family. Because the v1 plan also
requires a simultaneous interval for material/bounded classification but did not specify a Holm interval
construction, report conservative Bonferroni `(1−0.05/m)` simultaneous intervals over the same `m`
endpoints. This is stronger than marginal coverage and avoids pretending Holm-adjusted p-values are
ordinary confidence bounds. Report marginal and simultaneous intervals together.

Classification uses the simultaneous interval and `tau=0.05` nats:

- `material_support` / `material_suppress`: the simultaneous interval clears `+tau` / `−tau`, the
  Holm-adjusted p-value is ≤0.05, and direction matches discovery;
- `detected_subfloor`: Holm-significant and the simultaneous interval excludes zero but does not clear
  `tau`;
- `bounded_small`: the entire simultaneous interval lies inside `[−tau,+tau]`;
- otherwise `inconclusive`.

Both role endpoints must validate in their frozen direction for a `cross_role_general` relation to
survive. Failure of one role does not rename it post hoc as role-associated. A context relation survives
only through its own frozen contrast. Familywise and marginal 80%-power MDEs are both reported; only the
familywise MDE can support a familywise bounded-null statement.

## A1.4 Controls, groups, and progression

- L3H1/B is a donor-partner instrument control. Its one-way gene-clustered 95% CI must exclude zero in
  the positive direction before a null B-mechanism result is called interpretable; it remains outside the
  atlas candidate family and gains no atlas label automatically.
- Importance-matched control and exact-group point estimates, clustered intervals, intervention sizes,
  and masked-health effects are reported now. They remain diagnostic until placed with QK, V, and
  prespecified MLP/path endpoints in the candidate's later within-endpoint max-|t| family.
- Only whole-effect relations that survive untouched internal validation advance to QK/V/path assays and
  representative DAG export. If none survive, no mechanistic story or visualizer artifact is forced.

## A1.5 Plumbing evidence available before outcomes

The four-packet discovery smoke exercised A/B heads, A/B MLP writes, mixed head+MLP exact groups, health,
target, immutable resume, and package/split boundaries. Runtime source identities were exactly zero.
An independent verifier compared every singleton result with the earlier 102-component model preflight:
maximum target and health absolute error `0.0`; duplicate physical-computation aliases also differed by
`0.0`. No internal-validation file or model outcome was loaded.
