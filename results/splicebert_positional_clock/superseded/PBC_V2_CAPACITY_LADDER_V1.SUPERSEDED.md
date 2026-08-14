# `pbc_v2_capacity_ladder_v1` — superseded

`pbc_v2_capacity_ladder_v1.json` and its sibling manifest are retained here for provenance only.
They are superseded by
[`../development/pbc_v2_capacity_ladder_v2.json`](../development/pbc_v2_capacity_ladder_v2.json)
(+ [`manifest`](../development/pbc_v2_capacity_ladder_v2.manifest.json)).

## Why

An adversarial review of the branch that produced v1 confirmed eight defects, filed as A1–A8 in the
task-6 brief. Two are why v1 itself cannot stand as the record:

- **A8 (manifest hygiene):** v1's manifest pinned the producing runner at sha256 `769c1257…`, but
  the committed file was `5df4324b…` (`git_dirty: true` — the runner was edited after the run and
  then committed). An auditor checking out the commit and hashing the runner gets a different
  digest than the one in the manifest, so the manifest could not prove which code produced the
  artifact. v2's manifest is generated from, and pins the sha256 of, the code as actually committed.
- **A1–A2, A4–A6 (code and schema fixes):** the runner and `gfmcd.splicing.pbc_v2_capacity` changed
  in ways that alter the artifact's own field names and content — `legacy_block_sd_9cell` /
  `LEGACY_BLOCK_VARIANCE_9CELL` renamed to name what the number actually is (a 95% upper bound from
  the stopped 120-block development role, not a measured 9-cell block SD); `residual_holdout_unique_genes`
  renamed to `residual_unique_genes_per_slot_not_jointly_realizable` plus a new, genuinely joint
  `joint_residuals` figure computed by max-flow; the `direction` block gained `estimate_class` /
  `lower_bound_reason` and renamed `blocks_needed_worst_case` to `blocks_needed_floor`; a new
  consistency check between the PBC-v1 certificate and the frozen legacy variance; and
  `no_feasible_rung_reason`'s prose fixed to name the margin condition and drop a tautological
  `feasible=True` restatement. `SCHEMA` was bumped to `.v2` because the shape genuinely changed.

The underlying gene pool, census, and certificate are unchanged, and the ladder's numeric
conclusion is unchanged (no rung is simultaneously exactly feasible and Psi-powered at the frozen
SESOI) — only the artifact's field names, added honesty fields, and manifest self-consistency
changed. See `results/splicebert_positional_clock/RESULTS_INDEX.md` for the current record and
`docs/PREREG_CONFORMANCE.md` / the donor-circuit `CORRECTIONS.md` conventions this move follows.

Do not build on v1. Cite v2.
