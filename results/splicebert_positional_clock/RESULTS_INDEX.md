# Positional clock (PBC v2 successor) — results index

> **Tier: T2 — provenance.** Model-free construction-stage capacity/power record. This family
> contains no model outcome.

**The directory name `splicebert_positional_clock` is PROVISIONAL** and may be renamed before
anything freezes. Everything below therefore lives under `development/` or `superseded/`, not at
the family root, so a later rename costs a move, not a retraction. Do not cite this path as a
stable identifier yet.

Machine-readable entry point: [`../MANIFEST.csv`](../MANIFEST.csv) / [`../MANIFEST.json`](../MANIFEST.json).
Filter the `artifact` column on `results/splicebert_positional_clock/`. Column semantics and the four
`status` values are defined in [`../RESULTS_INDEX.md`](../RESULTS_INDEX.md).

## Relationship to PBC v1

This family reads two inputs from the closed PBC v1 study but never writes into it:

* the hash-pinned candidate census the PBC v2 successor project produced
  (`candidate_census_v1.json`, 2.34 GB, sha256 `9be356ac26ff18aa750b0552684d4a94fa6b8963fee01666608b38cc7de25614`,
  an INPUT that lives outside this repo at
  `gfm-circuit-discovery-pbc-successor/results/splicebert_positional_boundary_coherence/development/`
  and is never copied in);
* the committed capacity certificate
  [`results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.json`](../splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.json),
  read-only, which supplies both the `slot_gene_sets` filter configuration and the seven per-slot
  `gene_ids_sha256` values every run must reproduce before any rung is evaluated.

`studies/splicebert_positional_boundary_coherence/` and
`results/splicebert_positional_boundary_coherence/` are **never written to** from this family; that
study is closed. See `CLAUDE.md` and `AGENTS.md` for the standing rule that the successor study
inherits blacklists, geometry and sign-free variance for sizing from PBC v1, never its archived code
or its signed magnitudes.

## Layout

| Path | What may be written there |
|---|---|
| `./development/` | construction, calibration, smoke, and other pre-freeze output — everything this provisional study has produced so far |
| `./superseded/` | artifacts replaced by a later version of themselves, retained for provenance — see each one's sibling `*.SUPERSEDED.md` note for why |

## Checkpoint binding

**None.** This family is model-free, stage `construction`: no SpliceBERT checkpoint is loaded by
anything here. Every artifact records `model_execution: false`, `checkpoint_loaded: null`,
`cohort_selection_attempted: false` explicitly rather than omitting the fields.

## What every artifact here carries

1. an immutable JSON result: counts, hashes, exact-feasibility verdicts and power projections only
   — **no gene ID ever appears** (verified recursively by
   `tests/test_pbc_v2_capacity_runner.py`: no string in the payload matches `^ENSG\d`);
2. a sibling `*.manifest.json` with command, code revision and dirty state, input paths + sha256s
   (census, certificate), parameters, seeds (none — the analysis is exact arithmetic and max-flow,
   no RNG anywhere in it), environment, exact counts, and output hash.

`scripts/splicebert_positional_clock/run_pbc_v2_capacity.py` calls
`gfmcd.artifact_guard.guard_output_path` before writing either file, so a rerun that would overwrite
an existing artifact refuses rather than replacing it. Every new file here needs a row in
`../MANIFEST.csv` **and** `../MANIFEST.json`; run `uv run python scripts/verify_manifest.py` to check.

## Runs

| Run / result | Stage | Status | Artifact | Manifest | Claim IDs |
|---|---|---|---|---|---|
| PBC v2 capacity/power ladder v2 | construction | `BUILT` | [`development/pbc_v2_capacity_ladder_v2.json`](development/pbc_v2_capacity_ladder_v2.json) | [`manifest`](development/pbc_v2_capacity_ladder_v2.manifest.json) | — |
| PBC v2 capacity/power ladder v1 | construction | `superseded` | [`superseded/pbc_v2_capacity_ladder_v1.json`](superseded/pbc_v2_capacity_ladder_v1.json) | [`manifest`](superseded/pbc_v2_capacity_ladder_v1.manifest.json) | — |

### PBC v2 capacity/power ladder v2

Drives the four model-free analysis layers in `gfmcd.splicing.pbc_v2_capacity` across the fixed rung
ladder `N ∈ (600, 660, 729, 780, 819, 864, 909)`. For each rung it records the block allocation
across the three phase pairs; per-slot and per-union demand, available, margin, deficit, and
`residual_unique_genes_per_slot_not_jointly_realizable` (`available - demand`, carrying a note that
it double-counts genes shared across slots and is **not** a jointly buildable holdout); a genuinely
joint `joint_residuals` figure computed by max-flow (see below); the exact `feasibility()` verdict
(max-flow / min-cut, not counting — a Hall-type joint-oversubscription obstruction can fail even
when every individual slot and union passes); and `power_projection(N)`.

v2 supersedes v1 — see
[`superseded/PBC_V2_CAPACITY_LADDER_V1.SUPERSEDED.md`](superseded/PBC_V2_CAPACITY_LADDER_V1.SUPERSEDED.md)
for why (a confirmed manifest/code sha256 mismatch, plus renamed/added honesty fields). The
underlying gene pool and numeric conclusion are unchanged; v1 is retained for provenance, not for
citation.

**Selection rule** (recorded verbatim in the artifact, `selection_rule`): *the largest N whose every
slot and every union clears a margin of 0.25 AND whose exact feasibility passes AND whose Ψ verdict
is `powered_for_every_rho`.*

**Real-run outcome: `selected_rung` is `null`.** No rung on this ladder satisfies the conjunction.
Each of the three conditions is individually satisfiable somewhere on the ladder, but never by the
same rung:

* margin ≥ 0.25 holds on **every** rung, at every slot and union;
* exact feasibility passes only at **N=600** — every rung from N=660 through N=864 fails it because
  `donor_phase_2` and `acceptor_phase_2` share 800 genes in the real pool (available 809 and 854
  respectively; union 863), a two-slot joint-oversubscription obstruction plain per-slot counting
  cannot see (e.g. at N=660, demand 880 against the 863-gene union, shortfall 17). At **N=909** the
  binding certificate is a *different*, six-slot set — `{acceptor_phase_0..2, donor_phase_0..2}` —
  with union 3227, demand 3636, shortfall 409: a minimum cut returns a *maximum-deficiency*
  violator, so the reported cut can change with N even though the two-slot
  `{acceptor_phase_2, donor_phase_2}` obstruction is still present underneath at N=909 (it is simply
  no longer the tightest one there);
* the Ψ verdict first reaches `powered_for_every_rho` at **N=729** — but N=729 is already infeasible
  (shortfall 109 against the `{acceptor_phase_2, donor_phase_2}` min-cut).

**The feasibility ceiling is a ladder property, not the pool's own ceiling.** N=600 is the largest
*evaluated rung* that is exactly feasible — it is not the pool's constructible maximum. Under the
runner's own exact largest-remainder allocation, the rotated-lattice pool's true ceiling is
**N=647** (N=647 → the `{acceptor_phase_2, donor_phase_2}` joint demand is 862 ≤ the 863-gene union;
N=648 → 864, infeasible). 647 is not on the fixed ladder, so it never appears as a `feasible: true`
row above, but it is strictly below the power floor (729) exactly as 600 is — the conclusion is
unchanged either way; no ladder rung, nor the pool's own ceiling, is on both sides of the
feasibility/power conjunction at once.

The genuinely joint (max-flow-exact) residual for the rungs that are actually feasible — on this
ladder, only **N=600**: **5,790** genes remain unassigned overall (`|union of all seven slots| -
max_flow`, i.e. 8,790 available across the whole pool minus the 3,000 genes the exact solver spent
satisfying N=600's demand), and the `{donor_phase_2, acceptor_phase_2}` pair specifically retains
**63** genes beyond its own combined demand (`|donor_phase_2 ∪ acceptor_phase_2|` = 863, combined
demand 800). The per-slot `residual_unique_genes_per_slot_not_jointly_realizable` figures printed
elsewhere in the artifact (e.g. "N=909 leaves 203 unspent donor-φ2 genes") double-count genes shared
between slots and are not a number to publish as a build capacity — see `joint_residuals`, above,
for the rungs where a jointly buildable figure exists at all (only the feasible ones; every
infeasible rung's joint figure is negative by construction, meaning no genes are left over at all).

All seven `slot_hash_verification` booleans are `true`: the streamed census pool reproduces the
committed certificate's seven `gene_ids_sha256` values exactly before any rung was evaluated (a hard
gate — the run refuses to write an artifact on any mismatch, not merely log one). The certificate's
`sample_size_plan.variance` is also checked against the module's frozen
`LEGACY_RESIDUAL_VARIANCE_UPPER_BOUND_DEV120` before any rung is evaluated, and the run refuses to
proceed on drift; today it matches (`0.115942293009`).

`--smoke` exercises the same demand/feasibility/power layers on a small, fully synthetic pool and
returns before `--census` is ever opened; it is not an authoritative record (its manifest `status` is
`SMOKE_ONLY_NON_AUTHORIZING`) and is not committed here.
