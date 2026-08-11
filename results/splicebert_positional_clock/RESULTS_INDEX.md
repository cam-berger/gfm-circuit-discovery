# Positional clock (PBC v2 successor) — results index

> **Tier: T2 — provenance.** Model-free construction-stage capacity/power record. This family
> contains no model outcome.

**The directory name `splicebert_positional_clock` is PROVISIONAL** and may be renamed before
anything freezes. Everything below therefore lives under `development/`, not at the family root, so
a later rename costs a move, not a retraction. Do not cite this path as a stable identifier yet.

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
| PBC v2 capacity/power ladder v1 | construction | `BUILT` | [`development/pbc_v2_capacity_ladder_v1.json`](development/pbc_v2_capacity_ladder_v1.json) | [`manifest`](development/pbc_v2_capacity_ladder_v1.manifest.json) | — |

### PBC v2 capacity/power ladder v1

Drives the four model-free analysis layers in `gfmcd.splicing.pbc_v2_capacity` (committed at
`6d84ee4b`, tests green — imported as a finished dependency, not modified) across the fixed rung
ladder `N ∈ (600, 660, 729, 780, 819, 864, 909)`. For each rung it records the block allocation
across the three phase pairs; per-slot and per-union demand, available, margin, deficit, and the
**residual holdout pool** (`available - demand`, so the power / capacity-margin / future-holdout
trade-off is visible in one table); the exact `feasibility()` verdict (max-flow / min-cut, not
counting — a Hall-type joint-oversubscription obstruction can fail even when every individual slot
and union passes); and `power_projection(N)`.

**Selection rule** (recorded verbatim in the artifact, `selection_rule`): *the largest N whose every
slot and every union clears a margin of 0.25 AND whose exact feasibility passes AND whose Ψ verdict
is `powered_for_every_rho`.*

**Real-run outcome: `selected_rung` is `null`.** No rung on this ladder satisfies the conjunction.
Each of the three conditions is individually satisfiable somewhere on the ladder, but never by the
same rung:

* margin ≥ 0.25 holds on **every** rung, at every slot and union;
* exact feasibility passes only at **N=600** — every rung from N=660 up fails it, because
  `donor_phase_2` and `acceptor_phase_2` share 800 genes in the real pool (available 809 and 854
  respectively), a joint-oversubscription obstruction plain per-slot counting cannot see;
* the Ψ verdict first reaches `powered_for_every_rho` at **N=729** — but N=729 is already infeasible
  (shortfall 109 against the `{acceptor_phase_2, donor_phase_2}` min-cut).

The feasibility ceiling (600) sits strictly below the power floor (729); no ladder rung is on both
sides at once. `no_feasible_rung_reason` states this in the artifact so no reader has to infer it.
The residual `donor_phase_2` holdout the brief flags is exact and reproduced: **N=909 leaves 203
unspent donor-φ2 genes, N=819 leaves 263, N=729 leaves 323.**

All seven `slot_hash_verification` booleans are `true`: the streamed census pool reproduces the
committed certificate's seven `gene_ids_sha256` values exactly before any rung was evaluated (a hard
gate — the run refuses to write an artifact on any mismatch, not merely log one).

`--smoke` exercises the same demand/feasibility/power layers on a small, fully synthetic pool and
returns before `--census` is ever opened; it is not an authoritative record (its manifest `status` is
`SMOKE_ONLY_NON_AUTHORIZING`) and is not committed here.
