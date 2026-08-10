# Positional boundary coherence — results index

> **Tier: T2 — provenance.** This family is **empty**. Nothing here controls a claim, because nothing
> here exists yet.

Machine-readable entry point: [`../MANIFEST.csv`](../MANIFEST.csv) / [`../MANIFEST.json`](../MANIFEST.json).
Filter the `artifact` column on `results/splicebert_positional_boundary_coherence/`. Column semantics
and the four `status` values are defined in [`../RESULTS_INDEX.md`](../RESULTS_INDEX.md).

The study that will write here is
[`studies/splicebert_positional_boundary_coherence/`](../../studies/splicebert_positional_boundary_coherence/README.md).
Its frozen design is `PREREGISTRATION.md`; its claim and result ledgers are `CLAIMS.md` and `RESULTS.md`
in the same directory and are also empty. **This file exists so that the first gated run has a declared
place to land, not because a run has happened.**

## Layout, declared before the first write

Declared in advance so the first run cannot scatter. This tree follows the repository's dominant
convention — canonical artifacts at the family root, development output in a subdirectory — not the
`canonical/` variant used by `results/origin-crossmodel/` and `results/origin-siblings/`.

| Path | What may be written there |
|---|---|
| `./` | target and Phase-III confirmation artifacts named by the study's `CLAIMS.md` / `RESULTS.md`, each with its sibling `*.manifest.json` and its independent `*.audit.json` |
| `./development/` | construction, calibration, preflight, candidate-screen, and smoke output — everything produced at the `construction` or `development` stage |
| `./superseded/` | a later version of an artifact that already shipped; the original is retained, never deleted |

A development artifact is **never promoted** by moving it to the family root. Promotion is a new run at
the `target` or `phase3_reserve` stage against a sealed authorization, and it writes a new file.

## What every artifact here will have to carry

The study's own artifact contract (`README.md` §"Artifact contract") is stricter than the repository
minimum, and the stricter one governs:

1. an immutable JSON result with per-block endpoints sufficient to rederive every reported statistic;
2. a sibling manifest with command, code revision and dirty state, input hashes, checkpoint hash,
   parameters, seeds, environment, exact sample counts, exclusions, and output hashes;
3. a freeze/authorization artifact for any later stage; and
4. an **independently produced audit** that recomputes gates, families, decision class, chronology, and
   hashes without importing the analysis verdict function.

`scripts/splicebert_positional_boundary_coherence/` runners call `gfmcd.artifact_guard.guard_output_path`
before writing, so a rerun that would overwrite an existing artifact refuses rather than replacing it.
Every new file here needs a row in `../MANIFEST.csv` **and** `../MANIFEST.json`; run
`uv run python scripts/verify_manifest.py` to check.

## Checkpoint binding

Every model artifact in this family must record `SpliceBERT.1024nt`, `pytorch_model.bin` sha256
`2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`. A mismatch is fatal, not a warning.
No result from the stage-1 510-nt pair or from the five GTEx fine-tuned checkpoints belongs here.

| Run / result | Stage | Status | Artifact | Manifest | Audit | Claim IDs |
|---|---|---|---|---|---|---|

The table is empty and stays empty until a gated run completes.
