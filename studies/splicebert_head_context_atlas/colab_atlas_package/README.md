# Colab package — wider SpliceBERT Head–Context Atlas discovery screen

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

This package runs the original atlas objective: measure **all 96 attention heads and all six MLPs**
across six overlapping biological context labels, using a balanced discovery cohort. Competition is not
the organizing question.

## What the screen measures

- **A — native reconstruction:** loss of the masked splice-site margin/log-probability when one head
  output or MLP residual write is zeroed.
- **B — partner dependence:** loss of the native target readout when that component's state is scrubbed
  to the state induced by destroying the annotated donor–acceptor partner.
- **Collateral health:** the same A/B interventions on genuinely **masked**, non-site nucleotide probes,
  measured on a deterministic 512-packet panel.

Every packet is evaluated once. The 6,487 matched case/control pairs are joined afterward to estimate
general versus context-associated use. Labels remain overlapping; all non-focal labels were matched
exactly.

## Hard boundaries

- Discovery only. chr8/9 and historically touched chr11/16/19 are absent from the packaged sequences.
- No acceptor-coordinate shortcut: donor masks `donor_win`; acceptor masks `acceptor_win-1`.
- No head function is named from the screen alone. QK/OV/MLP/group/DAG validation follows shortlisting.
- No unmasked “MLM health” proxy. Four non-site bases per packet are actually masked.
- No GTF/FASTA download is needed: the outcome-independent sequence windows are bundled.

## Local evidence before packaging

- Cohort audit: 6,487 discovery pairs / 11,807 packets / 10,900 sequences; all ten role-valid contrasts
  balanced; 0 audit failures.
- Model preflight: four packets covering all six labels and both target roles; 816 component-estimand
  rows; 0 failures; head/MLP source-identity error exactly 0.
- Bulk-runner smoke: 2,448 comparisons against the independent preflight artifact; maximum absolute
  difference 0.0; validation and merge both pass.
- Test suite: 25 passed.

## Colab order

1. Mount Drive, unpack the zip, install requirements (never reinstall Colab's CUDA Torch).
2. Verify every bundled file hash and CUDA availability.
3. Run tests and the four-packet cross-device reference smoke.
4. Run a 32-packet benchmark; inspect projected GPU hours, storage, and peak memory.
5. Run the 512-packet **health phase first**. It must pass both masked-health floors.
6. Set the explicit GO flag and run the full 11,807-packet target phase; output goes to Drive and resumes
   at packet-shard granularity.
7. Validate every shard/hash/component grid, then merge to gzip JSONL matrices.

The notebook `run_atlas_screen_colab.ipynb` implements this sequence.

## Stage 2 continuation

After the audited discovery and whole-component validation, the original target execution used
`run_atlas_stage2_colab_v3.ipynb`. The completed-run recovery and post-audit handoff use
`run_atlas_stage2_recovery_a3.ipynb` with `gfm_atlas_stage2_colab_overlay_v6.zip` and its hash sidecar.
The repaired overlay carries the complete Stage 2 sibling-import closure, every frozen whole-validation
runtime reference declared by the audited package, the sealed internal-validation cohort, frozen packages,
and 256 passing health shards; it contains no Stage 2 target outcomes. The notebook is self-contained under
**Runtime → Run all**, writes to `/MyDrive/gfm_atlas_stage2_v1_results`, and resumes target shards there. It
does not read or reuse the completed discovery `target_shards` directory.

The superseded Stage 2 v1 overlay omitted `run_atlas_mechanism_validation.py`; the v2 notebook's
`PYTHONPATH` workaround could not repair an absent file. Both failed artifacts remain on Drive for
provenance. Versioned overlays are built by `build_stage2_overlay.py`. Overlay v2 imported successfully but
omitted the package-declared frozen
whole-validation row tables needed at target execution. Overlay v3 supersedes it; the builder regression
test now verifies both code-import and package-declared data-reference closure, plus exclusion of Stage 2
target outcomes.

Overlay v6 additionally carries the representative-DAG builder, its independent validator, and the
measured screening-graph exporter. It still carries no target shard, target manifest, conditional result,
or mechanism label. Its SHA-256 is
`c12e9a2096e416e0ee59e2fe9f3c2a0ad7f46831ac5569c397020760b9bf3e5d` (288 entries).

The A3 recovery preserves the original one-gate target failure, the first portability-failed A3 audit,
and all target shards. It writes only new `*_a3_v2` merge/analysis/audit paths. Mechanism labels may be
read only after both the A3 execution audit and independent analysis audit pass with zero failures.

## Reproducing the Stage-2 target run

[`RUN_STAGE2_COLAB.md`](RUN_STAGE2_COLAB.md) is the operational runbook for the audited stage-two target:
overlay hash verification, the target-phase command, and the audit/merge/analyze/reconstruct sequence,
together with the A3 recovery contract. It is the only written reproduction path for that result — read it
before touching a Stage-2 artifact, and do not use it to overwrite the completed run.
