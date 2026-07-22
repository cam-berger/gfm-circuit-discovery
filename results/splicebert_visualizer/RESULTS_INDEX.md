# Segment-attribution visualizer bundles — results index

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Machine-readable entry point: [`../MANIFEST.csv`](../MANIFEST.csv) /
[`../MANIFEST.json`](../MANIFEST.json). Column semantics and the four `status` values are defined in
[`../RESULTS_INDEX.md`](../RESULTS_INDEX.md).

This directory holds export bundles for the segment-attribution visualizer, written by
`scripts/splicebert/export_segment_attribution_bundle.py`. They are classified `infrastructure`: they
carry no `claim_id` and support no claim. Per
`studies/splicebert_segment_attribution/README.md`, a bundle must not be committed as a claim
artifact, and its screening graphs are attribution rankings — not causal effects, and not promotable
past the attribution rung of the ladder.

The study's own README lists the outstanding validation steps (exact-zero/matched-reference patching
of top-ranked routes, planted and real positive controls, influence-preserving pruning, a separate QK
graph, and cross-gene replication) that would be required before any bundle motif could be named a
mechanism.

## Integrity check

```bash
uv run python scripts/verify_manifest.py
```

Every file added, removed, or moved here must be reflected in `results/MANIFEST.{csv,json}` in the
same change.
