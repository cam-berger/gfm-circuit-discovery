# Preregistration v2: corrected native GTEx classifier coordinate

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Status: **FROZEN BEFORE CORRECTED PREFLIGHT OR PAIRED-ACCEPTOR OUTCOMES.**

Frozen on 2026-07-29 after the v1 preflight exposed a donor-coordinate error, but before any classifier
score at the corrected donor coordinate and before any paired-acceptor corruption was evaluated.

This protocol inherits every cohort, checkpoint, intervention, threshold, multiplicity rule, behavior
gate, and representation-stage boundary from
[`PREREGISTRATION_NATIVE_CLASSIFIER.md`](PREREGISTRATION_NATIVE_CLASSIFIER.md), with exactly one
correction:

- v1 incorrectly read the donor class at `donor_pos`, the first intronic `G` of `GT`;
- v2 reads the donor class at `donor_pos-1`, the last exonic base immediately before `GT`.

This correction follows the upstream authors' public data validation code, which finds positions carrying
class `1` and checks their donor motif at `sequence[label+1:label+3]`. The acceptor class remains at
`next_acceptor_pos`, the first exonic base after `AG`, matching the authors' check at
`sequence[label-2:label]`.

Local `GT→CC` and `GT→AA` sensitivity interventions still alter the motif at
`[donor_pos, donor_pos+2)`. The other-donor specificity readout is likewise shifted from the other `GT`
motif start to one nucleotide before it. No thresholds are relaxed in response to the failed v1 values.

The immutable v1 failure remains part of the record and has no biological interpretation. Only a v2
preflight pass may authorize the frozen paired-acceptor target.
