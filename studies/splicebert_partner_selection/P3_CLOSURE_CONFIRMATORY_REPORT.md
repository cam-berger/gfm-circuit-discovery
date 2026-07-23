# P3 closure — clean confirmatory report

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Result:** SpliceBERT causally uses its learned period-3 position direction in a phase-aligned manner
to support donor–acceptor coupling in CDS blocks. The matched experiment does not establish that this
use is enriched over non-CDS exons.

## What was tested

The intervention replaced only the learned P3 component of SpliceBERT's position embedding inside two
annotation-safe 36-nt blocks. For a shift `r`, the native P3 vector at position `x` was replaced by its
value at `x±r`. Every nucleotide, token ID, codon, splice motif, MaxEnt score, coordinate, and sequence
length stayed unchanged.

Offsets `{3,6,9}` preserve modulo-3 register. Their immediate neighbors `{2,4,5,7,8,10}` move the
carrier off register by one nucleotide. The established donor–acceptor coupling readout was measured
under both directions and both blocks. Identical surgery in an annotation-clear upstream-intron block
was subtracted within gene:

`Ψ_C = S(C_CDS) − S(C_upstream_intron)`.

The package froze 943 primary genes, 150 learned-direction-control genes, 52 matched CDS/non-CDS-exon
pairs, 19 hidden-axis control seeds, all thresholds, code/data/checkpoint hashes, and a non-runnable
457-gene reserve. No untouched outcome was read before package and smoke audits passed.

## Confirmatory results

| Question | Result | Frozen verdict |
|---|---:|---|
| Is phase-aligned P3 use material in CDS blocks beyond the intronic surgery? | **+0.141 [0.110,0.172] nats**, n=943 | material positive |
| Is the effect specific to the learned P3 hidden direction? | target **+0.135**; 19 controls **−0.0054..+0.0012**; paired excess **+0.136 [0.051,0.222]**; exact rank p=0.05 | pass |
| Is use enriched in CDS over matched non-CDS exons? | **+0.006 [−0.080,+0.093]**, n=52 pairs | inconclusive |

All 11 instrument gates passed. Primary MDE80 was 0.045 nats, below the 0.05-nat SESOI. The result is
not driven by one gene (leave-one-gene mean range 0.138–0.143) or one chromosome (leave-one-chromosome
means 0.135–0.145).

The effect is visually simple. Mean coupling under in-register CDS shifts is ≈0.757; adjacent
off-register shifts lower it to ≈0.614. Under the identical upstream-intron surgery, in-register
coupling is ≈0.757 and off-register coupling ≈0.755. Thus the primary contrast comes from local CDS
carrier alignment, not generic position-embedding damage.

## Why the direction controls matter

Each of the 19 controls is a signed permutation of the learned P3 component's hidden axis. These
controls retain its period-3 waveform, Fourier support, full spatial Gram matrix, every row norm, block,
direction, and perturbation RMS. Only alignment with the checkpoint's learned hidden features changes.
Their near-zero effects show that periodic geometry and intervention size alone do not explain C36.

## Licensed conclusion

> In the frozen SpliceBERT checkpoint, the learned P3 position direction is causally used in a
> phase-aligned manner to support donor–acceptor coupling in CDS blocks.

This closes the clean **present-day-use** gap left by C34's gate-qualified verification. Together with
C34/C35, it supports a distributed positional-routing role whose effect reaches coupling and is mediated
mainly through L3H1 attention under the prior carrier-patching intervention.

## Explicitly not established

- **Coding enrichment:** the matched boundary was inconclusive, not a null and not positive.
- **Reading-frame sensing:** modulo-3 sensitivity in CDS blocks does not show that SpliceBERT represents
  annotated transcript frame or that the spliceosome senses frame.
- **Training origin:** a frozen-checkpoint intervention identifies current use, not why the carrier was
  learned.
- **Biological mutation effect:** hidden position-embedding surgery is not a nucleotide mutation.
- **Other models or long introns:** the result is scoped to this checkpoint and frozen assay domain.

## Evidence and reproducibility

- Frozen protocol: `preregistrations/PREREGISTRATION_P3_CLOSURE.md`
- Pre-run red-team: `redteam/REDTEAM_SYNTHESIS_P3_CLOSURE.md`
- Package and audit: `results/phase-lattice/p3_phase_surgery_confirmation_package_v1*.json`
- Sealed development smoke and audit: `results/phase-lattice/p3_phase_surgery_confirmation_smoke_v1/`
- Immutable target shards (lossless archive), per-shard hash manifests, and independent analysis:
  `results/phase-lattice/p3_phase_surgery_confirmation_v1/`. The archive inventory and SHA-256 are in
  `confirmation_shards.archive.json`; extract it in place to reconstruct the three shard directories.

The one interrupted execution resumed from immutable shards: 943 primary and 109/150 control shards
were already complete; the runner hash-verified and skipped them, then finished the remaining controls
and context pairs. No aggregate outcome was opened until all three arm manifests were complete.
