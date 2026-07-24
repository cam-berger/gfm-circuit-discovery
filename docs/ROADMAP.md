# Research roadmap

> **Tier: T1 — canonical.** Current, authoritative. Read this.

## North star

Build a transferable, evidence-gated practice for causal mechanistic discovery in genomic foundation
models, then use cross-model comparisons to separate recurring computational motifs from solutions tied
to one architecture, objective, corpus, or checkpoint.

## Current baseline

The SpliceBERT case now reaches behavior-level causal evidence for donor–acceptor coupling, an A0→A12
address/payload/delivery account, a period-3 routing carrier (C34/C35), and an audited Head–Context Atlas.
The structured core is real but not sparse-faithful: the computation has no mid-size faithfulness knee.
The separate multispecies ≈10.63 carrier stops at post-softmax attention.

## Immediate scientific tracks

### Track 1 — cross-context causal transduction

**Question:** does the donor-circuit information state generalize across genes and sequence contexts, or
does patching work only within a matched clean/corrupt example?

- Use Atlas tags to define single-pair, clustered, multi-pair, cassette/complex, and matched-control
  cohorts from GENCODE v50 comprehensive.
- Derive `U_info` or a candidate site-choice direction on discovery genes only.
- Transplant it into gene-disjoint targets while matching site type, distance, native margin/strength,
  coding status, sequence composition, and intervention norm.
- Require dose-ordered target-margin movement above within-example, matched-random-direction, scrambled
  source, and non-target-position controls.
- Use two-way/dyadic inference whenever sources or targets are reused.

**Exit condition:** an independently derived direction predictably moves the intended native output in a
new gene/context with qualified health and specificity controls—or an instrument-limited/null result is
reported at a licensed detection floor.

### Track 2 — payload semantics and site-choice steering

**Question:** what acceptor-local latent is carried, and can it be used to steer which site a donor favors?

- Treat MaxEnt, motif scores, attention, attribution, and SVD directions as candidate generators only.
- Learn candidate directions on discovery genes using paired site-choice contrasts, not absolute strength
  labels alone.
- Separate existence, specificity, and semantics: first show that a direction causally changes a target
  margin; then test whether the change is partner-specific and whether external biological variables
  predict the dose response.
- Use matched competitor and decoy edits, gene-disjoint source/target splits, multi-draw stability, and
  two-way dyadic inference.

**Exit condition:** a direction causes predictable, dose-responsive reallocation between matched sites
without generic MLM damage, with its semantic label no stronger than the external validation supports.

## Open scientific problems

1. **Payload semantic identity:** established as acceptor-local and not bulk composition; absolute MaxEnt
   naming is withdrawn (R22).
2. **Distributed generation:** the no-knee faithfulness curve explains why a small injected core can be
   sufficient without being a self-contained generator; adaptive minimality remains open.
3. **Context generality:** Atlas components validate across matched cohorts, but cross-context state
   transduction remains untested.
4. **Period-3 origin:** present-day use is causal, and origin is now **corpus-attributed** (C39/C40,
   2026-07-23): period-3 is strong in the 72-species stage-1 sibling and absent (powered null) in the
   human-only sibling, and absent from single-nt transformer MLMs trained on ORF-free ncRNA — a
   multi-species, codon-linked origin. Only training *dynamics* (the trajectory) still need checkpoint
   series or controlled retraining. See [`CARRIER_ORIGIN_SYNTHESIS_2026-07-23.md`](CARRIER_ORIGIN_SYNTHESIS_2026-07-23.md).
5. **≈10.63 carrier:** its general downstream write and behavioral function remain open (C12/C37), but its
   **origin is now corpus-attributed** (C39): strong in 72-species, absent in human-only, and absent from
   the human genome's own dinucleotide spectrum (≈10.63 is not a human-sequence statistic; the genome's
   real short-range period is nucleosomal ≈10.298 nt). Multi-species training built it.

## Transfer milestone — second real GFM

Caduceus-PS is the prepared architectural contrast. Its v2 behavioral-transfer design integrates the
five-lens red-team; the CUDA/Colab package is locally dry-run validated but the model experiment is not
run. Before deployment:

1. confirm the frozen default forks (permuted-partner meaningfulness floor, distance-matched primary plus
   strength-caliper secondary, no PH companion);
2. pin a compatible CUDA/Torch/Mamba environment and pass the coordinate/long-range-channel preflight;
3. run behavioral qualification before any SSM mechanism screen;
4. if behavior qualifies, design architecture-native state-space localization and interventions rather
   than searching for SpliceBERT head analogues.

**Exit condition:** a second GFM reaches a justified point on the same evidence ladder, including an
honest architecture null if both local and long-range instruments are certified.

## Repository milestones

- **Done (2026-07-24):** the current record is consolidated on `master`; keep `master` as the default branch so the repository link exposes the current project.
- Keep `CLAIMS.md` compact and `CORRECTIONS.md` complete.
- Use [`../REPRODUCE.md`](../REPRODUCE.md) as the single environment/reproduction entry point.
- Preserve immutable artifacts and manifests; keep large reproducible shards outside Git.
