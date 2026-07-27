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
5. **≈10.63 carrier — the donor-readout write is OPEN and NOT EVALUABLE as preregistered (R36).** B1 v2
   preregistered the OV endpoint as `S_ov` (the `U_info`-projected `suff_recovery`) plus the `Ω` subspace
   overlap, each against a covariance-matched null with a `PP-ovplant` gate. **None of it was computed**, and
   it cannot be computed within the protocol's scope: every per-head OV branch requires an **L0** reader —
   the red-team's must-fix #7, because deeper readers' value-restore absorbs upstream global-P propagation —
   and Preflight R found **no L0 head reaching Level A** (9 scanned, 0 passed; `R_1024` is L2/L3/L4 only).
   The **A-mixed branch** ("the carrier writes a carrier-specific component onto the donor readout subspace
   that is behaviourally redundant for coupling") is therefore **unexcluded and not excludable here** —
   protocol **Endpoint B**. Closing it needs either an owner-authorized amendment extending per-head OV
   claims to deep readers (which re-admits the confound must-fix #7 removed, weakening any positive) or a
   reader set with a qualifying L0 head. **Do not simply run `S_ov` on `R_1024`** — that computes the
   confounded quantity the design excluded on purpose. **Disposition (2026-07-25): NOT PURSUED this
   cycle** — recorded as a bounded limitation at Endpoint B, not an active track. Its general downstream write and behavioral function
   also remain open (C12/C37). Separately, its
   **origin is now corpus-attributed** (C39): strong in 72-species, absent in human-only, and **not
   detectable** in the human genome's own dinucleotide spectrum above that unanchored assay's 5–10%
   modulation floor (a sensitivity limit, not an established absence — R16/C15). The nucleosomal
   comparator, ≈10.298 nt, is measured in **mouse** ES-cell chromatin (Voong 2016, mm9), not in human;
   it sits 0.33 nt from the carrier, a powered refutation (R30). Multi-species training built it.
6. **No human dyad-anchored nucleosome measurement exists in this record (R30).** The ≈10.298 comparator
   that carries the powered refutation is measured in mouse ES cells (Voong 2016, mm9); every dyad script
   in the tree is mm9. The refutation does not *depend* on the mouse estimate transferring to human at
   better than 0.33 nt — 10.630 falls outside the 10.2–10.5 bp DNA-helical range recorded from the
   literature — but the cross-species step is an assumption, not a measurement, and it is the most likely
   line of attack on the project's flagship biological claim. Closing it needs an external human
   dyad-anchored map (e.g. a chemical-cleavage or MNase dyad call set on hg38), not new model compute.
   Until then, always name the organism (see `docs/NOMENCLATURE.md`, standing phrase SP2).
   **Disposition (2026-07-25): NOT PURSUED — accepted as a disclosed limitation.** A human MNase dyad set
   is obtainable now but MNase's AT-cleavage bias directly confounds a WW-periodicity readout, so it would
   need its own bias control before any number is reportable; no mature human *chemical-cleavage* dyad map
   with comparable provenance was located. The refutation does not depend on closing this.

## Transfer milestone — second real GFM

The active milestone is the DNABERT-2 Biological Reconstruction Atlas. It asks which annotated
elements show reconstruction advantage beyond matched motif/token/composition controls, then qualifies
source regions before internal screening. Its infrastructure is implemented, but the preflight is not
frozen or run and no biological claim is licensed. Before target deployment:

1. pin every admitted annotation release and pass coordinate/evidence/split audits;
2. freeze calibrated, site-class-specific behavior and precision thresholds;
3. pass positive, negative, sensitivity, resolution, token-geometry, balance, invariant, and health gates;
4. run the development census and seal candidate/threshold rules before confirmation;
5. localize and intervene internally only for behavior- and context-qualified site classes.

**Exit condition:** DNABERT-2 reaches a justified point on the same evidence ladder for one or more
site classes, including an honest qualified null or instrument limit.

Caduceus-PS remains the prepared architecture contrast. Its v2 behavioral-transfer design integrates
the five-lens red-team; the CUDA/Colab package is locally dry-run validated but the model experiment is not
run. Before any later deployment:

1. confirm the frozen default forks (permuted-partner meaningfulness floor, distance-matched primary plus
   strength-caliper secondary, no PH companion);
2. pin a compatible CUDA/Torch/Mamba environment and pass the coordinate/long-range-channel preflight;
3. run behavioral qualification before any SSM mechanism screen;
4. if behavior qualifies, design architecture-native state-space localization and interventions rather
   than searching for SpliceBERT head analogues.

**Exit condition:** Caduceus supplies a later architecture-native comparison, including an honest
architecture null if both local and long-range instruments are certified.

## Repository milestones

- **Done (2026-07-24):** the current record is consolidated on `master`; keep `master` as the default branch so the repository link exposes the current project.
- Keep `CLAIMS.md` compact and `CORRECTIONS.md` complete.
- Use [`../REPRODUCE.md`](../REPRODUCE.md) as the single environment/reproduction entry point.
- Preserve immutable artifacts and manifests; keep large reproducible shards outside Git.
