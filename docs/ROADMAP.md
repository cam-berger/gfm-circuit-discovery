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
3. **Transfer specificity:** C38 establishes strong acceptor-side transfer with unresolved content
   specificity; C41 shows that the final donor residual is a generic readout axis. Earlier-layer,
   non-additive, and wider-context specificity remain open.
4. **Period-3 origin:** present-day use is causal, and C39/C40 associate the carrier with multi-species
   pretraining. Coding specificity and training dynamics remain unresolved, and the sibling comparison
   cannot isolate species diversity from corpus size, composition, sampling, or optimization history.
   See the live [`CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md).
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
   in the tree is mm9. It has been argued that the refutation does not *depend* on the mouse estimate
   transferring to human at better than 0.33 nt, because 10.630 falls outside the 10.2–10.5 bp
   DNA-helical range recorded from the literature. **That fallback is now contested**
   (`REVIEW_OPEN_QUESTIONS.md` B4: a random-sequence estimate of 10.6 ± 0.1 bp would place 10.630 inside
   the range), so treat the powered 0.33-nt comparison as the refutation and do not restate the range leg
   until B4 is ruled on. Either way the cross-species step is an assumption, not a measurement, and it is
   the most likely line of attack on the project's flagship biological claim. Closing it needs an
   external human dyad-anchored map (e.g. a chemical-cleavage or MNase dyad call set on hg38), not new
   model compute.
   Until then, always name the organism (see `docs/NOMENCLATURE.md`, standing phrase SP2).
   **Disposition (2026-07-25): NOT PURSUED — accepted as a disclosed limitation.** A human MNase dyad set
   is obtainable now but MNase's AT-cleavage bias directly confounds a WW-periodicity readout, so it would
   need its own bias control before any number is reportable; no mature human *chemical-cleavage* dyad map
   with comparable provenance was located. The refutation does not depend on closing this.

## Disposition — exon-edge compatibility (coding-boundary phase)

**Consolidated and archived, 2026-08-04.** The arc asked whether donor–acceptor coupling tracks codon
reading-frame phase across a joined boundary. It reached a **replicated behaviour** (C43) and a
**stopped mechanism attempt** (C44), and its remaining seven claim rows reached no evidence level this
project records. Bulk stayed on `codex/intron-exon-phase` at `fd26cc1b`; see R45.

What is genuinely open, in priority order:

1. **Does C43 survive outside the chimeric lattice?** Everything rests on constructs joining a donor
   module from one gene to an acceptor module from another at **one fixed 120-nt span**. A native-context
   assay — or even the same lattice at two more separations — is the cheapest thing that would tell us
   whether this is a property of the model's partner selection or of the construct family.
2. **Stop-avoidance versus phase.** Mismatched joins carry ~0.61 more in-frame stop codons than matched
   ones. The current diagnostics argue against that being the driver (`ρ = −0.093`; high-load blocks show
   a *lower* `Ψ`), but nothing yet **manipulates** stop load independently of phase, which is what would
   settle it.
3. **A capable mechanism instrument.** C44 **closes nothing** — the protocol closes its local family
   (learned-P3 carrier surgery, L3H1 A/V replacement, local L3H1 Q/K) only on a *powered negative*, and
   this is not one: four authorization gates failed and the target was never exposed. That family is
   attempted-and-unresolved. A successor needs a **newly preregistered capable** instrument; the failed
   authorization path may not be reopened or relaxed. Note the reserved 384-block target is the C43
   replication cohort, so it now carries a known behavioural covariate.
4. **E4's lead, not E4's claim.** A post-layer-2 acceptor plane decodes boundary phase at 0.894 held-out
   (0.973 exact-distance-matched), but its conditional-permutation specificity gate failed with 16/99
   nulls at or above target. A successor needs an exchangeable null with genuinely mixed strata and fresh
   outcomes. It is the one part of the arc whose audit is still re-runnable from history (1.96 MB, no
   genome, no checkpoint).

**Exit condition:** either C43 reproduces in a context that is not a fixed-geometry chimera, or the arc
is recorded as a construct-scoped behavioural curiosity. No mechanism claim is licensed until (3).

### Successor — positional boundary coherence (preregistered 2026-08-10, not executed)

[`studies/splicebert_positional_boundary_coherence/`](../studies/splicebert_positional_boundary_coherence/README.md)
is the sanctioned successor to (2) and (3). Its `PREREGISTRATION.md` freezes a three-phase design: a
model-free composition-balanced cohort on genes disjoint from every C43/C44/GFS25 role, a period-3-first
attenuation factorial with an independent dose preflight and capability gate per carrier, and an optional
register-relocation and route-mediation phase. Stop load is one of the prespecified balanced covariates,
which is what (2) asked for; the failed C44 authorization path is not reopened, which is what (3)
requires.

**It has produced nothing.** No cohort, no preflight, no target, no claim row — the design is frozen and
the executable target package is not. Treat this as a *planned* track, not progress against the exit
condition above. Two things it does **not** address: it keeps the fixed 120-nt chimeric geometry, so (1)
stays fully open, and it does not revisit E4's decoding lead (4). Its own strongest possible outcome is a
scoped model-mechanism statement — read its decision hierarchy before quoting any tier by name.

## Transfer milestone — second real GFM

**There is no active second-GFM milestone in this tree.** Two were prepared and both left the working
tree in the 2026-07-31 cleanup; each remains in Git history and is revivable as a design asset.

- **DNABERT-2 Biological Reconstruction Atlas** — reached a bounded stopping point rather than a
  finding. Its multisource preflight qualified and its 13-class development census ran, but no admitted
  class beat every empirical control, so confirmation stayed sealed and internal screening stopped
  before localization. Five classes were adequately powered for the development gate; eight retained
  explicitly underpowered estimates. It is stopped, not abandoned.
- **Caduceus-PS** — the prepared architecture contrast. Its v2 behavioral-transfer design integrates a
  five-lens red-team and its Colab package was locally dry-run validated, but **the model experiment was
  never run** and it needs CUDA.

The only cross-model work that produced live claims is the carrier-origin pair **C39/C40**
(`scripts/crossmodel/`, `results/origin-siblings/`, `results/origin-crossmodel/`), which is a
published-checkpoint natural experiment rather than a full second-model study.

**Exit condition for reviving either:** a second GFM reaches a justified point on the same evidence
ladder — including, and explicitly counting as success, an honest qualified null or instrument limit.
Read [`TRANSFER_GUIDE.md`](TRANSFER_GUIDE.md) first; it states what a second model must supply.

## Repository milestones

- **Done (2026-07-24):** the current record is consolidated on `master`; keep `master` as the default branch so the repository link exposes the current project.
- Keep `CLAIMS.md` compact and `CORRECTIONS.md` complete.
- Use [`../REPRODUCE.md`](../REPRODUCE.md) as the single environment/reproduction entry point.
- Preserve immutable artifacts and manifests; keep large reproducible shards outside Git.
