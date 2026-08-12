# Positional-proxy modifiability — results index

> **Tier: T2 — provenance.** One development-stage **PILOT**. No claim ledger row is backed by
> anything in this directory, and none should be until a multi-seed run exists.

Machine-readable entry point: [`../MANIFEST.csv`](../MANIFEST.csv) / [`../MANIFEST.json`](../MANIFEST.json).
Filter the `artifact` column on `results/splicebert_positional_proxy_modifiability/`. Column
semantics and the four `status` values are defined in [`../RESULTS_INDEX.md`](../RESULTS_INDEX.md).

**This is the first tree in this repository whose artifacts were produced by *continued
pretraining* of SpliceBERT, and the first under this study's name to load a checkpoint at all.**
The study's `README.md` statement that "no SpliceBERT checkpoint is loaded anywhere in this study"
was true until 2026-08-12 and has been corrected; `PPM1` itself remains a model-free row, backed by
artifacts in `results/splicebert_positional_clock/`, and is untouched by anything here.

## Runs

| Run / result | Stage | Status | Artifact | Manifest | Claim IDs |
|---|---|---|---|---|---|
| Paired-lineage pilot v1 | pilot | `BUILT` | [`lineage_pilot_v1.json`](lineage_pilot_v1.json) | [`manifest`](lineage_pilot_v1.manifest.json) | — |

## Paired-lineage pilot v1

`scripts/splicebert_positional_proxy_modifiability/run_lineage_pilot.py`. Continued pretraining
from the base SpliceBERT.1024nt checkpoint (sha256 `2ad91428…0149d`, verified before load — a
mismatch is fatal) on two corpora that differ in exactly one integer's residue mod 3.

### Read this before reading a number

**The premise this study was opened on is falsified, and this run does not revive it.**
`DECISIONS.md` (2026-08-12, final) records that the learned period-3 carrier is *necessary* for the
annotated-phase preference `Ψ` but **does not encode which register is preferred** — relocating the
carrier does not relocate the preference. So no result here may be read as evidence that SpliceBERT
uses a trainable *positional surrogate* for a biological relation. The only question this run asks
is the weaker survivor: **does phase-decoupled continued pretraining change `Ψ` at all, by any
route?**

**ONE SEED PER ARM. This is not a training-process claim.** The replicate for a corpus intervention
is the **training seed**, not the evaluation gene, and no repeated-training variance exists anywhere
in this record (`pbc_v2_capacity.power_projection`'s `cross_lineage` block refuses to project it).
Every interval below is a resampling interval over **evaluation cells** for **one lineage pair**.
None of them covers training-seed variability, and none may be quoted as though it did. A second
seed per arm could move `ΔΨ` by an amount this design cannot bound.

### The stop rule, which ran before either arm was trained

The base checkpoint was evaluated first, on the same cells. It reproduced the recorded
base-checkpoint reading almost exactly, so the run was permitted to proceed:

| quantity | observed | recorded reference | tolerance |
|---|---|---|---|
| `Ψ` | **+0.25625** | +0.256 | ±0.06 |
| coupling mean | **+0.04918** | +0.049 | — |
| cells | **479** | 479 | ±80 |

Reference source: `studies/splicebert_positional_proxy_modifiability/DECISIONS.md`, 2026-08-12
(later) and (final). Had it missed, the runner would have written a `base_reproduction_failed`
artifact and trained nothing.

### Design

| | L1 | L2 |
|---|---|---|
| label | phase-**preserving** | phase-**decoupled** |
| paired-indel offset | 3 | 4 or 5 (per window) |
| donor-relative register shift | 0 | 1 or 2 |
| windows at each residue | 3440 | 1728 at shift 2, 1712 at shift 1 |

The operator is a **length-preserving paired indel**: delete `offset` bases deep in the *upstream*
intron, insert a tandem duplication of `offset` exonic bases at `donor_pos - 30`. Deleting upstream
rather than downstream is deliberate — a downstream deletion moves the donor and shrinks the
donor→acceptor span, so the arms would differ in span as well as in residue. As built,
`donor_pos`, `next_acceptor_pos`, the span, and the window length are **byte-preserved and identical
across arms**; the upstream exon gains exactly `offset` bases, which is the frame statement itself.

**Held identical across arms** — and verified identical in the artifact, not merely intended: the
source windows, the window order, the crop plan, the masking draws (keyed by seed, window index and
epoch, never by iteration order), the optimizer and schedule, the step count, and the batch
composition. Token budget **11,421,118 tokens per arm** and **1,709,259 masked positions per arm**,
equal to the digit. Corpus sha256s differ, so the operator demonstrably did something.

**Two limits of the operator, stated rather than buried.**

1. The final **30 exonic bases** before the donor, and the **whole downstream exon**, keep their
   native register; only the bulk of the upstream exon is shifted as a block. A model reading the
   upstream frame solely from those last 30 bases would see no manipulation at all.
2. The arms differ by **1–2 bases of edit size** as well as by the residue, which is unavoidable —
   the residue cannot change without changing the integer. At `k = 1` (offsets 3 vs 4/5) that
   relative size difference is at its largest; a confirmatory design should repeat at a larger `k`
   (e.g. 30 vs 31/32), where edit size is nearly matched and only the residue moves.

### Corpus and leakage control

3,440 windows from chr1–chr4 and chr10, drawn from 74,106 candidate windows across 1,623 source
genes. **chr8 and chr9 were never drawn from** (globally sealed); the runner refuses them by name.

| exclusion | windows |
|---|---|
| not assay-eligible (no intron-mate acceptor, non-GT donor, span outside 60–150) | 68,491 |
| upstream window prefix not wholly intronic | 1,511 |
| **evaluation-parent gene ID** (`normalize_gene_id`) | **143** |
| **near-identity k-mer hit** against a parent window (k=31, ≥1 shared 31-mer) | **339** |
| indel geometry does not fit | 182 |
| sealed chromosome | 0 |

All 186 evaluation-parent genes were excluded by version-normalised ID. **140 of them lie on a
training chromosome** (chr10) and were therefore genuinely at risk — the artifact records this as
`evaluation_genes_at_risk_on_training_chromosomes` precisely so that a "0 dropped" can never be
mistaken for a working screen. An earlier launch of this run drew from chr1–chr5, where *no*
evaluation gene was at risk and the gene-ID screen removed nothing; it was discarded and rerun.
chr11 (the coupling study's validation split) carries the other 46 parents and was left out of
training.

**Residual risk, unchanged:** the k-mer screen detects near-identity only. Distant paralogs diverged
past roughly one substitution per 31 bases are not flagged — see
`parent_exclusion.near_identity_hits`' derived sensitivity floor.

### Result

`Ψ = mean(coupling | δ = 0) − mean(coupling | δ ≠ 0)`, on 479 chimeric cells, readout
`CouplingTask.margin` at the masked donor with an `AG→AC` corruption at the acceptor.

| arm | `Ψ` | 95% resampling interval | coupling mean | masked MLM top-1 | period-3 band power |
|---|---|---|---|---|---|
| base | +0.25625 | [+0.19080, +0.32296] | +0.04918 | 0.4727 | 197,361.8 |
| **L1** phase-preserving | **+0.64208** | [+0.52120, +0.77147] | −0.13388 | 0.4736 | 198,760.6 |
| **L2** phase-decoupled | **+0.29748** | [+0.21372, +0.39477] | −0.00374 | 0.4739 | 198,592.5 |

**Primary endpoint `ΔΨ = Ψ_L1 − Ψ_L2` = +0.34460**, under three resampling schemes computed on the
same cluster draw (so the difference is paired):

| clustering | clusters | `ΔΨ` interval |
|---|---|---|
| donor-gene\|acceptor-gene pair | 479 | [+0.26960, +0.42276] |
| donor gene | 180 | [+0.25755, +0.44128] |
| acceptor gene | 180 | [+0.25929, +0.43347] |

**The pair key is degenerate and is reported first only for comparability with the base reading.**
Every donor parent is crossed with every acceptor parent of its rank exactly once, so
`donor_gene|acceptor_gene` is unique to a single cell and resampling it is an ordinary **cell**
bootstrap, not a gene-clustered one. The donor-gene and acceptor-gene schemes genuinely group (a
parent contributes three cells each) and are the ones to quote for effect existence. All three
exclude zero.

**Where the effect comes from.** Decomposed against base, only one arm moved:

| contrast | estimate | interval |
|---|---|---|
| L1 − base | +0.38584 | [+0.28780, +0.49349] |
| L2 − base | +0.04124 | **[−0.02766, +0.12238]** — includes zero |

So `ΔΨ` is carried by **L1 rising**, not by L2 falling. On this one lineage pair, the
phase-decoupled corpus left `Ψ` statistically indistinguishable from base, while the
phase-preserving corpus more than doubled it. Per-δ means make the mechanism of the shift visible:
L1 both raised the matched class (+0.242 → +0.349) *and* drove the mismatched classes sharply
negative (−0.016/−0.013 → −0.305/−0.282), i.e. it became markedly more discriminative; L2 moved far
less (+0.220, −0.084/−0.071).

### Three things this does NOT show

1. **Not a change in carrier AMPLITUDE — and that is all this readout can say.** Period-3 band power
   rose by +0.71% (L1) and +0.62% (L2): negligible, and *the same* in both arms, so the arms are not
   separated by a change in carrier strength. **This does not show the effect bypassed the
   carrier.** `period3_band_power` sums squared moduli, so a phase rotation of the band — this
   study's own carrier operator, `positional_spectrum.phase_rotate_band`, the one the relocation
   test used — leaves the number *exactly* unchanged (verified at relative `1e-16`; memorialised in
   `tests/test_lineage_pretraining.py::test_period3_band_power_is_blind_to_a_phase_rotation`). A
   phase-only change, or a change in how downstream heads read an unchanged carrier, would produce
   precisely the reading above. The route is **not determined** by this pilot; the phase-sensitive
   `Ω3` relocation readout is what could settle it, and it was not run on the trained arms. See
   `DECISIONS.md`, 2026-08-12 (same day, bounding the entry above).
2. **Not degradation.** Masked MLM top-1 at non-site positions is 0.4727 / 0.4736 / 0.4739 across
   base / L1 / L2 — flat. No arm broke the language model, so `R12`/`R13` (never read a mechanism
   from a destroyed behaviour) does not bite here. The probe is `CouplingTask.mlm_health`, which
   **masks** its probe positions; `CouplingScorer.mlm_health` does not mask and once returned 0.998
   for a model with 94 of 96 heads ablated.
3. **Not a measurement on the base instrument.** Both arms shifted the overall coupling mean
   negative (+0.049 → −0.134 and −0.004), i.e. destroying the partner acceptor now *raises* the
   donor margin on average. The trained checkpoints are no longer the instrument the base reading
   was calibrated on, and cross-checkpoint `Ψ` comparisons inherit that caveat.

### Relationship to PPM1

`PPM1` records that a standalone single-checkpoint `Ψ` panel is not constructible at the frozen
SESOI, and is explicitly **silent** on the paired `ΔΨ` endpoint, which it calls "pilot-dependent and
unresolved" pending a **seed-level** covariance pilot. **This run is not that pilot and does not
resolve it.** It has one seed per arm, so it supplies no estimate of the same-block L1/L2
correlation across training seeds that `PPM1`'s scope names (`r ≥ 0.621` at N=544, `r ≥ 0.494` at
N=726). `PPM1` is unchanged by this artifact.

### Reproduction

Wall clock **25.8 minutes** end to end on one MPS device: 736 s for L1 and 733 s for L2 (2,000
steps each, batch 8, AdamW lr 1e-4, 200 warmup steps then linear decay, grad-norm clip 1.0, ~5
epochs over 3,440 windows), the remainder in census streaming, corpus construction, three
evaluations and 3 × 2,000 bootstrap replicates.

Objective is the published SpliceBERT recipe (Chen et al. 2024): 15% masking, 80% `[MASK]` / 10%
random base / 10% unchanged-but-scored, variable-length windows from the ladder (256, 512, 768,
1024) with every crop constrained to contain the manipulated region. **The trainer is a
reimplementation** — the official SpliceBERT repository ships architecture only, no pretraining
code.

**Trained weights are not persisted.** Binaries are never committed here; both arms are reproducible
from the base checkpoint hash, the two corpus sha256s and the single seed (20260812), all recorded
in the artifact and its sibling manifest.
