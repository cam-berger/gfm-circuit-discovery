# Positional-proxy modifiability — results index

> **Tier: T2 — provenance.** Two development-stage pilots. No claim ledger row is backed by
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
| Three-arm carrier dependence v1 | engineering pilot | `BUILT` | [`lineage_carrier_dependence_v1.json`](lineage_carrier_dependence_v1.json) | [`manifest`](lineage_carrier_dependence_v1.manifest.json) | — |

**Read the carrier-dependence section before quoting anything from the paired-lineage pilot.** The
later run adds the native no-edit control arm the pilot lacked, and that arm reverses the pilot's
central reading: the rise in `Ψ` under continued pretraining is **generic continuation**, not
phase-preservation.

## Paired-lineage pilot v1

`scripts/splicebert_positional_proxy_modifiability/run_lineage_pilot.py`. Continued pretraining
from the base SpliceBERT.1024nt checkpoint (sha256 `2ad91428…0149d`, verified before load — a
mismatch is fatal) on two corpora that differ in exactly one integer's residue mod 3.

### Read this before reading a number

**The premise this study was opened on is not supported in the form proposed, and this run does not
revive it.** `DECISIONS.md` (2026-08-12, CORRECTION) records the corrected reading: the learned
period-3 carrier **band-specifically and disproportionately supports** the annotated-phase
preference `Ψ`, but **does not encode which register is preferred** — relocating the carrier does
not relocate the preference.

Two boundaries on that, both load-bearing. *Support* rather than *necessity*: the decisive
attenuation also cuts coupling MAGNITUDE to 34.8%, so the claim rests on band specificity (control
bands retain magnitude 107% and `Ψ` 111%) and on `Ψ` falling faster, never on the behaviour being
unharmed. And an unchanged position-embedding carrier rules against **rewriting the input carrier**,
not against **learning a different way to read it** — Q/K, attention, OV/MLP and content–position
interactions were never measured here.

So no result here may be read as evidence that SpliceBERT uses a trainable *positional surrogate*
for a biological relation; equally, none may be read as ruling one out. The only question this run
asks is the weaker survivor: **does phase-decoupled continued pretraining change `Ψ` at all, by any
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

---

## Three-arm carrier dependence v1

`scripts/splicebert_positional_proxy_modifiability/run_lineage_carrier_dependence.py`. An
**engineering pilot**, not a re-run of the pilot above. It exists to answer two questions the pilot
could not: *is the trained preference actually about phase?*, and *does it depend on the learned
period-3 carrier?*

### What changed relative to the paired-lineage pilot, and why

| | pilot v1 | this run |
|---|---|---|
| arms | L1, L2 | **L0 (native, no edit)**, L1, L2 |
| `k` in `offset = 3k + residue` | 1 (offsets 3 vs 4/5) | **10** (offsets 30 vs 31/32) |
| training seeds | 1 | **2** |
| carrier readout | band **power** (phase-blind) | **complex** band coefficients: per-bin amplitude *and* phase, per-bin and per-hidden-dimension energy |
| carrier-dependence endpoint | none | per-arm dose ladder **and** a difference of differences against power-matched control bands |
| health probes | masked MLM top-1 | masked MLM top-1, **coupling distribution**, **masked donor-G top-1**, **masked acceptor-G top-1**, at every dose |
| weights | not persisted | **persisted outside the repo**, hash-recorded, with per-cell primitives |

**`k = 10` reduces the edit-size confound; it does not remove it.** L1 inserts 30 bases per window
and L2 inserts 31 or 32, so the arms still differ by 1–2 bases of edit *size* as well as by residue
mod 3 — 75,930 against 79,707 inserted bases per corpus, a 5.0% mismatch, where `k = 1` gave 33–67%.
The residue cannot change without changing the integer, so no setting of `k` removes this.

### The stop rule, which ran before any arm was loaded

| quantity | observed | recorded reference | tolerance |
|---|---|---|---|
| `Ψ` | **+0.25625** | +0.256 | ±0.06 |
| coupling mean | **+0.04918** | +0.049 | — |
| cells | **479** | 479 | ±80 |

Identical to the pilot's base reading in every printed digit, so the instrument has not drifted.

### The base dose ladder reproduces the recorded base-checkpoint attenuation exactly

`DECISIONS.md` (2026-08-12, final) records a dose-graded attenuation run whose runner and artifact
were never committed. This run reproduces it from a committed runner, to the digit:

| condition | `Ψ` | `Ψ` retention | previously recorded |
|---|---|---|---|
| intact | +0.25625 | — | +0.25625 |
| P3 @ 0.75 | +0.15817 | 61.7% | 61.7% |
| P3 @ 0.50 | +0.02734 | 10.7% | 10.7% |
| P3 @ 0.25 | +0.00835 | 3.3% | 3.3% |
| P3 @ 0.00 | +0.00501 | 2.0% | 2.0% |
| control 95–101 @ 0.50 | +0.28496 | 111.2% | 111.2% |

Signed coupling retention also reproduces: 114.3% for P3 @ 0.50 (recorded 114%) and 76.7% for the
control at the same dose (recorded 76.7%).

**And it qualifies that record.** The recorded entry reads the 114% signed-mean ratio as the general
behaviour being "fully intact" and calls the necessity result established "without destroying
anything". The magnitude and motif readouts, which that run did not have, disagree:

| condition | `Ψ` ret | coupling **signed** ret | coupling **magnitude** ret | MLM ret | acceptor-AG top-1 |
|---|---|---|---|---|---|
| P3 @ 0.75 | 61.7% | 141.1% | 78.2% | 99.8% | 0.7307 |
| **P3 @ 0.50** | **10.7%** | **114.3%** | **34.8%** | 97.1% | **0.6159** |
| P3 @ 0.25 | 3.3% | 40.7% | 15.2% | 94.2% | 0.4823 |
| P3 @ 0.00 | 2.0% | 25.4% | 10.7% | 92.5% | 0.4154 |
| control1 @ 0.50 | 111.2% | 76.7% | 107.0% | 100.4% | 0.7578 |
| control2 @ 0.50 | 116.1% | 93.3% | 109.2% | 100.5% | 0.7307 |
| control3 @ 0.50 | 116.6% | 80.2% | 112.1% | 100.4% | 0.7474 |

Intact acceptor-AG top-1 is 0.7474. At the dose the record calls fully intact, the coupling
*distribution* has collapsed to about a third of its intact magnitude and the 3′ AG motif probe has
lost 13 points. The **band specificity still holds** — every control band retains `Ψ`, coupling
magnitude, MLM and the motif probe — so the damage is carrier-specific rather than generic. But
"without destroying anything" is too strong. The supportable statement is that period-3 attenuation
degrades `Ψ` **faster** than it degrades the general behaviour, band-specifically, with the general
behaviour measurably degraded as well.

### Result 1 — the rise under continued pretraining is GENERIC CONTINUATION

`Ψ` on the same 479 annotated-phase chimeric cells. Intervals are **evaluation-parent resampling
only** (donor-gene clustered, shared draws across every arm and dose so all contrasts are paired).

| arm | seed 20260812 | seed 20260813 |
|---|---|---|
| base | +0.25625 [+0.19327, +0.32182] | (same checkpoint) |
| **L0 native, no edit** | **+0.65616** [+0.52317, +0.80166] | **+0.73215** [+0.58131, +0.89119] |
| L1 phase-preserving | +0.57863 [+0.46520, +0.70041] | +0.60421 [+0.46663, +0.75517] |
| L2 phase-decoupled | +0.38725 [+0.29050, +0.49293] | +0.45628 [+0.33578, +0.58509] |

**`L0 > L1 > L2` in both seeds.** The native arm, which carries no edit at all, produces the largest
`Ψ`; `L1 − L0` is **negative and excludes zero** in both seeds (−0.07754 [−0.13710, −0.01859] and
−0.12794 [−0.20031, −0.06177] — seed 20260812's interval straddles the `0.05` SESOI so its
materiality is unresolved, seed 20260813's is material). So the pilot's central reading — that phase-preserving continuation
*strengthened* the boundary preference 2.5× — is **not supported once the generic-continuation
control exists**. Continued pretraining on these windows raises `Ψ` whatever the frame does, and the
phase-decoupled edit *reduces* that rise rather than the phase-preserving edit causing it.

The edit-matched contrast `L1 − L2` remains positive and resolved (+0.19138 [+0.13296, +0.24693] and
+0.14793 [+0.08688, +0.20957]), so residue mod 3 does move `Ψ` — but between two *edited* arms that
both sit below native, i.e. as differential suppression, not as reinforcement.

Across the two seeds these are a **range**, never an interval: `Ψ_L0` 0.656 to 0.732, `Ψ_L1` 0.579
to 0.604, `Ψ_L2` 0.387 to 0.456. Two draws bound nothing.

### Result 2 — training did NOT move the carrier, and this time that is MEASURED

The pilot could only report band power, which sums squared moduli and is therefore blind to a phase
rotation *and* to redistribution across bins or hidden dimensions. This run reports the complex
coefficients on rFFT bins 339–345 of the mean-centred 1026-row position table.

| comparison | band power vs base | max \|per-bin rotation\| | equivalent shift | min coherence | per-bin energy TV | per-dimension energy TV |
|---|---|---|---|---|---|---|
| seed 20260812 L0 | 100.737% | 0.569° | 0.0047 nt | 0.99874 | 0.0044 | 0.0020 |
| seed 20260812 L1 | 100.854% | 0.596° | 0.0049 nt | 0.99839 | 0.0038 | 0.0021 |
| seed 20260812 L2 | 100.630% | 0.576° | 0.0048 nt | 0.99883 | 0.0027 | 0.0020 |
| seed 20260813 L0 | 100.597% | 1.036° | 0.0086 nt | 0.99867 | 0.0032 | 0.0019 |
| seed 20260813 L1 | 100.658% | 0.850° | 0.0071 nt | 0.99859 | 0.0035 | 0.0021 |
| seed 20260813 L2 | 100.518% | 0.591° | 0.0049 nt | 0.99853 | 0.0032 | 0.0020 |
| **L1 vs L2**, seed 20260812 | — | **0.342°** | 0.0029 nt | 0.99902 | 0.0020 | 0.0017 |
| **L1 vs L2**, seed 20260813 | — | **0.291°** | 0.0024 nt | 0.99864 | 0.0023 | 0.0017 |

Every rotation is under **1.1 degrees**, equivalent to under **0.009 nucleotides** of translation at
the period-3 bin; coherence stays above 0.998, so what little changed *is* well described as a
rotation-and-scale rather than a restructuring; and the per-dimension energy profile is essentially
unmoved (cosine ≥ 0.99999, top-8 dimension overlap 1.000 everywhere). The arms are not separated by
carrier phase, by carrier amplitude, or by carrier redistribution.

Base per-bin amplitudes and phases, for reference: bin 341 amplitude 232.5 at −31.07°, bin 342
amplitude 266.6 at −123.64°, bin 343 amplitude 234.5 at +64.73° (read at the base table's dominant
hidden dimension for each bin, so the arms are read at the same coordinate).

### Result 3 — every checkpoint's `Ψ` depends on the carrier, the trained ones included

`Ψ` retention under period-3 attenuation, beside control-band retention at the same doses:

| checkpoint | P3 @0.75 | P3 @0.50 | P3 @0.25 | P3 @0 | control1 @0.50 | control2 @0.50 | control3 @0.50 |
|---|---|---|---|---|---|---|---|
| base | 61.7% | 10.7% | 3.3% | 2.0% | 111.2% | 116.1% | 116.6% |
| seed 20260812 L0 | 84.4% | 14.7% | −0.8% | −0.9% | 97.6% | 100.1% | 102.8% |
| seed 20260812 L1 | 81.2% | **15.8%** | 1.7% | 2.0% | 98.1% | 109.4% | 104.3% |
| seed 20260812 L2 | 83.3% | 16.1% | 0.7% | −0.0% | 101.9% | 107.5% | 104.0% |
| seed 20260813 L0 | 84.1% | 18.1% | 0.0% | 1.2% | 97.0% | 104.9% | 103.2% |
| seed 20260813 L1 | 85.3% | **17.8%** | −3.3% | −3.5% | 97.0% | 109.3% | 104.4% |
| seed 20260813 L2 | 83.2% | 19.5% | 2.5% | 2.2% | 99.6% | 109.1% | 106.3% |

**L1's elevated preference collapses under carrier attenuation just as the base's does.** Halving
the period-3 band leaves 15.8% and 17.8% of L1's `Ψ`, against 10.7% for base; power-matched control
bands at the same dose leave 97–110%. Training did not move the behaviour onto a different
substrate.

**R12/R13, applied rather than waved at.** Coupling magnitude retention at P3 @ 0.50 is 28–38%
across the trained arms while masked MLM retention is 97–99% and the acceptor-AG motif probe is
essentially unmoved (0.797–0.814 against intact 0.785–0.827); at the same dose the control bands
hold coupling magnitude at 98–109%. So the 0.50 dose is band-specifically damaging but leaves both
generic and motif competence intact, and is read. **The 0.25 and 0.00 doses are NOT read**: there the
acceptor-AG probe falls to 0.58–0.72 and coupling magnitude to 15–25%, which is a destroyed
behaviour, and `Ψ` retentions of "≈0%" at those doses are reported for completeness only.

Note that **signed** coupling retention is not interpretable on any trained arm: continued
pretraining drove the overall coupling mean negative (−0.083 to −0.137 against base +0.049), so a
signed ratio against that baseline can exceed 1 while the behaviour shrinks. The artifact flags this
per checkpoint with `signed_retention_interpretable`, which is `false` for all six trained arms and
`true` only for base.

### Result 4 — the primary endpoint: the TRAINING CONTRAST is carrier-dependent

A per-arm ladder cannot say whether the *training effect* needs the carrier. That is a difference of
differences, `[Ψ_X − Ψ_Y]_intact − [Ψ_X − Ψ_Y]_attenuated`, resampled on the same draws so all four
terms are paired:

| seed | pair | intact gap | band @0.50 | dosed gap | gap retention | DoD | 95% CI |
|---|---|---|---|---|---|---|---|
| 20260812 | L1−L2 | +0.19138 | **period3** | +0.02913 | **15.2%** | **+0.16225** | [+0.10292, +0.21858] |
| 20260812 | L1−L2 | +0.19138 | control1 | +0.17316 | 90.5% | +0.01822 | [−0.00000, +0.03778] |
| 20260812 | L1−L2 | +0.19138 | control2 | +0.21687 | 113.3% | −0.02549 | [−0.04751, −0.00301] |
| 20260812 | L1−L2 | +0.19138 | control3 | +0.20083 | 104.9% | −0.00945 | [−0.02765, +0.00970] |
| 20260813 | L1−L2 | +0.14793 | **period3** | +0.01840 | **12.4%** | **+0.12953** | [+0.07410, +0.18678] |
| 20260813 | L1−L2 | +0.14793 | control1 | +0.13141 | 88.8% | +0.01652 | [−0.00467, +0.03912] |
| 20260813 | L1−L2 | +0.14793 | control2 | +0.16254 | 109.9% | −0.01461 | [−0.03913, +0.01226] |
| 20260813 | L1−L2 | +0.14793 | control3 | +0.14597 | 98.7% | +0.00196 | [−0.01748, +0.02216] |

The same pattern holds for `L2 − L0`: gap retention 12.7% and 15.8% under period-3 at 0.50 against
89–101% under every control band. For `L1 − L0` the intact gap is small (−0.078, −0.128) and the
period-3 `DoD` intervals exclude zero (−0.073 [−0.133, −0.011] and −0.103 [−0.183, −0.020]) while
the control `DoD`s mostly cover it, but the gap is close enough to zero that its retention ratios
swing sign and should not be quoted as percentages.

At the milder 0.75 dose, where coupling magnitude is still 83–86%, the `L1 − L2` gap largely
**survives** (77.1% and 91.6% retained). So the dependence appears between the 0.75 and 0.50 rungs,
which is also where `Ψ` itself falls off a cliff.

### What this run does NOT show

1. **Not a training-process claim.** Two training seeds, one lineage triple each. Every interval
   covers **evaluation-parent resampling** and nothing else; the across-seed numbers are a **range**
   over two draws and bound nothing. This is *observed in two exploratory lineage triples*, not
   demonstrated.
2. **Not an equivalence anywhere.** Contrasts are classified against the frozen `0.05` SESOI, and a
   zero-covering interval that also covers material effects is written `unresolved`. No arm is
   reported as having "left `Ψ` at baseline" — at `k = 10`, `L2 − base` is in fact **+0.131
   [+0.052, +0.219]** and **+0.200 [+0.099, +0.305]**, i.e. resolved and material.
3. **`L1 − L0` is not edit-matched.** L0 carries no indel, so it conflates "phase-preserving edit"
   with "any edit". `L1 − L2` is the edit-matched contrast. Both are reported; neither substitutes
   for the other.
4. **The carrier statement is about *these* arms.** "Aggregate period-3 band power was similar" plus
   "per-bin phase moved by under 1.1°" plus "per-dimension energy is unmoved" is what was measured.
   That the arms differ in `Ψ` while the carrier does not move says the *route* is not a change in
   the carrier — it does not say the carrier is uninvolved, and Result 3 shows it plainly is.
5. **`Ω3` was not run.** It tests whether an *imposed* relocation moves behaviour, which is a
   different question from whether *training* moved the learned Fourier phase. No `Ω3` capability
   preflight was run here and no `Ω3` number is reported.
6. **Not the base instrument.** All six trained checkpoints shifted the overall coupling mean
   negative, so cross-checkpoint `Ψ` comparisons inherit that caveat.

### Bands, and how they were frozen

`positional_spectrum.select_power_matched_controls` on the **base** table, with the carrier and its
representable harmonic protected:

| band | bins | base power | ratio to period-3 | role |
|---|---|---|---|---|
| period3 | 339–345 | 197,361.8 | 1.000 | carrier |
| control1 | 95–101 | 199,441.4 | 1.011 | power-matched control |
| control2 | 19–25 | 201,205.1 | 1.019 | power-matched control |
| control3 | 42–48 | 177,408.0 | 0.899 | power-matched control |

Protected from selection: bins 339–345 (period 3.0) and 168–174 (period 6.0). **Period 1.5 maps to
bin 684, above the 513-bin Nyquist limit of a 1026-row table, so it has no rFFT bin at all** — it is
aliased away rather than protected, and the artifact says so rather than silently protecting
nothing. Two of the three controls (95–101, 19–25) are the same bands the earlier uncommitted base
run used; its third (96–102) overlaps 95–101 and is refused by the selector's non-overlap rule,
which chose 42–48 instead. The same bands are applied to every arm, so the intervention is identical
across arms.

### Corpus, edit accounting and composition controls

2,531 windows per arm from chr1–chr4 and chr10 across 1,365 source genes, drawn from 74,106
candidates; **chr8 and chr9 were never drawn from**. The `k = 10` geometry rejects 1,184 windows
that `k = 1` accepted (182), which is why the corpus is smaller than the pilot's 3,440.

| | L0 | L1 | L2 |
|---|---|---|---|
| edited windows | 0 | 2,531 | 2,531 |
| inserted bases | 0 | 75,930 | 79,707 |
| deleted bases | 0 | 75,930 | 79,707 |
| edit operations | 0 | 5,062 | 5,062 |
| offsets | — | 30 | 31–32 |
| 1-mer TV vs L0 | 0 | 0.00279 | 0.00290 |
| 3-mer TV vs L0 | 0 | 0.00617 | 0.00633 |

`L1` vs `L2` composition distance is **0.00011 (1-mer) and 0.00036 (3-mer)** — an order of magnitude
closer to each other than either is to native, which is the quantitative form of "edit-matched".
Total cropped bases are identical across all three arms, and with an identical batch plan and step
count the token budget is therefore matched. Figures are for seed 20260812; seed 20260813 differs
only in the last digits (79,712 inserted for L2).

Leakage control is unchanged from the pilot: all 186 evaluation-parent genes excluded by
version-normalised ID (143 windows dropped), plus a 31-mer near-identity screen against the parent
windows (246 dropped). 140 of the 186 parents lie on chr10, a training chromosome, so the gene-ID
screen was operating on a real risk rather than reporting a vacuous zero.

### Persistence and reproduction

**Trained weights and every per-cell array are persisted, outside this repository**, with sha256
recorded in the artifact's `persistence` block: six full state dicts, seven position tables, seven
per-condition primitive archives (per-cell coupling and both margin halves for all 17 conditions),
and the complex period-3 band coefficients per checkpoint. Every statistic above is rederivable from
those files without a forward pass. No binary is committed.

**Provenance note on the `training` block.** The artifact's `training` block reads
`reused_persisted_weights: true` for all six arms. The six arms were trained in earlier passes of
the same runner (2,000 steps each, batch 8, AdamW lr 1e-4, ~712–761 s per arm on one MPS device,
loss 1.09–1.15 → 1.04–1.11); two of those passes were evicted by the execution harness mid-run, and
the final assembly pass loaded the persisted state dicts rather than retraining. Reuse was verified
lossless: reloaded arms reproduce their training-pass `Ψ` to every printed digit. The step counts and
loss curves therefore live in the run logs rather than in the artifact, which is a real gap in the
record and is stated here rather than papered over.
