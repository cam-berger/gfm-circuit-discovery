# PREREGISTRATION — P3 closure: coding-context-enriched carrier use

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN v1 — 2026-07-21; no untouched model outcome has been read.** The package, independent
audit, development smoke, and clean-revision checks in §9 remain mandatory execution gates.

## 0. The one claim under test

> In the frozen SpliceBERT checkpoint, the learned period-3 position component is causally used in a
> coding-context-enriched, phase-aligned manner to support donor–acceptor coupling.

This is a model-mechanism claim. It does **not** establish why training created the carrier, that the
spliceosome senses reading frame, or that the same mechanism exists in another GFM. “Coding-context
enriched” is licensed only if the matched non-CDS exon contrast passes.

## 1. Why the intervention changed after development

Prior evidence remains external to the new family:

- C8: deleting bins 339–345 collapses coupling from about 0.946 to 0.027.
- C34: the carrier contribution is spatially distributed across the donor–acceptor span.
- C35: under carrier patching, coupling recovery is mediated mainly through L3H1 attention rather
  than carrier-dependent L3H1 values.

The outcome-aware insertion and fixed-length sequence-rotation diagnostics are preserved as failed or
confounded development routes. Strict sequence rotations had zero construction yield; relaxed rotations
changed local MaxEnt 5′ splice landscapes, and the apparent effect moved toward zero after nuisance
adjustment. They cannot support a phase claim.

The replacement intervention changes **no nucleotide or token ID**. Therefore sequence composition,
codons, stops, splice motifs, MaxEnt scores, coordinates, and window length are exact invariants.

## 2. Cohorts and exposure firewall

The gene is the inferential unit. One example and two blocks per context are frozen per gene.

- Excluded as spent: `chr8`, `chr9`, `chr11`, `chr16`, `chr19`.
- Stable role namespace: `p3-closure-v1`, weights development:confirmation:reserve = 2:2:1.
- Primary confirmation census: all **943** eligible `confirmation`-role CDS genes.
- Primary reserve: **457** eligible CDS genes; no result-dependent extension is allowed.
- Hidden-direction specificity subset: first **150** primary genes in the package’s predeclared stable
  order.
- Non-CDS boundary census: all **52** eligible untouched non-CDS-exon genes (32 confirmation + 20
  reserve), matched outcome-blind to 52 CDS genes; 15 pairs are within gene.

### CDS blocks

Each target block is 36 nt, lies wholly in a comprehensive-GENCODE CDS with one unambiguous codon
register, is clear of splice boundaries, and is source-safe for shifts through ±10. Two spatially
separated blocks are selected in the current exon. The within-gene control uses two annotation-clear
blocks in the upstream intron.

### Non-CDS exon blocks

Each target block is fully exonic in the same gene/strand and overlaps no comprehensive-GENCODE CDS.
The active BASIC substrate contains no eligible non-protein-coding genes, so this is called a
**non-CDS exon** control—not a general noncoding-gene cohort. Matching is exact on strand, takes
same-gene pairs first, then uses optimal pooled-SD-scaled assignment and deterministic balance pruning
over exon length, splice geometry, block geometry, and block GC. The frozen match retains all 52 pairs
with maximum absolute SMD 0.208 (<0.25).

No model output, native coupling, attention, or margin enters cohort construction or matching.

## 3. Nucleotide-invariant local carrier surgery

Let `P` be the learned 1026×512 position table. Mean-center `P`, take its row-wise rFFT, retain only
bins 339–345, and invert to obtain `C3`. For a selected token-coordinate block `B` and offset `r`,

`P'_x = P_x - C3_x + C3_(x+r)` for a left shift,

`P'_x = P_x - C3_x + C3_(x-r)` for a right shift,

for `x in B`; every position outside `B` is unchanged. Clean and acceptor-corrupted forwards receive
the same surgery.

Offsets are `r={2,3,4,5,6,7,8,9,10}` in both directions and both blocks. Because the selected carrier
is nearly period 3, `r={3,6,9}` are near-native while neighboring offsets are equally local
off-register interventions. For a scalar outcome `Y`,

`S(Y)=mean_k[Y_(3k) - (Y_(3k-1)+Y_(3k+1))/2], k={1,2,3}`.

Directions and blocks are averaged within gene before inference.

## 4. Readouts and estimands

### Primary behavior

Coupling is the established native model behavior

`C = donor_margin(clean) - donor_margin(partner_acceptor_destroyed)`.

The single primary estimand is

`Psi_C = S(C_CDS) - S(C_upstream_intron)`.

A positive value means in-register carrier phase preserves more coupling in CDS blocks than neighboring
off-register carrier phase, beyond the same-example intronic effect.

### Secondary coding-context boundary

For every frozen matched pair,

`Delta_context = Psi_C(CDS gene) - Psi_C(non-CDS-exon gene)`.

This is secondary and opens only after the primary material-positive gate. It licenses
“coding-context enriched” only if its simultaneous 95% CI excludes zero in the positive direction.

### Descriptive routing diagnostics

The site-excluded L3H1 period-3 phasor, modulo-class masses, dominant class, attention normalization,
and phase angle are recorded but are **not confirmatory endpoints**. Development showed that the
signed global phasor was small and sampling-unstable even though coupling was strong. C35 remains the
controlling causal routing evidence; this study does not manufacture a new attention claim.

## 5. P3-direction specificity

On the frozen 150-gene subset, repeat the one-cycle `{2,3,4}` contrast for the learned `C3` and 19
pre-seeded controls. Each control is a signed permutation of `C3`’s hidden axis. This orthogonal
transform preserves, exactly up to floating precision:

- Fourier support and period-3 spatial waveform;
- complete spatial Gram matrix and every row norm;
- block, phase, shift direction, and intervention RMS;
- nucleotide sequence and every annotation.

Only alignment with SpliceBERT’s learned hidden features changes. The target passes specificity iff:

1. its mean `Psi_C` exceeds all 19 controls (one-sided add-one rank `p=1/20=0.05`), and
2. the gene-paired target-minus-mean-control 95% CI excludes zero.

Controls are fixed by seeds `20260721..20260739`; none may be selected or dropped by outcome.

## 6. Instrument gates

All must pass before a biological verdict is written:

| Gate | Frozen threshold |
|---|---:|
| P3 reconstruction error | ≤1e-6 |
| Repeat-forward determinism | ≤1e-8 max absolute difference |
| Zero-shift no-op | ≤1e-8 max absolute difference |
| Full attention row | finite and within 1e-4 of 1 |
| Mechanical modulo recovery | off-register minus 3/6/9 RMS ≥0.04, both directions |
| Continuous MLM health | paired true-base-log-probability 95% lower CI > −0.05 nat |
| Hidden-direction control RMS mismatch | ≤1e-7 |
| Hidden-direction control L2-ratio error | ≤5e-6 |
| Substantive positive control | global P3 deletion leaves ≤10% of native coupling; native >0.05 |
| Context match | ≥30 pairs and max absolute SMD ≤0.25 |

The full-restore identity and carrier reconstruction are plumbing checks, not scientific positive
controls. The global deletion control is substantive and can fail.

Development calibration (outcome-aware, never confirmation):

- 61 CDS genes: `Psi_C=+0.207`, multiplicity-adjusted CI `[+0.041,+0.373]`.
- 30-gene direction-control subset: P3 `+0.286`; 19 controls `[-0.0049,+0.0075]`;
  paired P3-minus-control CI `[+0.012,+0.558]`.
- 33 balanced development context pairs: CDS-minus-non-CDS `+0.230`, adjusted CI
  `[+0.033,+0.427]`.
- Continuous MLM-health deltas: CDS `−0.021 [−0.031,−0.010]`, non-CDS
  `−0.006 [−0.013,+0.0002]`; both pass the −0.05 floor.

## 7. Power and inference

The primary has one endpoint and no primary multiplicity penalty. In development, gene SD was 0.541.
At `N=943`, the projected SE is 0.0176 and 95% half-width about 0.037; a 50% shrinkage of the
development mean still has projected lower bound about 0.067, above the 0.05-nat SESOI. The projected
80% detection MDE is about 0.052 nat. The census is fixed; no extension is allowed.

Use gene-first studentized bootstrap inference with 20,000 frozen-seed draws. Report the two-sided 95%
CI, one-sided material test against `tau_C=0.05`, sign fraction, gene SD, MDE, chromosome, strand,
annotated donor phase, leave-one-gene, and leave-one-chromosome sensitivity.

Classification uses the entire interval:

- **material positive:** lower CI > +0.05;
- **detectable, materiality unresolved:** lower CI > 0 but not >0.05;
- **bounded null:** interval wholly inside ±0.05;
- **inconclusive:** otherwise.

The context and hidden-direction families are gatekept by a material primary. Their intervals are
reported independently; neither can rescue a failed primary.

## 8. Frozen decision table

| Outcome | Licensed conclusion |
|---|---|
| Primary material; direction specificity passes; context CI >0 | Learned P3 direction causally supports donor–acceptor coupling in a coding-context-enriched, phase-aligned manner. |
| Primary material; direction specificity passes; context fails/inconclusive | Learned P3 direction causally supports phase-aligned coupling in CDS blocks; coding enrichment unresolved. |
| Primary material; direction specificity fails | A same-geometry local perturbation affects coupling; P3-direction specificity unresolved. |
| Primary detectable but lower CI ≤0.05 | P3 phase sensitivity detected; materiality unresolved. |
| Primary bounded null | No effect larger than ±0.05 in the frozen domain. |
| Any instrument gate fails | Instrument-limited; no biological interpretation. |

Never write “the model learned codon frame” from this experiment alone. A positive result is compatible
with coding-associated triplet statistics but does not identify training origin.

## 9. Freeze and execution requirements

Before the first untouched forward pass:

1. freeze one package containing exact CDS IDs, non-CDS IDs, matched pairs, control subset, seeds,
   thresholds, checkpoint/GTF/FASTA hashes, and every development-artifact hash;
2. independently audit package counts, role assignments, sequence hashes, disjointness, match balance,
   and exposure exclusions;
3. run the final runner on development IDs only and require all gates to pass;
4. commit all code and frozen artifacts; require a clean worktree and exact commit hash;
5. seal the confirmatory runner and independent analysis hashes.

The target runner is resumable and writes immutable shards. It may run the primary, specificity subset,
and matched context arm, but it must not analyze or print outcome summaries until every planned shard is
complete. Confirmation runs once. Any post-outcome correction is outcome-aware verification, not clean
confirmation.

## 10. Publication outputs

Required outputs are: frozen package and audit; immutable row shards and manifests; independent gene-level
analysis; per-gene/per-pair tables; control and health memorials; figure data for the 3/6/9 recovery curve,
P3-versus-direction-null distribution, and CDS-versus-non-CDS contrast; claim-ledger update; and an explicit
scope paragraph separating present-day model use from biological/training origin.
