# Preregistration v1: causal localization of native splice-classification context

> **Tier: T2 — provenance.** Frozen before any context-shuffle model outcome was
> evaluated. The cohort geometry and perturbation mechanics may be audited on
> annotations and strings before the preflight, but validation outcomes remain closed
> until train discovery has frozen the regions.

## 1. Question and evidence boundary

The released GTEx fine-tunes contain a native three-class token classifier. This study
asks:

> Which equal-width sequence regions contain information that the model uses for donor
> or acceptor classification, and at what encoder depth does a confirmed distal
> region's information become integrated into the classified target token?

The study is intentionally not a coupling assay. Its two independent behavioral
endpoints are:

- `donor margin = logit(donor) - logit(no site)` at `donor_pos - 1`, the authors'
  last-exonic-base donor label; and
- `acceptor margin = logit(acceptor) - logit(no site)` at `next_acceptor_pos`, the
  authors' first-exonic-base acceptor label.

For endpoint `t` and relative window `w`, the signed causal damage is

`D(t,w) = margin(clean,t) - margin(dinucleotide-shuffle(clean,w),t)`.

Positive damage means the intact order in that region supports the target class.
Negative damage is retained and reported as possible suppressive context; neither sign
is renamed as a biological mechanism without a later semantic intervention.

An input shuffle is causal for the model on the assayed sequence distribution, but is a
synthetic intervention. Attention, gradients, nucleotide enrichment, and layerwise
representational similarity may describe a behaviorally confirmed region but cannot
select it or establish biological identity.

## 2. Cohorts, checkpoints, and split discipline

- **Discovery:** published seed 2020 only, annotation-only training cohort, `n=128`,
  deterministic tag `gtex-native-context-localization-discovery-v1`.
- **Confirmation:** all five published seeds 2020–2024, untouched validation cohort,
  `n=200`, deterministic tag `gtex-native-context-localization-confirmation-v1`.
- **Depth trace:** the first `n=64` examples of that same frozen validation cohort; no
  separate outcome-conditioned example selection.
- **Unit of resampling/permutation:** gene. All examples and raw paired values remain in
  the artifacts.
- **Sequence domain:** the existing 1,024-nt annotation-derived coupling-study windows,
  but no coupling effect is used for selection or inclusion.

The sealed chr8/9 test set is not opened. Models are loaded and freed sequentially; no
checkpoint seed may be selected or headlined alone.

## 3. Coarse equal-width scan

All discovery windows are 32 nt, target-relative, non-overlapping, and present in every
eligible 1,024-nt sequence by construction.

**Donor endpoint, anchor `d = donor_pos`:**

`[-112,-80), [-80,-48), [-48,-16), [-16,+16), [+16,+48),`
`[+48,+80), [+80,+112), [+112,+144), [+144,+176)`.

**Acceptor endpoint, anchor `a = next_acceptor_pos`:**

`[-176,-144), [-144,-112), [-112,-80), [-80,-48), [-48,-16),`
`[-16,+16), [+16,+48)`.

The deliberately offset central bin contains the canonical splice dinucleotide away
from a shuffle endpoint. The donor scan spans proximal exon, donor, downstream intron,
and the full annotated 60–150-nt interval to the next acceptor. The acceptor scan spans
160 nt of upstream intron, the acceptor, and 48 nt of downstream exon. These are
sequence-defined axes, not claims that a bin is a branchpoint, polypyrimidine tract, or
enhancer.

Every window is perturbed with a deterministic Eulerian shuffle preserving:

1. length;
2. the complete directed dinucleotide multiset;
3. mono-nucleotide composition as a consequence;
4. the first and last nucleotide of the window; and
5. every base outside the window.

For a frozen `(example, task, window, draw)` key, 64 deterministic Eulerian candidates
are generated and the one with maximum Hamming distance is chosen, with lexical
tie-breaking. Candidate choice is outcome-blind. Draw 0 is the primary intervention;
draw 1 is an independently keyed ordering-robustness intervention in confirmation.
Near-identity perturbations are an instrument failure, not silently retained.

All 16 scan effects form one discovery multiplicity family. Exact add-one gene sign-flip
permutation p-values are Benjamini-Hochberg corrected across that complete family.

## 4. Train-only selection and frozen matched controls

The central `[-16,+16)` window for each endpoint is always carried forward as an assay
calibration; it is not evidence-selected.

Separately for each endpoint, at most one **distal candidate** is selected from windows
not overlapping `[-16,+16)`:

1. positive mean damage at least `0.05` logit units;
2. 95% gene-bootstrap interval wholly above zero;
3. BH `q <= 0.05` in the 16-window family; then
4. largest mean damage, with relative start coordinate as the tie-break.

If no distal window passes, distal localization for that endpoint stops and the bounded
failure is preserved.

For a selected distal candidate, its equal-width **matched sham** is frozen as the
same-endpoint, non-central, non-candidate window with smallest absolute train mean
damage, with relative start as tie-break. Because both candidate and sham were selected
on train, the candidate-minus-sham contrast is evaluated only on validation.

The candidate's two pre-specified 16-nt halves are frozen without inspecting subwindow
outcomes. They are a resolution assay, not a new scan. Both halves are always reported;
no half is selected after validation.

## 5. Hard preflight

The seed-2020 train preflight uses `n=16` disjointly tagged examples and must pass before
coarse discovery:

- **positive controls:** exact local `GT→CC` donor destruction and local `AG→CC`
  acceptor destruction each lower their own mean native margin by at least `5.0`, with
  at least 75% of examples in the expected direction;
- **negative control:** an identity input and an all-false residual patch reproduce the
  unhooked margin within `2e-5`;
- **sensitivity:** shuffling the motif-centered `[-8,+8)` window lowers each endpoint's
  mean margin by at least `1.0`;
- **resolution:** motif-centered 16-nt shuffle damage exceeds an adjacent equal-width
  window by at least `0.5` (donor sham `[+8,+24)`, acceptor sham `[-24,-8)`);
- **invariants:** every shuffle preserves the five quantities in section 3 and changes
  at least 25% of window positions;
- **batch invariance:** subset and full-batch margins agree within `2e-5`;
- **source-patch positive control:** replacing the corrupt input's shuffled-region
  embedding outputs with their clean values restores the clean target margin within
  `2e-5`; and
- **target-patch positive control:** replacing the corrupt final-layer target state
  with its clean value restores the clean target margin within `2e-5`.

Any failure freezes interpretation and triggers instrument diagnosis. Thresholds are
not relaxed after seeing the result.

## 6. Five-seed confirmation

For each endpoint, validation evaluates the central calibration window and, if
authorized, the frozen distal candidate, matched sham, and two 16-nt halves.

Primary confirmation uses draw 0. A distal region confirms only if:

1. candidate damage has a five-seed family-mean 95% gene-bootstrap interval above zero;
2. mean damage is at least `0.05`;
3. the signed effect is positive in all five seeds;
4. candidate-minus-matched-sham has a 95% interval above zero;
5. exact add-one gene sign-flip `q <= 0.05` after BH correction across every primary
   endpoint/condition and candidate-minus-sham contrast evaluated; and
6. independent draw 1 has the same sign in all five seeds and a family-mean interval
   above zero.

The two frozen halves localize resolution descriptively. Their contrast is corrected in
the same primary family, but a winning half is not promoted unless a future
preregistration independently confirms it.

## 7. Causal depth trace for confirmed distal regions

Only a confirmed distal 32-nt region is traced. For clean and draw-0 corrupt inputs,
clean activations are patched into the corrupt run at:

1. the **source-region tokens**, and separately
2. the **single classified target token**,

at the embedding output and after each of the six encoder layers. Recovery is

`R = margin(patched) - margin(corrupt)`,

reported in raw logit units and as the ratio of gene-mean recovery to gene-mean
behavioral damage. Per-example ratios are forbidden.

Interpretation is based on the paired profiles:

- source-region recovery at early depth establishes that the perturbed information is
  still recoverable locally;
- target-token recovery first becoming positive bounds when the region's influence has
  arrived at the classified position;
- declining source recovery accompanied by increasing target recovery supports
  transfer/integration, not a literal one-path routing claim; and
- final target-state recovery is an instrument ceiling, while final distal-source
  recovery should be near zero because the classifier is position-wise.

The earliest-arrival layer is the first layer whose target-patch family-mean 95%
interval is above zero and whose recovery is at least `0.05`, provided every later
layer through the final layer has the same positive sign. Source and target effects are
BH-corrected across the complete `confirmed region × endpoint × depth × patch-site`
family. No attention head, nucleotide grammar, or spliceosomal factor is named from
this trace alone.

## 8. Licensed claims and next experiment

A confirmed input region licenses:

> Preserving local dinucleotide statistics, nucleotide order in this relative region is
> causally required for part of the native donor/acceptor classifier margin on the
> assayed annotation-derived sequences.

A source/target depth profile can additionally bound when that information becomes
integrated into the classifier target state.

It does **not** establish a branchpoint, PPT, enhancer/silencer, RNA-binding protein,
spliceosome assembly step, natural-variant effect, or mediation through `L5H6`/the final
MLP. Those become separate contingent assays using motif/composition-preserving edits,
matched-size shams, natural-variant or MPSA validation, and component/path patching
after a region is causally confirmed.
