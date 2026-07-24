# Exploratory P3 shift diagnostic v1

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status:** frozen exploratory diagnostic, 2026-07-21. This is not a confirmatory study and cannot
identify the training origin of the period-3 carrier. It asks how the frozen SpliceBERT checkpoint
currently responds when sequence content, splice-site geometry, and coding frame are shifted relative
to its learned position table.

**Amendment A1 (after mechanical smoke, before target run):** the original `post_acceptor_sham` was
mislocalized: 30 nt after the partner acceptor lies inside the next coding exon, so its +1/+2 arms were
frameshifts rather than a noncoding edit control. The smoke artifact is retained and not interpreted.
The sham is replaced by `upstream_intron_sham`, 30 nt before the current exon's acceptor, which is
intronic in transcript orientation. It leaves the mature coding sequence unchanged while translating
the current donor and downstream partner together. Target-minus-sham is now the interpreted insertion
contrast. No target-run outcome had been read at this amendment.

**Amendment A2 (outcome-aware audit):** v1 incorrectly gated health at absolute top-1 accuracy above
chance. The established carrier-surgery standard is stricter: each deleted cell must retain at least
50% of its corresponding intact headroom over chance. The original eight-example health subsample put
one cell at 43%. The runner is corrected to the relative-headroom gate and v2 repeats the deterministic
matrix with health measured on all 24 examples and 40 probes per example. Because v1 outcomes were seen,
v2 is an outcome-aware deterministic verification of an exploratory diagnostic, not confirmation.

## Question

Does the period-3-dependent part of donor–acceptor coupling and L3H1 routing follow:

1. absolute token position;
2. donor-relative or acceptor-relative position;
3. splice-preserving intronic geometry; or
4. a coding-exon frame disruption?

The carrier itself has no single scalar phase. The attention readout is therefore the phase selected by
L3H1: its donor-query attention peak and its mass across the three key-offset residue classes.

## Cohort

- Official `SpliceBERT.1024nt` checkpoint.
- Unconditional validation examples, selected without model outcomes.
- Internal protein-coding exons with an unambiguous CDS phase, exon length sufficient for an internal
  edit, and donor-to-partner-acceptor distance sufficient for a deep-intronic edit.
- The same examples appear in every cell. Results are gene-clustered.

## Perturbations

For each example and shift `d in {0,1,2,3}`:

- `window_translation`: translate the complete natural 1024-nt genomic window by `d` positions using
  genuine upstream genomic flank. Donor and acceptor move together; their relative distance and local
  content are unchanged.
- `intron_insert`: insert `d` real nucleotides halfway between donor and partner acceptor. The donor is
  fixed, the acceptor moves by `d`, and the mature coding sequence is unchanged.
- `exon_insert`: insert `d` real nucleotides 30 nt upstream of the donor. Donor and acceptor move
  together; `d=1,2` disrupt downstream coding register and `d=3` preserves it.
- `upstream_intron_sham`: make the same insertion 30 nt before the current exon acceptor. It is removed
  by splicing and translates the measured donor and downstream partner together. This estimates generic
  noncoding insertion/truncation, absolute translation, and nonlinear edit-dose effects.

Synthetic insertions use a deterministic, locally composition-matched string chosen without model
outputs and forbidden from increasing local `AG`, `GT`, or `GC` motif counts. Sequence length remains
1024 by dropping the same number of far-right bases.

## Model conditions and readouts

Each cell is evaluated with:

- intact position embeddings;
- the exact C8 period-3 deletion, RFFT bins 339–345 removed.

Readouts:

1. native masked-donor margin;
2. donor–acceptor coupling: clean donor margin minus partner-acceptor-destroyed margin;
3. L3H1 donor-query attention mass to the four-token acceptor window;
4. maximum-attention key within the donor–acceptor span;
5. attention mass across exact absolute classes `x%3 = {0,1,2}` and key offsets modulo 3 relative to
   donor and acceptor;
6. masked-MLM health on non-site positions.

For a continuous outcome `Y`, define the within-example phase contrast

`S(Y) = mean(Y_0, Y_3) - mean(Y_1, Y_2)`.

This contrast is zero for a linear edit-dose trend. The carrier-specific contrast is

`S_intact(Y) - S_deleted(Y)`.

For synthetic insertion arms, the interpreted location-specific contrast subtracts the corresponding
carrier-specific `upstream_intron_sham` value. Raw arm values remain diagnostic.

All intervals are gene-block bootstrap intervals. They are descriptive because the cohort size and
decision thresholds were not powered and frozen in advance.

## Peak tracking

For every shifted example, report whether the L3H1 peak remains fixed in:

- absolute nucleotide position;
- absolute position residue modulo 3;
- donor-relative offset; or
- acceptor-relative offset.

Also report all three residue-class masses. A single argmax is not sufficient when comb teeth are tied.
The absolute convention is always `x%3`: index 0 is class 0, index 1 class 1, index 2 class 2, index 3
class 0 again.

## Preflight and stopping rules

- `d=0` is byte-identical in all four constructors.
- Every sequence is 1024 nt and contains the original canonical donor `GT` and partner acceptor `AG`
  at the updated coordinates.
- Site-local contexts remain byte-identical after coordinate adjustment.
- No synthetic insertion increases local `AG`, `GT`, or `GC` counts.
- Period-3 deletion plus the removed delta reconstructs the original position table to `<1e-6`.
- Attention tensors must be live and normalized; L3H1 peak must be in the nucleotide span.
- Every intact arm must remain above chance, and each carrier-deleted arm must retain at least 50% of
  its corresponding intact masked-MLM headroom over chance; otherwise it is instrument-limited.

## Interpretation boundary

A carrier-dependent translation effect supports present-day absolute-position sensitivity. A
carrier-dependent intronic effect supports present-day donor/acceptor offset sensitivity despite an
unchanged mature coding frame. An exon-specific effect exceeding the sham and intron effects motivates a
larger coding-frame study, but does not itself establish reading-frame semantics because the exonic
insertion is synthetic. Establishing why the carrier was learned still requires checkpoint trajectories,
controlled retraining, corpus/objective ablation, or cross-model evidence.
