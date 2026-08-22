# Acceptor-Recognition Circuit — exploratory extension to the SpliceBERT proposal

> **Status: exploratory, unledgered, not paper evidence.** This directory is not a
> `studies/` arc. It has no frozen preregistration, no claim ledger, and no row in
> `results/MANIFEST.csv`. Nothing here may be cited in the manuscript. See
> [`README.md`](../README.md) and [`RESULTS.md`](../RESULTS.md) at the repository root.

Traces the reciprocal, unmapped half of the SpliceBERT donor-circuit account:
how the model's masked-`AG` reconstruction depends on the 3′ context — the
branch point (BP) and polypyrimidine tract (PPT). Motivated by the donor
circuit's own result that the BP/PPT region contributes ~0 to the *donor*
payload, this asks where and how those 3′ signals are used at the acceptor.

Two metrics mirror the donor circuit's `m(x)`:

- **`m_acc`** — masked-AG logit margin: mask the acceptor's `AG` dinucleotide,
  score the model's confidence for the true base at each position against
  the other three nucleotides, sum the two per-position margins.
- **`Δppt`** — the drop in `m_acc` when the BP/PPT region is corrupted,
  against a matched control.

---

## Audit notice — 2026-08-22

An independent methods audit found defects that invalidate every number this
directory had previously published. **All result files and figures have been
deleted rather than reprinted**, the pipeline has been repaired, and the
outstanding numbers must be regenerated before anything here is quoted again.
The withdrawn values remain recoverable from git history at `ba01eb6` — they are
gone from the working tree because they were misleading, not because they were
inconvenient.

**What was wrong**

1. **Nothing was reproducible.** Every corruption was seeded from
   `random.Random(hash(seq) & 0xFFFFFFFF)`. CPython salts `str` hashing per
   process, so every run drew different corruptions. The committed JSON, the
   committed figures, and the write-up were three different unrecoverable runs,
   and they disagreed with each other — including on two conclusions the prose
   reported as null whose committed CIs excluded zero.

2. **The "PPT" window destroyed part of the acceptor consensus.** `ppt_end` was
   `198`, so the window's last position was 197 — the −3 pyrimidine of the
   canonical `YAG` 3′ splice site, which is a pyrimidine in 94.4% of this
   dataset's acceptor rows. Every "the tract dominates the branch point" claim
   was confounded with destroying the 3′ss consensus itself. The donor arc's own
   result names that base as part of the effective acceptor window `yAG|r`.

3. **The branch-point locator scored RNA against DNA.** `_PYRIMIDINES` was
   `set("CU")` and the highest-weight term tested `== "U"`, applied to GUE
   sequences whose alphabet is `ACGT`. The 2.0-weight term never fired and the
   pyrimidine tests reduced to "is it C". Phase 3's branch-point null was a
   statement about a broken scorer.

4. **Phase 4 nominated heads by raw attention.** See
   [Phase 4](#phase-4--descriptive-attention-census-not-a-head-nomination) below.
   This is the failure mode of Jain & Wallace, *Attention is not Explanation*
   (NAACL 2019), and the donor arc contains its own counterexample: `L2H15`
   ranks 91st of 96 by attention excess and 2nd by ablation effect on the
   margin. Ranking by attention would have discarded half that circuit.

**What was repaired**

| Fix | Where |
|---|---|
| Deterministic per-sequence RNG, stable across processes and machines | `acceptors/data.py: sequence_rng` |
| Tract windows stop short of the `YAG` −3 pyrimidine | `configs/experiment.yaml: ppt_end`, `acceptors/data.py: YAG_MINUS3_GUARD, locate_tract` |
| Branch-point scorer reads DNA and RNA identically | `acceptors/data.py: _bp_score` |
| Corruption alphabet matches the input alphabet; tokenizer folding asserted, not assumed | `acceptors/data.py: BASES`, `acceptors/model.py` |
| Checkpoint pinned to a revision sha | `configs/model.yaml: revision` |
| Every result writes a sibling `*.manifest.json` with a content hash of the exact cohort | `acceptors/provenance.py`, all six runners |
| Length-matched BP-vs-tract arm, so region identity is not confounded with window size | `experiments/phase3_real_coordinates.py` |
| Regression tests for all of the above, model-free | `tests/test_reproducibility.py` |

**Still open, deliberately not changed here** — the `_bp_score` offsets are
`y(-2) U(-1) A(0) n(+1) y(+2)`, but `yUnAy` with the branch-point A at offset 0
places them one position over. Correcting that changes which adenosine the
locator selects, i.e. it changes the instrument, so it is a design decision for
the study owner rather than a mechanical fix. Every Phase 3 branch-point
statement is conditional on resolving it.

```bash
python -m pytest tests -q          # model-free; no checkpoint or network needed
```

---

## Data: GUE `splice_reconstructed`

Real human 3′/5′ splice-site windows from
[`leannmlindsey/GUE`](https://huggingface.co/datasets/leannmlindsey/GUE)
(config `splice_reconstructed`, itself GENCODE/DeepSplice-derived). Every row is
a 400 nt sequence with the splice-relevant dinucleotide fixed at index 200 by
construction:

| label | meaning | consensus |
|---|---|---|
| 0 | acceptor | `AG` at `[198, 200)` |
| 1 | donor (reserved for the shared-transport check) | `GT` at `[200, 202)` |
| 2 | non-splice negative | no consensus guaranteed at 200 |

Loaded directly via `pandas.read_csv` on the dataset's resolved CSV URL. The
measured alphabet is `A/C/G/T`.

**The dataset carries no gene, chromosome, or coordinate metadata** — only
`sequence` and `label`. Gene-clustered inference and gene-disjoint splits are
therefore impossible here, not merely absent, and the bootstrap resamples rows
i.i.d. Reproducing the sampling call also shows 19 of the 200 sampled "test"
acceptor windows appearing verbatim in the GUE train split, with 19 duplicate
rows inside test. The frozen model was not trained on GUE splits, so this is not
train-on-test contamination, but it does inflate effective *n* and breaks the
independence the intervals assume. This is the single largest structural
difference from the `studies/` arcs and the reason nothing here can be promoted
without re-siting it on GENCODE.

### Coordinates (`configs/experiment.yaml: coords`)

BP and PPT boundaries are **literature-typical positional approximations**
relative to the fixed AG at 198–200 (GUE carries no per-sequence branch-point
annotation):

- PPT: `[183, 197)` — ~4–17 nt upstream of the AG, stopping short of the −3
  pyrimidine of the `YAG` consensus
- BP: `[160, 183)` — ~18–40 nt upstream of the AG

### Corruption method

Region corruption is **content-destroying**, not composition-preserving: each
corrupted base is redrawn from a balanced ACGT distribution using a Markov
process that provably cannot introduce an `AG` substring anywhere in the region
*or at either seam* with the flanking real sequence (`acceptors/data.py:
random_no_ag`, `corrupt_region`).

The **random-AG** control destroys the entire window this way and forces `AG`
back in at the true position (bare AG, zero genuine context). The **decoy AG**
for the AG-scanning check is the first naturally-occurring `AG` strictly
downstream of the true site (search window `[201, 280)`).

Two asymmetries to keep in mind when reading the order-vs-composition contrast:
`corrupt_region` is constrained to contain no `AG` while `shuffle_region_in_place`
is not, so the two arms differ in more than composition; and the decoy-AG search
is exon-side only, with the true acceptor `AG` left unmasked and visible, so it
is a preference test rather than a scanning test.

## Setup

```bash
pip install -r requirements.txt
python experiments/phase0_pilot.py
```

CPU-only (`configs/model.yaml: device: auto`). Each runner writes its results
plus a sibling `*.manifest.json` recording the cohort content hash, the pinned
checkpoint revision, the seed, and the environment. `phase0_colab.ipynb` runs
Phase 0 standalone on a GPU Colab runtime for larger sample sizes.

---

## What each phase does

Results are withdrawn pending regeneration (see the audit notice). What follows
is what each phase *measures*, and at which rung of the evidence ladder — which
is the part that survived the audit and the part that determines what a rerun
would license.

### Phase 0 — the six-condition pilot

`experiments/phase0_pilot.py`. Six conditions on the same acceptor windows:
clean, BP-corrupted, PPT-corrupted, both, random-AG floor, and downstream decoy
AG, with a matched non-splice control arm. Establishes whether context is
required at all and whether the model prefers the correctly-positioned site.

### Phase 1 — control fix, and order versus composition

`experiments/phase1_control_fix.py`. Replaces the GUE `label==2` control — which
is not reliably intronic, so it never read as a clean zero — with a
far-intronic content-destroying control that serves as the honest noise floor.
Adds shuffle-in-place arms to separate a bulk-composition signal from an
order/identity one.

### Phase 2 — absolute `m_acc` against the random-AG floor

`experiments/phase2_absolute_floor.py`. A Δ alone cannot distinguish a real
interaction from a corruption that has already hit its floor. This re-analyses
Phase 0's per-sample absolute margins against the random-AG floor. No model run;
it derives entirely from Phase 0's CSV and records that file's hash in its
manifest.

### Phase 3 — per-sequence BP/tract localisation

`experiments/phase3_real_coordinates.py`. Locates a branch-point candidate per
sequence by consensus scan instead of a fixed offset, defines the tract between
it and the acceptor, and now also runs a **length-matched arm** — a tract window
of exactly the BP window's width — so that the BP-versus-tract contrast is not
confounded with how much sequence each condition destroys. Read the
`bp_vs_ppt_length_matched` row, not the raw BP-versus-PPT difference.

### Phase 4 — descriptive attention census, not a head nomination

`experiments/phase4_head_content_control.py`. **This phase ranks heads by raw
post-softmax attention mass. That is a descriptive census. It is not a causal
nomination and no head it surfaces may be described as a tract reader, a
branch-point reader, or a circuit component.**

What it measures, precisely: for a given (layer, head), how much of the
masked-AG query row's attention mass lands in a positional window, and whether
that mass changes when the window's content is destroyed or scrambled. A change
tells you the head's **QK circuit is sensitive to those tokens**. It tells you
nothing about whether the head's OV output moves `m_acc` — which is the only
question that matters for a mechanism, and which Phase 5 answers separately.

Three properties of the census that constrain how far it can be read:

- **The classifier is one-sided.** It counts a head as content-sensitive only
  when attention *drops* under corruption, and files every significant *rise*
  as a positional-locality artifact. Across the 84 interval tests in the last
  recorded run, significant rises outnumbered significant drops roughly 1.6 to
  1 — corruption perturbs nearly every head's attention in some direction, so
  "attention changed" has very little specificity.
- **No multiplicity correction is applied**, and every interval shares one
  bootstrap seed, so their Monte-Carlo errors are perfectly correlated.
- **Selection and testing happen on the same rows.** Heads are ranked top-k on
  a sample and then tested on that same sample, so the surviving effect sizes
  are winner's-curse inflated.

Whatever a rerun produces, the correct summary sentence is of the form *"these
heads' attention patterns are content-sensitive at this query position"*, and it
must be reported alongside Phase 5 rather than ahead of it.

### Phase 5 — the intervention ladder

`experiments/phase5_intervention_ladder.py`. Mean-ablation (necessity),
restoration patching (mediation), and the Q/V split (routing versus content),
via forward hooks on `SpliceBertSelfAttention`, numerically checked against a
self-patch no-op before running at scale.

Two limits are structural and no rerun of this script removes them:

- **The estimand is narrow.** The intervention overwrites a head's output at the
  two masked-AG token positions only — 2 of 402. A null licenses *"this head's
  direct AG-row output is not a large channel"*. It does not license *"this head
  is not causal"*, because a head can write useful information at another
  position for a later layer to read.
- **The coverage is inherited from Phase 4.** Only heads the attention census
  surfaced were ever ablated — 5 of 96 in the last recorded run. A head with low
  attention mass and high causal effect is structurally invisible to this
  design, which is exactly what `L2H15` is in the donor arc. **A null from this
  design therefore cannot support a "the computation is distributed" conclusion**;
  that conclusion needs the exhaustive sweep in next-steps item 1.

The last recorded run also had **no discrimination**: its designated negative
control produced the largest single-head ablation effect of any head tested, and
was the sole reason the joint test reached significance. Any rerun must report
the control alongside the targets and treat a control-indistinguishable target
as a non-result.

---

## What this directory does and does not license

**Does:** a behavioural question — whether the frozen model's masked-`AG`
reconstruction depends on upstream 3′ context, and whether that dependence is
order-driven or composition-driven. That question is well posed, the assay is
sound once the −3 confound is removed, and it is worth answering properly.

**Does not:** any statement about *which components* implement that dependence.
No head in this directory has ever been nominated by an effect on `m_acc` or by
an attribution method, so no head from here belongs in a figure, a claim, or a
sentence — in either direction. The positive reading (these heads read the
tract) and the negative reading (the computation is distributed across many
heads) are both unlicensed by the same defect.

## Known approximations and caveats

- **The branch-point locator is a hand-set consensus scan**, not branchpointer
  or LaBranchoR — those need either genomic coordinates GUE does not carry or a
  separate pretrained model. Its offsets are also unresolved (see the audit
  notice). Any branch-point null is a claim about this locator's candidates.
- **No gene identifiers exist**, so there is no clustering unit and no
  gene-disjoint split. See the data section.
- **The `label==2` non-splice control is superseded** by Phase 1's far-intronic
  control; it is not reliably intronic.
- **The decoy-AG comparison uses only the subset of samples with a downstream
  AG in the search window**, so its *n* is smaller than the cohort.
- **All phases share one fixed-seed sample**, so the phases are not independent
  replications of each other.
- **The checkpoint is `multimolecule/splicebert`**, pinned by revision. It is
  the SpliceBERT.1024nt architecture (1026 position embeddings, 6 layers, 16
  heads, hidden 512) but is *not* bound to the repository's audited weight file
  sha256 `2ad91428…0149d`. Do not carry numbers between the two without checking.

## Layout

```
configs/                 model.yaml (checkpoint/revision/device), experiment.yaml (data/coords/params)
acceptors/
  config.py              config loading
  data.py                GUE loading, corruption/shuffle, deterministic RNG, BP locator
  model.py               SpliceBERT wrapper, masked readouts, tokenizer assertion
  metrics.py             m_acc
  stats.py               bootstrap CIs
  attention.py           per-head attention-mass readout (Phase 4)
  patching.py            causal hooks: capture, replacement, mean-ablation (Phase 5)
  provenance.py          sibling *.manifest.json writer
experiments/             phase0..phase5 runners
visualization/           per-phase figure scripts (regenerate from results/)
tests/                   model-free regression tests for the audited defects
results/                 empty; regenerated results + manifests land here
phase0_colab.ipynb       standalone GPU-runnable copy of Phase 0
```

## Next steps

Ordered so that nothing depends on an unrepaired input.

1. **Regenerate every phase** under the repaired pipeline and check the
   manifests agree on the cohort hash. Until this is done there are no numbers.
2. **Nominate components causally, not by attention.** Rank all 96 heads with
   `gfmcd.circuits.attribution.head_attribution_scan` — one forward and one
   backward pass covers every head — then confirm the shortlist with an
   exhaustive 96-head mean-ablation sweep on `m_acc`, carrying over the design
   in `scripts/splicebert/run_ablation_scan.py`: a per-head distance-matched
   non-site specificity null, so a head must lose the *tract* effect
   specifically rather than degrade everything. This is the only route to a
   defensible statement about components, in either direction.
3. **Widen the intervention beyond the AG row** — MLP outputs and other token
   positions — and run a whole-layer scrub as an upper bound, so "explains X% of
   the effect" has a calibrated denominator.
4. **Resolve the `_bp_score` offsets**, then re-run Phase 3. Until then no
   branch-point conclusion is available at any sign.
5. **Re-site on GENCODE** if any of this is ever to be promoted: real branch-point
   annotations, gene identifiers, gene-clustered inference, and a gene-disjoint
   held-out split. GUE cannot support any of the four.
6. **Shared-transport question** — deferred until step 2 identifies something
   that actually mediates the effect.

**One-line status:** the behavioural question is live and worth answering; the
pipeline is now reproducible and the two confounds in the tract and locator
windows are fixed; every previously published number is withdrawn pending
regeneration; and no component-level claim exists here in either direction,
because none was ever tested against the margin.
