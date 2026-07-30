# DNABERT-2 Biological Reconstruction Atlas

> **Tier: T2 — provenance.** Study overview. `CLAIMS.md` controls scientific wording.

## Question

Across biologically annotated sequence elements, what information does DNABERT-2 use, and which
internal components causally support a qualified model behavior?

The study starts with behavior. Attention, gradients, and attribution nominate candidates only
after preflight. Necessity, sufficiency, faithfulness, and mediation require interventions with
matched nulls and untouched confirmation data.

## Results at a glance

### Baseline DNABERT-2

- The excess-reconstruction census tested 13 classes from GENCODE, EPDnew, PolyASite, ENCODE,
  and Mercer lariat-seq data. No class beat every empirical control in development. Five classes
  were adequately powered; eight remain underpowered. Confirmation stayed sealed.
- A separate geometry-qualified context-margin assay replicated proximal-context effects for
  nine classes: start codon, enhancer anchor, poly(A) cleavage and signal, branch point,
  polypyrimidine tract, and CEBPB, CTCF, and REST peak anchors.
- These are context-behavior effects. They do not identify motifs, biological function, or an
  internal circuit.

### Promoter-fine-tuned DNABERT-2

The promoter study uses `magicslabnu/DNABERT-2-finetuned-prom_300_all`. All EPDnew atlas windows
were excluded because they occur in the GUE fine-tuning source. The replacement cohort uses
GENCODE v50 MANE Select TSSs, same-locus composition-matched pseudo-TSSs, and exact
forward/reverse GUE exclusion.

| Result | Licensed conclusion | Boundary |
|---|---|---|
| Classifier behavior | The clean TSS-versus-pseudo-TSS margin replicated. | The target-preserving context effect did not replicate. |
| Head atlas | An eight-head `[CLS]` set passed joint necessity and sufficiency against layer-profile-matched nulls. | Only the set, not every member, received the full matched-null test. |
| Circuit battery | L11H2/L11H5 is a replicated late output-carrier pair. | The query-gating and early-to-late path models failed. |
| Faithfulness | The pair or eight heads alone retains about 14–15% of the contrast; eight heads plus all MLP writes retains about 61%. | The behavior depends on a distributed MLP scaffold, not a sparse head-only circuit. |
| Address/payload | The promoter margin depends on correct L11H2/L11H5 value-to-attention alignment. | Target-specific key-address dependence stayed below the practical threshold. |
| Source localization | Equal-dose masking nominated nominal +11 to +21 nt downstream of the TSS. | No regional late-pair V effect or independent CAGE, cCRE, CpG, or motif qualification replicated. |
| Classification impact | Masking the nominated region reduced AUROC in both splits and caused 7/231 real-promoter label flips. | Balanced accuracy did not change because control corrections offset target errors. |
| Native-base qualification | Exact mono- and linear-dinucleotide-preserving reordering failed the development positive gate. | The mask effect is not evidence that native sequence order is necessary. |
| All-node atlas | L11MLP ranked first in both splits; L11H2/L11H5 remained the leading heads. | No individual node passed the complete replicated accuracy gate because the L11MLP confirmation interval touched zero. |
| L11MLP low-rank structure | Four of 768 dimensions captured 89.7%/84.5% of the full class-margin effect and beat matched random subspaces in both splits. | This reused D10 cohorts post-hoc; the complement remained positive and hard-label intervals included zero. |
| TSS mask polarity (D12) | Masking +1:+31 costs more target-specific margin than masking −39:−9 (+0.3262, Holm 0.0096). | The bounded downstream domain and its distal edge were **refuted**; D8's tile 26 stands alone. |
| Arrangement vs composition (D14) | The polarity is **not** carried by nucleotide arrangement: a shuffle holding mono- and dinucleotide counts exactly fixed retains 12.9% (`mask − reorder` +0.4172, Holm 0.0066). | Composition is **not** established as the carrier — P2 was declared underpowered before the run. |
| Region density (D15) | At equal dose, −9:+31 carries materially more of the decision than far-upstream windows (+0.3490, Holm 0.0058); per-token density ratio 4.13×. | Magnitude is skew-inflated (median +0.1464, below the SESOI) and the per-locus sign test fails (104/186, p=0.1234). Regional, not positional. |
| Attention vs density (D15) | The carrier heads L11H2/L11H5 attend at their ALiBi-forced positional prior (excess +0.046/+0.032) — attention mass does **not** track information density. | Descriptive; attention and behaviour were measured on different cohorts. Head reach does not predict causal effect (null, 4 tests). |
| Base vs fine-tuned (D16) | The carriers were **already** at the prior before fine-tuning (Pearson 0.951 across 144 heads); one head (L9H7) relocated onto window indices 199–247. | n=1 third-party checkpoint pair; `[CLS]`-row attention only; no causal claim. |
| L9H7 addressing (D17) | L9H7 addresses a **fixed window position, not TSS content**: 88–92% of the band mass is present in windows containing no TSS, and under rotation attention does not follow the content. | **Retracts D16's "TSS reader" naming.** D16's relocation measurement stands. |

The current circuit trace is therefore:

1. a distributed representation maintained by MLP writes;
2. late L11H2/L11H5 output transport with value-payload alignment dependence — but **with no
   demonstrated source**: those heads attend at their ALiBi-forced positional prior and do not
   track where the information actually is (D15);
3. a final L11MLP write whose dominant class-margin effect is carried in a rank-4 subspace;
4. the supervised classifier readout.

This ordering is a causal priority map, not a complete directed circuit. The nucleotide content of
the routed representation remains unresolved, and **no head in the model has been shown to
recognise TSS sequence content** — the one head fine-tuning moved (L9H7) turned out to address a
window position rather than a motif (D17), and contributes ~19× less than the carriers it did not
change.

## Evidence status

| Phase | Status |
|---|---|
| Collection and assay infrastructure | Implemented and preflighted |
| Baseline excess-reconstruction census | Stopped at the development gate |
| Baseline context-margin atlas | Nine classes replicated |
| Promoter classifier behavior | Clean margin replicated; context mechanism failed confirmation |
| Promoter head and circuit tests | Joint head set and late pair replicated |
| Promoter payload decomposition | Value-alignment dependence replicated |
| Promoter source localization | Mask-sensitive region replicated; biological and native-base qualification failed |
| Promoter all-node accuracy atlas | Completed; no single node fully qualified |
| Promoter L11MLP low-rank assay | Rank-4 causal carrier supported post-hoc on reused cohorts; fresh transfer remains open |
| Promoter TSS mask polarity (D12) | Polarity replicated; bounded domain and distal edge refuted |
| Arrangement vs composition (D14) | Arrangement refuted as the carrier; composition left open at declared-insufficient power |
| Region density and attention prior (D15) | Density confirmed; attention shown not to track it — descriptive, cross-cohort |
| Base vs fine-tuned attention (D16) | Carriers already at prior before fine-tuning; one head relocated |
| L9H7 addressing (D17) | Positional, not content — D16's "TSS reader" naming retracted |

Failed preflights and negative outcomes remain in the immutable artifacts and ledgers. They are
not rewritten as biological nulls. D13 was **retired unrun** and its preregistration is kept as a
design record; D16's "TSS reader" wording is **retracted** by D17 with the original text preserved
in place rather than deleted.

## What this study does not establish

Stated once, plainly, because every individual claim carries bounds but the *body of work* needs
its own stopping line.

1. **No biological promoter element or motif.** No tested span qualified against CAGE, ENCODE cCRE,
   CpG-island, or EPD motif annotation on confirmation. Nothing here identifies TATA, Inr, a GC box,
   or any other element.
2. **Composition is unresolved.** D14 refuted *arrangement* as the carrier of the D12 polarity but
   could not establish composition: P2 was declared underpowered before the run (MDE 0.448). D15's
   contrast is **regional**, not positional, and cannot separate position from the composition
   resident at that position.
3. **No source-to-component mediation.** D8's regional `L11H2/L11H5` value swap was null at all 30
   tiles (max +0.00124 against a +0.005 SESOI). No experiment links any behaviourally implicated
   sequence region to any internal component.
4. **D15–D17 are attention-only.** No OV decomposition, no mediation, no path patching. "Relay" is
   an inference from architecture, not a measurement: no relay position or layer was identified.
5. **The readout is uncharacterised.** Since the carriers' attention geometry is unchanged between
   checkpoints (D16), whatever makes this checkpoint a promoter classifier must live largely in the
   readout — OV matrices, MLPs, the classification head — which no experiment here has examined.
   This is the largest open gap.
6. **No native-sequence necessity.** D9 failed its development gate and is bounded to tile 26.
7. **n=1 checkpoint pair, third-party.** This project fine-tuned nothing. D16/D17 compare two
   published checkpoints with an unknown recipe, data order, and seed; nothing here speaks to what a
   different fine-tuning run would produce.
8. **No transcriptional function in cells.** Every claim is about a model.

## Measurements

- **Baseline reconstruction:** true-token log probability per nucleotide in the full BPE cover,
  compared separately with same-motif, composition/token-matched, same-sequence, and low-order
  nucleotide controls.
- **Context margin:** the true target versus geometry-identical Hamming-one alternatives after
  equal-dose proximal and distal context corruption.
- **Promoter behavior:** promoter logit minus non-promoter logit for a real TSS versus its
  paired pseudo-TSS.
- **Causal support:** reciprocal activation exchange, complement scrub, K/V permutation,
  equal-dose source masking, and matched-size or full-family null inference.
- **Accuracy atlas:** balanced paired-accuracy loss after exchanging one of 144 head contexts or
  12 MLP writes at `[CLS]`.
- **Low-rank structure:** class-margin, accuracy, and AUROC loss after exchanging only a fitted
  L11MLP write subspace, compared with full exchange, its complement, and matched random
  subspaces.
- **Regional density:** target-specific margin loss under equal-dose contiguous span masking —
  identical BPE token count and matched nucleotide cover in target and control — so that any
  artifact depending only on "k tokens were masked" cancels in a between-region difference.
- **Attention vs the architectural prior:** `[CLS]`-row attention reconstructed from the `Wqkv`
  projections and allocated across TSS-relative nucleotide bands, reported as **measured minus the
  ALiBi-only prior**. Raw band mass is not interpretable: symmetric ALiBi with `[CLS]` at window
  index 0 forces 49–89% of every head's mass into `far_upstream` regardless of content.
- **Positional vs content addressing:** circular rotation of the input window, which preserves
  nucleotide content exactly while moving its offset, paired with an in-place content shuffle as an
  out-of-distribution-severity-matched comparator.

Exact definitions, thresholds, correction families, and opening rules are in
[`PROTOCOL.md`](PROTOCOL.md) and [`preregistrations/PREREGISTRATION.md`](preregistrations/PREREGISTRATION.md).

## Repository map

| Path | Purpose |
|---|---|
| `src/gfmcd/adapters/dnabert2/` | Pinned model loading and hook interfaces |
| `src/gfmcd/reconstruction_atlas/` | BPE-aware assays, controls, interventions, inference, and artifact contracts |
| `scripts/dnabert2_atlas/` | Collection, preflight, execution, audit, and summarization commands |
| `configs/dnabert2_atlas/` | Versioned site catalogs and phase configs |
| `results/dnabert2_reconstruction_atlas/` | Immutable compact results and manifests |
| `studies/dnabert2_reconstruction_atlas/` | Protocol, ledgers, source records, and this overview |

Bulk collections, activations, candidate pools, and per-example shards remain ignored runtime
data. Every result-producing command requires an explicit new output path.

## Read next

1. [`CLAIMS.md`](CLAIMS.md) — authoritative claim wording and open hypotheses
2. [`RESULTS.md`](RESULTS.md) — run index and numerical summaries
3. [`PROTOCOL.md`](PROTOCOL.md) — measurements and phase gates
4. [`preregistrations/`](preregistrations/README.md) — the seven frozen protocols, D1–D17, with
   their amendments, retirements, and conformance caveat
5. [`DECISIONS.md`](DECISIONS.md) — design changes and failed controls
6. [`DATA_CATALOG.md`](DATA_CATALOG.md) — biological sources
7. [`archive/LEGACY_DNABERT2_STUDIES.md`](archive/LEGACY_DNABERT2_STUDIES.md) — historical branches
