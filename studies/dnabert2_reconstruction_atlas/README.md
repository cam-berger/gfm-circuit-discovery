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

The current circuit trace is therefore:

1. a distributed representation maintained by MLP writes;
2. late L11H2/L11H5 output transport with value-payload alignment dependence;
3. a final L11MLP write whose dominant class-margin effect is carried in a rank-4 subspace;
4. the supervised classifier readout.

This ordering is a causal priority map, not a complete directed circuit. The nucleotide content
of the routed representation remains unresolved.

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

Failed preflights and negative outcomes remain in the immutable artifacts and ledgers. They are
not rewritten as biological nulls.

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

Exact definitions, thresholds, correction families, and opening rules are in
[`PROTOCOL.md`](PROTOCOL.md) and [`PREREGISTRATION.md`](PREREGISTRATION.md).

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
4. [`PREREGISTRATION.md`](PREREGISTRATION.md) — frozen amendments
5. [`DECISIONS.md`](DECISIONS.md) — design changes and failed controls
6. [`DATA_CATALOG.md`](DATA_CATALOG.md) — biological sources
7. [`archive/LEGACY_DNABERT2_STUDIES.md`](archive/LEGACY_DNABERT2_STUDIES.md) — historical branches
