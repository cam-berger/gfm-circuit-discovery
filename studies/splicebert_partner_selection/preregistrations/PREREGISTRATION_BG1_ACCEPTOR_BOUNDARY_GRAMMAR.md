# Pre-registration — BG1/BG2 acceptor-boundary grammar and payload route

> **Tier: T2 — provenance.** Frozen protocol; it does not supersede the claim ledger.

**Status: FROZEN; validation target outcomes not run or inspected.** Train is used for assay
preflight and precision calibration. Chr11/16/19 is a **verification** set because it has been used
previously in this repository, not a fresh confirmatory population. BG2 is conditional on a passing
BG1 verification result and uses a gene-disjoint role.

BG1 tests an eight-cell, AG-preserving synthetic panel at biological -3 and +1. BG2 asks whether the
corresponding edited-A value changes can produce donor-side lift through the already established
L3H1 acceptor-window route. A failed or anomalous preflight stops target execution.

## 1. Biological motivation and interpretation boundary

U2AF1-centred 3'-splice-site recognition has experimentally resolved sequence organization around
the invariant AG. U2AF1 variants alter 3'-site sequence preferences in patients, cells, and
biochemical assays ([Ilagan et al., 2015](https://doi.org/10.1101/gr.181016.114)); structural and
affinity measurements resolve -3/+1 contacts and preferences
([Yoshida et al., 2020](https://doi.org/10.1038/s41467-020-18559-6)). These findings motivate the
frozen -3 T-over-A and +1 G-over-A contrasts; they do not identify a model component.

The tested parallel is **functional and positional only**. No result licenses a statement that
SpliceBERT implements U2AF1, binds RNA, performs spliceosomal recognition or catalysis, reproduces
protein biochemistry, predicts cellular splice usage, or contains a representation identifiable as
U2AF1. “Payload” remains a model-internal causal object, not U2AF1 or “splice-site strength.”

## 2. Frozen model, splits, roles, and supply

- **Checkpoint weights SHA-256:**
  `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`.
- **Base cohort:** the repository `CouplingTask` data and `eligible` filter: internal examples with a
  partner acceptor, native donor `GT`, and donor-to-partner span 60--150 nt. Native partner `AG`,
  canonical bases, coordinate bounds, and strand are additionally checked.
- **Train:** chr1--7, chr10, chr12--15, chr17--18, chr20--22, chrX, chrY. Train outcomes calibrate
  the assay and precision only; they cannot support the paper claim.
- **Verification:** chr11, chr16, chr19.
- **Chr8/9:** the optional hypothesis-specific replication flag is frozen **disabled**. Its
  motif-clean behaviour cohort cannot clear the target supply gate, so this protocol authorizes no
  test-split run. Any future use needs a new protocol and may never be described as sealed or fresh.

Sequences are already in transcript orientation. Let `d` be the masked donor and `a =
next_acceptor_pos` the first exonic nucleotide after the partner acceptor. The core is
`sequence[a-2:a] == "AG"`; biological -3, +1, -4, and +2 are array positions `a-3`, `a`, `a-4`, and
`a+1`, respectively. Strand does not change this model-array mapping.

Before outcomes are read, each gene is assigned by SHA-256 of
`"gfmcd-bg1-role-v1\0splicebert-boundary-grammar-bg1-v1\0" + gene_id` to fixed integer-weighted
roles **behaviour:route:reserve = 3:2:1**. The first eight digest bytes, big-endian modulo six, select
the cumulative bucket. A gene cannot cross roles on any split. Within each role and panel, one
eligible example per gene is selected by the frozen SHA-256 ordering over `(gene_id, example_id)`.
Canonical targets use **all** eligible selected genes (`n=None`), never an outcome-selected subset.
The train behaviour analysis uses exactly the first 384 eligible behaviour genes in that frozen
ordering, the route assay uses the first 24 route genes, and behaviour health/specificity controls use
the first 96 behaviour genes. These are fixed analysis sizes, not adaptable post-outcome minima.

Hard supply gates are:

| cohort | minimum |
|---|---:|
| train behaviour preflight | 384 genes, both strands |
| train route preflight | 24 genes, both strands |
| validation BG1 behaviour | 128 genes, at least 32 per strand |
| validation attenuation arm | 80 genes, at least 20 per strand |
| validation BG2 route | 96 genes, at least 24 per strand |

The canonical +1/+2 paired validation support is 57 genes and is therefore frozen
`PRESPECIFIED_DIAGNOSTIC_ONLY`. The -3/-4 arm may be claim-bearing only if it passes both its supply
gate and the train-derived MDE gate. Neither arm may be replaced.

## 3. Exact panels and positioned motif inventory

### 3.1 Eight-cell target

For every otherwise eligible example, construct all cells `s_i(x,y)` with
`x in {A,C,G,T}` at -3 and `y in {A,G}` at +1, changing only `a-3` and `a`. The core `AG` is fixed.
A gene enters common support only if all eight cells are unique, have native length and exact declared
alleles, differ from native only at the declared positions, and pass the motif invariant below.

For a sequence `s`, define the full positioned inventory

`I(s) = sorted[(j, s[j:j+2]) for j=0..len(s)-2 if s[j:j+2] in {AG,GT,GC}]`.

The comparison set is **the native sequence plus all eight constructed cells**. Eligibility requires
`I(native) == I(cell)` for every cell, in addition to the fixed core `AG`. Thus no edit may create,
destroy, or move any positioned `AG`, `GT`, or `GC`. Since only declared bases change, code may check
the affected dinucleotide registers plus the separately asserted core, but this must be exactly
equivalent to the full native-inclusive inventory.

### 3.2 Flanking panels

Two separate paired supports retain every nonedited native base:

- upstream: one-position -3 A/T and -4 A/T panels on the same behaviour genes;
- downstream: one-position +1 A/G and +2 A/G panels on the same behaviour genes.

Native and every cell in each two-cell panel must pass the same exact edit, core, and positioned
inventory checks. Supports and counts are reported separately and never pooled. Failure of supply,
motif hygiene, or power makes the arm diagnostic-only. There is no relaxed-inventory, other-base,
branchpoint, decoy-anchor, naturally occurring, or outcome-selected replacement.

## 4. BG1 readouts, estimands, and decision

All cells are scored at the same masked donor. With raw base logits `z_b`, the primary readout is

`M_lin = z_G - (z_A + z_C + z_T)/3`.

The repository bridge is `M_max = z_G - max(z_A,z_C,z_T)`. It is always reported as a **secondary
bridge** and is never a BG1 target-decision gate or BG2 authorization condition. There is no learned
probe, probability transform, accuracy threshold, or normalization.

For `m_i(x,y)=M_lin(s_i(x,y))`, the two co-primary gene-level contrasts are

`theta_-3 = mean_i{[m_i(T,A)+m_i(T,G)-m_i(A,A)-m_i(A,G)]/2}`

and

`theta_+1 = mean_i{(1/4) sum_x[m_i(x,G)-m_i(x,A)]}`.

Both directions have a smallest effect of scientific interest of **0.05 raw logit units**.

### 4.1 Frozen co-primary family

Inference uses exactly the two linear contrasts as one family:

1. studentized simultaneous **two-sided** 95% max-t intervals from 4,999 gene bootstraps, seed
   `20260809`; and
2. one-sided maximum-signflip add-one p-values from 9,999 gene-level Rademacher draws. For each draw,
   the family reference is the maximum of its two null contrast means, and
   `p_k=(1+#{null family max >= observed_k})/10,000`.

The machine verdict is `SUPPORTED_VERIFICATION` only if both simultaneous lower bounds exceed 0.05
and both max-family p-values are at most 0.05. `NOT_SUPPORTED` means at least one simultaneous upper
bound is at or below 0.05; all remaining cases are `INCONCLUSIVE`. The max-margin summaries cannot
change these verdicts.

### 4.2 Single inferential secondary

The only inferential secondary target-panel contrast is the declared T/A-by-G/A interaction

`gamma = [m(T,G)-m(T,A)] - [m(A,G)-m(A,A)]`.

It receives its own two-sided studentized 95% interval and two-sided exact add-one signflip p-value.
It cannot rescue a failed primary family. All eight cell means, the C and G -3 levels, the max-margin
versions, and any additional profile views are descriptive; there is no 13-contrast family. A
combinatorial-interaction statement requires this single secondary to pass its own two-sided test.

### 4.3 Attenuation

On separate paired supports define

`alpha_up = mean{[M(-3=T)-M(-3=A)]-[M(-4=T)-M(-4=A)]}`

and

`alpha_down = mean{[M(+1=G)-M(+1=A)]-[M(+2=G)-M(+2=A)]}`.

An enabled arm passes only if its simultaneous lower bound exceeds 0.05 and its one-sided add-one
p-value is at most 0.05 after Holm correction across all enabled attenuation arms. Supply and MDE
status are frozen before validation scoring. The +1/+2 arm is diagnostic-only because `n=57<80`;
the -3/-4 arm is claim-bearing only if its projected MDE also passes. Diagnostic arms are still
reported when technically runnable but license no attenuation wording.

## 5. Train preflight and precision gates

Validation is traversed before unblinding only for model-free cell construction, supply/strand
counts, memberships, and hashes. No validation model forward is allowed before train `PASS`.

### 5.1 Projected MDE and analysis calibration

For a train gene-level endpoint with standard deviation `s_train`, frozen validation size `N`, and
the studentized max-t critical value `c`, define

`projected_MDE80 = (c + z_0.80) * s_train/sqrt(N)`, where `z_0.80=0.841621...`.

The target family, each potentially claim-bearing attenuation arm, and the two route calibration
directions require projected MDE at most **0.05**. Route boundary means observed on train are precision
calibration only, not positive biological controls and not target evidence.

The frozen analysis calibration uses 200 worlds per configuration. At the global boundary
`(0.05,0.05)`, the probability that **either** endpoint clears its complete interval-plus-signflip gate
must be at most 0.075. For the conjunctive claim, both least-favourable union-null configurations are
also exercised: `(0.05,0.10)` and `(0.10,0.05)`. The larger probability that the complete joint verdict
passes across those configurations must be at most 0.075. A planted total effect of `(0.10,0.10)`—0.05
above the claim threshold in both endpoints—must pass the complete two-endpoint decision in at least
80% of worlds. Thresholds, critical values, validation `N`, and outcomes are recorded without adaptive
revision.

### 5.2 Behaviour assay controls

Global behaviour preflight requires:

- exact checkpoint/data/declaration/protocol hashes; role and split gene disjointness; supply gates;
- native-inclusive panel construction, donor `GT`, partner `AG`, coordinate, mask-token, strand, and
  exact linear/max arithmetic invariants;
- batch-size and reverse-order agreement within `1e-5`;
- a motif-clean partner `AG->AC` positive control whose gene-bootstrap `clean-corrupt` interval has
  linear lower bound above 0.05 and max-margin lower bound above zero; and
- matched nonpartner specificity, defined as partner effect minus the distance-matched nonpartner
  `AG->AC` effect, with linear interval lower bound above 0.05.

### 5.3 Route assay controls

On at least 24 train route genes with both strands, the route preflight exercises the actual L3H1
edited-address reconstruction path. It requires:

- exact value linearity and edited-head reconstruction errors at most `1e-4`;
- natural donor-only injection identity error at most `1e-4` for both margins;
- nonzero eager attention with row-sum error at most `1e-4`;
- batch/order and all construction/coordinate invariants; and
- a motif-clean `AG->AC` acceptor-window (`ACC4`) edited-address recovery positive control over doses
  `0,0.25,0.5,1.0`, monotone within tolerance `1e-5`, with full-dose mean linear lift at least 0.05.

The synthetic route MDE uses the exact BG2 estimands: -3 T-to-A is averaged within gene across +1=A/G,
and +1 G-to-A is averaged within gene across -3=A/C/G/T. These route directions are used only for the
projected-MDE calculation; their train means do not gate the hypothesis. Cross-gene and
signed-coordinate target controls are covered by unit/invariant tests and are not required to form
target-like sham distributions or false-positive families during preflight.

Any failed invariant, anomalous positive/specificity control, non-finite value, hash drift, inadequate
primary supply/resolution, or irreproducible cohort yields `INSTRUMENT_UNRESOLVED` and stops validation
target execution. There is no force override for a canonical result.

## 6. BG2 edited-A acceptor-window value recovery

BG2 requires a hash-bound BG1 validation artifact with `target.verdict ==
"SUPPORTED_VERIFICATION"`, its manifest, and a passing independent BG1 audit that exactly matches a
fresh recomputation; max-margin behaviour is not an authorization gate. It then uses **all** validation
route-role genes passing the eight-cell common support, with at least 96 total and 24 per strand. These
genes are disjoint from BG1 behaviour genes.

The six frozen directed cell pairs are:

- -3: source T and edited baseline A at each +1 background A/G, averaged within gene; and
- +1: source G and edited baseline A at each -3 background A/C/G/T, averaged within gene.

At the layer-3 input, let `r_s-r_e` be the source-minus-edited residual change over biological
`ACC4={-3,-2,-1,+1}` (array offsets `-3,-2,-1,0` around `a`). Holding the edited donor attention row
fixed,

`delta_V[j]=(r_s[j]-r_e[j]) @ (W_V^L3H1)^T`,

`o_e=sum_all_j A_e[d,j]V_e[j]`, and

`o_rec=o_e+sum_{j in ACC4}A_e[d,j]delta_V[j]`.

Only the L3H1 donor-token head context is replaced by `o_rec`; all other heads, tokens, attention
weights, and activations remain from the edited run. This is a route-transmission sufficiency estimand,
not an attention or representation association. A positive result establishes causal transmission
and sufficiency through this intervention; it does not by itself estimate a complete mediated share.

### 6.1 Primary lift and secondary ratio

For each direction, primary raw lift is

`lambda=M_lin(injected edited-A)-M_lin(natural edited-A)`.

The primary lower bound is the fifth percentile of 2,000 gene-bootstrap draws, hence a 95% one-sided
lower bound, and must exceed zero; the two-sided 2.5/97.5 percentiles are also reported. BG2 has no 0.05
route SESOI. Max-margin lift is a secondary bridge. The secondary recovery fraction is the max-margin
ratio of means

`R_max=mean(lambda_max)/mean[M_max(source)-M_max(edited A)]`,

with a gene-bootstrap ratio interval and explicit denominator. A nonpositive or near-zero denominator
(`<=1e-9`) makes the ratio null. Ratios and max margins never determine route support.

The transition order is `(-3,+1)`. The base bootstrap seed is `20260809`; seed offsets are frozen as
0/1 for primary lift, 10/11 for true-minus-cross, 20/21 for the max ratio, 30/31 for behaviour,
40/41 for max lift, 50/51 for cross lift, and 60/61 for the max denominator. Every bootstrap resamples
genes with replacement and retains the within-gene background average.

### 6.2 Cross-gene and direction controls

Every control retains target edited attention and downstream state. Its attention-weighted L3H1
donor-head delta is scaled per target to match the Euclidean norm of the true donor-head delta within
relative error `1e-3`; this is final donor-output norm matching.

- **Cross-gene:** one deterministic different-gene source per target, half-rolled within exact
  directed pair, strand, and coarse span bin (60--104 or 105--150 nt). The paired gene-bootstrap
  lower bound for `true linear lift - cross-gene linear lift` must exceed zero.
- **Direction:** exactly 19 signed hidden-coordinate permutations with seeds
  `2026080900,...,2026080918`, shared across the four window rows/examples before per-target norm
  matching. Each yields both transition means. For endpoint `k`,
  `p_k=(1+#{max(null_-3,null_+1)>=observed_k})/20`; this one-sided max-family add-one p-value must be
  at most 0.05.

The structured cross-gene control and signed-coordinate family are never pooled. Target-time
linearity, reconstruction, natural-injection identity, attention, norm-match, collision, membership,
and finite-value invariants must pass. Any anomaly yields `INSTRUMENT_UNRESOLVED` regardless of
endpoint values.

`BG2_ROUTE_SUPPORTED` requires, for **both** directions, raw-linear-lift lower bound above zero,
paired true-minus-cross-gene lower bound above zero, and signed-coordinate max-family `p<=0.05`.
Otherwise the result is `BG2_INCONCLUSIVE`. BG2 cannot rescue or redefine BG1.

## 7. Artifacts, audit, and claim language

Canonical runs write immutable JSON plus manifests:

- `results/splicebert_coupling/bg1_boundary_preflight_v1.json` and `.manifest.json`;
- `results/splicebert_coupling/bg1_boundary_grammar_val_v1.json` and `.manifest.json`;
- `results/splicebert_coupling/bg2_boundary_payload_recovery_val_v1.json` and `.manifest.json`.

Results record frozen memberships/hashes, declared cells or directed pairs, reported per-gene inputs,
the native sequences needed to independently reconstruct every synthetic cell and positioned motif
inventory, summary statistics, gates, seeds, and verdicts. Manifests bind every checkpoint/tokenizer
file, GTF, FASTA,
this protocol, declaration, runners, command, parameters, seeds, code revision/dirty state, and output
hashes. The independent model-free auditor writes a separate immutable `.audit.json`; it verifies the
manifest inventory, reconstructs all panels/motif inventories and claim statistics, enforces supply,
and reproduces deterministic cross-gene assignments. It does not replay checkpoint forwards or prove
the provenance of a signed-transform outcome from activation tensors; those intervention mechanics
remain runner invariants covered by the preflight and focused tests. An audit `PASS` is required before
any wording becomes claim-bearing. BG2 additionally binds and freshly recomputes the BG1 audit before
model loading. Full resampling null arrays need not be stored because they are deterministically
recomputed from the per-gene rows and frozen seeds.

If BG1 and BG2 pass, the strongest licensed wording is:

> On a frozen synthetic panel preserving the acceptor AG and every positioned AG/GT/GC motif,
> SpliceBERT's masked-donor prediction showed material -3 T-over-A and +1 G-over-A effects in the
> primary raw-logit margin. On gene-disjoint route examples, the corresponding edited-A
> acceptor-window value changes produced donor-side lift through L3H1 above a matched cross-gene
> control and final-output-norm-matched direction shams. Together with the previously established
> predominantly relative partner address, this supports a model circuit combining a relative
> positional address with acceptor-local, sequence-structured 3'-boundary information. The -3/+1
> organization is a functional positional parallel to U2AF1-centred 3'-site recognition; it does not
> imply that SpliceBERT implements U2AF1 or spliceosomal biochemistry.

If the secondary max-margin results are concordant, that bridge may be stated separately. If the
powered -3/-4 attenuation arm passes, one may add that the -3 effect is materially stronger than its
-4 counterpart. The +1/+2 arm is diagnostic-only and cannot support a +2 attenuation claim. The
interaction may be named only if its single frozen two-sided test passes.

If only BG1 passes, wording stops at AG-preserving causal effects on model-native donor prediction
and may not name L3H1 as their route. The scope remains one SpliceBERT checkpoint, synthetic
counterfactuals, short internal introns that fit the assay, and recorded genes. No result establishes
biological use, protein identity, organism-level splicing, cross-model generality, or natural effect
size.
