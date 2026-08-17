# DRAFT R5a — paired-donor address and donor-local read in masked-acceptor reconstruction

> **Status: DRAFT — NOT FROZEN.** The fail-closed runner and model-free census audit are
> implemented, but every seal pin remains the literal nonbinding `None`. This document therefore
> authorizes no model score, R3c outcome exposure, or chr8/9 access.

## 1. Question and evidence boundary

For the native **single masked acceptor-G margin**, does `L3H1` use an address selective for the
paired window **relative to the fixed pseudo-window and adjacent-window controls** to read the paired
upstream donor-local state in the four-token window

`W_D = d + [-1, 0, 1, 2]`,

and does `L2H15` causally construct the address rather than the value contribution?

The position claim is tested directly, not inferred from paired-window preference. R5a swaps the
**position-embedding assignment** between the paired and adjacent four-token windows while preserving
the two-window position multiset, separately swaps their word-embedding assignment, and crosses the
two shifts in an exact 2x2. It then bridges the position shift to the acceptor margin with the
complete L3H1 value tensor held numerically fixed. Passing the basic position path licenses only “sensitive
to the paired↔adjacent positional-embedding assignment,” not a general positional rule.

R5a can support that scoped model-internal address/read role. “Position-sensitive” and
“paired-window-selective” never mean selective among all possible source positions. It cannot establish an acceptor-side state,
downstream delivery, semantic identity, a biological function, or a biological mechanism. Attention
or QK measurements without a behavior-changing intervention remain candidate/decomposition evidence.
The `A @ V` identity and its four-cell interaction are never, by themselves, evidence that a head
“binds” two variables.

Every behavioral endpoint in R5a is the acceptor margin

`m_A = logit(G) - max(logit(A), logit(C), logit(T))`

at the single masked acceptor `G`. A masked-donor score, donor accuracy, or forward donor-coupling
endpoint is forbidden, including as a positive control.

## 2. Fixed-sequence entry and stopping rule

The programme is a fixed sequence: **R3c -> R5a -> R5b -> R5c**.

R5a may be frozen only after a frozen R3c acceptor-only confirmation has:

1. passed its complete assay preflight and familywise calibration;
2. shown partner-specific acceptor-direction necessity for `L3H1` under its frozen component rule;
3. retained `L2H15` as either a resolved acceptor necessity candidate or an explicitly
   estimand-dependent address candidate; and
4. bound the exact checkpoint, data, cohort, registry, runner, and statistical-core hashes.

Failure of item 2 stops the entire R5 programme. Failure of item 3 removes only the nested
`L2H15_ADDRESS_REUSE` outcome; it does not permit substituting another addresser after inspecting R3c.
R5b cannot open unless R5a returns `L3H1_DONOR_LOCAL_READER_SUPPORTED`. R5c cannot open through this
document. A failed memorial, plumbing/health control anomaly, missing cohort capacity, hash drift, or
classifier calibration failure returns `INSTRUMENT_UNRESOLVED` and stops before target interpretation.
A resolved active scientific comparator is classified under Section 10, not relabeled as an anomaly.

Before the **first** model score on the R5 mechanism-evaluation reservation, the complete R5a, R5b,
and R5c methods must all be frozen together: protocol text and hashes, runners, helper/statistical
core, component and path registries, rank/null construction, thresholds, seeds, cohort/construction
identity, and artifact schemas. Stagewise gating determines whether later frozen methods execute; it
does not permit R5b or R5c to be redesigned after an R5a evaluation result is seen.

The executable seal is one canonical `gfmcd.r5.stack_binding/1`, computed by
`gfmcd.splicing.r5_stack_binding`. It binds normalized copies of **all three** R5 runners (only literal
`FROZEN_*` assignment values are normalized), exact bytes of all three protocols, the dynamically
executed R3c runner, the recursively discovered repository-local Python import closure and package
initializers, the executable family/path/control registries, the schema registry, and the stable
Python/package runtime identity. Every R5 artifact serializes the identical payload as
`r5_stack_binding` and its canonical digest as `r5_stack_fingerprint_sha256`; each handoff recomputes
and compares both. A per-runner fingerprint is additional and cannot substitute for this joint seal.

## 3. Reserved cohort, fixed before any model score

R5a--R5c share one model-free census and two gene-disjoint reservations. Construction occurs before
loading the adapter and reads annotation/sequence only on validation chromosomes `chr11/chr16/chr19`.
It never fetches or materializes chr8/9.

1. Reconstruct and exclude the complete historical R3 gene set under its immutable identity.
2. Exclude every gene appearing in any R3c discovery, preflight, or confirmation cohort. Unused rows
   from a used gene remain excluded.
3. Apply the frozen R1c/R3c acceptor-only structural and motif-hygiene rules: canonical internal
   donor/acceptor pair, single acceptor-G readout, paired `GT -> AC` donor edit, matched pseudo donor,
   exact edit-size matching, and zero planted donor/acceptor motifs. The pseudo donor must be distinct
   from the paired donor and satisfy the literal **40-nt caliper**
   `abs(pseudo_donor_position - paired_donor_position) <= 40`. No native margin, activation,
   attention, or prior outcome may enter selection. A row is eligible only if the deterministic
   adjacent-window construction in Section 4 also succeeds; this check occurs before within-gene
   row choice.
4. Within each eligible gene, choose exactly one row: minimize the lexicographic tuple
   `(sha256("r5-acceptor-mechanism-v1|row|" + example_id).hexdigest(), example_id)`. Assign that
   gene by `int.from_bytes(sha256("r5-acceptor-mechanism-v1|bucket|" + gene_id).digest()[:8],`
   `"big") mod 2`: `0` is instrument preflight and `1` is mechanism evaluation. Within a bucket
   order genes by `(sha256("r5-acceptor-mechanism-v1|order|" + gene_id).hexdigest(), gene_id)`.
5. Fix the first **128 genes** in each ordered bucket, hence exactly 128 rows and 128 genes per
   reservation, and require both strands. Cohort and construction hashes cover the complete eligible
   census, ordered one-row-per-gene choices, exclusions, and every clean, paired-donor-corrupt, and
   pseudo-donor-corrupt sequence.

Before freeze, an immutable outcome-blind audit must bind both capacities and literal hashes here
and in the runner. Source identity is exact: GENCODE v50 **comprehensive** GTF, 124,527,720 bytes,
SHA-256 `83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453`, and the local
hg38 FASTA, 3,273,481,150 bytes, SHA-256
`5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51`. The v50 basic
GTF is ineligible even if it appears structurally compatible. The implemented development census
finds 6,064 input rows, 1,233 excluded genes
(`cc8c23308bfe0c39b70c77cd07d94945838cf03da513c8aa9550542425d0093f`), 366 eligible
rows/272 one-row-per-gene choices, and bucket capacities 142/130. The first-128 ordered row hashes
are `572289f8d2e8e6406e93596994b6bb5d3df04cd6571ff95bea85211ba4f358a9` and
`935d62284518b0ad72b398b2d7f41cb034015d63bb6617ca7e43a5b534ddbf95`; construction hashes are
`3143a893894de702476145589652f3157001218ec24c1742f2b1cedd0eebe0fd` and
`7de99171bd678dada527b9411c350166cdc3a512020b4bdd18f1e28af5c08312`.
Until those bytes receive review, the runner's future census pins remain deliberately
`None`: complete eligible-gene count `None`; bucket-0/bucket-1 eligible-gene counts `None`/`None`;
ordered selected-cohort hashes `None`/`None`; construction hashes `None`/`None`; and excluded-gene
registry hash `None`. These are not wildcards and cannot bind a run. If either reservation is short,
the design returns to DRAFT before model scoring; rows or genes may not be borrowed from R3c or
chr8/9. The preflight reservation sets tolerances and validates the exact decision path. Only a
fresh clean-worktree preflight with
literal protocol and instrument pins can bind the evaluation reservation.

There is no free exclusion-list input in the production chain. After authenticating the exact R3c
target artifact **and its clean sibling manifest**, R5a derives `gfmcd.r5.exclusion_registry/1`
directly from the R3c historical-exclusion list and both fixed R3c gene buckets. It writes that
registry as a fresh immutable sibling output of the R5a package; the R5a manifest binds its path,
byte size, and SHA-256, while the registry self-digest and its embedded R3c artifact/manifest hashes
bind the derivation. R5b must consume this output and reject an alternate user-supplied gene list.

## 4. Fixed components, positions, and activation states

- Reader: `L3H1`.
- Address-builder hypothesis: `L2H15`.
- Fixed negative head: `L0H2`.
- Readout query: the single masked acceptor-G token.
- Paired donor-local window: exactly `d + [-1,0,1,2]`; no adaptive expansion or motif-centered
  recentering.
- Controls: the R3c/R1c matched pseudo donor window and one deterministic four-token adjacent window.
  The frozen bank uses salt `r5-adjacent-bank-v2` and eight increasing four-token shells. For
  `s in {1,...,8}`, candidates are
  `W_up(s)=d+[-(4s+1),...,-(4s-2)]` and
  `W_down(s)=d+[(4s-1),...,(4s+2)]`; maximum absolute offset is 34 nt. The final hash bit orders
  upstream-first versus downstream-first within each shell. Search shells in increasing order and
  choose the first in-bounds candidate disjoint from the paired window, pseudo window, and acceptor
  mask and containing no `GT` or `AG` in its one-base-padded span. If the bank has no admissible
  candidate, the row is ineligible. The original two-window rule was rejected model-free because it
  left only 61/58 genes; the eight-shell bank yields 142/130. This remains a local reference bank,
  not exhaustive positional matching.

For every source leg, `+` is the clean paired-donor state and `-` is the same row after the frozen
donor corruption. Pseudo-source legs replace the matched pseudo donor with the identical two-base
operator while leaving the real donor intact. All captures are same-example and shape-aligned; a
minibatch roll or cross-gene activation source is forbidden.

Write the three sequence states explicitly as `x0` (native clean), `xD` (paired donor corrupted), and
`xP` (matched pseudo donor corrupted, real donor intact). For source leg `ell in {D,P}`, define
`x_ell+ = x0` and `x_ell- in {xD,xP}`. Every partner endpoint below is the `D` leg, every pseudo
endpoint is the `P` leg, and every specificity endpoint is the paired row difference `D-P`. No arm
may instead use `xD` as the pseudo recipient or compare activations captured from different examples.

## 5. R5a-POSITION — paired↔adjacent position-versus-content assignment

This family is the only R5a experiment that can promote “paired-window-selective” to the narrower
“sensitive to the paired↔adjacent positional-embedding assignment.” It adapts the frozen A1
word-versus-position component intervention to the acceptor direction. BERT forms its embedding from
separate word and position components before embedding LayerNorm. At the eight positions
`W_left union W_adjacent`, swap captured component outputs at matched offsets while leaving the
acceptor query untouched:

| cell | word component | position component |
|---|---|---|
| `full` | native | native |
| `word_shift` | left↔adjacent swap | native |
| `position_shift` | native | left↔adjacent swap |
| `joint_shift` | left↔adjacent swap | left↔adjacent swap |

The pairwise swap preserves the two-window multiset separately for each component. It is not a K/V
swap: the model reruns from the edited embedding component, so L3H1 QK and attention may respond. Run
the four cells for comparisons `paired_adjacent` and `pseudo_adjacent` and heads `L3H1` and `L0H2`.
Record three measures per cell:

1. `qk_preference`: mean pre-softmax logits to the left window minus the adjacent window;
2. `attention_preference`: mean post-softmax mass to the left window minus the adjacent window; and
3. `acceptor_margin`: reconstruct the chosen head's acceptor-query context as `A_cell @ V_full`,
   where the **entire clean V tensor is bit-identical in all four cells**, inject only that query/head
   context into one clean recipient, and pass the result through the downstream network and complete
   `model.cls` to the single acceptor-G margin.

For each measure derive exactly:

- `position_effect_at_native_content = full - position_shift`;
- `position_effect_at_shifted_content = word_shift - joint_shift`;
- `content_effect_at_native_position = full - word_shift`;
- `content_effect_at_shifted_position = position_shift - joint_shift`; and
- `interaction_position_x_content = full - word_shift - position_shift + joint_shift`.

The complete family contains exactly 60 ordered rows: two heads x two comparisons x three measures
x five quantities. Its canonical ID is
`POSITION|head=<L3H1,L0H2>|comparison=<paired_adjacent,pseudo_adjacent>|`
`measure=<qk_preference,attention_preference,acceptor_margin>|quantity=<literal order above>`.

Two nested paths are frozen. `P_POSITION_SENSITIVE_L3_READER` requires both position simple effects
material positive in all three L3H1 paired-adjacent measures, the corresponding L3H1
pseudo-adjacent position effects equivalent, and both position effects for both L0H2 comparisons
equivalent. Content may also matter, so content equivalence is **not** required for this basic
sensitivity claim. `P_POSITION_DOMINANT_L3_READER` additionally requires both L3H1 paired-adjacent
content simple effects equivalent in all three measures. All content and interaction rows remain in
the simultaneous family and are reported regardless of either decision.

Preflight requires exact word/position submodule capture; an involutive swap memorial; only the
declared eight component positions change; the acceptor query and input IDs remain fixed; recomputed
QK softmax matches checkpoint attention `<1e-5`; the fixed-V digest is identical across all cells;
`A@V` matches captured context `<1e-5`; full-cell query injection reproduces the native acceptor
margin `<1e-4`; and replaying captured final hidden state through the checkpoint's complete
`model.cls` reproduces every logit and the acceptor-G margin `<1e-5`. If hooks cannot expose these
separate components or satisfy these numerical identities, R5a returns `INSTRUMENT_UNRESOLVED` and
the position-sensitive wording stays closed.

## 6. R5a-QK — causal address intervention

Record the pre-softmax `L3H1` logits and post-softmax row from the masked acceptor query to `W_D`, the
pseudo window, and the adjacent window. These measurements are descriptive. The causal arm replaces
exactly four selected pre-softmax logits with their matched `-` values while holding the complete
`V` tensor fixed to `+`, then recomputes softmax and the downstream model. The intervention must leave
every unselected logit and every value vector bit-identical at capture precision.

The fixed arms, all one head by four logits, are:

1. `L3H1:paired_donor_window`;
2. `L3H1:pseudo_donor_window`;
3. `L3H1:adjacent_four_token_window`; and
4. `L0H2:paired_donor_window`.

For arm `(b,w)` and leg `ell`, capture the four selected logits from `x_ell-`, but run the recipient
on `x0`: replace only head `b`'s clean acceptor-query logits at window `w` by their same-row
`x_ell-` values, keep every value vector and all other logits from `x0`, and call its margin
`m_QK(b,w,ell)`. The raw effect is

`E_QK(b,w,ell) = m(x0) - m_QK(b,w,ell)`.

For each arm, `partner=E_QK(...,D)`, `pseudo=E_QK(...,P)`, and
`specific=E_QK(...,D)-E_QK(...,P)`. The complete QK family therefore has exactly 12 ordered rows
(four arms x three endpoints); no endpoint may be removed after preflight.
Its canonical row ID is
`QK|head=<head>|window=<paired,pseudo,adjacent>|endpoint=<partner,pseudo,specific>`; arms are ordered
exactly as the numbered registry and endpoints are ordered `partner,pseudo,specific`.

## 7. R5a-AV — donor-local value use under reconstructed L3H1 output

Use `acceptor_coupling_mechanism.av_four_cell_reconstruction`. For each leg `ell` and source window in
`{paired donor, pseudo donor, adjacent control}`, capture `A+`,`V+` from `x0` and `A_ell-`,`V_ell-`
from `x_ell-`. Keep values outside the selected four-token window at `V_ell-`; within the window
take `V+` or `V_ell-` according to the cell. `A+` and `A_ell-` are the complete L3H1
acceptor-query rows. Reconstruct and inject one L3H1 acceptor-query head output into the same
`x_ell-` recipient:

| cell | address | values | meaning |
|---|---|---|---|
| `baseline` | `A_ell-` | `V_ell-` | recipient identity anchor |
| `address_only` | `A+` | `V_ell-` | clean address with recipient values |
| `value_only` | `A_ell-` | window `V+`, elsewhere `V_ell-` | window value under fixed recipient address |
| `joint` | `A+` | window `V+`, elsewhere `V_ell-` | joint window reconstruction |

For cell `c`, record the raw lift
`L_AV(w,ell,c)=m(inject c into x_ell-) - m(x_ell-)`. For each window/leg the family contains the four
cell lifts plus the six outputs of `av_factorial_contrasts`: joint lift, both address simple effects,
both value simple effects, and `interaction_AxV`. It then contains the paired specificity difference
`L_AV(w,D,q)-L_AV(w,P,q)` for each of those ten quantities `q`. The complete A/V family has exactly
90 rows: three windows x `(10 partner + 10 pseudo + 10 specificity)`.
The interaction is reported but excluded from every claim gate: at head-context level it follows from
the bilinear architecture, and at behavior level it is not uniquely diagnostic of “binding.”
Because `A` is a full-row source state rather than a four-logit local edit, this family licenses
value use and whole-row address/value factorization only. The paired-window address-selectivity
claim rests on Section 6's exact four-logit QK surgery, not on an A/V cell label.
The canonical row ID is
`AV|window=<paired,pseudo,adjacent>|quantity=<quantity>|endpoint=<partner,pseudo,specific>`, where the
literal quantity order is `baseline,address_only,value_only,joint,joint_lift,`
`address_effect_at_value_minus,value_effect_at_address_minus,value_effect_at_address_plus,`
`address_effect_at_value_plus,interaction_AxV`. Windows and endpoints use their displayed order.

## 8. R5a-L2 — does L2H15 construct L3H1 address or values?

For builder `b`, leg `ell`, and each sequence `s in {x0,x_ell-}`, capture L3H1 `A_int(s),V_int(s)`
with `b` intact and `A_b-(s),V_b-(s)` under the same-example token-mean intervention on `b`. In the
builder-intervened recipient for that same sequence, inject the four L3H1 acceptor-query outputs
`(A_b-,V_b-)`, `(A_int,V_b-)`, `(A_b-,V_int)`, and `(A_int,V_int)`. Let
`Y_c(b,ell)=m_c(x0)-m_c(x_ell-)` and let `Y_b-(ell)` be the native builder-intervened coupling without
an L3 patch. The raw rescue cell is

`L_b(b,ell,c)=Y_c(b,ell)-Y_b-(ell)`.

The fixed builders are `L2H15` and size-matched negative `L0H2`; each uses exactly one all-token head
intervention plus one acceptor-query L3H1 output patch. For every builder/leg, the family contains the
four rescue cells and six factorial contrasts, then the ten paired `D-P` specificity contrasts. The
complete L2 family has exactly 60 rows: two builders x
`(10 partner + 10 pseudo + 10 specificity)`.
The canonical row ID is `L2|builder=<L2H15,L0H2>|quantity=<quantity>|endpoint=<partner,pseudo,specific>`,
using the literal ten-quantity order from Section 6. The address-role simple effect is fixed as
`address_effect_at_value_plus = joint-value_only`; the value-role simple effect is fixed as
`value_effect_at_address_plus = joint-address_only`. The other two simple effects remain
corrected/reportable rows but cannot be substituted into a gate. An address role requires the fixed
address simple effect to be material and the fixed value simple effect to be equivalent. A material
fixed value effect yields `MIXED_AV`, not an address-only claim. Noising/mean-ablation disagreement
inherited from R3c must be printed as
`ESTIMAND_DEPENDENT`; it cannot be averaged away.

## 9. Inference and matched-size controls

The unit of resampling/randomisation is the gene. All endpoints remain raw logit-margin units; ratios
are descriptive only. Use 4,000 whole-gene bootstrap/randomisation draws, exact add-one Monte Carlo
p-values, simultaneous intervals, SESOI `0.05`, seed 0, and at least 128 realized genes.

Across R5a--R5c, `alpha=0.05` is reused only by strict fixed-sequence gatekeeping: a later stage has no
testable claim unless the preceding stage's primary conjunction passes. Within R5a, the four
families below exhaust the stagewise alpha.

The four intervention-size families in Sections 5--8 are corrected separately at
`alpha = 0.05/4 = 0.0125`. They are not pooled because their intervention geometries differ. Within each,
every listed target and control row is corrected jointly. A target-minus-control contrast is included
in the same family whenever it enters a decision.

SESOI decisions use arbitrary-boundary inference, never a zero-null p-value:

1. `MATERIAL_POSITIVE` tests the one-sided closed-boundary null `effect <= +0.05` and requires the
   alternative strictly above `+0.05`;
2. `EQUIVALENT_WITHIN_SESOI` uses two one-sided tests (TOST) against `-0.05` and `+0.05`, with the
   alternative strictly inside the interval;
3. each declared mechanistic path is an intersection-union test whose path p-value is the maximum of
   its required boundary/TOST constituent p-values;
4. path p-values are Holm-adjusted across **every declared path in that intervention-size family**;
   and
5. as a strict second gate, every constituent's full-family simultaneous interval must cross the
   same required boundary (lower bound `>+0.05` for material-positive; lower bound `>-0.05` and upper
   bound `<+0.05` for equivalence). Touching any boundary is `UNRESOLVED`.

The exact add-one full-family zero-null max p-value is reported for directional calibration only; it
cannot decide materiality or equivalence. There is no global maximum-half-width or blanket
“resolution” gate: each endpoint and path is resolved only by its own arbitrary-boundary tests and
simultaneous interval.

The declared path registry is closed and uses the canonical row IDs above. Section 5 gives the exact
two-path position registry. In the remaining registry notation,
`QK(h,w,e)` expands to the exact string `QK|head=h|window=w|endpoint=e`, `AV(w,q,e)` to
`AV|window=w|quantity=q|endpoint=e`, and `L2(b,q,e)` to
`L2|builder=b|quantity=q|endpoint=e`. Let `E={partner,pseudo,specific}`,
`QJ=joint_lift`, `QV=value_effect_at_address_plus`, and `QA=address_effect_at_value_plus`. Set
expressions expand in the displayed order into literal IDs before hashing. The exact constituents are:

- `P_QK_L3_READER`: material-positive `QK(L3H1,paired,{partner,specific})`; equivalent
  `QK(L3H1,paired,pseudo)`; and equivalent `QK(h,w,E)` for arms
  `(h,w) in {(L3H1,pseudo),(L3H1,adjacent),(L0H2,paired)}`.
- `P_AV_DONOR_WINDOW_VALUE`: for both `q in {QJ,QV}`, material-positive
  `AV(paired,q,{partner,specific})`, equivalent `AV(paired,q,pseudo)`, and equivalent
  `AV(w,q,E)` for each `w in {pseudo,adjacent}`. The fixed value simple effect is therefore at
  clean address; neither the interaction nor the alternate simple effect can replace it.
- `P_L2H15_ADDRESS`: material-positive `L2(L2H15,q,{partner,specific})` for
  `q in {QA,QJ}`; equivalent `L2(L2H15,q,pseudo)` for `q in {QA,QJ}`; equivalent
  `L2(L2H15,QV,E)`; and all 30 `L2(L0H2,<literal-ten-quantity-order>,E)` rows equivalent.
- `P_L2H15_MIXED_AV`: material-positive `L2(L2H15,q,{partner,specific})` and equivalent
  `L2(L2H15,q,pseudo)` for every `q in {QA,QV,QJ}`, plus all 30
  `L2(L0H2,<literal-ten-quantity-order>,E)` rows equivalent.

The last two paths are the full builder Holm family even though their verdicts are mutually exclusive.
The compact notation expands to literal ordered row-ID lists in the frozen registry; the runner must
reject any missing, duplicate, reordered, or extra path constituent. All other family rows remain
jointly corrected and reported but cannot enter an undeclared follow-up path.

Calibration is explicitly a production-code engineering check, not a claim to have found a global
least-favourable configuration over every nuisance distribution. It executes exactly 27 coherent
primitive-algebra boundary signatures: position material, content lower/upper, pseudo-reference
lower/upper, and negative-head lower/upper; QK material, paired-pseudo lower/upper, and control-arm
lower/upper; A/V material, paired-pseudo lower/upper, and control-window lower/upper; L2 address
material, address-value lower/upper, address-pseudo lower/upper, negative-builder lower/upper, mixed
material, and mixed-pseudo lower/upper. Each face puts at least one declared conjunction edge exactly
at `+0.05`, `-0.05`, or `+0.05` as appropriate while keeping the other identifiable requirements on
coherent easy alternatives (`+0.40` for materiality and `0` for equivalence). Six all-alternative
plants cover the six literal paths. The generator starts from raw margin cells and rebuilds every
dependent endpoint, factorial contrast, boundary test, IUT, Holm decision, and simultaneous interval;
it never shifts a derived endpoint, p-value, or verdict. The ordered signature registry, primitive
arguments, assigned paths, and registry hash are serialized.

The engineering calibration tolerance is `epsilon_cal=0.01`. The frozen trial count is `N_cal=465`,
the smallest integer for which the exact one-sided 95% Clopper--Pearson upper bound at five
rejections is at most `alpha_family+epsilon_cal` and the exact one-sided 95% lower bound at 441
successes is at least `0.90`. Each boundary signature must have at most five false supports and the
corresponding upper bound must clear the tolerance; each path plant must have at least 441 successes
and its lower bound must clear `0.90`. Centered primitive noise has fixed standard deviation `0.025`;
one outer random sign per gene is shared across every primitive row before dependent quantities are
rebuilt. This check can detect gross classifier inflation or a broken decision path. It does **not**
establish scientific FWER. Formal nominal control comes only from the production arbitrary-boundary
tests, IUT logic, Holm/family allocation, and their sign-symmetry assumptions; none of those thresholds
is relaxed by calibration.

## 10. Mandatory preflight gates

All gates are conjunctive and failable:

1. exact checkpoint/config/tokenizer, GTF, FASTA, all three sealed R5 protocol/runner/helper/statistical-
   core hashes, code revision, command, environment, cohort, and construction hashes are present and
   match literal pins before any mechanism-evaluation score; the identical joint R5 stack payload and
   fingerprint also replay from the current source/runtime closure;
2. exactly 128 one-row-per-gene preflight rows, both strands, exact single-G mask, exact `GT -> AC`
   and pseudo edit
   size, coordinate invariants, and zero planted motifs pass;
3. native paired-donor acceptor coupling is material positive; pseudo coupling passes TOST and its
   simultaneous interval lies strictly inside `(-0.05,+0.05)`;
4. every attention row is finite, nonzero, and sums to one; the Section 5 component-shift,
   involution/change-mask, fixed-V, QK/attention, fixed-V margin bridge, and full-`model.cls` replay
   memorials pass; QK surgery changes exactly four logits, changes no `V`, and preserves every
   unselected logit;
5. `A @ V` reproduces the captured L3H1 pre-output-projection context with maximum absolute error
   `<1e-5`; applying the captured output projection reproduces the captured head write `<1e-5`;
6. every baseline reconstruct-inject reproduces its native `m(x_ell-)` `<1e-4`, and a no-op
   full-head replacement does likewise;
7. the clean-L3H1 same-example restore is measured as a real model-path continuity outcome; it is not
   replaced by a synthetic final-residual plant;
8. all 27 engineering boundary signatures and six path plants, arbitrary-boundary path tests, and
   endpoint-specific simultaneous-CI gates pass; and
9. bounded nucleotide-only MLM health passes at `>=0.50` on four deterministic non-site probes for
   each of the first 16 genes under seven fixed geometry policies: native; local-query zero stress for
   L3H1 and L0H2; exact all-token within-example-mean intervention for L2H15 and L0H2; and each builder
   combined with L3H1 local-query zero stress. These health policies are fixed plumbing bounds, not
   surrogates for the scientific cells. Independent paired-versus-pseudo corruption sensitivity is
   enforced by item 3, while every actual cell remains subject to the exact no-op, changed-coordinate,
   fixed-V, and reconstruction memorials in items 4--6.

Items 1--2 and 4--6, 8--9 are instrument gates. A failure there is `INSTRUMENT_UNRESOLVED` and
suppresses every R5a endpoint. Items 3 and 7 are model-behavior/path continuity requirements: with all
instrument gates healthy, a resolved failure is `ENTRY_BEHAVIOR_NOT_REPLICATED` or
`L3_PATH_NOT_REPLICATED`, a valid scoped stop rather than an instrument failure. `--force`, partial-
family reporting, or a post hoc tolerance change can create only a diagnostic artifact and cannot
authorize evaluation.

## 11. Frozen decision table

- `POSITION_SENSITIVE_READER_SUPPORTED` requires the basic Section 5 position path. It licenses only
  sensitivity to the paired↔adjacent position-embedding assignment under the fixed-V bridge.
- `POSITION_DOMINANT_READER_SUPPORTED` additionally requires the nested content-equivalence path;
  failure of this nested path does not retract a passing sensitivity result.
- `L3H1_DONOR_LOCAL_READER_SUPPORTED` requires `POSITION_SENSITIVE_READER_SUPPORTED`; the L3H1 paired-window QK partner endpoint and
  specificity are material positive; its pseudo endpoint is equivalent; the pseudo-window,
  adjacent-window, and L0H2 matched interventions are equivalent; A/V baseline identity passes;
  and every exact material/equivalence constituent of `P_AV_DONOR_WINDOW_VALUE` passes.
- `L2H15_ADDRESS_REUSE_SUPPORTED` additionally requires the paired-donor L2H15 address simple effect
  and specificity to be material positive, the value simple effect and pseudo arm to be equivalent,
  joint rescue to be material positive, and the complete L0H2 builder control to be equivalent.
- Material address and value effects produce `L2H15_MIXED_AV`; this does not license the stronger
  addresser wording.
- A real L3H1 effect accompanied by a resolved material pseudo/adjacent/negative-head control is
  `NONSELECTIVE_OR_CONTROL_ACTIVE`, a valid negative for the selective-reader path. Boundary-touching
  specificity is `SCIENTIFICALLY_UNRESOLVED`. Neither is `INSTRUMENT_UNRESOLVED` unless a plumbing,
  health, invariant, or classifier-capability gate also failed.
- All other resolved outcomes are bounded nulls for the exact intervention; they do not show that the
  model has no paired-donor information elsewhere.

Only `L3H1_DONOR_LOCAL_READER_SUPPORTED` opens R5b. The strongest possible R5a prose is: “Under a frozen
masked-acceptor assay, L3H1 is sensitive to the paired↔adjacent positional-embedding assignment and causally uses a paired-window-selective route and paired-donor-window values
relative to the fixed pseudo/adjacent controls for a **material portion of the paired-donor-dependent
component** of the acceptor margin; L2H15 [does/does not] supply the address under the tested
intervention.” It may
not say what the donor-local state means, that attention alone is causal, that the route is complete,
or that SpliceBERT implements spliceosomal recognition.

## 12. Provenance and outputs required before freeze

A future runner must write immutable primitive row cells, compact gene sufficient statistics, the
complete ordered family definitions, all derived algebra, gate map, hashes, command, seed, and a
manifest. It must rehash the GTF, FASTA, checkpoint bundle, R3c entry artifact, instrument sources,
code revision, and (for target) the preflight artifact and manifest after all model work and inference;
the manifest enumerates the exact bytes of every joint-stack Python-closure file and all three
protocols in addition to their normalized prospective identities. Drift suppresses artifact creation.
Figures may be generated only from recorded artifacts. The
protocol and instrument fingerprint must be literal pinned values before a target-bindable preflight;
this draft contains neither.

R3c entry authentication is semantic as well as byte-level: target/confirmatory status, held-out
verdict, all gates, acceptor-only single-G construction, absence of a masked-donor endpoint, exact
checkpoint, comprehensive-GTF/FASTA identities, fixed cohort/exclusion identities, artifact bytes,
and sibling-manifest bytes must all pass. R5a serializes both upstream hashes in its entry contract.
Only the corrected successor schema `gfmcd.r3c.acceptor_head_confirmation/2`, which carries the
instrument revision `r3c-v2.1-monotone-health-early-preflight-authorization` and the ordered seven-path
preflight-health/current-health intersection into target classification, is admissible. The consumed
L3H1 path must appear in `effective_authorized_paths`, every path-level health record must reproduce
that intersection, the reviewed v1 failure must keep `L2H15_mean_component` suppressed even if its
current target health passes, and the synthetic calibration must power all seven health-enabled paths; the
earlier `/1` development instrument cannot bind R5.
Before a bucket-1 target, its bucket-0 preflight must additionally match the current joint and local
method bindings, R3c artifact+manifest, checkpoint, sources, complete two-bucket census, bucket-0
reservation/construction, and stable runtime identity exactly; a method- or context-mismatched
preflight cannot authorize the target.

The checkpoint aggregate uses the R3c-compatible canonical SHA-256 of the ordered list of
`{relative_path,size_bytes,sha256}` records for every regular file under the checkpoint root; a dict
encoding or a hand-picked filename subset is not interchangeable.

The target artifact also exposes the exact downstream `gfmcd.r5.entry_contract/1`: boolean
`instrument_gates_pass`, boolean `claim_eligible`, `verdict`, `acceptor_G_only=true`,
`masked_donor_endpoint_constructed=false`, and the checkpoint aggregate, bucket-1 ordered-cohort,
and bucket-1 construction SHA-256 values. `claim_eligible=true` is possible only when the target
verdict is exactly `L3H1_DONOR_LOCAL_READER_SUPPORTED`; R5b/R5c must reject every other contract.
The contract also carries the joint-stack fingerprint, exact R3c artifact/manifest hashes, and the
manifest-authenticated exclusion-registry byte/self-digests.
