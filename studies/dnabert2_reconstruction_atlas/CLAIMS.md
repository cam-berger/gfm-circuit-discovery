# DNABERT-2 Reconstruction Atlas claim ledger

> **Tier: T1 — canonical for this study's claim wording.**

## Live scientific claims

**D1 — development behavior boundary.** In the frozen bounded GENCODE v50 protein-coding
development cohort, none of splice acceptor, splice donor, start codon, or stop codon meets the
predeclared excess-reconstruction gate over every empirical control. All four beat n-gram
baselines, but empirical advantages range from −0.130 to +0.023 nats per covered nucleotide;
the only positive value (donor versus composition/token matched, +0.023) is not
Holm-significant and fails the other empirical families.

This licenses no `NULL` verdict and no claim about the sealed confirmation split. It stops
context and internal localization for these classes under census v1.

**D2 — multisource development behavior boundary.** In the frozen multisource-v4 development
cohort, none of the 13 tested classes meets the predeclared excess-reconstruction gate over every
empirical control. All classes beat order-0/1/2 nucleotide baselines, with advantages from +0.060
to +0.251 nats per covered nucleotide. Empirical advantages instead range from −0.150 to +0.020;
the largest, PPT versus a same-sequence non-site (+0.020), has add-one
`p=0.2445` and full-family Holm `p=1.0`.

Poly(A) cleavage regions, splice acceptors, branch points, splice donors, and stop codons meet the
frozen 250-cluster behavior floor and still do not qualify. Promoters, poly(A) signals, CTCF/REST/
CEBPB peak cores, distal enhancers, polypyrimidine tracts, and start codons are development
estimates only because control-complete supply is below that floor. This licenses no `NULL` verdict
for any class and no claim about the sealed confirmation split. It stops context and internal
localization under multisource v4.

**D3 — replicated DNABERT-2 context-margin behavior.** Under the frozen context-margin-v1
amendment, nine classes replicate a positive target-preserving proximal-context effect on the
geometry-qualified alternative margin: start codon, distal-enhancer anchor, poly(A) cleavage
region, branch point, REST/CEBPB/CTCF peak anchors, poly(A) signal, and polypyrimidine tract.
Development/confirmation effects are respectively +0.0468/+0.0479, +0.0395/+0.0384,
+0.0298/+0.0348, +0.0412/+0.0304, +0.0136/+0.0213, +0.0215/+0.0183,
+0.0141/+0.0159, +0.0127/+0.0127, and +0.0109/+0.0125 nats per covered nucleotide.
Every confirmation result exceeds the frozen +0.01 SESOI, has a positive cluster-bootstrap lower
bound, and passes Holm correction over the opened family.

This licenses a replicated **model-behavior context effect**: the preregistered proximal context
supports the true target over its fixed eligible Hamming-one alternatives more than an
equal-width, equal-composition, equal-change-count distal corruption. It does not establish
biological function, a resolved motif, mediation, or a causal model component. The null is distal,
so proximity-sensitive encoding remains part of the licensed effect.

Promoter qualified in development (+0.0247) but did not replicate (+0.00066, 95% CI
−0.0117 to +0.0126, Holm `p=0.4548`). Stop codon failed development (+0.00330).
Splice acceptor and donor were quarantined before target scoring because the margin positive
control was unresolved; none of these four outcomes is silently pooled with the replicated set.

**D4 — fine-tuned promoter behavior replicates; context mechanism does not.** The selected
`prom_300_all` DNABERT-2 classifier replicates a positive clean promoter margin on
exact-GUE-excluded GENCODE v50 protein-coding MANE Select TSS windows relative to
composition-matched, TSS-excluded same-locus pseudo-TSS windows. Development and confirmation
effects are +2.0956
(95% CI +1.7813 to +2.4080) and +1.8892 (+1.6118 to +2.1867) classifier-logit-margin units;
both Holm-adjusted add-one values are 0.0002.

The target-preserving local-context difference-in-differences does **not** replicate. It is
+0.6655 (+0.3250 to +1.0080, Holm `p=0.0002`) in development and +0.2750
(−0.0684 to +0.6151, Holm `p=0.0569`) in confirmation. The confirmation point estimate exceeds
the +0.25 SESOI, but its interval and corrected p-value fail. D4 licenses supervised promoter
classification behavior on the declared external cohort. It licenses no context-specific
mechanism, internal component, necessity, or mediation claim.

**D5 — a development-frozen eight-head set causally supports the clean promoter contrast.**
The frozen promoter head atlas scanned all 144 attention heads at the unpadded `[CLS]`
destination on 192 unique GENCODE locus blocks per split. Development selected
L2H2, L4H6, L4H7, L9H7, L10H1, L10H5, L11H2, and L11H5 without inspecting confirmation.
Jointly replacing those target head-context slices with their paired pseudo-TSS values reduced
the target promoter margin by +1.0614 logits in development (95% CI +0.8179 to +1.3146) and
+0.7322 in confirmation (+0.5443 to +0.9289). The reciprocal target-to-pseudo-TSS transfer
raised the pseudo-TSS margin by +1.4166 (+1.1490 to +1.6863) and +1.1384
(+0.9041 to +1.3746). All four estimates exceed the frozen +0.60 SESOI. In each split and
direction, the candidate set exceeds all 63 distinct candidate-excluding random sets with the
exact same per-layer head counts; two-endpoint Holm-adjusted add-one `p=0.03125`.

Attribution/exact-patch head rankings remain positively associated in development
(Spearman 0.747/0.876 for necessity/sufficiency) and confirmation (0.602/0.799).
L11H2 and L11H5 have the largest individual exact-patch effects; L9H7 and L10H5 also have
positive corrected signs in both directions on both splits. Those individual rows remain
candidate priorities because only the eight-head group received a matched-head null test.
L9H7's target `[CLS]` attention is concentrated proximal-upstream (0.548 mass) and shifts
+0.0458 relative to controls, with a +0.0108 TSS-anchor shift; this is an attention clue, not
causal source localization.

D5 licenses replicated **candidate-set `[CLS]` head-output mediation** of D4's clean supervised
TSS-versus-pseudo-TSS contrast. It does not revive D4's failed local-context endpoint, establish
individual-head specificity, distinguish QK selection from OV content, identify a motif or
source position, resolve a downstream path, prove a sparse circuit, or establish transcriptional
function in cells.

**D6 — L11H2/L11H5 is a replicated late output-carrier core with a distributed MLP
scaffold.** The frozen circuit assay uses the 192 unique 1-Mb locus blocks per split that were
not used in D5. In development, exact joint L11H2/L11H5 patches have +0.6467 necessity
(95% CI +0.4278 to +0.8772) and +1.0680 sufficiency (+0.8251 to +1.3089). Confirmation
replicates +0.7004 necessity (+0.4833 to +0.9261) and +1.0932 sufficiency
(+0.8310 to +1.3631). In every split and direction, the pair exceeds all 45
candidate-excluding two-head pairs in layer 11; two-endpoint Holm-adjusted exact add-one
`p=0.04348`.

Strict complement scrub over all 144 `[CLS]` head contexts and 12 `[CLS]` MLP writes shows that
the eight D5 heads retain only 15.30% (95% CI 13.57% to 17.25%) in development and 13.73%
(12.07% to 15.56%) in confirmation. L11H2/L11H5 alone retains nearly the same
15.18%/13.62%. Retaining all MLP writes with the eight heads raises faithfulness to 61.04%
(57.30% to 65.51%) and 61.27% (57.30% to 65.61%). All MLP writes without native head writes
retain exactly zero, while all native head writes without MLP writes retain 34.36%/31.30%.
Empty and all-node anchors are exactly zero and one.

Two proposed mechanisms fail their frozen gates. Paired-control transfer of only the
L11H2/L11H5 target queries is effectively zero (−0.00004 development, −0.00034 confirmation)
and is not specific against the 45-pair family. Corrupting the six earlier D5 heads costs only
+0.0585/+0.0563 logits, below the +0.25 SESOI; therefore the small late-pair rescue cannot
license an early-six-to-late directed path even though its descriptive rescue fraction is
30.2%/31.6%.

D6 licenses a replicated **layer-specific two-head causal core at the late `[CLS]` head-output
node** and a continuously measured dependence on distributed MLP writes. The supported model is
a late output-carrier pair operating on an MLP-maintained representation. It is not a
target-dependent query gate, a demonstrated early-to-late chain, a complete sparse circuit, a
resolved K/V source or motif, or a claim about transcription in cells.

**D7 — The late L11H2/L11H5 core has replicated value-payload alignment dependence, not
detectable target-specific key-address dependence.** This separately frozen assay uses a newly
constructed cohort of 192 development and 192 confirmation locus blocks after excluding every
example and 1-Mb block in the original 768-locus promoter cohort. At a fixed dose of 40 genomic
tokens per sequence, deterministic V permutation leaves Q, K, QK logits, and post-softmax
attention probabilities native while reassigning which source value vector each probability
weights.

V permutation of L11H2/L11H5 causes a target-specific promoter-margin loss of +0.1026 logits in
development (95% CI +0.0517 to +0.1552) and +0.0615 in confirmation (+0.0030 to +0.1210). In
each split it exceeds every one of the 45 candidate-excluding two-head pairs in layer 11 and
passes the two-endpoint Holm gate (`p=0.04348`). By contrast, K permutation is only
+0.0073/+0.0084 logits, below the frozen absolute 0.05-logit SESOI in both splits. It therefore
does not support a target-specific content-key address role under this intervention.

The paired target-minus-control residual-write direction is stable between splits
(mean-direction cosine 0.9908), with a positive confirmation projection of +3.111
(95% CI +2.287 to +3.904) and 75.0% positive loci. These direction measurements are descriptive:
they received no semantic or direction-only causal test. Joint K/V permutation is also secondary;
its confirmation estimate is +0.0481 and its interval crosses zero.

D7 licenses a **causal value/source-state payload-routing dependence** for the replicated late
pair: the model's native attention probabilities must remain associated with the correct value
vectors for the promoter margin. Together with D6's null query-transfer result, it narrows the
pair away from target-specific Q or K addressing. It does not identify a source band, nucleotide
motif, promoter grammar, semantic residual direction, complete circuit, or transcriptional
function in cells.

**D8 — Equal-dose masking localizes a replicated classifier-relevant source region over the
nominal +11 to +21 nt interval, but the late-pair V payload and independent biological labels do
not localize there.** The outcome-blind assay uses the final 192 unused exact-GUE-excluded locus
blocks per split. All thirty 10-nt nominal tiles receive exactly two adjacent genomic BPE tokens per target
and pseudo-TSS sequence. Candidate generation is independent of CAGE, cCRE, CpG, and motif labels.

Tile 26 (+11 to +21 relative to the TSS) is the sole development-qualified region. Its
target-specific mask loss is +0.2452 logits (95% CI +0.0755 to +0.4285), its equal-dose
localization contrast is +0.2656 (+0.0953 to +0.4442), and its joint 60-test maximum sign-flip
`p=0.0260`. The unchanged confirmation test replicates at +0.2056 (+0.0447 to +0.3916;
`p=0.0082`). Actual BPE covers have median length 10 nt and range 6–15 across the experiment, so
this is not exact ten-nucleotide resolution.

The regional L11H2/L11H5 V swap does not qualify at any tile: the largest raw and localization
effects are +0.00124 and +0.00119, below the frozen +0.005/+0.002 SESOIs. ENCODE PLS overlap
qualifies in development but not confirmation. CAGE is underpowered at tile 26, CpG does not
qualify, and no geometry-qualified EPD TATA, Inr, CCAAT, or GC-box instance intersects the
nominated interval.

D8 licenses a replicated **classifier-level behavioral source region**. It does not license a
canonical promoter motif, a replicated biological annotation, a regional source for D7, a
mask-to-payload mediation claim, or transcriptional function in cells.

**D8 secondary performance boundary — the nominated mask reduces ranking quality and causes a
small number of target label flips, but not net balanced-accuracy loss.** In a user-requested
post-hoc evaluation of the unchanged tile-26 intervention, AUROC falls by 0.0332 in development
(95% pair-bootstrap CI −0.0604 to −0.0069) and 0.0207 in confirmation (−0.0431 to −0.0019).
Three of 107 and four of 124 real promoters flip from promoter to non-promoter. Accuracy changes
by −0.0093 and +0.0081, with both intervals spanning zero, because six pseudo-TSS false positives
also become correct negatives and other reverse transitions occur.

This supports a bounded hard-label and ranking consequence on the fresh GENCODE/pseudo-TSS
challenge. It does not show an accuracy loss on the original GUE task, independently validate
the selected region, or identify its biological content.

**D9 — the +11:+21 `[MASK]` nomination does not survive native sequence-order
qualification.** A fresh, outcome-blind cohort admits 60 unique locus blocks per split under an
exact mononucleotide- and linear-dinucleotide-preserving nonidentity reorder. Target and
pseudo-TSS edits have equal Hamming dose and matched local BPE retokenization geometry. The final
disjoint GUE preflight passes sensitivity, precision, identity, batch, geometry, and finiteness
checks.

Development target-specific promoter-margin loss is −0.0162 logits (95% CI −0.1762 to +0.1383;
add-one cluster-sign and Holm `p=0.5834`), failing the frozen +0.10 SESOI, positive interval, and
sign gates. Clean target-minus-control behavior remains strong at +2.321 (+1.392 to +3.266;
`p=0.0001`). Accuracy is unchanged at 0.650, AUROC changes descriptively from 0.7075 to 0.7133,
and one of 60 targets flips promoter→non-promoter.

D9 licenses a **failed development positive gate for exact-composition sequence-order
necessity under this intervention**. The interval still includes the +0.10 SESOI, so this is not
an equivalence null. D8 remains a post-tokenization mask-sensitive candidate, not established
native-sequence evidence. D9 does not show that +11:+21 is biologically inert, exclude every
native substitution, or resolve GC content: the GC-depletion instrument failed its frozen
precision gate. Confirmation, residual-source tracing, and head mediation remain sealed.

**D10 — L11MLP is the strongest accuracy-relevant CLS-write candidate, while no individual
head or MLP passes the complete replicated accuracy gate.** An outcome-blind supply exhausts
the final 100 unique 1-Mb blocks unused by D4 and D7–D9, split 50/50 before model outcomes. The
assay exchanges each of all 144 attention-head contexts and 12 MLP writes individually between
the real TSS and its paired pseudo-TSS at `[CLS]`.

L11MLP ranks first in both splits. Its exchange reduces balanced accuracy by 0.130 in development
(95% CI 0.030 to 0.230; full-family maximum-sign `p=0.0100`) and 0.120 in confirmation
(0.000 to 0.240; `p=0.0401`); AUROC falls by 0.2136 and 0.2832. It passes the frozen SESOI,
positive interval, and familywise gate in development, but the confirmation bootstrap lower
bound is exactly zero. It is therefore a strong replicated **candidate**, not a fully qualified
standalone correctness node.

L11H2 is the top head in both splits, with 0.040 accuracy loss in each; L11H5 ranks third/fourth
with 0.030/0.020 loss. Neither head has a positive bootstrap lower bound or survives the
156-node maximum-sign family in either split. D10 licenses an accuracy-first priority ordering
under reciprocal paired activation exchange. It does not license zero-ablation necessity,
individual-node sufficiency, a complete circuit, a nucleotide source, or biological promoter
function.

**D11 — a four-dimensional subspace is the dominant causal carrier of the L11MLP exchange
effect on the reused D10 cohorts.** The assay fits an uncentered SVD to the 50 development
target-minus-control L11MLP `[CLS]` writes, selects the smallest rank passing frozen causal
gates, and transfers the exact basis to the disjoint 50-pair confirmation split. Rank 4 uses
0.52% of the 768-dimensional write.

Projected reciprocal exchange captures 0.828 of the 0.922-logit full L11MLP class-margin loss in
development (89.7%) and 0.923 of 1.092 logits in confirmation (84.5%). It reduces AUROC by
0.184/0.236, compared with 0.214/0.283 for full exchange. The selected subspace beats all 31
dimension-matched random subspaces in each split (add-one `p=0.03125`). Target-from-control and
control-from-target directional effects are positive in both splits.

The orthogonal complement retains 0.079 logits in development (95% CI 0.002 to 0.155) and 0.156
in confirmation (0.070 to 0.246), so rank 4 is not a complete account of L11MLP. Its balanced
accuracy loss is 0.110/0.070, but both confidence intervals include zero. D11 therefore licenses
a **dominant low-rank causal carrier for class margin under reciprocal activation exchange**,
not a precise hard-label effect, standalone L11MLP necessity, neuron identity, an upstream
path, a nucleotide source, or a biological promoter mechanism.

This is explicitly post-hoc and cohort-reused. D10 selected L11MLP after both cohorts had been
scored, and no unused MANE 1-Mb blocks remained. Unchanged-basis transfer tests structural
compression, not fresh independent replication.

**D12 — A replicated downstream-vs-upstream mask-sensitivity polarity survives confirmation, but the
bounded downstream domain and its distal edge are refuted; D8's tile 26 stands alone as a
single-tile effect.** D12 is the confirmatory follow-up to D8, reusing D8's own sealed 192-pair
confirmation split under a protocol frozen before any of its five endpoints had been read beyond the
already-public `mask|tile26` value. Amendment 1 (pre-outcome) relaxed the original all-tile-required
completeness rule for E1 and E2 — which had collapsed E1 to an underpowered, zero-spanning
development estimate — to a looser rule requiring only one inner/downstream and one outer/upstream
tile present per locus, and added a second, sharper primary endpoint, E1b, needing just two tiles.

Of five scored endpoints, exactly one qualifies. **E2 (TSS polarity)** — mean target-specific mask
loss over downstream tiles 25/27 minus mean loss over upstream tiles 21/22/23 — replicates at
+0.3262 logits (95% CI +0.1110 to +0.5564, n=149), matching its development-predicted sign and
development point estimate of +0.4045, clearing the +0.15 SESOI with a positive lower bound, and
passing Holm correction (`p=0.0096`). This licenses a replicated **classifier-level
mask-sensitivity polarity**: masking the +1:+31 downstream span costs more target-specific promoter
margin than masking the −39:−9 upstream span.

**E1 (downstream edge)** and **E1b (sharp edge)** both fail. E1 is +0.0375 (95% CI −0.0964 to
+0.1666, n=138) against a development point estimate of +0.2291; E1b is +0.0496 (−0.0746 to
+0.1815, n=77) against +0.2631 in development. Both keep the predicted sign but clear neither their
+0.10 SESOI nor a zero-excluding interval nor Holm correction (`p=0.47` for each). Per D12's own
preregistered refutation rule — E1 and E1b failing together refutes the bounded-domain reading —
**the bounded downstream domain and the sharp distal edge near +31 are refuted**. D8's tile 26
result (+0.2056, replicated at the time) now stands alone as a single-tile effect, not a domain. E4
(full domain, descriptive-only because it includes the already-opened tile 26 and can never qualify)
is +0.0846 (−0.0442 to +0.2295, n=162).

The development-to-confirmation shrinkage is diagnostic, not incidental. The three endpoints that
contrast against the +31:+51 downstream flank collapse sharply: E1 shrinks from +0.2291 to +0.0375
(about 6.1×), E1b from +0.2631 to +0.0496 (about 5.3×), and E4 from +0.2524 to +0.0846 (about
3.0×) — all three fail. The two endpoints that never reference that flank shrink far less: E2 from
+0.4045 to +0.3262 (about 1.24×) and E3 from −0.2793 to −0.2164 (about 1.29×). Amendment 1 discloses
that the v2 endpoint definitions — including E1's completeness rule and E1b's addition — were
selected after inspecting several variants on the development split, which inflates the development
estimates reported there. This pattern is consistent with **selection-inflated development
estimates for the flank-contrasting endpoints**, not with sampling noise alone.

**E3 (upstream anti-evidence)** has the predicted negative sign and clears its 0.10 absolute SESOI
at −0.2164 (95% CI −0.4189 to +0.0005, n=128), but its interval misses excluding zero by 4.5e-04
and Holm `p=0.1476` fails. E3 licenses no null: it is preregistered as powered only for effects at
or above 0.10 absolute, so this near-miss calls for fresh data, not a conclusion that the upstream
reversal is absent.

The frozen instrument preflight is `QUALIFIED`, passing all four preregistered memorials plus a
batch-invariance check, but the four are not equally informative. **M2 (record anchor)** re-derived
all 124 recorded per-example `mask|tile26` confirmation values from D8's own record through an
independent code path, with a worst absolute difference of 1.133e-05 against a 1e-4 tolerance — this
is the historical-fidelity evidence that D12 is scoring the same instrument D8 used. **M1
(identity)** is a Q/K/V-style no-op patch (worst absolute difference 0.0 against the recorded value;
worst absolute value 2.38e-06) that is largely architecture-forced: correctly wired, it returns
near-zero almost regardless of checkpoint or tokens. It validates the patch machinery's internal
consistency, not fidelity to the historical run, and is flagged as such rather than counted as
independent confirmatory evidence alongside M2.

D12 licenses a replicated classifier-level mask-sensitivity polarity between the +1:+31 downstream
span and the −39:−9 upstream span, target-specifically. It does not license a bounded domain or any
distal edge — both are refuted here; a biological promoter element; a nucleotide motif;
native-sequence necessity; mediation by any internal component, including L11H2/L11H5 or L11MLP; or
transcriptional function in cells. Whether the polarity reflects GC/CpG composition rather than
position was open at the time of writing; **D14 answers the arrangement half of it** and D13 is
retired unrun. Tiles 0–20 and 24 of the 30-tile scan (22 tiles) remain sealed and unspent.

**D14 — the D12 polarity is NOT carried by nucleotide arrangement.** Three interventions applied
per 10-nt tile at D12's own E2 tiles, so the endpoint is structurally identical to the one D12
confirmed, scored on the sealed confirmation split (three-arm intersection, n=83). Arm A (`mask`)
is not re-estimated: D12's stored per-locus values are **replayed**, and the replay reproduces all
93 compared rows to a worst absolute difference of **0.0** — the positive control D13 never had.

| arm | what it destroys | what it holds fixed | E2 | 95% CI |
|---|---|---|---:|---|
| `mask` | content **and** manifold | nothing | +0.4791 | [+0.2288, +0.7358] |
| `gc_depletion` (dose 3) | GC/CpG content | purine/pyrimidine class; real bases | +0.2625 | [−0.0622, +0.5934] |
| `dinucleotide_reorder` | arrangement | mono- **and** linear-dinucleotide counts **exactly** | +0.0619 | [−0.0719, +0.1986] |

| endpoint | mean | 95% CI | sign-flip p | Holm p |
|---|---:|---|---:|---:|
| **P1** `mask − reorder` | **+0.4172** | **[+0.1705, +0.6608]** | 0.0022 | **0.0066** |
| P3 arrangement contribution | +0.0619 | [−0.0719, +0.1986] | 0.3616 | 0.4506 |
| P2 `gc_depletion − reorder` | +0.2006 | [−0.1191, +0.5238] | 0.2253 | 0.4506 |
| S1 `mask − gc_depletion` | +0.2166 | [−0.1307, +0.5561] | — | — |

Scrambling the bases within each tile while holding mono- and dinucleotide counts **exactly** fixed
(so GC *and* CpG counts are exactly preserved) leaves the polarity essentially intact: arrangement
carries **12.9%** of the mask effect, its median is **−0.0070**, and only 41/83 loci are positive —
a coin flip. Meanwhile `mask − reorder` survives Holm at `p=0.0066`. The result is **not** a
perturbation-size artifact: `gc_depletion` changes **3** nt and yields **+0.2625**, while `reorder`
changes **4.917** nt and yields **+0.0619**, so effect is **not monotone in Hamming dose**. Nor is
it a `[MASK]` off-manifold artifact: S1's CI spans zero, and `gc_depletion` uses only real bases.
Specificity: `reorder` at an inert far-upstream tile (tile 5, −199:−189) gives −0.0623, inside the
0.15 SESOI. Development-to-confirmation shrinkage is **1.27×** (+0.5303 → +0.4172), against the
5–6× that killed D12's E1/E1b.

**Bounds — what D14 does not license.** (i) *Not* "arrangement contributes nothing": the achieved
upper bound is **0.1986**, above the 0.15 SESOI, and reaching equivalence would need ≈474 loci.
(ii) *Not* "composition is the carrier": P2 was **declared underpowered before the run**
(MDE 0.448 against a development estimate of +0.327) and its CI spans zero; `gc_depletion`
reproduces ~55% of the mask effect but is not confirmed. (iii) "Arrangement" means arrangement
**modulo exact dinucleotide content** — such a shuffle preserves CpG count and much local structure,
so a motif can partially survive it. (iv) P1's magnitude is **skew-inflated**: median +0.2500, 10%
trimmed +0.3206, +0.2481 after dropping the five most extreme loci, and the non-parametric sign test
is marginal (51/83, exact binomial `p=0.0475`). The *direction* is robust across all variants and
five bootstrap seeds; the *magnitude* should be read as ≈0.25–0.32, not 0.42. (v) Seal status is a
**partial seal**: these loci were opened by D12 for the `mask` endpoints, but had never been scored
under either native arm, and P1/P2/P3/S1 had never been estimated on them.

Artifacts: `promoter_arrangement_composition_confirmation_v4.json` (per-locus values embedded),
`promoter_arrangement_composition_smoke_v1.json`,
`promoter_arrangement_composition_phase0_v1.json` (exploratory, development split).
Protocol: `PREREGISTRATION_D14_ARRANGEMENT_VS_COMPOSITION.md`.

**D15 — promoter information density is ~4× concentrated near the TSS, while the dominant carrier
heads attend at the ALiBi-forced positional prior.** Track B's first result. Three strands: an
architectural deduction (not evidence), a reused-artifact analysis (hypothesis-generating), and one
confirmatory endpoint on the sealed split.

**Architecture, not evidence.** DNABERT-2 has symmetric ALiBi and no position embeddings
(`bert_layers.py:394`), and the classifier reads `[CLS]` at window index 0 = TSS-relative −249. Because
`far_upstream` is `(−249,−50)` — 200 of 300 nt and the band nearest `[CLS]` — the ALiBi-only prior
places **49%–89% of every head's `[CLS]` mass in `far_upstream`, at all twelve slopes**. A head's
far-upstream attention dominance is therefore **architecture-forced and is not evidence of a content
preference**; all attention statements below are *measured minus prior*.

**Attention (descriptive, reused head-atlas artifacts).** L11H2 and L11H5 — the replicated causal core
(D6/D7), top two by exact patch — sit essentially **at** their positional prior (excess `far_up`
+0.046 and +0.032, ≈0.000 in every other band). They perform no measurable content-driven source
selection. The one candidate head that does redirect strongly onto the TSS neighbourhood, **L9H7**
(excess `prox_up` +0.401, `tss` +0.051, `prox_dn` +0.082), contributes about **19× less**
(necessity 0.0169 vs 0.3154).

**A failed prediction, recorded.** Head reach does **not** predict causal importance within layer:
mean within-layer Spearman 0.062–0.284 across four split×endpoint tests, sign-test p=0.146–1.000,
**nothing qualifies**. The D14 prereg §2 predicted that only long-reach heads could carry downstream
evidence to `[CLS]`; that prediction **fails**. A post-hoc slope-enrichment observation on D5's
eight-head set (p=0.0232) is **downgraded** — the statistic was chosen after seeing the head list and
the better-powered within-layer test is null.

**Confirmatory (sealed split, n=186).** Equal-dose contiguous span masking: 7 BPE tokens in target and
control, matched cover, contrasting the `−9:+31` region against five disjoint 40-nt far-upstream
windows. Any artifact that is a function only of "seven tokens were masked" cancels **exactly** in the
difference.

| endpoint | n | mean | 95% CI | Holm p | verdict |
|---|---:|---:|---|---:|---|
| **E_reg** `S − mean(F₁..F₅)` | 186 | **+0.3490** | **[+0.1153, +0.5892]** | **0.0058** | **QUALIFIES** |
| E_band `S − B` (dose 7 vs 30) | 114 | −0.3319 | [−0.9109, +0.2488] | 0.265 | fails; declared underpowered pre-run |
| N1 far-vs-far null control | 178 | +0.0806 | [−0.1153, +0.2910] | — | passes (< SESOI) |
| spec `S − U` (−49:−9) | 175 | +0.4504 | [+0.0747, +0.8169] | — | predicted sign, holds |

Per-token density is **0.0668** in `−9:+31` against **0.0162** in a far-upstream window — a **4.13×
ratio** (development 4.67×), both sides at identical dose. Development→confirmation shrinkage is
**1.29×** (+0.4495 → +0.3490), against the 5–6× that killed D12's E1/E1b. All memorials pass: clean
anchor +1.866, cover balance 0.828 nt, non-degeneracy 0.817 vs 0.113, batch invariance 5.25e-06.

D15 licenses: **at equal dose, width and contiguity the `−9:+31` region carries materially more of
the promoter decision than far-upstream windows do, while the carrier heads' attention is distributed
as the architectural prior dictates.** In short, **attention mass does not track information density**
— which is what makes D8's null (regional `L11H2/L11H5` V swap ≈0 at all 30 tiles) unsurprising rather
than anomalous.

**Bounds — what D15 does not license.**
(i) **E_reg is REGIONAL, not positional.** It cannot separate position from the composition resident
at that position — exactly what D14's P2 left open at declared-insufficient power. Calling it
positional would repeat the estimand drift the D14 red-team caught.
(ii) **The magnitude is skew-inflated and the per-locus sign test fails.** Median +0.1464 (*below* the
0.25 SESOI), 10%-trimmed +0.2899, 5/95-winsorised +0.3560, drop-5-extreme +0.2812, and only
**104/186 loci positive (exact sign test p=0.1234, not significant)**. The preregistered cluster
sign-flip test passes (raw p=0.0029) and the CI is stable across five seeds, so the *direction* is
robust, but read the magnitude as **≈0.15–0.29** and treat this as a distributional shift, not an
effect present at most loci.
(iii) **My "far-upstream is inert" hypothesis is REFUTED, by my own data.** E_band's point estimate is
**negative** (−0.3319): masking the whole far-upstream band costs **+0.8168** against **+0.4675** for
the signal window. Far-upstream is diffuse and low-density, **not** unimportant. E_band was declared
underpowered before the run (12% at the SESOI) and its CI spans zero, so no claim is made either way —
but nothing here supports inertness. The band arm is also **super-additive** on confirmation
(5× a window = +0.566 against a measured +0.817), so dose-30 masking is a different regime and is not
safely comparable to dose-7 arms.
(iv) **No mediation.** Attention mass is descriptive and is measured on a different cohort from the
behaviour (head atlas vs source-localization; §3e of the prereg). Nothing here shows any component
causally mediates the `S` effect, and attention mass is not information transfer.
(v) **"Relay" remains an inference**, not a measurement — no path was patched and no relay position or
layer was identified.
(vi) **Partial seal**, and no biological claim: no motif, no CpG-island mechanism, no transcription in
cells.

Artifacts: `promoter_region_density_confirmation_v1.json`, `promoter_region_density_smoke_v2.json`,
`promoter_alibi_reach_analysis_v1.json`.
Protocol: `PREREGISTRATION_D15_ATTENTION_CAUSALITY_DISSOCIATION.md` (Amendment 1, pre-outcome).

**D16 — fine-tuning built exactly one reader and left the readout layer's attention geometry
untouched.** The study's **first measurement on two checkpoints**. D1–D3 used base
`zhihan1996/DNABERT-2-117M`, D4–D15 used `magicslabnu/DNABERT-2-finetuned-prom_300_all`, and no prior
script loaded both — so nothing in the record was a before/after of anything. This is descriptive
(attention only) on all 384 `promoter_source_localization_v1` loci.

The comparison is licensed because the two checkpoints tokenize **identically**: 0 `input_id`
mismatches and 0 offset mismatches over all 768 cohort sequences (M1). The BPE offsets, the
nucleotide-band allocation and the ALiBi-only prior are therefore common to both models and cancel in
every difference below.

**Frozen §4 verdict: ALREADY AT PRIOR.** The carriers were pinned to the positional prior *before*
fine-tuning, and stayed there:

| head | base excess `far_up` | fine-tuned excess `far_up` | Δ | threshold |
|---|---:|---:|---:|---:|
| L11H2 | +0.0256 | +0.0463 | +0.0206 | 0.10 |
| L11H5 | +0.0141 | +0.0318 | +0.0178 | 0.10 |

Across all 144 heads the `[CLS]` attention geometry is largely preserved: Pearson correlation of
per-head excess `far_upstream` between checkpoints is **0.951**, mean |Δ| is 0.062 against a mean
|excess| of 0.34. **No layer-11 head moved into the TSS-proximal bands** (largest is L11H3 at
+0.0130). There is no population-wide drift off the attention-sink tokens (mean Δ`special` +0.0215,
62/144 decreasing — a coin flip).

**The one large change: L9H7 converted from an attention sink into a TSS reader.** Raw `[CLS]` mass:

| band | base | fine-tuned |
|---|---:|---:|
| `special` (`[CLS]`/`[SEP]`) | **0.3759** | **0.0038** |
| `far_upstream` | 0.3757 | 0.1450 |
| **`proximal_upstream`** | **0.1005** | **0.5495** |
| `tss_anchor` | 0.0112 | 0.0651 |
| `proximal_downstream` | 0.0236 | 0.1035 |

Summed TSS-proximal excess moves from **−0.055 to +0.528 (Δ +0.583)**. The next-largest such shift in
the whole model is L10H5 at **+0.043 — 14× smaller** — so this is a singleton, not a trend. The other
large movers (L9H10, L6H1, L5H1, L8H1, L9H0) are pure `special ↔ far_upstream` rebalances with
TSS-proximal shifts of 0.0000 to four decimals.

D16 licenses: **fine-tuning this checkpoint for promoter classification did not redistribute `[CLS]`
attention. It converted a single mid-network attention-sink head (L9H7) into a TSS-proximal reader and
left the readout layer's attention essentially as pretraining had it.** Read together with D15 — where
L9H7 contributes ~19× less to the decision than the two unchanged, positionally-pinned carriers — the
reader fine-tuning built is not the component that carries the decision.

**Bounds.** (i) **Descriptive only** — attention, not behaviour, not causation; nothing here shows a
changed head caused a changed output. (ii) **Attention mass is not information transfer**; a head at
its positional prior can still move information. (iii) **This does not fill the behavioural
before/after.** The base model has no promoter readout, so D15's endpoint has no base-model
counterpart; that comparison was **retired at design** (see `DECISIONS.md`, 2026-07-29) because every
affordable base-model readout is either non-functional (`[CLS]` is not read by an MLM) or
distance-confounded. (iv) `[CLS]`-row only. (v) Verification: the prior recomputes against D15's
artifact to 4.27e-07, and D16's fine-tuned excesses reproduce D15's — measured on a **different
cohort** through a different code path — to within 0.01; batch invariance is exactly 0.0.

Artifacts: `promoter_finetuning_attention_shift_v1.json`,
`promoter_finetuning_attention_shift_smoke_v1.json`.
Protocol: `PREREGISTRATION_D16_FINETUNING_ATTENTION_SHIFT.md`.

## Established infrastructure facts

These are implementation properties and prior instrument constraints, not claims about biological
reconstruction.

| ID | Fact | Evidence | Boundary |
|---|---|---|---|
| IF1 | The adapter pins DNABERT-2-117M checkpoint, tokenizer, config, and vendored MosaicBERT code by revision and SHA-256; an incomplete MLM head is a hard failure. | `src/gfmcd/adapters/dnabert2/` and adapter tests | Does not show the model reconstructs any biological element. |
| IF2 | Atlas masking selects the minimal BPE token cover from nucleotide offsets and masks token IDs without retokenizing. | `test_atlas_tokenization.py` | An assay invariant, not model behavior. |
| IF3 | Alternative margins are rejected unless full offsets and all unmasked token IDs remain identical. | `test_atlas_tokenization.py` | Does not guarantee adequate alternative supply. |
| IF4 | Context exchanges are ineligible if they change target nucleotides, length, target token indices, offsets, or cover. | `test_atlas_context.py` | Does not establish any informative region. |
| IF5 | Causal specs reject null sets whose component count differs from the candidate intervention. | `test_atlas_mechanism.py` | Does not establish necessity or mediation. |
| IF6 | The bounded GENCODE v50 collection has zero target/baseline cluster overlap and exact declared matching balance; splice donor, splice acceptor, start codon, and stop codon meet outcome-blind control supply, while Kozak does not. | `results/dnabert2_reconstruction_atlas/collection_qualification_v1.json` | Supply qualification is not a reconstruction outcome; Kozak is untested, not null. |
| IF7 | Preflight calibration v1 passed geometry, split, balance, finiteness, deterministic replay, and batch-invariance checks, but its randomly located ≥32-nt-buffer copy plant failed class-wide positive-control monotonicity and sensitivity. | `results/dnabert2_reconstruction_atlas/preflight_calibration_v1.json` | `INSTRUMENT_UNRESOLVED`; no target outcome or biological null is licensed. |
| IF8 | Calibration v2 validates the nearest eligible copy-plant placement with four-copy means of +0.09 to +0.21 nats per covered nucleotide, while exposing an over-strict monotonicity implementation and a positive-effect rather than null-noise resolution statistic. | `results/dnabert2_reconstruction_atlas/preflight_calibration_v2.json` | Development-only instrument diagnosis; the lock remains draft and no target outcome is licensed. |
| IF9 | Corrected calibration v3 shows all four admitted classes exceed the positive-control floor with ordered doses; null-noise MDEs are ≤0.0139 and CI half-widths ≤0.0096, supporting the frozen v1 thresholds. | `results/dnabert2_reconstruction_atlas/preflight_calibration_v3.json` and `configs/dnabert2_atlas/preflight_v1.frozen.json` | Calibrates the instrument; target execution still requires a separate frozen `QUALIFIED` report. |
| IF10 | Frozen preflight v1 is `QUALIFIED`: every mandatory positive, negative, sensitivity, resolution, invariant, balance, geometry, and model-health gate passes against the exact bound input hashes. | `results/dnabert2_reconstruction_atlas/preflight_qualified_v1.json` | Authorizes development behavior scoring only; it is not evidence that any biological class reconstructs excessively. |
| IF11 | Multisource-v2 adapters and frozen definitions cover 13 classes from GENCODE, EPDnew, PolyASite, ENCODE TF clusters, Mercer lariat-seq, and ENCODE cCREs. A model-free full build rehearsal fills 384 locus blocks per class in each split after strict coordinate/source attrition. | `SOURCE_MANIFEST_V2.json`, `preflight_v2.frozen.json`, adapter tests | Outcome-blind source and collection fact only; v2 target scoring still requires a new qualified preflight. |
| IF12 | V2 outcome-blind control supply is insufficient for seven-nt promoter, enhancer, and TF peak-core anchors; those classes are supply-quarantined, not biologically negative. | `results/dnabert2_reconstruction_atlas/collection_qualification_v2.json` | No model score was computed. V3 changes only target width for broad interval anchors and outcome-blind local candidate prioritization. |
| IF13 | V3 five-nt anchors make all 13 classes preflight-eligible, but eight classes remain below the frozen 250-locus behavior floor under 128 local candidates. | `results/dnabert2_reconstruction_atlas/collection_qualification_v3.json` | Outcome-blind supply fact only; V4 expands candidate search without relaxing controls or the behavior minimum. |
| IF14 | V4’s 512-position search retains the same 6,448 complete matched targets as v3. All 13 classes meet the preflight floor; five meet the frozen behavior floor and eight are known underpowered before model scoring. | `results/dnabert2_reconstruction_atlas/collection_qualification_v4.json` | Final outcome-blind supply boundary; underpowered does not mean biological null. |
| IF15 | Frozen multisource-v4 preflight is `QUALIFIED` for all 13 classes against the exact bound data, controls, config, checkpoint, and model-health checks. | `results/dnabert2_reconstruction_atlas/preflight_qualified_multisource_v4.json` | Authorizes development scoring only; not a biological reconstruction result. |
| IF16 | Context-margin supply retains 4,799 target-preserving structured/irrelevant variant pairs across all 13 classes with zero target-cover, geometry, dose, or split violations; every class supplies at least 82 loci on each target split. | `results/dnabert2_reconstruction_atlas/context_margin_v1_supply_audit.json` | Outcome-blind supply only; attrition is substantial and retained rather than repaired. |
| IF17 | The corrected frozen context-margin preflight is `QUALIFIED` for 11 admitted classes. Copy-margin positive controls, irrelevant-control equivalence, ≤0.006 MDE, ≤0.004 CI half-width, exact geometry/dose, and model health pass. Acceptor and donor remain preflight-quarantined. | `results/dnabert2_reconstruction_atlas/context_margin_v1_preflight_qualified_rerun.json` | Authorizes the context-margin development run; does not itself establish D3. The prior failed qualified artifact is preserved as a gate-membership implementation failure. |
| IF18 | Among compatible human all-promoter DNABERT-2 heads, `prom_300_all` has the stronger native GUE test result (MCC 0.8705, AUROC 0.9848) and a complete pinned two-class head. | `results/dnabert2_reconstruction_atlas/promoter_classifier_checkpoint_audit_v2.json` | Checkpoint selection only; it does not establish promoter behavior on an external cohort. |
| IF19 | Every selected 300-nt EPDnew atlas window occurs verbatim in GUE; the 70-nt core task also has 768/768 exact membership at its best one-base alignment. | `results/dnabert2_reconstruction_atlas/promoter_classifier_checkpoint_audit_v2.json` | EPDnew is excluded from the fine-tuned target run; scoring it would be training-source contaminated. |
| IF20 | The replacement GENCODE cohort contains 384 development and 384 confirmation MANE TSS locus blocks, zero exact forward/reverse GUE memberships, zero split leakage, exact anchor/composition/dose invariants, and bounded GC/CpG imbalance. | `results/dnabert2_reconstruction_atlas/promoter_classifier_cohort_audit_v1.json` | Outcome-blind cohort fact only. Exact exclusion does not rule out undocumented near-duplicates. |
| IF21 | Promoter preflight v1 is `INSTRUMENT_UNRESOLVED` because a one-draw permuted-label MCC and an over-tight CPU batch tolerance fail; no GENCODE target outcome was opened. | `results/dnabert2_reconstruction_atlas/promoter_classifier_preflight_v1.json` | Preserved instrument failure, not a promoter null. |
| IF22 | Promoter preflight v2 is `QUALIFIED` after replacing the volatile one-draw null with a 256-permutation center and rounding the numerical tolerance to \(10^{-5}\), while retaining all scientific SESOIs. | `results/dnabert2_reconstruction_atlas/promoter_classifier_preflight_v2.json` | Authorizes the D4 target run; preflight itself is not external-cohort behavior evidence. |
| IF23 | Reserved-GUE calibration validates the promoter head instrument: exact no-op patching is zero, batch drift is \(4.77\times10^{-7}\), all-head CLS transfer recovers the class contrast, attribution tracks exact patches, and the calibrated group exceeds exact layer-profile nulls. | `promoter_head_atlas_calibration_v1.json` and `promoter_head_group_calibration_v1.json` | Calibration only; its GUE-ranked heads are not target candidates and create no GENCODE internal claim. |
| IF24 | Frozen promoter-head preflight v1 is `QUALIFIED` on 32 GUE pairs disjoint from calibration: GUE MCC/AUROC, attribution agreement, full/no-op/same-class patches, projected sensitivity, null-set resolution, token geometry, head capture, and model health all pass. | `promoter_head_atlas_preflight_v1.json` | Authorizes the development head atlas; it is not a GENCODE component outcome. |
| IF25 | Frozen promoter-circuit preflight v1 is `QUALIFIED` on 16 new GUE pairs: all 156 declared CLS write nodes are covered, full scrub reproduces paired-control margins exactly, empty complement and self/query/restore no-ops are exactly zero, batch drift is \(4.77\times10^{-7}\), and full-scrub sensitivity is 8.14 logits. | `promoter_circuit_preflight_v1.json` | Authorizes D6 development only; GUE controls are not evidence about unused GENCODE loci. |
| IF26 | The expanded promoter-decomposition cohort supplies 768 exact-GUE-excluded MANE TSS/pseudo-TSS pairs per split. Outcome-blind selection binds 192 fresh example IDs and unique 1-Mb blocks per split after excluding the complete original promoter cohort; every sequence has at least 44 genomic tokens, supporting a fixed surgery dose of 40. Reserved-GUE calibration gives projected 192-pair MDEs of 0.0170 logits for K and 0.0456 for V at that dose. | `promoter_decomposition_cohort_audit_v1.json`, `promoter_decomposition_supply_audit_v1.json`, and `promoter_address_payload_calibration_v1.json` | Cohort, token-geometry, and calibration facts only; no fresh GENCODE model outcome was read before the lock. |
| IF27 | Frozen promoter address/payload preflight v1 is `QUALIFIED` on 32 reserved GUE pairs: identity error is \(2.38\times10^{-6}\), all-layer-11 K/V target-specific sensitivity is 0.0372/0.2049 mean absolute logits, projected MDE is 0.0032/0.0262, batch drift is zero, all 45 matched null pairs are present, and residual writes are finite. | `promoter_address_payload_preflight_v1.json` | Authorizes D7 development only; reserved-GUE controls are not evidence about fresh GENCODE loci. |
| IF28 | The source-localization supply binds 192 final unused locus blocks per split, 107–140/111–133 eligible pairs per tile, 105 CAGE-supported TSSs per split, 138/142 PLS TSSs, 70/73 CpG-island TSSs, and geometry-qualified motif disruptions without reading a target outcome. Reserved-GUE calibration gives identity error \(9.54\times10^{-7}\), mask sensitivity 0.463 mean absolute logits, and candidate regional-V projected MDE 0.00297. | `promoter_source_localization_supply_audit_v1.json` and `promoter_source_localization_calibration_v1.json` | Outcome-blind supply and calibration only. |
| IF29 | Frozen promoter source-localization preflight v1 is `QUALIFIED` on 32 new reserved-GUE pairs: all identity, mask/V sensitivity, precision, tile-resolution, batch, finiteness, and 45-null-family checks pass. | `promoter_source_localization_preflight_v1.json` | Authorizes D8 development only; not evidence about fresh GENCODE loci. |
| IF30 | Native-source preflights v7 and v8 are `INSTRUMENT_UNRESOLVED`: three-base GC depletion has projected MDE 0.403 at 40 loci, and a disjoint dinucleotide-only slice has projected MDE 0.364. No target outcome was read in either run. | `promoter_native_source_preflight_v7.json` and `promoter_native_source_preflight_v8.json` | Preserved instrument failures; neither is a biological null. |
| IF31 | Dinucleotide-only supply v9 binds 60 fresh unique locus blocks per split; frozen preflight v10 passes all eight checks on a third disjoint GUE slice with projected 60-locus MDE 0.156 and zero replay/batch drift. | `promoter_native_source_supply_audit_v9.json` and `promoter_native_source_preflight_v10.json` | Authorizes D9 development only. |
| IF32 | The all-node accuracy supply contains exactly the final 100 outcome-unseen unique 1-Mb blocks after excluding every D4/D7/D8/D9 example and block, with 50 pairs frozen per split. | `promoter_node_accuracy_supply_v1.json` | Outcome-blind supply fact only; it does not identify an important component. |
| IF33 | All-node preflight v1 is preserved as `INSTRUMENT_UNRESOLVED` because simultaneous layerwise exchange is not a terminal classifier-input swap. V2 changes only the positive control: exact terminal CLS residual exchange takes held-out GUE accuracy from 0.891 to 0.109 and AUROC from 0.977 to 0.023; all 156 hooks are finite and self/batch drift is zero. | `promoter_node_accuracy_preflight_v1.json` and `promoter_node_accuracy_preflight_v2.json` | V2 authorizes D10; neither preflight is a target-cohort component result. |
| IF34 | Frozen low-rank preflight v1 is `QUALIFIED` on a new GUE slice: clean accuracy/AUROC are 0.906/0.992, full L11MLP exchange loses 6.856 class-margin logits, rank-zero drift is zero, reconstruction error is \(9.54\times10^{-7}\), and deterministic-basis, orthonormality, batch, and finiteness checks pass. | `promoter_low_rank_preflight_v1.json` | Authorizes the D11 projection instrument only; it reads no D10 target outcome. |

## Open hypotheses

| ID | Hypothesis | Current status | First required bridge |
|---|---|---|---|
| H1 | At least one annotated site class has positive excess reconstruction over every required control. | not supported in multisource-v4 development; five adequately powered classes fail and eight classes remain underpowered | A newly preregistered behavior design or expanded control-complete supply; v4 confirmation stays sealed. |
| H2 | For at least one H1-qualified class, a biologically related context region outperforms matched irrelevant-region exchanges. | blocked by the v4 H1 gate | H1 plus token-geometry-invariant context interventions. |
| H3 | A bounded set of heads, MLPs, positions, or residual directions causally supports an H2-qualified behavior. | blocked by the v4 H1 gate | Frozen candidate screen, matched-size necessity, mediation, and path tests. |
| H4 | A geometry-qualified DNABERT-2 alternative margin has a replicated, class-specific target-preserving proximal-context effect. | supported for the nine D3 classes; promoter did not replicate, stop failed development, acceptor/donor are instrument-unresolved | A distance-matched or orthogonal context null would narrow the proximal-context interpretation further. |
| H5 | A bounded internal DNABERT-2 component set causally mediates a D3 context-margin effect. | open but untested for the nine D3 classes | Freeze candidates and matched-size necessity/mediation tests before inspecting internal effects. |
| H6 | A promoter-fine-tuned DNABERT-2 has a replicated target-specific local-context effect at training-corpus-excluded human TSSs. | not supported: clean promoter margin replicated, but the context difference failed confirmation | A new independently justified context intervention/null and a fresh sealed cohort; D4 does not authorize internal localization. |
| H7 | A bounded set of head outputs at `[CLS]` mediates the replicated clean promoter-classifier contrast. | refined by D7: L11H2/L11H5 is a replicated late output-carrier core with value-payload alignment dependence and a distributed MLP scaffold; target-query, content-key-address, and early-six-to-late models fail their gates | Freeze an equal-dose source-localization phase for the replicated V estimand; do not infer bands or motifs from the stable residual direction alone. |
| H8 | A bounded TSS-relative source band or sequence feature supplies the late pair's replicated value payload. | refuted as a bounded band: D12's confirmatory tile-domain test shows neither a bounded downstream domain nor a distal edge near +31 replicates (E1/E1b both fail confirmation); D8's +11:+21 tile stands alone as a single-tile effect. A distinct classifier-level downstream-vs-upstream mask-sensitivity polarity does replicate (D12 E2), but regional V was already null (D8), so no span tested has payload-mediation evidence. **Corrected 2026-07-29:** an earlier wording cited D9 ("exact-composition native reordering already failed development") in support of a general arrangement null. D9 scored **tile 26 only** (+11:+21), which is *not* in D12's E2 contrast `{25,27}` vs `{21,22,23}` — and D12's own E4, the only endpoint containing tile 26, also failed. D9's null is bounded to tile 26. **D14** has since tested the E2 tiles directly and finds arrangement carries 12.9% of the polarity (P3 `+0.0619`, CI `[−0.0719,+0.1986]`) while `mask − reorder` survives Holm at `p=0.0066` | D8/D12/D14 do not authorize mask-by-payload tracing or mediation. D13 is **retired unrun**; D14 replaced it and answers the arrangement half. Whether composition specifically is the carrier remains open — D14's P2 was declared underpowered pre-run. |
| H9 | One individual CLS head context or MLP write is independently necessary for correct promoter classification. | not qualified; L11MLP is rank 1 in both splits but its confirmation accuracy-loss CI touches zero, and no head passes the full-family gate | A preregistered targeted L11MLP intervention on new data, preferably including zero ablation and matched MLP nulls; D10 remains an atlas-level candidate result. |
| H10 | A low-dimensional L11MLP write subspace carries most of the promoter class-margin effect. | supported post-hoc on reused D10 cohorts: the unchanged rank-4 basis captures 89.7%/84.5% of full exchange and beats 31 matched random subspaces in both splits; the complement remains positive | Transfer the fixed basis to genuinely new examples, then test late-head-to-subspace mediation and rescue. |
| H11 | The `[CLS]` attention of the promoter classifier's causal carrier heads is allocated by the ALiBi positional prior rather than by where the promoter evidence is. | supported descriptively by D15: L11H2/L11H5 sit at their prior (excess +0.046/+0.032) while confirmatory per-token information density is 4.13× higher in −9:+31 than in far-upstream windows (E_reg +0.3490, Holm p=0.0058). Head reach does not predict causal effect (null, 4 tests). **Not** a mediation result: attention and behaviour are measured on different cohorts and attention mass is not information transfer. | A within-cohort test that pairs the attention measurement and the mask behaviour on the same loci, then a path-patching test of whether `−9:+31` reaches `[CLS]` directly or via intermediate positions. |
| H12 | TSS-proximal promoter evidence reaches `[CLS]` by multi-hop relay rather than by direct attention from the carrier heads. | **untested inference only.** D15 makes direct reading by L11H2/L11H5 hard to sustain (their attention there is ≈0.1%–11%) and explains D8's all-tile V-swap null, but D15 patches no path and identifies no relay position or layer. | Layer-resolved path patching: block or restore the residual stream at intermediate positions between `−9:+31` and `[CLS]`, and locate the depth at which the information becomes readable at `[CLS]`. |

Negative, retracted, and superseded rows will remain here permanently once evidence exists.
