# Preregistration: reverse engineering the native GTEx final-layer representation

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Status: **FROZEN BEFORE FINAL-BLOCK COMPONENT OUTCOMES.**

Frozen on 2026-07-29 after GFS8–GFS9, but before inspecting any per-head intervention,
final-layer MLP intervention, native-head attention map, or Q/K/V intervention in a GTEx-fine-tuned
checkpoint. The existing native behavior and layerwise projection artifacts are known and motivate this
study. They do not select a final-layer component.

## 1. Hypothesis and operational definition

GFS9 uses the phrase **co-adapted final-layer splice-classification representation** as a representation
candidate. In this study that phrase is not a conclusion unless all of the following are separated:

1. **Interface adaptation:** the released classifier reads the fine-tuned final donor state, while the same
   classifier is incompatible with the base final state. This is already established by GFS9.
2. **Final-block formation:** paired-acceptor evidence is nearly absent under the frozen classifier after
   layer 4 and becomes readable after layer 5. This is already established descriptively by GFS9.
3. **Causal component:** an exactly defined final-layer attention head or the final-layer MLP carries
   acceptor-dependent donor evidence under both clean-to-corrupt noising and corrupt-to-clean denoising.
4. **Computational role:** Q/K routing, V-carried content, and MLP transformation are measured separately.
   Attention weights alone cannot assign any role.
5. **Biological specificity:** the component effect is larger for the annotated paired acceptor than for a
   distance-matched non-partner `AG`, is not merely a shift at every donor in the same sequence, and retains
   local donor competence.
6. **Replication:** a train-selected component and its role reproduce on the annotation-only validation
   cohort across the five released fine-tuning seeds.

The full phrase is licensed only to the level reached. For example, a reproducible causal final-layer
component without source localization licenses “a causal final-layer carrier of paired-acceptor evidence,”
not “an acceptor-reading head.” A source-localized effect at the paired acceptor licenses a model
computation with the functional shape of cross-intron splice-site pairing, not spliceosome biochemistry.

## 2. Fixed model, cohort, behavior, and coordinates

- Base configuration/tokenizer: the hash-frozen `SpliceBERT.1024nt`.
- Native models: all five strict GTEx token-classifier loads, seeds 2020–2024.
- Discovery: seed 2020, annotation-only `train`, `n=128`, stable tag
  `gtex-native-mechanism-discovery-v1`.
- Confirmation: annotation-only `val`, `n=400`, stable tag `sealed-unc-val`, all five seeds.
- The chr8/9 test split remains sealed.
- Clean input is the unmasked 1,024-nt sequence. Corrupt input replaces only the annotated paired
  acceptor `AG` at `[next_acceptor_pos-2,next_acceptor_pos)` with `CC`.
- The primary readout is the donor-vs-no-site logit margin at `donor_pos-1`, the authors' class-1 label
  coordinate. Token index is therefore `donor_pos` after the one-token prefix.
- The primary behavior gap is `clean donor margin - corrupt donor margin`, in logits (“nats”).
- No example is selected on its behavior gap. Recovery is reported as a difference in margins; unstable
  per-example recovery ratios are forbidden.

The final encoder layer is layer index 5. Its self-attention context has 16 equal-width heads. At the
donor output token, later computation is position-wise, so a layer-5 head-context patch at that token
intervenes directly on that head's contribution to the only final state read by the classifier.

## 3. Instrument preflight

Preflight uses seed 2020 and the first 16 frozen discovery examples. It may inspect only reconstruction,
identity, no-op, and fixed positive-control outcomes. It may not rank heads.

Every check must pass:

1. strict native checkpoint loading, finite forward, eager attention, 16 heads, 6 layers, and
   one-token-per-nucleotide coordinates;
2. explicit classifier projection from the final hidden state reconstructs native logits with maximum
   absolute error `≤1e-5`;
3. the eager attention probabilities multiplied by projected values reconstruct the returned per-head
   context with maximum absolute error `≤2e-5`;
4. clean-source into clean-target head patching at the donor is identity (`≤2e-5` margin error), and an
   all-false token mask is a no-op (`≤2e-5`);
5. clean-source into clean-target MLP-write patching at the donor is identity (`≤2e-5`), and an empty
   position selection is a no-op (`≤2e-5`);
6. replacing the complete final hidden state at the donor in a clean run with its corrupt-run value
   reproduces the corrupt native donor margin (`≤2e-5`). This is the fixed positive control;
7. for predesignated `L5H0`, corrupt Q at the donor plus corrupt K and V at every valid source token
   reproduces corrupt head-context patching at the donor (`≤2e-5`). Q/K/V decomposition is invalid
   unless this exact memorial passes;
8. clean and corrupt interventions leave batch shapes, token coordinates, padding masks, and all
   unselected examples invariant.

Any failure freezes discovery. Thresholds cannot be relaxed; a correction requires an immutable failed
artifact and a new protocol version.

## 4. Discovery: which final-layer components carry the learned signal?

Discovery scans all 16 layer-5 heads. For each head `h`, only its context vector at the donor output token
is patched:

- **Noising necessity:** cache `h` on the corrupt input, patch it into the clean run, and define
  `damage_h = clean margin - noised-clean margin`.
- **Denoising sufficiency:** cache `h` on the clean input, patch it into the corrupt run, and define
  `recovery_h = denoised-corrupt margin - corrupt margin`.

The layer-5 MLP residual write at the donor is tested by the identical two directions, but is a whole
768-dimensional block and therefore belongs to a separate inference family. It is never pooled with
single-head controls.

For every estimand, uncertainty is a gene-grouped percentile bootstrap (`5,000` draws). Zero-effect tests
use `9,999` gene-level sign flips with exact add-one p-values. Benjamini–Hochberg correction spans the
32 head-by-direction tests; the two MLP directions form their own family.

A head becomes a frozen candidate only if:

- mean damage and mean recovery are each at least `0.05` logit units;
- both 95% gene-bootstrap interval lower bounds exceed zero; and
- both BH-adjusted head-family q-values are `≤0.10`.

Candidates are ranked by `min(mean damage, mean recovery)` and capped at three. If none qualifies, the
head-sparse hypothesis is falsified and confirmation stops: the final-layer change may be MLP-dominant or
distributed below single-head resolution.

For every selected candidate, three equal-size layer-5 control heads are drawn deterministically from the
non-candidate set using tag `gtex-native-mechanism-controls-v1`; candidate outcomes cannot influence which
eligible control is drawn. Candidate identities and controls are written to an immutable freeze artifact
before validation begins.

## 5. Confirmation and computational-role decomposition

The frozen seed-2020 candidate identities and control identities are applied unchanged to all five model
seeds on validation. This is an index-aligned replication because all five fine-tunes start from the same
pretrained architecture. A seed-specific rescan on validation is forbidden.

### 5.1 Primary causal confirmation

Primary endpoints are candidate noising damage and denoising recovery. The five-seed mean is formed
per example before gene-grouped inference. A candidate confirms only if:

- both family-mean 95% intervals are wholly above zero and both means are `≥0.05`;
- at least four of five seed estimates are positive in each direction;
- both family-mean candidate effects exceed the mean of their three matched control heads, with a 95%
  interval wholly above zero; and
- local donor `GT→CC` sensitivity under the candidate intervention retains at least 75% of its
  unintervened mean. This is a task-health floor, not evidence that the head is dedicated.

Multiplicity correction spans all frozen candidate-by-direction primary tests. Control contrasts are a
separate declared family. Candidate and control interventions always have identical dimensionality.

### 5.2 Routing versus transported content

For confirmed heads, projection patching is performed at layer 5:

- `QK`: corrupt Q at the donor query and corrupt K at all valid source tokens, with V left clean;
- `V`: corrupt V at all valid source tokens, with Q and K left clean;
- `QKV`: corrupt Q at the donor and corrupt K+V at all valid source tokens.

`QKV` must replay whole-head noising within `2e-5` per batch or that batch is invalid. The same
decomposition is repeated in the denoising direction. QK and V effects are non-additive because softmax
and downstream LayerNorm are nonlinear; they are reported as controlled interventions, not percentages
that must sum to one.

A head is called **routing-dominant** only if `(QK effect - V effect) ≥0.05` with a family-mean 95%
interval wholly above zero in both intervention directions. **Content-dominant** reverses that contrast.
All other confirmed heads are **mixed**. Attention maps remain descriptive even when a QK effect exists.

### 5.3 Final MLP interaction

Whole-MLP noising and denoising are confirmed separately. For each confirmed head, repeat head noising
while clamping the layer-5 MLP residual write at the donor to its clean value. The difference between the
ordinary head effect and this MLP-clamped effect measures controlled head×MLP interaction. Because the
MLP is downstream of attention with residual connections and LayerNorm, this is not claimed to be an
additive mediation fraction.

## 6. Biological source and specificity gates

Only confirmed causal heads enter source analysis. The scientific baseline is the known organization of
a canonical short intron: a local 5′ donor `GT`, its downstream paired 3′ terminal `AG`, and a broader
acceptor region that may include branch-point/polypyrimidine information. This study tests model
representations of those sequence-defined regions; it does not assign spliceosomal molecules.

### 6.1 Already-established behavior controls

GFS8 supplies the required model-level baseline: paired `AG→CC` and independent `AG→TT` effects,
near-null distance-matched non-partner-AG effects, a positive paired-vs-other-donor contrast, and a large
local acceptor manipulation check. These outcomes are replayed on the exact confirmation cohort.

### 6.2 Causal source localization

In the corrupt run, clean K+V projections for a confirmed head are restored separately at equal-width,
four-token source windows:

- paired acceptor: offsets `[-3,-2,-1,0]` around the acceptor label;
- donor boundary: offsets `[-2,-1,0,+1]` around the donor label;
- nearest eligible distance-matched non-partner `AG`, using the same offsets;
- immediate upstream and downstream four-token windows adjacent to the paired-acceptor window.

The full-valid-token K+V patch is the positive ceiling. An all-false patch is the no-op. A paired-acceptor
source claim requires positive clean-to-corrupt recovery with a 95% interval above zero and recovery
larger than every same-size control-window recovery with family-corrected inference. If donor-local wins,
the head is a donor-side carrier of already-assembled evidence, not a direct acceptor reader.

Q-only, K-only, and V-only patches are then run only at the winning frozen region. A key-only effect
supports address/routing; a value-only effect supports transported content; neither is interpreted without
the combined K+V positive result.

### 6.3 Specificity outputs

Every confirmed intervention also reports:

- its effect on the distance-matched other donor;
- its effect under the distance-matched non-partner `AG→CC` input;
- clean donor margin and local `GT→CC` destruction;
- the paired acceptor's own acceptor margin.

The head is “paired-acceptor-specific” only when its candidate effect exceeds both non-partner-AG and
other-donor effects with family-mean 95% intervals above zero. Otherwise it is described at the weaker
level supported (for example, a sequence-wide splice-pair signal).

Independent perturbations of branch points, polypyrimidine tracts, splice-site strength, or transcript
usage are explicitly outside v1. They require matched-edit shams and external predictors; attention to
those regions cannot name the encoded biological variable.

## 7. Representation convergence across seeds

For confirmed heads, clean-minus-corrupt donor-token head-context vectors are collected for the same
validation examples. Index-aligned cross-seed linear CKA and orthogonal-Procrustes cosine are descriptive
geometry. A shared causal role plus aligned geometry supports convergent realization; a shared causal role
without alignment supports functional convergence with seed-specific coordinates. Geometry alone cannot
promote a component.

## 8. Outcomes and stopping rules

The result is reported using the strongest row reached:

| Outcome | Licensed interpretation |
|---|---|
| no individual head; MLP positive | final-layer task adaptation is MLP-dominant or head-distributed |
| head positive; source unresolved | a causal final-layer carrier of paired-acceptor evidence |
| head positive; donor-local source | a donor-side carrier of already-assembled splice evidence |
| head positive; paired-acceptor K/V source | a final-layer head directly reads a paired-acceptor representation |
| QK dominant | routing-dominant carrier under the defined interventions |
| V dominant | content-dominant carrier under the defined interventions |
| mixed | inseparable routing/content role at this resolution |
| cross-seed causal + geometric replication | a convergently co-adapted final-layer representation |

An anomalous preflight or QKV replay freezes interpretation. A null is preserved and narrows the mechanism;
candidate rules, controls, windows, and thresholds are not revised to manufacture a sparse circuit.

