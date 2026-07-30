# Preregistration: exact final-MLP native-unit census and head→unit mediation

> **Tier: T2 — provenance.** Frozen after GFS19 but before capturing a post-GELU
> intermediate activation or evaluating any native MLP unit. GFS14–GFS19 fix the two
> sequence effects, five checkpoints, final MLP stage, L5 head triples, and
> checkpoint-specific compact output code. This study reuses the already-opened GFS18
> partitions and is exploratory.
>
> **Pre-unit-outcome MPS precision correction (2026-07-29):** two discovery
> invocations stopped at the all-unit algebra preflight after seed-2020 donor capture
> and before computing a one-unit effect or ranking. CPU float64 reconstruction from
> the captured float32 activations and weights differed from the MPS dense-kernel write
> by at most `1.59e-4`; the corresponding clean-minus-corrupt reconstruction differed
> by `2.08e-4`. Native final composition itself matched within `1.28e-6`, locating the
> discrepancy to backend matrix-multiplication precision rather than coordinates or
> hooks. The write/all-unit threshold below is therefore `5e-4`; the independent
> LayerNorm/classifier threshold remains `1e-5`. No unit effect, ranking, group,
> control, or evaluation outcome had been computed, and no scientific threshold or
> statistical family changed.

## 1. Question

GFS19 shows that a seed-specific rank-1 direction carries most of the context effect
and an aligned rank-4 subspace carries nearly all of it. Low output rank does not imply
a sparse implementation. The final SpliceBERT MLP has 2,048 post-GELU native units
whose writes sum exactly through a 512 x 2,048 output matrix.

This study asks:

1. how many native units are causally required to build the compact output code;
2. whether a small unit set or a distributed overcomplete population implements it;
3. which native units receive the causal changes induced by the frozen L5 head sets;
4. whether the individual confirmed heads feed the same or different unit groups.

Units are architectural coordinates, not biological features. A unit group can mediate
a model behavior without having a monosemantic biological interpretation.

## 2. Frozen cohorts, behavior, and components

Reconstruct the exact GFS18 discovery (`n=128`) and evaluation (`n=200`) cohorts,
including the original six prior-gene exclusions, namespace, within-gene cohort tags,
counterfactual shuffles, targets, and unopened chromosomes. Example IDs must replay
exactly and discovery/evaluation genes must not overlap.

Required checkpoint family: 2020–2024. Endpoints:

- donor `[-80,-48)` shuffle, donor-class margin at `donor_pos - 1`;
- acceptor `[+16,+48)` shuffle, acceptor-class margin at `next_acceptor_pos`.

Frozen L5 head sets:

- donor: `{L5H14,L5H8,L5H12}`;
- acceptor: `{L5H11,L5H14,L5H8}`.

All three singleton heads and the full triple enter the head→unit assay. No head is
reselected.

## 3. Exact unit definition and preflight

`a` is the post-activation output of `BertIntermediate` in layer 5. The MLP dense write
is exactly

`m = a W_out^T + b_out`.

Required gates:

1. existing final-layer mechanism preflight remains `PASS`;
2. architecture is 6 layers, 16 heads, hidden width 512, intermediate width 2,048;
3. captured intermediate activations and output matrices are finite;
4. reconstructing `m` from all 2,048 captured units matches the hooked MPS dense write
   within the frozen backend tolerance `5e-4`;
5. a zero-unit group is an exact no-op;
6. all 2,048 units reproduce the complete MLP-write intervention within `5e-4`;
7. NumPy final LayerNorm/classifier composition matches native margins within `1e-5`;
8. discovery and family evaluation behaviors pass mean `>=0.05`, interval above zero,
   corrected `q<=0.05`, and positive checkpoint signs.

Any failure stops interpretation.

## 4. Complete discovery census and frozen groups

For every seed, endpoint, example, and unit `j`, compute the exact one-unit
interchange, not a gradient approximation:

`delta_m_j = (a_clean,j - a_corrupt,j) W_out[:,j]`.

Noising subtracts `delta_m_j` from the clean write; recovery adds it to the corrupt
write. Both pass through the real final LayerNorm and classifier. Examples are
averaged within gene. Units are ranked by

`min(mean_gene_damage_j, mean_gene_recovery_j)`,

descending with unit index as the deterministic tie break. This is discovery-stage
causal ranking; it does not itself confirm a unit.

Freeze nested top groups at sizes

`k in {1,4,16,64,256,1024}`,

plus all 2,048 units as the exact ceiling. For each non-ceiling group freeze three
equal-count controls chosen only by nearest output-column norm among non-candidate
units. Controls never inspect activations or outcomes. At `k=1024`, the complement is
the only disjoint equal-count set and repeated draws may coincide.

## 5. Evaluation: sparsity/completeness curve

At each seed and endpoint, apply its frozen unit groups to the instance-specific
evaluation activation delta. Report damage, recovery, ratio to the complete MLP
intervention, and the three output-norm-matched controls.

Families:

1. 28 candidate effects: two endpoints x seven sizes (including 2,048) x two
   directions;
2. 24 candidate-minus-control contrasts: two endpoints x six controlled sizes x two
   directions.

Use 5,000 gene-grouped bootstrap draws, 9,999 exact add-one gene sign flips, and BH
correction within each enumerated family. Checkpoint-family values average the
seed-specific interventions within example before gene inference.

The minimum tested `k` whose damage and recovery ratio lower intervals reach 0.50 is
the 50%-completeness size; the analogous 0.80 crossing is the 80%-completeness size.
No interpolation is permitted. A sparse native implementation requires `k<=64` at
80%; a distributed implementation is licensed if the 50% crossing exceeds 256 or the
curve has no sharp knee. Intermediate outcomes are reported without forcing either
label.

## 6. Discovery and evaluation: head-induced unit paths

On discovery genes, target-token patching of each frozen triple produces exact
post-GELU changes:

- noising: `a_clean - a_patched-clean`;
- recovery: `a_patched-corrupt - a_corrupt`.

Apply each unit's induced write alone at the unpatched baseline and rank units by the
same bidirectional rule. Freeze nested head-induced groups at
`k in {1,4,16,64,256,1024}` with output-norm-matched controls.

On evaluation genes, replay the full triple and each singleton head. For every frozen
triple-derived unit group:

1. transplant only that group's actual head-induced MLP write into the unpatched
   baseline;
2. measure mediated damage/recovery;
3. compare the triple path with matched unit controls;
4. report each singleton head's mediated effect through the same group.

The complete 2,048-unit head-induced write is the positive path ceiling and must
reproduce the prior head→MLP indirect path within `1e-4`.

Families:

1. 28 triple→unit mediated effects across endpoints, seven sizes, and two directions;
2. 24 triple→unit candidate-minus-control contrasts across controlled sizes;
3. singleton-head paths are descriptive ownership measurements and receive no
   confirmatory component label.

Inference and correction match Section 5. The minimum 50% and 80% sizes are ratios to
the complete head-induced MLP path, not to total behavior. Because residual and
LayerNorm nonlinearities remain, paths are not assumed additive across units or heads.

## 7. Licensed outcomes

| Outcome | Licensed interpretation |
|---|---|
| context curve crosses 80% by 64 | a sparse native-unit implementation writes the compact task code |
| context curve needs >256 units | a low-rank output code is synthesized by a distributed overcomplete unit population |
| head-induced group crosses 80% by 64 | the frozen heads target a sparse downstream unit channel |
| head-induced group needs >256 | head inputs are transformed through a distributed native-unit population |
| singleton paths concentrate in different groups | candidate division of labor among upstream heads |
| singleton paths concentrate in the same groups | candidate convergence onto a shared downstream population |

The study establishes only causal implementation at native-unit resolution. It does
not name motifs, RBPs, spliceosomal factors, or reusable biological concepts. A
transcoder/sparse autoencoder is authorized only if the exact native-unit curves stay
distributed or strongly polysemantic; it would begin a new preregistered study.
