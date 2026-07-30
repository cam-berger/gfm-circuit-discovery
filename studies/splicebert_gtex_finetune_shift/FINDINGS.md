# SpliceBERT GTEx fine-tuning-shift findings

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

This is the concise entry point for the GTEx fine-tuning-shift study. Exact licensed wording is controlled
by [`CLAIMS.md`](CLAIMS.md); run provenance is in [`RESULTS.md`](RESULTS.md) and the
[result index](../../results/splicebert_gtex_finetune_shift/RESULTS_INDEX.md).

## Status at a glance

| Finding | Evidence reached | Current interpretation |
|---|---|---|
| Readout shift, not coupling erasure | fixed-MLM behavior plus native-head verification | Fine-tuning makes the original MLM readout broadly incompatible, while the released classifier still uses paired-acceptor information. |
| Late causal task representation | five-seed component interventions and path patching | Several layer-5 attention routes converge through the final MLP into a compact, checkpoint-rotated splice-classification code. |
| Sequence-context dependence | exact matched shuffles, depth localization, and head/MLP patching | Two nonlocal sequence-order regions causally affect native margins; their positive biological identities remain unresolved. |
| Positional continuity | Fourier analysis plus controlled native-margin surgery | Period 3 and ≈10.63 survive fine-tuning and support local donor/acceptor evidence, but not the paired-site coupling effect. |
| Sparse-feature boundary | multiple SAE/J-SAE preflights with matched controls | Downstream final-MLP differences are highly compressible, but no validated sparse explanation of the upstream base→tuned shift or biological feature set has been established. |
| Biological bridge | outcome-blind exact-dose construction | The named RBP-candidate confirmation cohort was infeasible under frozen controls, so no biological confirmation outcome was opened. |

## 1. Fine-tuning changes the readable computation

Through the inherited masked-LM readout, base donor–acceptor coupling falls from `+0.958` to
`−0.029…+0.068` across the five fine-tuned checkpoints, with a family shift of
`−0.932 [−1.038,−0.835]` (GFS4). General MLM health also falls, so this is readout incompatibility rather
than evidence that the encoder erased coupling.

The released classifier resolves that ambiguity. Paired-acceptor destruction lowers native donor evidence
by `0.646 [0.563,0.733]` logit units, positive in every seed, and exceeds non-partner controls (GFS8). The
classifier-aligned coupling signal emerges primarily in the final encoder block (GFS9).

## 2. A many-head to final-MLP causal route

`L5H6` is a reproducible, partner-specific causal carrier whose effect is value-leaning but mixed and whose
source is distributed (GFS11–GFS13). It is one route, not a universal context head.

Independent context localization identifies donor `[-80,−48)` and acceptor `[+16,+48)` sequence-order
dependencies (GFS14). Their information transfers toward the classified token across the last two layers
and converges almost entirely on the final MLP (GFS15–GFS16).

The complete layer-5 census confirms `L5H14` and `L5H8` for both contexts, with donor-selected `L5H12` and
acceptor-selected `L5H11` as additional assignments (GFS17). Joint triples carry only `28.6–34.4%` of
behavior but route `98.0–100.0%` of their effect through their induced final-MLP write (GFS18). This is a
structured, incomplete upstream core with directed MLP mediation.

## 3. Compact output geometry, mesoscale implementation

Raw seed-2020 directions do not transport at substantial magnitude across independently fine-tuned seeds.
When each seed is fit separately and aligned using discovery-only orthogonal maps, rank-4 subspaces recover
about 99% of the full-MLP effect on reused evaluation partitions (GFS19). The result is exploratory:
functional convergence does not imply a universal raw vector.

The exact 2,048-unit census shows how that compact write is implemented. Top-64 native-unit groups carry
roughly 64–81% of the effect; 256 units cross the 80% threshold for both endpoints (GFS20). Hundreds of
native units therefore synthesize a one-to-four-dimensional task write. The units are not licensed as
monosemantic features.

## 4. Biological sequence features remain unresolved

The confirmed context regions establish dependence on higher-than-dinucleotide sequence order, not an RBP,
branch point, polypyrimidine tract, enhancer, codon, or spliceosomal identity. Sparse discovery proposed
sequence candidates, but the shared-index instrument collapsed (GFS21).

An outcome-blind exact-dose construction then found too few genes with candidate perturbations and all
three matched controls: 21–33 complete genes per assignment versus a frozen minimum of 45 (GFS22). No
held-out classifier output, steering result, external candidate lock, or reporter outcome was opened.

## 5. Inherited positional carriers remain functional

Period 3 remains at `2.997` nt and ≈10.63 at `10.649–10.651` nt in all tuned checkpoints (GFS23).
Fine-tuning also adds ≈2.2-nt spectral bands, but controlled surgery finds those gains causally null for
native donor coupling (GFS24).

Health-preserving surgery shows that period 3 and ≈10.63 both support local donor and acceptor margins and
can be rescued by the corresponding base-model bands. Neither detectably changes paired-acceptor donor
coupling (GFS25). Fine-tuning therefore reuses inherited positional infrastructure for local site
classification while expressing the paired-site computation through a separate late route.

## 6. What the sparse assays do and do not establish

The J-SAE series validates exact Jacobian algebra and repeatedly preserves task-relevant final-MLP output
differences (GFS26–GFS32). Earlier dictionaries either collapse, fail absolute/sham calibration, or remain
descriptive rather than feature-interpretable.

The larger paired-delta repair improves output fidelity and graph concentration but fails input coverage and
training-health gates (GFS33). The direct base→fine-tuned model-difference J-SAE sharpens the boundary:
state/parameter/total final-MLP write differences reach FSEE `0.9985/0.9991/0.9995`, but the upstream state
shift reaches only `0.7853`, all eight input-contrast strata fail, and the graph advantage misses both frozen
floors (GFS34).

The strongest sparse conclusion is therefore **distributed upstream change compressed into a compact
downstream difference**. Feature identity, a sparse directed difference graph, biological semantics, and
steering remain closed.

## First missing bridges

1. A new sparse or structured instrument that faithfully represents the upstream base→fine-tuned state
   change and passes perturbation-stratified controls.
2. Fresh, outcome-sealed replication of the aligned compact output geometry and native-unit organization.
3. A feasible biological counterfactual family with exact-dose candidate and nuisance controls.
4. Causal identification of which components inject the inherited positional carriers into local
   classification evidence.
