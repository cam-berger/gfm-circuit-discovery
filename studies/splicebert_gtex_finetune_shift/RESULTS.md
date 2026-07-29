# Results index: SpliceBERT GTEx fine-tuning shift

> **Tier: T1 — canonical.** Current, authoritative. Read this.

The fixed-MLM gate is closed, while the native-classifier behavior and its gated representation bridge
are complete. Runtime-derived checkpoints live under ignored `data/models/` paths and must carry
`transplant.json` plus `transplant.manifest.json`.

Immutable results:

| Stage | Runner | Result |
|---|---|---|
| checkpoint/transplant preflight | `scripts/splicebert/run_gtex_finetune_preflight.py` | `results/splicebert_gtex_finetune_shift/preflight_v1.json` (PASS) |
| seed-2020 mechanics smoke | unchanged `scripts/splicebert/run_sealed_test.py` | `results/splicebert_gtex_finetune_shift/seed-2020/smoke_sealed_val_v1.failed.json` (stopped at behavior gate) |
| five-seed behavior validation | `scripts/splicebert/run_gtex_coupling_shift.py` using the unchanged canonical behavior/controls | `results/splicebert_gtex_finetune_shift/five_seed_behavior_v1.json` |
| native classifier preflight v1 | `scripts/splicebert/run_gtex_native_classifier_preflight.py` at the retracted first-intronic-base coordinate | `results/splicebert_gtex_finetune_shift/native_classifier_preflight_v1.json` (FAIL; preserved instrument error) |
| corrected native classifier preflight v2 | same runner, last-exonic-base donor label frozen before rerun | `results/splicebert_gtex_finetune_shift/native_classifier_preflight_v2.json` (51/51 PASS) |
| five-seed native classifier coupling | `scripts/splicebert/run_gtex_native_classifier_coupling.py` | `results/splicebert_gtex_finetune_shift/native_classifier_coupling_v1.json` (behavior gate PASS) |
| gated layerwise representation bridge | `scripts/splicebert/run_gtex_native_representation_bridge.py` | `results/splicebert_gtex_finetune_shift/native_representation_bridge_v1.json` (representation candidate) |
| native final-block intervention preflight | `scripts/splicebert/run_gtex_native_mechanism_preflight.py` | `results/splicebert_gtex_finetune_shift/native_mechanism_preflight_v1.json` (PASS) |
| train component discovery and freeze | `scripts/splicebert/run_gtex_native_mechanism_discovery.py` | `results/splicebert_gtex_finetune_shift/native_mechanism_discovery_v1.json` (`L5H6`, `L5H1`, `L5H3` frozen) |
| five-seed component confirmation | `scripts/splicebert/run_gtex_native_mechanism_confirmation.py` | `results/splicebert_gtex_finetune_shift/native_mechanism_confirmation_v1.json` (`L5H6` and final MLP confirm) |
| QK/V, MLP-interaction, source, and specificity | `scripts/splicebert/run_gtex_native_mechanism_role_source.py` | `results/splicebert_gtex_finetune_shift/native_mechanism_role_source_v1.json` (V-leaning mixed; distributed source) |
| native context-localization preflight | `scripts/splicebert/run_gtex_native_context_preflight.py` | `results/splicebert_gtex_finetune_shift/native_context_preflight_v1.json` (21/21 PASS) |
| train equal-width context discovery | `scripts/splicebert/run_gtex_native_context_discovery.py` | `results/splicebert_gtex_finetune_shift/native_context_discovery_v1.json` (regions and shams frozen) |
| five-seed context confirmation | `scripts/splicebert/run_gtex_native_context_confirmation.py` | `results/splicebert_gtex_finetune_shift/native_context_confirmation_v1.json` (both 32-nt regions confirm) |
| delivered-dose correction audit | `scripts/splicebert/audit_gtex_native_context_confirmation.py` | `results/splicebert_gtex_finetune_shift/native_context_confirmation_audit_v1.json` (full regions licensed; one acceptor half quarantined) |
| source-to-target depth trace | `scripts/splicebert/run_gtex_native_context_depth_trace.py` | `results/splicebert_gtex_finetune_shift/native_context_depth_trace_v1.json` (material arrival after layer 4) |
| frozen `L5H6`/MLP component bridge | `scripts/splicebert/run_gtex_native_context_component_bridge.py` | `results/splicebert_gtex_finetune_shift/native_context_component_bridge_v1.json` (MLP transfers; `L5H6` does not) |
| complete train-only L5 context-head census | `scripts/splicebert/run_gtex_native_context_head_census_discovery.py` | `results/splicebert_gtex_finetune_shift/native_context_head_census_discovery_v1.json` (all 16 heads; candidates/controls frozen) |
| fresh five-seed context-head confirmation | `scripts/splicebert/run_gtex_native_context_head_census_confirmation.py` | `results/splicebert_gtex_finetune_shift/native_context_head_census_confirmation_v1.json` (`L5H14`/`L5H8` shared; `L5H12` donor-selected; `L5H11` acceptor-selected) |
| fresh low-rank direction freeze | `scripts/splicebert/run_gtex_native_context_lowrank_discovery.py` | `results/splicebert_gtex_finetune_shift/native_context_lowrank_discovery_v1.json` (seed-2020 endpoint bases/directions frozen) |
| joint heads, head→MLP path, and low-rank confirmation | `scripts/splicebert/run_gtex_native_context_joint_lowrank_confirmation.py` | `results/splicebert_gtex_finetune_shift/native_context_joint_lowrank_confirmation_v1.json` (joint/path confirm; completeness and universal direction do not) |
| per-seed subspace/alignment freeze | `scripts/splicebert/run_gtex_native_context_per_seed_alignment_discovery.py` | `results/splicebert_gtex_finetune_shift/native_context_per_seed_alignment_discovery_v1.json` plus `.npz` |
| cross-seed causal alignment evaluation | `scripts/splicebert/run_gtex_native_context_per_seed_alignment_evaluation.py` | `results/splicebert_gtex_finetune_shift/native_context_per_seed_alignment_evaluation_v1.json` plus `.npz`; rank-4 verdict correction in `native_context_per_seed_alignment_evaluation_audit_v1.json` |
| complete 2,048-unit MLP discovery census | `scripts/splicebert/run_gtex_native_context_final_mlp_unit_discovery.py` | `results/splicebert_gtex_finetune_shift/native_context_final_mlp_unit_discovery_v1.json` (context/head groups and controls frozen) |
| native-unit completeness and head→unit evaluation | `scripts/splicebert/run_gtex_native_context_final_mlp_unit_evaluation.py` | `results/splicebert_gtex_finetune_shift/native_context_final_mlp_unit_evaluation_v1.json` plus `.npz` |

The five-seed artifact includes all 400 paired per-example values for the base and each seed, gene-grouped
bootstrap intervals, 9,999 gene-level sign flips with add-one p-values, BH correction across the 15
seed-by-contrast tests, and the three family-mean contrasts. It is verification tier because the comparison
runner was finalized after the seed-2020 mechanics smoke.

Every target artifact must be added here and to `results/MANIFEST.{csv,json}` only after its status and
claim scope are decided.

The native behavior result resolves the central ambiguity in GFS4: coupling was not erased from the
released fine-tuned model. It remains visible to the supervised head, while the original MLM readout
becomes broadly incompatible. The layerwise bridge places the classifier-aligned change primarily at the
last encoder block.

The gated causal follow-up now resolves part of that block. Seed-2020 train discovery nominated three
heads; untouched five-seed validation confirms only `L5H6`. Its paired-acceptor-dependent output causes
`0.081` damage and `0.142` recovery and retains 99.3% of local donor sensitivity. The final MLP carries
`0.640/0.643` damage/recovery, and clamping its clean write removes nearly all of `L5H6`'s damage,
establishing a controlled head×MLP interaction.

`L5H6` is value-leaning but misses the preregistered two-direction threshold for a content-dominant label.
It is partner-specific, yet no equal-width source window accounts for enough of its signal: full-sequence
K+V restoration recovers `0.136`, compared with `0.031` for the best four-token window. The result therefore
defines the co-adapted representation as a partner-specific distributed value signal transformed by the
final MLP—not a direct terminal-`AG` attention head.

The context-localization extension makes that representation more general and more modular. Exact
directed-dinucleotide-preserving shuffles establish two nonlocal sequence-order dependencies: donor
`[-80,−48)` contributes `0.808 [0.647,0.971]`, and acceptor `[+16,+48)` contributes
`1.054 [0.855,1.266]`. Their held-out candidate-minus-sham contrasts are `0.799` and `0.816`; both
replicate in every seed and under a second shuffle draw. These coordinates are sequence-defined and are
not yet licensed as exon enhancers, codons, or RBP motifs.

Clean-state patching shows the information remains locally recoverable through layer 3, begins material
arrival at the classifier target after layer 4, and is fully integrated after the final block. The final
MLP carries essentially all of both context effects, but the coupling-selected `L5H6` carries neither
under the frozen control rule. The resulting model is therefore **multiple late attention routes
converging on a shared final task transformation**, not one universal final-layer context head.

The completed L5 census identifies those routes at one-head resolution. The all-16 seed-2020 discovery
found 8/16 donor and 6/16 acceptor heads above the bidirectional eligibility floor, then froze the top
three candidates per endpoint. A fresh internal gene split, excluding every previously exposed train
gene and every historical validation/test chromosome, confirms all six assignments across the five
checkpoints. `L5H14` and `L5H8` are shared donor/acceptor context carriers; `L5H12` was selected and
confirmed for donor context, and `L5H11` for acceptor context. Damage ranges from `0.057` to `0.142`, and
recovery from `0.079` to `0.204`; every frozen candidate effect is positive in all five seeds and exceeds
its three equal-size controls. Non-top-three eligible heads remain exploratory, and positive descriptive
opposite-endpoint measurements prevent a stronger specialization claim. These are causal carriers, not
yet a complete circuit.

The frozen joint intervention now quantifies that boundary. The donor triple causes `0.356/0.411`
damage/recovery and the acceptor triple `0.347/0.416`, each well above three matched-size control sets.
They recover only `28.6–34.4%` of behavior, but their joint effects are mildly super-additive relative to
the singleton sum and every leave-one-out marginal is positive. They are therefore complementary causal
cores inside the broader head population, not redundant or complete circuits. Head→MLP path patching is
decisive: the induced MLP write reproduces `98.0–100.0%` of each triple effect, whereas the controlled
residual-bypass route contributes at most `2.1%`.

The low-rank arm bounds the coordinate-level interpretation. The seed-2020 discovery matrices are
spectrally concentrated, but the frozen one-dimensional task direction fails the five-seed sign gate.
Ranks 4/16/64 have detectable effects above dimension-matched random subspaces, yet even rank 64 has
ratio estimates of only `28.9–33.9%` of the full-MLP effect and no tested rank clears the frozen 50%
criterion. Seed-level
estimates show that most magnitude comes from the checkpoint on which the basis was fit. The publishable
mechanism is thus a seed-reproducible **functional path**—multiple heads into the final MLP—not a single
shared raw-coordinate direction. A per-seed fit plus preregistered subspace alignment is required before
claiming geometric convergence.

That alignment follow-up succeeds at exploratory tier on the reused partitions. Every checkpoint places
more than 99.6% of discovery MLP-delta energy in four dimensions. Discovery-only orthogonal maps let
cross-seed rank-4 bases recover about 99% of the full MLP causal effect, and transport the actual
example-conditioned delta at about 95–101%. Raw transport fails recovery. All aligned ordered-pair
endpoint estimates are positive and exceed dimension- or norm-matched controls. The preserved evaluation
output contains a summary-only rank-selection bug; its immutable correction audit applies the frozen
rank-4 gate without changing any outcome.

The exact native-unit census then distinguishes low output rank from implementation sparsity. Top-64
post-GELU groups carry `63.9–81.1%` across endpoint/direction combinations, while top-256 groups carry
`91.4–109.4%` and first clear the frozen 80% gate for both context effects. Head-responsive populations
are more concentrated for acceptor (80% at 64) than donor (80% at 256), reproduce the prior head→MLP
path within `2.1e-6`, and overlap the corresponding whole-context top-64 groups by 84–86%. The most
precise current implementation account is therefore a **mesoscale overcomplete native-unit population
writing a compact, checkpoint-rotated task code**. It remains exploratory because the alignment and unit
arms reused GFS18's opened gene partitions.
