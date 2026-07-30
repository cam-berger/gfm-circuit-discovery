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
| label-free biological sequence package | `scripts/splicebert/export_gtex_sparse_transcoder_sequence_package.py` | `results/splicebert_gtex_finetune_shift/sparse_transcoder_sequence_package_v1.json` |
| six-view final-MLP activation capture | `scripts/splicebert/run_gtex_sparse_transcoder_activation_capture.py` | `results/splicebert_gtex_finetune_shift/sparse_transcoder_activation_capture_v1.json` plus `.npz` |
| frozen sparse final-MLP grid | `scripts/splicebert/run_gtex_sparse_transcoder_training.py` | `results/splicebert_gtex_finetune_shift/sparse_transcoder_training_v1.json` plus `.pt` (`INSTRUMENT_UNRESOLVED`) |
| exact-dose held-out biological construction | `scripts/splicebert/freeze_gtex_native_biological_features_confirmation.py` | `results/splicebert_gtex_finetune_shift/native_biological_features_confirmation_construction_v1.json` (`STOPPED_MINIMUM_COHORT`; no model output) |
| named-carrier native-margin gate v1 | `scripts/splicebert/run_gtex_positional_native_margin_gate.py` | `results/splicebert_gtex_finetune_shift/positional_native_margin_gate_v1.json` (`STOPPED_INTERVENTION_PREFLIGHT`; full period-3 deletion unhealthy) |
| health-preserving named-carrier gate v2 | `scripts/splicebert/run_gtex_positional_native_margin_gate_v2.py` | `results/splicebert_gtex_finetune_shift/positional_native_margin_gate_v2.json` (period 3 and ≈10.63 local-margin screens PASS; coupling screens null) |
| Jacobian-SAE pilot and capacity repairs | `scripts/splicebert/run_gtex_jacobian_sae_instrument.py` | `results/splicebert_gtex_finetune_shift/jacobian_sae_instrument_v1.json`, `v2.json`, and `v3.json` plus `.pt` (v1/v2 unresolved; asymmetric v3 representation gate PASS) |
| five-seed/three-initialization J-SAE replication | `scripts/splicebert/run_gtex_jacobian_sae_replication.py` | `results/splicebert_gtex_finetune_shift/jacobian_sae_replication_v1.json` plus `.pt` (representation and stability gates PASS) |
| complete-write and residual-delta development gates | `scripts/splicebert/run_gtex_jacobian_sae_behavior_preflight.py`; `scripts/splicebert/run_gtex_jacobian_sae_delta_preflight.py` | `results/splicebert_gtex_finetune_shift/jacobian_sae_behavior_preflight_v1.json`; `jacobian_sae_delta_preflight_v1.json` (both unresolved; target deltas preserved, absolute/sham calibration fails) |
| outcome-blind fresh J-SAE cohort and activation capture | `scripts/splicebert/build_gtex_jacobian_sae_fresh_cohort.py`; `scripts/splicebert/capture_gtex_jacobian_sae_fresh_activations.py` | `results/splicebert_gtex_finetune_shift/jacobian_sae_fresh_cohort_v1.json`; `jacobian_sae_fresh_activation_capture_v1.json` plus `.npz` |
| gene-fresh residual-delta confirmation | `scripts/splicebert/run_gtex_jacobian_sae_fresh_delta_confirmation.py` | `results/splicebert_gtex_finetune_shift/jacobian_sae_fresh_delta_confirmation_v1.json` (`INSTRUMENT_UNRESOLVED`; all 30 target paths pass, 4 sham cells and 9 numerical decoder-sum cells fail) |
| larger paired-delta J-SAE pilot | `scripts/splicebert/run_gtex_jacobian_sae_large.py` | `results/splicebert_gtex_finetune_shift/jacobian_sae_large_v1.json` plus `.pt` (`INSTRUMENT_UNRESOLVED`; seed-2020 pilot stops the family before classifier opening) |

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

The first preregistered sparse decomposition is a preserved instrument failure, not a biological
negative. Its planted sparse detector/writer control passes, but none of the 12 width/activity settings
reaches the all-view `0.95` write-delta FVE and optimizer-stability floors. The numerically strongest
setting, `w128_k32`, has mean write-delta FVE `0.877` and worst-view FVE `0.530`. More importantly, every
joint fit routes all five fine-tuned views through private capacity while leaving their shared and
tuned-only blocks inactive. The frozen behavioral preflight was therefore not opened. This stops every
sparse-latent, feature-label, and steering claim from that instrument while leaving the exact native-unit
route available only if a fresh biological-sequence construction passes its frozen gate.

That exact-dose construction has now run on the 200 held-out genes and stops before model inference.
Candidate-only low/high synonymous recodings remain feasible for 101–114 genes per assignment, but the
complete candidate-plus-three-control sets shrink to 23 donor/DAZAP1, 33 donor/SFPQ, 22 acceptor/UNK,
and 21 acceptor/DAZAP1 genes once every control must match the candidate's exact low/high nucleotide
dose and all discovery-frozen nuisance calipers. Every assignment misses the required 60 examples from
45 genes. The correct interpretation is an isolation/feasibility failure of this confirmatory assay,
not a negative result about an RBP. No held-out classifier outcome, native-unit ranking, steering test,
external candidate lock, or reporter outcome was opened.

The positional-spectrum assay answers a separate continuity question. Both learned positional clocks
survive supervised fine-tuning: period 3 peaks at `2.997` nt in every tuned checkpoint and keeps
`1.055–1.063×` normalized power, while the ≈10.63 carrier peaks at `10.649–10.651` nt and keeps
`0.819–0.849×`. A post-hoc energy-ranked sensitivity shows that their dominant rank-2 planes are also
nearly unchanged (`0.988–0.992` and `0.996–0.997` overlap). The lower preregistered full-rank overlap is
driven by weak coefficient directions and is preserved as the primary geometry result.

The full scan finds two adjacent ≈`2.21–2.28`-nt bands that gain `2.74–2.76×` normalized power. They pass
all intervention-health controls but are causally null for native donor coupling: deletion and base-band
restoration effects are on the order of `10⁻⁴`, mixed across seeds, indistinguishable from three
power-matched bands, and have `q=0.9704`. Thus fine-tuning preserved the old positional carriers and added
high-frequency weight structure that the tested classifier behavior does not use.

The named carriers can be tested without pretending that weak other-frequency bands are comparable.
Three same-band phase controls preserve carrier power while matching the exact table-space intervention
dose. A full period-3 deletion first stopped because it reduced the frozen local-task health response
below 80% in four seeds. On the next 64 genes, health-preserving 50% period-3 attenuation lowers native
donor/acceptor margins by `1.657/1.724`, while full ≈10.63 removal lowers them by `1.530/0.983`.
Base-model bands rescue nearly the same amount, and attenuation-minus-phase contrasts remain
`1.060–1.543`; all local effects are positive in all five seeds with `q=0.00015`. Paired-site donor
coupling remains null for both carriers. The clocks are therefore functional for local site
classification but separable from the final-layer coupling route.

The larger paired-delta J-SAE repair is also a preserved instrument failure. On the frozen seed-2020
pilot, the 512/128 TopK-16 pair reconstructs held-out output writes at FVE `0.9980`, preserves the
reconstructed-input MLP output at `0.9928`, and reduces median E90 from `62` to `10` relative to the
matched paired-delta no-Jacobian control. Analytic/autograd and finite-difference Jacobians agree.
However, held-out input FVE is only `0.9408`, five input-delta strata miss their frozen floors, and
AuxK supplies `69.6%` of late loss. The preregistered pilot rule therefore stops the other
initializations, checkpoints, base comparison, and fresh behavior cohort. Enlarging the dictionary and
adding paired-delta training improved output-path fidelity and graph concentration but did not create a
healthy, complete sparse input representation.
