# Amendment & provenance closure — head_screen_v1 (2026-07-19)

Post-hoc audit of the completed all-96-head A/B/C/D screen (`head_effect_matrix.jsonl`,
`head_effect_summary.json`), triggered by Cam's integrity review. Nothing here required re-running the
screen; all fixes are analysis-side or documentation.

## 1. Split contamination (documented, resolved analytically)
- **What:** the runner (`run_all_head_screen.py::load_cohort`, GATE 12) excludes only
  `SEALED={chr8,chr9}`. It does **not** exclude the `val` chromosomes that `split_for` defines
  (`{chr11,chr16,chr19}`). So the matrix is **train (true discovery) + val**, not discovery-only, despite
  the config's `claim_boundary` saying "DISCOVERY genes".
- **Magnitude:** val = **1,067,904 / 4,069,632 rows (26%)**; 1,291 of 5,059 genes. chr8/9 **absent**
  (seal held; `sealed_excluded: true` in the manifest).
- **Resolution:** the primary re-analysis (`head_effect_reanalysis.json`) is **train-only**. Every
  headline is unchanged vs full: corr(C,D) 0.783 vs 0.797; corr(B,C) −0.075; C−D heads>0 89/96 vs 91/96;
  L3H1 donor B unchanged. ⇒ the contamination is a **labeling/hygiene** issue, not a result-changing one.
- **Consequence:** chr11/16/19 is now **touched** — no longer a clean internal held-out. **chr8/9 remains
  the sole untouched confirmation set** (which was always the design). Any confirmatory freeze goes to
  chr8/9 only; do **not** treat val as independent.

## 2. Provenance — CLOSED (evidence: Drive `gfm_head_screen_shards/run_manifest.json`)
| Item | Status |
|---|---|
| all 12 mechanical gates on the FULL run | `gate_failures: []` ✓ |
| chr8/9 sealed | `sealed_excluded: true` ✓ |
| determinism | `pythonhashseed: "0"` ✓ |
| precision/hardware | `float32 (TF32/AMP off)`, A100-80GB ✓ |
| resume integrity | `completed_shards` (10,598) == `expected_shards`, per-shard sha256 ✓ |
| scalar==vectorized, source-to-source no-op | repo test suite (9 tests) + preflight noop maxdiff 0.0 ✓ |
| GTF sha | `83fba3e9…13a453` (full) ✓ |
| checkpoint sha | **was truncated** `2ad914280149d`; real = `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`; on-disk checkpoint matches. Configs corrected. |

## 3. Corrections applied
1. **`group_faithfulness` renamed → `single_head_effect_concentration`** (`merge_head_screen.py`, docs).
   It is a rank-CONCENTRATION statistic of single-head |mean|, **NOT** a faithfulness curve — heads
   interact and single-head effects do not add. A true faithfulness-size curve needs **joint** patching of
   ranked head sets (the group-patch study).
2. **Checkpoint sha256** corrected to the full 64-char digest in both configs.
3. **GPU-hour projection note:** `benchmark_colab.py` projected **2.67 h** using **4,864 genes**, but the
   run had **10,598 examples** → actual **5.34 h** (`seconds: 19229.99`). Per-example time was slightly
   *better* than benchmarked; the projection simply used the wrong N (genes, not examples). Harmless.

## 4. Known gaps (not fixable without more caching / recompute)
- **A (zero-ablation) has no collateral-health** recorded → A "general-reconstruction" heads are candidates,
  not established general-use heads.
- **Per-example baseline / counterfactual behavioral values are not in the matrix** (only effects). Baselines
  needed to normalize effect sizes must be recomputed or re-exported.
- **Per-head MLM-health** not recorded (only condition-level). The cheap per-head `effect_collateral` is a
  proxy (max ≤7e-4), not a masked-MLM health gate.
- **Ontology covariates** (MaxEnt strength, composition, coding context, packet geometry) are **not** in the
  matrix and **not** per-example in `census_v1.json` (which is aggregate-only). `target_competitor_distance`
  and `site_type` ARE in the matrix. A covariate join requires per-example recomputation from sequence.

## 5. Artifact
- `head_effect_reanalysis.json` (in `~/Downloads`): **train-only**, gene-clustered (cluster=`gene_id`) 95%
  CIs on claim-bearing heads + aggregate; estimands = **signed for B, magnitude E|·| for C/D**; directional
  B (donor vs acceptor); C−D magnitude contrast; renamed concentration stat; provenance block.

## 6. Acceptor target-coordinate defect (second audit; 2026-07-19)

- **What:** historical `head_screen_v1` set the masked position to `w.target_win` for both site types.
  That coordinate is the donor G for a donor `GT`, but it is the coordinate immediately **after** an
  acceptor `AG`. The valid acceptor readout masks `w.target_win - 1` (the acceptor G). The runner then
  evaluated the logit for G at the wrongly masked first-exon coordinate.
- **Affected:** all 5,820 `alt3ss_acceptor` examples in the raw 96-head screen (2,234,880 A/B/C/D matrix
  rows). Acceptor head effects, donor-versus-acceptor head comparisons, pooled profiles that mix both site
  types, and the claimed reciprocal acceptor-side null are invalid.
- **Unaffected:** all 4,778 `alt5ss_donor` examples; the donor L3H1 B result; the earlier behavioral and
  specificity pilots, which use `CouplingScorer.acceptor_margin` and correctly mask `p-1`; chr8/9 remains
  untouched.
- **Code correction:** `target_base_position(w)` now owns the donor/acceptor coordinate conversion and has
  a direct regression test. Future screens record both the site anchor and actual masked coordinate.
- **Valid donor-only reanalysis:** `head_effect_donor_valid_reanalysis_v1.json` retains 3,549 train donor
  examples / 2,244 genes. L3H1 B = +0.114 [0.098,0.130] by gene-level normal interval. C and D remain the
  same head family (`r=0.804` signed; `r=0.942` magnitude); B remains distinct from C (`r=0.023`).
- **Claim correction:** “SpliceBERT routes acceptor information into donor reconstruction” remains
  supported on the donor readout. “The reciprocal is null/not the same circuit” is **withdrawn**—the
  reciprocal acceptor experiment has not been validly run.

## 7. Exact joint head/MLP closure

- `group_mlp_closure_v1.json` used 400 train-only donor genes, exact scrub-to-corrupt joint patches, all
  six individual MLP writes, and all six MLPs jointly. Mechanical source-to-source gates were exactly 0.
  Its B/MLP results are valid. Its C/D head order inherited the invalid pooled ranking and is superseded
  for ranking-dependent interpretation.
- `group_head_rerank_valid_donor_v2.json` changes only the C/D ranking input to the valid donor-only
  reanalysis and reruns the exact C/D head curves. L4H12→L3H1→L0H4 are the top post-hoc three. Their joint
  C−D magnitude excess is +0.0176 [0.0016,0.0374] on the same discovery sample; at k≥5 every CI crosses
  zero, and the all-96/behavioral excess is +0.0191 [−0.0244,0.0669]. This is a small discovery candidate,
  not a licensed competition-specific module.
- B is concentrated under the exact intervention: L3H1 alone = +0.0857 [0.0492,0.1280] versus behavioral
  B +0.1015; the top 10 = +0.1007. All 96 exactly reproduce the behavioral effect. This validates the B
  positive-control circuit but does not rescue competition specificity.
- No individual MLP or all-MLP C−D magnitude CI excludes zero. Attention and MLP patches represent
  overlapping causal cuts through a sequential network and must not be added.

**Disposition:** close the competition branch as exploratory/instrument-limited. Do not spend sealed data
or further Colab credits on a competition confirmation. Return to the multi-context atlas described in
`RESEARCH_SPINE.md`.
