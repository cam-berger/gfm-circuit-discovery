# Head–Context Atlas completion audit

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

> Environment note (2026-07-20): commands below preserve the original `gfm-thrust1` execution record.
> New reproduction uses the canonical `uv` environment in [`../../REPRODUCE.md`](../../REPRODUCE.md).

This checklist maps the program objective to authoritative evidence. A row is complete only when the
named artifact exists, its independent audit is current and passing, and the claim ledger uses no
stronger language than that evidence permits. `PENDING` is not a failure; it prevents partial execution
from being mistaken for program completion.

## Historical alternative-site / competition branch

| Requirement | Status | Authoritative evidence |
|---|---|---|
| Preserve raw provenance and disclose the actual discovery/validation split | **PASS** | `results/splicebert_head_context_atlas/competition_branch_closure_audit_v1.json` → `raw_provenance_and_split` |
| Withdraw the coordinate-invalid acceptor rows and every pooled/reciprocal claim that depends on them | **PASS** | closure audit → `coordinate_defect_disposition`; `AMENDMENT_HEAD_SCREEN.md`; `CLAIMS.md` HCA5/HCA8–HCA10 |
| Recompute signed B separately from magnitude C/D and paired magnitude C−D | **PASS** | closure audit → `corrected_signed_and_magnitude_reanalysis`; `head_effect_donor_valid_reanalysis_v1.json` |
| Test exact joint head groups rather than infer joint faithfulness from single-head concentration | **PASS** | closure audit → `exact_joint_head_closure`; `group_head_rerank_valid_donor_v2.json` |
| Test individual and all-six MLP interventions without adding overlapping causal cuts | **PASS** | closure audit → `exact_MLP_closure`; `group_mlp_closure_v1.json` |
| Close with bounded claims; do not spend chr8/9 or export a competition module | **PASS** | closure audit → `bounded_claims_and_program_alignment`; `CLAIMS.md` HCA7/HCA10 |

Fresh regression check: `PYTHONHASHSEED=0 conda run --no-capture-output -n gfm-thrust1 python -m
pytest -q tests/test_competition.py tests/test_head_context_screen.py` collected and passed 17 tests on
2026-07-19.

## Wider Head–Context Atlas

| Requirement | Status | Authoritative evidence |
|---|---|---|
| Represent overlapping single-pair, donor-cluster, acceptor-cluster, multiple-pair, cassette/complex, and matched-control contexts | **PASS** | `atlas_context_matched_v7.summary.json`; `atlas_context_matched_v7.audit.json` (`PREFLIGHT_PASS`, zero failures) |
| Keep discovery, internal validation, and chr8/9 confirmation gene/sequence disjoint | **PASS** | matched-cohort audit and discovery-analysis audit split checks |
| Screen all 96 attention heads plus six MLP writes under the frozen A/B estimands | **PASS** | `atlas_discovery_v1.manifest.json` (`COMPLETE`); `atlas_discovery_analysis_v1.audit.json` (`PASS`, zero failures) |
| Classify general use separately from matched context association | **PASS** | `atlas_discovery_analysis_v1.*`; HCA18: four general candidates and 20 context-associated discovery signals |
| Preserve context signals that fail precision rather than relabel them as null or validated | **PASS** | HCA18; all 20 signals fail the frozen `MDE80≤0.05` progression gate |
| Seal the mechanism package before reading the gene-disjoint internal-validation outcomes | **PASS** | `atlas_mechanism_package_v1.audit.json`; package SHA-256 `48ac6b1791f2ddcd8f9ce54918afbd0f6067d87b1fa1d88ab535105fc28144cf` |
| Validate shortlisted whole-component relations with familywise inference and health controls | **PASS** | `atlas_mechanism_validation_v1/validation_audit.json`; `atlas_mechanism_validation_analysis_v1.audit.json`; HCA19 |
| Decompose only identifiable candidate mechanisms with frozen QK/V/MLP/group/path estimands | **PASS** | A3-amended `validation_audit.a3_v2.json`; `atlas_stage2_mechanism_analysis_v1_a3_v2.*`; independent analysis audit PASS with zero failures; HCA20. The original failed target manifest and first failed A3 audit remain preserved. |
| Build measured representative case/control DAGs with explicit evidence levels and remainder accounting | **PASS** | Drive `atlas_representative_dags_v1_a3_v2.json` + manifest + independent audit; 8 representatives, `export_eligible=true`, zero failures; HCA21 |
| Expose only audited, validated evidence through `gfm-visualizer` | **PASS** | `gfm-visualizer/public/data/splicebert-head-context-atlas-v1.{json,manifest.json,audit.json}`; source bundle SHA-256 pinned to `fa9d974e…d241`; evidence SHA-256 `9223cb62…9c1e`; independent transport audit PASS with zero failures; real-artifact tests preserve 2,458 screening + 992 remainder routes and reject stale/mutated evidence; HCA22 |

Fresh post-A3 regression check: `PYTHONHASHSEED=0 conda run -n gfm-thrust1 python -m pytest -q
tests/test_atlas_context.py tests/test_atlas_matching.py tests/test_atlas_analysis.py
tests/test_atlas_mechanism_package.py tests/test_atlas_mechanism_validation.py
tests/test_stage2_colab_overlay.py` passed 41 tests on 2026-07-19.

Full repository regression check: `PYTHONHASHSEED=0 conda run --no-capture-output -n gfm-thrust1
python -m pytest -q` passed **192**, skipped 34, and failed 0 on 2026-07-19. The overlay regression
test reconstructs Stage 2 overlay v6, verifies sibling-import closure, every package-declared
whole-validation runtime reference by frozen size/hash, and the post-audit DAG export tooling.

Cross-repository release check: `npm run check` in `gfm-visualizer` passed all **33** tests, the real
HCA21 import suite, and the production build on 2026-07-19. The production verifier requires all three
Atlas files, recomputes the evidence and manifest hashes, and rejects any source bundle other than the
pinned HCA21 export.

## Completion rule

The program is complete only when every row above is `PASS`, the final claim ledger cites the immutable
artifacts, the discovery and visualization repositories both pass their relevant tests/build, and a
fresh final audit confirms that no discovery-only context signal or representative screening edge was
promoted to a biological-function or feature-complete-circuit claim.
