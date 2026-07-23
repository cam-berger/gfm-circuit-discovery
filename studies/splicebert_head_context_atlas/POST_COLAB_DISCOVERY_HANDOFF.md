# Post-Colab Head–Context Atlas handoff

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Status: operational runbook. The scientific rules are frozen in `ANALYSIS_PLAN_V1.md`,
`ANALYSIS_AMENDMENT_A1.md`, `MECHANISM_VALIDATION_PLAN_V1.md`, and
`MECHANISM_VALIDATION_AMENDMENT_A1.md` + `MECHANISM_VALIDATION_AMENDMENT_A2.md`.

## Required Colab outputs

Preserve the Drive folder `gfm_atlas_screen_v1_results/` unchanged. Before local analysis, retrieve at
least:

- `validation_audit.json`;
- `run_manifest.health.json` and `run_manifest.target.json`;
- `atlas_discovery_v1.target_matrix.jsonl.gz`;
- `atlas_discovery_v1.health_matrix.jsonl.gz`;
- `atlas_discovery_v1.pairs.jsonl.gz`;
- `atlas_discovery_v1.summary.json` and `.manifest.json`.

Do not retrieve or substitute `.validation.*` files at this stage. Discovery analysis and candidate
packaging must finish and independently audit first.

## Gate 1 — Colab completion

Require, literally:

- screen audit `status=PASS`, `n_failures=0`;
- health and target phase manifests complete with `validation_loaded=false`;
- 512 health shards and 11,807 target shards;
- merged counts: 52,224 health rows, 1,204,314 target rows, 6,487 pairs, 102 components;
- all input/output hashes in the copied manifests match the copied files.

Any mismatch stops the sequence. Do not repair an output matrix by dropping rows.

## Gate 2 — frozen discovery analysis

From the repository root, with `<MERGED>` pointing to the copied `atlas_discovery_v1` prefix:

```bash
PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/analyze_atlas_discovery.py \
  --merged-prefix <MERGED> \
  --out-prefix results/splicebert_head_context_atlas/atlas_discovery_analysis_v1

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/validate_atlas_discovery_analysis.py \
  --merged-prefix <MERGED> \
  --analysis-prefix results/splicebert_head_context_atlas/atlas_discovery_analysis_v1
```

Require `DISCOVERY_COMPLETE` and an independent audit `PASS` with zero failures. Read the shortlist only
after this audit. Preserve a zero-candidate result; do not force a rank-based candidate.

## Gate 3 — candidate package while validation remains sealed

```bash
PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/build_atlas_mechanism_package.py \
  --analysis-prefix results/splicebert_head_context_atlas/atlas_discovery_analysis_v1 \
  --merged-prefix <MERGED> \
  --cohort-prefix results/splicebert_head_context_atlas/atlas_context_matched_v7 \
  --out results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/validate_atlas_mechanism_package.py \
  --package results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json \
  --analysis-prefix results/splicebert_head_context_atlas/atlas_discovery_analysis_v1 \
  --merged-prefix <MERGED>
```

Require `FROZEN_BEFORE_VALIDATION_ACCESS`, `validation_loaded=false`, and package audit `PASS`. The
package's portable `.validation.*` references—not an operator-provided alternative—define the later
internal-validation inputs.

## Gate 4 — untouched whole/group validation

Only after Gate 3:

```bash
PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/run_atlas_mechanism_validation.py \
  --package results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json \
  --package-audit results/splicebert_head_context_atlas/atlas_mechanism_package_v1.audit.json \
  --output-dir results/splicebert_head_context_atlas/atlas_mechanism_validation_v1 \
  --phase health --health-n 256 --batch 8
```

Require `HEALTH_PASS` before running the same command with `--phase target`. Then independently audit,
merge, analyze, and independently reconstruct:

```bash
PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/validate_atlas_mechanism_validation.py \
  --run-dir results/splicebert_head_context_atlas/atlas_mechanism_validation_v1 \
  --package results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json \
  --package-audit results/splicebert_head_context_atlas/atlas_mechanism_package_v1.audit.json

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/merge_atlas_mechanism_validation.py \
  --run-dir results/splicebert_head_context_atlas/atlas_mechanism_validation_v1 \
  --prefix results/splicebert_head_context_atlas/atlas_mechanism_validation_v1_merged

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/analyze_atlas_mechanism_validation.py \
  --merged-prefix results/splicebert_head_context_atlas/atlas_mechanism_validation_v1_merged \
  --package results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json \
  --package-audit results/splicebert_head_context_atlas/atlas_mechanism_package_v1.audit.json \
  --validation-audit results/splicebert_head_context_atlas/atlas_mechanism_validation_v1/validation_audit.json \
  --prefix results/splicebert_head_context_atlas/atlas_mechanism_validation_analysis_v1

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/validate_atlas_mechanism_analysis.py \
  --merged-prefix results/splicebert_head_context_atlas/atlas_mechanism_validation_v1_merged \
  --analysis-prefix results/splicebert_head_context_atlas/atlas_mechanism_validation_analysis_v1 \
  --package results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json
```

Only `validated_candidate_ids` may advance to QK/OV/MLP/path assays and representative DAGs. Discovery
ranks, inconclusive candidates, controls, and smoke artifacts never enter `gfm-visualizer`.

## Gate 5 — freeze the conditional-mechanism package

If `validated_candidate_ids` is empty, stop successfully: there is no stage-two mechanism run. Otherwise:

```bash
PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/build_atlas_stage2_package.py \
  --whole-analysis-prefix results/splicebert_head_context_atlas/atlas_mechanism_validation_analysis_v1 \
  --stage1-package results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json \
  --stage1-package-audit results/splicebert_head_context_atlas/atlas_mechanism_package_v1.audit.json \
  --out results/splicebert_head_context_atlas/atlas_stage2_package_v1.json

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/validate_atlas_stage2_package.py \
  --package results/splicebert_head_context_atlas/atlas_stage2_package_v1.json \
  --whole-analysis-prefix results/splicebert_head_context_atlas/atlas_mechanism_validation_analysis_v1 \
  --stage1-package results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json
```

Require `FROZEN_AFTER_WHOLE_VALIDATION_BEFORE_STAGE2_FORWARD` and a current zero-failure package audit.
The package fixes every eligible endpoint, exact group, importance control, QK/V source, downstream MLP
layer, upstream attention group, representative, and joint max-|t| family before a conditional forward.

## Gate 6 — conditional QK/OV/MLP/path inference

Run the health phase first. Only after it passes, run target and the two independent audits:

```bash
PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/run_atlas_stage2_mechanisms.py \
  --package results/splicebert_head_context_atlas/atlas_stage2_package_v1.json \
  --package-audit results/splicebert_head_context_atlas/atlas_stage2_package_v1.audit.json \
  --stage1-package results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json \
  --output-dir results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1 \
  --phase health --health-n 256 --batch 8

# Repeat the command above with --phase target only after HEALTH PASS.

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/validate_atlas_stage2_mechanisms.py \
  --run-dir results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1 \
  --package results/splicebert_head_context_atlas/atlas_stage2_package_v1.json \
  --package-audit results/splicebert_head_context_atlas/atlas_stage2_package_v1.audit.json

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/merge_atlas_stage2_mechanisms.py \
  --run-dir results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1 \
  --package results/splicebert_head_context_atlas/atlas_stage2_package_v1.json \
  --prefix results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1_merged

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/analyze_atlas_stage2_mechanisms.py \
  --merged-prefix results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1_merged \
  --package results/splicebert_head_context_atlas/atlas_stage2_package_v1.json \
  --package-audit results/splicebert_head_context_atlas/atlas_stage2_package_v1.audit.json \
  --execution-audit results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1/validation_audit.json \
  --run-dir results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1 \
  --prefix results/splicebert_head_context_atlas/atlas_stage2_mechanism_analysis_v1

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/validate_atlas_stage2_analysis.py \
  --merged-prefix results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1_merged \
  --analysis-prefix results/splicebert_head_context_atlas/atlas_stage2_mechanism_analysis_v1 \
  --package results/splicebert_head_context_atlas/atlas_stage2_package_v1.json \
  --run-dir results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1
```

QKV bridges are identities, not family evidence. A-estimand QK/V is not a population decomposition.
Only simultaneously supported, independently audited conditional relations may enter representative
DAG construction. Conditional routing/value/path labels are issued only by the A2 simultaneous-interval
rules; no output is called binding or assigned a biological head function.

## Gate 7 — representative DAGs and export boundary

Only after the Gate 6 analysis audit passes:

```bash
PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/build_atlas_representative_dags.py \
  --package results/splicebert_head_context_atlas/atlas_stage2_package_v1.json \
  --package-audit results/splicebert_head_context_atlas/atlas_stage2_package_v1.audit.json \
  --analysis-prefix results/splicebert_head_context_atlas/atlas_stage2_mechanism_analysis_v1 \
  --stage2-run-dir results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1 \
  --cohort-prefix results/splicebert_head_context_atlas/atlas_context_matched_v7 \
  --output results/splicebert_head_context_atlas/atlas_representative_dags_v1.json

PYTHONHASHSEED=0 python scripts/splicebert_head_context_atlas/validate_atlas_representative_dags.py \
  --bundle results/splicebert_head_context_atlas/atlas_representative_dags_v1.json \
  --package results/splicebert_head_context_atlas/atlas_stage2_package_v1.json \
  --package-audit results/splicebert_head_context_atlas/atlas_stage2_package_v1.audit.json \
  --analysis-prefix results/splicebert_head_context_atlas/atlas_stage2_mechanism_analysis_v1
```

The case and matched negative-control graphs keep activation×Jacobian edges labeled `screening` or
`negative_control`; audited population effects live in a separate exact/QK/V_OV/MLP overlay. Hidden
remainder mass is reported. A-estimand QK/V state interchange is representative-only and descriptive.
Only a full, zero-failure audit with `export_eligible=true` may be adapted into `gfm-visualizer`; smoke
bundles are structurally `SMOKE_NOT_EXPORTABLE`.

## Split language

The v1 internal validation is an outcome-independent, gene- and sequence-disjoint hash split with
chr8/9/11/16/19 absent. It is not chr8/9 confirmation. chr8/9 remains sealed throughout this runbook.
