#!/usr/bin/env python
"""BG1 -- AG-preserving 3'-splice-site boundary grammar at the donor readout.

The primary intervention is the frozen eight-cell ``-3 x +1`` panel declared in
``gfmcd.splicing.boundary_grammar``.  Every cell retains the central AG and has the
same positioned AG/GT/GC inventory over all touched dinucleotides.  The native
masked-donor linear margin is primary; the established max-alternative margin is
reported from the same forward pass as a bridge to C1.

This target runner is authorized only by a passed, non-smoke, hash-bound BG1
development preflight.  Validation is verification-tier; chr8/9 requires an
explicit hypothesis-specific-replication acknowledgement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.boundary_grammar import (  # noqa: E402
    ANALYSIS_SEED,
    BG1_BOOTSTRAP_DRAWS,
    BG1_PERMUTATION_DRAWS,
    CHECKPOINT_SHA256,
    FASTA_SHA256,
    GTF_SHA256,
    PLUS2_G_MINUS_A,
    PLUS2_PANEL,
    PLUS1_G_MINUS_A_SINGLE,
    PLUS1_PANEL,
    MINUS3_PANEL,
    MINUS3_T_MINUS_A_SINGLE,
    MINUS4_PANEL,
    MINUS4_T_MINUS_A,
    MIN_BG1_TARGET_GENES,
    MIN_BG1_TARGET_GENES_PER_STRAND,
    MIN_OUTER_TARGET_GENES,
    MIN_OUTER_TARGET_GENES_PER_STRAND,
    MODEL_FILE_SHA256,
    ROLE_NAMESPACE,
    SESOI,
    TARGET_CONTRASTS,
    TARGET_INTERACTION_ID,
    TARGET_MINUS3_T_MINUS_A_ID,
    TARGET_PANEL,
    TARGET_PLUS1_G_MINUS_A_ID,
    aggregate_gene_cells,
    build_panel,
    cohort_declaration,
    cohort_sha256,
    construct_sha256,
    contrast,
    holm_adjust,
    load_boundary_examples,
    margins_from_logits,
    max_family_add_one_pvalues,
    panel_construct_declaration,
    preflight_config,
    select_gene_cohort,
    validate_panel,
)
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    eligible,
    load_adapter,
    pick_device,
)
from gfmcd.splicing.p3_stats import bootstrap_studentized_max_t, mde80  # noqa: E402


PREFLIGHT_SCHEMA = "gfmcd.splicebert.boundary_grammar.preflight.v1"
RESULT_SCHEMA = "gfmcd.splicebert.boundary_grammar.behavior.v1"
PROTOCOL = (
    REPO
    / "studies/splicebert_partner_selection/preregistrations/"
    "PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md"
)
DECLARATION = REPO / "src/gfmcd/splicing/boundary_grammar.py"
PRIMARY_IDS = (TARGET_MINUS3_T_MINUS_A_ID, TARGET_PLUS1_G_MINUS_A_ID)
MODEL_FILES = (
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def resolved(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def checkpoint_weights_path(path: str | Path) -> Path:
    candidate = resolved(path)
    if candidate.is_file():
        return candidate
    weights = candidate / "pytorch_model.bin"
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint weights not found: {weights}")
    return weights


def model_file_hashes(path: str | Path) -> dict[str, str]:
    """Bind every local file that controls model/tokenizer execution."""

    model_path = resolved(path)
    if model_path.is_file():
        raise RuntimeError(
            "canonical BG runs require the complete model/tokenizer directory, not weights alone"
        )
    hashes = {}
    for name in MODEL_FILES:
        source = model_path / name
        if not source.is_file():
            raise RuntimeError(f"frozen model artifact is missing: {source}")
        hashes[name] = sha256_file(source)
    if hashes["pytorch_model.bin"] != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint does not match the frozen SpliceBERT weights")
    if hashes != MODEL_FILE_SHA256:
        raise RuntimeError("model/tokenizer artifacts do not match the frozen declaration")
    return hashes


def validate_preflight(
    path: str | Path,
    *,
    checkpoint: str | Path,
    gtf: str | Path,
    fasta: str | Path,
    protocol: str | Path = PROTOCOL,
    declaration: str | Path = DECLARATION,
) -> dict:
    """Fail closed if the development authorization or any frozen input drifted."""

    preflight_path = resolved(path)
    try:
        result = json.loads(preflight_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read BG1 preflight: {preflight_path}") from exc
    if result.get("schema") != PREFLIGHT_SCHEMA:
        raise RuntimeError(f"wrong BG1 preflight schema: {result.get('schema')!r}")
    if result.get("PASS") is not True:
        raise RuntimeError("BG1 preflight PASS is not true; target execution is forbidden")
    if result.get("smoke") is not False:
        raise RuntimeError("a smoke preflight cannot authorize target execution")
    sources = {
        "checkpoint_sha256": checkpoint_weights_path(checkpoint),
        "protocol_sha256": resolved(protocol),
        "declaration_sha256": resolved(declaration),
        "gtf_sha256": resolved(gtf),
        "fasta_sha256": resolved(fasta),
    }
    expected_hashes = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "gtf_sha256": GTF_SHA256,
        "fasta_sha256": FASTA_SHA256,
    }
    for field, source in sources.items():
        observed = sha256_file(source) if source.is_file() else None
        if (
            observed is None
            or result.get(field) != observed
            or field in expected_hashes and observed != expected_hashes[field]
        ):
            raise RuntimeError(f"BG1 preflight {field} mismatch for {source}")
    if result.get("config") != preflight_config():
        raise RuntimeError("BG1 preflight config differs from the frozen declaration")
    if result.get("model_file_sha256") != model_file_hashes(checkpoint):
        raise RuntimeError("BG1 preflight model/tokenizer file hashes drifted")
    cohort = result.get("cohort", {})
    if (
        cohort.get("split") != "train"
        or not isinstance(cohort.get("ids"), dict)
        or not isinstance(cohort.get("genes"), dict)
        or not isinstance(cohort.get("counts"), dict)
    ):
        raise RuntimeError("BG1 preflight must bind train IDs by stable role")
    for role in ("behavior", "route"):
        ids = cohort["ids"].get(role)
        genes = cohort["genes"].get(role)
        if (
            not isinstance(ids, list)
            or not ids
            or not isinstance(genes, list)
            or len(ids) != len(genes)
            or len(ids) != int(cohort["counts"].get(role, -1))
            or len(set(ids)) != len(ids)
            or len(set(genes)) != len(genes)
        ):
            raise RuntimeError(f"BG1 preflight {role} cohort is malformed")
    if not isinstance(cohort.get("sha256"), str) or len(cohort["sha256"]) != 64:
        raise RuntimeError("BG1 preflight cohort hash is missing")
    failed = [
        name for name, check in result.get("checks", {}).items()
        if not isinstance(check, dict) or check.get("PASS") is not True
    ]
    if failed or not result.get("checks"):
        raise RuntimeError(f"BG1 preflight contains failed/missing checks: {failed}")
    return result


def membership_declaration(examples: Sequence) -> dict:
    return {
        "ids": [example.example_id for example in examples],
        "genes": [example.gene_id for example in examples],
        "strands": [example.strand for example in examples],
        "sha256": cohort_sha256(examples),
    }


def require_frozen_membership(
    preflight: Mapping[str, object], split: str, name: str, examples: Sequence
) -> None:
    frozen = preflight.get("target_cohorts", {}).get(split, {}).get(name)
    current = membership_declaration(examples)
    if frozen != current:
        raise RuntimeError(f"{split} {name} cohort differs from the outcome-blind preflight census")


def require_rebuilt_train_membership(preflight: Mapping[str, object], all_examples: Sequence) -> None:
    """Reconstruct the exact outcome-blind train roles bound by the preflight."""

    cohort = preflight["cohort"]
    train = eligible(all_examples, "train")
    valid = [example for example in train if panel_eligible(example, TARGET_PANEL)]
    rebuilt = {
        role: select_gene_cohort(valid, role=role, n=int(cohort["counts"][role]))
        for role in ("behavior", "route")
    }
    declaration = cohort_declaration(rebuilt)
    for field in ("ids", "genes", "sha256"):
        if declaration[field] != cohort.get(field):
            raise RuntimeError(f"train preflight cohort {field} does not reconstruct")


def panel_eligible(example, panel) -> bool:
    try:
        cells = build_panel(
            example.sequence,
            int(example.next_acceptor_pos),
            panel,
            strand=example.strand,
        )
        return validate_panel(example, cells, panel).passed
    except (TypeError, ValueError):
        return False


def select_panel_cohort(examples, panel, *, role: str, n: int | None):
    candidates = [example for example in examples if panel_eligible(example, panel)]
    return select_gene_cohort(candidates, role=role, n=n)


def select_paired_panel_cohort(examples, left_panel, right_panel, *, role: str, n: int | None):
    candidates = [
        example for example in examples
        if panel_eligible(example, left_panel) and panel_eligible(example, right_panel)
    ]
    return select_gene_cohort(candidates, role=role, n=n)


def score_panel(task: CouplingTask, examples: Sequence, panel) -> list[dict]:
    """Score a complete panel, recording enough cells for independent reanalysis."""

    jobs = []
    construct_hashes = {}
    for example in examples:
        cells = build_panel(
            example.sequence,
            int(example.next_acceptor_pos),
            panel,
            strand=example.strand,
        )
        validate_panel(example, cells, panel).require_passed()
        construct_hashes[example.example_id] = construct_sha256(
            cells, panel_id=panel.panel_id
        )
        for cell_id, sequence in cells.items():
            jobs.append((example, cell_id, sequence))

    rows = []
    with torch.no_grad():
        for start in range(0, len(jobs), task.batch):
            chunk = jobs[start:start + task.batch]
            donors = [int(example.donor_pos) for example, _cell, _sequence in chunk]
            encoded = task.encode([sequence for _example, _cell, sequence in chunk], donors)
            logits = task.ad.model(**encoded).logits.float()
            linear, maximum = margins_from_logits(
                logits,
                donors,
                base_token_ids=task.bid,
                token_offset=task.off,
            )
            for local_index, ((example, cell_id, sequence), linear_value, max_value) in enumerate(
                zip(chunk, linear, maximum)
            ):
                donor_token = int(example.donor_pos) + task.off
                base_logits = {
                    base: float(logits[local_index, donor_token, token_id])
                    for base, token_id in sorted(task.bid.items())
                }
                rows.append({
                    "example_id": example.example_id,
                    "gene_id": example.gene_id,
                    "chrom": example.chrom,
                    "strand": example.strand,
                    "donor_pos": int(example.donor_pos),
                    "next_acceptor_pos": int(example.next_acceptor_pos),
                    "partner_span": int(example.next_acceptor_pos - example.donor_pos),
                    "panel_id": panel.panel_id,
                    "cell_id": cell_id,
                    "linear_margin": float(linear_value),
                    "max_margin": float(max_value),
                    "base_logits": base_logits,
                    "sequence_sha256": hashlib.sha256(
                        sequence.encode("ascii")
                    ).hexdigest(),
                    "construct_sha256": construct_hashes[example.example_id],
                })
    return rows


def gene_contrast_matrix(rows: Sequence[Mapping[str, object]], panel, specs):
    aggregated = aggregate_gene_cells(rows, panel=panel)
    by_gene: dict[str, dict[str, dict[str, float]]] = {}
    for row in aggregated:
        by_gene.setdefault(str(row["gene_id"]), {})[str(row["cell_id"])] = {
            "linear": float(row["linear_margin"]),
            "max": float(row["max_margin"]),
        }
    genes = sorted(by_gene)
    linear = np.asarray([
        [contrast({cell: values["linear"] for cell, values in by_gene[gene].items()}, spec)
         for spec in specs]
        for gene in genes
    ], dtype=float)
    maximum = np.asarray([
        [contrast({cell: values["max"] for cell, values in by_gene[gene].items()}, spec)
         for spec in specs]
        for gene in genes
    ], dtype=float)
    return genes, linear, maximum


def signflip_null(matrix: np.ndarray, *, draws: int, seed: int) -> np.ndarray:
    if matrix.ndim != 2 or len(matrix) < 3 or draws < 99:
        raise ValueError("sign-flip null needs [genes,endpoints], >=3 genes, and >=99 draws")
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(draws, len(matrix), 1))
    return (signs * matrix[None, :, :]).mean(axis=1)


def simultaneous_summary(
    matrix: np.ndarray,
    *,
    names: Sequence[str],
    seed: int,
    n_bootstrap: int,
    n_permutation: int,
    p_two_sided: bool,
) -> dict:
    simultaneous = bootstrap_studentized_max_t(
        matrix,
        names=names,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    null = signflip_null(matrix, draws=n_permutation, seed=seed + 1)
    pvalues = max_family_add_one_pvalues(
        matrix.mean(axis=0), null, two_sided=p_two_sided
    )
    return {
        "names": list(simultaneous.names),
        "means": list(simultaneous.means),
        "standard_errors": list(simultaneous.standard_errors),
        "simultaneous_95_ci": [
            [lower, upper]
            for lower, upper in zip(simultaneous.lower, simultaneous.upper)
        ],
        "max_family_add_one_p": pvalues.tolist(),
        "critical_value": simultaneous.critical_value,
        "n_genes": simultaneous.n_genes,
        "n_bootstrap": simultaneous.n_bootstrap,
        "n_permutation": n_permutation,
        "p_alternative": "two-sided" if p_two_sided else "greater",
        "mde80": [
            mde80(matrix[:, index], critical_value=simultaneous.critical_value)
            for index in range(matrix.shape[1])
        ],
    }


def descriptive_summary(matrix: np.ndarray, *, names: Sequence[str]) -> dict:
    """Non-inferential bridge summary for endpoints outside the frozen claim family."""

    values = np.asarray(matrix, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != len(names)
        or values.shape[0] < 1
        or not np.isfinite(values).all()
    ):
        raise ValueError("descriptive matrix must be finite [genes,endpoints]")
    return {
        "names": list(names),
        "means": values.mean(axis=0).tolist(),
        "standard_deviations": values.std(axis=0, ddof=1).tolist(),
        "standard_errors": (
            values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        ).tolist(),
        "n_genes": int(values.shape[0]),
        "claim_bearing": False,
    }


def target_verdict(summary: Mapping[str, object]) -> str:
    intervals = summary["simultaneous_95_ci"]
    pvalues = summary["max_family_add_one_p"]
    if all(float(intervals[index][0]) > SESOI and float(pvalues[index]) <= 0.05
           for index in range(2)):
        return "SUPPORTED_VERIFICATION"
    if any(float(intervals[index][1]) <= SESOI for index in range(2)):
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def attenuation_summary(
    task: CouplingTask,
    examples,
    *,
    inner_panel,
    inner_spec,
    outer_panel,
    outer_spec,
    seed: int,
    n_bootstrap: int,
    n_permutation: int,
) -> tuple[dict, list[dict]]:
    inner_rows = score_panel(task, examples, inner_panel)
    outer_rows = score_panel(task, examples, outer_panel)
    genes_i, inner_linear, inner_max = gene_contrast_matrix(
        inner_rows, inner_panel, (inner_spec,)
    )
    genes_o, outer_linear, outer_max = gene_contrast_matrix(
        outer_rows, outer_panel, (outer_spec,)
    )
    if genes_i != genes_o:
        raise RuntimeError("paired inner/outer diagnostic genes drifted")
    linear = inner_linear[:, 0] - outer_linear[:, 0]
    maximum = inner_max[:, 0] - outer_max[:, 0]
    inference = simultaneous_summary(
        linear[:, None],
        names=(f"{inner_spec.contrast_id}_minus_{outer_spec.contrast_id}",),
        seed=seed,
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation,
        p_two_sided=False,
    )
    for row in inner_rows:
        row["diagnostic_role"] = "inner"
    for row in outer_rows:
        row["diagnostic_role"] = "outer"
    return {
        "inference_linear": inference,
        "mean_max_margin_attenuation": float(maximum.mean()),
        "n_genes": len(genes_i),
    }, inner_rows + outer_rows


def apply_attenuation_decisions(summaries: Mapping[str, dict], enabled: Mapping[str, bool]) -> str:
    """Apply Holm once across every enabled localization arm, then freeze verdicts."""

    active = [name for name in summaries if bool(enabled.get(name))]
    adjusted = holm_adjust([
        float(summaries[name]["inference_linear"]["max_family_add_one_p"][0])
        for name in active
    ]) if active else np.asarray([], dtype=float)
    adjusted_by_name = dict(zip(active, adjusted.tolist()))
    for name, summary in summaries.items():
        claim_bearing = name in adjusted_by_name
        summary["claim_bearing_from_preflight"] = claim_bearing
        summary["holm_p_across_enabled_arms"] = adjusted_by_name.get(name)
        lower = float(summary["inference_linear"]["simultaneous_95_ci"][0][0])
        if not claim_bearing:
            summary["verdict"] = "PRESPECIFIED_DIAGNOSTIC_ONLY"
        elif lower > SESOI and adjusted_by_name[name] <= 0.05:
            summary["verdict"] = "MATERIALLY_ATTENUATED_OUTSIDE_BOUNDARY"
        else:
            summary["verdict"] = "INCONCLUSIVE"
    if len(active) == 2 and all(
        summaries[name]["verdict"] == "MATERIALLY_ATTENUATED_OUTSIDE_BOUNDARY"
        for name in active
    ):
        return "BOTH_FLANKS_MATERIALLY_ATTENUATED"
    if not active:
        return "PRESPECIFIED_DIAGNOSTIC_ONLY"
    return "JOINT_LOCALIZATION_INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", default="results/splicebert_coupling/bg1_boundary_preflight_v1.json")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--stage",
        choices=("verification", "hypothesis-specific-replication"),
        default="verification",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=BG1_BOOTSTRAP_DRAWS)
    parser.add_argument("--n-permutation", type=int, default=BG1_PERMUTATION_DRAWS)
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--out", default=None
    )
    parser.add_argument(
        "--i-understand-chr8-chr9-is-a-hypothesis-specific-replication",
        action="store_true",
    )
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("batch must be positive")
    if (args.seed, args.n_bootstrap, args.n_permutation) != (
        ANALYSIS_SEED, BG1_BOOTSTRAP_DRAWS, BG1_PERMUTATION_DRAWS
    ):
        parser.error("canonical BG1 seed and resampling counts are frozen")
    if args.split == "val" and args.stage != "verification":
        parser.error("val must be labelled verification")
    if args.split == "test" and (
        args.stage != "hypothesis-specific-replication"
        or not args.i_understand_chr8_chr9_is_a_hypothesis_specific_replication
    ):
        parser.error(
            "test requires the hypothesis-specific-replication stage and explicit acknowledgement"
        )

    if args.out is None:
        args.out = (
            "results/splicebert_coupling/bg1_boundary_grammar_val_v1.json"
            if args.split == "val" else
            "results/splicebert_coupling/bg1_boundary_grammar_chr89_v1.json"
        )

    out = guard_output_path(resolved(args.out))
    manifest_path = guard_output_path(out.with_suffix(".manifest.json"))
    preflight = validate_preflight(
        args.preflight,
        checkpoint=args.model,
        gtf=args.gtf,
        fasta=args.fasta,
    )
    if args.split == "test" and preflight.get("test_replication_enabled") is not True:
        raise RuntimeError("chr8/9 replication was not prospectively enabled by the preflight")
    outer_power = preflight.get("diagnostics", {}).get("outer_power", {})

    torch.set_grad_enabled(False)
    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=True)
    task = CouplingTask(adapter, batch=args.batch)
    all_examples = load_boundary_examples(args.gtf, args.fasta)
    require_rebuilt_train_membership(preflight, all_examples)
    source = eligible(all_examples, args.split)
    target_examples = select_panel_cohort(
        source, TARGET_PANEL, role="behavior", n=None
    )
    target_strands = {strand: sum(x.strand == strand for x in target_examples) for strand in ("+", "-")}
    if (
        len(target_examples) < MIN_BG1_TARGET_GENES
        or min(target_strands.values()) < MIN_BG1_TARGET_GENES_PER_STRAND
    ):
        raise RuntimeError("target panel fails the frozen gene/strand supply gate")
    require_frozen_membership(preflight, args.split, "target_behavior", target_examples)
    target_rows = score_panel(task, target_examples, TARGET_PANEL)
    genes, target_linear_all, target_max_all = gene_contrast_matrix(
        target_rows, TARGET_PANEL, TARGET_CONTRASTS
    )
    primary_names = tuple(spec.contrast_id for spec in TARGET_CONTRASTS[:2])
    inference_linear = simultaneous_summary(
        target_linear_all[:, :2],
        names=primary_names,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
        p_two_sided=False,
    )
    descriptive_max = descriptive_summary(
        target_max_all[:, :2],
        names=primary_names,
    )
    interaction_linear = simultaneous_summary(
        target_linear_all[:, 2:],
        names=(TARGET_INTERACTION_ID,),
        seed=args.seed + 11,
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
        p_two_sided=True,
    )
    interaction_max = descriptive_summary(
        target_max_all[:, 2:],
        names=(TARGET_INTERACTION_ID,),
    )

    minus_examples = select_paired_panel_cohort(
        source,
        MINUS3_PANEL,
        MINUS4_PANEL,
        role="behavior",
        n=None,
    )
    plus_examples = select_paired_panel_cohort(
        source,
        PLUS1_PANEL,
        PLUS2_PANEL,
        role="behavior",
        n=None,
    )
    require_frozen_membership(
        preflight, args.split, "minus3_minus4_diagnostic", minus_examples
    )
    require_frozen_membership(
        preflight, args.split, "plus1_plus2_diagnostic", plus_examples
    )
    for label, examples in (
        ("minus3_minus4", minus_examples),
        ("plus1_plus2", plus_examples),
    ):
        strand_counts = {strand: sum(x.strand == strand for x in examples) for strand in ("+", "-")}
        if bool(outer_power.get(label, {}).get("claim_bearing")) and (
            len(examples) < MIN_OUTER_TARGET_GENES
            or min(strand_counts.values()) < MIN_OUTER_TARGET_GENES_PER_STRAND
        ):
            raise RuntimeError(f"claim-bearing {label} cohort fails its frozen supply gate")
    minus_summary, minus_rows = attenuation_summary(
        task,
        minus_examples,
        inner_panel=MINUS3_PANEL,
        inner_spec=MINUS3_T_MINUS_A_SINGLE,
        outer_panel=MINUS4_PANEL,
        outer_spec=MINUS4_T_MINUS_A,
        seed=args.seed + 20,
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
    )
    plus_summary, plus_rows = attenuation_summary(
        task,
        plus_examples,
        inner_panel=PLUS1_PANEL,
        inner_spec=PLUS1_G_MINUS_A_SINGLE,
        outer_panel=PLUS2_PANEL,
        outer_spec=PLUS2_G_MINUS_A,
        seed=args.seed + 30,
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
    )
    localization = {
        "minus3_vs_minus4": minus_summary,
        "plus1_vs_plus2": plus_summary,
    }
    localization_verdict = apply_attenuation_decisions(
        localization,
        {
            name: bool(outer_power.get(name, {}).get("claim_bearing"))
            for name in localization
        },
    )

    memberships = {
        "target_behavior": membership_declaration(target_examples),
        "minus3_minus4_diagnostic": membership_declaration(minus_examples),
        "plus1_plus2_diagnostic": membership_declaration(plus_examples),
    }
    constructs = [
        *(
            panel_construct_declaration(example, TARGET_PANEL)
            for example in target_examples
        ),
        *(
            panel_construct_declaration(example, MINUS3_PANEL)
            for example in minus_examples
        ),
        *(
            panel_construct_declaration(example, MINUS4_PANEL)
            for example in minus_examples
        ),
        *(
            panel_construct_declaration(example, PLUS1_PANEL)
            for example in plus_examples
        ),
        *(
            panel_construct_declaration(example, PLUS2_PANEL)
            for example in plus_examples
        ),
    ]
    verdict = target_verdict(inference_linear)
    result = {
        "schema": RESULT_SCHEMA,
        "status": args.stage.upper().replace("-", "_"),
        "preflight_path": str(resolved(args.preflight)),
        "preflight_sha256": sha256_file(resolved(args.preflight)),
        "config": preflight_config(),
        "cohort": {"memberships": memberships, "split": args.split, "counts": {
            "target_behavior": len(target_examples),
            "minus3_minus4_diagnostic": len(minus_examples),
            "plus1_plus2_diagnostic": len(plus_examples),
        }},
        "target": {
            "verdict": verdict,
            "sesoi": SESOI,
            "inference_linear": inference_linear,
            "descriptive_max_bridge": descriptive_max,
            "interaction_linear_secondary": interaction_linear,
            "descriptive_interaction_max": interaction_max,
            "primary_ids": list(PRIMARY_IDS),
            "interaction_id": TARGET_INTERACTION_ID,
        },
        "localization": {
            "joint_verdict": localization_verdict,
            **localization,
        },
        "constructs": constructs,
        "cells": target_rows + minus_rows + plus_rows,
        "wording_if_independent_audit_passes": (
            "On motif-inventory-matched 3'-boundary counterfactuals, the linear donor "
            "margin has material -3 T-over-A and +1 G-over-A causal effects while AG is fixed."
            if verdict == "SUPPORTED_VERIFICATION" else
            "No positive boundary-grammar wording is licensed by this result."
        ),
        "scope_boundary": (
            "Model-native masked-donor behavior at one checkpoint; no U2AF1, spliceosome, "
            "RNA-binding, transcript-usage, or biochemical identity claim."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")

    manifest = ExperimentManifest.create(
        "splicebert-partner-selection:bg1-boundary-grammar-v1",
        sys.argv,
        repo=REPO,
        status=result["status"],
        config={
            "split": args.split,
            "stage": args.stage,
            "n_target_genes": len(target_examples),
            "n_minus_diagnostic_genes": len(minus_examples),
            "n_plus_diagnostic_genes": len(plus_examples),
            "device": device,
            "verdict": verdict,
        },
        seeds={"cohort_roles": 0, "bootstrap_and_permutation": args.seed},
        notes=[
            "Validation is verification tier; chr8/9 is only a hypothesis-specific replication.",
            "Interaction is corrected secondary; flanking attenuation is claim-bearing only when preflight powered.",
        ],
    )
    model_path = resolved(args.model)
    model_inputs = (
        [model_path]
        if model_path.is_file()
        else [model_path / name for name in MODEL_FILES]
    )
    for path in (
        args.preflight,
        args.gtf,
        args.fasta,
        *model_inputs,
        PROTOCOL,
        DECLARATION,
        Path(__file__),
        REPO / "src/gfmcd/splicing/coupling.py",
        REPO / "src/gfmcd/splicing/circuit.py",
        REPO / "src/gfmcd/splicing/p3_stats.py",
        REPO / "src/gfmcd/circuits/hf_bert.py",
    ):
        manifest.add_input(resolved(path))
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({
        "target_verdict": verdict,
        "n_target_genes": len(target_examples),
        "minus3_vs_minus4": minus_summary["verdict"],
        "plus1_vs_plus2": plus_summary["verdict"],
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
