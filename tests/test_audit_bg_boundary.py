from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert/audit_bg_boundary.py"
SPEC = importlib.util.spec_from_file_location("audit_bg_boundary", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _record(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _manifest(path: Path, result: Path, inputs: list[Path]) -> Path:
    _write(path, {"inputs": [_record(item) for item in inputs], "outputs": [_record(result)]})
    return path


def _membership(prefix: str, n: int = 8) -> dict:
    return {
        "ids": [f"{prefix}_example_{index}" for index in range(n)],
        "genes": [f"{prefix}_gene_{index}" for index in range(n)],
        "strands": ["+" if index % 2 == 0 else "-" for index in range(n)],
        "sha256": hashlib.sha256(prefix.encode()).hexdigest(),
    }


def _native_sequence() -> str:
    sequence = list("C" * 1024)
    sequence[20:22] = "GT"
    sequence[98:100] = "AG"
    sequence[100] = "G"
    sequence[101] = "A"
    sequence[102] = "A"
    return "".join(sequence)


def _construct(panel: str, example: str, gene: str, strand: str) -> tuple[dict, dict[str, str]]:
    native = _native_sequence()
    cell_sequences = {}
    for cell, alleles in audit.PANEL_ALLELES[panel].items():
        edited = list(native)
        for position, base in alleles.items():
            edited[audit._biological_index(100, position)] = base
        cell_sequences[cell] = "".join(edited)
    hashes = {cell: hashlib.sha256(sequence.encode()).hexdigest() for cell, sequence in cell_sequences.items()}
    construct_hash = audit._canonical_hash({
        "panel_id": panel,
        "cells": [
            {"cell_id": cell, "sequence_sha256": hashes[cell]}
            for cell in sorted(hashes)
        ],
    })
    return ({
        "panel_id": panel,
        "example_id": example,
        "gene_id": gene,
        "chrom": "chr11",
        "strand": strand,
        "donor_pos": 20,
        "next_acceptor_pos": 100,
        "partner_span": 80,
        "native_sequence": native,
        "native_sequence_sha256": hashlib.sha256(native.encode()).hexdigest(),
        "cell_sequence_sha256": hashes,
        "construct_sha256": construct_hash,
    }, cell_sequences)


def _bg1_row(
    panel: str,
    example: str,
    gene: str,
    strand: str,
    cell: str,
    value: float,
    construct: dict,
) -> dict:
    return {
        "panel_id": panel,
        "example_id": example,
        "gene_id": gene,
        "chrom": "chr11",
        "strand": strand,
        "donor_pos": 20,
        "next_acceptor_pos": 100,
        "partner_span": 80,
        "cell_id": cell,
        "base_logits": {"A": 0.0, "C": 0.0, "G": value, "T": 0.0},
        "linear_margin": value,
        "max_margin": value,
        "sequence_sha256": construct["cell_sequence_sha256"][cell],
        "construct_sha256": construct["construct_sha256"],
    }


def _finish_construct_hashes(rows: list[dict]) -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["panel_id"], row["example_id"]), []).append(row)
    for (panel, _example), group in groups.items():
        digest = audit._canonical_hash({
            "panel_id": panel,
            "cells": sorted(
                [
                    {"cell_id": row["cell_id"], "sequence_sha256": row["sequence_sha256"]}
                    for row in group
                ],
                key=lambda item: item["cell_id"],
            ),
        })
        for row in group:
            row["construct_sha256"] = digest


def _bg1_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(audit, "BG1_BOOTSTRAP_DRAWS", 99)
    monkeypatch.setattr(audit, "BG1_PERMUTATION_DRAWS", 99)
    monkeypatch.setattr(audit, "MIN_BG1_TARGET_GENES", 2)
    monkeypatch.setattr(audit, "MIN_BG1_TARGET_GENES_PER_STRAND", 1)
    monkeypatch.setattr(audit, "MIN_OUTER_TARGET_GENES", 2)
    monkeypatch.setattr(audit, "MIN_OUTER_TARGET_GENES_PER_STRAND", 1)
    monkeypatch.setattr(audit, "_require_manifest_inventory", lambda *args, **kwargs: None)
    memberships = {
        "target_behavior": _membership("target", n=20),
        "minus3_minus4_diagnostic": _membership("minus"),
        "plus1_plus2_diagnostic": _membership("plus"),
    }
    route_membership = _membership("route")
    config = {"sesoi": audit.SESOI, "analysis": {
        "seed": audit.ANALYSIS_SEED,
        "bg1_bootstrap_draws": 99,
        "bg1_permutation_draws": 99,
        "localization_threshold": audit.SESOI,
        "localization_multiplicity": "Holm across enabled arms",
    }, "model_file_sha256": {}}
    preflight = {
        "schema": audit.PREFLIGHT_SCHEMA,
        "PASS": True,
        "smoke": False,
        "config": config,
        "checks": {"instrument": {"PASS": True}},
        "target_cohorts": {"val": {**memberships, "target_route": route_membership}},
        "diagnostics": {"outer_power": {
            "minus3_vs_minus4": {
                "claim_bearing": True, "target_n_genes": 8,
                "target_strands": {"+": 4, "-": 4}, "projected_target_mde80": 0.01,
            },
            "plus1_vs_plus2": {
                "claim_bearing": True, "target_n_genes": 8,
                "target_strands": {"+": 4, "-": 4}, "projected_target_mde80": 0.01,
            },
        }},
    }
    preflight_path = tmp_path / "preflight.json"
    _write(preflight_path, preflight)

    rows: list[dict] = []
    constructs: list[dict] = []
    role_panels = {
        "target_behavior": (audit.TARGET_PANEL,),
        "minus3_minus4_diagnostic": (audit.MINUS3_PANEL, audit.MINUS4_PANEL),
        "plus1_plus2_diagnostic": (audit.PLUS1_PANEL, audit.PLUS2_PANEL),
    }
    target_basis = dict(audit.TARGET_CONTRASTS)
    for role, panels in role_panels.items():
        membership = memberships[role]
        for gene_index, (example, gene, strand) in enumerate(
            zip(membership["ids"], membership["genes"], membership["strands"]), start=1
        ):
            for panel in panels:
                construct, _cell_sequences = _construct(panel, example, gene, strand)
                constructs.append(construct)
                for cell_index, cell in enumerate(audit.PANEL_CELLS[panel]):
                    if panel == audit.TARGET_PANEL:
                        value = gene_index * (
                            0.13 * target_basis["m3_T_minus_A"][cell]
                            + 0.17 * target_basis["p1_G_minus_A"][cell]
                            + 0.11 * target_basis["m3_by_p1_interaction"][cell]
                        ) + 0.003 * cell_index
                    elif panel == audit.MINUS3_PANEL:
                        value = gene_index * (0.25 if cell == "m3_T" else 0.0)
                    elif panel == audit.MINUS4_PANEL:
                        value = gene_index * (0.03 if cell == "m4_T" else 0.0)
                    elif panel == audit.PLUS1_PANEL:
                        value = gene_index * (0.28 if cell == "p1_G" else 0.0)
                    else:
                        value = gene_index * (0.02 if cell == "p2_G" else 0.0)
                    rows.append(_bg1_row(
                        panel, example, gene, strand, cell, value, construct
                    ))
    construct_map = {(item["panel_id"], item["example_id"]): item for item in constructs}
    for role, panels in role_panels.items():
        memberships[role]["sha256"] = audit._cohort_hash_from_constructs(
            memberships[role], panels[0], construct_map
        )
    route_constructs = {
        (audit.TARGET_PANEL, example): _construct(
            audit.TARGET_PANEL, example, gene, strand
        )[0]
        for example, gene, strand in zip(
            route_membership["ids"], route_membership["genes"], route_membership["strands"]
        )
    }
    route_membership["sha256"] = audit._cohort_hash_from_constructs(
        route_membership, audit.TARGET_PANEL, route_constructs
    )
    _write(preflight_path, preflight)
    groups = audit._bg1_groups(rows)
    primary_specs = audit.TARGET_CONTRASTS[:2]
    _, primary_linear = audit._gene_matrix(groups[audit.TARGET_PANEL], primary_specs, "linear_margin")
    _, interaction = audit._gene_matrix(groups[audit.TARGET_PANEL], audit.TARGET_CONTRASTS[2:], "linear_margin")
    target_summary = audit._simultaneous(
        primary_linear, [item[0] for item in primary_specs], seed=audit.ANALYSIS_SEED,
        n_bootstrap=99, n_permutation=99, two_sided=False,
    )
    target = {
        "sesoi": audit.SESOI,
        "primary_ids": [item[0] for item in primary_specs],
        "interaction_id": audit.TARGET_CONTRASTS[2][0],
        "inference_linear": target_summary,
        "descriptive_max_bridge": audit._descriptive(
            primary_linear, [item[0] for item in primary_specs]
        ),
        "interaction_linear_secondary": audit._simultaneous(
            interaction, [audit.TARGET_CONTRASTS[2][0]], seed=audit.ANALYSIS_SEED + 11,
            n_bootstrap=99, n_permutation=99, two_sided=True,
        ),
        "descriptive_interaction_max": audit._descriptive(
            interaction, [audit.TARGET_CONTRASTS[2][0]]
        ),
    }
    intervals, pvalues = target_summary["simultaneous_95_ci"], target_summary["max_family_add_one_p"]
    target["verdict"] = (
        "SUPPORTED_VERIFICATION"
        if all(intervals[i][0] > audit.SESOI and pvalues[i] <= 0.05 for i in range(2))
        else "NOT_SUPPORTED" if any(intervals[i][1] <= audit.SESOI for i in range(2))
        else "INCONCLUSIVE"
    )
    localization: dict[str, dict] = {}
    for name, (inner_panel, inner_weights, outer_panel, outer_weights, contrast_name, seed) in audit.LOCALIZATION.items():
        genes_i, inner = audit._gene_matrix(groups[inner_panel], (("inner", inner_weights),), "linear_margin")
        genes_o, outer = audit._gene_matrix(groups[outer_panel], (("outer", outer_weights),), "linear_margin")
        assert genes_i == genes_o
        _, inner_max = audit._gene_matrix(groups[inner_panel], (("inner", inner_weights),), "max_margin")
        _, outer_max = audit._gene_matrix(groups[outer_panel], (("outer", outer_weights),), "max_margin")
        localization[name] = {
            "inference_linear": audit._simultaneous(
                inner - outer, [contrast_name], seed=seed,
                n_bootstrap=99, n_permutation=99, two_sided=False,
            ),
            "mean_max_margin_attenuation": float((inner_max - outer_max).mean()),
            "n_genes": len(genes_i),
            "claim_bearing_from_preflight": True,
        }
    holm = audit._holm([
        localization[name]["inference_linear"]["max_family_add_one_p"][0]
        for name in audit.LOCALIZATION
    ])
    for (name, item), adjusted in zip(localization.items(), holm):
        item["holm_p_across_enabled_arms"] = adjusted
        item["verdict"] = (
            "MATERIALLY_ATTENUATED_OUTSIDE_BOUNDARY"
            if item["inference_linear"]["simultaneous_95_ci"][0][0] > audit.SESOI
            and adjusted <= 0.05 else "INCONCLUSIVE"
        )
    localization["joint_verdict"] = (
        "BOTH_FLANKS_MATERIALLY_ATTENUATED"
        if all(localization[name]["verdict"] == "MATERIALLY_ATTENUATED_OUTSIDE_BOUNDARY" for name in audit.LOCALIZATION)
        else "JOINT_LOCALIZATION_INCONCLUSIVE"
    )
    result = {
        "schema": audit.BG1_SCHEMA,
        "status": "VERIFICATION",
        "preflight_path": str(preflight_path),
        "preflight_sha256": _record(preflight_path)["sha256"],
        "config": config,
        "cohort": {
            "split": "val", "memberships": memberships,
            "counts": {role: len(value["ids"]) for role, value in memberships.items()},
        },
        "target": target,
        "localization": localization,
        "constructs": constructs,
        "cells": rows,
    }
    result_path = tmp_path / "bg1.json"
    manifest_path = tmp_path / "bg1.manifest.json"
    _write(result_path, result)
    return result_path, _manifest(manifest_path, result_path, [preflight_path])


def _bg2_logits(value: float) -> list[float]:
    return [0.0, 0.0, value, 0.0]


def _bg2_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(audit, "BG2_BOOTSTRAP_DRAWS", 99)
    monkeypatch.setattr(audit, "MIN_BG2_TARGET_GENES", 2)
    monkeypatch.setattr(audit, "MIN_BG2_TARGET_GENES_PER_STRAND", 1)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    behavior_path, behavior_manifest_path = _bg1_fixture(upstream, monkeypatch)
    behavior = json.loads(behavior_path.read_text())
    preflight_path = Path(behavior["preflight_path"])
    preflight = json.loads(preflight_path.read_text())
    behavior_audit_path = upstream / "bg1.audit.json"
    behavior_audit = audit.build_audit(behavior_path, behavior_manifest_path)
    assert behavior_audit["PASS"] is True
    _write(behavior_audit_path, behavior_audit)

    frozen = preflight["target_cohorts"]["val"]["target_route"]
    ids, genes, strands = frozen["ids"], frozen["genes"], frozen["strands"]
    constructs = [
        _construct(audit.TARGET_PANEL, example, gene, strand)[0]
        for example, gene, strand in zip(ids, genes, strands)
    ]
    construct_by_example = {item["example_id"]: item for item in constructs}
    cells: list[dict] = []
    for example, gene, strand in zip(ids, genes, strands):
        construct = construct_by_example[example]
        for pair_id, transition, background, reference_cell, edit_cell in audit.DIRECTED_PAIRS:
            edit, reference, true, cross = 1.0, 2.0, 1.8, 1.2
            residual_hash = hashlib.sha256(f"{example}/{pair_id}".encode()).hexdigest()
            cells.append({
                "example_id": example, "gene_id": gene, "chrom": "chr11", "strand": strand,
                "donor_pos": 20, "acceptor_pos": 100, "partner_span": 80,
                "pair_id": pair_id, "transition": transition, "background": background,
                "reference_cell": reference_cell, "edit_cell": edit_cell,
                "span_bin": "060_104", "residual_delta_sha256": residual_hash,
                "reference_sequence_sha256": construct["cell_sequence_sha256"][reference_cell],
                "edit_sequence_sha256": construct["cell_sequence_sha256"][edit_cell],
                "panel_construct_sha256": construct["construct_sha256"],
                "acceptor_tokens": [98, 99, 100, 101],
                "edited_attention_ACC4": [0.1, 0.2, 0.3, 0.4],
                "reference_base_logits_ACGT": _bg2_logits(reference),
                "edit_base_logits_ACGT": _bg2_logits(edit), "base_logit_order": ["A", "C", "G", "T"],
                "reference_linear_margin": reference, "reference_max_margin": reference,
                "edit_linear_margin": edit, "edit_max_margin": edit,
                "value_linearity_max_abs": 0.0,
                "edited_head_reconstruction_max_abs": 0.0,
                "edited_attention_row_sum_abs_error": 0.0,
                "true_head_delta_norm": 1.0,
                "true": {
                    "base_logits_ACGT": _bg2_logits(true), "linear_margin": true, "max_margin": true,
                    "linear_lift": true - edit, "max_lift": true - edit,
                },
                "signed_coordinate_shams": [
                    {
                        "seed": seed, "base_logits_ACGT": _bg2_logits(1.1),
                        "linear_margin": 1.1, "max_margin": 1.1,
                        "linear_lift": 0.1, "max_lift": 0.1,
                        "norm_scale": 1.0, "matched_head_delta_norm": 1.0,
                        "norm_relative_error": 0.0,
                        "transformed_residual_delta_sha256": hashlib.sha256(
                            f"{example}/{pair_id}/{seed}".encode()
                        ).hexdigest(),
                    }
                    for seed in audit.SIGNED_SHAM_SEEDS
                ],
            })
    strata: dict[tuple[str, str, str], list[dict]] = {}
    for cell in cells:
        strata.setdefault((cell["pair_id"], cell["strand"], cell["span_bin"]), []).append(cell)
    for rows in strata.values():
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"bg2-crossgene|{row['gene_id']}|{row['example_id']}".encode()
            ).hexdigest(),
        )
        shift = max(1, len(ordered) // 2)
        for index, target in enumerate(ordered):
            source = ordered[(index + shift) % len(ordered)]
            target["cross_gene"] = {
                "source_example_id": source["example_id"],
                "source_gene_id": source["gene_id"],
                "source_strand": source["strand"],
                "source_span_bin": source["span_bin"],
                "source_residual_delta_sha256": source["residual_delta_sha256"],
                "base_logits_ACGT": _bg2_logits(1.2),
                "linear_margin": 1.2,
                "max_margin": 1.2,
                "linear_lift": 0.2,
                "max_lift": 0.2,
                "norm_scale": 1.0,
                "matched_head_delta_norm": 1.0,
                "norm_relative_error": 0.0,
            }
    summaries = audit._bg2_expected_summaries(cells)
    result = {
        "schema": audit.BG2_SCHEMA, "status": "VERIFICATION",
        "verdict": "BG2_ROUTE_SUPPORTED",
        "preflight": {"path": str(preflight_path), "sha256": _record(preflight_path)["sha256"]},
        "behavior_authorization": {
            "path": str(behavior_path), "sha256": _record(behavior_path)["sha256"],
            "manifest_path": str(behavior_manifest_path),
            "manifest_sha256": _record(behavior_manifest_path)["sha256"],
            "audit_path": str(behavior_audit_path),
            "audit_sha256": _record(behavior_audit_path)["sha256"],
            "audit_schema": audit.AUDIT_SCHEMA, "audit_PASS": True,
            "schema": behavior["schema"], "status": behavior["status"],
            "verdict": behavior["target"]["verdict"], "primary_ids": behavior["target"]["primary_ids"],
            "preflight_sha256": behavior["preflight_sha256"],
        },
        "config": {
            "design": preflight["config"], "split": "val",
            "n_boot": 99, "signed_sham_seeds": list(audit.SIGNED_SHAM_SEEDS),
            "directed_pairs": [
                {"pair_id": row[0], "transition": row[1], "background": row[2], "reference_cell": row[3], "edit_cell": row[4]}
                for row in audit.DIRECTED_PAIRS
            ],
        },
        "cohort": {
            "ids": {"route": ids}, "genes": {"route": genes}, "strands": strands,
            "preflight_membership": frozen, "data_sha256": frozen["sha256"],
            "counts": {
                "route": len(ids), "route_plus_strand": strands.count("+"),
                "route_minus_strand": strands.count("-"),
            },
        },
        "diagnostics": {
            "value_linearity_max_abs": 0.0, "edited_head_reconstruction_max_abs": 0.0,
            "edited_attention_row_sum_max_abs": 0.0,
            "n_records": len(cells), "n_genes": len(ids),
            "all_true_head_deltas_nonzero": True, "capture_PASS": True,
            "cross_gene_norm_match_max_relative": 0.0,
            "signed_coordinate_norm_match_max_relative": 0.0, "norm_match_PASS": True,
            "edited_output_replay_linear_max_abs": 0.0,
            "edited_output_replay_max_margin_max_abs": 0.0, "edited_output_replay_PASS": True,
            "checkpoint_parameter_sha256_before": "a" * 64,
            "checkpoint_parameter_sha256_after": "a" * 64,
            "checkpoint_unmutated_PASS": True,
        },
        "summaries": summaries, "constructs": constructs, "cells": cells,
    }
    result_path = tmp_path / "bg2.json"
    manifest_path = tmp_path / "bg2.manifest.json"
    _write(result_path, result)
    return result_path, _manifest(
        manifest_path,
        result_path,
        [
            preflight_path, behavior_path, behavior_manifest_path, behavior_audit_path
        ],
    )


def test_bg1_audit_recomputes_panels_contrasts_and_holm(tmp_path, monkeypatch):
    result, manifest = _bg1_fixture(tmp_path, monkeypatch)
    report = audit.build_audit(result, manifest)
    assert report["PASS"] is True
    assert report["checks"]["localization_holm_and_verdict"] is True


def test_bg2_audit_recomputes_background_nulls_and_norms(tmp_path, monkeypatch):
    result, manifest = _bg2_fixture(tmp_path, monkeypatch)
    report = audit.build_audit(result, manifest)
    assert report["PASS"] is True
    assert report["recomputed"]["verdict"] == "BG2_ROUTE_SUPPORTED"


def test_audit_fails_closed_on_row_tamper_even_with_updated_manifest(tmp_path, monkeypatch):
    result, manifest = _bg2_fixture(tmp_path, monkeypatch)
    payload = json.loads(result.read_text())
    payload["cells"][0]["cross_gene"]["matched_head_delta_norm"] = 1.01
    _write(result, payload)
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["outputs"] = [_record(result)]
    _write(manifest, manifest_payload)
    report = audit.build_audit(result, manifest)
    assert report["PASS"] is False
    assert "norm_relative_error" in report["failures"][0]


def test_bg1_audit_rejects_broken_native_core_even_with_updated_manifest(tmp_path, monkeypatch):
    result, manifest = _bg1_fixture(tmp_path, monkeypatch)
    payload = json.loads(result.read_text())
    construct = payload["constructs"][0]
    native = list(construct["native_sequence"])
    native[98:100] = "AA"
    construct["native_sequence"] = "".join(native)
    construct["native_sequence_sha256"] = hashlib.sha256(
        construct["native_sequence"].encode()
    ).hexdigest()
    _write(result, payload)
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["outputs"] = [_record(result)]
    _write(manifest, manifest_payload)
    report = audit.build_audit(result, manifest)
    assert report["PASS"] is False
    assert "core AG" in report["failures"][0]


def test_bg2_audit_rejects_nonfrozen_crossgene_assignment(tmp_path, monkeypatch):
    result, manifest = _bg2_fixture(tmp_path, monkeypatch)
    payload = json.loads(result.read_text())
    target = payload["cells"][0]
    alternatives = [
        cell for cell in payload["cells"]
        if cell["pair_id"] == target["pair_id"]
        and cell["strand"] == target["strand"]
        and cell["example_id"] not in (
            target["example_id"], target["cross_gene"]["source_example_id"]
        )
    ]
    source = alternatives[0]
    target["cross_gene"].update({
        "source_example_id": source["example_id"],
        "source_gene_id": source["gene_id"],
        "source_strand": source["strand"],
        "source_span_bin": source["span_bin"],
        "source_residual_delta_sha256": source["residual_delta_sha256"],
    })
    _write(result, payload)
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["outputs"] = [_record(result)]
    _write(manifest, manifest_payload)
    report = audit.build_audit(result, manifest)
    assert report["PASS"] is False
    assert "deterministic half-roll" in report["failures"][0]


def test_audit_rejects_stale_manifest_output_hash(tmp_path, monkeypatch):
    result, manifest = _bg1_fixture(tmp_path, monkeypatch)
    result.write_text(result.read_text() + " ")
    report = audit.build_audit(result, manifest)
    assert report["PASS"] is False
    assert "manifest output" in report["failures"][0]


def test_required_manifest_inventory_rejects_omitted_code_or_data(tmp_path):
    code = tmp_path / "runner.py"
    data = tmp_path / "data.fa"
    code.write_text("pass\n")
    data.write_text("ACGT\n")
    records = {
        "inputs": {
            code.resolve(): hashlib.sha256(code.read_bytes()).hexdigest(),
            data.resolve(): hashlib.sha256(data.read_bytes()).hexdigest(),
        }
    }
    audit._require_manifest_inventory(
        records,
        code_paths=(code,),
        expected_hashes={"data": hashlib.sha256(data.read_bytes()).hexdigest()},
    )
    with pytest.raises(audit.AuditError, match="code/protocol"):
        audit._require_manifest_inventory(
            records, code_paths=(tmp_path / "missing.py",), expected_hashes={}
        )


def test_cli_refuses_to_overwrite_audit(tmp_path, monkeypatch):
    output = tmp_path / "already.json"
    output.write_text("{}")
    monkeypatch.setattr(
        audit.sys,
        "argv",
        [str(SCRIPT), "--result", str(tmp_path / "r.json"), "--manifest", str(tmp_path / "m.json"), "--out", str(output)],
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit.main()
