from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from gfmcd.splicing.r5_target_ledger import (
    R1CTargetSpentError,
    R5TargetSpentError,
    build_r1c_biological_target_identity,
    build_r5_biological_target_identity,
    complete_r1c_target_attempt,
    complete_r5_target_attempt,
    reserve_r1c_target_attempt,
    reserve_r5_target_attempt,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _logical(label: str) -> dict:
    return {
        "logical_name": f"{label}.json",
        "size_bytes": len(label) + 1,
        "sha256": _sha(label),
    }


def _r1c_method_context(
    label: str = "v1", *, biological_target_key: str | None = None
) -> dict:
    return {
        "protocol_sha256": _sha(f"protocol-{label}"),
        "instrument_fingerprint_sha256": _sha(f"instrument-{label}"),
        "r5_stack_fingerprint_sha256": _sha(f"stack-{label}"),
        "checkpoint_bundle_sha256": _sha("checkpoint"),
        "construction_contract_sha256": _sha(f"construction-contract-{label}"),
        "endpoint_policy": f"endpoint-{label}",
        "preflight_artifact": _logical(f"preflight-{label}"),
        "preflight_manifest": _logical(f"preflight-{label}.manifest"),
        "input_snapshot_sha256": _sha(f"snapshot-{label}"),
        "requested_artifact_logical_name": "r1c-target.json",
        "requested_manifest_logical_name": "r1c-target.manifest.json",
        "declared_completion_logical_name": (
            f"{biological_target_key}.completion.json"
            if biological_target_key is not None
            else "r1c-target.completion.json"
        ),
    }


def test_r1c_receipt_uses_git_common_dir_and_method_changes_cannot_reset_spend(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    identity = build_r1c_biological_target_identity(
        checkpoint_sha256=_sha("checkpoint"),
        gtf_sha256=_sha("gtf"),
        fasta_sha256=_sha("fasta"),
        fasta_fai_sha256=_sha("fasta-fai"),
    )
    receipt = reserve_r1c_target_attempt(
        repo=repo,
        identity=identity,
        method_context=_r1c_method_context(
            biological_target_key=identity["biological_target_key"]
        ),
    )
    path = Path(receipt["receipt_path"])
    assert path.parent == repo / ".git/gfmcd_target_attempts/r1c"
    assert path.is_file()
    assert receipt["schema"] == "gfmcd.r1c.target_attempt/2"
    assert receipt["sealed_cohort_constructed_before_record"] is False
    assert receipt["model_loaded_before_record"] is False
    assert receipt["output_inspected_before_record"] is False
    assert receipt["biological_target_key"] == identity["biological_target_key"]

    # Endpoint labels, construction, code, and stack are context only: none may mint
    # a fresh receipt for the same checkpoint and exact sealed cohort.
    with pytest.raises(R1CTargetSpentError, match="already spent"):
        reserve_r1c_target_attempt(
            repo=repo,
            identity=identity,
            method_context=_r1c_method_context(
                "redesigned",
                biological_target_key=identity["biological_target_key"],
            ),
        )


def test_r1c_completion_is_create_only_after_exact_artifact_and_manifest_publication(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    identity = build_r1c_biological_target_identity(
        checkpoint_sha256=_sha("checkpoint"),
        gtf_sha256=_sha("gtf"),
        fasta_sha256=_sha("fasta"),
        fasta_fai_sha256=_sha("fasta-fai"),
    )
    context = _r1c_method_context(
        biological_target_key=identity["biological_target_key"]
    )
    receipt = reserve_r1c_target_attempt(
        repo=repo, identity=identity, method_context=context
    )
    receipt_path = Path(receipt["receipt_path"])
    attempt_identity = {
        "logical_name": receipt_path.name,
        "size_bytes": receipt_path.stat().st_size,
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "attempt_id": receipt["attempt_id"],
        "biological_target_key": receipt["biological_target_key"],
        "status": receipt["status"],
    }
    cohort_digests = {
        "source_cohort_sha256": _sha("source-cohort"),
        "ordered_example_ids_sha256": _sha("ordered-example-ids"),
        "target_gene_ids_sha256": _sha("sealed-genes"),
        "ordered_gene_ids_sha256": _sha("ordered-gene-ids"),
        "ordered_six_position_rows_sha256": _sha("sealed-rows"),
        "ordered_partner_margin_levels_sha256": _sha("ordered-partner-margins"),
        "construction_sha256": _sha("observed-construction"),
    }
    artifact = tmp_path / context["requested_artifact_logical_name"]
    manifest = tmp_path / context["requested_manifest_logical_name"]
    artifact.write_text(
        json.dumps(
            {
                "target_attempt": attempt_identity,
                "target_completion": {
                    "logical_name": context["declared_completion_logical_name"]
                },
                "target_cohort_identity": cohort_digests,
            },
            sort_keys=True,
        )
        + "\n"
    )
    artifact_identity = {
        "logical_name": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    manifest.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "path": f"/old/checkout/{artifact.name}",
                        "size_bytes": artifact_identity["size_bytes"],
                        "sha256": artifact_identity["sha256"],
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n"
    )

    completion = complete_r1c_target_attempt(
        receipt=receipt,
        artifact_path=artifact,
        manifest_path=manifest,
        cohort_digests=cohort_digests,
        method_fingerprint_sha256=context["instrument_fingerprint_sha256"],
        r5_stack_fingerprint_sha256=context["r5_stack_fingerprint_sha256"],
        official_verdict="CONFIRMED_PAIRED_DONOR_ACCEPTOR_BEHAVIOR",
        all_gates_pass=True,
        gates={"instrument": True, "robustness": True},
    )
    completion_path = Path(completion["completion_path"])
    persisted = json.loads(completion_path.read_text())
    assert persisted["schema"] == "gfmcd.r1c.target_completion/1"
    assert persisted["status"] == "TARGET_PUBLICATION_COMPLETE"
    assert persisted["artifact"] == artifact_identity
    assert persisted["attempt"] == {
        key: attempt_identity[key]
        for key in ("logical_name", "size_bytes", "sha256")
    }
    assert persisted["cohort_digests"] == cohort_digests
    assert persisted["attempt_id"] == receipt["attempt_id"]
    assert persisted["biological_target_key"] == identity["biological_target_key"]
    with pytest.raises(R1CTargetSpentError, match="already spent"):
        complete_r1c_target_attempt(
            receipt=receipt,
            artifact_path=artifact,
            manifest_path=manifest,
            cohort_digests=cohort_digests,
            method_fingerprint_sha256=context["instrument_fingerprint_sha256"],
            r5_stack_fingerprint_sha256=context["r5_stack_fingerprint_sha256"],
            official_verdict="RETRY_FORBIDDEN",
            all_gates_pass=True,
            gates={"instrument": True, "robustness": True},
        )


def test_r5_target_receipt_is_stable_across_repackaging_and_create_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    identity = build_r5_biological_target_identity(
        stage="R5a",
        checkpoint_sha256=_sha("checkpoint"),
        target_gene_ids_sha256=_sha("genes"),
        ordered_rows_sha256=_sha("rows"),
        construction_sha256=_sha("construction"),
        endpoint_policy="single masked acceptor G hard margin",
    )
    first_context = {
        "r5_stack_fingerprint_sha256": _sha("stack-v1"),
        "stage_preflight_artifact_sha256": _sha("preflight"),
        "stage_preflight_manifest_sha256": _sha("preflight-manifest"),
        "upstream_package_chain_sha256": _sha("upstreams"),
    }
    receipt = reserve_r5_target_attempt(
        repo=repo,
        identity=identity,
        attempt_context=first_context,
    )
    assert Path(receipt["receipt_path"]).is_file()
    assert receipt["biological_target_key"] == identity["biological_target_key"]
    assert receipt["attempt_context"] == first_context

    # Changing code/package context cannot create a fresh biological spend key.
    second_context = dict(first_context)
    second_context["r5_stack_fingerprint_sha256"] = _sha("stack-v2")
    with pytest.raises(R5TargetSpentError, match="already spent"):
        reserve_r5_target_attempt(
            repo=repo,
            identity=identity,
            attempt_context=second_context,
        )

    # Nor can an outcome-informed method rename or construction change make the same
    # checkpoint/cohort/stage biologically fresh again.
    redesigned = build_r5_biological_target_identity(
        stage="R5a",
        checkpoint_sha256=_sha("checkpoint"),
        target_gene_ids_sha256=_sha("genes"),
        ordered_rows_sha256=_sha("rows"),
        construction_sha256=_sha("post-outcome-construction"),
        endpoint_policy="renamed post-outcome endpoint",
    )
    assert redesigned["biological_target_key"] == identity["biological_target_key"]
    with pytest.raises(R5TargetSpentError, match="already spent"):
        reserve_r5_target_attempt(
            repo=repo,
            identity=redesigned,
            attempt_context=second_context,
        )

    completion = complete_r5_target_attempt(
        receipt=receipt,
        result_identity={"size_bytes": 123, "sha256": _sha("result")},
        manifest_identity={"size_bytes": 456, "sha256": _sha("manifest")},
        official_disposition="SCIENTIFIC_STOP_POSITION_ASSIGNMENT_NOT_SUPPORTED",
    )
    assert Path(completion["completion_path"]).is_file()
    persisted = json.loads(Path(completion["completion_path"]).read_text())
    assert persisted["receipt_sha256"] == completion["receipt_sha256"]
    with pytest.raises(R5TargetSpentError, match="completion already exists"):
        complete_r5_target_attempt(
            receipt=receipt,
            result_identity={"size_bytes": 123, "sha256": _sha("result")},
            manifest_identity={"size_bytes": 456, "sha256": _sha("manifest")},
            official_disposition="RETRY",
        )
