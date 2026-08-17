"""Authenticated, relocation-safe upstream contracts for R5.

R5 has two different upstream dependencies and must not conflate them:

* the exact committed R3d target supplies the sole scientific entry decision, and
  only its preregistered Path A (L3H1 reuse) is decision-bearing;
* the exact committed R3c preflight and R3d exposure registry supply provenance
  and complete gene exclusions, not scientific entry support.

All package identities below are path-free.  Embedded historical absolute paths are
treated as provenance text; sibling binding is checked by logical basename, byte size,
and SHA-256 so an exact package remains usable after repository relocation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


R3D_ENTRY_SCHEMA = "gfmcd.r3d.acceptor_specificity_adjudication/1"
R5_ENTRY_SCHEMA = "gfmcd.r5.entry_contract/2"
R5_EXCLUSION_REGISTRY_SCHEMA = "gfmcd.r5.exclusion_registry/2"

R3D_TARGET_ARTIFACT_SHA256 = (
    "e33d37447cbba368a48beea108fc49c06c56adfbeef2c0f95f019c227d60cb88"
)
R3D_TARGET_MANIFEST_SHA256 = (
    "53f13aa4884b53a91eba6efa2091c7e7638672ed260bf6558f20547d88f899cd"
)
R3D_REGISTRY_ARTIFACT_SHA256 = (
    "17ad316d5f9c478e37b91f34d87bc1d0c6a714f411947419e9dbcced2443598c"
)
R3D_REGISTRY_MANIFEST_SHA256 = (
    "745d7a31837c3575edcd9816464f82fc7f23b440b00b06fde4f6185a5f50afde"
)
R3C_PREFLIGHT_ARTIFACT_SHA256 = (
    "c0c9306fe894ce73bfd981a14c85e2aec6660da33f01d9716cee4327b3ff2d3e"
)
R3C_PREFLIGHT_MANIFEST_SHA256 = (
    "4a53f96b365c7bed227c8e0c995cee65a022620994e8898bab988b354fce41bb"
)
R3D_TARGET_ATTEMPT_SHA256 = (
    "64f25327d64e1397941ec669a694ed3cdc3a93938c7615fe56fbe2a562d6f0c0"
)
R3D_TARGET_COMPLETION_SHA256 = (
    "16dd969702b0c74b30bb8cbae2ffb08fe41eca81b5be5f4d57326c618c96510f"
)

R3D_TARGET_ARTIFACT_SIZE = 10_267_233
R3D_TARGET_MANIFEST_SIZE = 13_255
R3D_REGISTRY_ARTIFACT_SIZE = 47_380
R3D_REGISTRY_MANIFEST_SIZE = 87_831
R3C_PREFLIGHT_ARTIFACT_SIZE = 33_361_804
R3C_PREFLIGHT_MANIFEST_SIZE = 9_620
R3D_TARGET_ATTEMPT_SIZE = 13_730
R3D_TARGET_COMPLETION_SIZE = 1_153

R3D_TARGET_GENERATING_COMMIT = "ac17370ceaea57122fe7773c07fd70a7d22315f9"
R3D_TARGET_PUBLICATION_COMMIT = "0b5dc1ec1e70f42cbab94bd7ea409051bc1d5a34"
R3C_PREFLIGHT_GENERATING_COMMIT = "bb22dff7e2f3ec4e3de73a8d45ad1740793b66f4"
R3D_TARGET_IDENTITY_SHA256 = (
    "b8ff2fedfd3add86c88d882a0b1c36788e2f41cafb583ff1179c4779d0d4f839"
)
R3D_TARGET_ATTEMPT_ID = (
    "3f140f56ccd69b991a98199af4c9f96c10aedd9e8256906651482e8fc997394d"
)
R3D_PATH_A_ALPHA = 0.05
R3D_PATH_A_INTERSECTION_UNION_P_VALUE = 0.00024993751562109475

HISTORICAL_R3_GENE_COUNT = 645
HISTORICAL_R3_GENE_IDS_SHA256 = (
    "0964c06136fd9a8fa9251ad13a2ca4d45e3f0c8cea45323dbdc1fe2092b8fda0"
)
R3C_FIXED_GENE_COUNT = 588
R3C_FIXED_GENE_IDS_SHA256 = (
    "67c745eef7d223d1b8b2769c87f5d818c955a042ae4a303459a473c1ca528a96"
)
R3C_FIXED_BUCKET_IDENTITIES = {
    "0": {
        "gene_count": 284,
        "gene_ids_sha256": (
            "8745477f1fe458173774a1a905d569574ed917169621cfbfacbd7b8a65746d6d"
        ),
    },
    "1": {
        "gene_count": 304,
        "gene_ids_sha256": (
            "484b2c4a11f47c608ea51bcaf2a4234d3197a56f6a10ffb38be9706e5fbd2c5e"
        ),
    },
}
R3D_REGISTRY_GENE_COUNT = 1_112
R3D_REGISTRY_GENE_IDS_SHA256 = (
    "378aa6b5b9408b6bb3311b883cc9ba984678dc287f442071a82dc8a9071efc09"
)
R3D_TARGET_GENE_COUNT = 3_909
R3D_TARGET_GENE_IDS_SHA256 = (
    "c41dbd0222f7a7364f9d8558e06854ad758228d8962b309dd42a54170c590cd4"
)
R3D_REGISTRY_AND_TARGET_UNION_GENE_COUNT = 5_021
R3D_REGISTRY_AND_TARGET_UNION_GENE_IDS_SHA256 = (
    "2d64dab36515e9d6344f73164a91c668e15d50f15003cfdd3a5a4ecc87e9fd13"
)
R5_PRIOR_EXPOSURE_UNION_GENE_COUNT = 6_254
R5_PRIOR_EXPOSURE_UNION_GENE_IDS_SHA256 = (
    "14f2cbb7d1aa4764da9825a481bdb1f2002e2d47022b0292f809494e163e9ed3"
)
R3D_ENTRY_DOES_NOT_LICENSE = frozenset(
    {
        "sealed-split or independent confirmation",
        "attention as mechanism",
        "address or payload identity",
        "acceptor-side write",
        "delivery or mediation",
        "sufficiency or faithfulness",
        "biological mechanism",
    }
)
R3D_ENTRY_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        "decision_path",
        "classification",
        "health_ok",
        "intersection_union_p_value",
        "alpha",
        "path_b_used_to_gate_entry",
        "licensed_entry",
        "not_sealed_or_independent_confirmation",
        "does_not_license",
    }
)
R5_EXPOSURE_SCOPE_LIMITATION = (
    "all historical candidate-selection exposure outside the four authenticated "
    "registries"
)
R5_EXCLUSION_REGISTRY_FIELDS = frozenset(
    {
        "schema",
        "derivation",
        "scope",
        "source_identity",
        "upstream_packages",
        "gene_sets",
        "r3c_gene_ids_by_bucket",
        "pairwise_intersection_counts",
        "all_pairwise_intersections_zero",
        "r3d_registry_and_target_union_gene_count",
        "r3d_registry_and_target_union_gene_ids_sha256",
        "prior_exposure_union_gene_count",
        "prior_exposure_union_gene_ids_sha256",
        "record_sha256",
    }
)

PINNED_R5_SOURCE_IDENTITY: dict[str, dict[str, Any]] = {
    "gtf": {
        "label": "GENCODE_v50_comprehensive",
        "size_bytes": 124_527_720,
        "sha256": "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453",
    },
    "fasta": {
        "label": "gfmcd_local_hg38.fa@local_GRCh38_reference_v1",
        "size_bytes": 3_273_481_150,
        "sha256": "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51",
    },
    "fasta_fai": {
        "label": "gfmcd_local_hg38.fa.fai@local_GRCh38_reference_v1",
        "size_bytes": 19_381,
        "sha256": "3b425de206296a5c8053023fa5ca61da43cfe78c1737c12e58c83367c7e83c21",
    },
}
PINNED_R5_REPOSITORY_RUNTIME_IDENTITY: dict[str, dict[str, Any]] = {
    "pyproject.toml": {
        "size_bytes": 963,
        "sha256": "bcd872b15c944e24372935963f0fdd9fa33613aa47c4c7119bb79ede4629fa30",
    },
    "uv.lock": {
        "size_bytes": 836_629,
        "sha256": "47a96aca98a0be346c7d8e227f7ef1cda13c41a99e6b5a19ab01b12bd02d8961",
    },
}
PINNED_CHECKPOINT_AGGREGATE_SHA256 = (
    "fb2a04e0c9b5247b44f03cd23581632d5924b86fcafaffa4c6f2b8d01ae1da8d"
)


def _identity(logical_name: str, size_bytes: int, sha256: str) -> dict[str, Any]:
    return {
        "logical_name": logical_name,
        "size_bytes": int(size_bytes),
        "sha256": str(sha256),
    }


UPSTREAM_PACKAGE_IDENTITIES = {
    "r3d_target": {
        "artifact": _identity(
            "r3d_acceptor_specificity_target_v1.json",
            R3D_TARGET_ARTIFACT_SIZE,
            R3D_TARGET_ARTIFACT_SHA256,
        ),
        "manifest": _identity(
            "r3d_acceptor_specificity_target_v1.manifest.json",
            R3D_TARGET_MANIFEST_SIZE,
            R3D_TARGET_MANIFEST_SHA256,
        ),
        "generating_commit": R3D_TARGET_GENERATING_COMMIT,
        "publication_commit": R3D_TARGET_PUBLICATION_COMMIT,
    },
    "r3d_registry": {
        "artifact": _identity(
            "r3d_train_exposure_registry_v1.json",
            R3D_REGISTRY_ARTIFACT_SIZE,
            R3D_REGISTRY_ARTIFACT_SHA256,
        ),
        "manifest": _identity(
            "r3d_train_exposure_registry_v1.manifest.json",
            R3D_REGISTRY_MANIFEST_SIZE,
            R3D_REGISTRY_MANIFEST_SHA256,
        ),
    },
    "r3c_preflight": {
        "artifact": _identity(
            "r3c_acceptor_head_preflight_v2_1.json",
            R3C_PREFLIGHT_ARTIFACT_SIZE,
            R3C_PREFLIGHT_ARTIFACT_SHA256,
        ),
        "manifest": _identity(
            "r3c_acceptor_head_preflight_v2_1.manifest.json",
            R3C_PREFLIGHT_MANIFEST_SIZE,
            R3C_PREFLIGHT_MANIFEST_SHA256,
        ),
        "generating_commit": R3C_PREFLIGHT_GENERATING_COMMIT,
        "role": "exclusion_provenance_only",
    },
    "r3d_target_attempt": _identity(
        f"{R3D_TARGET_IDENTITY_SHA256}.json",
        R3D_TARGET_ATTEMPT_SIZE,
        R3D_TARGET_ATTEMPT_SHA256,
    ),
    "r3d_target_completion": _identity(
        f"{R3D_TARGET_IDENTITY_SHA256}.completion.json",
        R3D_TARGET_COMPLETION_SIZE,
        R3D_TARGET_COMPLETION_SHA256,
    ),
}


class R5UpstreamProvenanceError(ValueError):
    """Raised when an upstream package or exposure registry differs."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_exact_json(
    path_value: str | Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path_value).expanduser().resolve(strict=True)
    observed = _identity(path.name, path.stat().st_size, sha256_file(path))
    if (
        observed["size_bytes"] != expected.get("size_bytes")
        or observed["sha256"] != expected.get("sha256")
        or observed["logical_name"] != expected.get("logical_name")
    ):
        raise R5UpstreamProvenanceError(f"{label} differs from its exact byte pin")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R5UpstreamProvenanceError(f"{label} is not valid JSON") from exc
    if not isinstance(record, dict):
        raise R5UpstreamProvenanceError(f"{label} must contain a JSON object")
    return record, dict(expected)


def _sibling_manifest_path(artifact_path: str | Path) -> Path:
    artifact = Path(artifact_path).expanduser()
    return artifact.with_suffix(".manifest.json")


def _validate_manifest_output(
    manifest: Mapping[str, Any], artifact_identity: Mapping[str, Any]
) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise R5UpstreamProvenanceError("upstream sibling manifest lacks outputs")
    logical_name = str(artifact_identity["logical_name"])
    matches = [
        item
        for item in outputs
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).name == logical_name
        and item.get("sha256") == artifact_identity["sha256"]
        and item.get("size_bytes") == artifact_identity["size_bytes"]
    ]
    if len(matches) != 1:
        raise R5UpstreamProvenanceError(
            "upstream sibling manifest does not bind exactly one logical output"
        )


def _sorted_unique_genes(
    values: Any,
    *,
    label: str,
    expected_count: int,
    expected_sha256: str,
) -> list[str]:
    if not isinstance(values, list):
        raise R5UpstreamProvenanceError(f"{label} gene list is missing")
    genes = list(map(str, values))
    if (
        genes != sorted(set(genes))
        or any(not gene for gene in genes)
        or len(genes) != expected_count
        or canonical_sha256(genes) != expected_sha256
    ):
        raise R5UpstreamProvenanceError(f"{label} gene identity differs")
    return genes


def validate_r3d_target_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate R3d's exact scientific-entry semantics using Path A only.

    Path B is intentionally neither read nor returned as an entry condition.  It was a
    candidate-generating component pattern and cannot authorize an R5 mechanism claim.
    """

    if (
        record.get("schema") != R3D_ENTRY_SCHEMA
        or record.get("experiment") != "R3d_acceptor_specificity_adjudication"
        or record.get("stage") != "target"
        or record.get("status") != "PROSPECTIVE_VERIFICATION"
        or record.get("tier") != "PROSPECTIVE_ENDPOINT_SPECIFIC_TRAIN_VERIFICATION"
        or record.get("not_sealed_or_independent_confirmation") is not True
        or record.get("split") != "train"
        or record.get("n") != R3D_TARGET_GENE_COUNT
        or record.get("n_genes") != R3D_TARGET_GENE_COUNT
        or record.get("primary_estimand")
        != "raw masked-acceptor-G logit-margin component drop"
        or record.get("verdict")
        != "PROSPECTIVE_TRAIN_ENDPOINT_VERIFICATION_COMPLETE"
        or record.get("all_gates_pass") is not True
        or record.get("official_scientific_disposition_gates_pass") is not True
    ):
        raise R5UpstreamProvenanceError("R3d target context differs")
    gates = record.get("gates")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
    ):
        raise R5UpstreamProvenanceError("R3d target has a failed global gate")
    git_identity = record.get("git_authorization_identity")
    if git_identity != {"commit": R3D_TARGET_GENERATING_COMMIT, "dirty": False}:
        raise R5UpstreamProvenanceError("R3d target clean Git identity differs")

    construction = record.get("construction")
    policy = construction.get("policy") if isinstance(construction, Mapping) else None
    if (
        not isinstance(policy, Mapping)
        or construction.get("schema") != "gfmcd.r3c.acceptor_only_construction/1"
        or policy.get("readout") != "mask exactly one G at next_acceptor_pos - 1"
        or policy.get("paired_operator") != "GT -> AC at donor_pos"
        or policy.get("pseudo_operator") != "GT -> AC at matched_decoy_donor"
        or policy.get("donor_readout_constructed") is not False
        or construction.get("rows") != R3D_TARGET_GENE_COUNT
    ):
        raise R5UpstreamProvenanceError(
            "R3d entry is not the exact masked-acceptor/no-masked-donor construction"
        )

    expected_requirements = {
        "within_example_token_mean_ablation|L3H1|partner": "MATERIAL_POSITIVE",
        "within_example_token_mean_ablation|L3H1|pseudo": "EQUIVALENT_WITHIN_SESOI",
        "within_example_token_mean_ablation|L3H1|specific": "MATERIAL_POSITIVE",
        "activation_noising|L3H1|partner": "MATERIAL_POSITIVE",
        "activation_noising|L3H1|pseudo": "EQUIVALENT_WITHIN_SESOI",
        "activation_noising|L3H1|specific": "MATERIAL_POSITIVE",
    }
    paths = record.get("paths")
    path_a = paths.get("L3H1_reused_component") if isinstance(paths, Mapping) else None
    if not isinstance(path_a, Mapping):
        raise R5UpstreamProvenanceError("R3d Path A is missing")
    rows = path_a.get("rows")
    health = path_a.get("health")
    if (
        path_a.get("path") != "L3H1_reused_component"
        or path_a.get("classification") != "SUPPORTED"
        or path_a.get("official_scientific_disposition") is not True
        or path_a.get("all_edges_reject") is not True
        or path_a.get("test_interval_coherence") is not True
        or path_a.get("strict_threshold") != "p < alpha"
        or path_a.get("requirements") != expected_requirements
        or not isinstance(health, Mapping)
        or health.get("ok") is not True
        or not isinstance(rows, Mapping)
        or set(rows) != set(expected_requirements)
        or any(
            not isinstance(rows[name], Mapping)
            or rows[name].get("required") != required
            or rows[name].get("classification") != required
            or rows[name].get("strict_requirement_pass") is not True
            or rows[name].get("strict_test_interval_coherence") is not True
            for name, required in expected_requirements.items()
        )
    ):
        raise R5UpstreamProvenanceError("R3d Path A is not supported exactly")
    alpha = path_a.get("alpha")
    p_value = path_a.get("intersection_union_p_value")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or isinstance(p_value, bool)
        or not isinstance(p_value, (int, float))
        or float(alpha) != R3D_PATH_A_ALPHA
        or float(p_value) != R3D_PATH_A_INTERSECTION_UNION_P_VALUE
        or not float(p_value) < float(alpha)
    ):
        raise R5UpstreamProvenanceError("R3d Path A fixed-sequence decision differs")

    does_not_license = record.get("does_not_license")
    if not isinstance(does_not_license, list) or not R3D_ENTRY_DOES_NOT_LICENSE.issubset(
        set(map(str, does_not_license))
    ):
        raise R5UpstreamProvenanceError("R3d entry limitation contract differs")

    return {
        "schema": R3D_ENTRY_SCHEMA,
        "decision_path": "L3H1_reused_component",
        "classification": "SUPPORTED",
        "health_ok": True,
        "intersection_union_p_value": float(p_value),
        "alpha": float(alpha),
        "path_b_used_to_gate_entry": False,
        "licensed_entry": (
            "registry-disjoint masked-acceptor endpoint support for L3H1 reuse"
        ),
        "not_sealed_or_independent_confirmation": True,
        "does_not_license": sorted(R3D_ENTRY_DOES_NOT_LICENSE),
    }


def _load_r3d_registry_package(
    registry_path: str | Path, manifest_path: str | Path | None
) -> dict[str, Any]:
    expected = UPSTREAM_PACKAGE_IDENTITIES["r3d_registry"]
    record, artifact_identity = _load_exact_json(
        registry_path, expected["artifact"], label="R3d direct-exposure registry"
    )
    manifest_value = (
        _sibling_manifest_path(registry_path)
        if manifest_path is None
        else Path(manifest_path)
    )
    manifest, manifest_identity = _load_exact_json(
        manifest_value,
        expected["manifest"],
        label="R3d direct-exposure registry manifest",
    )
    _validate_manifest_output(manifest, artifact_identity)
    genes = _sorted_unique_genes(
        record.get("gene_ids"),
        label="R3d direct-exposure registry",
        expected_count=R3D_REGISTRY_GENE_COUNT,
        expected_sha256=R3D_REGISTRY_GENE_IDS_SHA256,
    )
    scope = record.get("scope")
    if (
        record.get("schema") != "gfmcd.r3d.exposed_gene_registry/1"
        or record.get("status") != "REVIEWED_MODEL_FREE_EXPOSURE_SCOPE_REGISTRY"
        or record.get("gene_ids_sha256") != R3D_REGISTRY_GENE_IDS_SHA256
        or record.get("scope_id")
        != "claude_inverse_acceptor_and_clean_alt_head_screen_direct_exposures_v1"
        or not isinstance(scope, Mapping)
        or scope.get("global_freshness_claimed") is not False
        or manifest.get("schema")
        != "gfmcd.r3d.exposed_gene_registry_manifest/1"
        or manifest.get("status")
        != "REVIEWED_MODEL_FREE_EXPOSURE_SCOPE_REGISTRY"
    ):
        raise R5UpstreamProvenanceError("R3d exposure-registry semantics differ")
    inputs = record.get("input_bindings")
    if not isinstance(inputs, Mapping):
        raise R5UpstreamProvenanceError("R3d registry input bindings are missing")
    expected_inputs = {
        "annotation": PINNED_R5_SOURCE_IDENTITY["gtf"],
        "genome": PINNED_R5_SOURCE_IDENTITY["fasta"],
        "genome_index": PINNED_R5_SOURCE_IDENTITY["fasta_fai"],
    }
    for key, expected_source in expected_inputs.items():
        observed = inputs.get(key)
        if (
            not isinstance(observed, Mapping)
            or observed.get("size_bytes") != expected_source["size_bytes"]
            or observed.get("sha256") != expected_source["sha256"]
        ):
            raise R5UpstreamProvenanceError("R3d registry source bytes differ")
    return {
        "artifact": artifact_identity,
        "manifest": manifest_identity,
        "gene_ids": genes,
        "gene_count": len(genes),
        "gene_ids_sha256": canonical_sha256(genes),
        "scope_id": record["scope_id"],
        "global_freshness_claimed": False,
    }


def _load_target_attempts(
    record: Mapping[str, Any],
    attempt_path: str | Path | None,
    completion_path: str | Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if (attempt_path is None) != (completion_path is None):
        raise R5UpstreamProvenanceError(
            "R3d target attempt and completion must be supplied together"
        )
    if attempt_path is None:
        return None, None
    attempt, attempt_identity = _load_exact_json(
        attempt_path,
        UPSTREAM_PACKAGE_IDENTITIES["r3d_target_attempt"],
        label="R3d target-attempt receipt",
    )
    completion, completion_identity = _load_exact_json(
        completion_path,
        UPSTREAM_PACKAGE_IDENTITIES["r3d_target_completion"],
        label="R3d target-completion receipt",
    )
    embedded = record.get("target_attempt")
    if (
        attempt.get("schema") != "gfmcd.r3d.target_attempt/1"
        or attempt.get("status")
        != "TARGET_SPENT_BEFORE_COHORT_OR_MODEL_OUTCOME"
        or attempt.get("attempt_id") != R3D_TARGET_ATTEMPT_ID
        or attempt.get("target_identity_sha256") != R3D_TARGET_IDENTITY_SHA256
        or attempt.get("cohort_materialized_before_record") is not False
        or attempt.get("model_outcome_opened_before_record") is not False
        or not isinstance(embedded, Mapping)
        or embedded.get("sha256") != R3D_TARGET_ATTEMPT_SHA256
        or embedded.get("attempt_id") != R3D_TARGET_ATTEMPT_ID
        or embedded.get("target_identity_sha256") != R3D_TARGET_IDENTITY_SHA256
        or embedded.get("receipt") != attempt
    ):
        raise R5UpstreamProvenanceError("R3d target-attempt receipt differs")
    details = completion.get("details")
    result = details.get("result") if isinstance(details, Mapping) else None
    manifest = details.get("manifest") if isinstance(details, Mapping) else None
    classifications = (
        details.get("official_path_classifications")
        if isinstance(details, Mapping)
        else None
    )
    if (
        completion.get("schema") != "gfmcd.r3d.target_attempt_completion/1"
        or completion.get("status") != "TARGET_PUBLICATION_COMPLETE"
        or completion.get("attempt_id") != R3D_TARGET_ATTEMPT_ID
        or completion.get("attempt_sha256") != R3D_TARGET_ATTEMPT_SHA256
        or not isinstance(result, Mapping)
        or result.get("sha256") != R3D_TARGET_ARTIFACT_SHA256
        or result.get("size_bytes") != R3D_TARGET_ARTIFACT_SIZE
        or not isinstance(manifest, Mapping)
        or manifest.get("sha256") != R3D_TARGET_MANIFEST_SHA256
        or manifest.get("size_bytes") != R3D_TARGET_MANIFEST_SIZE
        or not isinstance(classifications, Mapping)
        or classifications.get("L3H1_reused_component") != "SUPPORTED"
        or details.get("verdict")
        != "PROSPECTIVE_TRAIN_ENDPOINT_VERIFICATION_COMPLETE"
    ):
        raise R5UpstreamProvenanceError("R3d target-completion receipt differs")
    return attempt_identity | {"attempt_id": R3D_TARGET_ATTEMPT_ID}, completion_identity


def load_r3d_entry_package(
    artifact_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    registry_path: str | Path | None = None,
    registry_manifest_path: str | Path | None = None,
    attempt_path: str | Path | None = None,
    completion_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the exact committed R3d target and its complete exposure package."""

    expected = UPSTREAM_PACKAGE_IDENTITIES["r3d_target"]
    record, artifact_identity = _load_exact_json(
        artifact_path, expected["artifact"], label="R3d target artifact"
    )
    manifest_value = (
        _sibling_manifest_path(artifact_path)
        if manifest_path is None
        else Path(manifest_path)
    )
    manifest, manifest_identity = _load_exact_json(
        manifest_value, expected["manifest"], label="R3d target manifest"
    )
    _validate_manifest_output(manifest, artifact_identity)
    entry = validate_r3d_target_record(record)
    if (
        manifest.get("experiment_id")
        != "splicebert_acceptor_coupling:r3d_acceptor_specificity_adjudication"
        or manifest.get("status") != "PROSPECTIVE_VERIFICATION"
        or manifest.get("git_dirty") is not False
        or manifest.get("git_commit") != R3D_TARGET_GENERATING_COMMIT
        or not isinstance(manifest.get("config"), Mapping)
        or manifest["config"].get("stage") != "target"
        or manifest["config"].get("n_genes") != R3D_TARGET_GENE_COUNT
        or manifest["config"].get("target_attempt_id") != R3D_TARGET_ATTEMPT_ID
        or manifest["config"].get("instrument_fingerprint_sha256")
        != record.get("instrument_fingerprint_sha256")
    ):
        raise R5UpstreamProvenanceError("R3d target manifest context differs")

    if registry_path is None:
        candidate = Path(artifact_path).expanduser().with_name(
            "r3d_train_exposure_registry_v1.json"
        )
        if not candidate.is_file():
            raise R5UpstreamProvenanceError(
                "R3d target requires its exact direct-exposure registry package"
            )
        registry_path = candidate
    registry = _load_r3d_registry_package(registry_path, registry_manifest_path)

    cohort = record.get("cohort_audit")
    target_identity = record.get("target_identity")
    target = target_identity.get("target") if isinstance(target_identity, Mapping) else None
    selected = cohort.get("selected_gene_ids") if isinstance(cohort, Mapping) else None
    if not isinstance(selected, list):
        raise R5UpstreamProvenanceError("R3d target gene list is missing")
    selected_strings = list(map(str, selected))
    target_genes = sorted(set(selected_strings))
    if (
        len(selected_strings) != R3D_TARGET_GENE_COUNT
        or len(target_genes) != R3D_TARGET_GENE_COUNT
        or canonical_sha256(target_genes) != R3D_TARGET_GENE_IDS_SHA256
        or not isinstance(target, Mapping)
        or target.get("genes_one_row_each") != R3D_TARGET_GENE_COUNT
        or target.get("gene_ids_sha256") != R3D_TARGET_GENE_IDS_SHA256
        or target_identity.get("registry_gene_ids_sha256")
        != R3D_REGISTRY_GENE_IDS_SHA256
        or cohort.get("registry_gene_intersection_count") != 0
        or set(target_genes) & set(registry["gene_ids"])
    ):
        raise R5UpstreamProvenanceError("R3d target/registry gene identity differs")
    registry_record = record.get("registry")
    if (
        not isinstance(registry_record, Mapping)
        or registry_record.get("global_gene_freshness_claimed") is not False
        or registry_record.get("gene_count") != R3D_REGISTRY_GENE_COUNT
        or registry_record.get("gene_ids_sha256") != R3D_REGISTRY_GENE_IDS_SHA256
        or registry_record.get("file_sha256") != R3D_REGISTRY_ARTIFACT_SHA256
    ):
        raise R5UpstreamProvenanceError("R3d target embeds a different registry identity")

    hashes = record.get("hashes")
    if not isinstance(hashes, Mapping):
        raise R5UpstreamProvenanceError("R3d target hash registry is missing")
    required_hashes = {
        "r3c_v2_1_preflight_sha256": R3C_PREFLIGHT_ARTIFACT_SHA256,
        "r3c_v2_1_preflight_manifest_sha256": R3C_PREFLIGHT_MANIFEST_SHA256,
        "exposed_gene_registry_sha256": R3D_REGISTRY_ARTIFACT_SHA256,
        "exposed_gene_registry_manifest_sha256": R3D_REGISTRY_MANIFEST_SHA256,
        "gtf_sha256": PINNED_R5_SOURCE_IDENTITY["gtf"]["sha256"],
        "fasta_sha256": PINNED_R5_SOURCE_IDENTITY["fasta"]["sha256"],
        "fasta_index_sha256": PINNED_R5_SOURCE_IDENTITY["fasta_fai"]["sha256"],
        "checkpoint_aggregate_sha256": PINNED_CHECKPOINT_AGGREGATE_SHA256,
    }
    if any(hashes.get(key) != value for key, value in required_hashes.items()):
        raise R5UpstreamProvenanceError("R3d target upstream/source hashes differ")
    checkpoint_and_data = target_identity.get("checkpoint_and_data")
    if checkpoint_and_data != {
        "gtf_sha256": required_hashes["gtf_sha256"],
        "fasta_sha256": required_hashes["fasta_sha256"],
        "fasta_index_sha256": required_hashes["fasta_index_sha256"],
        "checkpoint_aggregate_sha256": PINNED_CHECKPOINT_AGGREGATE_SHA256,
    }:
        raise R5UpstreamProvenanceError("R3d target checkpoint/data identity differs")

    attempt, completion = _load_target_attempts(
        record, attempt_path, completion_path
    )
    package = {
        "artifact": artifact_identity,
        "manifest": manifest_identity,
        "generating_commit": R3D_TARGET_GENERATING_COMMIT,
        "publication_commit": R3D_TARGET_PUBLICATION_COMMIT,
        "entry": entry,
        "checkpoint_and_data": dict(checkpoint_and_data),
        "target_gene_ids": target_genes,
        "target_gene_count": len(target_genes),
        "target_gene_ids_sha256": canonical_sha256(target_genes),
        "registry": registry,
        "attempt": attempt,
        "completion": completion,
        "global_gene_freshness_claimed": False,
    }
    package["package_contract_sha256"] = canonical_sha256(package)
    return package


def load_r3c_exclusion_package(
    artifact_path: str | Path, manifest_path: str | Path | None = None
) -> dict[str, Any]:
    """Load exact R3c preflight bytes solely for historical/fixed exclusions."""

    expected = UPSTREAM_PACKAGE_IDENTITIES["r3c_preflight"]
    record, artifact_identity = _load_exact_json(
        artifact_path, expected["artifact"], label="R3c preflight artifact"
    )
    manifest_value = (
        _sibling_manifest_path(artifact_path)
        if manifest_path is None
        else Path(manifest_path)
    )
    manifest, manifest_identity = _load_exact_json(
        manifest_value, expected["manifest"], label="R3c preflight manifest"
    )
    _validate_manifest_output(manifest, artifact_identity)
    if (
        record.get("schema") != "gfmcd.r3c.acceptor_head_confirmation/2"
        or record.get("instrument_revision")
        != "r3c-v2.1-monotone-health-early-preflight-authorization"
        or record.get("stage") != "preflight"
        or record.get("status") != "DEVELOPMENT"
        or record.get("split") != "val"
        or record.get("gene_bucket") != 0
        or record.get("all_gates_pass") is not True
        or record.get("verdict") != "TARGET_BINDABLE_PREFLIGHT_GO"
        or record.get("git_authorization_identity")
        != {"commit": R3C_PREFLIGHT_GENERATING_COMMIT, "dirty": False}
        or manifest.get("experiment_id")
        != "splicebert_acceptor_coupling:r3c_acceptor_head_confirmation"
        or manifest.get("status") != "DEVELOPMENT"
        or manifest.get("git_dirty") is not False
        or manifest.get("git_commit") != R3C_PREFLIGHT_GENERATING_COMMIT
        or not isinstance(manifest.get("config"), Mapping)
        or manifest["config"].get("stage") != "preflight"
    ):
        raise R5UpstreamProvenanceError("R3c exclusion-provenance context differs")
    cohort = record.get("cohort_audit")
    historical_record = (
        cohort.get("historical_reconstruction")
        if isinstance(cohort, Mapping)
        else None
    )
    census = cohort.get("current_census") if isinstance(cohort, Mapping) else None
    fixed = (
        census.get("fixed_selection_gene_ids_by_bucket")
        if isinstance(census, Mapping)
        else None
    )
    if not isinstance(historical_record, Mapping) or not isinstance(fixed, Mapping):
        raise R5UpstreamProvenanceError("R3c exclusion gene registries are missing")
    historical = _sorted_unique_genes(
        historical_record.get("excluded_gene_ids"),
        label="historical R3",
        expected_count=HISTORICAL_R3_GENE_COUNT,
        expected_sha256=HISTORICAL_R3_GENE_IDS_SHA256,
    )
    if set(fixed) != {"0", "1"}:
        raise R5UpstreamProvenanceError("R3c fixed cohort buckets differ")
    by_bucket: dict[str, list[str]] = {}
    for bucket in ("0", "1"):
        bucket_record = fixed[bucket]
        if not isinstance(bucket_record, Mapping):
            raise R5UpstreamProvenanceError("R3c fixed cohort bucket is malformed")
        expected_bucket = R3C_FIXED_BUCKET_IDENTITIES[bucket]
        genes = _sorted_unique_genes(
            bucket_record.get("gene_ids"),
            label=f"R3c fixed cohort bucket {bucket}",
            expected_count=expected_bucket["gene_count"],
            expected_sha256=expected_bucket["gene_ids_sha256"],
        )
        if bucket_record.get("gene_ids_sha256") != expected_bucket["gene_ids_sha256"]:
            raise R5UpstreamProvenanceError("R3c fixed cohort bucket differs")
        by_bucket[bucket] = genes
    r3c_genes = sorted(set(by_bucket["0"]) | set(by_bucket["1"]))
    if (
        len(r3c_genes) != R3C_FIXED_GENE_COUNT
        or canonical_sha256(r3c_genes) != R3C_FIXED_GENE_IDS_SHA256
        or set(historical) & set(r3c_genes)
    ):
        raise R5UpstreamProvenanceError("R3c exclusion union differs")
    package = {
        "artifact": artifact_identity,
        "manifest": manifest_identity,
        "generating_commit": R3C_PREFLIGHT_GENERATING_COMMIT,
        "role": "exclusion_provenance_only",
        "scientific_entry_authorized": False,
        "historical_r3_gene_ids": historical,
        "historical_r3_gene_count": len(historical),
        "historical_r3_gene_ids_sha256": canonical_sha256(historical),
        "r3c_gene_ids_by_bucket": by_bucket,
        "r3c_gene_ids": r3c_genes,
        "r3c_gene_count": len(r3c_genes),
        "r3c_gene_ids_sha256": canonical_sha256(r3c_genes),
    }
    package["package_contract_sha256"] = canonical_sha256(package)
    return package


def _validate_loaded_package_identity(
    package: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key in expected:
        if package.get(key) != expected.get(key):
            raise R5UpstreamProvenanceError(f"{label} package identity differs")
    recorded = package.get("package_contract_sha256")
    payload = {key: value for key, value in package.items() if key != "package_contract_sha256"}
    if recorded != canonical_sha256(payload):
        raise R5UpstreamProvenanceError(f"{label} package contract digest differs")


def _pairwise_intersections(gene_sets: Mapping[str, Sequence[str]]) -> dict[str, int]:
    names = list(gene_sets)
    output: dict[str, int] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            output[f"{left}__{right}"] = len(
                set(map(str, gene_sets[left])) & set(map(str, gene_sets[right]))
            )
    return output


def derive_r5_exclusion_registry(
    r3c_package: Mapping[str, Any], r3d_package: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive R5's complete four-source prior-exposure registry."""

    _validate_loaded_package_identity(
        r3c_package,
        UPSTREAM_PACKAGE_IDENTITIES["r3c_preflight"],
        label="R3c preflight",
    )
    _validate_loaded_package_identity(
        r3d_package,
        UPSTREAM_PACKAGE_IDENTITIES["r3d_target"],
        label="R3d target",
    )
    registry_package = r3d_package.get("registry")
    if not isinstance(registry_package, Mapping):
        raise R5UpstreamProvenanceError("R3d registry package is missing")
    if (
        registry_package.get("artifact")
        != UPSTREAM_PACKAGE_IDENTITIES["r3d_registry"]["artifact"]
        or registry_package.get("manifest")
        != UPSTREAM_PACKAGE_IDENTITIES["r3d_registry"]["manifest"]
    ):
        raise R5UpstreamProvenanceError("R3d registry package identity differs")
    entry = r3d_package.get("entry")
    checkpoint_and_data = r3d_package.get("checkpoint_and_data")
    if (
        r3c_package.get("role") != "exclusion_provenance_only"
        or r3c_package.get("scientific_entry_authorized") is not False
        or not isinstance(entry, Mapping)
        or set(entry) != set(R3D_ENTRY_CONTRACT_FIELDS)
        or entry.get("schema") != R3D_ENTRY_SCHEMA
        or entry.get("decision_path") != "L3H1_reused_component"
        or entry.get("classification") != "SUPPORTED"
        or entry.get("health_ok") is not True
        or entry.get("alpha") != R3D_PATH_A_ALPHA
        or entry.get("intersection_union_p_value")
        != R3D_PATH_A_INTERSECTION_UNION_P_VALUE
        or entry.get("path_b_used_to_gate_entry") is not False
        or entry.get("not_sealed_or_independent_confirmation") is not True
        or entry.get("licensed_entry")
        != "registry-disjoint masked-acceptor endpoint support for L3H1 reuse"
        or not isinstance(entry.get("does_not_license"), list)
        or set(map(str, entry.get("does_not_license", ())))
        != set(R3D_ENTRY_DOES_NOT_LICENSE)
        or checkpoint_and_data
        != {
            "gtf_sha256": PINNED_R5_SOURCE_IDENTITY["gtf"]["sha256"],
            "fasta_sha256": PINNED_R5_SOURCE_IDENTITY["fasta"]["sha256"],
            "fasta_index_sha256": PINNED_R5_SOURCE_IDENTITY["fasta_fai"][
                "sha256"
            ],
            "checkpoint_aggregate_sha256": PINNED_CHECKPOINT_AGGREGATE_SHA256,
        }
        or registry_package.get("global_freshness_claimed") is not False
        or r3d_package.get("global_gene_freshness_claimed") is not False
    ):
        raise R5UpstreamProvenanceError("R5 upstream scientific-role contract differs")

    gene_sets = {
        "historical_r3": list(map(str, r3c_package["historical_r3_gene_ids"])),
        "r3c_fixed_cohorts": list(map(str, r3c_package["r3c_gene_ids"])),
        "r3d_direct_exposure_registry": list(
            map(str, registry_package["gene_ids"])
        ),
        "r3d_target": list(map(str, r3d_package["target_gene_ids"])),
    }
    expected_sets = {
        "historical_r3": (
            HISTORICAL_R3_GENE_COUNT,
            HISTORICAL_R3_GENE_IDS_SHA256,
        ),
        "r3c_fixed_cohorts": (R3C_FIXED_GENE_COUNT, R3C_FIXED_GENE_IDS_SHA256),
        "r3d_direct_exposure_registry": (
            R3D_REGISTRY_GENE_COUNT,
            R3D_REGISTRY_GENE_IDS_SHA256,
        ),
        "r3d_target": (R3D_TARGET_GENE_COUNT, R3D_TARGET_GENE_IDS_SHA256),
    }
    records: dict[str, dict[str, Any]] = {}
    for name, genes in gene_sets.items():
        count, digest = expected_sets[name]
        if genes != sorted(set(genes)) or len(genes) != count or canonical_sha256(genes) != digest:
            raise R5UpstreamProvenanceError(f"{name} exposure identity differs")
        records[name] = {
            "gene_count": count,
            "gene_ids_sha256": digest,
            "gene_ids": genes,
        }
    intersections = _pairwise_intersections(gene_sets)
    if any(intersections.values()):
        raise R5UpstreamProvenanceError(
            "R5 prior-exposure source registries are not pairwise disjoint"
        )
    r3d_union = sorted(
        set(gene_sets["r3d_direct_exposure_registry"])
        | set(gene_sets["r3d_target"])
    )
    prior_union = sorted(set().union(*(set(values) for values in gene_sets.values())))
    if (
        len(r3d_union) != R3D_REGISTRY_AND_TARGET_UNION_GENE_COUNT
        or canonical_sha256(r3d_union)
        != R3D_REGISTRY_AND_TARGET_UNION_GENE_IDS_SHA256
        or len(prior_union) != R5_PRIOR_EXPOSURE_UNION_GENE_COUNT
        or canonical_sha256(prior_union)
        != R5_PRIOR_EXPOSURE_UNION_GENE_IDS_SHA256
    ):
        raise R5UpstreamProvenanceError("R5 prior-exposure union differs")
    payload = {
        "schema": R5_EXCLUSION_REGISTRY_SCHEMA,
        "derivation": (
            "authenticated exact R3c preflight fixed cohorts plus exact R3d "
            "direct-exposure registry and target; no free gene-list input"
        ),
        "scope": {
            "selection_reads_model_outputs": False,
            "global_gene_freshness_claimed": False,
            "registry_disjointness_scope_only": True,
            "r3d_sources_must_have_zero_validation_row_overlap": True,
            "does_not_cover": R5_EXPOSURE_SCOPE_LIMITATION,
        },
        "source_identity": PINNED_R5_SOURCE_IDENTITY,
        "upstream_packages": {
            "r3c_preflight": {
                key: r3c_package[key]
                for key in (
                    "artifact",
                    "manifest",
                    "generating_commit",
                    "role",
                    "scientific_entry_authorized",
                )
            },
            "r3d_target": {
                key: r3d_package[key]
                for key in (
                    "artifact",
                    "manifest",
                    "generating_commit",
                    "publication_commit",
                    "entry",
                    "attempt",
                    "completion",
                )
            },
            "r3d_registry": {
                key: registry_package[key] for key in ("artifact", "manifest", "scope_id")
            },
        },
        "gene_sets": records,
        "r3c_gene_ids_by_bucket": r3c_package["r3c_gene_ids_by_bucket"],
        "pairwise_intersection_counts": intersections,
        "all_pairwise_intersections_zero": True,
        "r3d_registry_and_target_union_gene_count": len(r3d_union),
        "r3d_registry_and_target_union_gene_ids_sha256": canonical_sha256(r3d_union),
        "prior_exposure_union_gene_count": len(prior_union),
        "prior_exposure_union_gene_ids_sha256": canonical_sha256(prior_union),
    }
    # Do not let a caller mutate loaded package objects or module-level pins through
    # references retained by the derived registry.
    payload = json.loads(json.dumps(payload))
    return payload | {"record_sha256": canonical_sha256(payload)}


def validate_r5_exclusion_registry(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and unpack a serialized R5 v2 prior-exposure registry."""

    if (
        set(record) != set(R5_EXCLUSION_REGISTRY_FIELDS)
        or record.get("schema") != R5_EXCLUSION_REGISTRY_SCHEMA
    ):
        raise R5UpstreamProvenanceError("R5 exclusion-registry schema differs")
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != canonical_sha256(payload):
        raise R5UpstreamProvenanceError("R5 exclusion-registry self-digest differs")
    scope = record.get("scope")
    if (
        record.get("derivation")
        != "authenticated exact R3c preflight fixed cohorts plus exact R3d "
        "direct-exposure registry and target; no free gene-list input"
        or record.get("source_identity") != PINNED_R5_SOURCE_IDENTITY
        or not isinstance(scope, Mapping)
        or set(scope)
        != {
            "selection_reads_model_outputs",
            "global_gene_freshness_claimed",
            "registry_disjointness_scope_only",
            "r3d_sources_must_have_zero_validation_row_overlap",
            "does_not_cover",
        }
        or scope.get("selection_reads_model_outputs") is not False
        or scope.get("global_gene_freshness_claimed") is not False
        or scope.get("registry_disjointness_scope_only") is not True
        or scope.get("r3d_sources_must_have_zero_validation_row_overlap") is not True
        or scope.get("does_not_cover") != R5_EXPOSURE_SCOPE_LIMITATION
    ):
        raise R5UpstreamProvenanceError("R5 exclusion-registry scope differs")
    upstream = record.get("upstream_packages")
    if not isinstance(upstream, Mapping) or set(upstream) != {
        "r3c_preflight",
        "r3d_target",
        "r3d_registry",
    }:
        raise R5UpstreamProvenanceError("R5 exclusion upstream packages are missing")
    r3c_upstream = upstream["r3c_preflight"]
    r3d_upstream = upstream["r3d_target"]
    registry_upstream = upstream["r3d_registry"]
    if (
        not isinstance(r3c_upstream, Mapping)
        or set(r3c_upstream)
        != {
            "artifact",
            "manifest",
            "generating_commit",
            "role",
            "scientific_entry_authorized",
        }
        or r3c_upstream.get("artifact")
        != UPSTREAM_PACKAGE_IDENTITIES["r3c_preflight"]["artifact"]
        or r3c_upstream.get("manifest")
        != UPSTREAM_PACKAGE_IDENTITIES["r3c_preflight"]["manifest"]
        or r3c_upstream.get("generating_commit")
        != R3C_PREFLIGHT_GENERATING_COMMIT
        or r3c_upstream.get("role") != "exclusion_provenance_only"
        or r3c_upstream.get("scientific_entry_authorized") is not False
        or not isinstance(r3d_upstream, Mapping)
        or set(r3d_upstream)
        != {
            "artifact",
            "manifest",
            "generating_commit",
            "publication_commit",
            "entry",
            "attempt",
            "completion",
        }
        or r3d_upstream.get("artifact")
        != UPSTREAM_PACKAGE_IDENTITIES["r3d_target"]["artifact"]
        or r3d_upstream.get("manifest")
        != UPSTREAM_PACKAGE_IDENTITIES["r3d_target"]["manifest"]
        or r3d_upstream.get("generating_commit")
        != R3D_TARGET_GENERATING_COMMIT
        or r3d_upstream.get("publication_commit")
        != R3D_TARGET_PUBLICATION_COMMIT
        or not isinstance(r3d_upstream.get("entry"), Mapping)
        or set(r3d_upstream["entry"]) != set(R3D_ENTRY_CONTRACT_FIELDS)
        or r3d_upstream["entry"].get("schema") != R3D_ENTRY_SCHEMA
        or r3d_upstream["entry"].get("decision_path")
        != "L3H1_reused_component"
        or r3d_upstream["entry"].get("classification") != "SUPPORTED"
        or r3d_upstream["entry"].get("health_ok") is not True
        or r3d_upstream["entry"].get("alpha") != R3D_PATH_A_ALPHA
        or r3d_upstream["entry"].get("intersection_union_p_value")
        != R3D_PATH_A_INTERSECTION_UNION_P_VALUE
        or r3d_upstream["entry"].get("path_b_used_to_gate_entry") is not False
        or r3d_upstream["entry"].get("not_sealed_or_independent_confirmation")
        is not True
        or r3d_upstream["entry"].get("licensed_entry")
        != "registry-disjoint masked-acceptor endpoint support for L3H1 reuse"
        or not isinstance(r3d_upstream["entry"].get("does_not_license"), list)
        or set(map(str, r3d_upstream["entry"].get("does_not_license", ())))
        != set(R3D_ENTRY_DOES_NOT_LICENSE)
        or not isinstance(registry_upstream, Mapping)
        or set(registry_upstream) != {"artifact", "manifest", "scope_id"}
        or registry_upstream.get("artifact")
        != UPSTREAM_PACKAGE_IDENTITIES["r3d_registry"]["artifact"]
        or registry_upstream.get("manifest")
        != UPSTREAM_PACKAGE_IDENTITIES["r3d_registry"]["manifest"]
        or registry_upstream.get("scope_id")
        != "claude_inverse_acceptor_and_clean_alt_head_screen_direct_exposures_v1"
    ):
        raise R5UpstreamProvenanceError("R5 exclusion upstream package identity differs")
    attempt = r3d_upstream.get("attempt")
    completion = r3d_upstream.get("completion")
    if (attempt is None) != (completion is None):
        raise R5UpstreamProvenanceError(
            "R5 exclusion registry has a partial R3d publication receipt pair"
        )
    if attempt is not None and (
        not isinstance(attempt, Mapping)
        or set(attempt) != {"logical_name", "size_bytes", "sha256", "attempt_id"}
        or {
            key: attempt.get(key)
            for key in ("logical_name", "size_bytes", "sha256")
        }
        != UPSTREAM_PACKAGE_IDENTITIES["r3d_target_attempt"]
        or attempt.get("attempt_id") != R3D_TARGET_ATTEMPT_ID
        or not isinstance(completion, Mapping)
        or dict(completion)
        != UPSTREAM_PACKAGE_IDENTITIES["r3d_target_completion"]
    ):
        raise R5UpstreamProvenanceError("R5 exclusion publication receipts differ")
    gene_records = record.get("gene_sets")
    if not isinstance(gene_records, Mapping):
        raise R5UpstreamProvenanceError("R5 exclusion-registry gene sets are missing")
    expected_sets = {
        "historical_r3": (
            HISTORICAL_R3_GENE_COUNT,
            HISTORICAL_R3_GENE_IDS_SHA256,
        ),
        "r3c_fixed_cohorts": (R3C_FIXED_GENE_COUNT, R3C_FIXED_GENE_IDS_SHA256),
        "r3d_direct_exposure_registry": (
            R3D_REGISTRY_GENE_COUNT,
            R3D_REGISTRY_GENE_IDS_SHA256,
        ),
        "r3d_target": (R3D_TARGET_GENE_COUNT, R3D_TARGET_GENE_IDS_SHA256),
    }
    if set(gene_records) != set(expected_sets):
        raise R5UpstreamProvenanceError("R5 exclusion-registry source set differs")
    gene_sets: dict[str, list[str]] = {}
    for name, (count, digest) in expected_sets.items():
        item = gene_records[name]
        if not isinstance(item, Mapping) or set(item) != {
            "gene_count",
            "gene_ids_sha256",
            "gene_ids",
        }:
            raise R5UpstreamProvenanceError(f"R5 exclusion set {name} is malformed")
        genes = _sorted_unique_genes(
            item.get("gene_ids"),
            label=name,
            expected_count=count,
            expected_sha256=digest,
        )
        if item.get("gene_count") != count or item.get("gene_ids_sha256") != digest:
            raise R5UpstreamProvenanceError(f"R5 exclusion set {name} identity differs")
        gene_sets[name] = genes
    by_bucket = record.get("r3c_gene_ids_by_bucket")
    if not isinstance(by_bucket, Mapping) or set(by_bucket) != {"0", "1"}:
        raise R5UpstreamProvenanceError("R5 exclusion R3c buckets differ")
    normalized_buckets = {
        bucket: list(map(str, by_bucket[bucket])) for bucket in ("0", "1")
    }
    if (
        any(
            genes != sorted(set(genes))
            or len(genes) != R3C_FIXED_BUCKET_IDENTITIES[bucket]["gene_count"]
            or canonical_sha256(genes)
            != R3C_FIXED_BUCKET_IDENTITIES[bucket]["gene_ids_sha256"]
            for bucket, genes in normalized_buckets.items()
        )
        or sorted(set(normalized_buckets["0"]) | set(normalized_buckets["1"]))
        != gene_sets["r3c_fixed_cohorts"]
    ):
        raise R5UpstreamProvenanceError("R5 exclusion R3c bucket union differs")
    intersections = _pairwise_intersections(gene_sets)
    recorded_intersections = record.get("pairwise_intersection_counts")
    if (
        not isinstance(recorded_intersections, Mapping)
        or set(recorded_intersections) != set(intersections)
        or any(
            type(value) is not int or value != 0
            for value in recorded_intersections.values()
        )
        or dict(recorded_intersections) != intersections
        or any(intersections.values())
        or record.get("all_pairwise_intersections_zero") is not True
    ):
        raise R5UpstreamProvenanceError("R5 exclusion sources are not disjoint")
    r3d_union = sorted(
        set(gene_sets["r3d_direct_exposure_registry"])
        | set(gene_sets["r3d_target"])
    )
    prior_union = sorted(set().union(*(set(values) for values in gene_sets.values())))
    if (
        len(r3d_union) != R3D_REGISTRY_AND_TARGET_UNION_GENE_COUNT
        or canonical_sha256(r3d_union)
        != R3D_REGISTRY_AND_TARGET_UNION_GENE_IDS_SHA256
        or record.get("r3d_registry_and_target_union_gene_count") != len(r3d_union)
        or record.get("r3d_registry_and_target_union_gene_ids_sha256")
        != canonical_sha256(r3d_union)
        or len(prior_union) != R5_PRIOR_EXPOSURE_UNION_GENE_COUNT
        or canonical_sha256(prior_union)
        != R5_PRIOR_EXPOSURE_UNION_GENE_IDS_SHA256
        or record.get("prior_exposure_union_gene_count") != len(prior_union)
        or record.get("prior_exposure_union_gene_ids_sha256")
        != canonical_sha256(prior_union)
    ):
        raise R5UpstreamProvenanceError("R5 exclusion union differs")
    return {
        "gene_sets": gene_sets,
        "historical_r3_gene_ids": gene_sets["historical_r3"],
        "r3c_gene_ids": gene_sets["r3c_fixed_cohorts"],
        "r3d_registry_gene_ids": gene_sets["r3d_direct_exposure_registry"],
        "r3d_target_gene_ids": gene_sets["r3d_target"],
        "prior_exposure_gene_ids": prior_union,
        "counts": {
            name: len(genes) for name, genes in gene_sets.items()
        }
        | {"prior_exposure_union": len(prior_union)},
        "prior_exposure_union_gene_ids_sha256": canonical_sha256(prior_union),
        "r3d_registry_and_target_union_gene_ids_sha256": canonical_sha256(r3d_union),
        "all_pairwise_intersections_zero": True,
        "record_sha256": record["record_sha256"],
    }


__all__ = [
    "HISTORICAL_R3_GENE_COUNT",
    "HISTORICAL_R3_GENE_IDS_SHA256",
    "PINNED_CHECKPOINT_AGGREGATE_SHA256",
    "PINNED_R5_REPOSITORY_RUNTIME_IDENTITY",
    "PINNED_R5_SOURCE_IDENTITY",
    "R3C_FIXED_GENE_COUNT",
    "R3C_FIXED_GENE_IDS_SHA256",
    "R3C_FIXED_BUCKET_IDENTITIES",
    "R3C_PREFLIGHT_ARTIFACT_SHA256",
    "R3C_PREFLIGHT_MANIFEST_SHA256",
    "R3D_ENTRY_SCHEMA",
    "R3D_PATH_A_ALPHA",
    "R3D_PATH_A_INTERSECTION_UNION_P_VALUE",
    "R3D_REGISTRY_AND_TARGET_UNION_GENE_COUNT",
    "R3D_REGISTRY_AND_TARGET_UNION_GENE_IDS_SHA256",
    "R3D_REGISTRY_ARTIFACT_SHA256",
    "R3D_REGISTRY_GENE_COUNT",
    "R3D_REGISTRY_GENE_IDS_SHA256",
    "R3D_REGISTRY_MANIFEST_SHA256",
    "R3D_TARGET_ARTIFACT_SHA256",
    "R3D_TARGET_ATTEMPT_SHA256",
    "R3D_TARGET_COMPLETION_SHA256",
    "R3D_TARGET_GENE_COUNT",
    "R3D_TARGET_GENE_IDS_SHA256",
    "R3D_TARGET_IDENTITY_SHA256",
    "R3D_TARGET_MANIFEST_SHA256",
    "R3D_TARGET_ATTEMPT_ID",
    "R3D_TARGET_GENERATING_COMMIT",
    "R3D_TARGET_PUBLICATION_COMMIT",
    "R5_ENTRY_SCHEMA",
    "R5_EXCLUSION_REGISTRY_SCHEMA",
    "R5_PRIOR_EXPOSURE_UNION_GENE_COUNT",
    "R5_PRIOR_EXPOSURE_UNION_GENE_IDS_SHA256",
    "R5UpstreamProvenanceError",
    "UPSTREAM_PACKAGE_IDENTITIES",
    "canonical_sha256",
    "derive_r5_exclusion_registry",
    "load_r3c_exclusion_package",
    "load_r3d_entry_package",
    "sha256_file",
    "validate_r3d_target_record",
    "validate_r5_exclusion_registry",
]
