"""Frozen external-input identities for positional-boundary coherence.

These hashes are execution identities, not scientific thresholds.  Production
entrypoints validate them before an expensive parse or model load so a file at
the expected path cannot silently substitute another checkpoint/reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from gfmcd.provenance import sha256_file


OFFICIAL_SPLICEBERT_WEIGHT_SHA256 = (
    "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d"
)

FROZEN_BIOLOGY_SHA256: dict[str, str] = {
    "fasta": "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51",
    "fai": "3b425de206296a5c8053023fa5ca61da43cfe78c1737c12e58c83367c7e83c21",
    "basic_gtf": "5b9ad9aa71f6ab5d55123c51f74723644a895bc0cbaa9cee8fbc11bb9dec0bf7",
    "comprehensive_gtf": (
        "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453"
    ),
    "maxent_donor_table": (
        "c64fa6aea3d8b71d6af69f5e3ece8f84de06f5450f6d4ba482fbbff6afcf8b5b"
    ),
    "maxent_acceptor_table": (
        "9e8a74dc795ae5c9c5b611d4343a1a5a70fff863956f4e00ae23a85f21beb3ae"
    ),
}

FROZEN_CONSTRUCTION_CONFIG_SHA256: dict[str, str] = {
    "source_spec": "c3a145aa70d52eeba8276456822ba6f8aed5acd7e65ae62f67e14d7aec6c40d4",
    "frozen_variance": (
        "aaaf0b554ba1d11dc7c03485f1db583b886f13d786bc0508b729d09d0645584d"
    ),
}


def require_sha256(path: str | Path, expected: str, *, label: str) -> Path:
    """Resolve *path* and fail unless its bytes match the frozen digest."""

    resolved = Path(path).expanduser().resolve(strict=True)
    observed = sha256_file(resolved)
    if observed != expected:
        raise RuntimeError(
            f"{label} sha256 mismatch: expected {expected}, observed {observed}"
        )
    return resolved


def validate_official_checkpoint(model: str | Path) -> Path:
    """Return the official weight file only after exact byte validation."""

    root = Path(model).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("official SpliceBERT checkpoint must be a directory")
    if (root / "model.safetensors").exists():
        raise RuntimeError(
            "official SpliceBERT checkpoint must not contain an alternate "
            "model.safetensors weight source"
        )
    return require_sha256(
        root / "pytorch_model.bin",
        OFFICIAL_SPLICEBERT_WEIGHT_SHA256,
        label="SpliceBERT.1024nt pytorch_model.bin",
    )


def validate_frozen_biology_inputs(
    paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Validate the complete hg38/GENCODE-v50/MaxEnt execution inventory."""

    if set(paths) != set(FROZEN_BIOLOGY_SHA256):
        missing = sorted(set(FROZEN_BIOLOGY_SHA256) - set(paths))
        extra = sorted(set(paths) - set(FROZEN_BIOLOGY_SHA256))
        raise RuntimeError(
            f"frozen biology input inventory mismatch: missing={missing}, extra={extra}"
        )
    return {
        label: require_sha256(paths[label], digest, label=label)
        for label, digest in FROZEN_BIOLOGY_SHA256.items()
    }


def validate_candidate_input_pins(
    candidate_package: Mapping[str, Any],
) -> dict[str, Path]:
    """Recheck the candidate package's external-input records against frozen bytes.

    The candidate builder remains a general recorder.  This is the execution
    gate: it accepts an already-produced census only when its recorded files
    still exist, its records are internally accurate, and their bytes match the
    pre-outcome identities.
    """

    pins = candidate_package.get("input_pins")
    if not isinstance(pins, Mapping):
        raise RuntimeError("candidate package lacks input_pins")
    maxent = pins.get("maxent")
    if not isinstance(maxent, Mapping):
        raise RuntimeError("candidate package lacks MaxEnt input pins")
    records = {
        "fasta": pins.get("fasta"),
        "fai": pins.get("fai"),
        "basic_gtf": pins.get("basic_gtf"),
        "comprehensive_gtf": pins.get("comprehensive_gtf"),
        "maxent_donor_table": maxent.get("donor_table"),
        "maxent_acceptor_table": maxent.get("acceptor_table"),
    }
    paths: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise RuntimeError(f"candidate package lacks a valid {label} record")
        expected = FROZEN_BIOLOGY_SHA256[label]
        if str(record.get("sha256", "")) != expected:
            raise RuntimeError(
                f"candidate package {label} pin is not the frozen sha256"
            )
        path = require_sha256(record["path"], expected, label=label)
        if type(record.get("size_bytes")) is not int or int(
            record["size_bytes"]
        ) != path.stat().st_size:
            raise RuntimeError(f"candidate package {label} size record is stale")
        paths[label] = path
    basic = records["basic_gtf"]
    comprehensive = records["comprehensive_gtf"]
    if (
        basic.get("release") != "GENCODE_v50"  # type: ignore[union-attr]
        or basic.get("coordinate_convention")  # type: ignore[union-attr]
        != "GTF_1_based_closed_parsed_as_0_based_half_open"
        or basic.get("phase_convention")  # type: ignore[union-attr]
        != "GTF_frame_to_conventional_intron_phase"
        or comprehensive.get("release") != "GENCODE_v50"  # type: ignore[union-attr]
        or comprehensive.get("role")  # type: ignore[union-attr]
        != "local_coding_strand_and_phase_safety_audit"
        or maxent.get("equations") != "Yeo_Burge_MaxEntScan_9mer_23mer_local_v1"
    ):
        raise RuntimeError("candidate package annotation/MaxEnt convention drift")
    return paths


def validate_frozen_construction_configs(
    paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Validate outcome-blind freshness and variance configuration bytes."""

    if set(paths) != set(FROZEN_CONSTRUCTION_CONFIG_SHA256):
        raise RuntimeError("frozen construction configuration inventory mismatch")
    return {
        label: require_sha256(paths[label], digest, label=label)
        for label, digest in FROZEN_CONSTRUCTION_CONFIG_SHA256.items()
    }


__all__ = [
    "FROZEN_BIOLOGY_SHA256",
    "FROZEN_CONSTRUCTION_CONFIG_SHA256",
    "OFFICIAL_SPLICEBERT_WEIGHT_SHA256",
    "require_sha256",
    "validate_candidate_input_pins",
    "validate_frozen_construction_configs",
    "validate_frozen_biology_inputs",
    "validate_official_checkpoint",
]
