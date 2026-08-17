"""Outcome-blind cohort construction shared by the R5 acceptor mechanism stages.

The three conditional mechanism assays must operate on the same rows.  This module keeps
selection independent of model outputs and makes the exact one-row-per-gene reservations,
adjacent control, sequence constructions, and hashes reusable by R5a, R5b, and R5c.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from gfmcd.splicing.coupling import (
    MOTIF_CLEAN_DONOR_ALT,
    CouplingExample,
    planted_donor_motifs,
    substitute,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout


R5_SALT = "r5-acceptor-mechanism-v1"
R5_ADJACENT_SALT = "r5-adjacent-bank-v2"
R5_VALIDATION_CHROMOSOMES = ("chr11", "chr16", "chr19")
R5_DONOR_PSEUDO_CALIPER = 40
R5_ROWS_PER_BUCKET = 128
R5_EXPECTED_SOURCE_IDENTITY = {
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
}
R5_EXPECTED_CENSUS_IDENTITY = {
    "schema": "gfmcd.r5.acceptor_mechanism_cohort/2",
    "input_rows": 6064,
    "historical_r3_gene_count": 645,
    "historical_r3_gene_ids_sha256": (
        "0964c06136fd9a8fa9251ad13a2ca4d45e3f0c8cea45323dbdc1fe2092b8fda0"
    ),
    "r3c_gene_count": 588,
    "r3c_gene_ids_sha256": (
        "67c745eef7d223d1b8b2769c87f5d818c955a042ae4a303459a473c1ca528a96"
    ),
    "combined_excluded_gene_count": 1233,
    "combined_excluded_gene_ids_sha256": (
        "cc8c23308bfe0c39b70c77cd07d94945838cf03da513c8aa9550542425d0093f"
    ),
    "exclusions": {
        "excluded_gene": 5648,
        "pseudo_donor_outside_40nt_caliper": 50,
    },
    "eligible_rows": 366,
    "eligible_genes": 272,
    "available_genes_by_bucket": {"0": 142, "1": 130},
    "selected_by_bucket": {
        "0": {
            "rows": 128,
            "genes": 128,
            "both_strands": True,
            "gene_ids_sha256": (
                "af644dd66d8963a714f07a236fae6302a789044aeb1e52c0bb844e731328abe4"
            ),
            "ordered_rows_sha256": (
                "572289f8d2e8e6406e93596994b6bb5d3df04cd6571ff95bea85211ba4f358a9"
            ),
            "construction_sha256": (
                "3143a893894de702476145589652f3157001218ec24c1742f2b1cedd0eebe0fd"
            ),
        },
        "1": {
            "rows": 128,
            "genes": 128,
            "both_strands": True,
            "gene_ids_sha256": (
                "e6c3099298431c99b8afde686215b3287f3e0f63eedfe3b650db4d48fc13c84c"
            ),
            "ordered_rows_sha256": (
                "935d62284518b0ad72b398b2d7f41cb034015d63bb6617ca7e43a5b534ddbf95"
            ),
            "construction_sha256": (
                "7de99171bd678dada527b9411c350166cdc3a512020b4bdd18f1e28af5c08312"
            ),
        },
    },
}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible object without whitespace or key-order ambiguity."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def r5_source_identity(gtf: str | Path, fasta: str | Path) -> dict[str, Any]:
    """Hash the exact annotation and reference bytes used to construct an R5 census."""

    records: dict[str, Any] = {}
    for key, value in (("gtf", gtf), ("fasta", fasta)):
        path = Path(value).expanduser().resolve(strict=True)
        records[key] = {
            "label": R5_EXPECTED_SOURCE_IDENTITY[key]["label"],
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return records


def validate_expected_r5_source_identity(
    gtf: str | Path, fasta: str | Path
) -> dict[str, Any]:
    """Reject a source-compatible-looking census built from different source bytes."""

    observed = r5_source_identity(gtf, fasta)
    if observed != R5_EXPECTED_SOURCE_IDENTITY:
        raise RuntimeError(
            "R5 source identity drifted; the historical R3/R5 cohort requires the "
            f"pinned comprehensive GTF and hg38 bytes: observed={observed}, "
            f"expected={R5_EXPECTED_SOURCE_IDENTITY}"
        )
    return observed


def r5_gene_bucket(gene_id: str) -> int:
    digest = hashlib.sha256(f"{R5_SALT}|bucket|{gene_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2


def r5_row_priority(example_id: str) -> tuple[str, str]:
    return (_sha256_text(f"{R5_SALT}|row|{example_id}"), str(example_id))


def r5_gene_priority(gene_id: str) -> tuple[str, str]:
    return (_sha256_text(f"{R5_SALT}|order|{gene_id}"), str(gene_id))


def _window(start: int) -> tuple[int, int, int, int]:
    return (start, start + 1, start + 2, start + 3)


def deterministic_adjacent_window(
    *,
    example_id: str,
    sequence: str,
    paired_donor: int,
    pseudo_donor: int,
    acceptor_g: int,
) -> tuple[int, int, int, int] | None:
    """Choose the frozen four-token adjacent control, or ``None`` if neither is valid."""

    side_bit = hashlib.sha256(
        f"{R5_ADJACENT_SALT}|{example_id}".encode()
    ).digest()[-1] & 1
    protected = (
        set(_window(paired_donor - 1))
        | set(_window(pseudo_donor - 1))
        | {int(acceptor_g)}
    )
    # Search a frozen local bank in increasing four-token shells.  The original two-window
    # design (shell 1 only) left just 61/58 genes after the predeclared exclusions.  The
    # eight-shell bank was chosen model-free before any R5 outcome and preserves the nearest
    # admissible control while recovering 142/130 genes on the validation census.
    for shell in range(1, 9):
        upstream = _window(paired_donor - (4 * shell + 1))
        downstream = _window(paired_donor + (4 * shell - 1))
        order = (
            (upstream, downstream)
            if side_bit == 0
            else (downstream, upstream)
        )
        for candidate in order:
            start, *_, end = candidate
            # The one-base-padded span must itself be in bounds because motif hygiene is
            # evaluated across both edges of the four-token control.
            if start < 1 or end + 1 >= len(sequence):
                continue
            if set(candidate) & protected:
                continue
            padded = sequence[start - 1 : end + 2]
            if "GT" in padded or "AG" in padded:
                continue
            return candidate
    return None


@dataclass(frozen=True)
class R5CohortRow:
    """One fully constructed, outcome-blind R5 row."""

    example: CouplingExample
    pseudo_donor: int
    adjacent_window: tuple[int, int, int, int]
    donor_alt: str = MOTIF_CLEAN_DONOR_ALT

    @property
    def acceptor_g(self) -> int:
        return acceptor_readout(self.example)

    @property
    def paired_window(self) -> tuple[int, int, int, int]:
        return _window(int(self.example.donor_pos) - 1)

    @property
    def pseudo_window(self) -> tuple[int, int, int, int]:
        return _window(int(self.pseudo_donor) - 1)

    @property
    def clean_sequence(self) -> str:
        return self.example.sequence

    @property
    def paired_corrupt_sequence(self) -> str:
        return substitute(
            self.example.sequence, int(self.example.donor_pos), self.donor_alt
        )

    @property
    def pseudo_corrupt_sequence(self) -> str:
        return substitute(self.example.sequence, int(self.pseudo_donor), self.donor_alt)

    def identity_record(self) -> dict[str, Any]:
        return {
            "example_id": str(self.example.example_id),
            "gene_id": str(self.example.gene_id),
            "chrom": str(self.example.chrom),
            "strand": str(self.example.strand),
            "paired_donor": int(self.example.donor_pos),
            "pseudo_donor": int(self.pseudo_donor),
            "acceptor_g": int(self.acceptor_g),
            "paired_window": list(self.paired_window),
            "pseudo_window": list(self.pseudo_window),
            "adjacent_window": list(self.adjacent_window),
            "donor_alt": self.donor_alt,
        }

    def construction_record(self) -> dict[str, Any]:
        return self.identity_record() | {
            "clean_sequence_sha256": _sha256_text(self.clean_sequence),
            "paired_corrupt_sequence_sha256": _sha256_text(
                self.paired_corrupt_sequence
            ),
            "pseudo_corrupt_sequence_sha256": _sha256_text(
                self.pseudo_corrupt_sequence
            ),
        }


def evaluate_r5_row(
    example: CouplingExample,
    pseudo_donor: int | None,
    *,
    excluded_gene_ids: set[str] | frozenset[str],
    donor_alt: str = MOTIF_CLEAN_DONOR_ALT,
    allowed_chromosomes: Sequence[str] = R5_VALIDATION_CHROMOSOMES,
    donor_pseudo_caliper: int = R5_DONOR_PSEUDO_CALIPER,
) -> tuple[R5CohortRow | None, str | None]:
    """Evaluate every outcome-blind eligibility rule and return one failure reason."""

    if str(example.chrom) not in set(map(str, allowed_chromosomes)):
        return None, "chromosome_not_allowed"
    if str(example.gene_id) in excluded_gene_ids:
        return None, "excluded_gene"
    if example.next_acceptor_pos is None:
        return None, "missing_intron_mate_acceptor"
    donor = int(example.donor_pos)
    q = acceptor_readout(example)
    if donor < 1 or donor + 2 >= len(example.sequence):
        return None, "paired_donor_out_of_bounds"
    if q < 1 or q >= len(example.sequence):
        return None, "acceptor_readout_out_of_bounds"
    if example.sequence[donor : donor + 2] != "GT":
        return None, "paired_donor_not_GT"
    if example.sequence[q - 1 : q + 1] != "AG":
        return None, "intron_mate_acceptor_not_AG"
    if pseudo_donor is None:
        return None, "missing_pseudo_donor"
    pseudo = int(pseudo_donor)
    if pseudo == donor:
        return None, "pseudo_donor_equals_paired_donor"
    if pseudo < 1 or pseudo + 2 >= len(example.sequence):
        return None, "pseudo_donor_out_of_bounds"
    if abs(pseudo - donor) > int(donor_pseudo_caliper):
        return None, "pseudo_donor_outside_40nt_caliper"
    if example.sequence[pseudo : pseudo + 2] != "GT":
        return None, "pseudo_donor_not_GT"
    if planted_donor_motifs(example.sequence, donor, donor_alt):
        return None, "paired_donor_edit_plants_splice_motif"
    if planted_donor_motifs(example.sequence, pseudo, donor_alt):
        return None, "pseudo_donor_edit_plants_splice_motif"
    adjacent = deterministic_adjacent_window(
        example_id=str(example.example_id),
        sequence=example.sequence,
        paired_donor=donor,
        pseudo_donor=pseudo,
        acceptor_g=q,
    )
    if adjacent is None:
        return None, "missing_admissible_adjacent_window"
    return (
        R5CohortRow(
            example=example,
            pseudo_donor=pseudo,
            adjacent_window=adjacent,
            donor_alt=donor_alt,
        ),
        None,
    )


def _pseudo_for(
    pseudo_donors: Mapping[str, int | None], example: CouplingExample
) -> int | None:
    value = pseudo_donors.get(str(example.example_id))
    return None if value is None else int(value)


def build_r5_reservations(
    examples: Sequence[CouplingExample],
    pseudo_donors: Mapping[str, int | None],
    *,
    historical_r3_gene_ids: Sequence[str],
    r3c_gene_ids: Sequence[str],
    rows_per_bucket: int = R5_ROWS_PER_BUCKET,
    require_capacity: bool = True,
) -> tuple[dict[int, tuple[R5CohortRow, ...]], dict[str, Any]]:
    """Build the two deterministic one-row-per-gene R5 reservations and their audit."""

    if isinstance(rows_per_bucket, bool) or not isinstance(rows_per_bucket, int):
        raise TypeError("rows_per_bucket must be an integer")
    if rows_per_bucket < 1:
        raise ValueError("rows_per_bucket must be positive")
    historical = sorted(set(map(str, historical_r3_gene_ids)))
    r3c = sorted(set(map(str, r3c_gene_ids)))
    excluded = frozenset(historical) | frozenset(r3c)
    exclusions: Counter[str] = Counter()
    eligible_by_gene: dict[str, list[R5CohortRow]] = {}
    for example in examples:
        row, reason = evaluate_r5_row(
            example,
            _pseudo_for(pseudo_donors, example),
            excluded_gene_ids=excluded,
        )
        if row is None:
            exclusions[str(reason)] += 1
            continue
        eligible_by_gene.setdefault(str(example.gene_id), []).append(row)

    chosen_by_gene = {
        gene: min(rows, key=lambda row: r5_row_priority(row.example.example_id))
        for gene, rows in eligible_by_gene.items()
    }
    available: dict[int, list[R5CohortRow]] = {0: [], 1: []}
    for gene, row in chosen_by_gene.items():
        available[r5_gene_bucket(gene)].append(row)
    for bucket in (0, 1):
        available[bucket].sort(key=lambda row: r5_gene_priority(row.example.gene_id))

    if require_capacity:
        short = {
            bucket: len(rows)
            for bucket, rows in available.items()
            if len(rows) < rows_per_bucket
        }
        if short:
            raise RuntimeError(
                "R5 one-row-per-gene reservation capacity is insufficient: "
                + ", ".join(
                    f"bucket {bucket} has {count}, requires {rows_per_bucket}"
                    for bucket, count in sorted(short.items())
                )
            )

    selected = {
        bucket: tuple(available[bucket][:rows_per_bucket]) for bucket in (0, 1)
    }
    selected_audit: dict[str, Any] = {}
    for bucket in (0, 1):
        rows = selected[bucket]
        identity = [row.identity_record() for row in rows]
        constructions = [row.construction_record() for row in rows]
        genes = [str(row.example.gene_id) for row in rows]
        selected_audit[str(bucket)] = {
            "rows": len(rows),
            "genes": len(set(genes)),
            "both_strands": {str(row.example.strand) for row in rows} == {"+", "-"},
            "gene_ids": genes,
            "gene_ids_sha256": canonical_sha256(genes),
            "ordered_rows_sha256": canonical_sha256(identity),
            "construction_sha256": canonical_sha256(constructions),
        }
    audit = {
        "schema": "gfmcd.r5.acceptor_mechanism_cohort/2",
        "selection_reads_model_outputs": False,
        "allowed_chromosomes": list(R5_VALIDATION_CHROMOSOMES),
        "sealed_chr8_chr9_opened": False,
        "donor_pseudo_caliper": R5_DONOR_PSEUDO_CALIPER,
        "adjacent_window_policy": {
            "salt": R5_ADJACENT_SALT,
            "shells": 8,
            "tokens_per_window": 4,
            "maximum_absolute_offset_from_donor": 34,
            "selection": "nearest admissible shell; hash-fixed side order within shell",
        },
        "rows_per_bucket": rows_per_bucket,
        "input_rows": len(examples),
        "historical_r3_gene_count": len(historical),
        "historical_r3_gene_ids_sha256": canonical_sha256(historical),
        "r3c_gene_count": len(r3c),
        "r3c_gene_ids_sha256": canonical_sha256(r3c),
        "combined_excluded_gene_count": len(excluded),
        "combined_excluded_gene_ids_sha256": canonical_sha256(sorted(excluded)),
        "exclusions": dict(sorted(exclusions.items())),
        "eligible_rows": sum(len(rows) for rows in eligible_by_gene.values()),
        "eligible_genes": len(chosen_by_gene),
        "available_genes_by_bucket": {
            str(bucket): len(available[bucket]) for bucket in (0, 1)
        },
        "selected_by_bucket": selected_audit,
    }
    return selected, audit


def validate_r5_reservation_audit(
    audit: Mapping[str, Any], *, rows_per_bucket: int = R5_ROWS_PER_BUCKET
) -> None:
    """Fail closed on the cohort properties needed by all three R5 stages."""

    if audit.get("schema") != "gfmcd.r5.acceptor_mechanism_cohort/2":
        raise ValueError("R5 cohort audit schema differs")
    if audit.get("selection_reads_model_outputs") is not False:
        raise ValueError("R5 cohort selection must be outcome-blind")
    if audit.get("sealed_chr8_chr9_opened") is not False:
        raise ValueError("R5 cohort must not open chr8/chr9")
    if audit.get("allowed_chromosomes") != list(R5_VALIDATION_CHROMOSOMES):
        raise ValueError("R5 cohort chromosome registry differs")
    if audit.get("rows_per_bucket") != rows_per_bucket:
        raise ValueError("R5 cohort reservation size differs")
    if audit.get("adjacent_window_policy") != {
        "salt": R5_ADJACENT_SALT,
        "shells": 8,
        "tokens_per_window": 4,
        "maximum_absolute_offset_from_donor": 34,
        "selection": "nearest admissible shell; hash-fixed side order within shell",
    }:
        raise ValueError("R5 adjacent-window bank differs")
    selected = audit.get("selected_by_bucket")
    if not isinstance(selected, Mapping) or set(selected) != {"0", "1"}:
        raise ValueError("R5 cohort lacks both fixed reservations")
    for bucket in ("0", "1"):
        record = selected[bucket]
        if (
            record.get("rows") != rows_per_bucket
            or record.get("genes") != rows_per_bucket
            or record.get("both_strands") is not True
        ):
            raise ValueError(
                f"R5 bucket {bucket} must contain {rows_per_bucket} genes and both strands"
            )
        for key in (
            "gene_ids_sha256",
            "ordered_rows_sha256",
            "construction_sha256",
        ):
            value = record.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"R5 bucket {bucket} lacks {key}")


def r5_census_identity(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the literal model-free identity that every R5 runner must bind."""

    selected = audit["selected_by_bucket"]
    return {
        "schema": audit["schema"],
        "input_rows": audit["input_rows"],
        "historical_r3_gene_count": audit["historical_r3_gene_count"],
        "historical_r3_gene_ids_sha256": audit[
            "historical_r3_gene_ids_sha256"
        ],
        "r3c_gene_count": audit["r3c_gene_count"],
        "r3c_gene_ids_sha256": audit["r3c_gene_ids_sha256"],
        "combined_excluded_gene_count": audit["combined_excluded_gene_count"],
        "combined_excluded_gene_ids_sha256": audit[
            "combined_excluded_gene_ids_sha256"
        ],
        "exclusions": audit["exclusions"],
        "eligible_rows": audit["eligible_rows"],
        "eligible_genes": audit["eligible_genes"],
        "available_genes_by_bucket": audit["available_genes_by_bucket"],
        "selected_by_bucket": {
            bucket: {
                key: selected[bucket][key]
                for key in (
                    "rows",
                    "genes",
                    "both_strands",
                    "gene_ids_sha256",
                    "ordered_rows_sha256",
                    "construction_sha256",
                )
            }
            for bucket in ("0", "1")
        },
    }


def validate_expected_r5_census_identity(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Abort if the full validation census drifts from the pre-outcome R5 identity."""

    observed = r5_census_identity(audit)
    if observed != R5_EXPECTED_CENSUS_IDENTITY:
        raise RuntimeError(
            "R5 model-free census identity drifted: "
            f"observed={observed}, expected={R5_EXPECTED_CENSUS_IDENTITY}"
        )
    return observed


__all__ = [
    "R5_ADJACENT_SALT",
    "R5_DONOR_PSEUDO_CALIPER",
    "R5_EXPECTED_CENSUS_IDENTITY",
    "R5_EXPECTED_SOURCE_IDENTITY",
    "R5_ROWS_PER_BUCKET",
    "R5_SALT",
    "R5_VALIDATION_CHROMOSOMES",
    "R5CohortRow",
    "build_r5_reservations",
    "canonical_sha256",
    "deterministic_adjacent_window",
    "evaluate_r5_row",
    "r5_gene_bucket",
    "r5_census_identity",
    "r5_gene_priority",
    "r5_row_priority",
    "r5_source_identity",
    "validate_expected_r5_source_identity",
    "validate_r5_reservation_audit",
    "validate_expected_r5_census_identity",
]
