"""Model-free construction and analysis primitives for BG1 boundary grammar.

The supplied sequence is always in transcript orientation (5' to 3'), including for a
minus-strand gene.  ``acceptor_pos`` is the first exonic nucleotide: the protected acceptor
``AG`` is therefore ``sequence[acceptor_pos - 2:acceptor_pos]`` and biological ``+1`` is
``sequence[acceptor_pos]``.  Keeping that convention here prevents runners from silently
mixing genomic and model-array coordinates.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


DNA_BASES = ("A", "C", "G", "T")
SPLICE_MOTIFS = frozenset(("AG", "GT", "GC"))

ROLE_NAMESPACE = "splicebert-boundary-grammar-bg1-v1"
BG1_NAMESPACE = ROLE_NAMESPACE
ROLE_WEIGHTS = (("behavior", 3), ("route", 2), ("reserve", 1))
SESOI = 0.05  # raw donor-logit units
CHECKPOINT_SHA256 = "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d"
GTF_SHA256 = "5b9ad9aa71f6ab5d55123c51f74723644a895bc0cbaa9cee8fbc11bb9dec0bf7"
FASTA_SHA256 = "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51"
MODEL_FILE_SHA256 = {
    "config.json": "51ea51e42951f18cc1d3455d7111fe0a41cebad476ba8314afc161a0e956303e",
    "pytorch_model.bin": CHECKPOINT_SHA256,
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer.json": "b380c8c93ba20313c0c0239b8c70f0c67253028ca76cfda4f55b46049abe4fa3",
    "tokenizer_config.json": "a8f9a3828553689c1cb215689fc82af7551c1bb6ba642606612e834fda0c607c",
    "vocab.txt": "ba67bacb61d3de7d57e11e04ba59020a2cc67dbedbd9ae65715117b9fd4f30bc",
}
ANALYSIS_SEED = 20260809
BG1_BOOTSTRAP_DRAWS = 4999
BG1_PERMUTATION_DRAWS = 9999
BG2_BOOTSTRAP_DRAWS = 2000
PREFLIGHT_BOOTSTRAP_DRAWS = 1999
PREFLIGHT_PERMUTATION_DRAWS = 1999
PREFLIGHT_CALIBRATION_REPLICATES = 200
MIN_TRAIN_BEHAVIOR_GENES = 384
MIN_TRAIN_ROUTE_GENES = 24
PREFLIGHT_HEALTH_GENES = 96
MIN_BG1_TARGET_GENES = 128
MIN_BG1_TARGET_GENES_PER_STRAND = 32
MIN_OUTER_TARGET_GENES = 80
MIN_OUTER_TARGET_GENES_PER_STRAND = 20
MIN_BG2_TARGET_GENES = 96
MIN_BG2_TARGET_GENES_PER_STRAND = 24

TARGET_PANEL_ID = "bg1_target_m3_by_p1"
MINUS3_PANEL_ID = "bg1_inner_m3"
PLUS1_PANEL_ID = "bg1_inner_p1"
MINUS4_PANEL_ID = "bg1_diagnostic_m4"
PLUS2_PANEL_ID = "bg1_diagnostic_p2"
TARGET_UPSTREAM_ALLELES = DNA_BASES
TARGET_DOWNSTREAM_ALLELES = ("A", "G")
MINUS3_ALLELES = ("A", "T")
PLUS1_ALLELES = ("A", "G")
MINUS4_ALLELES = ("A", "T")
PLUS2_ALLELES = ("A", "G")

TARGET_MINUS3_T_MINUS_A_ID = "m3_T_minus_A"
TARGET_PLUS1_G_MINUS_A_ID = "p1_G_minus_A"
TARGET_INTERACTION_ID = "m3_by_p1_interaction"
MINUS3_T_MINUS_A_SINGLE_ID = "m3_T_minus_A_single"
PLUS1_G_MINUS_A_SINGLE_ID = "p1_G_minus_A_single"
MINUS4_T_MINUS_A_ID = "m4_T_minus_A"
PLUS2_G_MINUS_A_ID = "p2_G_minus_A"


def _position_tag(position: int) -> str:
    return f"m{-position}" if position < 0 else f"p{position}"


@dataclass(frozen=True)
class PanelCell:
    """One declared factorial cell; positions use biological splice-site numbering."""

    cell_id: str
    alleles: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id must be non-empty")
        positions = [position for position, _base in self.alleles]
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("cell allele positions must be non-empty and unique")
        if any(position == 0 for position in positions):
            raise ValueError("biological splice-site position 0 does not exist")
        if any(base not in DNA_BASES for _position, base in self.alleles):
            raise ValueError("cell alleles must be uppercase DNA bases")


@dataclass(frozen=True)
class PanelSpec:
    """A complete, ordered intervention panel."""

    panel_id: str
    positions: tuple[int, ...]
    cells: tuple[PanelCell, ...]

    def __post_init__(self) -> None:
        if not self.panel_id or not self.positions or not self.cells:
            raise ValueError("panel id, positions, and cells must be non-empty")
        if len(set(self.positions)) != len(self.positions) or 0 in self.positions:
            raise ValueError("panel positions must be unique and cannot include zero")
        if {-2, -1}.intersection(self.positions):
            raise ValueError("a boundary panel cannot edit the protected core AG")
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("panel cell IDs must be unique")
        allele_states = []
        for cell in self.cells:
            if tuple(position for position, _base in cell.alleles) != self.positions:
                raise ValueError("every cell must declare every panel position in panel order")
            allele_states.append(cell.alleles)
        if len(set(allele_states)) != len(allele_states):
            raise ValueError("panel cells must have unique allele states")


@dataclass(frozen=True)
class ContrastSpec:
    """A named linear contrast over one complete panel."""

    contrast_id: str
    panel_id: str
    weights: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.contrast_id or not self.panel_id or not self.weights:
            raise ValueError("contrast id, panel id, and weights must be non-empty")
        names = [name for name, _weight in self.weights]
        values = [float(weight) for _name, weight in self.weights]
        if len(set(names)) != len(names) or not all(math.isfinite(value) for value in values):
            raise ValueError("contrast weights need unique cell IDs and finite values")
        if not any(value != 0.0 for value in values):
            raise ValueError("a contrast cannot be identically zero")
        if not math.isclose(sum(values), 0.0, abs_tol=1e-12):
            raise ValueError("contrast weights must sum to zero")


@dataclass(frozen=True)
class PanelValidationResult:
    checks: Mapping[str, bool]
    metrics: Mapping[str, object]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks.items() if not passed)

    def require_passed(self) -> None:
        if not self.passed:
            raise ValueError("boundary panel validation failed: " + ", ".join(self.failures))


def _factorial_cells() -> tuple[PanelCell, ...]:
    return tuple(
        PanelCell(f"m3_{upstream}__p1_{downstream}", ((-3, upstream), (1, downstream)))
        for upstream in TARGET_UPSTREAM_ALLELES
        for downstream in TARGET_DOWNSTREAM_ALLELES
    )


TARGET_PANEL = PanelSpec(TARGET_PANEL_ID, (-3, 1), _factorial_cells())
MINUS3_PANEL = PanelSpec(
    MINUS3_PANEL_ID,
    (-3,),
    tuple(PanelCell(f"m3_{base}", ((-3, base),)) for base in MINUS3_ALLELES),
)
PLUS1_PANEL = PanelSpec(
    PLUS1_PANEL_ID,
    (1,),
    tuple(PanelCell(f"p1_{base}", ((1, base),)) for base in PLUS1_ALLELES),
)
MINUS4_PANEL = PanelSpec(
    MINUS4_PANEL_ID,
    (-4,),
    tuple(PanelCell(f"m4_{base}", ((-4, base),)) for base in MINUS4_ALLELES),
)
PLUS2_PANEL = PanelSpec(
    PLUS2_PANEL_ID,
    (2,),
    tuple(PanelCell(f"p2_{base}", ((2, base),)) for base in PLUS2_ALLELES),
)


def _target_weights(kind: str) -> tuple[tuple[str, float], ...]:
    weights = []
    for cell in TARGET_PANEL.cells:
        alleles = dict(cell.alleles)
        if kind == "minus3":
            weight = 0.5 if alleles[-3] == "T" else -0.5 if alleles[-3] == "A" else 0.0
        elif kind == "plus1":
            weight = 0.25 if alleles[1] == "G" else -0.25
        elif kind == "interaction":
            weight = {
                ("T", "G"): 1.0,
                ("A", "G"): -1.0,
                ("T", "A"): -1.0,
                ("A", "A"): 1.0,
            }.get((alleles[-3], alleles[1]), 0.0)
        else:  # pragma: no cover - private, declaration-time guard
            raise ValueError(kind)
        weights.append((cell.cell_id, weight))
    return tuple(weights)


TARGET_MINUS3_T_MINUS_A = ContrastSpec(
    TARGET_MINUS3_T_MINUS_A_ID, TARGET_PANEL_ID, _target_weights("minus3")
)
TARGET_PLUS1_G_MINUS_A = ContrastSpec(
    TARGET_PLUS1_G_MINUS_A_ID, TARGET_PANEL_ID, _target_weights("plus1")
)
TARGET_INTERACTION = ContrastSpec(
    TARGET_INTERACTION_ID, TARGET_PANEL_ID, _target_weights("interaction")
)
MINUS3_T_MINUS_A_SINGLE = ContrastSpec(
    MINUS3_T_MINUS_A_SINGLE_ID, MINUS3_PANEL_ID, (("m3_A", -1.0), ("m3_T", 1.0))
)
PLUS1_G_MINUS_A_SINGLE = ContrastSpec(
    PLUS1_G_MINUS_A_SINGLE_ID, PLUS1_PANEL_ID, (("p1_A", -1.0), ("p1_G", 1.0))
)
MINUS4_T_MINUS_A = ContrastSpec(
    MINUS4_T_MINUS_A_ID, MINUS4_PANEL_ID, (("m4_A", -1.0), ("m4_T", 1.0))
)
PLUS2_G_MINUS_A = ContrastSpec(
    PLUS2_G_MINUS_A_ID, PLUS2_PANEL_ID, (("p2_A", -1.0), ("p2_G", 1.0))
)
TARGET_CONTRASTS = (
    TARGET_MINUS3_T_MINUS_A,
    TARGET_PLUS1_G_MINUS_A,
    TARGET_INTERACTION,
)
DIAGNOSTIC_CONTRASTS = (
    MINUS3_T_MINUS_A_SINGLE,
    MINUS4_T_MINUS_A,
    PLUS1_G_MINUS_A_SINGLE,
    PLUS2_G_MINUS_A,
)
ALL_CONTRASTS = TARGET_CONTRASTS + DIAGNOSTIC_CONTRASTS
CONTRAST_WEIGHTS = {
    contrast.contrast_id: dict(contrast.weights) for contrast in ALL_CONTRASTS
}
_BOUNDARY_DATASET_CACHE: dict[tuple[str, str], list[object]] = {}


def biological_to_array_offset(position: int) -> int:
    """Translate splice-site numbering (which has no zero) to a first-exon-base offset."""

    if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
        raise ValueError("biological position must be an integer")
    if position == 0:
        raise ValueError("biological splice-site position 0 does not exist")
    return int(position) if position < 0 else int(position) - 1


def biological_to_array_index(
    acceptor_pos: int,
    position: int,
    *,
    sequence_length: int | None = None,
    strand: str | None = None,
) -> int:
    """Return an array index in a transcript-oriented sequence.

    ``strand`` is checked but deliberately does not alter the result: minus-strand examples have
    already been reverse-complemented before entering the model.
    """

    if isinstance(acceptor_pos, bool) or not isinstance(acceptor_pos, (int, np.integer)):
        raise ValueError("acceptor_pos must be an integer")
    if strand not in (None, "+", "-"):
        raise ValueError("strand must be '+', '-', or None")
    index = int(acceptor_pos) + biological_to_array_offset(position)
    if sequence_length is not None and not 0 <= index < sequence_length:
        raise ValueError("biological position falls outside the sequence")
    return index


def touched_register_starts(
    acceptor_pos: int,
    biological_positions: Iterable[int],
    *,
    sequence_length: int,
    strand: str | None = None,
) -> tuple[int, ...]:
    """All dinucleotide starts whose contents can change under the declared edits."""

    starts = set()
    for position in biological_positions:
        index = biological_to_array_index(
            acceptor_pos, position, sequence_length=sequence_length, strand=strand
        )
        starts.update((index - 1, index))
    if any(start < 0 or start + 1 >= sequence_length for start in starts):
        raise ValueError("an edited nucleotide lacks complete flanking dinucleotide registers")
    return tuple(sorted(starts))


def positioned_motif_inventory(
    sequence: str,
    *,
    acceptor_pos: int,
    biological_positions: Iterable[int],
    strand: str | None = None,
) -> tuple[tuple[int, str], ...]:
    """Positioned ``AG``/``GT``/``GC`` inventory over every touched register."""

    starts = touched_register_starts(
        acceptor_pos,
        biological_positions,
        sequence_length=len(sequence),
        strand=strand,
    )
    return tuple((start, sequence[start:start + 2]) for start in starts
                 if sequence[start:start + 2] in SPLICE_MOTIFS)


def _field(item: object, name: str, default: object = None) -> object:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _target_example(example: object) -> tuple[str, int, str | None]:
    sequence = _field(example, "sequence")
    acceptor_pos = _field(example, "next_acceptor_pos")
    strand = _field(example, "strand")
    if not isinstance(sequence, str) or acceptor_pos is None:
        raise ValueError("example needs sequence and next_acceptor_pos")
    return sequence, int(acceptor_pos), str(strand) if strand is not None else None


def build_panel(
    sequence: str,
    acceptor_pos: int,
    panel: PanelSpec = TARGET_PANEL,
    *,
    strand: str | None = None,
    require_valid: bool = True,
) -> dict[str, str]:
    """Construct every panel cell by substituting only its declared nucleotide states."""

    cells: dict[str, str] = {}
    for cell in panel.cells:
        edited = list(sequence)
        for position, base in cell.alleles:
            index = biological_to_array_index(
                acceptor_pos, position, sequence_length=len(sequence), strand=strand
            )
            edited[index] = base
        cells[cell.cell_id] = "".join(edited)
    if require_valid:
        validate_panel(
            sequence, cells, panel, acceptor_pos=acceptor_pos, strand=strand
        ).require_passed()
    return cells


def build_target_panel(example: object) -> dict[str, str]:
    sequence, acceptor_pos, strand = _target_example(example)
    return build_panel(sequence, acceptor_pos, TARGET_PANEL, strand=strand)


def build_single_locus_panel(
    example: object,
    offset: int,
    alleles: Sequence[str],
    *,
    panel_id: str | None = None,
) -> dict[str, str]:
    """Build a strict single-position panel; fixed diagnostics use their canonical specs."""

    allele_tuple = tuple(alleles)
    if offset == -3 and allele_tuple == MINUS3_ALLELES:
        panel = MINUS3_PANEL
    elif offset == 1 and allele_tuple == PLUS1_ALLELES:
        panel = PLUS1_PANEL
    elif offset == -4 and allele_tuple == MINUS4_ALLELES:
        panel = MINUS4_PANEL
    elif offset == 2 and allele_tuple == PLUS2_ALLELES:
        panel = PLUS2_PANEL
    else:
        tag = _position_tag(offset)
        panel = PanelSpec(
            panel_id or f"single_{tag}",
            (offset,),
            tuple(PanelCell(f"{tag}_{base}", ((offset, base),)) for base in allele_tuple),
        )
    sequence, acceptor_pos, strand = _target_example(example)
    return build_panel(sequence, acceptor_pos, panel, strand=strand)


def validate_panel(
    sequence_or_example: str | object,
    cells: Mapping[str, str],
    panel: PanelSpec = TARGET_PANEL,
    *,
    acceptor_pos: int | None = None,
    strand: str | None = None,
) -> PanelValidationResult:
    """Audit the full panel, including exact positioned splice-motif hygiene."""

    if isinstance(sequence_or_example, str):
        sequence = sequence_or_example
        if acceptor_pos is None:
            raise ValueError("acceptor_pos is required when validating a raw sequence")
    else:
        sequence, example_acceptor, example_strand = _target_example(sequence_or_example)
        if acceptor_pos is None:
            acceptor_pos = example_acceptor
        if strand is None:
            strand = example_strand
    acceptor_pos = int(acceptor_pos)
    expected_ids = tuple(cell.cell_id for cell in panel.cells)
    ids_exact = set(cells) == set(expected_ids) and len(cells) == len(expected_ids)
    lengths_equal = ids_exact and all(
        isinstance(cells[cell_id], str) and len(cells[cell_id]) == len(sequence)
        for cell_id in expected_ids
    )
    try:
        indices = {
            position: biological_to_array_index(
                acceptor_pos, position, sequence_length=len(sequence), strand=strand
            )
            for position in panel.positions
        }
        starts = touched_register_starts(
            acceptor_pos, panel.positions, sequence_length=len(sequence), strand=strand
        )
        coordinate_bounds = True
    except ValueError:
        indices, starts, coordinate_bounds = {}, (), False

    core_in_bounds = 2 <= acceptor_pos <= len(sequence)
    native_core_ag = core_in_bounds and sequence[acceptor_pos - 2:acceptor_pos] == "AG"
    cell_core_ag = lengths_equal and all(
        cells[cell_id][acceptor_pos - 2:acceptor_pos] == "AG" for cell_id in expected_ids
    )
    declared = {indices[position] for position in panel.positions} if coordinate_bounds else set()
    only_declared = lengths_equal and coordinate_bounds and all(
        {index for index, (native, edited) in enumerate(zip(sequence, cells[cell_id]))
         if native != edited}.issubset(declared)
        for cell_id in expected_ids
    )
    exact_alleles = lengths_equal and coordinate_bounds and all(
        all(cells[cell.cell_id][indices[position]] == base for position, base in cell.alleles)
        for cell in panel.cells
    )

    inventories: dict[str, tuple[tuple[int, str], ...]] = {}
    if lengths_equal and coordinate_bounds:
        inventories = {
            "native": tuple(
                (start, sequence[start:start + 2]) for start in starts
                if sequence[start:start + 2] in SPLICE_MOTIFS
            ),
            **{
            cell_id: tuple(
                (start, cells[cell_id][start:start + 2]) for start in starts
                if cells[cell_id][start:start + 2] in SPLICE_MOTIFS
            )
            for cell_id in expected_ids
            },
        }
    inventories_identical = (
        len(inventories) == len(expected_ids) + 1
        and len(set(inventories.values())) == 1
    )
    no_extra_touched_motifs = (
        len(inventories) == len(expected_ids) + 1
        and all(not inventory for inventory in inventories.values())
    )
    checks = {
        "cell_ids_exact": ids_exact,
        "coordinate_bounds": coordinate_bounds,
        "length_equal": lengths_equal,
        "native_core_ag": native_core_ag,
        "cell_core_ag": cell_core_ag,
        "only_declared_loci_changed": only_declared,
        "declared_alleles_exact": exact_alleles,
        "positioned_inventories_identical": inventories_identical,
        "no_extra_touched_motifs": no_extra_touched_motifs,
    }
    metrics = {
        "panel_id": panel.panel_id,
        "acceptor_pos": acceptor_pos,
        "declared_array_indices": tuple(sorted(declared)),
        "touched_register_starts": starts,
        "positioned_inventories": inventories,
    }
    return PanelValidationResult(checks, metrics)


def boundary_relevant(example: object) -> bool:
    """Retain only base-eligible examples used by at least one frozen BG panel."""

    try:
        sequence, acceptor_pos, strand = _target_example(example)
        donor_pos = int(_field(example, "donor_pos"))
    except (TypeError, ValueError):
        return False
    if sequence[donor_pos:donor_pos + 2] != "GT" or not 60 <= acceptor_pos - donor_pos <= 150:
        return False

    def passes(panel: PanelSpec) -> bool:
        try:
            cells = build_panel(
                sequence,
                acceptor_pos,
                panel,
                strand=strand,
                require_valid=False,
            )
            return validate_panel(
                sequence,
                cells,
                panel,
                acceptor_pos=acceptor_pos,
                strand=strand,
            ).passed
        except (TypeError, ValueError):
            return False

    return (
        passes(TARGET_PANEL)
        or (passes(MINUS3_PANEL) and passes(MINUS4_PANEL))
        or (passes(PLUS1_PANEL) and passes(PLUS2_PANEL))
    )


def load_boundary_examples(gtf: str, fasta: str) -> list[object]:
    """Stream gene blocks and retain canonical ``CouplingExample`` rows used by BG1/BG2.

    Transcript support and annotated-site pools are gene-local, so one complete contiguous
    GENCODE gene block is sufficient to reproduce the stock builder exactly.  Processing one
    gene at a time avoids retaining the annotation-wide transcript graph or irrelevant 1,024-nt
    strings.  The stock builder still constructs each retained row, including ``other_ag``,
    ``other_gt``, real-site controls, intron lengths, and metadata.
    """

    key = (str(gtf), str(fasta))
    if key not in _BOUNDARY_DATASET_CACHE:
        from collections import defaultdict
        from itertools import groupby

        from gfmcd.splicing.coupling import CouplingConfig, build_coupling_dataset
        from gfmcd.splicing.dataset import PyfaidxGenome
        from gfmcd.splicing.gtf import TranscriptRecord, iter_exons

        genome = PyfaidxGenome(fasta)
        output = []
        seen_genes = set()
        for gene_id, gene_rows in groupby(iter_exons(gtf), key=lambda exon: exon.gene_id):
            if not gene_id:
                raise RuntimeError("GTF exon lacks gene_id")
            if gene_id in seen_genes:
                raise RuntimeError(
                    "GTF gene blocks are not contiguous; streaming would change support counts"
                )
            seen_genes.add(gene_id)
            grouped = defaultdict(list)
            for exon in gene_rows:
                if exon.transcript_id:
                    grouped[exon.transcript_id].append(exon)
            transcripts = []
            for transcript_id, exons in grouped.items():
                first = exons[0]
                if any(
                    (exon.chrom, exon.strand, exon.gene_id)
                    != (first.chrom, first.strand, first.gene_id)
                    for exon in exons
                ):
                    raise ValueError(f"inconsistent transcript record: {transcript_id}")
                transcripts.append(TranscriptRecord(
                    transcript_id=transcript_id,
                    gene_id=first.gene_id,
                    chrom=first.chrom,
                    strand=first.strand,
                    transcript_type=first.transcript_type,
                    tags=tuple(sorted({tag for exon in exons for tag in exon.tags})),
                    exons=tuple(exons),
                ))
            transcripts.sort(key=lambda item: (item.chrom, item.gene_id, item.transcript_id))
            output.extend(build_coupling_dataset(
                gtf,
                genome,
                CouplingConfig(),
                transcripts=transcripts,
                example_filter=boundary_relevant,
            ))
        # ``load_transcripts`` sorts by (chrom, gene, transcript).  The stable block sort
        # restores that global order while retaining each gene's transcript/exon order.
        output.sort(key=lambda item: (item.chrom, item.gene_id))
        _BOUNDARY_DATASET_CACHE[key] = output
    return _BOUNDARY_DATASET_CACHE[key]


def contrast(cell_values: Mapping[str, float], spec: ContrastSpec) -> float:
    missing = [cell_id for cell_id, _weight in spec.weights if cell_id not in cell_values]
    if missing:
        raise ValueError(f"missing contrast cells: {missing}")
    values = [float(cell_values[cell_id]) for cell_id, _weight in spec.weights]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("contrast cell values must be finite")
    return float(sum(float(cell_values[cell_id]) * weight for cell_id, weight in spec.weights))


def target_contrasts(cell_values: Mapping[str, float]) -> dict[str, float]:
    return {spec.contrast_id: contrast(cell_values, spec) for spec in TARGET_CONTRASTS}


def stable_gene_role(
    gene_id: str,
    *,
    namespace: str = ROLE_NAMESPACE,
    role_weights: Sequence[tuple[str, int]] = ROLE_WEIGHTS,
) -> str:
    """Assign all rows from one gene to one predeclared role by SHA-256."""

    if not gene_id or not namespace or not role_weights:
        raise ValueError("gene_id, namespace, and role_weights must be non-empty")
    names = [name for name, _weight in role_weights]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("role names must be non-empty and unique")
    if any(isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
           for _name, weight in role_weights):
        raise ValueError("role weights must be positive integers")
    total = sum(weight for _name, weight in role_weights)
    digest = hashlib.sha256(
        f"gfmcd-bg1-role-v1\0{namespace}\0{gene_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % total
    cumulative = 0
    for name, weight in role_weights:
        cumulative += weight
        if bucket < cumulative:
            return name
    raise AssertionError("unreachable role bucket")


def _stable_key(namespace: str, purpose: str, *parts: str) -> tuple[bytes, tuple[str, ...]]:
    payload = "\0".join(("gfmcd-bg1-v1", namespace, purpose, *parts)).encode("utf-8")
    return hashlib.sha256(payload).digest(), tuple(parts)


def select_gene_cohort(
    examples: Iterable[object],
    *,
    role: str | None,
    namespace: str = ROLE_NAMESPACE,
    role_weights: Sequence[tuple[str, int]] = ROLE_WEIGHTS,
    n: int | None = None,
) -> tuple[object, ...]:
    """Select one outcome-blind example per gene, independent of input/hash order."""

    valid_roles = {name for name, _weight in role_weights}
    if role is not None and role not in valid_roles:
        raise ValueError(f"role must be one of {sorted(valid_roles)} or None")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n <= 0):
        raise ValueError("n must be a positive integer or None")
    by_gene: dict[str, list[object]] = defaultdict(list)
    for example in examples:
        gene_id = str(_field(example, "gene_id", ""))
        example_id = str(_field(example, "example_id", ""))
        if not gene_id or not example_id:
            raise ValueError("every cohort example needs non-empty gene_id and example_id")
        if role is None or stable_gene_role(
            gene_id, namespace=namespace, role_weights=role_weights
        ) == role:
            by_gene[gene_id].append(example)
    selected = []
    for gene_id, candidates in by_gene.items():
        selected.append(min(
            candidates,
            key=lambda example: _stable_key(
                namespace, "representative", gene_id, str(_field(example, "example_id"))
            ),
        ))
    selected.sort(key=lambda example: _stable_key(
        namespace, "cohort-order", str(_field(example, "gene_id"))
    ))
    if n is not None:
        if len(selected) < n:
            raise RuntimeError(f"only {len(selected)} eligible genes for role {role!r}; asked for {n}")
        selected = selected[:n]
    return tuple(selected)


def margins_from_logits(
    logits: object,
    donor_positions: Sequence[int] | np.ndarray,
    *,
    base_token_ids: Mapping[str, int],
    token_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear and max G margins from the same batched logits tensor.

    ``logits`` may be ``[batch, sequence, vocabulary]`` or already selected
    ``[batch, vocabulary]``.  ``token_offset=1`` accounts for SpliceBERT's leading CLS token.
    """

    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    array = np.asarray(logits, dtype=np.float64)
    positions = np.asarray(donor_positions)
    if positions.ndim != 1 or any(isinstance(value, (bool, np.bool_)) for value in positions):
        raise ValueError("donor_positions must be a one-dimensional integer sequence")
    if not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("donor_positions must be integers")
    if array.ndim == 3:
        if array.shape[0] != len(positions):
            raise ValueError("logit batch and donor positions must have matched lengths")
        token_positions = positions.astype(int) + int(token_offset)
        if np.any(token_positions < 0) or np.any(token_positions >= array.shape[1]):
            raise ValueError("donor position falls outside the logit sequence")
        selected = array[np.arange(array.shape[0]), token_positions]
    elif array.ndim == 2:
        if array.shape[0] != len(positions):
            raise ValueError("selected logit rows and donor positions must have matched lengths")
        selected = array
    else:
        raise ValueError("logits must have shape [batch,vocab] or [batch,sequence,vocab]")
    if set(base_token_ids) != set(DNA_BASES):
        raise ValueError("base_token_ids must contain exactly A, C, G, and T")
    ids = [int(base_token_ids[base]) for base in DNA_BASES]
    if len(set(ids)) != 4 or min(ids) < 0 or max(ids) >= selected.shape[1]:
        raise ValueError("base token IDs must be distinct valid vocabulary indices")
    base_logits = {base: selected[:, int(base_token_ids[base])] for base in DNA_BASES}
    if not all(np.isfinite(values).all() for values in base_logits.values()):
        raise ValueError("base logits must be finite")
    alternatives = np.column_stack([base_logits[base] for base in ("A", "C", "T")])
    linear = base_logits["G"] - alternatives.mean(axis=1)
    maximum = base_logits["G"] - alternatives.max(axis=1)
    return linear, maximum


def aggregate_gene_cells(
    rows: Iterable[Mapping[str, object]],
    *,
    panel: PanelSpec = TARGET_PANEL,
    value_keys: Sequence[str] = ("linear_margin", "max_margin"),
    gene_key: str = "gene_id",
    cell_key: str = "cell_id",
) -> tuple[dict[str, object], ...]:
    """Average repeated rows within gene/cell and require each gene's complete panel."""

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {key: [] for key in value_keys}
    )
    expected = {cell.cell_id for cell in panel.cells}
    for row in rows:
        gene, cell = str(row[gene_key]), str(row[cell_key])
        if not gene or cell not in expected:
            raise ValueError("rows need a non-empty gene and a declared panel cell")
        for key in value_keys:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError("gene-cell values must be finite")
            grouped[(gene, cell)][key].append(value)
    genes = sorted({gene for gene, _cell in grouped})
    output = []
    for gene in genes:
        present = {cell for row_gene, cell in grouped if row_gene == gene}
        if present != expected:
            raise ValueError(f"gene {gene} does not have the complete {panel.panel_id} panel")
        for cell in (item.cell_id for item in panel.cells):
            item: dict[str, object] = {gene_key: gene, cell_key: cell}
            for key in value_keys:
                item[key] = float(np.mean(grouped[(gene, cell)][key]))
            item["n_rows"] = len(grouped[(gene, cell)][value_keys[0]])
            output.append(item)
    return tuple(output)


def add_one_pvalue(
    observed: float,
    null_statistics: Sequence[float],
    *,
    alternative: str = "greater",
) -> float:
    """Exact add-one randomization p-value against a precomputed null family."""

    null = np.asarray(null_statistics, dtype=np.float64)
    if null.ndim != 1 or null.size == 0 or not np.isfinite(null).all() or not math.isfinite(observed):
        raise ValueError("observed and null statistics must be finite and null non-empty")
    if alternative == "greater":
        exceed = null >= observed
    elif alternative == "less":
        exceed = null <= observed
    elif alternative == "two-sided":
        exceed = np.abs(null) >= abs(observed)
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return float((1 + np.count_nonzero(exceed)) / (null.size + 1))


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite one-dimensional array in [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def max_family_add_one_pvalues(
    observed: Sequence[float],
    null_statistics: np.ndarray,
    *,
    two_sided: bool = True,
) -> np.ndarray:
    """Add-one p-values using each null draw's maximum across the scanned family."""

    target = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null_statistics, dtype=np.float64)
    if (
        target.ndim != 1 or null.ndim != 2 or null.shape[1] != len(target)
        or null.shape[0] == 0 or not np.isfinite(target).all() or not np.isfinite(null).all()
    ):
        raise ValueError("null matrix must be finite [draw,family] and match observed")
    null_max = np.max(np.abs(null) if two_sided else null, axis=1)
    target_values = np.abs(target) if two_sided else target
    return np.asarray([
        (1 + np.count_nonzero(null_max >= value)) / (len(null_max) + 1)
        for value in target_values
    ], dtype=np.float64)


def panel_declaration(panel: PanelSpec) -> dict[str, object]:
    return {
        "panel_id": panel.panel_id,
        "positions": list(panel.positions),
        "cells": [
            {
                "cell_id": cell.cell_id,
                "alleles": [
                    {"position": position, "base": base} for position, base in cell.alleles
                ],
            }
            for cell in panel.cells
        ],
    }


def contrast_declaration(spec: ContrastSpec) -> dict[str, object]:
    return {
        "contrast_id": spec.contrast_id,
        "panel_id": spec.panel_id,
        "weights": [{"cell_id": cell_id, "weight": weight} for cell_id, weight in spec.weights],
    }


def design_declaration() -> dict[str, object]:
    """Exact JSON-ready declaration shared by preflight and target runners."""

    return {
        "namespace": ROLE_NAMESPACE,
        "role_weights": [list(item) for item in ROLE_WEIGHTS],
        "sesoi": SESOI,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "gtf_sha256": GTF_SHA256,
        "fasta_sha256": FASTA_SHA256,
        "model_file_sha256": dict(MODEL_FILE_SHA256),
        "analysis": {
            "seed": ANALYSIS_SEED,
            "bg1_bootstrap_draws": BG1_BOOTSTRAP_DRAWS,
            "bg1_permutation_draws": BG1_PERMUTATION_DRAWS,
            "bg2_bootstrap_draws": BG2_BOOTSTRAP_DRAWS,
            "bg2_bootstrap_lower_quantile": 0.05,
            "bg2_bootstrap_two_sided_interval_quantiles": [0.025, 0.975],
            "bg2_bootstrap_seed_schedule": {
                "primary": "20260809 + transition_index",
                "true_minus_cross": "20260819 + transition_index",
                "max_ratio": "20260829 + transition_index",
                "behavior": "20260839 + transition_index",
                "max_lift": "20260849 + transition_index",
                "cross": "20260859 + transition_index",
                "max_denominator": "20260869 + transition_index",
            },
            "preflight_bootstrap_draws": PREFLIGHT_BOOTSTRAP_DRAWS,
            "preflight_permutation_draws": PREFLIGHT_PERMUTATION_DRAWS,
            "preflight_calibration_replicates": PREFLIGHT_CALIBRATION_REPLICATES,
            "primary_inference": "studentized simultaneous max-t CI; one-sided max-signflip p",
            "max_margin_role": "secondary_bridge",
            "localization_threshold": SESOI,
            "localization_multiplicity": "Holm across enabled arms",
            "signed_coordinate_shams": 19,
        },
        "minimum_gene_supply": {
            "train_behavior": MIN_TRAIN_BEHAVIOR_GENES,
            "train_route": MIN_TRAIN_ROUTE_GENES,
            "train_health": PREFLIGHT_HEALTH_GENES,
            "validation_behavior": MIN_BG1_TARGET_GENES,
            "validation_behavior_per_strand": MIN_BG1_TARGET_GENES_PER_STRAND,
            "validation_outer": MIN_OUTER_TARGET_GENES,
            "validation_outer_per_strand": MIN_OUTER_TARGET_GENES_PER_STRAND,
            "validation_route": MIN_BG2_TARGET_GENES,
            "validation_route_per_strand": MIN_BG2_TARGET_GENES_PER_STRAND,
        },
        "splice_motifs": sorted(SPLICE_MOTIFS),
        "panels": {
            "target": panel_declaration(TARGET_PANEL),
            "minus3_inner": panel_declaration(MINUS3_PANEL),
            "minus4_diagnostic": panel_declaration(MINUS4_PANEL),
            "plus1_inner": panel_declaration(PLUS1_PANEL),
            "plus2_diagnostic": panel_declaration(PLUS2_PANEL),
        },
        "contrasts": [contrast_declaration(spec) for spec in ALL_CONTRASTS],
    }


def preflight_config() -> dict[str, object]:
    """Compatibility name for the frozen, model-free part of the BG preflight config."""

    return design_declaration()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def declaration_sha256() -> str:
    return _canonical_sha256(design_declaration())


def construct_sha256(cells: Mapping[str, str], *, panel_id: str = TARGET_PANEL_ID) -> str:
    payload = {
        "panel_id": panel_id,
        "cells": [
            {"cell_id": cell_id, "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest()}
            for cell_id, sequence in sorted(cells.items())
        ],
    }
    return _canonical_sha256(payload)


def panel_construct_declaration(example: object, panel: PanelSpec) -> dict[str, object]:
    """Persist one native sequence so a model-free audit can reconstruct every panel cell."""

    sequence, acceptor_pos, strand = _target_example(example)
    donor_pos = int(_field(example, "donor_pos"))
    cells = build_panel(sequence, acceptor_pos, panel, strand=strand)
    return {
        "panel_id": panel.panel_id,
        "example_id": str(_field(example, "example_id", "")),
        "gene_id": str(_field(example, "gene_id", "")),
        "chrom": str(_field(example, "chrom", "")),
        "strand": strand,
        "donor_pos": donor_pos,
        "next_acceptor_pos": acceptor_pos,
        "partner_span": acceptor_pos - donor_pos,
        "native_sequence": sequence,
        "native_sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "cell_sequence_sha256": {
            cell_id: hashlib.sha256(cell_sequence.encode("ascii")).hexdigest()
            for cell_id, cell_sequence in cells.items()
        },
        "construct_sha256": construct_sha256(cells, panel_id=panel.panel_id),
    }


def cohort_sha256(examples: Sequence[object]) -> str:
    rows = []
    for example in examples:
        sequence = str(_field(example, "sequence", ""))
        rows.append({
            "example_id": str(_field(example, "example_id", "")),
            "gene_id": str(_field(example, "gene_id", "")),
            "chrom": str(_field(example, "chrom", "")),
            "strand": str(_field(example, "strand", "")),
            "donor_pos": _field(example, "donor_pos"),
            "next_acceptor_pos": _field(example, "next_acceptor_pos"),
            "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        })
    return _canonical_sha256(rows)


def cohort_declaration(cohorts: Mapping[str, Sequence[object]]) -> dict[str, object]:
    """JSON-ready ordered IDs/genes by role plus a hash over those exact memberships."""

    ids = {
        role: [str(_field(example, "example_id", "")) for example in examples]
        for role, examples in cohorts.items()
    }
    genes = {
        role: [str(_field(example, "gene_id", "")) for example in examples]
        for role, examples in cohorts.items()
    }
    all_genes = [gene for role_genes in genes.values() for gene in role_genes]
    if len(all_genes) != len(set(all_genes)):
        raise ValueError("cohort roles must be gene-disjoint")
    payload: dict[str, object] = {"ids": ids, "genes": genes}
    payload["sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "ANALYSIS_SEED",
    "ALL_CONTRASTS",
    "BG1_NAMESPACE",
    "BG1_BOOTSTRAP_DRAWS",
    "BG1_PERMUTATION_DRAWS",
    "BG2_BOOTSTRAP_DRAWS",
    "CHECKPOINT_SHA256",
    "FASTA_SHA256",
    "GTF_SHA256",
    "CONTRAST_WEIGHTS",
    "ContrastSpec",
    "DIAGNOSTIC_CONTRASTS",
    "DNA_BASES",
    "MINUS4_ALLELES",
    "MINUS4_PANEL",
    "MINUS4_PANEL_ID",
    "MINUS4_T_MINUS_A",
    "MINUS4_T_MINUS_A_ID",
    "MINUS3_ALLELES",
    "MODEL_FILE_SHA256",
    "MINUS3_PANEL",
    "MINUS3_PANEL_ID",
    "MINUS3_T_MINUS_A_SINGLE",
    "MINUS3_T_MINUS_A_SINGLE_ID",
    "PLUS1_ALLELES",
    "PLUS1_G_MINUS_A_SINGLE",
    "PLUS1_G_MINUS_A_SINGLE_ID",
    "PLUS1_PANEL",
    "PLUS1_PANEL_ID",
    "PLUS2_ALLELES",
    "PLUS2_G_MINUS_A",
    "PLUS2_G_MINUS_A_ID",
    "PLUS2_PANEL",
    "PLUS2_PANEL_ID",
    "MIN_BG1_TARGET_GENES",
    "MIN_BG1_TARGET_GENES_PER_STRAND",
    "MIN_BG2_TARGET_GENES",
    "MIN_BG2_TARGET_GENES_PER_STRAND",
    "MIN_OUTER_TARGET_GENES",
    "MIN_OUTER_TARGET_GENES_PER_STRAND",
    "MIN_TRAIN_BEHAVIOR_GENES",
    "MIN_TRAIN_ROUTE_GENES",
    "PanelCell",
    "PanelSpec",
    "PanelValidationResult",
    "ROLE_NAMESPACE",
    "ROLE_WEIGHTS",
    "PREFLIGHT_BOOTSTRAP_DRAWS",
    "PREFLIGHT_CALIBRATION_REPLICATES",
    "PREFLIGHT_HEALTH_GENES",
    "PREFLIGHT_PERMUTATION_DRAWS",
    "SESOI",
    "SPLICE_MOTIFS",
    "TARGET_CONTRASTS",
    "TARGET_DOWNSTREAM_ALLELES",
    "TARGET_INTERACTION",
    "TARGET_INTERACTION_ID",
    "TARGET_MINUS3_T_MINUS_A",
    "TARGET_MINUS3_T_MINUS_A_ID",
    "TARGET_PANEL",
    "TARGET_PANEL_ID",
    "TARGET_PLUS1_G_MINUS_A",
    "TARGET_PLUS1_G_MINUS_A_ID",
    "TARGET_UPSTREAM_ALLELES",
    "add_one_pvalue",
    "aggregate_gene_cells",
    "biological_to_array_index",
    "biological_to_array_offset",
    "build_panel",
    "build_single_locus_panel",
    "build_target_panel",
    "boundary_relevant",
    "cohort_declaration",
    "cohort_sha256",
    "construct_sha256",
    "contrast",
    "contrast_declaration",
    "declaration_sha256",
    "design_declaration",
    "holm_adjust",
    "load_boundary_examples",
    "margins_from_logits",
    "max_family_add_one_pvalues",
    "panel_declaration",
    "panel_construct_declaration",
    "positioned_motif_inventory",
    "preflight_config",
    "select_gene_cohort",
    "stable_gene_role",
    "target_contrasts",
    "touched_register_starts",
    "validate_panel",
]
