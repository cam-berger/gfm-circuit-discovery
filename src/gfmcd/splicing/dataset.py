"""Build transcript-oriented exon-pair examples from a genome FASTA and GENCODE GTF."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Iterable, Protocol

from gfmcd.splicing.candidates import CandidateConfig, enumerate_candidate_pairs
from gfmcd.splicing.gtf import ExonRecord, TranscriptRecord, load_transcripts
from gfmcd.splicing.schema import SplicePairExample


_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1].upper()


class GenomeAccessor(Protocol):
    def fetch(self, chrom: str, start: int, end: int) -> str: ...

    def length(self, chrom: str) -> int: ...


class PyfaidxGenome:
    """Lazy adapter around ``pyfaidx.Fasta``.

    ``pyfaidx`` is imported only when this class is instantiated so the core test
    suite does not require the genome extra.
    """

    def __init__(self, path: str | Path):
        try:
            from pyfaidx import Fasta
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("install the genome extra: uv sync --extra genome") from exc
        self.path = Path(path)
        self._fasta = Fasta(str(path), as_raw=True, sequence_always_upper=True)

    def _resolve(self, chrom: str) -> str:
        if chrom in self._fasta:
            return chrom
        alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alt in self._fasta:
            return alt
        raise KeyError(f"chromosome {chrom!r} absent from {self.path}")

    def fetch(self, chrom: str, start: int, end: int) -> str:
        key = self._resolve(chrom)
        return str(self._fasta[key][start:end]).upper()

    def length(self, chrom: str) -> int:
        key = self._resolve(chrom)
        return len(self._fasta[key])


@dataclass(frozen=True)
class DatasetBuildConfig:
    left_flank: int = 200
    right_flank: int = 200
    max_sequence_length: int = 900  # leaves room for [CLS]/[SEP] under SpliceBERT's 1024 limit
    min_exon_length: int = 20
    max_exon_length: int = 500
    canonical_only: bool = True
    minimum_candidate_pairs: int = 2
    transcript_types: tuple[str, ...] = ("protein_coding",)
    required_tags: tuple[str, ...] = ()
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    split_method: str = "chromosome"
    val_chromosomes: tuple[str, ...] = ("chr17", "chr18")
    test_chromosomes: tuple[str, ...] = ("chr19", "chr20", "chr21", "chr22")
    exclude_chromosomes: tuple[str, ...] = ("chrX", "chrY", "chrM", "MT")
    deduplicate_exons: bool = True

    def validate(self) -> None:
        if self.left_flank < 2 or self.right_flank < 2:
            raise ValueError("flanks must leave room for canonical splice dinucleotides")
        if self.max_sequence_length > 1022:
            raise ValueError("SpliceBERT adds [CLS]/[SEP]; nucleotide sequence must be <=1022")
        if self.min_exon_length < 1 or self.max_exon_length < self.min_exon_length:
            raise ValueError("invalid exon length range")
        if self.minimum_candidate_pairs < 1:
            raise ValueError("minimum_candidate_pairs must be positive")
        if self.split_method not in {"chromosome", "gene_hash"}:
            raise ValueError("split_method must be chromosome or gene_hash")
        self.candidate.validate()

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


def _split_for(chrom: str, gene_id: str, cfg: DatasetBuildConfig) -> str:
    if cfg.split_method == "chromosome":
        normalized = chrom if chrom.startswith("chr") else f"chr{chrom}"
        if normalized in cfg.val_chromosomes:
            return "val"
        if normalized in cfg.test_chromosomes:
            return "test"
        return "train"
    digest = int(hashlib.sha256(gene_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if digest < 80 else "val" if digest < 90 else "test"


def _extract_oriented_window(
    genome: GenomeAccessor,
    exon: ExonRecord,
    cfg: DatasetBuildConfig,
) -> tuple[str, int, int, int, int] | None:
    """Return sequence, acceptor, donor, genomic window start/end, or ``None``."""
    if exon.length < cfg.min_exon_length or exon.length > cfg.max_exon_length:
        return None
    total = cfg.left_flank + exon.length + cfg.right_flank
    if total > cfg.max_sequence_length:
        return None

    if exon.strand == "+":
        window_start = exon.start - cfg.left_flank
        window_end = exon.end + cfg.right_flank
    else:
        # After reverse complementation, the high-coordinate genomic flank is
        # transcript-upstream (left in the oriented sequence).
        window_start = exon.start - cfg.right_flank
        window_end = exon.end + cfg.left_flank

    if window_start < 0 or window_end > genome.length(exon.chrom):
        return None
    sequence = genome.fetch(exon.chrom, window_start, window_end)
    if exon.strand == "-":
        sequence = reverse_complement(sequence)
    sequence = sequence.upper().replace("U", "T")
    acceptor = cfg.left_flank
    donor = cfg.left_flank + exon.length
    return sequence, acceptor, donor, window_start, window_end


def _iter_internal_exons(transcripts: Iterable[TranscriptRecord]) -> Iterable[tuple[TranscriptRecord, ExonRecord, int]]:
    for transcript in transcripts:
        ordered = transcript.ordered_exons()
        for index, exon in enumerate(ordered[1:-1], start=2):
            yield transcript, exon, index


def build_exon_pair_examples(
    gtf_path: str | Path,
    genome: GenomeAccessor,
    cfg: DatasetBuildConfig | None = None,
    *,
    allowed_chromosomes: set[str] | None = None,
    max_examples: int | None = None,
) -> tuple[list[SplicePairExample], dict[str, int]]:
    """Build examples and return ``(examples, skip_counts)``.

    The function contains no model-specific code.  Dataset construction can be
    audited independently from SpliceBERT and its checkpoints.
    """
    cfg = cfg or DatasetBuildConfig()
    cfg.validate()
    transcripts = load_transcripts(
        gtf_path,
        transcript_types=cfg.transcript_types,
        required_tags=cfg.required_tags,
        allowed_chromosomes=allowed_chromosomes,
    )
    excluded = {x if x.startswith("chr") else f"chr{x}" for x in cfg.exclude_chromosomes}
    examples: list[SplicePairExample] = []
    skips: dict[str, int] = {}
    seen_exons: set[tuple[str, int, int, str]] = set()

    def skip(reason: str) -> None:
        skips[reason] = skips.get(reason, 0) + 1

    for transcript, exon, ordinal in _iter_internal_exons(transcripts):
        normalized_chrom = exon.chrom if exon.chrom.startswith("chr") else f"chr{exon.chrom}"
        if normalized_chrom in excluded:
            skip("excluded_chromosome")
            continue
        key = (exon.chrom, exon.start, exon.end, exon.strand)
        if cfg.deduplicate_exons and key in seen_exons:
            skip("duplicate_exon")
            continue
        seen_exons.add(key)

        extracted = _extract_oriented_window(genome, exon, cfg)
        if extracted is None:
            skip("length_or_boundary")
            continue
        sequence, true_acceptor, true_donor, window_start, window_end = extracted
        acceptor_motif = sequence[true_acceptor - 2 : true_acceptor]
        donor_motif = sequence[true_donor : true_donor + 2]
        if cfg.canonical_only and (
            acceptor_motif not in cfg.candidate.acceptor_motifs
            or donor_motif not in cfg.candidate.donor_motifs
        ):
            skip("noncanonical_true_site")
            continue
        try:
            acceptors, donors, pairs, true_index = enumerate_candidate_pairs(
                sequence,
                cfg.candidate,
                true_acceptor=true_acceptor,
                true_donor=true_donor,
            )
        except ValueError:
            skip("candidate_generation")
            continue
        if true_index is None:  # pragma: no cover - guarded by enumerate_candidate_pairs
            skip("missing_true_pair")
            continue
        if len(pairs) < cfg.minimum_candidate_pairs:
            skip("no_decoy_pair")
            continue

        exon_number = exon.exon_number if exon.exon_number is not None else ordinal
        example_id = (
            f"{exon.chrom}:{exon.start}-{exon.end}:{exon.strand}:"
            f"{transcript.transcript_id}"
        )
        example = SplicePairExample(
            example_id=example_id,
            sequence=sequence,
            acceptors=acceptors,
            donors=donors,
            pairs=pairs,
            true_pair_index=true_index,
            split=_split_for(exon.chrom, exon.gene_id, cfg),
            chrom=exon.chrom,
            strand=exon.strand,
            gene_id=exon.gene_id,
            transcript_id=exon.transcript_id,
            exon_number=exon_number,
            genomic_exon_start=exon.start,
            genomic_exon_end=exon.end,
            metadata={
                "genomic_window_start": window_start,
                "genomic_window_end": window_end,
                "true_acceptor_motif": acceptor_motif,
                "true_donor_motif": donor_motif,
                "transcript_tags": list(transcript.tags),
                "coordinate_convention": "0-based half-open genomic; 0-based transcript-oriented sequence",
            },
        )
        example.validate()
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break

    examples.sort(key=lambda x: (x.split, x.chrom, x.genomic_exon_start, x.example_id))
    return examples, skips


def split_examples(examples: Iterable[SplicePairExample]) -> dict[str, list[SplicePairExample]]:
    out = {"train": [], "val": [], "test": []}
    for example in examples:
        out.setdefault(example.split, []).append(example)
    return out
