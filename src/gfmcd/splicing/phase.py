"""Splice (intron) phase from GENCODE CDS features.

Intron phase at a splice donor = the reading-frame position where the intron interrupts the
codon: 0 = between whole codons, 1 = after the 1st base, 2 = after the 2nd. It is a property of
the CODING boundary, so it is defined only for exons that are wholly a CDS.

Computed from the GTF ``frame`` (a.k.a. GFF3 phase): for a CDS of length ``L`` (0-based
half-open) with frame ``f``, the number of coding bases before its first whole codon is
``(3 - f) mod 3``, so the phase at its 3' (donor) end is ``((3 - f) + L) mod 3``. This is
STRAND-AGNOSTIC because ``frame`` is already expressed in translation coordinates.

Validated on GENCODE v50: CDS chr1:924431-924948 (0-based, frame 0, L=517) -> phase 1, and the
next CDS in that transcript has frame 2 = (3 - 1) mod 3, as required.
"""
from __future__ import annotations

from collections import defaultdict

from gfmcd.splicing.gtf import _open_text, attr_first, parse_attributes


def intron_phase(start_0: int, end: int, frame: int) -> int:
    """Intron phase (0/1/2) at the donor (3') end of a CDS, from its 0-based coords and frame."""
    return ((3 - frame) + (end - start_0)) % 3


def load_cds_frames(path, *, allowed_chromosomes=None) -> dict[tuple, int]:
    """Map ``(gene_id, start_0, end) -> donor intron phase`` from every CDS feature.

    Keys are 0-based half-open exon coordinates (matching ``ExonRecord``/`CouplingExample`
    metadata). Exons whose phase disagrees across transcripts are AMBIGUOUS and dropped.
    """
    phases: dict[tuple, set] = defaultdict(set)
    with _open_text(path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9 or cols[2] != "CDS":
                continue
            if allowed_chromosomes is not None and cols[0] not in allowed_chromosomes:
                continue
            frame_txt = cols[7]
            if frame_txt not in ("0", "1", "2"):
                continue
            start_0, end = int(cols[3]) - 1, int(cols[4])
            gid = attr_first(parse_attributes(cols[8]), "gene_id")
            phases[(gid, start_0, end)].add(intron_phase(start_0, end, int(frame_txt)))
    return {k: next(iter(v)) for k, v in phases.items() if len(v) == 1}


def example_phase(example, cds_lookup: dict[tuple, int]) -> int | None:
    """Donor intron phase for a CouplingExample, or None if its internal exon is not a clean CDS.

    A missing key means the exon is not wholly a CDS in any transcript (UTR-containing or
    noncoding boundary) -- i.e. this is also the coding-vs-noncoding stratifier.
    """
    md = example.metadata
    return cds_lookup.get((example.gene_id, md["genomic_exon_start"], md["genomic_exon_end"]))
