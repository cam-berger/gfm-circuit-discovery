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


# ---------------------------------------------------------------------------------------------
# Per-base codon position (for the period-3 whole-codon masking test, thread 2a)
# ---------------------------------------------------------------------------------------------
def codon_slot(g: int, cds_start_0: int, cds_end: int, strand: str, frame: int) -> int:
    """Codon slot (0/1/2) of genomic base ``g`` inside a CDS (translation-relative offset)."""
    offset = (g - cds_start_0) if strand == "+" else (cds_end - 1 - g)
    return (offset - frame) % 3


def load_cds_intervals(path, *, allowed_chromosomes=None) -> dict[str, list[tuple]]:
    """Map ``gene_id -> [(start_0, end, strand, frame), ...]`` for every distinct CDS feature."""
    out: dict[str, list[tuple]] = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    with _open_text(path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9 or cols[2] != "CDS" or cols[7] not in ("0", "1", "2"):
                continue
            if allowed_chromosomes is not None and cols[0] not in allowed_chromosomes:
                continue
            gid = attr_first(parse_attributes(cols[8]), "gene_id")
            key = (int(cols[3]) - 1, int(cols[4]), cols[6], int(cols[7]))
            if key not in seen[gid]:
                seen[gid].add(key)
                out[gid].append(key)
    return dict(out)


def window_codon_slot(example, seq_pos: int, cds_intervals: dict[str, list[tuple]]) -> int | None:
    """Codon slot of window position ``seq_pos`` in an example, or None (non-CDS or ambiguous)."""
    md = example.metadata
    ws, we = md["genomic_window_start"], md["genomic_window_end"]
    g = (ws + seq_pos) if example.strand == "+" else (we - 1 - seq_pos)
    slots = {codon_slot(g, s0, e, strand, frame)
             for (s0, e, strand, frame) in cds_intervals.get(example.gene_id, ())
             if strand == example.strand and s0 <= g < e}
    return slots.pop() if len(slots) == 1 else None
