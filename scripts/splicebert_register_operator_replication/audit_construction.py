#!/usr/bin/env python3
"""Independently audit the approved 342-block, 120-nt operator cohort.

The auditor owns its FASTA reader, coordinate arithmetic, sequence assembly,
selection, probe mapping, translation, and pseudo-label reconstruction. It never
imports the production constructor, a model adapter, or an archived module.
The historical cohort is an explicitly supplied, hash-pinned external input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402

SCHEMA = "gfmcd.register_operator.construction.v1"
AUDIT_SCHEMA = "gfmcd.register_operator.construction_audit.v1"
COHORT_SHA256 = "478838e64a3a3eff2bca5868212852a5b833d0ccf74ff720bb15cd00cf6fd5ca"
SELECTED_IDS_SHA256 = "972d388cc031b6debdbb5b1c531afb7a8619630462f6bf4bb380d6a989417436"
FASTA_SHA256 = "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51"
FAI_SHA256 = "3b425de206296a5c8053023fa5ca61da43cfe78c1737c12e58c83367c7e83c21"
CONFIG = {"sequence_length": 1022, "donor_pos": 392, "acceptor_pos": 512,
          "donor_intron_flank": 32, "acceptor_intron_flank": 64, "bridge_decoy_start": 11}
DONOR_PROBES = (64, 96, 128, 160, 192, 224, 256, 288, 320, 344, 360, 376)
ACCEPTOR_PROBES = (528, 544, 560, 576, 608, 640, 672, 704, 736, 768, 800, 832)
PSEUDO_SEEDS = tuple(range(2026080401, 2026080500))
ALLOWED_CHROMS = ({f"chr{i}" for i in range(1, 23)} | {"chrX"}) - {"chr8", "chr9"}
SALT = "cc-maxmargin-frozen-c43-hygiene-v1"
TRIM_NAMESPACE = "whole-block-trim-to-171-per-strand"
FRAGMENT_FIELDS = ("source_id", "gene_id", "kind", "phase", "chrom", "strand",
                   "genomic_start", "genomic_end", "boundary", "sequence_sha256")
SOURCE_FIELDS = ("source_id", "gene_id", "chrom", "strand", "boundary_phase",
                 "intron_length", "donor_boundary", "acceptor_boundary", "native_donor_pos",
                 "upstream", "downstream", "native_window", "donor_fragment", "acceptor_fragment")
CELL_FIELDS = ("p", "q", "donor_phase", "acceptor_phase", "phase_gap", "compatible",
               "sequence_sha256", "source_sequence_hashes", "junction_translation")


class AuditFailure(ValueError):
    """A reconstruction or conformance check failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _integer(value: Any, label: str) -> int:
    require(type(value) is int, f"{label}: expected an integer")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    require(type(actual) is type(expected), f"{label}: type mismatch")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"{label}: keys mismatch")
        for key, value in expected.items():
            _equal(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{label}: length mismatch")
        for index, value in enumerate(expected):
            _equal(actual[index], value, f"{label}[{index}]")
    else:
        require(actual == expected, f"{label}: mismatch")


def _numbers_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{label}: keys")
        for key in expected:
            _numbers_equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{label}: length")
        for i, value in enumerate(expected):
            _numbers_equal(actual[i], value, f"{label}[{i}]")
    elif type(expected) is float:
        require(type(actual) in {int, float} and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), f"{label}: number")
    else:
        _equal(actual, expected, label)


def _read_json(path: str | Path) -> tuple[dict[str, Any], bytes]:
    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    payload = gzip.decompress(raw) if source.suffix == ".gz" else raw
    value = json.loads(payload)
    require(isinstance(value, dict), f"{source}: expected an object")
    return value, payload


class AuditFasta:
    """Auditor-owned indexed FASTA access with no model dependencies."""

    def __init__(self, fasta: str | Path, fai: str | Path):
        self.path = Path(fasta).expanduser().resolve(strict=True)
        self.records: dict[str, tuple[int, int, int, int]] = {}
        for line in Path(fai).read_text().splitlines():
            fields = line.split("\t")
            require(len(fields) >= 5, "FAI: fewer than five fields")
            name = fields[0]
            length, offset, bases, width = (int(x) for x in fields[1:5])
            require(name not in self.records and length > 0 and offset >= 0
                    and bases > 0 and width >= bases, "FAI: invalid or duplicate record")
            self.records[name] = (length, offset, bases, width)
        require(bool(self.records), "FAI: empty")

    def fetch(self, chrom: str, start: int, end: int) -> str:
        require(chrom in self.records, f"FASTA: absent chromosome {chrom}")
        length, offset, bases, width = self.records[chrom]
        require(0 <= start < end <= length, "FASTA: interval outside chromosome")
        start_line, start_offset = divmod(start, bases)
        end_line, end_offset = divmod(end - 1, bases)
        first = offset + start_line * width + start_offset
        last = offset + end_line * width + end_offset
        with self.path.open("rb") as handle:
            handle.seek(first)
            raw = handle.read(last - first + 1)
        sequence = raw.replace(b"\n", b"").replace(b"\r", b"").decode("ascii").upper()
        require(len(sequence) == end - start and not set(sequence) - set("ACGT"),
                "FASTA: incomplete or ambiguous sequence")
        return sequence


def oriented(genome: Any, chrom: str, start: int, end: int, strand: str) -> str:
    require(strand in {"+", "-"}, "source: invalid strand")
    sequence = genome.fetch(chrom, start, end)
    if strand == "-":
        sequence = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    return sequence


def aligned_window(genome: Any, chrom: str, strand: str, boundary: int,
                   position: int) -> tuple[str, dict[str, Any]]:
    if strand == "+":
        start, end = boundary - position, boundary - position + 1022
    else:
        start, end = boundary + position - 1022, boundary + position
    sequence = oriented(genome, chrom, start, end, strand)
    return sequence, {"chrom": chrom, "strand": strand, "boundary": boundary,
                      "boundary_position": position, "window_start": start,
                      "window_end": end, "sequence_sha256": sequence_hash(sequence)}


def junction_translation(sequence: str, phase: int) -> dict[str, Any]:
    require(phase in {0, 1, 2} and len(sequence) == 1022, "translation: invalid phase/length")
    joined = sequence[392 - phase:392] + sequence[512:548 - phase]
    require(len(joined) == 36, "translation: not twelve codons")
    stops = [i + 1 for i in range(12) if joined[3 * i:3 * i + 3] in {"TAA", "TAG", "TGA"}]
    return {"codon_count": 12, "stop_count": len(stops), "stop_any": bool(stops),
            "first_stop_codon_1based": stops[0] if stops else None,
            "stop_codon_indices_1based": stops,
            "source_spliced_window_sha256": sequence_hash(joined)}


def block_stops(cells: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {(r["p"], r["q"]): r["junction_translation"]["stop_count"] for r in cells}
    require(len(cells) == 9 and set(counts) == {(p, q) for p in range(3) for q in range(3)},
            "stop matrix: incomplete lattice")
    diagonal = statistics.fmean(counts[p, p] for p in range(3))
    plus = statistics.fmean(counts[p, (p + 1) % 3] for p in range(3))
    minus = statistics.fmean(counts[p, (p - 1) % 3] for p in range(3))
    return {"stop_count_matrix": [[counts[p, q] for q in range(3)] for p in range(3)],
            "diagonal_mean": diagonal, "plus_mismatch_mean": plus, "minus_mismatch_mean": minus,
            "omega_mismatch_minus_diagonal": 0.5 * (plus + minus) - diagonal}


def canonical_stop_omega(matrix: list[list[int]]) -> float:
    """Use an exact integer contrast so mathematically equal stop loads stay tied."""
    require(len(matrix) == 3 and all(len(row) == 3 for row in matrix),
            "canonical stop contrast: expected a three by three matrix")
    require(all(type(value) is int and 0 <= value <= 12 for row in matrix for value in row),
            "canonical stop contrast: expected counts from zero to twelve")
    diagonal_sum = sum(matrix[p][p] for p in range(3))
    off_diagonal_sum = sum(matrix[p][q] for p in range(3) for q in range(3) if p != q)
    return (off_diagonal_sum - 2 * diagonal_sum) / 6


def _source_reference(source: Mapping[str, Any], phase: int, genome: Any) -> dict[str, Any]:
    chrom, strand, gene = source["chrom"], source["strand"], source["gene_id"]
    upstream, downstream = source["upstream"], source["downstream"]
    us, ue, ds, de = (upstream["start"], upstream["end"], downstream["start"], downstream["end"])
    require(ue > us and de > ds, "source: invalid exon interval")
    donor, acceptor = (ue, ds) if strand == "+" else (us, de)
    distance = acceptor - donor if strand == "+" else donor - acceptor
    _equal(source["source_id"], f"{chrom}:{us}-{ue}:{ds}-{de}:{strand}:{gene}", "source identity")
    for field, expected in (("boundary_phase", phase), ("donor_boundary", donor),
                            ("acceptor_boundary", acceptor), ("intron_length", distance),
                            ("native_donor_pos", 512 - distance)):
        _equal(source[field], expected, f"source.{field}")
    require(64 <= distance <= 512, "source: native geometry outside C43 domain")
    for exon in (upstream, downstream):
        frame = _integer(exon["frame"], "GTF frame")
        require(frame in {0, 1, 2}, "GTF frame outside 0..2")
        _equal(exon["start_phase"], (3 - frame) % 3, "CDS start phase")
        _equal(exon["end_phase"], (3 - frame + exon["end"] - exon["start"]) % 3, "CDS end phase")
    require(upstream["end_phase"] == phase == downstream["start_phase"], "source: phase invariant")
    donor_seq, donor_window = aligned_window(genome, chrom, strand, donor, 392)
    native_seq, native_window = aligned_window(genome, chrom, strand, acceptor, 512)
    _equal(dict(source["native_window"]), native_window, "native window coordinates/hash")
    require(donor_seq[392:394] == "GT" and native_seq[510:512] == "AG"
            and native_seq[512 - distance:514 - distance] == "GT", "source: canonical sites")
    fragments = {}
    for kind, sequence, window, lo, hi, boundary in (
        ("donor", donor_seq, donor_window, 0, 424, donor),
        ("acceptor", native_seq, native_window, 448, 1022, acceptor),
    ):
        start, end = ((window["window_start"] + lo, window["window_start"] + hi)
                      if strand == "+" else (window["window_end"] - hi, window["window_end"] - lo))
        expected = {"source_id": source["source_id"], "gene_id": gene, "kind": kind,
                    "phase": phase, "chrom": chrom, "strand": strand,
                    "genomic_start": start, "genomic_end": end, "boundary": boundary,
                    "sequence_sha256": sequence_hash(sequence[lo:hi])}
        actual = source[f"{kind}_fragment"]
        for key in FRAGMENT_FIELDS:
            _equal(actual[key], expected[key], f"{kind} fragment.{key}")
        _equal(oriented(genome, chrom, start, end, strand), sequence[lo:hi], "fragment coordinate mapping")
        fragments[kind] = sequence[lo:hi]
    return {"donor_sequence": donor_seq, "native_sequence": native_seq,
            "donor_aligned_window": donor_window, "fragments": fragments,
            "cc_unsafe": native_seq[509] == "G"}


def reconstruct_block(block: Mapping[str, Any], genome: Any) -> dict[str, Any]:
    """Reconstruct and validate one frozen block without the producer's helpers."""
    chrom, strand = block["chrom"], block["strand"]
    require(chrom in ALLOWED_CHROMS and strand in {"+", "-"}, "block: forbidden chromosome/strand")
    sources = {}
    genes = []
    for role in ("donors", "acceptors"):
        require(set(block[role]) == {"0", "1", "2"}, "block: incomplete source phases")
        for phase in range(3):
            row = block[role][str(phase)]
            require(row["chrom"] == chrom and row["strand"] == strand, "source/block stratum mismatch")
            genes.append(row["gene_id"].split(".", 1)[0])
            sources[role, phase] = _source_reference(row, phase, genome)
    bridge = block["bridge"]
    require(bridge["chrom"] == chrom and bridge["strand"] == strand
            and bridge["kind"] == "bridge" and bridge["phase"] is None
            and bridge["boundary"] is None, "bridge: identity or stratum")
    start, end = bridge["genomic_start"], bridge["genomic_end"]
    middle = oriented(genome, chrom, start, end, strand)
    require(end - start == 24 and middle[0] == "A" and middle[-1] == "T"
            and middle[11:13] == "AG" and middle.count("AG") == 1, "bridge: geometry/motifs")
    _equal(sequence_hash(middle), bridge["sequence_sha256"], "bridge sequence hash")
    ins, ine = bridge["source_intron"]
    require(ins + 8 <= start < end <= ine - 8, "bridge: intron clearance")
    _equal(bridge["source_id"], f"{chrom}:{ins}-{ine}:{strand}:{bridge['gene_id']}:bridge", "bridge source identity")
    genes.append(bridge["gene_id"].split(".", 1)[0])
    require(len(set(genes)) == 7, "block: source genes are not distinct")
    original_cells = block["cells"]
    require(len(original_cells) == 9, "block: expected nine cells")
    by_pair = {(r["p"], r["q"]): r for r in original_cells}
    require(set(by_pair) == {(p, q) for p in range(3) for q in range(3)}, "block: duplicate/missing phase cell")
    cells = []
    for p in range(3):
        for q in range(3):
            row = by_pair[p, q]
            for key, value in (("donor_phase", p), ("acceptor_phase", q), ("phase_gap", (q-p) % 3), ("compatible", p == q)):
                _equal(row[key], value, f"cell.{key}")
            d, a = block["donors"][str(p)], block["acceptors"][str(q)]
            sequence = sources["donors", p]["fragments"]["donor"] + middle + sources["acceptors", q]["fragments"]["acceptor"]
            require(len(sequence) == 1022 and sequence[392:394] == "GT"
                    and sequence[510:512] == "AG" and sequence[435:437] == "AG", "cell: geometry or motif")
            require(sequence[423:425] not in {"AG", "GT", "GC"}
                    and sequence[447:449] not in {"AG", "GT", "GC"}, "cell: splice motif across seam")
            _equal(sequence_hash(sequence), row["sequence_sha256"], "cell sequence hash")
            _equal(row["source_sequence_hashes"], {"donor_fragment_sha256": d["donor_fragment"]["sequence_sha256"],
                   "acceptor_fragment_sha256": a["acceptor_fragment"]["sequence_sha256"], "bridge_sha256": bridge["sequence_sha256"]}, "cell fragment identity")
            translated = junction_translation(sequence, p)
            _equal(row["junction_translation"], translated, "cell junction translation")
            probes = []
            for role, source, native, positions in (
                ("donor", d, sources["donors", p]["donor_sequence"], DONOR_PROBES),
                ("acceptor", a, sources["acceptors", q]["native_sequence"], ACCEPTOR_PROBES),
            ):
                for position in positions:
                    require(sequence[position] == native[position], "protected probe: source base mapping")
                    probes.append({"position": position, "role": role, "source_id": source["source_id"],
                                   "native_position": position, "source_base": sequence[position]})
            cells.append({**{key: row[key] for key in CELL_FIELDS}, "donor_source_id": d["source_id"],
                          "acceptor_source_id": a["source_id"], "bridge_source_id": bridge["source_id"],
                          "donor_mask": 392, "acceptor": 512, "protected_probes": probes})
    stops = block_stops(cells)
    _numbers_equal(block["stop_diagnostics"], stops, "block stop diagnostics")
    return {"block_id": block["block_id"], "chrom": chrom, "strand": strand,
            "genes": genes, "sources": sources, "cells": cells, "stop_diagnostics": stops,
            "cc_unsafe": any(r["cc_unsafe"] for r in sources.values())}


def verify_production_block(actual: Mapping[str, Any], original: Mapping[str, Any],
                            reconstructed: Mapping[str, Any]) -> None:
    for key in ("block_id", "chrom", "strand"):
        _equal(actual[key], original[key], f"production block.{key}")
    for role in ("donors", "acceptors"):
        require(set(actual[role]) == {"0", "1", "2"}, "production source phases")
        for phase in range(3):
            src = actual[role][str(phase)]
            for key in SOURCE_FIELDS:
                _equal(src[key], original[role][str(phase)][key], f"production source.{key}")
            aligned = src["donor_aligned_window"]
            expected = reconstructed["sources"][role, phase]["donor_aligned_window"]
            for key in ("chrom", "strand", "window_start", "window_end", "sequence_sha256"):
                _equal(aligned[key], expected[key], f"production donor aligned window.{key}")
            for key in ("boundary", "boundary_position"):
                if key in aligned:
                    _equal(aligned[key], expected[key], f"production donor aligned window.{key}")
    for key in (*FRAGMENT_FIELDS, "source_intron"):
        _equal(actual["bridge"][key], original["bridge"][key], f"production bridge.{key}")
    require(len(actual["cells"]) == 9, "production cells: expected nine")
    cells = {(r["p"], r["q"]): r for r in actual["cells"]}
    require(len(cells) == 9, "production cells: duplicate phase pair")
    for expected in reconstructed["cells"]:
        row = cells[expected["p"], expected["q"]]
        for key, value in expected.items():
            _equal(row[key], value, f"production cell.{key}")
    _numbers_equal(actual["stop_diagnostics"], reconstructed["stop_diagnostics"], "production block stop diagnostics")


def derive_selection(reconstructed: list[dict[str, Any]], *, target_per_strand: int = 171) -> list[str]:
    pool = [r for r in reconstructed if not r["cc_unsafe"]]
    key = lambda r: (r["chrom"], r["strand"])
    counts = Counter(key(r) for r in pool)
    pool = [r for r in pool if counts[key(r)] >= 2]
    counts = Counter(key(r) for r in pool)
    selected = []
    for strand in ("+", "-"):
        part = [r for r in pool if r["strand"] == strand]
        require(len(part) >= target_per_strand, "selection: insufficient strand capacity")
        removed = set()
        def rank(row: Mapping[str, Any]) -> tuple[str, str]:
            text = row["block_id"]
            return hashlib.sha256(f"{SALT}\0{TRIM_NAMESPACE}\0{text}".encode()).hexdigest(), text
        for row in sorted(part, key=rank):
            if len(part) - len(removed) == target_per_strand:
                break
            if counts[key(row)] <= 2:
                continue
            removed.add(row["block_id"])
            counts[key(row)] -= 1
        keep = [r["block_id"] for r in part if r["block_id"] not in removed]
        require(len(keep) == target_per_strand, "selection: minimum stratum size prevents trimming")
        selected.extend(keep)
    return sorted(selected)


def pseudo_assignments(block_ids: list[str], seed: int) -> list[dict[str, Any]]:
    require(len(block_ids) > 0 and len(block_ids) % 6 == 0 and len(set(block_ids)) == len(block_ids),
            "pseudo labels: block count must be unique and divisible by six")
    ordered = sorted(block_ids, key=lambda b: (hashlib.sha256(f"crossed-phase-null-v1\0{seed}\0{b}".encode()).hexdigest(), b))
    choices = list(permutations(range(3)))
    assigned = {b: list(choices[i // (len(block_ids) // 6)]) for i, b in enumerate(ordered)}
    return [{"block_id": b, "acceptor_label_permutation": assigned[b]} for b in block_ids]


def _audit_documents(package: Mapping[str, Any], historical: Mapping[str, Any],
                     selection: Mapping[str, Any], genome: Any) -> dict[str, Any]:
    _equal(package.get("schema"), SCHEMA, "production schema")
    for key, value in CONFIG.items():
        _equal(package["config"][key], value, f"config.{key}")
    for key, value in (("span_nt", 120), ("token_length", 1024)):
        if key in package["config"]:
            _equal(package["config"][key], value, f"config.{key}")
    original = historical["blocks"]
    require(len(original) == 384 and len({b["block_id"] for b in original}) == 384, "original: expected 384 unique blocks")
    require(len(package["original_blocks"]) == 384, "production: expected 384 original blocks")
    produced = {b["block_id"]: b for b in package["original_blocks"]}
    require(set(produced) == {b["block_id"] for b in original}, "production: duplicate/missing original block")
    reconstructed = []
    for block in original:
        ref = reconstruct_block(block, genome)
        verify_production_block(produced[block["block_id"]], block, ref)
        reconstructed.append(ref)
    genes = [gene for row in reconstructed for gene in row["genes"]]
    require(len(genes) == len(set(genes)) == 2688, "original: global gene reuse")
    ids = derive_selection(reconstructed)
    _equal(digest(ids), SELECTED_IDS_SHA256, "rederived selected ID hash")
    _equal(selection["block_ids"], ids, "approved selection IDs")
    _equal(selection["block_ids_sha256"], SELECTED_IDS_SHA256, "approved selection ID hash")
    _equal(package["selected_block_ids"], ids, "production selected IDs")
    selected = [r for r in reconstructed if r["block_id"] in set(ids)]
    require(len(selected) == 342 and all(not r["cc_unsafe"] for r in selected), "selected: count or CC hygiene")
    strata = Counter(f"{r['chrom']}:{r['strand']}" for r in selected)
    require(min(strata.values()) >= 2, "selected: singleton stratum")
    _equal(dict(Counter(r["strand"] for r in selected)), {"+": 171, "-": 171}, "selected strand counts")
    omega = {r["block_id"]: canonical_stop_omega(r["stop_diagnostics"]["stop_count_matrix"])
             for r in selected}
    stop_summary = {"selected_omega_median": float(statistics.median(omega.values())),
                    "selected_omega_by_block": omega,
                    "mean_stop_count_matrix": [[statistics.fmean(r["stop_diagnostics"]["stop_count_matrix"][p][q] for r in selected) for q in range(3)] for p in range(3)]}
    _numbers_equal(package["stop_diagnostics"], stop_summary, "selected stop diagnostics")
    # Floating subtraction can split a mathematically exact median tie. Tolerance
    # is appropriate for the historical summaries, but not these frozen ranks.
    _equal(package["stop_diagnostics"]["selected_omega_by_block"], omega,
           "selected canonical stop contrasts")
    _equal(package["stop_diagnostics"]["selected_omega_median"],
           stop_summary["selected_omega_median"], "selected canonical stop median")
    old_ids = sorted(r["block_id"] for r in reconstructed)
    family = historical["pseudo_phase_family"]["roles"]["replication"]
    _equal(family["block_ids"], old_ids, "historical pseudo block IDs")
    new_hashes = {}
    for seed in PSEUDO_SEEDS:
        old_assignments = pseudo_assignments(old_ids, seed)
        stored = family["assignments_by_seed"][str(seed)]
        _equal(stored["assignments"], old_assignments, f"historical pseudo seed {seed}")
        _equal(stored["assignment_sha256"], digest(old_assignments), "historical pseudo assignment hash")
        new_hashes[str(seed)] = digest(pseudo_assignments(ids, seed))
    new_family = {"seeds": list(PSEUDO_SEEDS), "assignment_sha256_by_seed": new_hashes}
    if "pseudo_phase_family" in package:
        _equal(package["pseudo_phase_family"], new_family, "production pseudo family")
    return {"historical_blocks": 384, "historical_cells_reconstructed": 3456,
            "historical_native_windows_reconstructed": 2304, "historical_fragment_hashes_reconstructed": 4992,
            "historical_donor_aligned_windows_reconstructed": 2304,
            "historical_protected_probe_pairs_checked": 82944, "historical_source_genes": 2688,
            "selected_blocks": 342, "selected_cells": 3078, "selected_source_genes": 2394,
            "selected_block_ids_sha256": digest(ids), "selected_strata": dict(sorted(strata.items())),
            "cc_unsafe_original_blocks": sorted(r["block_id"] for r in reconstructed if r["cc_unsafe"]),
            "selected_stop_diagnostics": stop_summary, "historical_pseudo_families_reconstructed": 99,
            "selected_pseudo_phase_family": new_family,
            "production_pseudo_family_present": "pseudo_phase_family" in package}


def audit_construction(package_path: str | Path, *, fasta_path: str | Path,
                       fai_path: str | Path, frozen_cohort_path: str | Path,
                       selection_path: str | Path) -> dict[str, Any]:
    """Strict real-cohort audit; failure is returned without any execution permission."""
    checks: dict[str, bool] = {}
    report: dict[str, Any] = {"schema": AUDIT_SCHEMA, "stage": "construction",
        "model_loaded": False, "model_outcomes_read": False, "target_execution_authorized": False,
        "checks": checks, "failures": []}
    try:
        package, _ = _read_json(package_path)
        historical, historical_bytes = _read_json(frozen_cohort_path)
        selection, _ = _read_json(selection_path)
        require(hashlib.sha256(historical_bytes).hexdigest() == COHORT_SHA256, "frozen C43 cohort SHA-256 mismatch")
        checks["frozen_historical_cohort_pin"] = True
        bindings = {"frozen_cohort": frozen_cohort_path, "selection": selection_path,
                    "fasta": fasta_path, "fai": fai_path}
        actual = {key: file_record(Path(path).expanduser().resolve(strict=True)) for key, path in bindings.items()}
        require(actual["fasta"]["sha256"] == FASTA_SHA256 and actual["fai"]["sha256"] == FAI_SHA256,
                "hg38 or FAI SHA-256 mismatch")
        for key, record in actual.items():
            for field in ("sha256", "size_bytes"):
                _equal(package["inputs"][key][field], record[field], f"input binding {key}.{field}")
        checks["all_reference_input_bindings"] = True
        report["input_bindings"] = actual
        report["reconstruction"] = _audit_documents(package, historical, selection, AuditFasta(fasta_path, fai_path))
        checks.update({name: True for name in ("complete_historical_reconstruction", "source_identity_and_phase",
            "global_gene_disjointness", "approved342_selection", "CC_native_and_target_hygiene",
            "geometry_seams_and_masks", "protected_probe_mapping", "stop_codon_diagnostics",
            "historical_pseudo_family", "production_schema_conformance")})
    except (AuditFailure, KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
    report["passed"] = not report["failures"]
    report["status"] = "AUDIT_PASS" if report["passed"] else "AUDIT_FAIL"
    return report


def write_audit(*, package_path: str | Path, fasta_path: str | Path, fai_path: str | Path,
                frozen_cohort_path: str | Path, selection_path: str | Path,
                output_path: str | Path, command: list[str] | None = None) -> tuple[Path, Path, dict[str, Any]]:
    output = Path(output_path).expanduser().absolute()
    require(output.suffix == ".json", "audit output must end in .json")
    manifest_path = output.with_suffix(".manifest.json")
    for path in (output, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite immutable audit artifact: {path}")
    report = audit_construction(package_path, fasta_path=fasta_path, fai_path=fai_path,
                               frozen_cohort_path=frozen_cohort_path, selection_path=selection_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    manifest = ExperimentManifest.create("splicebert-register-operator-replication:construction-audit-v1",
        command if command is not None else sys.argv, repo=REPO, status=report["status"],
        config={"stage": "construction", "original_blocks": 384, "selected_blocks": 342, "span_nt": 120},
        notes=["Independent model-free reference reconstruction; no archived module or producer constructor imported.",
               "Construction audit alone never authorizes model execution."])
    for path in (package_path, frozen_cohort_path, selection_path, fasta_path, fai_path, Path(__file__), REPO / "src/gfmcd/provenance.py"):
        manifest.add_input(path)
    manifest.add_output(output)
    with manifest_path.open("x") as handle:
        json.dump(manifest.to_dict(), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return output, manifest_path, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--fai", required=True)
    parser.add_argument("--frozen-cohort", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output, manifest, report = write_audit(package_path=args.package, fasta_path=args.fasta,
        fai_path=args.fai, frozen_cohort_path=args.frozen_cohort, selection_path=args.selection,
        output_path=args.out)
    print(json.dumps({"status": report["status"], "passed": report["passed"],
                      "audit": str(output), "manifest": str(manifest), "failures": report["failures"]}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
