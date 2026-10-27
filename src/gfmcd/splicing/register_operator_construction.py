"""Coordinate reconstruction for the prospectively chosen CC/max verification.

This new implementation reads historical data only. It does not import or revive
the archived exon-edge program. An independent auditor checks all reconstructed
fragments, requests and source mappings against the immutable historical cohort.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

from gfmcd.provenance import file_record

DONOR = 392
ACCEPTOR = 512
LENGTH = 1022
BRIDGE_START = 435
PROBES = {
    "donor": (64, 96, 128, 160, 192, 224, 256, 288, 320, 344, 360, 376),
    "acceptor": (528, 544, 560, 576, 608, 640, 672, 704, 736, 768, 800, 832),
}
COHORT_SHA = "478838e64a3a3eff2bca5868212852a5b833d0ccf74ff720bb15cd00cf6fd5ca"
SELECTED_IDS_SHA = "972d388cc031b6debdbb5b1c531afb7a8619630462f6bf4bb380d6a989417436"


def sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


class Genome:
    def __init__(self, path: str | Path):
        self.fasta = Fasta(str(path), as_raw=True, sequence_always_upper=True,
                           build_index=False, rebuild=False)

    def fetch(self, chrom: str, start: int, end: int, strand: str) -> str:
        if strand not in {"+", "-"} or not 0 <= start < end <= len(self.fasta[chrom]):
            raise ValueError("invalid oriented reference interval")
        seq = str(self.fasta[chrom][start:end])
        if strand == "-":
            seq = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        if len(seq) != end - start or set(seq) - set("ACGT"):
            raise ValueError("reference window contains invalid bases or length")
        return seq

    def record(self, row: dict, *, native: bool = False) -> str:
        start, end = ("window_start", "window_end") if native else ("genomic_start", "genomic_end")
        seq = self.fetch(row["chrom"], int(row[start]), int(row[end]), row["strand"])
        if sha(seq) != row["sequence_sha256"]:
            raise ValueError("reference sequence hash mismatch")
        return seq


def donor_window(source: dict, genome: Genome) -> tuple[dict, str]:
    boundary = int(source["donor_boundary"])
    start = boundary - DONOR if source["strand"] == "+" else boundary - (LENGTH - DONOR)
    seq = genome.fetch(source["chrom"], start, start + LENGTH, source["strand"])
    return {"chrom": source["chrom"], "strand": source["strand"],
            "boundary": boundary, "boundary_position": DONOR,
            "window_start": start, "window_end": start + LENGTH,
            "sequence_sha256": sha(seq)}, seq


def edit_acceptor(sequence: str, start: int, replacement: str, *, require_clean: bool = True) -> str:
    if replacement not in {"AC", "AT", "CC"} or sequence[start:start + 2] != "AG":
        raise ValueError("invalid acceptor corruption")
    edited = sequence[:start] + replacement + sequence[start + 2:]
    if require_clean:
        for position in range(start - 1, start + 2):
            if (edited[position:position + 2] in {"AG", "GT", "GC"}
                    and edited[position:position + 2] != sequence[position:position + 2]):
                raise ValueError("corruption plants an overlapping splice motif")
    return edited


def junction_translation(sequence: str, p: int) -> dict:
    junction = sequence[DONOR - p:DONOR] + sequence[ACCEPTOR:ACCEPTOR + 36 - p]
    stops = [i // 3 + 1 for i in range(0, 36, 3) if junction[i:i + 3] in {"TAA", "TAG", "TGA"}]
    return {"codon_count": 12, "first_stop_codon_1based": stops[0] if stops else None,
            "source_spliced_window_sha256": sha(junction), "stop_any": bool(stops),
            "stop_codon_indices_1based": stops, "stop_count": len(stops)}


def exact_stop_omega(counts) -> float:
    """Canonical rational contrast: equal stop loads must retain equal ranks."""
    matrix = np.asarray(counts)
    if matrix.shape != (3, 3) or not np.issubdtype(matrix.dtype, np.integer) or (matrix < 0).any():
        raise ValueError("stop load must be a nonnegative integer3x3 matrix")
    diagonal = int(np.trace(matrix))
    off = int(matrix.sum()) - diagonal
    return (off - 2 * diagonal) / 6


def reconstruct_block(block: dict, genome: Genome) -> dict:
    """Rebuild all six sources and all nine cells; check metadata before scoring."""
    output = {"record": block, "cells": {}, "native": {}, "donor_aligned": {}, "fragments": {}}
    genes = {block["bridge"]["gene_id"].split(".")[0]}
    bridge = genome.record(block["bridge"])
    if len(bridge) != 24 or bridge[0] != "A" or bridge[-1] != "T" or bridge[11:13] != "AG":
        raise ValueError("bridge geometry changed")
    if sum(bridge[i:i + 2] == "AG" for i in range(23)) != 1:
        raise ValueError("bridge has an unexpected AG")
    for role, plural in (("donor", "donors"), ("acceptor", "acceptors")):
        if set(block[plural]) != {"0", "1", "2"}:
            raise ValueError("incomplete source phases")
        for p in range(3):
            source = block[plural][str(p)]
            gene = source["gene_id"].split(".")[0]
            if gene in genes:
                raise ValueError("a block reuses a source gene")
            genes.add(gene)
            if (source["boundary_phase"] != p or source["chrom"] != block["chrom"]
                    or source["strand"] != block["strand"]):
                raise ValueError("source phase/stratum mismatch")
            for kind, length in (("donor", 424), ("acceptor", 574)):
                fragment = source[kind + "_fragment"]
                if (fragment["source_id"] != source["source_id"] or fragment["phase"] != p
                        or fragment["gene_id"] != source["gene_id"]
                        or fragment["boundary"] != source[kind + "_boundary"]):
                    raise ValueError("fragment identity mismatch")
                seq = genome.record(fragment)
                if len(seq) != length:
                    raise ValueError("fragment length mismatch")
                output["fragments"][role, p, kind] = seq
            native = genome.record(source["native_window"], native=True)
            d = int(source["native_donor_pos"])
            if len(native) != LENGTH or native[d:d + 2] != "GT" or native[510:512] != "AG":
                raise ValueError("native sites mismatch")
            output["native"][role, p] = native
            dw, aligned = donor_window(source, genome)
            if aligned[:424] != output["fragments"][role, p, "donor"]:
                raise ValueError("donor protected source mapping mismatch")
            if "donor_aligned_window" in source and source["donor_aligned_window"] != dw:
                raise ValueError("donor aligned window record changed")
            output["donor_aligned"][role, p] = aligned
            if native[448:] != output["fragments"][role, p, "acceptor"]:
                raise ValueError("acceptor protected source mapping mismatch")
    if block["chrom"] in {"chr8", "chr9", "chrY"}:
        raise ValueError("sealed or excluded chromosome")
    seen = set()
    for cell in block["cells"]:
        p, q = cell["p"], cell["q"]
        if p not in range(3) or q not in range(3) or (p, q) in seen:
            raise ValueError("duplicate/invalid phase cell")
        seen.add((p, q))
        if any(cell[k] != v for k, v in {"donor_phase": p, "acceptor_phase": q,
                                        "phase_gap": (q - p) % 3, "compatible": p == q}.items()):
            raise ValueError("misassigned phase metadata")
        sequence = output["fragments"]["donor", p, "donor"] + bridge + output["fragments"]["acceptor", q, "acceptor"]
        if len(sequence) != LENGTH or sha(sequence) != cell["sequence_sha256"]:
            raise ValueError("assembled sequence mismatch")
        if sequence[392:394] != "GT" or sequence[435:437] != "AG" or sequence[510:512] != "AG":
            raise ValueError("assembled site geometry mismatch")
        if any(sequence[s - 1:s + 1] in {"AG", "GT", "GC"} for s in (424, 448)):
            raise ValueError("seam plants splice motif")
        if junction_translation(sequence, p) != cell["junction_translation"]:
            raise ValueError("annotated-frame stop diagnostic changed")
        output["cells"][p, q] = sequence
    if len(seen) != 9:
        raise ValueError("missing phase cell")
    output["genes"] = genes
    return output


def build_construction(cohort_path: Path, selection_path: Path, fasta_path: Path) -> dict:
    raw = cohort_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != COHORT_SHA:
        raise ValueError("wrong historical cohort")
    cohort = json.loads(raw)
    selection = json.loads(selection_path.read_bytes())
    chosen = selection["block_ids"]
    ids_digest = hashlib.sha256(json.dumps(chosen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if selection["block_ids_sha256"] != SELECTED_IDS_SHA or ids_digest != SELECTED_IDS_SHA or len(chosen) != 342:
        raise ValueError("wrong approved selection")
    genome = Genome(fasta_path)
    original_blocks = []
    allgenes = set()
    omega, stopmatrices = {}, []
    for original in cohort["blocks"]:
        block = {k: original[k] for k in ("block_id", "chrom", "strand", "bridge", "stop_diagnostics")}
        for plural in ("donors", "acceptors"):
            block[plural] = {p: {k: v for k, v in source.items() if k not in {"site_covariates", "transcript_ids"}}
                             for p, source in original[plural].items()}
            for source in block[plural].values():
                source["donor_aligned_window"] = donor_window(source, genome)[0]
        block["cells"] = [{k: c[k] for k in ("p", "q", "donor_phase", "acceptor_phase", "phase_gap",
                                                    "compatible", "sequence_sha256", "junction_translation",
                                                    "source_sequence_hashes")} for c in original["cells"]]
        rebuilt = reconstruct_block(block, genome)
        if allgenes.intersection(rebuilt["genes"]):
            raise ValueError("cross-block source-gene reuse")
        allgenes.update(rebuilt["genes"])
        for cell in block["cells"]:
            p, q = cell["p"], cell["q"]
            cell.update(donor_source_id=block["donors"][str(p)]["source_id"],
                        acceptor_source_id=block["acceptors"][str(q)]["source_id"],
                        bridge_source_id=block["bridge"]["source_id"], donor_mask=DONOR, acceptor=ACCEPTOR)
            cell["protected_probes"] = []
            for role, phase in (("donor", p), ("acceptor", q)):
                source = block[role + "s"][str(phase)]
                aligned = rebuilt["donor_aligned"][role, phase] if role == "donor" else rebuilt["native"][role, phase]
                for pos in PROBES[role]:
                    base = rebuilt["cells"][p, q][pos]
                    if aligned[pos] != base:
                        raise ValueError("protected nucleotide/source mismatch")
                    cell["protected_probes"].append({"position": pos, "role": role,
                        "source_id": source["source_id"], "native_position": pos, "source_base": base})
        if block["block_id"] in chosen:
            for sequence in (*rebuilt["native"].values(), *rebuilt["cells"].values()):
                edit_acceptor(sequence, 510, "CC")
            counts = np.array([[junction_translation(rebuilt["cells"][p, q], p)["stop_count"] for q in range(3)] for p in range(3)])
            value = exact_stop_omega(counts)
            omega[block["block_id"]] = value
            stopmatrices.append(counts)
        original_blocks.append(block)
    if len(original_blocks) != 384 or set(omega) != set(chosen):
        raise ValueError("cohort membership incomplete")
    strata = Counter(f"{b['chrom']}:{b['strand']}" for b in original_blocks if b["block_id"] in chosen)
    strands = Counter(b["strand"] for b in original_blocks if b["block_id"] in chosen)
    if min(strata.values()) < 2 or strands != {"+": 171, "-": 171}:
        raise ValueError("approved stratification changed")
    return {"schema": "gfmcd.register_operator.construction.v1", "status": "CONSTRUCTED",
            "config": {**cohort["config"], "selected_blocks": 342, "span_nt": 120},
            "inputs": {k: file_record(v) for k, v in {"frozen_cohort": cohort_path,
                "selection": selection_path, "fasta": fasta_path, "fai": Path(str(fasta_path) + ".fai")}.items()},
            "original_blocks": original_blocks, "selected_block_ids": chosen,
            "stop_diagnostics": {"selected_omega_median": float(np.median(list(omega.values()))),
                "selected_omega_by_block": omega, "mean_stop_count_matrix": np.mean(stopmatrices, axis=0).tolist()},
            "model_loaded": False, "model_forward_count": 0}


@dataclass(frozen=True)
class Request:
    request_id: str
    sequence: str
    position: int
    true_base: str

    def __post_init__(self):
        if (len(self.sequence) != LENGTH or set(self.sequence) - set("ACGT") or
                not 0 <= self.position < LENGTH or self.sequence[self.position] != self.true_base):
            raise ValueError("invalid masked request")


def requests_for_blocks(blocks: list[dict], genome: Genome, *, stage: str, include_cc: bool) -> list[Request]:
    """Preserve historical insertion/dedup order and batch boundaries exactly."""
    if stage not in {"native", "cells"}:
        raise ValueError("unknown request stage")
    unique: dict[str, Request] = {}

    def add(request: Request):
        if request.request_id in unique and unique[request.request_id] != request:
            raise ValueError("request ID collision")
        unique[request.request_id] = request

    for record in blocks:
        block = reconstruct_block(record, genome)
        if stage == "native":
            for role in ("donor", "acceptor"):
                for p in range(3):
                    source = record[role + "s"][str(p)]
                    seq = block["native"][role, p]
                    key = f"native:{record['block_id']}:{role}:p{p}:{source['source_id']}"
                    d = source["native_donor_pos"]
                    add(Request(key + ":donor_intact", seq, d, "G"))
                    if include_cc:
                        add(Request(key + ":donor_true_acceptor_cc", edit_acceptor(seq, 510, "CC"), d, "G"))
                    add(Request(key + ":donor_true_acceptor_ac", edit_acceptor(seq, 510, "AC"), d, "G"))
                    add(Request(key + ":acceptor_intact", seq, 511, "G"))
            continue
        for p in range(3):
            for q in range(3):
                seq = block["cells"][p, q]
                key = f"{record['block_id']}:p{p}:q{q}"
                add(Request(key + ":donor_intact", seq, 392, "G"))
                if include_cc:
                    add(Request(key + ":donor_true_acceptor_cc", edit_acceptor(seq, 510, "CC"), 392, "G"))
                for op in ("AC", "AT"):
                    add(Request(key + ":donor_true_acceptor_" + op.lower(), edit_acceptor(seq, 510, op), 392, "G"))
                add(Request(key + ":donor_bridge_ac", edit_acceptor(seq, 435, "AC"), 392, "G"))
                add(Request(key + ":acceptor_intact", seq, 511, "G"))
                sources = []
                for role, phase in (("donor", p), ("acceptor", q)):
                    source_id = record[role + "s"][str(phase)]["source_id"]
                    source_seq = block["donor_aligned"][role, phase] if role == "donor" else block["native"][role, phase]
                    for pos in PROBES[role]:
                        base = seq[pos]
                        if source_seq[pos] != base:
                            raise ValueError("protected source coordinate mismatch")
                        add(Request(f"{key}:health:{role}:{pos}:construct", seq, pos, base))
                        sources.append(Request(f"source:{role}:{source_id}:health:{pos}", source_seq, pos, base))
                for request in sources:
                    add(request)
    return list(unique.values())
