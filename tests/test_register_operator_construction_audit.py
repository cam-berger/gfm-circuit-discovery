"""The construction gate must reject plausible coordinate and identity defects."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "register_operator_construction_auditor",
    REPO / "scripts/splicebert_register_operator_replication/audit_construction.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def sha(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def rc(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


class Genome:
    def __init__(self, sequence):
        self.sequence = sequence

    def fetch(self, chrom, start, end):
        assert chrom == "chr1" and 0 <= start < end <= len(self.sequence)
        return self.sequence[start:end]


def window(genome, strand, boundary, position):
    start = boundary - position if strand == "+" else boundary + position - 1022
    end = start + 1022
    sequence = genome.fetch("chr1", start, end)
    if strand == "-":
        sequence = rc(sequence)
    return sequence, {"chrom": "chr1", "strand": strand, "boundary": boundary,
                      "boundary_position": position, "window_start": start,
                      "window_end": end, "sequence_sha256": sha(sequence)}


def translation(donor, acceptor, phase):
    # Twelve codons beginning in the authentic upstream register, including its
    # residual bases. This fixture has varied native intron lengths intentionally.
    spliced = (donor[-phase:] if phase else "") + acceptor[:36-phase]
    codons = [spliced[i:i+3] for i in range(0, 36, 3)]
    stops = [i for i, codon in enumerate(codons, 1) if codon in ("TAA", "TAG", "TGA")]
    return {"codon_count": 12, "stop_count": len(stops), "stop_any": bool(stops),
            "first_stop_codon_1based": stops[0] if stops else None,
            "stop_codon_indices_1based": stops,
            "source_spliced_window_sha256": sha(spliced)}


def diagnostics(cells):
    matrix = [[next(c["junction_translation"]["stop_count"] for c in cells
                    if c["p"] == p and c["q"] == q) for q in range(3)] for p in range(3)]
    diagonal = sum(matrix[i][i] for i in range(3))/3
    plus = sum(matrix[i][(i+1)%3] for i in range(3))/3
    minus = sum(matrix[i][(i-1)%3] for i in range(3))/3
    return {"stop_count_matrix": matrix, "diagonal_mean": diagonal,
            "plus_mismatch_mean": plus, "minus_mismatch_mean": minus,
            "omega_mismatch_minus_diagonal": (plus+minus)/2-diagonal}


def make_block(strand="+", unsafe=None, duplicate_gene=False):
    bases = list("C" * 17000)
    coordinate_rows = []
    for index in range(6):
        role, phase = ("donors" if index < 3 else "acceptors"), index % 3
        start, distance = 1000 + index * 2000, 90 + index * 9
        oriented = list("C" * 1600)
        oriented[600:602] = "GT"
        oriented[600+distance-2:600+distance] = "AG"
        oriented[600+distance:600+distance+36] = "TAACCG" * 6
        if unsafe == (role, phase):
            oriented[600+distance-3] = "G"
        text = "".join(oriented)
        bases[start:start+1600] = text if strand == "+" else rc(text)
        donor = start + (600 if strand == "+" else 1000)
        acceptor = donor + (distance if strand == "+" else -distance)
        gene_index = 0 if duplicate_gene and index == 5 else index
        coordinate_rows.append((role, phase, donor, acceptor, distance, f"GENE{gene_index}"))
    bridge_text = "A" + "C"*10 + "AG" + "C"*10 + "T"
    bases[14500:14524] = bridge_text if strand == "+" else rc(bridge_text)
    genome = Genome("".join(bases))
    block = {"block_id": f"fixture:chr1:{strand}:b000", "chrom": "chr1", "strand": strand,
             "donors": {}, "acceptors": {}, "cells": []}
    sequences = {}
    for role, phase, donor, acceptor, distance, gene in coordinate_rows:
        us, ue = (donor-300-phase, donor) if strand == "+" else (donor, donor+300+phase)
        ds, de = (acceptor, acceptor+300) if strand == "+" else (acceptor-300, acceptor)
        source_id = f"chr1:{us}-{ue}:{ds}-{de}:{strand}:{gene}"
        upstream = {"start": us, "end": ue, "frame": 0, "start_phase": 0, "end_phase": phase}
        downstream = {"start": ds, "end": de, "frame": (3-phase)%3,
                      "start_phase": phase, "end_phase": phase}
        dseq, dwin = window(genome, strand, donor, 392)
        aseq, awin = window(genome, strand, acceptor, 512)
        row = {"source_id": source_id, "gene_id": gene, "chrom": "chr1", "strand": strand,
               "boundary_phase": phase, "intron_length": distance,
               "donor_boundary": donor, "acceptor_boundary": acceptor,
               "native_donor_pos": 512-distance, "upstream": upstream, "downstream": downstream,
               "native_window": awin}
        for kind, seq, win, lo, hi, boundary in (("donor", dseq, dwin, 0, 424, donor),
                                               ("acceptor", aseq, awin, 448, 1022, acceptor)):
            gs, ge = ((win["window_start"]+lo, win["window_start"]+hi) if strand == "+"
                      else (win["window_end"]-hi, win["window_end"]-lo))
            row[kind+"_fragment"] = {"source_id": source_id, "gene_id": gene, "kind": kind,
                "phase": phase, "chrom": "chr1", "strand": strand, "genomic_start": gs,
                "genomic_end": ge, "boundary": boundary, "sequence_sha256": sha(seq[lo:hi])}
        block[role][str(phase)] = row
        sequences[role, phase] = dseq, aseq, dwin
    bridge = {"source_id": f"chr1:14400-14600:{strand}:GENE6:bridge", "gene_id": "GENE6",
              "kind": "bridge", "phase": None, "chrom": "chr1", "strand": strand,
              "genomic_start": 14500, "genomic_end": 14524, "boundary": None,
              "source_intron": [14400, 14600], "sequence_sha256": sha(bridge_text)}
    block["bridge"] = bridge
    for p in range(3):
        for q in range(3):
            dseq, aseq = sequences["donors", p][0], sequences["acceptors", q][1]
            assembled = dseq[:424] + bridge_text + aseq[448:]
            block["cells"].append({"p": p, "q": q, "donor_phase": p, "acceptor_phase": q,
                "phase_gap": (q-p)%3, "compatible": p == q, "sequence_sha256": sha(assembled),
                "source_sequence_hashes": {
                    "donor_fragment_sha256": sha(dseq[:424]),
                    "acceptor_fragment_sha256": sha(aseq[448:]), "bridge_sha256": sha(bridge_text)},
                "junction_translation": translation(dseq[:392], aseq[512:], p)})
    block["stop_diagnostics"] = diagnostics(block["cells"])
    produced = copy.deepcopy(block)
    for role in ("donors", "acceptors"):
        for p in range(3):
            produced[role][str(p)]["donor_aligned_window"] = sequences[role, p][2]
    for cell in produced["cells"]:
        p, q = cell["p"], cell["q"]
        d, a = block["donors"][str(p)], block["acceptors"][str(q)]
        cell.update(donor_source_id=d["source_id"], acceptor_source_id=a["source_id"],
                    bridge_source_id=bridge["source_id"], donor_mask=392, acceptor=512)
        probes = []
        for role, src, native, positions in (
            ("donor", d, sequences["donors", p][0], (64,96,128,160,192,224,256,288,320,344,360,376)),
            ("acceptor", a, sequences["acceptors", q][1], (528,544,560,576,608,640,672,704,736,768,800,832)),
        ):
            probes.extend({"position": pos, "role": role, "source_id": src["source_id"],
                           "native_position": pos, "source_base": native[pos]} for pos in positions)
        cell["protected_probes"] = probes
    return genome, block, produced


@pytest.mark.parametrize("strand", ["+", "-"])
def test_complete_reference_and_production_reconstruction_on_both_strands(strand):
    genome, block, produced = make_block(strand)
    ref = audit.reconstruct_block(block, genome)
    audit.verify_production_block(produced, block, ref)
    assert len(ref["cells"]) == 9 and len(set(ref["genes"])) == 7
    assert not ref["cc_unsafe"]
    assert all(len(cell["protected_probes"]) == 24 for cell in ref["cells"])


@pytest.mark.parametrize("role", ["donors", "acceptors"])
@pytest.mark.parametrize("strand", ["+", "-"])
def test_cc_hygiene_checks_native_acceptors_for_both_roles(role, strand):
    genome, block, _ = make_block(strand, unsafe=(role, 1))
    assert audit.reconstruct_block(block, genome)["cc_unsafe"]


@pytest.mark.parametrize("defect, message", [
    ("mask", "donor_mask"), ("probe_position", "protected_probes"),
    ("probe_source", "protected_probes"), ("probe_base", "protected_probes"),
    ("probe_coordinate_type", "protected_probes"), ("phase_type", "upstream.frame"),
    ("donor_window", "donor aligned window"), ("source_identity", "donor_source_id"),
    ("fragment_identity", "source_sequence_hashes"), ("stop", "junction_translation"),
    ("missing_cell", "expected nine"), ("duplicate_cell", "duplicate phase pair"),
])
def test_production_mapping_gate_rejects_plausible_defects(defect, message):
    genome, original, broken = make_block("-")
    ref = audit.reconstruct_block(original, genome)
    cell = broken["cells"][0]
    if defect == "mask":
        cell["donor_mask"] = 393
    elif defect == "probe_position":
        cell["protected_probes"][0]["native_position"] += 1
    elif defect == "probe_source":
        cell["protected_probes"][0]["source_id"] = cell["acceptor_source_id"]
    elif defect == "probe_base":
        cell["protected_probes"][0]["source_base"] = "A"
    elif defect == "probe_coordinate_type":
        cell["protected_probes"][0]["native_position"] = 64.0
    elif defect == "phase_type":
        broken["donors"]["0"]["upstream"]["frame"] = False
    elif defect == "donor_window":
        src = broken["donors"]["0"]
        src["donor_aligned_window"] = src["native_window"]
    elif defect == "source_identity":
        cell["donor_source_id"] = broken["donors"]["1"]["source_id"]
    elif defect == "fragment_identity":
        cell["source_sequence_hashes"]["bridge_sha256"] = "0" * 64
    elif defect == "stop":
        cell["junction_translation"]["stop_count"] += 1
    elif defect == "missing_cell":
        broken["cells"].pop()
    elif defect == "duplicate_cell":
        broken["cells"][-1] = copy.deepcopy(broken["cells"][0])
    with pytest.raises(audit.AuditFailure, match=message):
        audit.verify_production_block(broken, original, ref)


@pytest.mark.parametrize("defect, message", [
    ("phase", "CDS end phase"), ("coordinate", "fragment.genomic_start"),
    ("hash", "cell sequence hash"), ("native", "native window coordinates/hash"),
    ("bridge", "bridge source identity"), ("chrom", "forbidden chromosome"),
])
def test_independent_reference_gate_rejects_defects(defect, message):
    genome, block, _ = make_block()
    if defect == "phase":
        block["donors"]["1"]["upstream"]["end_phase"] = 2
    elif defect == "coordinate":
        block["donors"]["0"]["donor_fragment"]["genomic_start"] += 1
    elif defect == "hash":
        block["cells"][0]["sequence_sha256"] = "0" * 64
    elif defect == "native":
        block["acceptors"]["2"]["native_window"]["window_start"] -= 1
    elif defect == "bridge":
        block["bridge"]["source_id"] = "wrong"
    elif defect == "chrom":
        block["chrom"] = "chr8"
    with pytest.raises(audit.AuditFailure, match=message):
        audit.reconstruct_block(block, genome)


def test_seven_coordinate_sources_still_require_seven_distinct_genes():
    genome, block, _ = make_block(duplicate_gene=True)
    with pytest.raises(audit.AuditFailure, match="source genes are not distinct"):
        audit.reconstruct_block(block, genome)


def test_junction_translation_includes_donor_residual_bases():
    sequence = list("C" * 1022)
    sequence[390:392] = "TA"
    sequence[512:515] = "ACC"
    two = audit.junction_translation("".join(sequence), 2)
    zero = audit.junction_translation("".join(sequence), 0)
    assert two["stop_codon_indices_1based"] == [1]
    assert two["codon_count"] == 12 and two["first_stop_codon_1based"] == 1
    assert zero["stop_count"] == 0


def test_stop_omega_is_six_off_diagonal_mean_minus_three_diagonal_mean():
    matrix = [[1, 3, 5], [7, 2, 9], [11, 13, 4]]
    cells = [{"p": p, "q": q, "junction_translation": {"stop_count": matrix[p][q]}}
             for p in range(3) for q in range(3)]
    result = audit.block_stops(cells)
    assert result["stop_count_matrix"] == matrix
    assert result["omega_mismatch_minus_diagonal"] == pytest.approx(8 - 7/3)


def test_chr17_minus_b007_exact_stop_tie_remains_at_the_frozen_median():
    # This is the model-free junction stop-count matrix for the approved source
    # block replication:chr17:-:b007, independently reconstructed from hg38.
    matrix = [[0, 0, 3], [2, 1, 0], [1, 0, 0]]
    old_mean_subtraction = 6 / 6 - 1 / 3
    frozen_median = 2 / 3
    assert old_mean_subtraction > frozen_median
    canonical = audit.canonical_stop_omega(matrix)
    assert canonical == frozen_median
    assert not canonical > frozen_median
    # Equal rational contrasts must also remain tied for Spearman ranks.
    same_contrast_without_diagonal_stops = [[0, 0, 3], [1, 0, 0], [0, 0, 0]]
    assert canonical == audit.canonical_stop_omega(same_contrast_without_diagonal_stops)
    with pytest.raises(audit.AuditFailure, match="canonical"):
        audit._equal({"b007": old_mean_subtraction}, {"b007": canonical},
                     "selected canonical stop contrasts")


def test_selection_removes_unsafe_and_singletons_then_hash_trims_preserving_strata():
    rows = [{"block_id": f"{chrom}:{strand}:{i}", "chrom": chrom, "strand": strand,
             "cc_unsafe": False} for strand in ("+", "-")
            for chrom, count in (("chr1", 4), ("chr2", 2)) for i in range(count)]
    rows += [{"block_id": "singleton", "chrom": "chr14", "strand": "-", "cc_unsafe": False},
             {"block_id": "unsafe", "chrom": "chr3", "strand": "+", "cc_unsafe": True}]
    ids = audit.derive_selection(rows, target_per_strand=5)
    assert len(ids) == 10 and "singleton" not in ids and "unsafe" not in ids
    assert ids == audit.derive_selection(list(reversed(rows)), target_per_strand=5)
    chosen = [r for r in rows if r["block_id"] in ids]
    assert Counter(r["strand"] for r in chosen) == {"+": 5, "-": 5}
    assert min(Counter((r["chrom"], r["strand"]) for r in chosen).values()) >= 2
    with pytest.raises(audit.AuditFailure, match="minimum stratum size"):
        audit.derive_selection(rows, target_per_strand=3)


def test_pseudo_assignment_is_exactly_balanced_and_does_not_cycle():
    ids = [f"block{i}" for i in range(12)]
    assigned = audit.pseudo_assignments(ids, 2026080401)
    assert [r["block_id"] for r in assigned] == ids
    assert set(Counter(tuple(r["acceptor_label_permutation"]) for r in assigned).values()) == {2}
    ranked = sorted(ids, key=lambda b: hashlib.sha256(f"crossed-phase-null-v1\0{2026080401}\0{b}".encode()).hexdigest())
    lookup = {r["block_id"]: r["acceptor_label_permutation"] for r in assigned}
    assert lookup[ranked[0]] == lookup[ranked[1]] == [0, 1, 2]
    assert lookup[ranked[2]] == [0, 2, 1]
    with pytest.raises(audit.AuditFailure, match="unique and divisible"):
        audit.pseudo_assignments(ids[:-1], 1)


def test_indexed_fasta_handles_line_breaks_and_reverse_strand(tmp_path):
    fasta, fai = tmp_path / "test.fa", tmp_path / "test.fa.fai"
    fasta.write_bytes(b">chr1\r\nACGTAC\r\nGGTACC\r\nTG\r\n")
    fai.write_text("chr1\t14\t7\t6\t8\n")
    reader = audit.AuditFasta(fasta, fai)
    assert reader.fetch("chr1", 4, 12) == "ACGGTACC"
    assert audit.oriented(reader, "chr1", 4, 12, "-") == "GGTACCGT"
    with pytest.raises(audit.AuditFailure, match="outside chromosome"):
        reader.fetch("chr1", 0, 15)


def test_real_audit_rejects_unpinned_historical_input_before_fasta_access(tmp_path):
    paths = [tmp_path / name for name in ("package.json", "cohort.json", "selection.json")]
    for path in paths:
        path.write_text("{}")
    result = audit.audit_construction(paths[0], frozen_cohort_path=paths[1], selection_path=paths[2],
                                     fasta_path=tmp_path / "absent.fa", fai_path=tmp_path / "absent.fai")
    assert result["status"] == "AUDIT_FAIL"
    assert result["failures"] == ["AuditFailure: frozen C43 cohort SHA-256 mismatch"]
    assert not result["model_loaded"] and not result["target_execution_authorized"]


@pytest.mark.parametrize("reserved", ["output", "manifest", "dangling_symlink"])
def test_audit_output_is_immutable_even_when_only_manifest_exists(tmp_path, reserved, monkeypatch):
    output = tmp_path / "audit.json"
    manifest = output.with_suffix(".manifest.json")
    if reserved == "output":
        output.write_text("keep")
    elif reserved == "manifest":
        manifest.write_text("keep")
    else:
        output.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(audit, "audit_construction", lambda *a, **k: pytest.fail("must guard before audit"))
    with pytest.raises(FileExistsError, match="immutable"):
        audit.write_audit(package_path="unused", fasta_path="unused", fai_path="unused",
                          frozen_cohort_path="unused", selection_path="unused", output_path=output)


def test_failed_audit_is_preserved_with_input_and_output_manifest(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text("{}")
    report = {"status": "AUDIT_FAIL", "passed": False, "failures": ["deliberate"]}
    monkeypatch.setattr(audit, "audit_construction", lambda *a, **k: report)
    output, manifest_path, observed = audit.write_audit(package_path=source, fasta_path=source,
        fai_path=source, frozen_cohort_path=source, selection_path=source,
        output_path=tmp_path / "audit.json", command=["test-audit"])
    assert observed == report == json.loads(output.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "AUDIT_FAIL" and manifest["command"] == ["test-audit"]
    assert manifest["git_commit"] and manifest["outputs"][0]["sha256"] == sha(output.read_text())
    assert any(r["path"].endswith("audit_construction.py") for r in manifest["inputs"])
