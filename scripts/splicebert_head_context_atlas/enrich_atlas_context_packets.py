#!/usr/bin/env python
"""Sequence-enrich chromosome-train atlas packets without reading model outcomes.

The annotation census includes val/test counts, but this stage deliberately constructs
genome sequences only for chromosome-train packets.  chr11/16/19 and sealed chr8/9 are
not read.  One sequence is stored per junction/window; donor and acceptor packet rows
reference that sequence and carry role-specific covariates.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.atlas_context import atlas_analysis_split  # noqa: E402
from gfmcd.splicing.dataset import PyfaidxGenome, reverse_complement  # noqa: E402
from gfmcd.splicing.specificity import acc_score, don_score  # noqa: E402

DEFAULT_PACKETS = REPO / "results/splicebert_head_context_atlas/atlas_context_packets_v1.jsonl"
DEFAULT_FASTA: str | None = None  # supply via --fasta
DEFAULT_GTF: str | None = None  # supply via --gtf
DONOR_MOTIFS = {"GT", "GC"}


def gc_fraction(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else float("nan")


def pyrimidine_fraction(seq: str) -> float:
    return (seq.count("C") + seq.count("T")) / len(seq) if seq else float("nan")


def max_homopolymer(seq: str) -> int:
    best = run = 0
    previous = None
    for base in seq:
        run = run + 1 if base == previous else 1
        best = max(best, run)
        previous = base
    return best


def finite_or_none(value: float):
    return float(value) if math.isfinite(value) else None


def parse_intervals(gtf_path: str) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """0-based half-open exon/CDS intervals by gene; annotation only."""
    intervals = defaultdict(lambda: {"CDS": [], "exon": []})
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in {"CDS", "exon"}:
                continue
            attrs = fields[8]
            marker = 'gene_id "'
            start = attrs.find(marker)
            if start < 0:
                continue
            start += len(marker)
            gene = attrs[start:attrs.find('"', start)]
            intervals[gene][fields[2]].append((int(fields[3]) - 1, int(fields[4])))
    return intervals


def in_intervals(position: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in intervals)


def boundary_region(packet: dict, intervals) -> tuple[str, bool, bool]:
    if packet["target_role"] == "donor":
        # First intron base -> adjacent exonic base on the transcript-upstream side.
        position = packet["donor"] - 1 if packet["strand"] == "+" else packet["donor"] + 1
    else:
        position = packet["acceptor"]
    gene_intervals = intervals.get(packet["gene_id"], {"CDS": [], "exon": []})
    in_cds = in_intervals(position, gene_intervals["CDS"])
    in_exon = in_intervals(position, gene_intervals["exon"])
    return ("CDS" if in_cds else "UTR_exon" if in_exon else "noncoding_or_other"), in_cds, in_exon


def fetch_sequence(packet: dict, genome: PyfaidxGenome) -> str:
    seq = genome.fetch(packet["chrom"], packet["window_start"], packet["window_end"])
    if packet["strand"] == "-":
        seq = reverse_complement(seq)
    return seq.upper().replace("U", "T")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", default=str(DEFAULT_PACKETS))
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--out", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_packets_enriched_v1.jsonl"
    ))
    ap.add_argument("--sequences", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_sequences_v1.jsonl"
    ))
    ap.add_argument("--summary", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_enrichment_v1.json"
    ))
    ap.add_argument("--limit", type=int, default=None, help="packet rows, smoke only")
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    packet_path = Path(args.packets)
    out_path = Path(args.out)
    sequence_path = Path(args.sequences)
    summary_path = Path(args.summary)
    for path in (out_path, sequence_path, summary_path, summary_path.with_suffix(".manifest.json")):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable artifact {path}")

    from maxentpy.maxent import load_matrix3, load_matrix5
    matrix5, matrix3 = load_matrix5(), load_matrix3()
    print("[atlas-enrich] parsing exon/CDS intervals", file=sys.stderr)
    intervals = parse_intervals(args.gtf)
    genome = PyfaidxGenome(args.fasta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sequence_path.parent.mkdir(parents=True, exist_ok=True)

    n_seen = n_train = n_written = n_sequences = 0
    drops = Counter()
    analysis_counts = Counter()
    category_counts = Counter()
    last_junction = None
    last_sequence = None
    last_sequence_row = None
    with packet_path.open() as source, out_path.open("w") as out, sequence_path.open("w") as seq_out:
        for line in source:
            if args.limit is not None and n_seen >= args.limit:
                break
            packet = json.loads(line)
            n_seen += 1
            if packet["split"] != "train":
                continue
            n_train += 1
            junction = packet["junction_id"]
            if junction != last_junction:
                last_junction = junction
                last_sequence = fetch_sequence(packet, genome)
                last_sequence_row = None
                if len(last_sequence) != packet["window_end"] - packet["window_start"]:
                    drops["length"] += 1
                    last_sequence = None
                elif set(last_sequence) - set("ACGT"):
                    drops["non_acgt"] += 1
                    last_sequence = None
                elif last_sequence[packet["donor_win"]:packet["donor_win"] + 2] not in DONOR_MOTIFS:
                    drops["donor_motif"] += 1
                    last_sequence = None
                elif last_sequence[packet["acceptor_win"] - 2:packet["acceptor_win"]] != "AG":
                    drops["acceptor_motif"] += 1
                    last_sequence = None
                else:
                    donor_strength = finite_or_none(don_score(last_sequence, packet["donor_win"], matrix5))
                    acceptor_strength = finite_or_none(acc_score(last_sequence, packet["acceptor_win"], matrix3))
                    if donor_strength is None or acceptor_strength is None:
                        drops["maxent_context"] += 1
                        last_sequence = None
                    else:
                        last_sequence_row = {
                            "sequence_id": junction,
                            "chrom": packet["chrom"],
                            "strand": packet["strand"],
                            "window_start": packet["window_start"],
                            "window_end": packet["window_end"],
                            "donor_win": packet["donor_win"],
                            "acceptor_win": packet["acceptor_win"],
                            "donor_strength": donor_strength,
                            "acceptor_strength": acceptor_strength,
                            "sequence": last_sequence,
                            "sequence_sha256": hashlib.sha256(last_sequence.encode()).hexdigest(),
                        }
                        seq_out.write(json.dumps(last_sequence_row, sort_keys=True) + "\n")
                        n_sequences += 1
            if last_sequence is None or last_sequence_row is None:
                continue

            target_pos = packet["target_anchor_win"]
            partner_pos = packet["partner_anchor_win"]
            target_local = last_sequence[max(0, target_pos - 25):target_pos + 25]
            partner_local = last_sequence[max(0, partner_pos - 25):partner_pos + 25]
            role = packet["target_role"]
            target_strength = (
                last_sequence_row["donor_strength"] if role == "donor"
                else last_sequence_row["acceptor_strength"]
            )
            partner_strength = (
                last_sequence_row["acceptor_strength"] if role == "donor"
                else last_sequence_row["donor_strength"]
            )
            masked_native = last_sequence[packet["masked_base_win"]]
            if masked_native != "G":
                raise RuntimeError(
                    f"coordinate gate failed for {packet['packet_id']}: masked native={masked_native!r}"
                )
            region, in_cds, in_exon = boundary_region(packet, intervals)
            analysis_split = atlas_analysis_split(packet["gene_id"], packet["split"])
            distance = packet["target_partner_distance"]
            phase_10 = 2 * math.pi * distance / 10.63
            enriched = {
                **packet,
                "sequence_id": junction,
                "analysis_split": analysis_split,
                "masked_native_base": masked_native,
                "target_strength": target_strength,
                "partner_strength": partner_strength,
                "strength_difference": target_strength - partner_strength,
                "gc_window": gc_fraction(last_sequence),
                "gc_target_local50": gc_fraction(target_local),
                "gc_partner_local50": gc_fraction(partner_local),
                "pyrimidine_target_local50": pyrimidine_fraction(target_local),
                "pyrimidine_partner_local50": pyrimidine_fraction(partner_local),
                "donor_motif_count": sum(
                    last_sequence[i:i + 2] in DONOR_MOTIFS for i in range(len(last_sequence) - 1)
                ),
                "acceptor_motif_count": sum(
                    last_sequence[i:i + 2] == "AG" for i in range(len(last_sequence) - 1)
                ),
                "max_homopolymer": max_homopolymer(last_sequence),
                "target_boundary_region": region,
                "target_in_cds": in_cds,
                "target_in_exon": in_exon,
                "distance_mod3": distance % 3,
                "distance_phase10_63_sin": math.sin(phase_10),
                "distance_phase10_63_cos": math.cos(phase_10),
                "target_position_fraction": target_pos / (len(last_sequence) - 1),
                "sequence_gate": "canonical_GT_or_GC_AG; ACGT-only; native masked base G",
            }
            out.write(json.dumps(enriched, sort_keys=True) + "\n")
            n_written += 1
            analysis_counts[analysis_split] += 1
            for tag in packet["tags"]:
                category_counts[tag] += 1

    report = {
        "schema": "head_context_atlas.sequence_enrichment/1",
        "status": "SEQUENCE_ONLY_NO_MODEL_OUTCOMES",
        "input_packet_rows_seen": n_seen,
        "chromosome_train_packet_rows": n_train,
        "valid_enriched_packets": n_written,
        "valid_unique_sequences": n_sequences,
        "dropped_unique_sequences_by_gate": dict(sorted(drops.items())),
        "analysis_split_packet_counts": dict(sorted(analysis_counts.items())),
        "category_packet_counts": dict(sorted(category_counts.items())),
        "sealed_guard": {
            "included_chromosome_split": "train only",
            "excluded": ["chr8", "chr9", "chr11", "chr16", "chr19"],
            "model_or_margin_read": False,
        },
        "covariates": [
            "target/partner MaxEnt (stratifiers only)", "window/local GC", "local pyrimidine",
            "motif burden", "homopolymer", "boundary coding region", "P3/10.63 distance phase",
            "packet position", "transcript support and graph/packet geometry from annotation",
        ],
        "deferred_gate": "proper masked-MLM health is required in the model preflight",
    }
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_sequence_enrichment_v1",
        command=["python"] + sys.argv,
        status="SEQUENCE_ONLY_NO_MODEL_OUTCOMES",
    )
    for path in (packet_path, Path(args.fasta), Path(args.gtf)):
        manifest.add_input(str(path))
    for path in (sequence_path, out_path, summary_path):
        manifest.add_output(str(path))
    manifest.write(str(summary_path.with_suffix(".manifest.json")))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {sequence_path}\nWrote {out_path}\nWrote {summary_path}")


if __name__ == "__main__":
    main()
