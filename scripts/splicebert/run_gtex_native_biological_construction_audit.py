#!/usr/bin/env python
"""Correct the outcome-blind MaxEnt-constrained candidate construction audit.

The v1 discovery artifact is immutable.  Its first implementation optimized
PSAM affinity and rejected MaxEnt failures only after selecting a winner.  This
audit keeps every frozen candidate, control, threshold, cohort, and beam width,
but searches for the best PSAM sequence *within* the already frozen MaxEnt
constraint.  It never loads a model or reads a new model outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.biological_cohort import ACCEPTOR_WINDOW, DONOR_WINDOW  # noqa: E402
from gfmcd.splicing.biological_features import (  # noqa: E402
    beam_synonymous_optimize,
    complete_codon_starts,
    kmer_count_vector,
    load_rbns_psam_directory,
    psam_region_score,
)
from gfmcd.splicing.circuit import _dataset, eligible  # noqa: E402
from gfmcd.splicing.phase import load_cds_intervals  # noqa: E402
from gfmcd.splicing.splice_landscape import (  # noqa: E402
    compare_landscapes,
    maxent_scorers,
)
from run_gtex_native_biological_features_discovery import (  # noqa: E402
    codon_pair_log_preference,
    descriptor_values,
    no_new_splice_validator,
    sha256_text,
)


MAXENT_DELTA_BAND = 1.0
MIN_DISCOVERY_EXAMPLES = 40
MIN_DISCOVERY_GENES = 30


def exon_scorer(record: dict[str, Any]):
    names = tuple(record["kmer_names"])
    mean = np.asarray(record["scaler_mean"], dtype=np.float64)
    scale = np.asarray(record["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(record["coefficient"], dtype=np.float64)
    intercept = float(record["intercept"])

    def score(sequence: str) -> float:
        vector, observed = kmer_count_vector(sequence)
        if observed != names:
            raise RuntimeError("frozen exon-identity k-mer basis mismatch")
        standardized = (vector - mean) / scale
        return float(standardized @ coefficient + intercept)

    return score


def constrained_validator(clean: str, start: int, end: int, score3, score5):
    motif_validator = no_new_splice_validator(clean, start, end)

    def validate(candidate: str) -> bool:
        if not motif_validator(candidate):
            return False
        changes = (
            compare_landscapes(
                clean,
                candidate,
                start,
                end,
                window_length=23,
                scorer=score3,
            ),
            compare_landscapes(
                clean,
                candidate,
                start,
                end,
                window_length=9,
                scorer=score5,
            ),
        )
        return all(
            abs(change.delta_max) <= MAXENT_DELTA_BAND
            and not change.new_competitive_site
            for change in changes
        )

    return validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--basic-gtf", required=True)
    parser.add_argument("--comprehensive-gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--rbns-psams", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    discovery_path = Path(args.discovery).expanduser().resolve()
    discovery = json.loads(discovery_path.read_text())
    if discovery.get("status") != "COMPLETE":
        raise RuntimeError("discovery artifact is incomplete")
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    basic_gtf = Path(args.basic_gtf).expanduser().resolve()
    comprehensive_gtf = Path(args.comprehensive_gtf).expanduser().resolve()
    fasta = Path(args.fasta).expanduser().resolve()
    rbns_root = Path(args.rbns_psams).expanduser().resolve()

    by_id = {
        example.example_id: example
        for example in eligible(_dataset(str(basic_gtf), str(fasta)), "train")
    }
    frozen_ids = [
        row["example_id"] for row in discovery["cohort"]["discovery"]["examples"]
    ]
    if any(example_id not in by_id for example_id in frozen_ids):
        raise RuntimeError("a frozen discovery example could not be reconstructed")
    examples = [by_id[example_id] for example_id in frozen_ids]
    cds_intervals = load_cds_intervals(comprehensive_gtf)
    psams = load_rbns_psam_directory(rbns_root)
    exon_score = exon_scorer(discovery["sequence_baselines"]["exon_identity"])
    codon_counts = Counter(discovery["sequence_baselines"]["codon_counts"])
    codon_pair_counts = Counter(
        {
            tuple(name.split(":")): int(count)
            for name, count in discovery["sequence_baselines"][
                "codon_pair_counts"
            ].items()
        }
    )
    score3, score5 = maxent_scorers()
    specs = {
        "donor": {
            "window": DONOR_WINDOW,
            "anchors": [int(example.donor_pos) for example in examples],
        },
        "acceptor": {
            "window": ACCEPTOR_WINDOW,
            "anchors": [int(example.next_acceptor_pos) for example in examples],
        },
    }
    audit: dict[str, Any] = {}
    for endpoint, spec in specs.items():
        audit[endpoint] = {}
        for candidate in discovery["candidate_freeze"][endpoint]:
            candidate_name = candidate["rbp"]
            frozen = discovery["construction_freeze"][endpoint][candidate_name]
            family_names = [candidate_name, *frozen["matched_controls"]]
            families: dict[str, Any] = {}
            for family_name in family_names:
                print(
                    f"{endpoint} {candidate_name}: constructing {family_name}",
                    flush=True,
                )
                psam = psams[family_name]
                rows = []
                for example, anchor in zip(examples, spec["anchors"]):
                    start, end = spec["window"].absolute(
                        anchor,
                        len(example.sequence),
                    )
                    objective = lambda sequence, model=psam, s=start, e=end: (
                        psam_region_score(sequence, model, start=s, end=e)
                    )
                    validator = constrained_validator(
                        example.sequence,
                        start,
                        end,
                        score3,
                        score5,
                    )
                    try:
                        low, low_record = beam_synonymous_optimize(
                            example,
                            spec["window"],
                            cds_intervals,
                            anchor=anchor,
                            objective=objective,
                            maximize=False,
                            min_changed_bases=1,
                            max_changed_bases=4,
                            beam_size=256,
                            validator=validator,
                        )
                        high, high_record = beam_synonymous_optimize(
                            example,
                            spec["window"],
                            cds_intervals,
                            anchor=anchor,
                            objective=objective,
                            maximize=True,
                            min_changed_bases=1,
                            max_changed_bases=4,
                            beam_size=256,
                            validator=validator,
                        )
                        success = (
                            low != high
                            and objective(high) > objective(low)
                            and validator(low)
                            and validator(high)
                        )
                        starts = complete_codon_starts(
                            example,
                            spec["window"],
                            cds_intervals,
                            anchor=anchor,
                        )
                        low_values = descriptor_values(
                            low,
                            example,
                            window=spec["window"],
                            anchor=anchor,
                            codon_starts=starts,
                            codon_counts=codon_counts,
                            psams=psams,
                            exon_score=exon_score,
                        )
                        high_values = descriptor_values(
                            high,
                            example,
                            window=spec["window"],
                            anchor=anchor,
                            codon_starts=starts,
                            codon_counts=codon_counts,
                            psams=psams,
                            exon_score=exon_score,
                        )
                        nuisance = {
                            "off_target_psam_l1": float(
                                sum(
                                    abs(
                                        high_values[f"psam:{other}"]
                                        - low_values[f"psam:{other}"]
                                    )
                                    for other in psams
                                    if other != family_name
                                )
                            ),
                            "codon_preference_high_minus_low": float(
                                high_values["codon_preference"]
                                - low_values["codon_preference"]
                            ),
                            "exon_identity_high_minus_low": float(
                                high_values["exon_identity"]
                                - low_values["exon_identity"]
                            ),
                            "rna_accessibility_high_minus_low": float(
                                high_values["rna_accessibility"]
                                - low_values["rna_accessibility"]
                            ),
                            "mfe_high_minus_low": float(
                                high_values["mfe"] - low_values["mfe"]
                            ),
                            "gc3_high_minus_low": int(
                                sum(high[position] in "GC" for position in starts)
                                - sum(low[position] in "GC" for position in starts)
                            ),
                            "codon_pair_high_minus_low": float(
                                codon_pair_log_preference(
                                    high,
                                    starts,
                                    codon_pair_counts,
                                )
                                - codon_pair_log_preference(
                                    low,
                                    starts,
                                    codon_pair_counts,
                                )
                            ),
                            "canonical_ag_gt_high_minus_low": int(
                                sum(
                                    high[position : position + 2] in {"AG", "GT"}
                                    for position in range(start, end - 1)
                                )
                                - sum(
                                    low[position : position + 2] in {"AG", "GT"}
                                    for position in range(start, end - 1)
                                )
                            ),
                        }
                    except RuntimeError:
                        low = high = example.sequence
                        low_record = high_record = {}
                        nuisance = {"construction_failed": True}
                        success = False
                    rows.append(
                        {
                            "example_id": example.example_id,
                            "gene_id": example.gene_id,
                            "success": bool(success),
                            "low_sha256": sha256_text(low),
                            "high_sha256": sha256_text(high),
                            "score_low": float(objective(low)),
                            "score_high": float(objective(high)),
                            "low": low_record,
                            "high": high_record,
                            "nuisance": nuisance,
                        }
                    )
                families[family_name] = {
                    "role": (
                        "candidate"
                        if family_name == candidate_name
                        else "matched_control"
                    ),
                    "success_count": sum(row["success"] for row in rows),
                    "success_gene_count": len(
                        {row["gene_id"] for row in rows if row["success"]}
                    ),
                    "rows": rows,
                }
            candidate_rows = families[candidate_name]["rows"]
            candidate_doses = {
                row["example_id"]: (
                    row["low"].get("changed_bases"),
                    row["high"].get("changed_bases"),
                )
                for row in candidate_rows
                if row["success"]
            }
            for control_name in frozen["matched_controls"]:
                control = families[control_name]
                matched = [
                    row
                    for row in control["rows"]
                    if row["success"]
                    and row["example_id"] in candidate_doses
                    and (
                        row["low"].get("changed_bases"),
                        row["high"].get("changed_bases"),
                    )
                    == candidate_doses[row["example_id"]]
                ]
                control["candidate_dose_matched_success_count"] = len(matched)
                control["candidate_dose_matched_gene_count"] = len(
                    {row["gene_id"] for row in matched}
                )
            candidate_pass = (
                families[candidate_name]["success_count"] >= MIN_DISCOVERY_EXAMPLES
                and families[candidate_name]["success_gene_count"]
                >= MIN_DISCOVERY_GENES
            )
            audit[endpoint][candidate_name] = {
                "expected_high_minus_low_margin_sign": frozen[
                    "expected_high_minus_low_margin_sign"
                ],
                "matched_controls": frozen["matched_controls"],
                "candidate_construction_pass": bool(candidate_pass),
                "families": families,
            }

    result = {
        "experiment": "gtex_native_biological_construction_audit_v2",
        "status": "COMPLETE",
        "model_loaded": False,
        "source_discovery": str(discovery_path),
        "correction": (
            "v1 optimized PSAM affinity before applying the MaxEnt band. "
            "v2 searches the frozen beam within the same band."
        ),
        "frozen_thresholds": {
            "max_changed_bases": 4,
            "min_changed_bases": 1,
            "beam_size": 256,
            "max_abs_maxent_delta": MAXENT_DELTA_BAND,
            "no_new_competitive_site": True,
            "min_discovery_examples": MIN_DISCOVERY_EXAMPLES,
            "min_discovery_genes": MIN_DISCOVERY_GENES,
        },
        "audit": audit,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "gtex-native-biological-construction-audit-v2",
        sys.argv,
        repo=REPO,
        status="COMPLETE",
        config=result["frozen_thresholds"],
        inputs=[
            file_record(path)
            for path in (
                discovery_path,
                basic_gtf,
                comprehensive_gtf,
                fasta,
            )
        ],
        outputs=[file_record(output)],
        notes=[
            "No model was loaded and no candidate was reselected.",
            "The immutable v1 construction failure remains recorded.",
        ],
    )
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                endpoint: {
                    name: {
                        "pass": row["candidate_construction_pass"],
                        "candidate_success": row["families"][name]["success_count"],
                    }
                    for name, row in rows.items()
                }
                for endpoint, rows in audit.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
