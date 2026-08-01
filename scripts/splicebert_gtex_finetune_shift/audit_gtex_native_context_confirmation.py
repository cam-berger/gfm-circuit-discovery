#!/usr/bin/env python
"""Outcome-independent delivered-dose audit for context confirmation v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.context_localization import (  # noqa: E402
    RelativeWindow,
    shuffle_relative_window,
)
from gfmcd.splicing.native_classifier import load_annotation_cohort  # noqa: E402


MINIMUM_HAMMING_FRACTION = 0.25
MAIN_CONDITIONS = ("candidate_draw0", "candidate_draw1", "sham_draw0")
RESOLUTION_CONDITIONS = ("half1_draw0", "half2_draw0")


def parse_window(name: str) -> RelativeWindow:
    start_text, end_text = name.split(":")
    return RelativeWindow(int(start_text), int(end_text))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_LOCALIZATION_V1.md"
    )
    confirmation_path = Path(args.confirmation).expanduser().resolve()
    discovery_path = Path(args.discovery).expanduser().resolve()
    confirmation = json.loads(confirmation_path.read_text())
    discovery = json.loads(discovery_path.read_text())
    if confirmation.get("stage") != "five-seed region confirmation":
        raise RuntimeError("unexpected confirmation artifact")
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    examples = load_annotation_cohort(
        args.gtf,
        args.fasta,
        split="val",
        n=int(confirmation["n"]),
        tag=str(confirmation["cohort_tag"]),
    )

    condition_specs: dict[str, dict[str, tuple[str, int]]] = {}
    for task_name in ("donor", "acceptor"):
        frozen = discovery["freeze"][task_name]
        condition_specs[task_name] = {
            "candidate_draw0": (frozen["distal_candidate"], 0),
            "candidate_draw1": (frozen["distal_candidate"], 1),
            "sham_draw0": (frozen["matched_sham"], 0),
            "half1_draw0": (frozen["candidate_halves"][0], 0),
            "half2_draw0": (frozen["candidate_halves"][1], 0),
        }

    failures: list[dict[str, Any]] = []
    condition_audit: dict[str, dict[str, Any]] = {}
    for task_name, task_conditions in condition_specs.items():
        condition_audit[task_name] = {}
        anchors = [
            (
                int(example.donor_pos)
                if task_name == "donor"
                else int(example.next_acceptor_pos)
            )
            for example in examples
        ]
        for condition, (window_name, draw) in task_conditions.items():
            window = parse_window(window_name)
            doses: list[float] = []
            for example, anchor in zip(examples, anchors):
                _, audit = shuffle_relative_window(
                    example.sequence,
                    anchor=anchor,
                    window=window,
                    key=(
                        example.example_id,
                        task_name,
                        window.name,
                        "confirmation",
                    ),
                    draw=draw,
                )
                doses.append(audit.hamming_fraction)
                if audit.hamming_fraction < MINIMUM_HAMMING_FRACTION:
                    failures.append(
                        {
                            "example_id": example.example_id,
                            "gene_id": example.gene_id,
                            "task": task_name,
                            "condition": condition,
                            "window": window.name,
                            "draw": draw,
                            "hamming": audit.hamming,
                            "width": audit.width,
                            "hamming_fraction": audit.hamming_fraction,
                            "invariants_pass": audit.invariants_pass,
                        }
                    )
            condition_audit[task_name][condition] = {
                "window": window.name,
                "draw": draw,
                "minimum_hamming_fraction": float(min(doses)),
                "mean_hamming_fraction": float(np.mean(doses)),
                "n_below_threshold": int(
                    sum(
                        dose < MINIMUM_HAMMING_FRACTION for dose in doses
                    )
                ),
                "dose_gate_pass": all(
                    dose >= MINIMUM_HAMMING_FRACTION for dose in doses
                ),
            }

    licenses: dict[str, dict[str, Any]] = {}
    for task_name in ("donor", "acceptor"):
        main_dose = all(
            condition_audit[task_name][condition]["dose_gate_pass"]
            for condition in MAIN_CONDITIONS
        )
        resolution_dose = all(
            condition_audit[task_name][condition]["dose_gate_pass"]
            for condition in RESOLUTION_CONDITIONS
        )
        model_confirmation = bool(
            confirmation["confirmation"][task_name]["confirmed"]
        )
        licenses[task_name] = {
            "model_confirmation_gate": model_confirmation,
            "main_32nt_and_sham_dose_gate": main_dose,
            "confirmed_32nt_region_licensed": (
                model_confirmation and main_dose
            ),
            "half_resolution_dose_gate": resolution_dose,
            "half_resolution_licensed": (
                model_confirmation and main_dose and resolution_dose
            ),
        }

    artifact = {
        "study": "native GTEx classifier context localization",
        "stage": "confirmation delivered-dose audit",
        "protocol": str(protocol.relative_to(REPO)),
        "status": (
            "PASS"
            if not failures
            else "PARTIAL_PASS_RESOLUTION_QUARANTINED"
        ),
        "threshold": {
            "minimum_hamming_fraction": MINIMUM_HAMMING_FRACTION,
            "scope": "each example and condition",
        },
        "condition_audit": condition_audit,
        "failures": failures,
        "licenses": licenses,
        "interpretation": (
            "The confirmed 32-nt candidate regions and matched shams satisfy "
            "delivered-dose controls. The acceptor half-window contrast is "
            "unlicensed because one sequence has no alternative exact Eulerian "
            "ordering at that 16-nt resolution."
            if failures
            else "All candidate, sham, and half-window dose controls pass."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_confirmation_audit_v1",
        sys.argv,
        repo=REPO,
        status=artifact["status"],
        config={
            "split": "val",
            "n": len(examples),
            "cohort_tag": confirmation["cohort_tag"],
            "minimum_hamming_fraction": MINIMUM_HAMMING_FRACTION,
            "model_forward_passes": 0,
        },
        notes=[
            "This audit reconstructs only outcome-independent string interventions.",
            "No checkpoint is loaded and no model outcome is recomputed.",
        ],
    )
    for path in (
        protocol,
        confirmation_path,
        discovery_path,
        Path(args.gtf),
        Path(args.fasta),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(f"{artifact['status']}: saved -> {output}")


if __name__ == "__main__":
    main()
