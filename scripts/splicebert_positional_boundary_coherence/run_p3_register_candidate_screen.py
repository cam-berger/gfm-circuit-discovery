#!/usr/bin/env python
"""Run the model-backed 102-component Phase-III development screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.circuit import load_adapter, pick_device  # noqa: E402
from gfmcd.splicing.pbc_register_candidate_runner import (  # noqa: E402
    HfCandidateMeasurementProvider,
    execute_candidate_screen,
)
from gfmcd.splicing.pbc_register_candidates import (  # noqa: E402
    DEVELOPMENT_BLOCKS,
    TOKEN_SUPPORT,
    validate_candidate_screen,
)
from gfmcd.splicing.pbc_register_runner import register_cell_from_mapping  # noqa: E402
from gfmcd.splicing.pbc_register_stage import P3_BINS  # noqa: E402


COHORT_SCHEMA = "gfmcd.positional_boundary_coherence.cohort.v1"


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def development_cells(cohort: Mapping[str, Any]) -> tuple[Any, ...]:
    """Validate the exact outcome-blind 120-block development lattice."""

    if cohort.get("schema") != COHORT_SCHEMA:
        raise RuntimeError("wrong positional-boundary-coherence cohort schema")
    rows = tuple(
        row
        for row in cohort.get("cells", [])
        if isinstance(row, Mapping) and row.get("split_role") == "development"
    )
    by_block: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(str(row.get("block_id", "")), []).append(row)
    if len(by_block) != DEVELOPMENT_BLOCKS:
        raise RuntimeError(
            f"candidate screen needs {DEVELOPMENT_BLOCKS} development blocks; "
            f"found {len(by_block)}"
        )
    expected = {(left, right) for left in range(3) for right in range(3)}
    for block_id, local in by_block.items():
        if not block_id or len(local) != 9:
            raise RuntimeError("candidate-screen block is not a complete nine-cell lattice")
        if {
            (int(row["donor_phase"]), int(row["acceptor_phase"])) for row in local
        } != expected:
            raise RuntimeError("candidate-screen block lacks the complete 3x3 lattice")
    cells = tuple(register_cell_from_mapping(row) for row in rows)
    if any(
        len(cell.intact_sequence) != 1022
        or cell.donor_position != 392
        or cell.acceptor_position != 512
        for cell in cells
    ):
        raise RuntimeError("candidate screen drifted from the frozen 1022/392/512 geometry")
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--p3-dose", required=True, type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=36)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output, manifest_path = output_paths(args.out)
    cohort_path = Path(args.cohort).expanduser().resolve()
    cohort = json.loads(cohort_path.read_text())
    # Census, split, geometry, and output path all fail before checkpoint loading.
    cells = development_cells(cohort)
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise RuntimeError("--model must name the official local SpliceBERT checkpoint")
    adapter = load_adapter(str(model_path), pick_device(args.device), eager=True)
    provider = HfCandidateMeasurementProvider(
        adapter,
        p3_bins=P3_BINS,
        p3_attenuation_dose=args.p3_dose,
        direction="right",
    )
    result = execute_candidate_screen(
        cells, provider=provider, batch_size=args.batch
    )
    validate_candidate_screen(result)
    result["model"] = dict(getattr(adapter.metadata(), "__dict__", {}))
    result["design"] = {
        "p3_bins": list(P3_BINS),
        "p3_attenuation_dose": float(args.p3_dose),
        "direction": "right",
        "token_support": list(TOKEN_SUPPORT),
        "corruption": "acceptor_AG_to_AC",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-p3-register-candidate-screen-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_DISCOVERY",
        config={
            "stage": "development",
            "n_blocks": DEVELOPMENT_BLOCKS,
            "components": 102,
            "token_support": list(TOKEN_SUPPORT),
            "p3_attenuation_dose": float(args.p3_dose),
            "batch_size": int(args.batch),
        },
    )
    manifest.add_input(cohort_path)
    for name in (
        "pytorch_model.bin",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ):
        manifest.add_input(model_path / name)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_blocks": result["n_blocks"],
                "components": len(result["candidate_universe"]),
                "patch_events": result["tensor_stream"]["patch_events"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
