#!/usr/bin/env python
"""Post-hoc dominant-energy sensitivity for the frozen positional spectrum.

The primary assay compares every numerically nonzero coefficient direction.
This explicitly exploratory analysis asks whether its lower full-rank overlap
is driven by weak tail directions while the dominant carrier plane remains
aligned. It changes no primary threshold or causal conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.gtex_finetune import (  # noqa: E402
    checkpoint_seed,
    load_gtex_state_dict,
)
from gfmcd.splicing.positional_spectrum import (  # noqa: E402
    band_coefficient_matrix,
    dominant_subspace_comparison,
    validate_table,
)

RANKS = (1, 2, 4, 8, 14)


def _table(state: Mapping[str, Any]) -> np.ndarray:
    hits = [
        value
        for key, value in state.items()
        if key.endswith("position_embeddings.weight")
    ]
    if len(hits) != 1 or not isinstance(hits[0], torch.Tensor):
        raise ValueError("exactly one tensor position table is required")
    return validate_table(
        hits[0].detach().cpu().to(dtype=torch.float64).numpy(),
        shape=(1026, 512),
    )


def _load_base(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("base weights are not a state mapping")
    return _table(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum", required=True)
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    spectrum_path = Path(args.spectrum).expanduser().resolve()
    spectrum = json.loads(spectrum_path.read_text())
    if spectrum.get("status") != "COMPLETE":
        raise RuntimeError("controlling spectrum is not complete")
    base_path = Path(args.base_weights).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output = guard_output_path(Path(args.out).expanduser().resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    base = _load_base(base_path)
    checkpoints = {
        checkpoint_seed(path): path
        for path in sorted(checkpoint_dir.glob("*.pt"))
        if checkpoint_seed(path) is not None
    }
    if set(checkpoints) != set(range(2020, 2025)):
        raise RuntimeError("complete five-checkpoint family is required")

    records: dict[str, Any] = {}
    for band_name, bins in spectrum["analysis"]["named_bands"].items():
        base_coefficients = band_coefficient_matrix(base, bins)
        records[band_name] = {}
        for seed, checkpoint in sorted(checkpoints.items()):
            if (
                sha256_file(checkpoint)
                != spectrum["sources"]["fine_tuned"][str(seed)]["sha256"]
            ):
                raise RuntimeError(f"checkpoint hash mismatch for seed {seed}")
            tuned = _table(load_gtex_state_dict(checkpoint))
            records[band_name][str(seed)] = dominant_subspace_comparison(
                base_coefficients,
                band_coefficient_matrix(tuned, bins),
                RANKS,
            )

    rank2_overlap = {
        band: [
            records[band][str(seed)]["2"]["symmetric_projection_overlap"]
            for seed in range(2020, 2025)
        ]
        for band in records
    }
    payload = {
        "experiment_id": "gtex-positional-spectrum-dominant-subspace-sensitivity-v1",
        "status": "COMPLETE_EXPLORATORY",
        "post_hoc": True,
        "primary_estimands_unchanged": True,
        "spectrum_source": file_record(spectrum_path),
        "ranks": list(RANKS),
        "records": records,
        "rank2_summary": {
            band: {
                "values": values,
                "minimum": float(min(values)),
                "maximum": float(max(values)),
                "mean": float(np.mean(values)),
            }
            for band, values in rank2_overlap.items()
        },
        "interpretation_boundary": (
            "Leading-energy subspace alignment is an exploratory sensitivity "
            "descriptor. It does not replace the preregistered full numerical-rank "
            "overlap, establish behavior, or identify a biological frequency."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "gtex-positional-spectrum-dominant-subspace-sensitivity-v1",
        sys.argv,
        repo=REPO,
        status="COMPLETE_EXPLORATORY",
        config={
            "ranks": list(RANKS),
            "post_hoc": True,
            "spectrum_sha256": sha256_file(spectrum_path),
        },
    )
    for path in [spectrum_path, base_path, *checkpoints.values()]:
        manifest.add_input(path)
    manifest.outputs = [file_record(output)]
    manifest.write(manifest_path)
    print(json.dumps(payload["rank2_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
