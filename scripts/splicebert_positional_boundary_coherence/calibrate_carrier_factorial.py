#!/usr/bin/env python
"""Produce deterministic Phase-II null/power calibration evidence.

This is the only authorizing producer for
``gfmcd.pbc.carrier_factorial_fixture_calibration.v1``.  Each replicate is
generated from a frozen sign-symmetric block model and passed to the real
``analyze_fixture_arrays`` entry point, which in turn calls the same
simultaneous interval, exact-add-one, MDE, classification, and verdict code as
the carrier-factorial target analysis.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.pbc_carrier_factorial import (  # noqa: E402
    MIN_PSEUDO_PHASE_CONTROLS,
    MIN_STRUCTURED_CONTROLS,
)
from analyze_carrier_factorial import (  # noqa: E402
    CONFIRMATORY_BOOTSTRAP_DRAWS,
    CONFIRMATORY_BOOTSTRAP_SEED,
    analyze_fixture_arrays,
)


CALIBRATION_SCHEMA = "gfmcd.pbc.carrier_factorial_fixture_calibration.v1"
SMOKE_SCHEMA = "gfmcd.pbc.carrier_factorial_fixture_calibration_smoke.v1"
CALIBRATION_REPLICATES = 1_000
CALIBRATION_BLOCKS = 600
CALIBRATION_SEED = 2026081063
CALIBRATION_NOISE_SD = 0.16
PLANTED_EFFECTS = (0.050, 0.075, 0.100)
STRATA_NAMES = (
    "chr1|+",
    "chr1|-",
    "chr2|+",
    "chr2|-",
    "chr3|+",
    "chr3|-",
    "chr4|+",
    "chr4|-",
    "chr5|+",
    "chr5|-",
    "chr6|+",
    "chr6|-",
)


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def _strata(n_blocks: int) -> list[str]:
    if n_blocks < len(STRATA_NAMES) * 5 or n_blocks % len(STRATA_NAMES):
        raise ValueError("fixture blocks must divide evenly across twelve strata")
    return [
        name
        for name in STRATA_NAMES
        for _ in range(n_blocks // len(STRATA_NAMES))
    ]


def _control_sufficient_statistics(
    rng: np.random.Generator,
    *,
    n_controls: int,
    n_blocks: int,
    noise_sd: float,
) -> dict[str, list[float]]:
    """Generate consistent control and across-control block margins.

    These two margins are the smallest raw inputs used by the exact add-one and
    paired mean-control families.  Their grand means are forced to agree, as
    they would for a materialized control-by-block matrix.
    """

    control_means = rng.normal(
        0.0, noise_sd / np.sqrt(n_blocks), size=n_controls
    )
    block_mean = rng.normal(
        float(control_means.mean()),
        noise_sd / np.sqrt(n_controls),
        size=n_blocks,
    )
    block_mean += float(control_means.mean() - block_mean.mean())
    return {
        "control_means": [float(value) for value in control_means],
        "block_mean": [float(value) for value in block_mean],
    }


def generate_fixture(
    *,
    effect: float,
    seed: int,
    n_blocks: int = CALIBRATION_BLOCKS,
    noise_sd: float = CALIBRATION_NOISE_SD,
) -> dict[str, Any]:
    """Generate one fully numeric sign-symmetric or planted fixture."""

    rng = np.random.default_rng(seed)
    values = {
        name: [
            float(value)
            for value in rng.normal(float(effect), noise_sd, size=n_blocks)
        ]
        for name in ("clean", "N3", "N10", "J")
    }
    directions = {
        name: [
            float(value)
            for value in rng.normal(float(effect), noise_sd, size=n_blocks)
        ]
        for name in ("Dplus", "Dminus", "Psi0", "Psi1", "Psi2")
    }

    def structured() -> dict[str, list[float]]:
        return _control_sufficient_statistics(
            rng,
            n_controls=MIN_STRUCTURED_CONTROLS,
            n_blocks=n_blocks,
            noise_sd=noise_sd,
        )

    return {
        "strata": _strata(n_blocks),
        "values": values,
        "directions": directions,
        "specificity": {
            "clean": {
                "pseudo": {
                    **_control_sufficient_statistics(
                        rng,
                        n_controls=MIN_PSEUDO_PHASE_CONTROLS,
                        n_blocks=n_blocks,
                        noise_sd=noise_sd,
                    ),
                    "assignment_exact": True,
                }
            },
            "N3": {"phase": structured(), "axis": structured()},
            "N10": {"phase": structured(), "axis": structured()},
            "J": {"joint": structured()},
        },
    }


def _fixture_sha256(fixture: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        fixture, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def analyze_replicate(
    *,
    replicate: int,
    effect: float,
    group_index: int,
    n_blocks: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    seed = CALIBRATION_SEED + group_index * 1_000_000 + replicate
    fixture = generate_fixture(effect=effect, seed=seed, n_blocks=n_blocks)
    analysis = analyze_fixture_arrays(
        fixture,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=CONFIRMATORY_BOOTSTRAP_SEED,
    )
    return {
        "replicate": int(replicate),
        "generator_seed": int(seed),
        "input_sha256": _fixture_sha256(fixture),
        **analysis,
    }


def generate_calibration(
    *,
    replicates: int = CALIBRATION_REPLICATES,
    n_blocks: int = CALIBRATION_BLOCKS,
    n_bootstrap: int = CONFIRMATORY_BOOTSTRAP_DRAWS,
    authorizing: bool = True,
) -> dict[str, Any]:
    if authorizing and (
        replicates != CALIBRATION_REPLICATES
        or n_blocks != CALIBRATION_BLOCKS
        or n_bootstrap != CONFIRMATORY_BOOTSTRAP_DRAWS
    ):
        raise ValueError("authorizing calibration sizes are frozen")
    if replicates < 1:
        raise ValueError("calibration requires at least one replicate")
    null_rows = [
        analyze_replicate(
            replicate=index,
            effect=0.0,
            group_index=0,
            n_blocks=n_blocks,
            n_bootstrap=n_bootstrap,
        )
        for index in range(replicates)
    ]
    planted = {
        f"{effect:.3f}": [
            analyze_replicate(
                replicate=index,
                effect=effect,
                group_index=group_index,
                n_blocks=n_blocks,
                n_bootstrap=n_bootstrap,
            )
            for index in range(replicates)
        ]
        for group_index, effect in enumerate(PLANTED_EFFECTS, start=1)
    }
    contract_specs = (
        ("identity", 0.0, 10),
        ("no_op", 0.0, 11),
        ("planted_positive", 0.10, 12),
        ("sign_reversed", -0.10, 13),
    )
    contract_fixtures = {
        name: analyze_replicate(
            replicate=0,
            effect=effect,
            group_index=group_index,
            n_blocks=n_blocks,
            n_bootstrap=n_bootstrap,
        )
        for name, effect, group_index in contract_specs
    }
    return {
        "schema": CALIBRATION_SCHEMA if authorizing else SMOKE_SCHEMA,
        "stage": "development",
        "status": "COMPLETE" if authorizing else "SMOKE_ONLY_NON_AUTHORIZING",
        "analysis_engine": "analyze_carrier_factorial.analyze_fixture_arrays",
        "decision_engine": "pbc_carrier_factorial.carrier_phase2_decision",
        "replicates": int(replicates),
        "effects": [float(value) for value in PLANTED_EFFECTS],
        "generator": {
            "algorithm": "independent_gaussian_block_fixtures_v1",
            "base_seed": CALIBRATION_SEED,
            "bit_generator": "PCG64",
            "n_blocks": int(n_blocks),
            "strata": list(STRATA_NAMES),
            "noise_sd": CALIBRATION_NOISE_SD,
            "sign_symmetric_null": True,
            "stored_inputs": "control_and_block_margins_sufficient_for_exact_families",
        },
        "analysis_parameters": {
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": CONFIRMATORY_BOOTSTRAP_SEED,
        },
        "contract_fixtures": contract_fixtures,
        "null_replicates": null_rows,
        "planted_replicates": planted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = generate_calibration()
    output, manifest_path = output_paths(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-carrier-factorial-fixture-calibration-v1",
        sys.argv,
        repo=REPO,
        status=result["status"],
        config={
            "stage": "development",
            "schema": CALIBRATION_SCHEMA,
            "replicates": CALIBRATION_REPLICATES,
            "effects": list(PLANTED_EFFECTS),
            "n_blocks": CALIBRATION_BLOCKS,
            "n_bootstrap": CONFIRMATORY_BOOTSTRAP_DRAWS,
        },
        seeds={
            "fixture_generation": CALIBRATION_SEED,
            "analysis_bootstrap": CONFIRMATORY_BOOTSTRAP_SEED,
        },
    )
    for path in (
        REPO / "studies/splicebert_positional_boundary_coherence/PREREGISTRATION.md",
        REPO / "src/gfmcd/splicing/pbc_carrier_factorial.py",
        SCRIPT_DIR / "analyze_carrier_factorial.py",
        Path(__file__).resolve(),
    ):
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "replicates_per_group": result["replicates"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
