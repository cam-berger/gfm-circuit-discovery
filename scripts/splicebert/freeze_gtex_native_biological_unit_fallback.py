#!/usr/bin/env python
"""Freeze the model-free Lock C for the native-unit biological fallback.

This command reads no classifier outputs and loads no model.  It freezes the
candidate signs, discovery/confirmation sequence identities, checkpoint file
identities, exact native-unit intervention, matched controls, directed
head-to-unit bridge, nested size curve, GFS19/GFS20 comparisons, and one BH
family before discovery or confirmation is allowed to run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gtex_native_biological_unit_fallback import (  # noqa: E402
    CONTROL_DRAWS,
    EXPECTED_SEEDS,
    GROUP_SIZES,
    INTERMEDIATE_SIZE,
    LOCK_SCHEMA,
    N_BOOT,
    N_PERM,
    candidate_contract,
    normalize_confirmation_states,
    normalize_discovery_audit,
)


def parse_checkpoints(values: list[str]) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in checkpoints:
            raise ValueError(f"duplicate checkpoint seed {seed}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints[seed] = path
    if tuple(sorted(checkpoints)) != EXPECTED_SEEDS:
        raise ValueError("complete checkpoint family 2020--2024 required")
    return checkpoints


def model_files(base_model: Path) -> list[Path]:
    required = [base_model / "config.json"]
    tokenizer = next(
        (
            base_model / name
            for name in ("tokenizer.json", "vocab.txt")
            if (base_model / name).is_file()
        ),
        None,
    )
    if tokenizer is None:
        raise FileNotFoundError("base model tokenizer.json or vocab.txt")
    required.append(tokenizer)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    return required


def compatible_contracts(discovery: dict, confirmation: dict) -> None:
    def core(package: dict) -> list[dict]:
        values = [
            {
                "candidate_id": value["candidate_id"],
                "endpoint": value["endpoint"],
                "expected_high_minus_low_margin_sign": value[
                    "expected_high_minus_low_margin_sign"
                ],
                "matched_controls": value["matched_controls"],
                "family_roles": {
                    key: family["role"]
                    for key, family in value["families"].items()
                },
            }
            for value in package["candidates"]
        ]
        return sorted(
            values,
            key=lambda value: (value["endpoint"], value["candidate_id"]),
        )

    if core(discovery) != core(confirmation):
        raise RuntimeError(
            "discovery and confirmation candidate/sign/control contracts differ"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-construction-audit", required=True)
    parser.add_argument("--discovery-sequence-package", required=True)
    parser.add_argument("--confirmation-construction", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gfs19-discovery", required=True)
    parser.add_argument("--gfs19-arrays", required=True)
    parser.add_argument("--gfs18-head-census", required=True)
    parser.add_argument("--gfs20-discovery", required=True)
    parser.add_argument("--expected-confirmation-genes", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    discovery_path = Path(args.discovery_construction_audit).expanduser().resolve()
    discovery_sequence_path = Path(
        args.discovery_sequence_package
    ).expanduser().resolve()
    confirmation_path = Path(args.confirmation_construction).expanduser().resolve()
    discovery = normalize_discovery_audit(
        json.loads(discovery_path.read_text()),
        json.loads(discovery_sequence_path.read_text()),
    )
    confirmation = normalize_confirmation_states(
        json.loads(confirmation_path.read_text()),
        expected_genes=args.expected_confirmation_genes,
    )
    compatible_contracts(discovery, confirmation)
    overlap = set(discovery["gene_ids"]) & set(confirmation["gene_ids"])
    if overlap:
        raise RuntimeError(
            f"discovery/confirmation gene firewall failed for {len(overlap)} genes"
        )

    gfs19_path = Path(args.gfs19_discovery).expanduser().resolve()
    gfs19_arrays_path = Path(args.gfs19_arrays).expanduser().resolve()
    gfs18_path = Path(args.gfs18_head_census).expanduser().resolve()
    gfs20_path = Path(args.gfs20_discovery).expanduser().resolve()
    gfs19 = json.loads(gfs19_path.read_text())
    gfs18 = json.loads(gfs18_path.read_text())
    gfs20 = json.loads(gfs20_path.read_text())
    if (
        gfs19.get("status") != "COMPLETE"
        or gfs18.get("status") != "COMPLETE"
        or gfs20.get("status") != "COMPLETE"
    ):
        raise RuntimeError("GFS18, GFS19, and GFS20 discovery artifacts must be COMPLETE")
    if tuple(sorted(map(int, gfs20.get("rankings", {})))) != EXPECTED_SEEDS:
        raise RuntimeError("GFS20 does not contain the frozen five-checkpoint family")
    if not gfs19_arrays_path.is_file():
        raise FileNotFoundError(gfs19_arrays_path)

    checkpoints = parse_checkpoints(args.checkpoint)
    base_model = Path(args.base_model).expanduser().resolve()
    base_files = model_files(base_model)
    output = guard_output_path(Path(args.out).expanduser().resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    checkpoint_records = {
        str(seed): {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for seed, path in checkpoints.items()
    }
    head_bridge: dict[str, dict] = {}
    for endpoint in ("donor", "acceptor"):
        freeze = gfs18["freeze"][endpoint]
        candidate_heads = list(freeze["candidate_heads"])
        if candidate_heads != list(gfs20["frozen_heads"][endpoint]):
            raise RuntimeError(f"GFS18/GFS20 frozen heads disagree for {endpoint}")
        control_map = freeze["control_heads"]
        control_draws = [
            [control_map[head][draw] for head in candidate_heads]
            for draw in range(CONTROL_DRAWS)
        ]
        if any(len(set(draw)) != len(candidate_heads) for draw in control_draws):
            raise RuntimeError(f"GFS18 head-control draw is not unique for {endpoint}")
        head_bridge[endpoint] = {
            "candidate_heads": candidate_heads,
            "matched_head_control_draws": control_draws,
        }
    result = {
        "schema": LOCK_SCHEMA,
        "status": "FROZEN_LOCK_C",
        "model_loaded": False,
        "classifier_outcomes_read": False,
        "sparse_instrument_status": "INSTRUMENT_UNRESOLVED",
        "sparse_weights_used": False,
        "object": (
            "exact architectural post-GELU L5 native-unit populations; "
            "distributed/non-monosemantic coordinates, not sparse features"
        ),
        "sources": {
            "discovery_construction": {
                "path": str(discovery_path),
                "sha256": sha256_file(discovery_path),
                "n_genes": discovery["n_genes"],
            },
            "discovery_sequence_package": {
                "path": str(discovery_sequence_path),
                "sha256": sha256_file(discovery_sequence_path),
            },
            "confirmation_construction": {
                "path": str(confirmation_path),
                "sha256": sha256_file(confirmation_path),
                "n_genes": confirmation["n_genes"],
            },
            "gfs19_discovery": {
                "path": str(gfs19_path),
                "sha256": sha256_file(gfs19_path),
            },
            "gfs19_arrays": {
                "path": str(gfs19_arrays_path),
                "sha256": sha256_file(gfs19_arrays_path),
            },
            "gfs18_head_census": {
                "path": str(gfs18_path),
                "sha256": sha256_file(gfs18_path),
            },
            "gfs20_discovery": {
                "path": str(gfs20_path),
                "sha256": sha256_file(gfs20_path),
            },
            "base_model": {
                "path": str(base_model),
                "files": [
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in base_files
                ],
            },
            "checkpoints": checkpoint_records,
        },
        "gene_firewall": {
            "discovery_n_genes": discovery["n_genes"],
            "confirmation_n_genes": confirmation["n_genes"],
            "overlap": [],
            "PASS": True,
        },
        "candidate_contract": {
            "discovery": candidate_contract(discovery),
            "confirmation": candidate_contract(confirmation),
        },
        "orientation": {
            "positive_sign": "preferred=high, adverse=low",
            "negative_sign": "preferred=low, adverse=high",
            "estimands": (
                "damage = preferred margin - preferred margin after subtracting "
                "the frozen unit write delta; recovery = adverse margin after "
                "adding the frozen unit write delta - adverse margin"
            ),
        },
        "capture": {
            "layer": 5,
            "mlp_input_width": 512,
            "post_gelu_width": INTERMEDIATE_SIZE,
            "writer_definition": "columns of L5 output.dense.weight",
            "native_write_width": 512,
            "endpoint_margin": "endpoint class logit minus no-site class logit",
            "dropout": "model.eval(), exact released classifier",
        },
        "discovery": {
            "ranking": (
                "per checkpoint/endpoint/candidate, descending minimum of "
                "gene-mean exact one-unit damage and recovery"
            ),
            "candidate_family_only": True,
            "confirmation_outcomes_available": False,
            "group_sizes": list(GROUP_SIZES),
            "nested_prefixes": True,
        },
        "controls": {
            "draws": CONTROL_DRAWS,
            "exact_size_and_unique": True,
            "candidate_units_excluded": True,
            "nested_prefixes": True,
            "matched_distance": (
                "robust-standardized squared Euclidean distance over log writer-column "
                "norm, log discovery activation RMS, positive activation rate, and "
                "sequence-change rate; greedy without replacement"
            ),
            "uniform_random_secondary_null": True,
            "matching_uses_discovery_only": True,
        },
        "directed_bridge": {
            "upstream_components": "frozen endpoint-specific GFS18 L5 head triple",
            "intervention": (
                "patch adverse heads into preferred and preferred heads into adverse "
                "at the target token, recompute exact post-GELU activations"
            ),
            "downstream_filter": (
                "project only the frozen candidate-responsive native-unit population"
            ),
            "controls": "same matched and uniform frozen native-unit populations",
            "frozen_head_sets": head_bridge,
            "separate_verdict": True,
        },
        "comparisons": {
            "gfs20": (
                "same-size overlap/Jaccard with frozen context and head-induced "
                "native-unit ranking prefixes"
            ),
            "gfs19": (
                "per-example fraction of selected native write energy in the "
                "checkpoint/endpoint rank-4 task-code basis"
            ),
            "candidate_generation_only": True,
        },
        "inference": {
            "checkpoint_replication": (
                "average five checkpoint effects within example; separately require "
                "all five checkpoint point estimates to have the expected positive sign"
            ),
            "bootstrap": {
                "unit": "gene",
                "draws": N_BOOT,
                "interval": "two-sided percentile 95%",
            },
            "sign_flip": {
                "unit": "gene",
                "draws": N_PERM,
                "tail": "two-sided",
                "add_one": True,
            },
            "bh_family": {
                "name": "full_native_unit_confirmation_family",
                "members": (
                    "all endpoints x candidate/control sequence families x "
                    "8/16/32/64/128/256 sizes x damage/recovery x "
                    "candidate-population/matched-null/uniform-null and all "
                    "direct selected-minus-null contrasts, including the directed bridge"
                ),
                "pooled_across_sizes": True,
                "pooled_across_endpoints": True,
                "pooled_across_candidate_and_control_sequence_families": True,
                "q_field": "q_bh_full_native_unit_family",
            },
        },
        "preflight": {
            "architecture": "6 layers x 16 heads x 512 hidden x 2048 MLP",
            "write_reconstruction_tolerance": 5e-4,
            "composition_tolerance": 1e-5,
            "all_unit_delta_tolerance": 5e-4,
            "zero_population_exact": True,
            "rank4_basis_orthonormal_tolerance": 1e-4,
            "fail_closed": True,
        },
        "checkpoints": checkpoint_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_biological_unit_fallback_lock_v1",
        sys.argv,
        repo=REPO,
        status="FROZEN",
        config={
            "group_sizes": list(GROUP_SIZES),
            "control_draws": CONTROL_DRAWS,
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
            "expected_confirmation_genes": args.expected_confirmation_genes,
        },
        seeds={"checkpoint_family": list(EXPECTED_SEEDS)},
        notes=[
            "No model was loaded and no classifier outcome was read.",
            "The unresolved sparse instrument is excluded by construction.",
            "Native units are architectural, distributed/non-monosemantic coordinates.",
        ],
    )
    for path in (
        discovery_path,
        discovery_sequence_path,
        confirmation_path,
        gfs19_path,
        gfs19_arrays_path,
        gfs18_path,
        gfs20_path,
        *base_files,
        *checkpoints.values(),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(output)


if __name__ == "__main__":
    main()
