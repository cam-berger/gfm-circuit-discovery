#!/usr/bin/env python
"""Complete train-only L5 head census for confirmed native context effects."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.adapters.splicebert import SpliceBertAdapter  # noqa: E402
from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.circuits.hf_bert import patch_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.rigor import benjamini_hochberg  # noqa: E402
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.context_localization import (  # noqa: E402
    RelativeWindow,
    shuffle_relative_window,
)
from gfmcd.splicing.grouped_stats import (  # noqa: E402
    grouped_bootstrap,
    grouped_signflip_pvalue,
)
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    ACCEPTOR_LABEL,
    DONOR_LABEL,
    NativeClassifierTask,
    load_annotation_gene_hash_cohort,
)
from gfmcd.splicing.native_mechanism import (  # noqa: E402
    cache_final_layer_inputs,
    capture_final_layer_run,
    deterministic_control_heads,
    donor_token_positions,
    position_mask,
    replay_final_layer,
    select_head_candidates,
)


N_DISCOVERY = 128
N_BOOT = 5000
N_PERM = 9999
FINAL_LAYER = 5
NUM_HEADS = 16
MIN_EFFECT = 0.05
MAX_DISCOVERY_Q = 0.10
MAX_CANDIDATES = 3
CONTROLS_PER_CANDIDATE = 3
MAX_REPLAY_ERROR = 2e-5
NAMESPACE = "gtex-native-context-head-census-v1"
COHORT_TAG = "gtex-native-context-head-census-discovery-v1"
FORBIDDEN_CHROMS = frozenset({"chr8", "chr9", "chr11", "chr16", "chr19"})


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model weights not found in {model_dir}")


def parse_window(name: str) -> RelativeWindow:
    start_text, end_text = name.split(":")
    return RelativeWindow(int(start_text), int(end_text))


def classifier_margin(
    model: Any,
    hidden: torch.Tensor,
    token_positions: torch.Tensor,
    *,
    label: int,
) -> torch.Tensor:
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    positions = token_positions.to(device=hidden.device)
    logits = model.classifier(model.dropout(hidden[rows, positions])).float()
    margin = logits[:, label] - logits[:, 0]
    if not torch.isfinite(margin).all():
        raise RuntimeError("non-finite classifier margin")
    return margin


def summarize(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    result = grouped_bootstrap(values, groups, n_boot=N_BOOT, seed=seed)
    result["p_signflip_add_one"] = grouped_signflip_pvalue(
        values,
        groups,
        n_perm=N_PERM,
        seed=seed + 100_000,
    )
    return result


def gene_from_example_id(example_id: str) -> str:
    try:
        return example_id.rsplit(":", 1)[1]
    except IndexError as exc:
        raise ValueError(f"cannot recover gene from {example_id!r}") from exc


def raw_gene_ids(artifact: dict[str, Any]) -> set[str]:
    genes = {
        str(row["gene_id"])
        for row in artifact.get("raw", [])
        if isinstance(row, dict) and "gene_id" in row
    }
    genes.update(
        gene_from_example_id(str(example_id))
        for example_id in artifact.get("example_ids", [])
    )
    return genes


def prerequisite_gate(
    *,
    mechanism_preflight: dict[str, Any],
    context_preflight: dict[str, Any],
    context_audit: dict[str, Any],
    component_bridge: dict[str, Any],
) -> dict[str, bool]:
    replay_errors = [
        float(error)
        for task_errors in component_bridge.get(
            "confirmation_replay_max_abs_error",
            {},
        ).values()
        for error in task_errors.values()
    ]
    return {
        "mechanism_preflight_pass": mechanism_preflight.get("status") == "PASS",
        "context_preflight_pass": context_preflight.get("status") == "PASS",
        "donor_region_licensed": bool(
            context_audit.get("licenses", {})
            .get("donor", {})
            .get("confirmed_32nt_region_licensed")
        ),
        "acceptor_region_licensed": bool(
            context_audit.get("licenses", {})
            .get("acceptor", {})
            .get("confirmed_32nt_region_licensed")
        ),
        "component_bridge_complete": component_bridge.get("status") == "COMPLETE",
        "component_bridge_replay": bool(replay_errors)
        and max(replay_errors) <= MAX_REPLAY_ERROR,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-preflight", required=True)
    parser.add_argument("--mechanism-discovery", required=True)
    parser.add_argument("--context-preflight", required=True)
    parser.add_argument("--context-discovery", required=True)
    parser.add_argument("--context-audit", required=True)
    parser.add_argument("--component-bridge", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=N_DISCOVERY)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != N_DISCOVERY:
        raise ValueError(f"head-census discovery is frozen at n={N_DISCOVERY}")
    if args.n_boot != N_BOOT or args.n_perm != N_PERM:
        raise ValueError(
            f"inference is frozen at {N_BOOT} bootstraps/{N_PERM} sign flips"
        )
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_HEAD_CENSUS_V1.md"
    )
    paths = {
        "mechanism_preflight": Path(args.mechanism_preflight).resolve(),
        "mechanism_discovery": Path(args.mechanism_discovery).resolve(),
        "context_preflight": Path(args.context_preflight).resolve(),
        "context_discovery": Path(args.context_discovery).resolve(),
        "context_audit": Path(args.context_audit).resolve(),
        "component_bridge": Path(args.component_bridge).resolve(),
    }
    artifacts = {
        name: json.loads(path.read_text()) for name, path in paths.items()
    }
    gate = prerequisite_gate(
        mechanism_preflight=artifacts["mechanism_preflight"],
        context_preflight=artifacts["context_preflight"],
        context_audit=artifacts["context_audit"],
        component_bridge=artifacts["component_bridge"],
    )
    if not all(gate.values()):
        raise RuntimeError(f"head-census prerequisite gate failed: {gate}")

    context_discovery = artifacts["context_discovery"]
    if not context_discovery.get("validation_authorized"):
        raise RuntimeError("context discovery did not authorize either region")
    task_windows = {
        task_name: parse_window(
            context_discovery["freeze"][task_name]["distal_candidate"]
        )
        for task_name in ("donor", "acceptor")
    }
    expected_windows = {"donor": "-80:-48", "acceptor": "+16:+48"}
    if {
        task_name: window.name for task_name, window in task_windows.items()
    } != expected_windows:
        raise RuntimeError("frozen context windows do not match census protocol")

    prior_gene_ids = set()
    for name in (
        "mechanism_preflight",
        "mechanism_discovery",
        "context_preflight",
        "context_discovery",
    ):
        prior_gene_ids.update(raw_gene_ids(artifacts[name]))
    if len(prior_gene_ids) < 128:
        raise RuntimeError("prior-exposed gene exclusion set is unexpectedly small")

    base_model = Path(args.base_model).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)
    examples = load_annotation_gene_hash_cohort(
        args.gtf,
        args.fasta,
        base_split="train",
        analysis_partition="discovery",
        namespace=NAMESPACE,
        n=args.n,
        tag=COHORT_TAG,
        excluded_gene_ids=sorted(prior_gene_ids),
    )
    if {example.gene_id for example in examples} & prior_gene_ids:
        raise RuntimeError("discovery cohort contains a prior-exposed gene")
    normalized_chroms = {
        chrom if chrom.startswith("chr") else f"chr{chrom}"
        for chrom in (example.chrom for example in examples)
    }
    if normalized_chroms & FORBIDDEN_CHROMS:
        raise RuntimeError("discovery cohort opened a forbidden chromosome")
    groups = np.asarray([example.gene_id for example in examples], dtype=object)

    model, tokenizer, report = load_gtex_token_classifier(
        base_model,
        checkpoint,
        device=device,
        attn_implementation="eager",
    )
    if report.seed != 2020:
        raise ValueError(f"discovery requires seed 2020, found {report.seed}")
    task = NativeClassifierTask(
        model,
        tokenizer,
        device=device,
        batch_size=args.batch,
    )
    adapter = SpliceBertAdapter(
        model,
        tokenizer,
        source=str(checkpoint),
        device=device,
    )
    if adapter.num_layers != 6 or adapter.num_heads != NUM_HEADS:
        raise RuntimeError("frozen architecture is 6 layers x 16 heads")

    clean_sequences = [example.sequence for example in examples]
    task_specs: dict[str, dict[str, Any]] = {
        "donor": {
            "anchors": [int(example.donor_pos) for example in examples],
            "targets": [int(example.donor_pos) - 1 for example in examples],
            "label": DONOR_LABEL,
            "window": task_windows["donor"],
        },
        "acceptor": {
            "anchors": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "targets": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "label": ACCEPTOR_LABEL,
            "window": task_windows["acceptor"],
        },
    }
    corrupt_sequences: dict[str, list[str]] = {}
    shuffle_audits: dict[str, list[dict[str, Any]]] = {}
    for task_name, spec in task_specs.items():
        corrupt_sequences[task_name] = []
        shuffle_audits[task_name] = []
        for example, anchor in zip(examples, spec["anchors"]):
            shuffled, audit = shuffle_relative_window(
                example.sequence,
                anchor=anchor,
                window=spec["window"],
                key=(
                    example.example_id,
                    task_name,
                    spec["window"].name,
                    "head-census-v1",
                ),
                draw=0,
            )
            if audit.hamming_fraction < 0.25:
                raise RuntimeError("head-census shuffle failed dose floor")
            corrupt_sequences[task_name].append(shuffled)
            shuffle_audits[task_name].append(audit.to_dict())

    arrays: dict[str, dict[str, Any]] = {}
    for task_index, (task_name, spec) in enumerate(task_specs.items(), start=1):
        arrays[task_name] = {
            "behavior": [],
            "heads": {
                head: {"damage": [], "recovery": []}
                for head in range(NUM_HEADS)
            },
        }
        for start in range(0, len(examples), args.batch):
            stop = min(start + args.batch, len(examples))
            clean_encoded = task.encode(clean_sequences[start:stop])
            corrupt_encoded = task.encode(
                corrupt_sequences[task_name][start:stop]
            )
            target_tokens = donor_token_positions(
                spec["targets"][start:stop],
                prefix_tokens=task.prefix_tokens,
                device=device,
            )
            target_mask = position_mask(
                target_tokens,
                int(clean_encoded["input_ids"].shape[1]),
            )
            clean_inputs = cache_final_layer_inputs(
                model,
                clean_encoded,
                layer=FINAL_LAYER,
            )
            corrupt_inputs = cache_final_layer_inputs(
                model,
                corrupt_encoded,
                layer=FINAL_LAYER,
            )
            clean_run = capture_final_layer_run(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
            corrupt_run = capture_final_layer_run(
                adapter,
                corrupt_inputs,
                layer=FINAL_LAYER,
            )
            clean_margin = classifier_margin(
                model,
                clean_run.hidden,
                target_tokens,
                label=int(spec["label"]),
            )
            corrupt_margin = classifier_margin(
                model,
                corrupt_run.hidden,
                target_tokens,
                label=int(spec["label"]),
            )
            arrays[task_name]["behavior"].extend(
                (clean_margin - corrupt_margin).detach().cpu().tolist()
            )
            clean_source = {FINAL_LAYER: clean_run.heads}
            corrupt_source = {FINAL_LAYER: corrupt_run.heads}
            for head in range(NUM_HEADS):
                with patch_heads(
                    adapter,
                    corrupt_source,
                    [(FINAL_LAYER, head)],
                    token_mask=target_mask,
                ):
                    noised_hidden = replay_final_layer(
                        adapter,
                        clean_inputs,
                        layer=FINAL_LAYER,
                    )
                noised = classifier_margin(
                    model,
                    noised_hidden,
                    target_tokens,
                    label=int(spec["label"]),
                )
                arrays[task_name]["heads"][head]["damage"].extend(
                    (clean_margin - noised).detach().cpu().tolist()
                )
                with patch_heads(
                    adapter,
                    clean_source,
                    [(FINAL_LAYER, head)],
                    token_mask=target_mask,
                ):
                    denoised_hidden = replay_final_layer(
                        adapter,
                        corrupt_inputs,
                        layer=FINAL_LAYER,
                    )
                denoised = classifier_margin(
                    model,
                    denoised_hidden,
                    target_tokens,
                    label=int(spec["label"]),
                )
                arrays[task_name]["heads"][head]["recovery"].extend(
                    (denoised - corrupt_margin).detach().cpu().tolist()
                )
            print(
                f"endpoint {task_index}/2 {task_name}: {stop}/{len(examples)}",
                end="\r",
                flush=True,
            )
            del (
                clean_encoded,
                corrupt_encoded,
                clean_inputs,
                corrupt_inputs,
                clean_run,
                corrupt_run,
            )
        print()

    for task_name in task_specs:
        arrays[task_name]["behavior"] = np.asarray(
            arrays[task_name]["behavior"],
            dtype=np.float64,
        )
        for head in range(NUM_HEADS):
            for direction in ("damage", "recovery"):
                arrays[task_name]["heads"][head][direction] = np.asarray(
                    arrays[task_name]["heads"][head][direction],
                    dtype=np.float64,
                )

    behavior_summaries = {
        task_name: summarize(
            arrays[task_name]["behavior"],
            groups,
            seed=180_000 + task_index,
        )
        for task_index, task_name in enumerate(task_specs)
    }
    behavior_qvalues = benjamini_hochberg(
        [
            float(behavior_summaries[task_name]["p_signflip_add_one"])
            for task_name in task_specs
        ]
    )
    for task_name, qvalue in zip(task_specs, behavior_qvalues):
        behavior_summaries[task_name]["q_bh_2_behavior_tests"] = float(qvalue)

    head_summaries: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    pvalue_entries: list[tuple[str, int, str, float]] = []
    summary_seed = 200_000
    for task_name in task_specs:
        head_summaries[task_name] = {}
        for head in range(NUM_HEADS):
            head_summaries[task_name][head] = {}
            for direction in ("damage", "recovery"):
                result = summarize(
                    arrays[task_name]["heads"][head][direction],
                    groups,
                    seed=summary_seed,
                )
                head_summaries[task_name][head][direction] = result
                pvalue_entries.append(
                    (
                        task_name,
                        head,
                        direction,
                        float(result["p_signflip_add_one"]),
                    )
                )
                summary_seed += 1
    if len(pvalue_entries) != 64:
        raise RuntimeError(f"expected 64 census tests, found {len(pvalue_entries)}")
    qvalues = benjamini_hochberg([entry[3] for entry in pvalue_entries])
    for (task_name, head, direction, _), qvalue in zip(
        pvalue_entries,
        qvalues,
    ):
        head_summaries[task_name][head][direction][
            "q_bh_64_tests"
        ] = float(qvalue)

    behavior_gates = {
        task_name: {
            "mean_at_least_0_05": bool(
                behavior_summaries[task_name]["estimate"] >= MIN_EFFECT
            ),
            "ci_above_zero": bool(
                behavior_summaries[task_name]["ci_low"] > 0
            ),
            "q_at_most_0_05": bool(
                behavior_summaries[task_name]["q_bh_2_behavior_tests"] <= 0.05
            ),
        }
        for task_name in task_specs
    }
    candidates: dict[str, list[int]] = {}
    controls: dict[str, dict[int, list[int]]] = {}
    for task_name in task_specs:
        candidates[task_name] = (
            select_head_candidates(
                head_summaries[task_name],
                min_effect=MIN_EFFECT,
                max_q=MAX_DISCOVERY_Q,
                max_candidates=MAX_CANDIDATES,
                q_field="q_bh_64_tests",
            )
            if all(behavior_gates[task_name].values())
            else []
        )
        controls[task_name] = deterministic_control_heads(
            candidates[task_name],
            num_heads=NUM_HEADS,
            controls_per_candidate=CONTROLS_PER_CANDIDATE,
            tag=f"gtex-native-context-head-census-controls-v1:{task_name}",
        )

    raw = []
    for index, example in enumerate(examples):
        raw.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "chrom": example.chrom,
                "tasks": {
                    task_name: {
                        "behavior": float(arrays[task_name]["behavior"][index]),
                        "heads": {
                            f"L{FINAL_LAYER}H{head}": {
                                direction: float(
                                    arrays[task_name]["heads"][head][direction][
                                        index
                                    ]
                                )
                                for direction in ("damage", "recovery")
                            }
                            for head in range(NUM_HEADS)
                        },
                        "shuffle_audit": shuffle_audits[task_name][index],
                    }
                    for task_name in task_specs
                },
            }
        )

    freeze = {
        task_name: {
            "candidate_heads": [
                f"L{FINAL_LAYER}H{head}" for head in candidates[task_name]
            ],
            "control_heads": {
                f"L{FINAL_LAYER}H{candidate}": [
                    f"L{FINAL_LAYER}H{head}" for head in candidate_controls
                ]
                for candidate, candidate_controls in controls[task_name].items()
            },
            "confirmation_authorized": bool(candidates[task_name]),
        }
        for task_name in task_specs
    }
    result = {
        "study": "native GTEx context head census",
        "stage": "complete L5 train discovery",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "evidence_tier": "exploratory all-head discovery",
        "device": device,
        "split": "train gene-hash discovery",
        "cohort_tag": COHORT_TAG,
        "cohort_namespace": NAMESPACE,
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "prior_exposed_genes_excluded": len(prior_gene_ids),
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMS),
        "checkpoint": report.to_dict(),
        "regions": {
            task_name: spec["window"].name
            for task_name, spec in task_specs.items()
        },
        "prerequisite_gate": gate,
        "behavior_summaries": behavior_summaries,
        "behavior_gates": behavior_gates,
        "head_summaries": {
            task_name: {
                f"L{FINAL_LAYER}H{head}": result
                for head, result in task_results.items()
            }
            for task_name, task_results in head_summaries.items()
        },
        "multiplicity_family": len(pvalue_entries),
        "candidate_rule": {
            "min_damage_and_recovery": MIN_EFFECT,
            "both_ci_low_above_zero": True,
            "both_q_bh_64_tests_at_most": MAX_DISCOVERY_Q,
            "max_candidates_per_endpoint": MAX_CANDIDATES,
            "rank": "descending min(damage,recovery), head index tie-break",
        },
        "freeze": freeze,
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_head_census_discovery_v1",
        sys.argv,
        repo=REPO,
        status="EXPLORATORY",
        config={
            "split": result["split"],
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "cohort_tag": COHORT_TAG,
            "cohort_namespace": NAMESPACE,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "heads_scanned": NUM_HEADS,
            "multiplicity_family": len(pvalue_entries),
            "candidate_rule": result["candidate_rule"],
        },
        seeds={
            "checkpoint": 2020,
            "summary_base": 200_000,
        },
        notes=[
            "All 16 layer-5 heads were scanned for both endpoints and directions.",
            "Prior train-exposed genes and all validation/test chromosomes were excluded.",
            "Candidate and equal-size control identities are frozen before confirmation.",
            "The fresh split is analysis-sealed, not claimed absent from checkpoint training.",
        ],
    )
    for path in (
        protocol,
        *paths.values(),
        model_weight(base_model),
        base_model / "config.json",
        base_model / "tokenizer.json",
        checkpoint,
        Path(args.gtf),
        Path(args.fasta),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "behavior_gates": behavior_gates,
                "freeze": freeze,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"COMPLETE: saved -> {output}")
    del task, adapter, model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
