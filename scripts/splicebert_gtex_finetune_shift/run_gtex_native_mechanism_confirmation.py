#!/usr/bin/env python
"""Five-seed validation of frozen native GTEx final-layer components."""

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
from gfmcd.splicing.circuit import clamp_write, pick_device  # noqa: E402
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.grouped_stats import (  # noqa: E402
    grouped_bootstrap,
    grouped_bootstrap_ratio,
    grouped_signflip_pvalue,
)
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    NativeClassifierTask,
    load_annotation_cohort,
)
from gfmcd.splicing.native_mechanism import (  # noqa: E402
    cache_final_layer_inputs,
    capture_final_layer_run,
    donor_margin,
    donor_token_positions,
    position_mask,
    replay_final_layer,
)

EXPECTED_SEEDS = frozenset(range(2020, 2025))
FINAL_LAYER = 5
MIN_EFFECT = 0.05
MAX_Q = 0.05
MIN_HEALTH_RETENTION = 0.75
REPLAY_ATOL = 2e-5


def parse_checkpoints(values: list[str]) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for value in values:
        try:
            seed_text, path_text = value.split("=", 1)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError("--checkpoint must be SEED=PATH") from exc
        if seed in checkpoints:
            raise ValueError(f"duplicate seed {seed}")
        checkpoints[seed] = Path(path_text).expanduser().resolve()
    if set(checkpoints) != set(EXPECTED_SEEDS):
        raise ValueError(f"complete seed family required; found {sorted(checkpoints)}")
    return checkpoints


def head_number(name: str) -> int:
    prefix = f"L{FINAL_LAYER}H"
    if not name.startswith(prefix):
        raise ValueError(f"expected a final-layer head, found {name!r}")
    return int(name.removeprefix(prefix))


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model weights not found in {model_dir}")


def summary(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    n_perm: int | None,
    seed: int,
) -> dict[str, Any]:
    result = grouped_bootstrap(values, groups, n_boot=n_boot, seed=seed)
    if n_perm is not None:
        result["p_signflip_add_one"] = grouped_signflip_pvalue(
            values,
            groups,
            n_perm=n_perm,
            seed=seed + 100_000,
        )
    return result


def apply_qvalues(
    entries: list[tuple[dict[str, Any], str]],
    *,
    field: str,
) -> None:
    qvalues = benjamini_hochberg(
        [float(result["p_signflip_add_one"]) for result, _ in entries]
    )
    for (result, _), qvalue in zip(entries, qvalues):
        result[field] = float(qvalue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--native-behavior", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=9999)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != 400:
        raise ValueError("native-mechanism confirmation is frozen at n=400")
    if args.n_boot != 5000 or args.n_perm != 9999:
        raise ValueError("confirmation inference is frozen at 5000/9999")
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_MECHANISM_V1.md"
    )
    discovery_path = Path(args.discovery).expanduser().resolve()
    discovery = json.loads(discovery_path.read_text())
    freeze = discovery.get("freeze", {})
    if not freeze.get("validation_authorized"):
        raise RuntimeError("train discovery did not authorize validation")
    candidate_names = list(freeze["candidate_heads"])
    candidates = [head_number(name) for name in candidate_names]
    controls = {
        head_number(candidate): [head_number(name) for name in names]
        for candidate, names in freeze["control_heads"].items()
    }
    if set(candidates) != set(controls):
        raise RuntimeError("candidate/control freeze is incomplete")
    all_heads = sorted(
        set(candidates)
        | {head for values in controls.values() for head in values}
    )
    if len(all_heads) != len(candidates) * 4:
        raise RuntimeError("confirmation requires three unique controls per candidate")

    native_path = Path(args.native_behavior).expanduser().resolve()
    native = json.loads(native_path.read_text())
    if not native.get("native_coupling_gate", {}).get("PASS"):
        raise RuntimeError("native behavior gate is not open")
    base_model = Path(args.base_model).expanduser().resolve()
    checkpoints = parse_checkpoints(args.checkpoint)
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)
    examples = load_annotation_cohort(
        args.gtf,
        args.fasta,
        split="val",
        n=args.n,
        tag="sealed-unc-val",
    )
    example_ids = [example.example_id for example in examples]
    groups = np.asarray([example.gene_id for example in examples], dtype=object)

    seed_arrays: dict[str, dict[str, Any]] = {}
    seed_summaries: dict[str, Any] = {}
    replay: dict[str, Any] = {}
    for seed, checkpoint in sorted(checkpoints.items()):
        print(f"seed {seed}: frozen component confirmation")
        model, tokenizer, report = load_gtex_token_classifier(
            base_model,
            checkpoint,
            device=device,
            attn_implementation="eager",
        )
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
        arrays: dict[str, Any] = {
            "clean_margin": [],
            "corrupt_margin": [],
            "local_donor_margin": [],
            "heads": {
                head: {"damage": [], "recovery": []}
                for head in all_heads
            },
            "local_intervened_drop": {
                head: [] for head in candidates
            },
            "mlp": {"damage": [], "recovery": []},
        }

        for start in range(0, len(examples), args.batch):
            chunk = examples[start : start + args.batch]
            clean_encoded = task.encode([example.sequence for example in chunk])
            corrupt_encoded = task.encode(
                [
                    substitute(
                        example.sequence,
                        int(example.next_acceptor_pos) - 2,
                        "CC",
                    )
                    for example in chunk
                ]
            )
            local_encoded = task.encode(
                [
                    substitute(example.sequence, example.donor_pos, "CC")
                    for example in chunk
                ]
            )
            donor_tokens = donor_token_positions(
                [example.donor_pos - 1 for example in chunk],
                prefix_tokens=task.prefix_tokens,
                device=device,
            )
            donor_mask = position_mask(
                donor_tokens,
                int(clean_encoded["input_ids"].shape[1]),
            )
            donor_positions_2d = donor_tokens[:, None]
            clean_inputs = cache_final_layer_inputs(
                model, clean_encoded, layer=FINAL_LAYER
            )
            corrupt_inputs = cache_final_layer_inputs(
                model, corrupt_encoded, layer=FINAL_LAYER
            )
            local_inputs = cache_final_layer_inputs(
                model, local_encoded, layer=FINAL_LAYER
            )
            clean_run = capture_final_layer_run(
                adapter, clean_inputs, layer=FINAL_LAYER
            )
            corrupt_run = capture_final_layer_run(
                adapter, corrupt_inputs, layer=FINAL_LAYER
            )
            local_run = capture_final_layer_run(
                adapter, local_inputs, layer=FINAL_LAYER
            )
            clean_margin = donor_margin(model, clean_run.hidden, donor_tokens)
            corrupt_margin = donor_margin(model, corrupt_run.hidden, donor_tokens)
            local_margin = donor_margin(model, local_run.hidden, donor_tokens)
            arrays["clean_margin"].extend(clean_margin.detach().cpu().tolist())
            arrays["corrupt_margin"].extend(corrupt_margin.detach().cpu().tolist())
            arrays["local_donor_margin"].extend(local_margin.detach().cpu().tolist())
            clean_source = {FINAL_LAYER: clean_run.heads}
            corrupt_source = {FINAL_LAYER: corrupt_run.heads}

            noised_margins: dict[int, torch.Tensor] = {}
            for head in all_heads:
                with patch_heads(
                    adapter,
                    corrupt_source,
                    [(FINAL_LAYER, head)],
                    token_mask=donor_mask,
                ):
                    noised_hidden = replay_final_layer(
                        adapter, clean_inputs, layer=FINAL_LAYER
                    )
                noised = donor_margin(model, noised_hidden, donor_tokens)
                noised_margins[head] = noised
                arrays["heads"][head]["damage"].extend(
                    (clean_margin - noised).detach().cpu().tolist()
                )

                with patch_heads(
                    adapter,
                    clean_source,
                    [(FINAL_LAYER, head)],
                    token_mask=donor_mask,
                ):
                    denoised_hidden = replay_final_layer(
                        adapter, corrupt_inputs, layer=FINAL_LAYER
                    )
                denoised = donor_margin(model, denoised_hidden, donor_tokens)
                arrays["heads"][head]["recovery"].extend(
                    (denoised - corrupt_margin).detach().cpu().tolist()
                )

            for head in candidates:
                with patch_heads(
                    adapter,
                    corrupt_source,
                    [(FINAL_LAYER, head)],
                    token_mask=donor_mask,
                ):
                    local_patched_hidden = replay_final_layer(
                        adapter, local_inputs, layer=FINAL_LAYER
                    )
                local_patched = donor_margin(
                    model, local_patched_hidden, donor_tokens
                )
                arrays["local_intervened_drop"][head].extend(
                    (noised_margins[head] - local_patched).detach().cpu().tolist()
                )

            with clamp_write(
                adapter,
                [(FINAL_LAYER, "mlp")],
                {(FINAL_LAYER, "mlp"): corrupt_run.mlp_write},
                positions=donor_positions_2d,
            ):
                mlp_noised_hidden = replay_final_layer(
                    adapter, clean_inputs, layer=FINAL_LAYER
                )
            mlp_noised = donor_margin(model, mlp_noised_hidden, donor_tokens)
            arrays["mlp"]["damage"].extend(
                (clean_margin - mlp_noised).detach().cpu().tolist()
            )

            with clamp_write(
                adapter,
                [(FINAL_LAYER, "mlp")],
                {(FINAL_LAYER, "mlp"): clean_run.mlp_write},
                positions=donor_positions_2d,
            ):
                mlp_denoised_hidden = replay_final_layer(
                    adapter, corrupt_inputs, layer=FINAL_LAYER
                )
            mlp_denoised = donor_margin(model, mlp_denoised_hidden, donor_tokens)
            arrays["mlp"]["recovery"].extend(
                (mlp_denoised - corrupt_margin).detach().cpu().tolist()
            )
            print(
                f"  {min(start + len(chunk), len(examples))}/{len(examples)}",
                end="\r",
                flush=True,
            )
        print()

        arrays["clean_margin"] = np.asarray(arrays["clean_margin"], dtype=np.float64)
        arrays["corrupt_margin"] = np.asarray(arrays["corrupt_margin"], dtype=np.float64)
        arrays["local_donor_margin"] = np.asarray(
            arrays["local_donor_margin"], dtype=np.float64
        )
        for head in all_heads:
            for direction in ("damage", "recovery"):
                arrays["heads"][head][direction] = np.asarray(
                    arrays["heads"][head][direction], dtype=np.float64
                )
        for head in candidates:
            arrays["local_intervened_drop"][head] = np.asarray(
                arrays["local_intervened_drop"][head], dtype=np.float64
            )
        for direction in ("damage", "recovery"):
            arrays["mlp"][direction] = np.asarray(
                arrays["mlp"][direction], dtype=np.float64
            )
        seed_arrays[str(seed)] = arrays

        expected = native["raw"][str(seed)]
        if [row["example_id"] for row in expected] != example_ids:
            raise RuntimeError(f"seed {seed} native artifact cohort mismatch")
        expected_clean = np.asarray(
            [row["clean_donor_margin"] for row in expected], dtype=np.float64
        )
        expected_gap = np.asarray(
            [row["partner_cc"] for row in expected], dtype=np.float64
        )
        replay[str(seed)] = {
            "clean_margin_max_abs_error": float(
                np.max(np.abs(arrays["clean_margin"] - expected_clean))
            ),
            "coupling_max_abs_error": float(
                np.max(
                    np.abs(
                        (arrays["clean_margin"] - arrays["corrupt_margin"])
                        - expected_gap
                    )
                )
            ),
            "atol": REPLAY_ATOL,
        }
        replay[str(seed)]["PASS"] = (
            replay[str(seed)]["clean_margin_max_abs_error"] <= REPLAY_ATOL
            and replay[str(seed)]["coupling_max_abs_error"] <= REPLAY_ATOL
        )

        local_drop = arrays["clean_margin"] - arrays["local_donor_margin"]
        seed_summaries[str(seed)] = {
            "checkpoint": report.to_dict(),
            "behavior_gap": summary(
                arrays["clean_margin"] - arrays["corrupt_margin"],
                groups,
                n_boot=args.n_boot,
                n_perm=None,
                seed=seed,
            ),
            "local_donor_drop": summary(
                local_drop,
                groups,
                n_boot=args.n_boot,
                n_perm=None,
                seed=seed + 1,
            ),
            "heads": {
                f"L{FINAL_LAYER}H{head}": {
                    direction: summary(
                        arrays["heads"][head][direction],
                        groups,
                        n_boot=args.n_boot,
                        n_perm=None,
                        seed=seed + 100 * head + index,
                    )
                    for index, direction in enumerate(("damage", "recovery"))
                }
                for head in all_heads
            },
            "mlp": {
                direction: summary(
                    arrays["mlp"][direction],
                    groups,
                    n_boot=args.n_boot,
                    n_perm=None,
                    seed=seed + 50_000 + index,
                )
                for index, direction in enumerate(("damage", "recovery"))
            },
        }
        del task, adapter, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    names = [str(seed) for seed in sorted(EXPECTED_SEEDS)]
    family: dict[str, Any] = {
        "heads": {},
        "candidate_vs_controls": {},
        "health": {},
        "mlp": {},
    }
    primary_entries: list[tuple[dict[str, Any], str]] = []
    control_entries: list[tuple[dict[str, Any], str]] = []
    for candidate in candidates:
        candidate_name = f"L{FINAL_LAYER}H{candidate}"
        family["heads"][candidate_name] = {}
        family["candidate_vs_controls"][candidate_name] = {}
        for direction_index, direction in enumerate(("damage", "recovery")):
            candidate_values = np.mean(
                np.stack(
                    [
                        seed_arrays[name]["heads"][candidate][direction]
                        for name in names
                    ]
                ),
                axis=0,
            )
            candidate_summary = summary(
                candidate_values,
                groups,
                n_boot=args.n_boot,
                n_perm=args.n_perm,
                seed=70_000 + 100 * candidate + direction_index,
            )
            candidate_summary["positive_seed_count"] = int(
                sum(
                    seed_summaries[name]["heads"][candidate_name][direction][
                        "estimate"
                    ]
                    > 0
                    for name in names
                )
            )
            family["heads"][candidate_name][direction] = candidate_summary
            primary_entries.append((candidate_summary, candidate_name + direction))

            control_mean = np.mean(
                np.stack(
                    [
                        seed_arrays[name]["heads"][head][direction]
                        for name in names
                        for head in controls[candidate]
                    ]
                ),
                axis=0,
            )
            contrast = candidate_values - control_mean
            contrast_summary = summary(
                contrast,
                groups,
                n_boot=args.n_boot,
                n_perm=args.n_perm,
                seed=80_000 + 100 * candidate + direction_index,
            )
            family["candidate_vs_controls"][candidate_name][
                direction
            ] = contrast_summary
            control_entries.append(
                (contrast_summary, candidate_name + direction)
            )

        local_drop_family = np.mean(
            np.stack(
                [
                    seed_arrays[name]["clean_margin"]
                    - seed_arrays[name]["local_donor_margin"]
                    for name in names
                ]
            ),
            axis=0,
        )
        intervened_drop_family = np.mean(
            np.stack(
                [
                    seed_arrays[name]["local_intervened_drop"][candidate]
                    for name in names
                ]
            ),
            axis=0,
        )
        family["health"][candidate_name] = grouped_bootstrap_ratio(
            intervened_drop_family,
            local_drop_family,
            groups,
            n_boot=args.n_boot,
            seed=90_000 + candidate,
        )

    apply_qvalues(
        primary_entries,
        field=f"q_bh_{len(primary_entries)}_candidate_tests",
    )
    apply_qvalues(
        control_entries,
        field=f"q_bh_{len(control_entries)}_control_contrasts",
    )
    mlp_entries: list[tuple[dict[str, Any], str]] = []
    for direction_index, direction in enumerate(("damage", "recovery")):
        values = np.mean(
            np.stack(
                [seed_arrays[name]["mlp"][direction] for name in names]
            ),
            axis=0,
        )
        result_summary = summary(
            values,
            groups,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=100_000 + direction_index,
        )
        result_summary["positive_seed_count"] = int(
            sum(
                seed_summaries[name]["mlp"][direction]["estimate"] > 0
                for name in names
            )
        )
        family["mlp"][direction] = result_summary
        mlp_entries.append((result_summary, direction))
    apply_qvalues(mlp_entries, field="q_bh_2_mlp_tests")

    confirmed: list[str] = []
    verdicts: dict[str, Any] = {}
    primary_q_field = f"q_bh_{len(primary_entries)}_candidate_tests"
    control_q_field = f"q_bh_{len(control_entries)}_control_contrasts"
    for candidate in candidates:
        name = f"L{FINAL_LAYER}H{candidate}"
        checks = {}
        for direction in ("damage", "recovery"):
            effect = family["heads"][name][direction]
            contrast = family["candidate_vs_controls"][name][direction]
            checks[f"{direction}_effect"] = bool(
                effect["estimate"] >= MIN_EFFECT
                and effect["ci_low"] > 0
                and effect[primary_q_field] <= MAX_Q
                and effect["positive_seed_count"] >= 4
            )
            checks[f"{direction}_vs_controls"] = bool(
                contrast["ci_low"] > 0
                and contrast[control_q_field] <= MAX_Q
            )
        checks["local_donor_health"] = bool(
            family["health"][name]["estimate"] >= MIN_HEALTH_RETENTION
        )
        verdicts[name] = {"checks": checks, "PASS": all(checks.values())}
        if verdicts[name]["PASS"]:
            confirmed.append(name)

    mlp_checks = {
        direction: bool(
            family["mlp"][direction]["estimate"] >= MIN_EFFECT
            and family["mlp"][direction]["ci_low"] > 0
            and family["mlp"][direction]["q_bh_2_mlp_tests"] <= MAX_Q
            and family["mlp"][direction]["positive_seed_count"] >= 4
        )
        for direction in ("damage", "recovery")
    }
    all_replay = all(row["PASS"] for row in replay.values())
    result = {
        "study": "native GTEx final-layer mechanism confirmation",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "PASS" if all_replay else "FAIL_INSTRUMENT",
        "device": device,
        "split": "val",
        "cohort_tag": "sealed-unc-val",
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "freeze": freeze,
        "replay": replay,
        "seed_summaries": seed_summaries,
        "family": family,
        "verdicts": {
            "heads": verdicts,
            "confirmed_heads": confirmed,
            "head_mechanism_gate": bool(confirmed) and all_replay,
            "mlp": {"checks": mlp_checks, "PASS": all(mlp_checks.values())},
            "role_decomposition_authorized": bool(confirmed) and all_replay,
        },
        "thresholds": {
            "min_effect": MIN_EFFECT,
            "max_q": MAX_Q,
            "min_positive_seeds": 4,
            "min_local_donor_health_retention": MIN_HEALTH_RETENTION,
            "replay_atol": REPLAY_ATOL,
        },
        "raw": {
            name: [
                {
                    "example_id": examples[index].example_id,
                    "gene_id": examples[index].gene_id,
                    "clean_margin": seed_arrays[name]["clean_margin"][index],
                    "corrupt_margin": seed_arrays[name]["corrupt_margin"][index],
                    "local_donor_drop": (
                        seed_arrays[name]["clean_margin"][index]
                        - seed_arrays[name]["local_donor_margin"][index]
                    ),
                    "heads": {
                        f"L{FINAL_LAYER}H{head}": {
                            "damage": seed_arrays[name]["heads"][head]["damage"][
                                index
                            ],
                            "recovery": seed_arrays[name]["heads"][head][
                                "recovery"
                            ][index],
                            **(
                                {
                                    "local_intervened_drop": seed_arrays[name][
                                        "local_intervened_drop"
                                    ][head][index]
                                }
                                if head in candidates
                                else {}
                            ),
                        }
                        for head in all_heads
                    },
                    "mlp": {
                        direction: seed_arrays[name]["mlp"][direction][index]
                        for direction in ("damage", "recovery")
                    },
                }
                for index in range(len(examples))
            ]
            for name in names
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else value
            ),
        )
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_mechanism_confirmation_v1",
        sys.argv,
        repo=REPO,
        status="CONFIRMATORY" if all_replay else "FAILED",
        config={
            "split": "val",
            "n": args.n,
            "batch_size": args.batch,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "cohort_tag": "sealed-unc-val",
            "final_layer": FINAL_LAYER,
            "candidate_heads": candidate_names,
            "control_heads": freeze["control_heads"],
            "thresholds": result["thresholds"],
        },
        seeds={
            "checkpoint_min": min(EXPECTED_SEEDS),
            "checkpoint_max": max(EXPECTED_SEEDS),
        },
        inputs=[
            file_record(protocol),
            file_record(discovery_path),
            file_record(native_path),
            file_record(base_model / "config.json"),
            file_record(model_weight(base_model)),
            *[file_record(path) for path in checkpoints.values()],
            file_record(args.gtf),
            file_record(args.fasta),
        ],
        notes=[
            "Candidates and three equal-size controls each were frozen on train.",
            "All intervention identities were applied unchanged across five validation seeds.",
            "Local donor health is a ratio of means under the same paired-corrupt head clamp in both arms.",
            "Role/source analysis remains gated on this confirmation artifact.",
        ],
    )
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "replay": replay,
                "confirmed_heads": confirmed,
                "head_verdicts": verdicts,
                "mlp": result["verdicts"]["mlp"],
                "family": family,
            },
            indent=2,
        )
    )
    print(f"saved -> {output}")
    if not all_replay:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
