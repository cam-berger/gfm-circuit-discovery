#!/usr/bin/env python
"""QK/V, MLP-interaction, source, and specificity analysis for confirmed GTEx heads."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import itertools
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
from gfmcd.splicing.circuit import (  # noqa: E402
    clamp_write,
    matched_decoy_acceptor,
    matched_decoy_donor,
    patch_proj,
    pick_device,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.grouped_stats import (  # noqa: E402
    grouped_bootstrap,
    grouped_signflip_pvalue,
)
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    NativeClassifierTask,
    linear_cka,
    load_annotation_cohort,
)
from gfmcd.splicing.native_mechanism import (  # noqa: E402
    FinalLayerRun,
    cache_final_layer_inputs,
    capture_final_layer_run,
    donor_margin,
    donor_token_positions,
    orthogonal_procrustes_cosine,
    position_mask,
    replay_final_layer,
)

EXPECTED_SEEDS = frozenset(range(2020, 2025))
FINAL_LAYER = 5
REPLAY_ATOL = 2e-5
MIN_ROLE_DIFFERENCE = 0.05
MIN_SOURCE_EFFECT = 0.05
MAX_Q = 0.05
REGIONS = ("acceptor", "donor", "nonpartner_ag", "upstream", "downstream")


def parse_checkpoints(values: list[str]) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in checkpoints:
            raise ValueError(f"duplicate seed {seed}")
        checkpoints[seed] = Path(path_text).expanduser().resolve()
    if set(checkpoints) != set(EXPECTED_SEEDS):
        raise ValueError(f"complete seed family required; found {sorted(checkpoints)}")
    return checkpoints


def head_number(name: str) -> int:
    prefix = f"L{FINAL_LAYER}H"
    if not name.startswith(prefix):
        raise ValueError(f"expected final-layer head, found {name}")
    return int(name.removeprefix(prefix))


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model weights not found in {model_dir}")


def infer(
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


def add_qvalues(results: list[dict[str, Any]], field: str) -> None:
    qvalues = benjamini_hochberg(
        [float(result["p_signflip_add_one"]) for result in results]
    )
    for result, qvalue in zip(results, qvalues):
        result[field] = float(qvalue)


def projection_intervention(
    adapter: SpliceBertAdapter,
    target_inputs: Any,
    source: FinalLayerRun,
    *,
    head: int,
    query_positions: torch.Tensor,
    source_positions: torch.Tensor,
    patch_query: bool,
    patch_key: bool,
    patch_value: bool,
) -> torch.Tensor:
    with ExitStack() as stack:
        if patch_query:
            stack.enter_context(
                patch_proj(
                    adapter,
                    FINAL_LAYER,
                    head,
                    "query",
                    source.query,
                    query_positions,
                )
            )
        if patch_key:
            stack.enter_context(
                patch_proj(
                    adapter,
                    FINAL_LAYER,
                    head,
                    "key",
                    source.key,
                    source_positions,
                )
            )
        if patch_value:
            stack.enter_context(
                patch_proj(
                    adapter,
                    FINAL_LAYER,
                    head,
                    "value",
                    source.value,
                    source_positions,
                )
            )
        return replay_final_layer(adapter, target_inputs, layer=FINAL_LAYER)


def window_tensor(
    centers: list[int],
    offsets: tuple[int, ...],
    *,
    prefix_tokens: int,
    device: str,
) -> torch.Tensor:
    return torch.as_tensor(
        [
            [center + prefix_tokens + offset for offset in offsets]
            for center in centers
        ],
        dtype=torch.long,
        device=device,
    )


def attention_mass(
    attention: torch.Tensor,
    *,
    head: int,
    query_positions: torch.Tensor,
    source_positions: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(attention.shape[0], device=attention.device)[:, None]
    queries = query_positions[:, None]
    return attention[rows, head, queries, source_positions].sum(dim=1)


def nanmean_stack(values: list[np.ndarray]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(values), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", required=True)
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

    if args.n != 400 or args.n_boot != 5000 or args.n_perm != 9999:
        raise ValueError("role/source assay is frozen at n=400, 5000/9999")
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_MECHANISM_V1.md"
    )
    confirmation_path = Path(args.confirmation).expanduser().resolve()
    confirmation = json.loads(confirmation_path.read_text())
    verdicts = confirmation.get("verdicts", {})
    if not verdicts.get("role_decomposition_authorized"):
        raise RuntimeError("component confirmation did not authorize role analysis")
    confirmed_names = list(verdicts["confirmed_heads"])
    if len(confirmed_names) != 1:
        raise RuntimeError("v1 role runner requires exactly one confirmed head")
    head_name = confirmed_names[0]
    head = head_number(head_name)
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
    decoy_acceptors = [matched_decoy_acceptor(example) for example in examples]
    decoy_donors = [matched_decoy_donor(example) for example in examples]

    seed_arrays: dict[str, dict[str, Any]] = {}
    qkv_replay: dict[str, float] = {}
    confirmation_replay: dict[str, float] = {}
    for seed, checkpoint in sorted(checkpoints.items()):
        print(f"seed {seed}: {head_name} role/source")
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
            "clean": [],
            "corrupt": [],
            "whole": {"damage": [], "recovery": []},
            "qk": {"damage": [], "recovery": []},
            "v": {"damage": [], "recovery": []},
            "qkv": {"damage": [], "recovery": []},
            "mlp_clamped_damage": [],
            "mlp_interaction": [],
            "source": {region: [] for region in REGIONS},
            "source_full": [],
            "source_noop": [],
            "nonpartner_head_damage": [],
            "other_donor_head_damage": [],
            "delta_context": [],
            "attention_clean": {region: [] for region in REGIONS},
            "attention_corrupt": {region: [] for region in REGIONS},
        }
        max_qkv_error = 0.0
        for start in range(0, len(examples), args.batch):
            chunk = examples[start : start + args.batch]
            chunk_decoy_acceptors = decoy_acceptors[start : start + len(chunk)]
            chunk_decoy_donors = decoy_donors[start : start + len(chunk)]
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
            nonpartner_encoded = task.encode(
                [
                    (
                        substitute(example.sequence, int(decoy) - 2, "CC")
                        if decoy is not None
                        else example.sequence
                    )
                    for example, decoy in zip(chunk, chunk_decoy_acceptors)
                ]
            )
            donor_tokens = donor_token_positions(
                [example.donor_pos - 1 for example in chunk],
                prefix_tokens=task.prefix_tokens,
                device=device,
            )
            sequence_length = int(clean_encoded["input_ids"].shape[1])
            donor_mask = position_mask(donor_tokens, sequence_length)
            donor_positions_2d = donor_tokens[:, None]
            all_positions = torch.arange(
                sequence_length,
                device=donor_tokens.device,
                dtype=torch.long,
            )[None, :].expand(len(chunk), sequence_length)
            empty_positions = torch.empty(
                (len(chunk), 0),
                device=donor_tokens.device,
                dtype=torch.long,
            )
            clean_inputs = cache_final_layer_inputs(
                model, clean_encoded, layer=FINAL_LAYER
            )
            corrupt_inputs = cache_final_layer_inputs(
                model, corrupt_encoded, layer=FINAL_LAYER
            )
            nonpartner_inputs = cache_final_layer_inputs(
                model, nonpartner_encoded, layer=FINAL_LAYER
            )
            clean_run = capture_final_layer_run(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
                output_attentions=True,
            )
            corrupt_run = capture_final_layer_run(
                adapter,
                corrupt_inputs,
                layer=FINAL_LAYER,
                output_attentions=True,
            )
            nonpartner_run = capture_final_layer_run(
                adapter,
                nonpartner_inputs,
                layer=FINAL_LAYER,
            )
            if clean_run.attention is None or corrupt_run.attention is None:
                raise RuntimeError("eager attention capture failed")
            clean_margin = donor_margin(model, clean_run.hidden, donor_tokens)
            corrupt_margin = donor_margin(model, corrupt_run.hidden, donor_tokens)
            arrays["clean"].extend(clean_margin.detach().cpu().tolist())
            arrays["corrupt"].extend(corrupt_margin.detach().cpu().tolist())
            clean_source = {FINAL_LAYER: clean_run.heads}
            corrupt_source = {FINAL_LAYER: corrupt_run.heads}

            with patch_heads(
                adapter,
                corrupt_source,
                [(FINAL_LAYER, head)],
                token_mask=donor_mask,
            ):
                whole_noised_hidden = replay_final_layer(
                    adapter, clean_inputs, layer=FINAL_LAYER
                )
            whole_noised = donor_margin(model, whole_noised_hidden, donor_tokens)
            whole_damage = clean_margin - whole_noised
            arrays["whole"]["damage"].extend(
                whole_damage.detach().cpu().tolist()
            )
            with patch_heads(
                adapter,
                clean_source,
                [(FINAL_LAYER, head)],
                token_mask=donor_mask,
            ):
                whole_denoised_hidden = replay_final_layer(
                    adapter, corrupt_inputs, layer=FINAL_LAYER
                )
            whole_denoised = donor_margin(
                model, whole_denoised_hidden, donor_tokens
            )
            whole_recovery = whole_denoised - corrupt_margin
            arrays["whole"]["recovery"].extend(
                whole_recovery.detach().cpu().tolist()
            )

            for direction, target_inputs, source, baseline, sign in (
                ("damage", clean_inputs, corrupt_run, clean_margin, -1),
                ("recovery", corrupt_inputs, clean_run, corrupt_margin, 1),
            ):
                qk_hidden = projection_intervention(
                    adapter,
                    target_inputs,
                    source,
                    head=head,
                    query_positions=donor_positions_2d,
                    source_positions=all_positions,
                    patch_query=True,
                    patch_key=True,
                    patch_value=False,
                )
                v_hidden = projection_intervention(
                    adapter,
                    target_inputs,
                    source,
                    head=head,
                    query_positions=donor_positions_2d,
                    source_positions=all_positions,
                    patch_query=False,
                    patch_key=False,
                    patch_value=True,
                )
                qkv_hidden = projection_intervention(
                    adapter,
                    target_inputs,
                    source,
                    head=head,
                    query_positions=donor_positions_2d,
                    source_positions=all_positions,
                    patch_query=True,
                    patch_key=True,
                    patch_value=True,
                )
                qk_margin = donor_margin(model, qk_hidden, donor_tokens)
                v_margin = donor_margin(model, v_hidden, donor_tokens)
                qkv_margin = donor_margin(model, qkv_hidden, donor_tokens)
                if direction == "damage":
                    qk_effect = baseline - qk_margin
                    v_effect = baseline - v_margin
                    qkv_effect = baseline - qkv_margin
                    whole_effect = whole_damage
                else:
                    qk_effect = qk_margin - baseline
                    v_effect = v_margin - baseline
                    qkv_effect = qkv_margin - baseline
                    whole_effect = whole_recovery
                arrays["qk"][direction].extend(
                    qk_effect.detach().cpu().tolist()
                )
                arrays["v"][direction].extend(v_effect.detach().cpu().tolist())
                arrays["qkv"][direction].extend(
                    qkv_effect.detach().cpu().tolist()
                )
                max_qkv_error = max(
                    max_qkv_error,
                    float(
                        (qkv_effect - whole_effect)
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    ),
                )

            with patch_heads(
                adapter,
                corrupt_source,
                [(FINAL_LAYER, head)],
                token_mask=donor_mask,
            ), clamp_write(
                adapter,
                [(FINAL_LAYER, "mlp")],
                {(FINAL_LAYER, "mlp"): clean_run.mlp_write},
                positions=donor_positions_2d,
            ):
                mlp_clamped_hidden = replay_final_layer(
                    adapter, clean_inputs, layer=FINAL_LAYER
                )
            mlp_clamped_margin = donor_margin(
                model, mlp_clamped_hidden, donor_tokens
            )
            mlp_clamped_damage = clean_margin - mlp_clamped_margin
            arrays["mlp_clamped_damage"].extend(
                mlp_clamped_damage.detach().cpu().tolist()
            )
            arrays["mlp_interaction"].extend(
                (whole_damage - mlp_clamped_damage).detach().cpu().tolist()
            )

            acceptor_centers = [
                int(example.next_acceptor_pos) for example in chunk
            ]
            donor_centers = [example.donor_pos - 1 for example in chunk]
            safe_decoy_centers = [
                int(decoy) if decoy is not None else acceptor
                for decoy, acceptor in zip(
                    chunk_decoy_acceptors, acceptor_centers
                )
            ]
            windows = {
                "acceptor": window_tensor(
                    acceptor_centers,
                    (-3, -2, -1, 0),
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                ),
                "donor": window_tensor(
                    donor_centers,
                    (-2, -1, 0, 1),
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                ),
                "nonpartner_ag": window_tensor(
                    safe_decoy_centers,
                    (-3, -2, -1, 0),
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                ),
                "upstream": window_tensor(
                    acceptor_centers,
                    (-7, -6, -5, -4),
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                ),
                "downstream": window_tensor(
                    acceptor_centers,
                    (1, 2, 3, 4),
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                ),
            }
            for region, positions in windows.items():
                restored_hidden = projection_intervention(
                    adapter,
                    corrupt_inputs,
                    clean_run,
                    head=head,
                    query_positions=donor_positions_2d,
                    source_positions=positions,
                    patch_query=False,
                    patch_key=True,
                    patch_value=True,
                )
                restored_margin = donor_margin(
                    model, restored_hidden, donor_tokens
                )
                recovery = restored_margin - corrupt_margin
                if region == "nonpartner_ag":
                    missing = torch.as_tensor(
                        [
                            decoy is None
                            for decoy in chunk_decoy_acceptors
                        ],
                        device=recovery.device,
                    )
                    recovery = recovery.masked_fill(missing, torch.nan)
                arrays["source"][region].extend(
                    recovery.detach().cpu().tolist()
                )
                clean_mass = attention_mass(
                    clean_run.attention,
                    head=head,
                    query_positions=donor_tokens,
                    source_positions=positions,
                )
                corrupt_mass = attention_mass(
                    corrupt_run.attention,
                    head=head,
                    query_positions=donor_tokens,
                    source_positions=positions,
                )
                if region == "nonpartner_ag":
                    clean_mass = clean_mass.masked_fill(missing, torch.nan)
                    corrupt_mass = corrupt_mass.masked_fill(missing, torch.nan)
                arrays["attention_clean"][region].extend(
                    clean_mass.detach().cpu().tolist()
                )
                arrays["attention_corrupt"][region].extend(
                    corrupt_mass.detach().cpu().tolist()
                )

            full_hidden = projection_intervention(
                adapter,
                corrupt_inputs,
                clean_run,
                head=head,
                query_positions=donor_positions_2d,
                source_positions=all_positions,
                patch_query=False,
                patch_key=True,
                patch_value=True,
            )
            full_margin = donor_margin(model, full_hidden, donor_tokens)
            arrays["source_full"].extend(
                (full_margin - corrupt_margin).detach().cpu().tolist()
            )
            noop_hidden = projection_intervention(
                adapter,
                corrupt_inputs,
                clean_run,
                head=head,
                query_positions=donor_positions_2d,
                source_positions=empty_positions,
                patch_query=False,
                patch_key=True,
                patch_value=True,
            )
            noop_margin = donor_margin(model, noop_hidden, donor_tokens)
            arrays["source_noop"].extend(
                (noop_margin - corrupt_margin).detach().cpu().tolist()
            )

            with patch_heads(
                adapter,
                {FINAL_LAYER: nonpartner_run.heads},
                [(FINAL_LAYER, head)],
                token_mask=donor_mask,
            ):
                nonpartner_patched_hidden = replay_final_layer(
                    adapter, clean_inputs, layer=FINAL_LAYER
                )
            nonpartner_patched = donor_margin(
                model, nonpartner_patched_hidden, donor_tokens
            )
            nonpartner_damage = clean_margin - nonpartner_patched
            missing_acceptor = torch.as_tensor(
                [decoy is None for decoy in chunk_decoy_acceptors],
                device=nonpartner_damage.device,
            )
            nonpartner_damage = nonpartner_damage.masked_fill(
                missing_acceptor, torch.nan
            )
            arrays["nonpartner_head_damage"].extend(
                nonpartner_damage.detach().cpu().tolist()
            )

            safe_other_donor_labels = [
                int(decoy) - 1 if decoy is not None else example.donor_pos - 1
                for example, decoy in zip(chunk, chunk_decoy_donors)
            ]
            other_tokens = donor_token_positions(
                safe_other_donor_labels,
                prefix_tokens=task.prefix_tokens,
                device=device,
            )
            other_mask = position_mask(other_tokens, sequence_length)
            clean_other = donor_margin(model, clean_run.hidden, other_tokens)
            with patch_heads(
                adapter,
                corrupt_source,
                [(FINAL_LAYER, head)],
                token_mask=other_mask,
            ):
                other_patched_hidden = replay_final_layer(
                    adapter, clean_inputs, layer=FINAL_LAYER
                )
            other_patched = donor_margin(
                model, other_patched_hidden, other_tokens
            )
            other_damage = clean_other - other_patched
            missing_donor = torch.as_tensor(
                [decoy is None for decoy in chunk_decoy_donors],
                device=other_damage.device,
            )
            other_damage = other_damage.masked_fill(missing_donor, torch.nan)
            arrays["other_donor_head_damage"].extend(
                other_damage.detach().cpu().tolist()
            )

            rows = torch.arange(len(chunk), device=donor_tokens.device)
            clean_delta = clean_run.heads[rows, head, donor_tokens]
            corrupt_delta = corrupt_run.heads[rows, head, donor_tokens]
            arrays["delta_context"].extend(
                (clean_delta - corrupt_delta).detach().cpu().tolist()
            )
            print(
                f"  {min(start + len(chunk), len(examples))}/{len(examples)}",
                end="\r",
                flush=True,
            )
        print()

        for key in ("clean", "corrupt", "mlp_clamped_damage", "mlp_interaction",
                    "source_full", "source_noop", "nonpartner_head_damage",
                    "other_donor_head_damage", "delta_context"):
            arrays[key] = np.asarray(arrays[key], dtype=np.float64)
        for family in ("whole", "qk", "v", "qkv"):
            for direction in ("damage", "recovery"):
                arrays[family][direction] = np.asarray(
                    arrays[family][direction], dtype=np.float64
                )
        for family in ("source", "attention_clean", "attention_corrupt"):
            for region in REGIONS:
                arrays[family][region] = np.asarray(
                    arrays[family][region], dtype=np.float64
                )
        seed_arrays[str(seed)] = arrays
        qkv_replay[str(seed)] = max_qkv_error

        expected = confirmation["raw"][str(seed)]
        if [row["example_id"] for row in expected] != example_ids:
            raise RuntimeError("confirmation cohort mismatch")
        expected_damage = np.asarray(
            [row["heads"][head_name]["damage"] for row in expected],
            dtype=np.float64,
        )
        expected_recovery = np.asarray(
            [row["heads"][head_name]["recovery"] for row in expected],
            dtype=np.float64,
        )
        confirmation_replay[str(seed)] = float(
            max(
                np.max(np.abs(arrays["whole"]["damage"] - expected_damage)),
                np.max(np.abs(arrays["whole"]["recovery"] - expected_recovery)),
            )
        )
        del task, adapter, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    names = [str(seed) for seed in sorted(EXPECTED_SEEDS)]
    family: dict[str, Any] = {
        "role": {},
        "mlp_interaction": {},
        "source": {},
        "source_pairwise": {},
        "specificity": {},
        "attention_descriptive": {},
        "geometry_descriptive": {},
    }
    for component_index, component in enumerate(("whole", "qk", "v", "qkv")):
        family["role"][component] = {}
        for direction_index, direction in enumerate(("damage", "recovery")):
            values = np.mean(
                np.stack(
                    [seed_arrays[name][component][direction] for name in names]
                ),
                axis=0,
            )
            family["role"][component][direction] = infer(
                values,
                groups,
                n_boot=args.n_boot,
                n_perm=None,
                seed=110_000 + 10 * component_index + direction_index,
            )
    role_contrasts = []
    for direction_index, direction in enumerate(("damage", "recovery")):
        values = np.mean(
            np.stack(
                [
                    seed_arrays[name]["qk"][direction]
                    - seed_arrays[name]["v"][direction]
                    for name in names
                ]
            ),
            axis=0,
        )
        result = infer(
            values,
            groups,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=120_000 + direction_index,
        )
        family["role"].setdefault("qk_minus_v", {})[direction] = result
        role_contrasts.append(result)
    add_qvalues(role_contrasts, "q_bh_2_role_contrasts")
    if all(
        family["role"]["qk_minus_v"][direction]["estimate"]
        >= MIN_ROLE_DIFFERENCE
        and family["role"]["qk_minus_v"][direction]["ci_low"] > 0
        and family["role"]["qk_minus_v"][direction][
            "q_bh_2_role_contrasts"
        ]
        <= MAX_Q
        for direction in ("damage", "recovery")
    ):
        role_class = "routing-dominant"
    elif all(
        family["role"]["qk_minus_v"][direction]["estimate"]
        <= -MIN_ROLE_DIFFERENCE
        and family["role"]["qk_minus_v"][direction]["ci_high"] < 0
        and family["role"]["qk_minus_v"][direction][
            "q_bh_2_role_contrasts"
        ]
        <= MAX_Q
        for direction in ("damage", "recovery")
    ):
        role_class = "content-dominant"
    else:
        role_class = "mixed"
    family["role"]["classification"] = role_class

    for index, metric in enumerate(("mlp_clamped_damage", "mlp_interaction")):
        values = np.mean(
            np.stack([seed_arrays[name][metric] for name in names]),
            axis=0,
        )
        family["mlp_interaction"][metric] = infer(
            values,
            groups,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=130_000 + index,
        )
    add_qvalues(
        list(family["mlp_interaction"].values()),
        "q_bh_2_mlp_interaction_tests",
    )

    source_results = []
    source_values: dict[str, np.ndarray] = {}
    for region_index, region in enumerate(REGIONS):
        source_values[region] = nanmean_stack(
            [seed_arrays[name]["source"][region] for name in names]
        )
        result = infer(
            source_values[region],
            groups,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=140_000 + region_index,
        )
        family["source"][region] = result
        source_results.append(result)
    add_qvalues(source_results, "q_bh_5_source_regions")
    for index, metric in enumerate(("source_full", "source_noop")):
        values = np.mean(
            np.stack([seed_arrays[name][metric] for name in names]),
            axis=0,
        )
        family["source"][metric] = infer(
            values,
            groups,
            n_boot=args.n_boot,
            n_perm=None,
            seed=145_000 + index,
        )

    pair_results: list[dict[str, Any]] = []
    for pair_index, (left, right) in enumerate(itertools.combinations(REGIONS, 2)):
        contrast = source_values[left] - source_values[right]
        result = infer(
            contrast,
            groups,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=150_000 + pair_index,
        )
        family["source_pairwise"][f"{left}_minus_{right}"] = result
        pair_results.append(result)
    add_qvalues(pair_results, "q_bh_10_source_pairwise")

    winning_region = None
    for region in REGIONS:
        source_result = family["source"][region]
        wins = (
            source_result["estimate"] >= MIN_SOURCE_EFFECT
            and source_result["ci_low"] > 0
            and source_result["q_bh_5_source_regions"] <= MAX_Q
        )
        for other in REGIONS:
            if other == region:
                continue
            if REGIONS.index(region) < REGIONS.index(other):
                contrast = family["source_pairwise"][f"{region}_minus_{other}"]
                estimate = contrast["estimate"]
                ci_low = contrast["ci_low"]
            else:
                contrast = family["source_pairwise"][f"{other}_minus_{region}"]
                estimate = -contrast["estimate"]
                ci_low = -contrast["ci_high"]
            wins = (
                wins
                and estimate > 0
                and ci_low > 0
                and contrast["q_bh_10_source_pairwise"] <= MAX_Q
            )
        if wins:
            if winning_region is not None:
                raise RuntimeError("more than one source region passed unique-winner rule")
            winning_region = region

    paired_damage = np.mean(
        np.stack([seed_arrays[name]["whole"]["damage"] for name in names]),
        axis=0,
    )
    nonpartner_damage = nanmean_stack(
        [seed_arrays[name]["nonpartner_head_damage"] for name in names]
    )
    other_donor_damage = nanmean_stack(
        [seed_arrays[name]["other_donor_head_damage"] for name in names]
    )
    specificity_values = {
        "nonpartner_ag_head_damage": nonpartner_damage,
        "other_donor_head_damage": other_donor_damage,
        "paired_minus_nonpartner_ag": paired_damage - nonpartner_damage,
        "paired_minus_other_donor": paired_damage - other_donor_damage,
    }
    specificity_results = []
    for index, (metric, values) in enumerate(specificity_values.items()):
        result = infer(
            values,
            groups,
            n_boot=args.n_boot,
            n_perm=(
                args.n_perm if metric.startswith("paired_minus") else None
            ),
            seed=160_000 + index,
        )
        family["specificity"][metric] = result
        if metric.startswith("paired_minus"):
            specificity_results.append(result)
    add_qvalues(specificity_results, "q_bh_2_specificity_contrasts")

    for region_index, region in enumerate(REGIONS):
        for state_index, state in enumerate(("clean", "corrupt")):
            values = nanmean_stack(
                [
                    seed_arrays[name][f"attention_{state}"][region]
                    for name in names
                ]
            )
            family["attention_descriptive"].setdefault(region, {})[state] = infer(
                values,
                groups,
                n_boot=args.n_boot,
                n_perm=None,
                seed=170_000 + 10 * region_index + state_index,
            )

    for left, right in itertools.combinations(names, 2):
        x = seed_arrays[left]["delta_context"]
        y = seed_arrays[right]["delta_context"]
        family["geometry_descriptive"][f"{left}_vs_{right}"] = {
            "linear_cka": linear_cka(x, y),
            "orthogonal_procrustes_cosine": orthogonal_procrustes_cosine(x, y),
        }

    qkv_pass = all(error <= REPLAY_ATOL for error in qkv_replay.values())
    confirmation_pass = all(
        error <= REPLAY_ATOL for error in confirmation_replay.values()
    )
    noop_pass = abs(family["source"]["source_noop"]["estimate"]) <= REPLAY_ATOL
    source_gate = {
        "winning_region": winning_region,
        "PASS": winning_region is not None,
        "kv_region_followup_authorized": (
            winning_region is not None
            and qkv_pass
            and confirmation_pass
            and noop_pass
        ),
    }
    result = {
        "study": "native GTEx final-layer role and biological source",
        "protocol": str(protocol.relative_to(REPO)),
        "status": (
            "PASS"
            if qkv_pass and confirmation_pass and noop_pass
            else "FAIL_INSTRUMENT"
        ),
        "device": device,
        "split": "val",
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "head": head_name,
        "qkv_replay_max_abs_error": qkv_replay,
        "confirmation_replay_max_abs_error": confirmation_replay,
        "family": family,
        "verdicts": {
            "role_classification": role_class,
            "source": source_gate,
            "paired_acceptor_specific": bool(
                family["specificity"]["paired_minus_nonpartner_ag"]["ci_low"] > 0
                and family["specificity"]["paired_minus_other_donor"]["ci_low"] > 0
                and all(
                    result["q_bh_2_specificity_contrasts"] <= MAX_Q
                    for result in specificity_results
                )
            ),
            "qkv_replay": qkv_pass,
            "confirmation_replay": confirmation_pass,
            "source_noop": noop_pass,
        },
        "thresholds": {
            "replay_atol": REPLAY_ATOL,
            "min_role_difference": MIN_ROLE_DIFFERENCE,
            "min_source_effect": MIN_SOURCE_EFFECT,
            "max_q": MAX_Q,
        },
        "raw": {
            name: [
                {
                    "example_id": examples[index].example_id,
                    "gene_id": examples[index].gene_id,
                    "decoy_acceptor": decoy_acceptors[index],
                    "decoy_donor": decoy_donors[index],
                    "whole": {
                        direction: seed_arrays[name]["whole"][direction][index]
                        for direction in ("damage", "recovery")
                    },
                    "qk": {
                        direction: seed_arrays[name]["qk"][direction][index]
                        for direction in ("damage", "recovery")
                    },
                    "v": {
                        direction: seed_arrays[name]["v"][direction][index]
                        for direction in ("damage", "recovery")
                    },
                    "qkv": {
                        direction: seed_arrays[name]["qkv"][direction][index]
                        for direction in ("damage", "recovery")
                    },
                    "mlp_clamped_damage": seed_arrays[name][
                        "mlp_clamped_damage"
                    ][index],
                    "mlp_interaction": seed_arrays[name]["mlp_interaction"][index],
                    "source": {
                        region: seed_arrays[name]["source"][region][index]
                        for region in REGIONS
                    },
                    "source_full": seed_arrays[name]["source_full"][index],
                    "source_noop": seed_arrays[name]["source_noop"][index],
                    "nonpartner_head_damage": seed_arrays[name][
                        "nonpartner_head_damage"
                    ][index],
                    "other_donor_head_damage": seed_arrays[name][
                        "other_donor_head_damage"
                    ][index],
                    "delta_context": seed_arrays[name]["delta_context"][index],
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
        "splicebert_gtex_finetune_shift:native_mechanism_role_source_v1",
        sys.argv,
        repo=REPO,
        status="CONFIRMATORY" if result["status"] == "PASS" else "FAILED",
        config={
            "split": "val",
            "n": args.n,
            "batch_size": args.batch,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "head": head_name,
            "regions": list(REGIONS),
            "thresholds": result["thresholds"],
        },
        seeds={
            "checkpoint_min": min(EXPECTED_SEEDS),
            "checkpoint_max": max(EXPECTED_SEEDS),
        },
        inputs=[
            file_record(protocol),
            file_record(confirmation_path),
            file_record(base_model / "config.json"),
            file_record(model_weight(base_model)),
            *[file_record(path) for path in checkpoints.values()],
            file_record(args.gtf),
            file_record(args.fasta),
        ],
        notes=[
            "QKV must replay whole-head intervention in both directions.",
            "Five equal-size source regions and all ten pairwise contrasts are corrected separately.",
            "Attention mass and cross-seed geometry are descriptive only.",
            "K-only/V-only source analysis remains gated on a unique frozen winning region.",
        ],
    )
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "head": head_name,
                "role": family["role"],
                "mlp_interaction": family["mlp_interaction"],
                "source": family["source"],
                "source_gate": source_gate,
                "specificity": family["specificity"],
                "geometry": family["geometry_descriptive"],
                "replay": {
                    "qkv": qkv_replay,
                    "confirmation": confirmation_replay,
                },
            },
            indent=2,
        )
    )
    print(f"saved -> {output}")
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
