#!/usr/bin/env python
"""Development-only preflight for BG1 behavior and BG2 payload recovery.

This command is the initial authorization artifact for both target runners; BG2 additionally
requires an independently audited BG1 result.  It is restricted to the train chromosome split,
exercises the real panel/statistics/injection paths, and writes immutable, hash-bound checks.
Construction identities are recorded as forced
plumbing checks; biological interpretation depends on the independently capable positive,
specificity, resolution, false-positive, and route-sensitivity gates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from run_bg1_boundary_grammar import (  # noqa: E402
    DECLARATION,
    PREFLIGHT_SCHEMA,
    PROTOCOL,
    checkpoint_weights_path,
    gene_contrast_matrix,
    membership_declaration,
    model_file_hashes,
    panel_eligible,
    score_panel,
    select_panel_cohort,
    select_paired_panel_cohort,
    signflip_null,
    simultaneous_summary,
)
from run_bg2_boundary_payload_recovery import (  # noqa: E402
    DIRECTED_PAIRS,
    HEAD,
    LAYER,
    build_donor_patch,
    edited_attention_delta,
)
from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.circuits.hf_bert import capture_head_outputs, patch_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.boundary_grammar import (  # noqa: E402
    ANALYSIS_SEED,
    CHECKPOINT_SHA256,
    FASTA_SHA256,
    GTF_SHA256,
    MINUS3_PANEL,
    MINUS3_T_MINUS_A_SINGLE,
    MINUS4_PANEL,
    MINUS4_T_MINUS_A,
    MIN_BG1_TARGET_GENES,
    MIN_BG1_TARGET_GENES_PER_STRAND,
    MIN_BG2_TARGET_GENES,
    MIN_BG2_TARGET_GENES_PER_STRAND,
    MIN_OUTER_TARGET_GENES,
    MIN_OUTER_TARGET_GENES_PER_STRAND,
    MIN_TRAIN_BEHAVIOR_GENES,
    MIN_TRAIN_ROUTE_GENES,
    PLUS1_G_MINUS_A_SINGLE,
    PLUS1_PANEL,
    PLUS2_G_MINUS_A,
    PLUS2_PANEL,
    PREFLIGHT_BOOTSTRAP_DRAWS,
    PREFLIGHT_CALIBRATION_REPLICATES,
    PREFLIGHT_HEALTH_GENES,
    PREFLIGHT_PERMUTATION_DRAWS,
    SESOI,
    TARGET_CONTRASTS,
    TARGET_PANEL,
    build_panel,
    cohort_declaration,
    margins_from_logits,
    max_family_add_one_pvalues,
    load_boundary_examples,
    preflight_config,
    select_gene_cohort,
    validate_panel,
)
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    eligible,
    load_adapter,
    matched_decoy_acceptor,
    pick_device,
)
from gfmcd.splicing.coupling import destroy_acceptor_clean, substitute  # noqa: E402
from gfmcd.splicing.payload import value_weight_head  # noqa: E402
from scipy.stats import norm  # noqa: E402


RESULT_DEFAULT = "results/splicebert_coupling/bg1_boundary_preflight_v1.json"
LINEARITY_TOL = 1e-4
INJECTION_TOL = 1e-4
BATCH_TOL = 1e-5
MAX_CALIBRATED_FPR = 0.075
MIN_PLANTED_POWER = 0.80
MIN_ROUTE_LIFT = 0.05


def resolved(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def mean_ci(values: Sequence[float], *, draws: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 3 or not np.isfinite(array).all():
        raise ValueError("mean CI needs at least three finite values")
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def score_sequences(task: CouplingTask, sequences: Sequence[str], donors: Sequence[int]):
    linear, maximum = [], []
    with torch.no_grad():
        for start in range(0, len(sequences), task.batch):
            seq = sequences[start:start + task.batch]
            pos = donors[start:start + task.batch]
            encoded = task.encode(seq, pos)
            logits = task.ad.model(**encoded).logits.float()
            lin, mx = margins_from_logits(
                logits, pos, base_token_ids=task.bid, token_offset=task.off
            )
            linear.extend(lin.tolist())
            maximum.extend(mx.tolist())
    return np.asarray(linear), np.asarray(maximum)


def analysis_calibration(
    matrix: np.ndarray,
    *,
    critical_value: float,
    seed: int,
    replicates: int,
    n_permutation: int,
) -> dict:
    """Power and worst-case composite-null audit of the frozen joint decision."""

    centered = matrix[:, :2] - matrix[:, :2].mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    planted_hits = 0
    endpoint_family_hits = 0
    composite_hits = np.zeros(2, dtype=int)

    def endpoint_passes(world: np.ndarray, world_seed: int) -> np.ndarray:
        standard_errors = world.std(axis=0, ddof=1) / math.sqrt(len(world))
        pvalues = max_family_add_one_pvalues(
            world.mean(axis=0),
            signflip_null(world, draws=n_permutation, seed=world_seed),
            two_sided=False,
        )
        return (
            world.mean(axis=0) - critical_value * standard_errors > SESOI
        ) & (pvalues <= 0.05)

    for index in range(replicates):
        sample = centered[rng.integers(0, len(centered), size=len(centered))]
        planted = sample + 2 * SESOI
        planted_hits += int(np.all(endpoint_passes(planted, seed + 10_000 + index)))

        global_boundary = sample + SESOI
        endpoint_family_hits += int(np.any(endpoint_passes(
            global_boundary, seed + 20_000 + index
        )))

        # For the intersection claim, the union null's least-favourable configurations
        # place one endpoint at the SESOI boundary while the other is materially positive.
        for boundary_endpoint in range(2):
            shifts = np.full(2, 2 * SESOI)
            shifts[boundary_endpoint] = SESOI
            composite = sample + shifts
            composite_hits[boundary_endpoint] += int(np.all(endpoint_passes(
                composite,
                seed + 30_000 + boundary_endpoint * replicates + index,
            )))
    composite_rates = (composite_hits / replicates).tolist()
    return {
        "replicates": replicates,
        "permutation_draws_per_world": n_permutation,
        "claim_threshold": SESOI,
        "planted_total_effect": 2 * SESOI,
        "planted_excess_above_threshold": SESOI,
        "planted_power": planted_hits / replicates,
        "endpoint_familywise_false_positive_rate": endpoint_family_hits / replicates,
        "joint_claim_composite_null_false_positive_rate_by_boundary_endpoint": composite_rates,
        "joint_claim_worst_case_false_positive_rate": max(composite_rates),
    }


def projected_mde80(
    matrix: np.ndarray, *, critical_value: float, target_n: int
) -> list[float]:
    """Project train gene-level dispersion onto the frozen target gene count."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or len(values) < 3 or target_n < 3:
        raise ValueError("projected MDE needs [train genes,endpoints] and target_n >= 3")
    scale = (critical_value + norm.ppf(0.80)) / math.sqrt(target_n)
    return [float(scale * values[:, index].std(ddof=1)) for index in range(values.shape[1])]


def route_instrument_preflight(adapter, task: CouplingTask, examples: Sequence) -> dict:
    """Exercise edited-address value recovery on both frozen directed contrasts."""

    Wv = value_weight_head(adapter).to(adapter.device)
    head_width = adapter.hidden_size // adapter.num_heads
    max_linearity_error = 0.0
    max_reconstruction_error = 0.0
    max_identity_linear_error = 0.0
    max_identity_max_error = 0.0
    max_attention_row_error = 0.0
    pair_cells = {
        str(pair["pair_id"]): (
            str(pair["transition"]),
            str(pair["reference_cell"]),
            str(pair["edit_cell"]),
        )
        for pair in DIRECTED_PAIRS
    }
    dose_by_pair = {
        "acceptor_AG_to_AC_positive": {
            alpha: [] for alpha in (0.0, 0.25, 0.5, 1.0)
        },
        **{
            pair_id: {alpha: [] for alpha in (0.0, 0.25, 0.5, 1.0)}
            for pair_id in pair_cells
        },
    }

    experiments = (("acceptor_AG_to_AC_positive", None), *pair_cells.items())
    for pair_id, pair in experiments:
        for start in range(0, len(examples), task.batch):
            chunk = examples[start:start + task.batch]
            if pair is None:
                reference_sequences = [example.sequence for example in chunk]
                edit_sequences = [
                    destroy_acceptor_clean(
                        example.sequence, int(example.next_acceptor_pos)
                    )
                    for example in chunk
                ]
            else:
                _transition, reference_id, edit_id = pair
                panels = [build_panel(
                    example.sequence,
                    int(example.next_acceptor_pos),
                    TARGET_PANEL,
                    strand=example.strand,
                ) for example in chunk]
                reference_sequences = [panel[reference_id] for panel in panels]
                edit_sequences = [panel[edit_id] for panel in panels]
            donors = [int(example.donor_pos) for example in chunk]
            reference = task.encode(reference_sequences, donors)
            edit = task.encode(edit_sequences, donors)
            donor_tokens = torch.as_tensor(
                [donor + task.off for donor in donors], device=adapter.device
            )
            batch_index = torch.arange(len(chunk), device=adapter.device)
            acc4 = torch.as_tensor([
                [int(example.next_acceptor_pos) + task.off + offset
                 for offset in (-3, -2, -1, 0)]
                for example in chunk
            ], device=adapter.device)
            gather_batch = batch_index[:, None].expand_as(acc4)

            with torch.no_grad():
                reference_output = adapter.model(**reference, output_hidden_states=True)
                box = {}
                edit_heads = capture_head_outputs(
                    adapter,
                    lambda: box.__setitem__(
                        "output",
                        adapter.model(
                            **edit, output_hidden_states=True, output_attentions=True
                        ),
                    ),
                )
                edit_output = box["output"]
                residual_reference = reference_output.hidden_states[LAYER]
                residual_edit = edit_output.hidden_states[LAYER]
                value_module = adapter.attention_self_modules[LAYER].value
                values_reference = value_module(residual_reference)
                values_edit = value_module(residual_edit)
                lo, hi = HEAD * head_width, (HEAD + 1) * head_width
                values_reference = values_reference[:, :, lo:hi]
                values_edit = values_edit[:, :, lo:hi]
                attention = edit_output.attentions[LAYER][
                    batch_index, HEAD, donor_tokens, :
                ]
                natural_edit = edit_heads[LAYER][
                    batch_index, HEAD, donor_tokens
                ]
                reconstructed = torch.einsum("bs,bsd->bd", attention, values_edit)
                residual_delta = (
                    residual_reference - residual_edit
                )[gather_batch, acc4]
                true_value_delta = (
                    values_reference - values_edit
                )[gather_batch, acc4]
                linear_value_delta = residual_delta @ Wv.T
                attention_acc4 = attention[gather_batch, acc4]
                delivered = edited_attention_delta(
                    attention_acc4, Wv, residual_delta
                )

                edit_linear, edit_max = margins_from_logits(
                    edit_output.logits.float(),
                    donors,
                    base_token_ids=task.bid,
                    token_offset=task.off,
                )
                max_linearity_error = max(
                    max_linearity_error,
                    float((true_value_delta - linear_value_delta).abs().max()),
                )
                max_reconstruction_error = max(
                    max_reconstruction_error,
                    float((reconstructed - natural_edit).abs().max()),
                )
                max_attention_row_error = max(
                    max_attention_row_error,
                    float((attention.sum(dim=-1) - 1.0).abs().max()),
                )

                for alpha in (0.0, 0.25, 0.5, 1.0):
                    recovered = natural_edit + alpha * delivered
                    source, mask = build_donor_patch(
                        recovered,
                        donor_tokens=donor_tokens,
                        num_heads=adapter.num_heads,
                        sequence_length=edit["input_ids"].shape[1],
                    )
                    with patch_heads(
                        adapter, {LAYER: source}, [(LAYER, HEAD)], token_mask=mask
                    ):
                        patched_logits = adapter.model(**edit).logits.float()
                    patched_linear, patched_max = margins_from_logits(
                        patched_logits,
                        donors,
                        base_token_ids=task.bid,
                        token_offset=task.off,
                    )
                    dose_by_pair[pair_id][alpha].extend(
                        (patched_linear - edit_linear).tolist()
                    )
                    if alpha == 0.0:
                        max_identity_linear_error = max(
                            max_identity_linear_error,
                            float(np.max(np.abs(patched_linear - edit_linear))),
                        )
                        max_identity_max_error = max(
                            max_identity_max_error,
                            float(np.max(np.abs(patched_max - edit_max))),
                        )

    dose_summary = {}
    dose_pass = True
    for pair_id, values in dose_by_pair.items():
        means = [float(np.mean(values[alpha])) for alpha in (0.0, 0.25, 0.5, 1.0)]
        monotone = all(right + 1e-5 >= left for left, right in zip(means, means[1:]))
        live = means[-1] >= MIN_ROUTE_LIFT
        dose_summary[pair_id] = {
            "alpha": [0.0, 0.25, 0.5, 1.0],
            "mean_linear_lift": means,
            "monotone": monotone,
            "full_dose_live": live,
        }
        if pair_id == "acceptor_AG_to_AC_positive":
            dose_pass = dose_pass and monotone and live

    boundary_values = {}
    for transition in ("minus3_T_to_A", "plus1_G_to_A"):
        pair_ids = [
            pair_id for pair_id, (name, _reference, _edit) in pair_cells.items()
            if name == transition
        ]
        arrays = [np.asarray(dose_by_pair[pair_id][1.0], dtype=float) for pair_id in pair_ids]
        if not arrays or any(len(values) != len(examples) for values in arrays):
            raise RuntimeError(f"route preflight did not retain complete {transition} backgrounds")
        boundary_values[transition] = np.mean(np.column_stack(arrays), axis=1).tolist()
    return {
        "value_linearity": max_linearity_error,
        "head_reconstruction": max_reconstruction_error,
        "injection_identity_linear": max_identity_linear_error,
        "injection_identity_max": max_identity_max_error,
        "attention_row_sum": max_attention_row_error,
        "dose": dose_summary,
        "dose_PASS": dose_pass,
        "boundary_full_dose_values": boundary_values,
    }


def outer_power(
    task: CouplingTask,
    source,
    *,
    inner_panel,
    inner_spec,
    outer_panel,
    outer_spec,
    n: int | None,
    seed: int,
    n_bootstrap: int,
    n_permutation: int,
    target_examples: Sequence,
) -> dict:
    examples = select_paired_panel_cohort(
        source, inner_panel, outer_panel, role="behavior", n=n
    )
    if len(examples) < 3:
        return {"claim_bearing": False, "n_genes": len(examples), "reason": "support"}
    inner_rows = score_panel(task, examples, inner_panel)
    outer_rows = score_panel(task, examples, outer_panel)
    genes_i, inner, _inner_max = gene_contrast_matrix(inner_rows, inner_panel, (inner_spec,))
    genes_o, outer, _outer_max = gene_contrast_matrix(outer_rows, outer_panel, (outer_spec,))
    if genes_i != genes_o:
        raise RuntimeError("outer preflight pairing drifted")
    attenuation = inner[:, 0] - outer[:, 0]
    inference = simultaneous_summary(
        attenuation[:, None],
        names=(f"{inner_spec.contrast_id}_minus_{outer_spec.contrast_id}",),
        seed=seed,
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation,
        p_two_sided=False,
    )
    achieved_mde = projected_mde80(
        attenuation[:, None],
        critical_value=float(inference["critical_value"]),
        target_n=len(target_examples),
    )[0] if len(target_examples) >= 3 else None
    target_strands = {
        strand: sum(example.strand == strand for example in target_examples)
        for strand in ("+", "-")
    }
    return {
        "claim_bearing": bool(
            len(target_examples) >= MIN_OUTER_TARGET_GENES
            and min(target_strands.values()) >= MIN_OUTER_TARGET_GENES_PER_STRAND
            and achieved_mde is not None
            and achieved_mde <= SESOI
        ),
        "train_n_genes": len(examples),
        "target_n_genes": len(target_examples),
        "target_strands": target_strands,
        "projected_target_mde80": achieved_mde,
        "required_n_genes": MIN_OUTER_TARGET_GENES,
        "required_per_strand": MIN_OUTER_TARGET_GENES_PER_STRAND,
        "required_mde_max": SESOI,
        "train_membership": membership_declaration(examples),
        "target_membership": membership_declaration(target_examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--n-behavior", type=int, default=MIN_TRAIN_BEHAVIOR_GENES)
    parser.add_argument("--n-route", type=int, default=MIN_TRAIN_ROUTE_GENES)
    parser.add_argument("--n-outer", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=PREFLIGHT_BOOTSTRAP_DRAWS)
    parser.add_argument("--n-permutation", type=int, default=PREFLIGHT_PERMUTATION_DRAWS)
    parser.add_argument(
        "--calibration-replicates", type=int, default=PREFLIGHT_CALIBRATION_REPLICATES
    )
    parser.add_argument("--health-n", type=int, default=PREFLIGHT_HEALTH_GENES)
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--enable-test-replication", action="store_true")
    parser.add_argument("--out", default=RESULT_DEFAULT)
    args = parser.parse_args()
    if min(args.n_behavior, args.n_route, args.batch, args.health_n) <= 0:
        parser.error("cohort and batch sizes must be positive")
    if min(args.n_bootstrap, args.n_permutation) < 99:
        parser.error("resampling counts must be >=99")
    if args.calibration_replicates < 40:
        parser.error("calibration needs at least 40 replicates")
    if not args.smoke and (
        args.n_behavior != MIN_TRAIN_BEHAVIOR_GENES
        or args.n_route != MIN_TRAIN_ROUTE_GENES
        or args.n_outer is not None
        or args.health_n != PREFLIGHT_HEALTH_GENES
        or args.seed != ANALYSIS_SEED
        or args.n_bootstrap != PREFLIGHT_BOOTSTRAP_DRAWS
        or args.n_permutation != PREFLIGHT_PERMUTATION_DRAWS
        or args.calibration_replicates != PREFLIGHT_CALIBRATION_REPLICATES
    ):
        parser.error("canonical preflight cohort, seed, and resampling settings are frozen")

    out = guard_output_path(resolved(args.out))
    manifest_path = guard_output_path(out.with_suffix(".manifest.json"))
    if not PROTOCOL.is_file():
        raise FileNotFoundError(f"frozen BG1 protocol is missing: {PROTOCOL}")
    frozen_model_hashes = model_file_hashes(args.model)
    if frozen_model_hashes.get("pytorch_model.bin") != CHECKPOINT_SHA256:
        raise RuntimeError("the requested model is not the frozen SpliceBERT checkpoint")
    frozen_gtf_hash = sha256_file(resolved(args.gtf))
    frozen_fasta_hash = sha256_file(resolved(args.fasta))
    if frozen_gtf_hash != GTF_SHA256:
        raise RuntimeError("annotation does not match the frozen GENCODE basic GTF")
    if frozen_fasta_hash != FASTA_SHA256:
        raise RuntimeError("FASTA does not match the frozen hg38 sequence")
    if args.smoke:
        args.n_behavior = min(args.n_behavior, 12)
        args.n_route = min(args.n_route, 4)
        args.health_n = min(args.health_n, 8)
        args.n_outer = min(args.n_outer or 12, 12)
        args.n_bootstrap = min(args.n_bootstrap, 199)
        args.n_permutation = min(args.n_permutation, 199)
        args.calibration_replicates = min(args.calibration_replicates, 40)

    torch.set_grad_enabled(False)
    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=True)
    task = CouplingTask(adapter, batch=args.batch)
    all_examples = load_boundary_examples(args.gtf, args.fasta)
    source = eligible(all_examples, "train")
    valid_target = [example for example in source if panel_eligible(example, TARGET_PANEL)]
    behavior = select_gene_cohort(valid_target, role="behavior", n=args.n_behavior)
    route = select_gene_cohort(valid_target, role="route", n=args.n_route)
    cohort = cohort_declaration({"behavior": behavior, "route": route})
    cohort.update({
        "split": "train",
        "counts": {"behavior": len(behavior), "route": len(route)},
    })

    target_cohorts = {}
    target_examples_by_split = {}
    for split in ("val", "test"):
        target_source = eligible(all_examples, split)
        target_behavior = select_panel_cohort(
            target_source, TARGET_PANEL, role="behavior", n=None
        )
        target_route = select_panel_cohort(
            target_source, TARGET_PANEL, role="route", n=None
        )
        target_minus = select_paired_panel_cohort(
            target_source, MINUS3_PANEL, MINUS4_PANEL, role="behavior", n=None
        )
        target_plus = select_paired_panel_cohort(
            target_source, PLUS1_PANEL, PLUS2_PANEL, role="behavior", n=None
        )
        target_examples_by_split[split] = {
            "target_behavior": target_behavior,
            "target_route": target_route,
            "minus3_minus4_diagnostic": target_minus,
            "plus1_plus2_diagnostic": target_plus,
        }
        target_cohorts[split] = {
            name: membership_declaration(examples)
            for name, examples in target_examples_by_split[split].items()
        }

    target_rows = score_panel(task, behavior, TARGET_PANEL)
    _genes, target_linear_all, _target_max = gene_contrast_matrix(
        target_rows, TARGET_PANEL, TARGET_CONTRASTS
    )
    target_linear = target_linear_all[:, :2]
    target_inference = simultaneous_summary(
        target_linear,
        names=tuple(spec.contrast_id for spec in TARGET_CONTRASTS[:2]),
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
        p_two_sided=False,
    )
    calibration = analysis_calibration(
        target_linear,
        critical_value=float(target_inference["critical_value"]),
        seed=args.seed + 100,
        replicates=args.calibration_replicates,
        n_permutation=args.n_permutation,
    )

    health = behavior[:min(args.health_n, len(behavior))]
    donors = [int(example.donor_pos) for example in health]
    native_sequences = [example.sequence for example in health]
    partner_sequences = [
        destroy_acceptor_clean(example.sequence, int(example.next_acceptor_pos))
        for example in health
    ]
    decoy_positions = [matched_decoy_acceptor(example) for example in health]
    has_decoy = np.asarray([position is not None for position in decoy_positions])
    decoy_sequences = [
        substitute(example.sequence, int(position) - 2, "AC")
        if position is not None else example.sequence
        for example, position in zip(health, decoy_positions)
    ]
    native_linear, native_max = score_sequences(task, native_sequences, donors)
    partner_linear, partner_max = score_sequences(task, partner_sequences, donors)
    decoy_linear, _decoy_max = score_sequences(task, decoy_sequences, donors)
    partner_effect = native_linear - partner_linear
    partner_max_effect = native_max - partner_max
    specificity = partner_effect[has_decoy] - (native_linear - decoy_linear)[has_decoy]
    partner_ci = mean_ci(partner_effect, draws=args.n_bootstrap, seed=args.seed + 200)
    partner_max_ci = mean_ci(
        partner_max_effect, draws=args.n_bootstrap, seed=args.seed + 202
    )
    specificity_ci = (
        mean_ci(specificity, draws=args.n_bootstrap, seed=args.seed + 201)
        if len(specificity) >= 3 else None
    )

    probe = health[:min(args.batch, len(health))]
    probe_sequences = [example.sequence for example in probe]
    probe_donors = [int(example.donor_pos) for example in probe]
    probe_linear, probe_max = score_sequences(task, probe_sequences, probe_donors)
    singleton_task = CouplingTask(adapter, batch=1)
    singleton_linear, singleton_max = score_sequences(
        singleton_task, probe_sequences, probe_donors
    )
    reversed_linear, reversed_max = score_sequences(
        task, list(reversed(probe_sequences)), list(reversed(probe_donors))
    )
    reversed_linear = reversed_linear[::-1]
    reversed_max = reversed_max[::-1]
    batch_error = float(max(
        np.max(np.abs(probe_linear - singleton_linear)),
        np.max(np.abs(probe_max - singleton_max)),
        np.max(np.abs(probe_linear - reversed_linear)),
        np.max(np.abs(probe_max - reversed_max)),
    ))

    first = behavior[0]
    first_cells = build_panel(
        first.sequence,
        int(first.next_acceptor_pos),
        TARGET_PANEL,
        strand=first.strand,
    )
    construction = validate_panel(first, first_cells, TARGET_PANEL)
    encoded = task.encode([next(iter(first_cells.values()))], [int(first.donor_pos)])
    donor_mask_exact = int(encoded["input_ids"][0, int(first.donor_pos) + task.off]) == task.mask_id
    route_metrics = route_instrument_preflight(adapter, task, route)
    route_matrix = np.column_stack([
        route_metrics["boundary_full_dose_values"][transition]
        for transition in ("minus3_T_to_A", "plus1_G_to_A")
    ])
    route_inference = simultaneous_summary(
        route_matrix,
        names=("minus3_T_to_A", "plus1_G_to_A"),
        seed=args.seed + 250,
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
        p_two_sided=False,
    )

    minus_outer = outer_power(
        task,
        source,
        inner_panel=MINUS3_PANEL,
        inner_spec=MINUS3_T_MINUS_A_SINGLE,
        outer_panel=MINUS4_PANEL,
        outer_spec=MINUS4_T_MINUS_A,
        n=args.n_outer,
        seed=args.seed + 300,
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
        target_examples=target_examples_by_split["val"]["minus3_minus4_diagnostic"],
    )
    plus_outer = outer_power(
        task,
        source,
        inner_panel=PLUS1_PANEL,
        inner_spec=PLUS1_G_MINUS_A_SINGLE,
        outer_panel=PLUS2_PANEL,
        outer_spec=PLUS2_G_MINUS_A,
        n=args.n_outer,
        seed=args.seed + 301,
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
        target_examples=target_examples_by_split["val"]["plus1_plus2_diagnostic"],
    )

    val_behavior = target_examples_by_split["val"]["target_behavior"]
    val_route = target_examples_by_split["val"]["target_route"]
    primary_mde = projected_mde80(
        target_linear,
        critical_value=float(target_inference["critical_value"]),
        target_n=len(val_behavior),
    )
    route_mde = projected_mde80(
        route_matrix,
        critical_value=float(route_inference["critical_value"]),
        target_n=len(val_route),
    )
    size_floor = 3 if args.smoke else MIN_TRAIN_BEHAVIOR_GENES
    route_floor = 2 if args.smoke else MIN_TRAIN_ROUTE_GENES
    val_behavior_strands = {
        strand: sum(example.strand == strand for example in val_behavior)
        for strand in ("+", "-")
    }
    val_route_strands = {
        strand: sum(example.strand == strand for example in val_route)
        for strand in ("+", "-")
    }
    test_behavior = target_examples_by_split["test"]["target_behavior"]
    test_behavior_strands = {
        strand: sum(example.strand == strand for example in test_behavior)
        for strand in ("+", "-")
    }
    checks = {
        "cohort_size_and_gene_disjointness": {
            "PASS": bool(
                len(behavior) >= size_floor
                and len(route) >= route_floor
                and {example.strand for example in route} == {"+", "-"}
            ),
            "behavior_genes": len(behavior),
            "route_genes": len(route),
            "thresholds": [size_floor, route_floor],
        },
        "both_strands": {
            "PASS": (
                {example.strand for example in behavior} == {"+", "-"}
                and {example.strand for example in route} == {"+", "-"}
            ),
            "behavior_strands": sorted({example.strand for example in behavior}),
            "route_strands": sorted({example.strand for example in route}),
        },
        "validation_supply": {
            "PASS": bool(
                len(val_behavior) >= MIN_BG1_TARGET_GENES
                and min(val_behavior_strands.values()) >= MIN_BG1_TARGET_GENES_PER_STRAND
                and len(val_route) >= MIN_BG2_TARGET_GENES
                and min(val_route_strands.values()) >= MIN_BG2_TARGET_GENES_PER_STRAND
            ),
            "behavior": {
                "n": len(val_behavior),
                "strands": val_behavior_strands,
                "thresholds": [MIN_BG1_TARGET_GENES, MIN_BG1_TARGET_GENES_PER_STRAND],
            },
            "route": {
                "n": len(val_route),
                "strands": val_route_strands,
                "thresholds": [MIN_BG2_TARGET_GENES, MIN_BG2_TARGET_GENES_PER_STRAND],
            },
        },
        "optional_test_replication_supply": {
            "PASS": bool(
                not args.enable_test_replication
                or (
                    len(test_behavior) >= MIN_BG1_TARGET_GENES
                    and min(test_behavior_strands.values()) >= MIN_BG1_TARGET_GENES_PER_STRAND
                )
            ),
            "requested": bool(args.enable_test_replication),
            "n": len(test_behavior),
            "strands": test_behavior_strands,
            "thresholds": [MIN_BG1_TARGET_GENES, MIN_BG1_TARGET_GENES_PER_STRAND],
        },
        "panel_construction": {
            "PASS": construction.passed,
            "forced": True,
            "checks": dict(construction.checks),
        },
        "donor_mask_coordinate": {
            "PASS": donor_mask_exact,
            "forced": True,
        },
        "batch_and_order_invariance": {
            "PASS": batch_error <= BATCH_TOL,
            "max_abs_error": batch_error,
            "threshold": BATCH_TOL,
        },
        "positive_partner_corruption": {
            "PASS": bool(partner_ci[0] > SESOI and partner_max_ci[0] > 0.0),
            "linear": {"mean": float(partner_effect.mean()), "ci": list(partner_ci)},
            "max": {"mean": float(partner_max_effect.mean()), "ci": list(partner_max_ci)},
            "thresholds": {"linear": SESOI, "max": 0.0},
        },
        "partner_vs_nonpartner_specificity": {
            "PASS": bool(specificity_ci is not None and specificity_ci[0] > SESOI),
            "n": int(len(specificity)),
            "mean": float(specificity.mean()) if len(specificity) else None,
            "ci": list(specificity_ci) if specificity_ci is not None else None,
            "threshold": SESOI,
        },
        "target_resolution": {
            "PASS": bool(args.smoke or all(value <= SESOI for value in primary_mde)),
            "projected_validation_mde80": primary_mde,
            "target_n": len(val_behavior),
            "threshold": SESOI,
            "smoke_relaxed": args.smoke,
        },
        "planted_sensitivity": {
            "PASS": calibration["planted_power"] >= MIN_PLANTED_POWER,
            **calibration,
            "threshold": MIN_PLANTED_POWER,
        },
        "endpoint_familywise_false_positive_rate": {
            "PASS": calibration["endpoint_familywise_false_positive_rate"] <= MAX_CALIBRATED_FPR,
            "value": calibration["endpoint_familywise_false_positive_rate"],
            "threshold": MAX_CALIBRATED_FPR,
        },
        "joint_claim_composite_null_false_positive_rate": {
            "PASS": calibration["joint_claim_worst_case_false_positive_rate"] <= MAX_CALIBRATED_FPR,
            "value": calibration["joint_claim_worst_case_false_positive_rate"],
            "by_boundary_endpoint": calibration[
                "joint_claim_composite_null_false_positive_rate_by_boundary_endpoint"
            ],
            "threshold": MAX_CALIBRATED_FPR,
        },
        "value_linearity": {
            "PASS": route_metrics["value_linearity"] <= LINEARITY_TOL,
            "max_abs_error": route_metrics["value_linearity"],
            "threshold": LINEARITY_TOL,
        },
        "head_reconstruction": {
            "PASS": route_metrics["head_reconstruction"] <= LINEARITY_TOL,
            "max_abs_error": route_metrics["head_reconstruction"],
            "threshold": LINEARITY_TOL,
        },
        "injection_identity": {
            "PASS": max(route_metrics["injection_identity_linear"], route_metrics["injection_identity_max"]) <= INJECTION_TOL,
            "linear_error": route_metrics["injection_identity_linear"],
            "max_error": route_metrics["injection_identity_max"],
            "threshold": INJECTION_TOL,
        },
        "attention_live": {
            "PASS": route_metrics["attention_row_sum"] <= LINEARITY_TOL,
            "row_sum_error": route_metrics["attention_row_sum"],
            "threshold": LINEARITY_TOL,
        },
        "route_dose_sensitivity": {
            "PASS": route_metrics["dose_PASS"],
            "dose": route_metrics["dose"],
            "minimum_full_lift": MIN_ROUTE_LIFT,
        },
        "route_resolution": {
            "PASS": bool(args.smoke or all(value <= SESOI for value in route_mde)),
            "projected_validation_mde80": route_mde,
            "target_n": len(val_route),
            "threshold": SESOI,
            "smoke_relaxed": args.smoke,
        },
    }
    passed = all(check["PASS"] for check in checks.values())
    weights = checkpoint_weights_path(args.model)
    gtf_path = resolved(args.gtf)
    fasta_path = resolved(args.fasta)
    result = {
        "schema": PREFLIGHT_SCHEMA,
        "PASS": bool(passed),
        "smoke": bool(args.smoke),
        "status": "DEVELOPMENT_PREFLIGHT",
        "checkpoint_sha256": sha256_file(weights),
        "model_file_sha256": frozen_model_hashes,
        "gtf_sha256": frozen_gtf_hash,
        "fasta_sha256": frozen_fasta_hash,
        "protocol_sha256": sha256_file(PROTOCOL),
        "declaration_sha256": sha256_file(DECLARATION),
        "config": preflight_config(),
        "cohort": cohort,
        "target_cohorts": target_cohorts,
        "test_replication_enabled": bool(args.enable_test_replication),
        "checks": checks,
        "diagnostics": {
            "target_train_inference": target_inference,
            "route_train_inference": route_inference,
            "route_instrument": route_metrics,
            "outer_power": {
                "minus3_vs_minus4": minus_outer,
                "plus1_vs_plus2": plus_outer,
            },
            "native_max_margin_mean": float(native_max.mean()),
        },
        "cells": target_rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")

    manifest = ExperimentManifest.create(
        "splicebert-partner-selection:bg1-boundary-preflight-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_PREFLIGHT_PASS" if passed else "DEVELOPMENT_PREFLIGHT_FAILED",
        config={
            "smoke": args.smoke,
            "n_behavior": len(behavior),
            "n_route": len(route),
            "device": device,
            "PASS": passed,
        },
        seeds={"stable_roles": 0, "analysis": args.seed},
        notes=[
            "Train split only; no target role or target split can be authorized by a smoke artifact.",
            "Outer attenuation remains diagnostic unless its separate outcome-blind power flag is true.",
        ],
    )
    for path in (
        gtf_path,
        fasta_path,
        PROTOCOL,
        DECLARATION,
        Path(__file__),
        REPO / "scripts/splicebert/run_bg1_boundary_grammar.py",
        REPO / "scripts/splicebert/run_bg2_boundary_payload_recovery.py",
        REPO / "src/gfmcd/splicing/coupling.py",
        REPO / "src/gfmcd/splicing/circuit.py",
        REPO / "src/gfmcd/splicing/p3_stats.py",
        REPO / "src/gfmcd/splicing/payload.py",
        REPO / "src/gfmcd/circuits/hf_bert.py",
    ):
        manifest.add_input(resolved(path))
    model_path = resolved(args.model)
    if model_path.is_dir():
        for name in sorted(frozen_model_hashes):
            manifest.add_input(model_path / name)
    else:
        manifest.add_input(model_path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({
        "PASS": passed,
        "smoke": args.smoke,
        "failed": [name for name, check in checks.items() if not check["PASS"]],
        "outer_claim_bearing": {
            "minus3_vs_minus4": minus_outer["claim_bearing"],
            "plus1_vs_plus2": plus_outer["claim_bearing"],
        },
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
