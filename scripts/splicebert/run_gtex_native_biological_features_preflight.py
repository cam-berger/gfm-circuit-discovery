#!/usr/bin/env python
"""Preflight the fresh CDS-only biological-feature assay across five checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.rigor import benjamini_hochberg  # noqa: E402
from gfmcd.splicing.biological_cohort import (  # noqa: E402
    ACCEPTOR_WINDOW,
    DONOR_WINDOW,
    collect_prior_gene_exclusion,
    load_fresh_cds_cohort,
)
from gfmcd.splicing.biological_features import (  # noqa: E402
    codon_amino_acid,
    feature_delta,
    load_rbns_psam_directory,
    maximal_synonymous_recode,
    psam_score_vector,
    synonymous_saturation_variants,
    vienna_mfe,
    vienna_unpaired_mean,
)
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.context_localization import (  # noqa: E402
    RelativeWindow,
    shuffle_relative_window,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.grouped_stats import (  # noqa: E402
    grouped_bootstrap,
    grouped_signflip_pvalue,
)
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    ACCEPTOR_LABEL,
    DONOR_LABEL,
    NativeClassifierTask,
)
from gfmcd.splicing.native_mechanism import (  # noqa: E402
    cache_final_layer_inputs,
    capture_final_layer_run,
    compose_target_mlp_hidden,
    donor_token_positions,
)
from gfmcd.splicing.p3_annotation import load_annotation_safety_index  # noqa: E402
from gfmcd.splicing.phase import load_cds_intervals  # noqa: E402


EXPECTED_SEEDS = frozenset(range(2020, 2025))
N_PREFLIGHT = 16
N_BOOT = 5000
N_PERM = 9999
FINAL_LAYER = 5
MIN_REGION_EFFECT = 0.05
MIN_LOCAL_EFFECT = 5.0
MAX_IDENTITY_ERROR = 2e-5
MAX_Q = 0.05
DONOR_SHAM = RelativeWindow(16, 48)
ACCEPTOR_SHAM = RelativeWindow(-80, -48)


def parse_checkpoints(values: list[str]) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in found:
            raise ValueError(f"duplicate checkpoint seed {seed}")
        found[seed] = Path(path_text).expanduser().resolve()
    if set(found) != EXPECTED_SEEDS:
        raise ValueError(f"complete seed family required; found {sorted(found)}")
    return found


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _target_rows(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[rows, positions.to(tensor.device)]


def composition_errors(
    model: Any,
    tokenizer: Any,
    sequences: list[str],
    specs: dict[str, dict[str, Any]],
    *,
    device: str,
) -> dict[str, float]:
    """Verify direct replay and exact native final-MLP composition identities."""

    task = NativeClassifierTask(model, tokenizer, device=device, batch_size=len(sequences))
    encoded = task.encode(sequences)
    adapter = SpliceBertAdapter(model, tokenizer, device=device)
    inputs = cache_final_layer_inputs(model, encoded, layer=FINAL_LAYER)
    run = capture_final_layer_run(adapter, inputs, layer=FINAL_LAYER)
    with torch.no_grad():
        ordinary = model.bert(**encoded, return_dict=True).last_hidden_state
    results = {
        "final_layer_replay_max_abs_error": float(
            (ordinary - run.hidden).abs().max().detach().cpu()
        )
    }
    for endpoint, spec in specs.items():
        nucleotide_positions = spec["targets"][: len(sequences)]
        tokens = donor_token_positions(
            nucleotide_positions,
            prefix_tokens=task.prefix_tokens,
            device=device,
        )
        native_hidden = _target_rows(run.hidden, tokens)
        native_logits = model.classifier(model.dropout(native_hidden)).float()
        native_margin = (
            native_logits[:, int(spec["label"])] - native_logits[:, 0]
        )
        composed_hidden = compose_target_mlp_hidden(
            run,
            tokens,
            _target_rows(run.mlp_write, tokens),
            adapter.encoder_layers[FINAL_LAYER].output.LayerNorm,
        )
        composed_logits = model.classifier(model.dropout(composed_hidden)).float()
        composed_margin = (
            composed_logits[:, int(spec["label"])] - composed_logits[:, 0]
        )
        results[f"{endpoint}_complete_mlp_composition_max_abs_error"] = float(
            (native_margin - composed_margin).abs().max().detach().cpu()
        )
    return results


def summary(values: np.ndarray, groups: np.ndarray, *, seed: int) -> dict[str, Any]:
    row = grouped_bootstrap(values, groups, n_boot=N_BOOT, seed=seed)
    row["p_signflip_add_one"] = grouped_signflip_pvalue(
        values,
        groups,
        n_perm=N_PERM,
        seed=seed + 50_000,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--basic-gtf", required=True)
    parser.add_argument("--comprehensive-gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--prior-results", required=True)
    parser.add_argument("--rbns-psams", required=True)
    parser.add_argument("--n", type=int, default=N_PREFLIGHT)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != N_PREFLIGHT:
        raise ValueError(f"preflight is frozen at n={N_PREFLIGHT}")
    torch.set_grad_enabled(False)
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    checkpoints = parse_checkpoints(args.checkpoint)
    base_model = Path(args.base_model).expanduser().resolve()
    basic_gtf = Path(args.basic_gtf).expanduser().resolve()
    comprehensive_gtf = Path(args.comprehensive_gtf).expanduser().resolve()
    fasta = Path(args.fasta).expanduser().resolve()
    prior_results = Path(args.prior_results).expanduser().resolve()
    rbns_root = Path(args.rbns_psams).expanduser().resolve()
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_BIOLOGICAL_FEATURES_V1.md"
    )

    prior = collect_prior_gene_exclusion(prior_results)
    annotation_index = load_annotation_safety_index(comprehensive_gtf)
    cds_intervals = load_cds_intervals(comprehensive_gtf)
    examples, annotation_audits, cohort_counts = load_fresh_cds_cohort(
        str(basic_gtf),
        str(fasta),
        annotation_index=annotation_index,
        cds_intervals=cds_intervals,
        analysis_partition="discovery",
        n=args.n,
        tag="gtex-native-biological-features-preflight-v1",
        excluded_gene_ids=prior.gene_ids,
        discovery_percent=50,
    )
    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    clean_sequences = [example.sequence for example in examples]
    models = load_rbns_psam_directory(rbns_root)

    specs: dict[str, dict[str, Any]] = {
        "donor": {
            "window": DONOR_WINDOW,
            "sham": DONOR_SHAM,
            "anchors": [int(example.donor_pos) for example in examples],
            "targets": [int(example.donor_pos) - 1 for example in examples],
            "label": DONOR_LABEL,
        },
        "acceptor": {
            "window": ACCEPTOR_WINDOW,
            "sham": ACCEPTOR_SHAM,
            "anchors": [int(example.next_acceptor_pos) for example in examples],
            "targets": [int(example.next_acceptor_pos) for example in examples],
            "label": ACCEPTOR_LABEL,
        },
    }

    sequence_conditions: dict[str, dict[str, list[str]]] = {}
    sequence_audits: dict[str, list[dict[str, Any]]] = {}
    translation_ok = True
    new_splice_ok = True
    recode_dose_pass: list[bool] = []
    for endpoint, spec in specs.items():
        sequence_conditions[endpoint] = {"target": [], "sham": [], "local": []}
        sequence_audits[endpoint] = []
        for example, anchor in zip(examples, spec["anchors"]):
            target, target_audit = shuffle_relative_window(
                example.sequence,
                anchor=anchor,
                window=spec["window"],
                key=(example.example_id, endpoint, "biological-preflight-target"),
                draw=0,
            )
            sham, sham_audit = shuffle_relative_window(
                example.sequence,
                anchor=anchor,
                window=spec["sham"],
                key=(example.example_id, endpoint, "biological-preflight-sham"),
                draw=0,
            )
            sequence_conditions[endpoint]["target"].append(target)
            sequence_conditions[endpoint]["sham"].append(sham)
            if endpoint == "donor":
                local = substitute(example.sequence, int(example.donor_pos), "CC")
            else:
                local = substitute(
                    example.sequence,
                    int(example.next_acceptor_pos) - 2,
                    "CC",
                )
            sequence_conditions[endpoint]["local"].append(local)

            saturation = synonymous_saturation_variants(
                example,
                spec["window"],
                cds_intervals,
                anchor=anchor,
            )
            for variant in saturation:
                translation_ok &= (
                    codon_amino_acid(variant.original_codon)
                    == codon_amino_acid(variant.alternate_codon)
                )
            recodes = []
            for draw in (0, 1):
                sequence, changes = maximal_synonymous_recode(
                    example,
                    spec["window"],
                    cds_intervals,
                    anchor=anchor,
                    draw=draw,
                    forbid_new_splice_dinucleotides=True,
                )
                recodes.append((sequence, changes))
                start, end = spec["window"].absolute(anchor, len(sequence))
                clean_positions = {
                    pos
                    for pos in range(start, end - 1)
                    if example.sequence[pos : pos + 2] in {"AG", "GT"}
                }
                changed_positions = {
                    pos
                    for pos in range(start, end - 1)
                    if sequence[pos : pos + 2] in {"AG", "GT"}
                }
                new_splice_ok &= changed_positions <= clean_positions
                dose = sum(
                    left != right
                    for left, right in zip(example.sequence[start:end], sequence[start:end])
                )
                recode_dose_pass.append(dose >= 0.20 * 30)
            sequence_audits[endpoint].append(
                {
                    "example_id": example.example_id,
                    "target_shuffle": target_audit.to_dict(),
                    "sham_shuffle": sham_audit.to_dict(),
                    "n_saturation_variants": len(saturation),
                    "recode_draws": [
                        {
                            "hamming": sum(
                                left != right
                                for left, right in zip(example.sequence, sequence)
                            ),
                            "n_changed_codons": len(changes),
                        }
                        for sequence, changes in recodes
                    ],
                }
            )

    first = examples[0]
    donor_start, donor_end = DONOR_WINDOW.absolute(
        int(first.donor_pos), len(first.sequence)
    )
    identity_psam = psam_score_vector(
        first.sequence,
        models,
        start=donor_start,
        end=donor_end,
    )
    context_start = donor_start - 84
    context_end = donor_end + 85
    fold_context = first.sequence[context_start:context_end]
    fold_region_start = donor_start - context_start
    fold_region_end = donor_end - context_start
    identity_accessibility = vienna_unpaired_mean(
        fold_context,
        start=fold_region_start,
        end=fold_region_end,
    )
    identity_mfe = vienna_mfe(fold_context)
    sequence_checks = {
        "cohort_size": len(examples) == N_PREFLIGHT,
        "genes_unique": len(set(groups.tolist())) == N_PREFLIGHT,
        "prior_genes_excluded": not set(groups.tolist()) & set(prior.gene_ids),
        "all_annotation_gates": all(
            audit.passed
            for rows in annotation_audits.values()
            for audit in rows.values()
        ),
        "ten_complete_codons_each": all(
            len(audit.complete_codon_starts) == 10
            for rows in annotation_audits.values()
            for audit in rows.values()
        ),
        "synonymous_translation_exact": bool(translation_ok),
        "no_new_ag_gt_in_recodes": bool(new_splice_ok),
        "max_recode_dose_90_percent": float(np.mean(recode_dose_pass)) >= 0.90,
        "shuffle_invariants": all(
            all(
                row[condition][name]
                for name in (
                    "length_preserved",
                    "outside_preserved",
                    "endpoints_preserved",
                    "mononucleotides_preserved",
                    "dinucleotides_preserved",
                )
            )
            for rows in sequence_audits.values()
            for row in rows
            for condition in ("target_shuffle", "sham_shuffle")
        ),
        "shuffle_dose": all(
            row[condition]["hamming_fraction"] >= 0.25
            for rows in sequence_audits.values()
            for row in rows
            for condition in ("target_shuffle", "sham_shuffle")
        ),
        "all_79_psams_loaded": len(models) == 79,
        "psam_identity_exact": feature_delta(identity_psam, identity_psam)
        == {name: 0.0 for name in sorted(identity_psam)},
        "rna_instruments_finite": bool(
            np.isfinite(identity_accessibility) and np.isfinite(identity_mfe)
        ),
    }
    if not all(sequence_checks.values()):
        raise RuntimeError(f"sequence preflight failed: {sequence_checks}")

    seed_effects: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    seed_reports: dict[str, Any] = {}
    numerical_checks: dict[str, dict[str, float]] = {}
    for seed, checkpoint in sorted(checkpoints.items()):
        print(f"checkpoint {seed}: {len(examples)} fresh CDS-only genes", flush=True)
        model, tokenizer, report = load_gtex_token_classifier(
            base_model,
            checkpoint,
            device=device,
        )
        task = NativeClassifierTask(
            model,
            tokenizer,
            device=device,
            batch_size=args.batch,
        )
        clean_logits = task.logits(clean_sequences)
        identity_logits = task.logits([sequence[:] for sequence in clean_sequences[:2]])
        numerical = {
            "identity_input_max_abs_error": float(
                np.max(np.abs(clean_logits[:2] - identity_logits))
            ),
            "final_projection_max_abs_error": task.final_projection_error(
                clean_sequences[:2]
            ),
        }
        if seed == 2020:
            numerical.update(
                composition_errors(
                    model,
                    tokenizer,
                    clean_sequences[:2],
                    specs,
                    device=device,
                )
            )
        numerical_checks[str(seed)] = numerical
        seed_effects[seed] = {}
        for endpoint, spec in specs.items():
            clean_margin = task.score(
                clean_logits,
                spec["targets"],
                label=int(spec["label"]),
            ).margin
            seed_effects[seed][endpoint] = {}
            for condition, sequences in sequence_conditions[endpoint].items():
                changed_logits = task.logits(sequences)
                changed_margin = task.score(
                    changed_logits,
                    spec["targets"],
                    label=int(spec["label"]),
                ).margin
                seed_effects[seed][endpoint][condition] = clean_margin - changed_margin
            seed_effects[seed][endpoint]["target_minus_sham"] = (
                seed_effects[seed][endpoint]["target"]
                - seed_effects[seed][endpoint]["sham"]
            )
        seed_reports[str(seed)] = report.to_dict()
        del task, model, tokenizer, clean_logits, identity_logits
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    family_summaries: dict[str, dict[str, Any]] = {}
    primary_entries: list[tuple[str, str, float]] = []
    for endpoint_index, endpoint in enumerate(("donor", "acceptor")):
        family_summaries[endpoint] = {}
        for condition_index, condition in enumerate(
            ("target", "sham", "target_minus_sham", "local")
        ):
            values = np.mean(
                np.stack(
                    [
                        seed_effects[seed][endpoint][condition]
                        for seed in sorted(EXPECTED_SEEDS)
                    ]
                ),
                axis=0,
            )
            row = summary(
                values,
                groups,
                seed=70_000 + endpoint_index * 100 + condition_index,
            )
            row["seed_estimates"] = {
                str(seed): float(seed_effects[seed][endpoint][condition].mean())
                for seed in sorted(EXPECTED_SEEDS)
            }
            family_summaries[endpoint][condition] = row
            if condition in {"target", "target_minus_sham"}:
                primary_entries.append(
                    (endpoint, condition, float(row["p_signflip_add_one"]))
                )
    qvalues = benjamini_hochberg([entry[2] for entry in primary_entries])
    for (endpoint, condition, _), qvalue in zip(primary_entries, qvalues):
        family_summaries[endpoint][condition]["q_bh_primary_family"] = float(qvalue)

    endpoint_gates: dict[str, dict[str, bool]] = {}
    for endpoint in ("donor", "acceptor"):
        target = family_summaries[endpoint]["target"]
        contrast = family_summaries[endpoint]["target_minus_sham"]
        local = family_summaries[endpoint]["local"]
        endpoint_gates[endpoint] = {
            "target_effect_at_least_0.05": float(target["estimate"]) >= MIN_REGION_EFFECT,
            "target_ci_above_zero": float(target["ci_low"]) > 0.0,
            "target_positive_5_of_5": all(
                float(value) > 0.0 for value in target["seed_estimates"].values()
            ),
            "target_q_at_most_0.05": float(target["q_bh_primary_family"]) <= MAX_Q,
            "target_minus_sham_ci_above_zero": float(contrast["ci_low"]) > 0.0,
            "target_minus_sham_q_at_most_0.05": (
                float(contrast["q_bh_primary_family"]) <= MAX_Q
            ),
            "local_effect_at_least_5": float(local["estimate"]) >= MIN_LOCAL_EFFECT,
            "local_effect_at_least_5_each_seed": all(
                float(value) >= MIN_LOCAL_EFFECT
                for value in local["seed_estimates"].values()
            ),
        }
    numerical_gate = all(
        value <= MAX_IDENTITY_ERROR
        for rows in numerical_checks.values()
        for value in rows.values()
    )
    passed = all(sequence_checks.values()) and numerical_gate and all(
        all(rows.values()) for rows in endpoint_gates.values()
    )

    raw = []
    for index, example in enumerate(examples):
        raw.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "chrom": example.chrom,
                "annotation": {
                    endpoint: asdict(audit)
                    for endpoint, audit in annotation_audits[example.example_id].items()
                },
                "seed_effects": {
                    str(seed): {
                        endpoint: {
                            condition: float(values[index])
                            for condition, values in conditions.items()
                        }
                        for endpoint, conditions in seed_effects[seed].items()
                    }
                    for seed in sorted(EXPECTED_SEEDS)
                },
            }
        )
    result = {
        "experiment": "gtex_native_biological_features_preflight_v1",
        "status": "PASS" if passed else "FAIL",
        "protocol": str(protocol.resolve()),
        "device": device,
        "cohort": {
            "n": len(examples),
            "n_genes": len(set(groups.tolist())),
            "counts": cohort_counts,
            "prior_gene_count": len(prior.gene_ids),
            "prior_gene_source_aggregate_sha256": prior.aggregate_sha256,
            "prior_gene_source_files": list(prior.source_files),
        },
        "sequence_checks": sequence_checks,
        "sequence_instrument": {
            "rbns_model_count": len(models),
            "identity_accessibility": identity_accessibility,
            "identity_mfe": identity_mfe,
            "max_recode_dose_pass_fraction": float(np.mean(recode_dose_pass)),
            "audits": sequence_audits,
        },
        "numerical_checks": numerical_checks,
        "numerical_gate": numerical_gate,
        "family_summaries": family_summaries,
        "endpoint_gates": endpoint_gates,
        "checkpoint_reports": seed_reports,
        "raw": raw,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    manifest = ExperimentManifest.create(
        "gtex-native-biological-features-preflight-v1",
        sys.argv,
        repo=REPO,
        status=result["status"],
        config={
            "n": N_PREFLIGHT,
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
            "min_region_effect": MIN_REGION_EFFECT,
            "min_local_effect": MIN_LOCAL_EFFECT,
            "max_identity_error": MAX_IDENTITY_ERROR,
        },
        inputs=[
            file_record(path)
            for path in (
                protocol,
                basic_gtf,
                comprehensive_gtf,
                fasta,
                *checkpoints.values(),
            )
        ],
        outputs=[file_record(output)],
        notes=[
            "All cohort filtering and sequence construction preceded model loading.",
            "RNAplfold settings: W=150, L=100, u=7 on 201 nt.",
        ],
    )
    manifest.write(manifest_path)
    print(json.dumps({"status": result["status"], "gates": endpoint_gates}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
