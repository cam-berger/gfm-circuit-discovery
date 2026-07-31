#!/usr/bin/env python
"""Five-seed paired-acceptor coupling assay through the released GTEx classifier."""

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

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.rigor import benjamini_hochberg  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    matched_decoy_donor,
    pick_device,
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
    load_annotation_cohort,
)

EXPECTED_SEEDS = frozenset(range(2020, 2025))
SCANNED_METRICS = (
    "partner_cc",
    "specificity_vs_other_donor",
    "specificity_vs_nonpartner_ag",
    "partner_tt",
)


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
        raise ValueError(
            f"complete seed family required; found {sorted(checkpoints)}"
        )
    return checkpoints


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def nearest_nonpartner_ag(example: Any) -> int | None:
    candidates = [position for position in example.other_ag if position >= 3]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda position: abs(position - int(example.next_acceptor_pos)),
    )


def score_seed(
    base_model: Path,
    checkpoint: Path,
    examples: list[Any],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    model, tokenizer, report = load_gtex_token_classifier(
        base_model,
        checkpoint,
        device=device,
    )
    task = NativeClassifierTask(
        model,
        tokenizer,
        device=device,
        batch_size=batch_size,
    )
    donor_labels = [example.donor_pos - 1 for example in examples]
    acceptors = [int(example.next_acceptor_pos) for example in examples]
    other_donors = [matched_decoy_donor(example) for example in examples]
    nonpartner_ags = [nearest_nonpartner_ag(example) for example in examples]
    clean_sequences = [example.sequence for example in examples]
    partner_cc_sequences = [
        substitute(example.sequence, int(example.next_acceptor_pos) - 2, "CC")
        for example in examples
    ]
    partner_tt_sequences = [
        substitute(example.sequence, int(example.next_acceptor_pos) - 2, "TT")
        for example in examples
    ]
    nonpartner_sequences = [
        (
            substitute(example.sequence, int(position) - 2, "CC")
            if position is not None
            else example.sequence
        )
        for example, position in zip(examples, nonpartner_ags)
    ]

    clean_logits = task.logits(clean_sequences)
    partner_cc_logits = task.logits(partner_cc_sequences)
    partner_tt_logits = task.logits(partner_tt_sequences)
    nonpartner_logits = task.logits(nonpartner_sequences)
    clean = task.score(clean_logits, donor_labels, label=DONOR_LABEL)
    partner_cc_margin = task.score(
        partner_cc_logits,
        donor_labels,
        label=DONOR_LABEL,
    ).margin
    partner_tt_margin = task.score(
        partner_tt_logits,
        donor_labels,
        label=DONOR_LABEL,
    ).margin
    nonpartner_margin = task.score(
        nonpartner_logits,
        donor_labels,
        label=DONOR_LABEL,
    ).margin
    partner_cc = clean.margin - partner_cc_margin
    partner_tt = clean.margin - partner_tt_margin
    nonpartner_ag = clean.margin - nonpartner_margin
    nonpartner_ag[
        np.asarray([position is None for position in nonpartner_ags], dtype=bool)
    ] = np.nan

    other_donor = np.full(len(examples), np.nan, dtype=np.float64)
    keep_other = [
        index for index, position in enumerate(other_donors) if position is not None
    ]
    if keep_other:
        other_clean = task.score(
            clean_logits[keep_other],
            [int(other_donors[index]) - 1 for index in keep_other],
            label=DONOR_LABEL,
        ).margin
        other_corrupt = task.score(
            partner_cc_logits[keep_other],
            [int(other_donors[index]) - 1 for index in keep_other],
            label=DONOR_LABEL,
        ).margin
        other_donor[keep_other] = other_clean - other_corrupt

    clean_acceptor = task.score(
        clean_logits,
        acceptors,
        label=ACCEPTOR_LABEL,
    )
    corrupt_acceptor = task.score(
        partner_cc_logits,
        acceptors,
        label=ACCEPTOR_LABEL,
    )
    acceptor_manipulation = clean_acceptor.margin - corrupt_acceptor.margin

    result = {
        "partner_cc": partner_cc,
        "partner_tt": partner_tt,
        "other_donor": other_donor,
        "nonpartner_ag": nonpartner_ag,
        "specificity_vs_other_donor": partner_cc - other_donor,
        "specificity_vs_nonpartner_ag": partner_cc - nonpartner_ag,
        "acceptor_manipulation": acceptor_manipulation,
        "clean_donor_margin": clean.margin,
        "clean_donor_probability": clean.probability,
        "clean_acceptor_margin": clean_acceptor.margin,
        "clean_acceptor_probability": clean_acceptor.probability,
        "other_donor_positions": other_donors,
        "nonpartner_ag_positions": nonpartner_ags,
        "report": report.to_dict(),
    }
    del task, model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return result


def summarize(
    result: dict[str, Any],
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    metrics = (
        "partner_cc",
        "partner_tt",
        "other_donor",
        "nonpartner_ag",
        "specificity_vs_other_donor",
        "specificity_vs_nonpartner_ag",
        "acceptor_manipulation",
        "clean_donor_margin",
        "clean_acceptor_margin",
    )
    summary = {
        metric: grouped_bootstrap(
            result[metric],
            groups,
            n_boot=n_boot,
            seed=seed + index,
        )
        for index, metric in enumerate(metrics)
    }
    summary["clean_donor_probability_mean"] = float(
        np.mean(result["clean_donor_probability"])
    )
    summary["clean_donor_positive_fraction"] = float(
        np.mean(result["clean_donor_margin"] > 0)
    )
    summary["clean_acceptor_probability_mean"] = float(
        np.mean(result["clean_acceptor_probability"])
    )
    summary["clean_acceptor_positive_fraction"] = float(
        np.mean(result["clean_acceptor_margin"] > 0)
    )
    summary["checkpoint"] = result["report"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
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
        raise ValueError("native-classifier target cohort is frozen at n=400")
    preflight_path = Path(args.preflight).resolve()
    protocol_path = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CLASSIFIER_V2.md"
    )
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("native-classifier preflight did not pass")
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
    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    results: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for seed, checkpoint in sorted(checkpoints.items()):
        print(f"seed {seed}: native coupling on {len(examples)} fixed examples")
        name = str(seed)
        results[name] = score_seed(
            base_model,
            checkpoint,
            examples,
            device=device,
            batch_size=args.batch,
        )
        summaries[name] = summarize(
            results[name],
            groups,
            n_boot=args.n_boot,
            seed=seed,
        )

    pvalue_entries: list[tuple[str, str, float]] = []
    for seed in sorted(EXPECTED_SEEDS):
        name = str(seed)
        for metric_index, metric in enumerate(SCANNED_METRICS):
            stat = grouped_bootstrap(
                results[name][metric],
                groups,
                n_boot=args.n_boot,
                seed=seed + 100 * metric_index,
            )
            pvalue = grouped_signflip_pvalue(
                results[name][metric],
                groups,
                n_perm=args.n_perm,
                seed=seed + 1000 * metric_index,
            )
            stat["p_add_one_two_sided"] = pvalue
            summaries[name][metric] = stat
            pvalue_entries.append((name, metric, pvalue))
    individual_q = benjamini_hochberg(
        [entry[2] for entry in pvalue_entries]
    )
    for (name, metric, _), qvalue in zip(pvalue_entries, individual_q):
        summaries[name][metric]["q_bh_20_tests"] = float(qvalue)

    fine_names = [str(seed) for seed in sorted(EXPECTED_SEEDS)]
    family: dict[str, Any] = {}
    family_pvalues: list[float] = []
    for metric_index, metric in enumerate(SCANNED_METRICS):
        values = np.nanmean(
            np.stack([results[name][metric] for name in fine_names]),
            axis=0,
        )
        stat = grouped_bootstrap(
            values,
            groups,
            n_boot=args.n_boot,
            seed=9000 + metric_index,
        )
        pvalue = grouped_signflip_pvalue(
            values,
            groups,
            n_perm=args.n_perm,
            seed=10000 + metric_index,
        )
        stat["p_add_one_two_sided"] = pvalue
        stat["seed_estimates"] = {
            name: summaries[name][metric]["estimate"] for name in fine_names
        }
        stat["n_positive_seeds"] = int(
            sum(value > 0 for value in stat["seed_estimates"].values())
        )
        family[metric] = stat
        family_pvalues.append(pvalue)
    for metric, qvalue in zip(
        SCANNED_METRICS,
        benjamini_hochberg(family_pvalues),
    ):
        family[metric]["q_bh_4_family_tests"] = float(qvalue)

    manipulation_values = np.mean(
        np.stack(
            [results[name]["acceptor_manipulation"] for name in fine_names]
        ),
        axis=0,
    )
    family["acceptor_manipulation"] = grouped_bootstrap(
        manipulation_values,
        groups,
        n_boot=args.n_boot,
        seed=11000,
    )
    gate_checks = {
        "partner_cc_interval_positive": family["partner_cc"]["ci_low"] > 0,
        "partner_cc_replication": family["partner_cc"]["n_positive_seeds"] >= 4,
        "partner_tt_interval_positive": family["partner_tt"]["ci_low"] > 0,
        "specificity_vs_other_donor_positive": (
            family["specificity_vs_other_donor"]["ci_low"] > 0
        ),
        "specificity_vs_nonpartner_ag_positive": (
            family["specificity_vs_nonpartner_ag"]["ci_low"] > 0
        ),
        "acceptor_manipulation_positive": (
            family["acceptor_manipulation"]["ci_low"] > 0
        ),
    }
    gate_pass = all(gate_checks.values())

    raw: dict[str, list[dict[str, Any]]] = {}
    for name, result in results.items():
        raw[name] = [
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "partner_cc": float(result["partner_cc"][index]),
                "partner_tt": float(result["partner_tt"][index]),
                "other_donor": (
                    float(result["other_donor"][index])
                    if np.isfinite(result["other_donor"][index])
                    else None
                ),
                "nonpartner_ag": (
                    float(result["nonpartner_ag"][index])
                    if np.isfinite(result["nonpartner_ag"][index])
                    else None
                ),
                "acceptor_manipulation": float(
                    result["acceptor_manipulation"][index]
                ),
                "clean_donor_margin": float(
                    result["clean_donor_margin"][index]
                ),
                "clean_donor_probability": float(
                    result["clean_donor_probability"][index]
                ),
                "clean_acceptor_margin": float(
                    result["clean_acceptor_margin"][index]
                ),
                "clean_acceptor_probability": float(
                    result["clean_acceptor_probability"][index]
                ),
                "other_donor_position": result["other_donor_positions"][index],
                "nonpartner_ag_position": result["nonpartner_ag_positions"][index],
            }
            for index, example in enumerate(examples)
        ]

    artifact = {
        "study": "splicebert_gtex_finetune_shift",
        "stage": "native classifier coupling",
        "protocol": str(protocol_path.relative_to(REPO)),
        "status": "VERIFICATION_FROZEN_AFTER_GFS4",
        "scope": (
            "Released three-class classifiers on the fixed validation cohort; "
            "no base native-head comparison and no mechanism claim."
        ),
        "models": summaries,
        "family_mean": family,
        "native_coupling_gate": {
            "checks": gate_checks,
            "PASS": gate_pass,
            "representation_bridge_authorized": gate_pass,
        },
        "multiplicity": {
            "individual_family": "BH across 4 estimands x 5 seeds = 20",
            "family_mean": "BH across the 4 five-seed-mean estimands",
            "permutation": (
                f"{args.n_perm} gene-level sign flips; exact add-one correction"
            ),
        },
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, default=_json_default) + "\n")

    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_classifier_coupling",
        sys.argv,
        repo=REPO,
        status=artifact["status"],
        config={
            "split": "val",
            "cohort_tag": "sealed-unc-val",
            "n": args.n,
            "batch": args.batch,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "device": device,
            "scanned_metrics": list(SCANNED_METRICS),
            "readout": "class_1_donor_logit_minus_class_0_no_site_logit",
            "donor_label_coordinate": "donor_motif_start_minus_1",
        },
        seeds={
            "published_checkpoints_start": 2020,
            "bootstrap_family": 9000,
            "signflip_family": 10000,
        },
        notes=[
            "Native-head protocol was frozen after GFS4 and before any native-head forward.",
            "All five published fine-tuning seeds were run sequentially.",
            "Representation bridge remains gated by the artifact's frozen behavior checks.",
        ],
    )
    for path in (
        preflight_path,
        protocol_path,
        base_model / "config.json",
        base_model / "tokenizer.json",
        Path(args.gtf),
        Path(args.fasta),
        *checkpoints.values(),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(
        f"saved -> {output}; native coupling gate "
        f"{'PASS' if gate_pass else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
