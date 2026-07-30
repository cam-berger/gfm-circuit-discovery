#!/usr/bin/env python
"""Discover sequence features for the fresh CDS-only classifier dependencies."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.biological_cohort import (  # noqa: E402
    ACCEPTOR_WINDOW,
    DONOR_WINDOW,
    collect_prior_gene_exclusion,
    load_fresh_cds_cohort,
)
from gfmcd.splicing.biological_features import (  # noqa: E402
    beam_synonymous_optimize,
    codon_log_preference,
    complete_codon_starts,
    kmer_count_vector,
    load_rbns_psam_directory,
    maximal_synonymous_recode,
    psam_region_score,
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
from gfmcd.splicing.grouped_stats import grouped_bootstrap  # noqa: E402
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    ACCEPTOR_LABEL,
    DONOR_LABEL,
    NativeClassifierTask,
)
from gfmcd.splicing.p3_annotation import (  # noqa: E402
    load_annotation_safety_index,
    validate_annotation_block,
    window_block_to_genomic,
)
from gfmcd.splicing.phase import load_cds_intervals  # noqa: E402
from gfmcd.splicing.splice_landscape import (  # noqa: E402
    compare_landscapes,
    maxent_scorers,
)


N_DISCOVERY = 128
N_CONFIRMATION = 200
N_BASELINE = 400
N_BOOT = 5000
SEED = 2020
DONOR_SHAM = RelativeWindow(16, 48)
ACCEPTOR_SHAM = RelativeWindow(-80, -48)
ELASTIC_GRID = tuple(
    (alpha, l1_ratio)
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1)
    for l1_ratio in (0.1, 0.5, 0.9)
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def segment(example, window: RelativeWindow, anchor: int, sequence: str | None = None) -> str:
    start, end = window.absolute(anchor, len(example.sequence))
    return (sequence or example.sequence)[start:end]


def annotation_intronic(
    example,
    window: RelativeWindow,
    anchor: int,
    annotation_index,
) -> bool:
    start, end = window.absolute(anchor, len(example.sequence))
    metadata = example.metadata
    genomic_start, genomic_end = window_block_to_genomic(
        start,
        end,
        window_start=int(metadata["genomic_window_start"]),
        window_end=int(metadata["genomic_window_end"]),
        strand=example.strand,
    )
    return validate_annotation_block(
        annotation_index,
        chrom=example.chrom,
        start=genomic_start,
        end=genomic_end,
        strand=example.strand,
        gene_id=example.gene_id,
        context="upstream_intron",
        splice_clearance=0,
    ).passed


def fit_exon_identity(
    examples,
    annotation_index,
) -> tuple[dict[str, Any], Any]:
    """Fit the frozen annotation-only 1–4-mer exon-versus-intron baseline."""

    vectors = []
    labels = []
    groups = []
    names: tuple[str, ...] | None = None
    for example in examples:
        for window, sham, anchor in (
            (DONOR_WINDOW, DONOR_SHAM, int(example.donor_pos)),
            (
                ACCEPTOR_WINDOW,
                ACCEPTOR_SHAM,
                int(example.next_acceptor_pos),
            ),
        ):
            if not annotation_intronic(example, sham, anchor, annotation_index):
                continue
            positive, current_names = kmer_count_vector(segment(example, window, anchor))
            negative, negative_names = kmer_count_vector(segment(example, sham, anchor))
            if current_names != negative_names:
                raise RuntimeError("k-mer basis changed between segments")
            names = current_names
            vectors.extend((positive, negative))
            labels.extend((1, 0))
            groups.extend((example.gene_id, example.gene_id))
    x = np.asarray(vectors, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=object)
    if x.shape[0] < 500 or len(np.unique(group_array)) < 100 or names is None:
        raise RuntimeError("insufficient annotation-only exon-identity training data")
    best_c = None
    best_loss = np.inf
    folds = GroupKFold(n_splits=5)
    cv_rows = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        fold_losses = []
        for train, test in folds.split(x, y, group_array):
            scaler = StandardScaler().fit(x[train])
            model = LogisticRegression(
                C=c_value,
                max_iter=2000,
                random_state=0,
            ).fit(scaler.transform(x[train]), y[train])
            probability = model.predict_proba(scaler.transform(x[test]))[:, 1]
            eps = np.finfo(np.float64).eps
            loss = -np.mean(
                y[test] * np.log(np.maximum(probability, eps))
                + (1 - y[test]) * np.log(np.maximum(1 - probability, eps))
            )
            fold_losses.append(float(loss))
        cv_rows.append({"C": c_value, "fold_log_loss": fold_losses})
        candidate_loss = float(np.mean(fold_losses))
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_c = c_value
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=float(best_c),
        max_iter=2000,
        random_state=0,
    ).fit(scaler.transform(x), y)

    def score(sequence: str) -> float:
        vector, observed_names = kmer_count_vector(sequence)
        if observed_names != names:
            raise RuntimeError("k-mer basis mismatch")
        return float(model.decision_function(scaler.transform(vector[None]))[0])

    record = {
        "n_segments": int(x.shape[0]),
        "n_genes": int(len(np.unique(group_array))),
        "positive_fraction": float(y.mean()),
        "kmer_names": list(names),
        "selected_C": float(best_c),
        "cv": cv_rows,
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "training_accuracy": float(model.score(scaler.transform(x), y)),
    }
    return record, score


def fit_elastic_grid(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Nested grouped CV for the preregistered descriptor horse race."""

    outer = GroupKFold(n_splits=5)
    fold_rows = []
    coefficients = []
    predictions = np.empty_like(y, dtype=np.float64)
    for outer_index, (train, test) in enumerate(outer.split(x, y, groups)):
        inner_groups = groups[train]
        inner = GroupKFold(n_splits=3)
        grid_losses = []
        for alpha, l1_ratio in ELASTIC_GRID:
            losses = []
            for inner_train, inner_test in inner.split(
                x[train],
                y[train],
                inner_groups,
            ):
                scaler = StandardScaler().fit(x[train][inner_train])
                model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=10000,
                    random_state=0,
                ).fit(
                    scaler.transform(x[train][inner_train]),
                    y[train][inner_train],
                )
                prediction = model.predict(scaler.transform(x[train][inner_test]))
                losses.append(mean_squared_error(y[train][inner_test], prediction))
            grid_losses.append((float(np.mean(losses)), alpha, l1_ratio))
        _, alpha, l1_ratio = min(grid_losses)
        scaler = StandardScaler().fit(x[train])
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10000,
            random_state=0,
        ).fit(scaler.transform(x[train]), y[train])
        prediction = model.predict(scaler.transform(x[test]))
        predictions[test] = prediction
        coefficients.append(model.coef_)
        fold_rows.append(
            {
                "fold": outer_index,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "mse": float(mean_squared_error(y[test], prediction)),
                "n_train": len(train),
                "n_test": len(test),
            }
        )
    return {
        "folds": fold_rows,
        "coefficient": np.asarray(coefficients, dtype=np.float64),
        "predictions": predictions,
        "mse": float(mean_squared_error(y, predictions)),
    }


def descriptor_values(
    sequence: str,
    example,
    *,
    window: RelativeWindow,
    anchor: int,
    codon_starts: tuple[int, ...],
    codon_counts: Counter,
    psams,
    exon_score,
) -> dict[str, float]:
    start, end = window.absolute(anchor, len(sequence))
    psam = psam_score_vector(sequence, psams, start=start, end=end)
    values = {f"psam:{name}": float(score) for name, score in psam.items()}
    region = sequence[start:end]
    context_start = start - 84
    context_end = end + 85
    context = sequence[context_start:context_end]
    values.update(
        {
            "codon_preference": codon_log_preference(
                sequence,
                codon_starts,
                codon_counts,
            ),
            "exon_identity": float(exon_score(region)),
            "rna_accessibility": vienna_unpaired_mean(
                context,
                start=start - context_start,
                end=end - context_start,
            ),
            "mfe": vienna_mfe(context),
        }
    )
    return values


def psam_information(model) -> float:
    components = []
    for component in model.components:
        matrix = component.matrix / np.maximum(
            component.matrix.sum(axis=1, keepdims=True),
            np.finfo(np.float64).eps,
        )
        log_matrix = np.zeros_like(matrix)
        np.log2(matrix, out=log_matrix, where=matrix > 0)
        entropy = -np.sum(matrix * log_matrix, axis=1)
        components.append(float(np.sum(2.0 - entropy)))
    return float(max(components))


def codon_pair_log_preference(
    sequence: str,
    codon_starts: tuple[int, ...],
    pair_counts: Counter[tuple[str, str]],
) -> float:
    codons = [sequence[start : start + 3] for start in codon_starts]
    denominator = float(sum(pair_counts.values()) + 61 * 61)
    return float(
        sum(
            np.log((pair_counts[(left, right)] + 1.0) / denominator)
            for left, right in zip(codons, codons[1:])
        )
    )


def no_new_splice_validator(clean: str, start: int, end: int):
    clean_positions = {
        position
        for position in range(start, end - 1)
        if clean[position : position + 2] in {"AG", "GT"}
    }

    def validate(candidate: str) -> bool:
        candidate_positions = {
            position
            for position in range(start, end - 1)
            if candidate[position : position + 2] in {"AG", "GT"}
        }
        return candidate_positions <= clean_positions

    return validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--basic-gtf", required=True)
    parser.add_argument("--comprehensive-gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--prior-results", required=True)
    parser.add_argument("--rbns-psams", required=True)
    parser.add_argument("--n", type=int, default=N_DISCOVERY)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.n != N_DISCOVERY:
        raise ValueError(f"discovery is frozen at n={N_DISCOVERY}")
    preflight_path = Path(args.preflight).expanduser().resolve()
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("biological-feature preflight did not pass")
    torch.set_grad_enabled(False)
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    base_model = Path(args.base_model).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
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

    # Everything through sequence construction and feature-baseline fitting is
    # annotation/sequence-only and completes before the checkpoint is loaded.
    prior = collect_prior_gene_exclusion(prior_results)
    annotation_index = load_annotation_safety_index(comprehensive_gtf)
    cds_intervals = load_cds_intervals(comprehensive_gtf)
    examples, annotation_audits, discovery_counts = load_fresh_cds_cohort(
        str(basic_gtf),
        str(fasta),
        annotation_index=annotation_index,
        cds_intervals=cds_intervals,
        analysis_partition="discovery",
        n=N_DISCOVERY,
        tag="gtex-native-biological-features-discovery-v1",
        excluded_gene_ids=prior.gene_ids,
        discovery_percent=50,
        require_folding_context=True,
    )
    discovery_genes = {example.gene_id for example in examples}
    confirmation_examples, confirmation_audits, confirmation_counts = (
        load_fresh_cds_cohort(
            str(basic_gtf),
            str(fasta),
            annotation_index=annotation_index,
            cds_intervals=cds_intervals,
            analysis_partition="confirmation",
            n=N_CONFIRMATION,
            tag="gtex-native-biological-features-confirmation-v1",
            excluded_gene_ids=prior.gene_ids,
            additional_excluded_gene_ids=discovery_genes,
            discovery_percent=50,
            require_folding_context=True,
        )
    )
    confirmation_genes = {example.gene_id for example in confirmation_examples}
    baseline_examples, _, baseline_counts = load_fresh_cds_cohort(
        str(basic_gtf),
        str(fasta),
        annotation_index=annotation_index,
        cds_intervals=cds_intervals,
        analysis_partition="discovery",
        n=N_BASELINE,
        tag="gtex-native-biological-features-baseline-v1",
        excluded_gene_ids=prior.gene_ids,
        additional_excluded_gene_ids=discovery_genes | confirmation_genes,
        discovery_percent=50,
        require_folding_context=True,
    )
    if discovery_genes & confirmation_genes:
        raise RuntimeError("discovery and confirmation genes overlap")
    exon_record, exon_score = fit_exon_identity(
        baseline_examples,
        annotation_index,
    )
    codon_counts: Counter[str] = Counter()
    codon_pair_counts: Counter[tuple[str, str]] = Counter()
    for example in baseline_examples:
        for window, anchor in (
            (DONOR_WINDOW, int(example.donor_pos)),
            (ACCEPTOR_WINDOW, int(example.next_acceptor_pos)),
        ):
            starts = complete_codon_starts(
                example,
                window,
                cds_intervals,
                anchor=anchor,
            )
            codons = [example.sequence[start : start + 3] for start in starts]
            codon_counts.update(codons)
            codon_pair_counts.update(zip(codons, codons[1:]))
    psams = load_rbns_psam_directory(rbns_root)

    specs = {
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
    feature_names = tuple(
        [f"psam:{name}" for name in sorted(psams)]
        + ["codon_preference", "exon_identity", "rna_accessibility", "mfe"]
    )
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    macro_sequences: dict[str, dict[str, list[str]]] = {}
    clean_features: dict[str, list[dict[str, float]]] = {}
    for endpoint, spec in specs.items():
        print(f"constructing {endpoint} synonymous saturation package", flush=True)
        variant_rows[endpoint] = []
        macro_sequences[endpoint] = {
            "target_shuffle": [],
            "sham_shuffle": [],
            "max_recode_draw0": [],
            "max_recode_draw1": [],
        }
        clean_features[endpoint] = []
        for example_index, (example, anchor) in enumerate(
            zip(examples, spec["anchors"])
        ):
            starts = complete_codon_starts(
                example,
                spec["window"],
                cds_intervals,
                anchor=anchor,
            )
            clean_values = descriptor_values(
                example.sequence,
                example,
                window=spec["window"],
                anchor=anchor,
                codon_starts=starts,
                codon_counts=codon_counts,
                psams=psams,
                exon_score=exon_score,
            )
            clean_features[endpoint].append(clean_values)
            target, _ = shuffle_relative_window(
                example.sequence,
                anchor=anchor,
                window=spec["window"],
                key=(example.example_id, endpoint, "biological-discovery-target"),
                draw=0,
            )
            sham, _ = shuffle_relative_window(
                example.sequence,
                anchor=anchor,
                window=spec["sham"],
                key=(example.example_id, endpoint, "biological-discovery-sham"),
                draw=0,
            )
            macro_sequences[endpoint]["target_shuffle"].append(target)
            macro_sequences[endpoint]["sham_shuffle"].append(sham)
            for draw in (0, 1):
                recoded, _ = maximal_synonymous_recode(
                    example,
                    spec["window"],
                    cds_intervals,
                    anchor=anchor,
                    draw=draw,
                    forbid_new_splice_dinucleotides=True,
                )
                macro_sequences[endpoint][f"max_recode_draw{draw}"].append(recoded)
            for variant in synonymous_saturation_variants(
                example,
                spec["window"],
                cds_intervals,
                anchor=anchor,
            ):
                changed_values = descriptor_values(
                    variant.sequence,
                    example,
                    window=spec["window"],
                    anchor=anchor,
                    codon_starts=starts,
                    codon_counts=codon_counts,
                    psams=psams,
                    exon_score=exon_score,
                )
                start, end = spec["window"].absolute(anchor, len(example.sequence))
                relative_codon = (variant.codon_start - start) / 29.0
                gc_clean = sum(base in "GC" for base in example.sequence[start:end])
                gc_changed = sum(base in "GC" for base in variant.sequence[start:end])
                variant_rows[endpoint].append(
                    {
                        "_sequence": variant.sequence,
                        "example_index": example_index,
                        "example_id": example.example_id,
                        "gene_id": example.gene_id,
                        "codon_start": variant.codon_start,
                        "relative_codon": float(relative_codon),
                        "original_codon": variant.original_codon,
                        "alternate_codon": variant.alternate_codon,
                        "changed_positions": list(variant.changed_positions),
                        "hamming": variant.hamming,
                        "gc_delta": float(gc_clean - gc_changed),
                        "descriptor_delta": {
                            name: float(clean_values[name] - changed_values[name])
                            for name in feature_names
                        },
                    }
                )

    print("sequence package complete; opening seed-2020 classifier outcomes", flush=True)
    model, tokenizer, checkpoint_report = load_gtex_token_classifier(
        base_model,
        checkpoint,
        device=device,
    )
    if checkpoint_report.seed != SEED:
        raise RuntimeError(
            f"discovery requires seed {SEED}, loaded {checkpoint_report.seed}"
        )
    task = NativeClassifierTask(
        model,
        tokenizer,
        device=device,
        batch_size=args.batch,
    )
    clean_sequences = [example.sequence for example in examples]
    clean_logits = task.logits(clean_sequences)
    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    discovery_summaries: dict[str, Any] = {}
    regression_records: dict[str, Any] = {}
    selected: dict[str, list[dict[str, Any]]] = {}
    for endpoint_index, (endpoint, spec) in enumerate(specs.items()):
        print(
            f"scoring {endpoint}: {len(variant_rows[endpoint])} single-codon variants",
            flush=True,
        )
        clean_margin = task.score(
            clean_logits,
            spec["targets"],
            label=int(spec["label"]),
        ).margin
        condition_summaries = {
            "clean_margin": grouped_bootstrap(
                clean_margin,
                groups,
                n_boot=N_BOOT,
                seed=81_000 + endpoint_index,
            )
        }
        for condition_index, (condition, sequences) in enumerate(
            macro_sequences[endpoint].items()
        ):
            logits = task.logits(sequences)
            margin = task.score(
                logits,
                spec["targets"],
                label=int(spec["label"]),
            ).margin
            damage = clean_margin - margin
            condition_summaries[condition] = grouped_bootstrap(
                damage,
                groups,
                n_boot=N_BOOT,
                seed=82_000 + endpoint_index * 100 + condition_index,
            )
            condition_summaries[condition]["per_example"] = damage.tolist()
        rows = variant_rows[endpoint]
        sequences = [row["_sequence"] for row in rows]
        positions = [spec["targets"][int(row["example_index"])] for row in rows]
        logits = task.logits(sequences)
        variant_margin = task.score(logits, positions, label=int(spec["label"])).margin
        outcome = np.asarray(
            [
                clean_margin[int(row["example_index"])] - variant_margin[index]
                for index, row in enumerate(rows)
            ],
            dtype=np.float64,
        )
        for row, value in zip(rows, outcome):
            row["margin_damage"] = float(value)
        descriptor_x = np.asarray(
            [
                [float(row["descriptor_delta"][name]) for name in feature_names]
                for row in rows
            ],
            dtype=np.float64,
        )
        nuisance_names = ("hamming", "gc_delta", "relative_codon")
        nuisance_x = np.asarray(
            [[float(row[name]) for name in nuisance_names] for row in rows],
            dtype=np.float64,
        )
        full_x = np.concatenate((nuisance_x, descriptor_x), axis=1)
        row_groups = np.asarray([row["gene_id"] for row in rows], dtype=object)
        full_fit = fit_elastic_grid(full_x, outcome, row_groups)
        nuisance_fit = fit_elastic_grid(nuisance_x, outcome, row_groups)
        coefficient = full_fit["coefficient"][:, len(nuisance_names) :]
        signs = np.sign(coefficient)
        stable_count = np.maximum(
            np.sum(signs > 0, axis=0),
            np.sum(signs < 0, axis=0),
        )
        median_abs = np.median(np.abs(coefficient), axis=0)
        improvement_folds = [
            nuisance_fit["folds"][index]["mse"] - full_fit["folds"][index]["mse"]
            for index in range(5)
        ]
        eligible = [
            index
            for index, name in enumerate(feature_names)
            if name.startswith("psam:")
            and stable_count[index] >= 4
            and median_abs[index] > 1e-8
            and sum(value > 0 for value in improvement_folds) >= 4
        ]
        ranked = sorted(
            eligible,
            key=lambda index: (-median_abs[index], feature_names[index]),
        )
        chosen: list[dict[str, Any]] = []
        for index in ranked:
            profile = descriptor_x[:, index]
            if any(
                abs(float(np.corrcoef(profile, descriptor_x[:, row["index"]])[0, 1]))
                > 0.90
                for row in chosen
                if np.std(profile) > 0
                and np.std(descriptor_x[:, row["index"]]) > 0
            ):
                continue
            nonzero = coefficient[:, index][np.abs(coefficient[:, index]) > 1e-8]
            sign = int(np.sign(np.median(nonzero))) if nonzero.size else 0
            chosen.append(
                {
                    "index": index,
                    "descriptor": feature_names[index],
                    "rbp": feature_names[index].removeprefix("psam:"),
                    "coefficient_sign": sign,
                    "median_abs_standardized_coefficient": float(median_abs[index]),
                    "stable_fold_count": int(stable_count[index]),
                }
            )
            if len(chosen) == 2:
                break
        selected[endpoint] = chosen
        regression_records[endpoint] = {
            "feature_names": list(feature_names),
            "nuisance_names": list(nuisance_names),
            "n_rows": len(rows),
            "n_genes": len(np.unique(row_groups)),
            "full_model": {
                "folds": full_fit["folds"],
                "mse": full_fit["mse"],
            },
            "nuisance_model": {
                "folds": nuisance_fit["folds"],
                "mse": nuisance_fit["mse"],
            },
            "mse_improvement": float(nuisance_fit["mse"] - full_fit["mse"]),
            "fold_mse_improvement": improvement_folds,
            "coefficient_by_fold": coefficient.tolist(),
            "stable_fold_count": stable_count.tolist(),
            "median_abs_standardized_coefficient": median_abs.tolist(),
            "eligible_psam_indices": eligible,
            "selected": chosen,
        }
        discovery_summaries[endpoint] = condition_summaries
    del task, model, tokenizer, clean_logits
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # Match three sequence-only PSAM controls and freeze high/low synonymous
    # constructions without opening any confirmation model.
    construction_freeze: dict[str, Any] = {}
    score_acceptor, score_donor = maxent_scorers()
    for endpoint, spec in specs.items():
        clean_profiles = {
            name: np.asarray(
                [values[f"psam:{name}"] for values in clean_features[endpoint]]
            )
            for name in sorted(psams)
        }
        metadata = {
            name: {
                "width": max(component.width for component in model.components),
                "information": psam_information(model),
                "natural_variance": float(np.var(clean_profiles[name])),
            }
            for name, model in psams.items()
        }
        construction_freeze[endpoint] = {}
        for candidate in selected[endpoint]:
            name = candidate["rbp"]
            target_meta = metadata[name]
            control_names = sorted(
                (other for other in psams if other != name),
                key=lambda other: (
                    abs(metadata[other]["width"] - target_meta["width"]),
                    abs(metadata[other]["information"] - target_meta["information"]),
                    abs(
                        np.log1p(metadata[other]["natural_variance"])
                        - np.log1p(target_meta["natural_variance"])
                    ),
                    other,
                ),
            )[:3]
            family_names = [name, *control_names]
            family_rows = {}
            for family_name in family_names:
                model = psams[family_name]
                rows = []
                for example, anchor in zip(examples, spec["anchors"]):
                    start, end = spec["window"].absolute(
                        anchor,
                        len(example.sequence),
                    )
                    objective = lambda sequence, m=model, s=start, e=end: psam_region_score(
                        sequence,
                        m,
                        start=s,
                        end=e,
                    )
                    validator = no_new_splice_validator(
                        example.sequence,
                        start,
                        end,
                    )
                    try:
                        low, low_audit = beam_synonymous_optimize(
                            example,
                            spec["window"],
                            cds_intervals,
                            anchor=anchor,
                            objective=objective,
                            maximize=False,
                            min_changed_bases=1,
                            max_changed_bases=4,
                            beam_size=256,
                            validator=validator,
                        )
                        high, high_audit = beam_synonymous_optimize(
                            example,
                            spec["window"],
                            cds_intervals,
                            anchor=anchor,
                            objective=objective,
                            maximize=True,
                            min_changed_bases=1,
                            max_changed_bases=4,
                            beam_size=256,
                            validator=validator,
                        )
                        success = (
                            high != low
                            and objective(high) > objective(low)
                            and validator(high)
                            and validator(low)
                        )
                        clean_to_low_3 = compare_landscapes(
                            example.sequence,
                            low,
                            start,
                            end,
                            window_length=23,
                            scorer=score_acceptor,
                        )
                        clean_to_high_3 = compare_landscapes(
                            example.sequence,
                            high,
                            start,
                            end,
                            window_length=23,
                            scorer=score_acceptor,
                        )
                        clean_to_low_5 = compare_landscapes(
                            example.sequence,
                            low,
                            start,
                            end,
                            window_length=9,
                            scorer=score_donor,
                        )
                        clean_to_high_5 = compare_landscapes(
                            example.sequence,
                            high,
                            start,
                            end,
                            window_length=9,
                            scorer=score_donor,
                        )
                        maxent_pass = all(
                            abs(change.delta_max) <= 1.0
                            and not change.new_competitive_site
                            for change in (
                                clean_to_low_3,
                                clean_to_high_3,
                                clean_to_low_5,
                                clean_to_high_5,
                            )
                        )
                        success = success and maxent_pass
                        starts = complete_codon_starts(
                            example,
                            spec["window"],
                            cds_intervals,
                            anchor=anchor,
                        )
                        low_values = descriptor_values(
                            low,
                            example,
                            window=spec["window"],
                            anchor=anchor,
                            codon_starts=starts,
                            codon_counts=codon_counts,
                            psams=psams,
                            exon_score=exon_score,
                        )
                        high_values = descriptor_values(
                            high,
                            example,
                            window=spec["window"],
                            anchor=anchor,
                            codon_starts=starts,
                            codon_counts=codon_counts,
                            psams=psams,
                            exon_score=exon_score,
                        )
                        off_target_l1 = float(
                            sum(
                                abs(
                                    high_values[f"psam:{other}"]
                                    - low_values[f"psam:{other}"]
                                )
                                for other in psams
                                if other != family_name
                            )
                        )
                        nuisance = {
                            "off_target_psam_l1": off_target_l1,
                            "codon_preference_high_minus_low": float(
                                high_values["codon_preference"]
                                - low_values["codon_preference"]
                            ),
                            "exon_identity_high_minus_low": float(
                                high_values["exon_identity"]
                                - low_values["exon_identity"]
                            ),
                            "rna_accessibility_high_minus_low": float(
                                high_values["rna_accessibility"]
                                - low_values["rna_accessibility"]
                            ),
                            "mfe_high_minus_low": float(
                                high_values["mfe"] - low_values["mfe"]
                            ),
                            "gc3_high_minus_low": float(
                                sum(high[position] in "GC" for position in starts)
                                - sum(low[position] in "GC" for position in starts)
                            ),
                            "codon_pair_high_minus_low": (
                                codon_pair_log_preference(
                                    high,
                                    starts,
                                    codon_pair_counts,
                                )
                                - codon_pair_log_preference(
                                    low,
                                    starts,
                                    codon_pair_counts,
                                )
                            ),
                            "canonical_ag_gt_high_minus_low": int(
                                sum(
                                    high[position : position + 2] in {"AG", "GT"}
                                    for position in range(start, end - 1)
                                )
                                - sum(
                                    low[position : position + 2] in {"AG", "GT"}
                                    for position in range(start, end - 1)
                                )
                            ),
                            "maxent3_clean_to_low_delta_max": float(
                                clean_to_low_3.delta_max
                            ),
                            "maxent3_clean_to_high_delta_max": float(
                                clean_to_high_3.delta_max
                            ),
                            "maxent5_clean_to_low_delta_max": float(
                                clean_to_low_5.delta_max
                            ),
                            "maxent5_clean_to_high_delta_max": float(
                                clean_to_high_5.delta_max
                            ),
                            "maxent_band_pass": bool(maxent_pass),
                        }
                    except RuntimeError:
                        low = high = example.sequence
                        low_audit = high_audit = {}
                        success = False
                        nuisance = {"construction_failed": True}
                    rows.append(
                        {
                            "example_id": example.example_id,
                            "gene_id": example.gene_id,
                            "success": success,
                            "low_sha256": sha256_text(low),
                            "high_sha256": sha256_text(high),
                            "low": low_audit,
                            "high": high_audit,
                            "score_low": float(objective(low)),
                            "score_high": float(objective(high)),
                            "nuisance": nuisance,
                        }
                    )
                family_rows[family_name] = {
                    "role": "candidate" if family_name == name else "matched_control",
                    "metadata": metadata[family_name],
                    "success_count": sum(row["success"] for row in rows),
                    "rows": rows,
                }
            construction_freeze[endpoint][name] = {
                "expected_high_minus_low_margin_sign": candidate["coefficient_sign"],
                "matched_controls": control_names,
                "families": family_rows,
            }

    serializable_rows = {}
    for endpoint, rows in variant_rows.items():
        serializable_rows[endpoint] = [
            {key: value for key, value in row.items() if key != "_sequence"}
            for row in rows
        ]
    result = {
        "experiment": "gtex_native_biological_features_discovery_v1",
        "status": "COMPLETE",
        "protocol": str(protocol.resolve()),
        "preflight": str(preflight_path),
        "device": device,
        "cohort": {
            "discovery": {
                "n": len(examples),
                "n_genes": len(discovery_genes),
                "counts": discovery_counts,
                "examples": [
                    {
                        "example_id": example.example_id,
                        "gene_id": example.gene_id,
                        "chrom": example.chrom,
                        "donor_pos": int(example.donor_pos),
                        "next_acceptor_pos": int(example.next_acceptor_pos),
                        "annotation": {
                            endpoint: asdict(audit)
                            for endpoint, audit in annotation_audits[
                                example.example_id
                            ].items()
                        },
                    }
                    for example in examples
                ],
            },
            "confirmation_freeze": {
                "n": len(confirmation_examples),
                "n_genes": len(confirmation_genes),
                "counts": confirmation_counts,
                "examples": [
                    {
                        "example_id": example.example_id,
                        "gene_id": example.gene_id,
                        "chrom": example.chrom,
                        "annotation": {
                            endpoint: asdict(audit)
                            for endpoint, audit in confirmation_audits[
                                example.example_id
                            ].items()
                        },
                    }
                    for example in confirmation_examples
                ],
            },
            "baseline": {
                "n": len(baseline_examples),
                "counts": baseline_counts,
                "gene_ids": [example.gene_id for example in baseline_examples],
            },
            "prior_gene_count": len(prior.gene_ids),
            "prior_gene_source_aggregate_sha256": prior.aggregate_sha256,
        },
        "sequence_baselines": {
            "codon_counts": dict(sorted(codon_counts.items())),
            "codon_pair_counts": {
                f"{left}:{right}": count
                for (left, right), count in sorted(codon_pair_counts.items())
            },
            "exon_identity": exon_record,
            "psam_count": len(psams),
        },
        "macro_condition_summaries": discovery_summaries,
        "regression": regression_records,
        "candidate_freeze": selected,
        "construction_freeze": construction_freeze,
        "single_codon_rows": serializable_rows,
        "checkpoint_report": checkpoint_report.to_dict(),
        "evidence_boundary": (
            "Discovery coefficients and sequence-only high/low constructions are "
            "candidate generators; no RBP identity or causal feature is established."
        ),
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    manifest = ExperimentManifest.create(
        "gtex-native-biological-features-discovery-v1",
        sys.argv,
        repo=REPO,
        status="COMPLETE",
        config={
            "n_discovery": N_DISCOVERY,
            "n_confirmation_frozen": N_CONFIRMATION,
            "n_baseline": N_BASELINE,
            "seed": SEED,
            "n_boot": N_BOOT,
            "elastic_grid": ELASTIC_GRID,
        },
        inputs=[
            file_record(path)
            for path in (
                protocol,
                preflight_path,
                basic_gtf,
                comprehensive_gtf,
                fasta,
                checkpoint,
            )
        ],
        outputs=[file_record(output)],
        notes=[
            "Confirmation cohort identities were frozen before any confirmation outcome.",
            "RBP high/low construction did not access classifier outcomes.",
            "RNAplfold settings: W=150, L=100, u=7 on 201 nt.",
        ],
    )
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidates": selected,
                "macro": {
                    endpoint: {
                        name: row["estimate"]
                        for name, row in rows.items()
                        if name != "clean_margin"
                    }
                    for endpoint, rows in discovery_summaries.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
