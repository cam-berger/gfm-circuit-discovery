#!/usr/bin/env python
"""Run the phase-gated DNABERT-2 excess-reconstruction screen."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from gfmcd.adapters.dnabert2 import (
    DNABERT2_REVISION,
    Dnabert2Adapter,
    load_dnabert2_mlm,
)
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.analysis import (
    SiteReconstruction,
    compute_excess_reconstruction,
    summarize_by_site_class,
)
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.baselines import NucleotideNGramBaseline
from gfmcd.reconstruction_atlas.gates import (
    frozen_config_sha256,
    load_preflight_report,
)
from gfmcd.reconstruction_atlas.schema import (
    SiteReference,
    load_collection,
    load_matched_sets,
)
from gfmcd.reconstruction_atlas.scoring import ReconstructionScorer, ScoringRequest
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    holm_adjust,
    paired_cluster_sign_permutation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--baseline-training", required=True)
    parser.add_argument("--preflight-config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--split", choices=("development",), default="development")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preflight = load_preflight_report(args.preflight_report)
    preflight.assert_qualified(
        expected_config_sha256=frozen_config_sha256(args.preflight_config),
        expected_checkpoint_revision=DNABERT2_REVISION,
        expected_input_sha256={
            "collection": sha256_file(args.collection),
            "matches": sha256_file(args.matches),
            "baseline_training": sha256_file(args.baseline_training),
        },
    )
    collection = load_collection(args.collection)
    background = load_collection(args.baseline_training)
    selected = {
        record.example_id: record for record in collection if record.split == args.split
    }
    if not selected:
        raise SystemExit(f"collection has no {args.split} examples")
    if {record.homology_cluster for record in selected.values()} & {
        record.homology_cluster for record in background
    }:
        raise SystemExit("baseline-training and target homology clusters overlap")
    matches = [
        matched
        for matched in load_matched_sets(args.matches)
        if matched.target.example_id in selected
    ]
    if not matches:
        raise SystemExit("no matched sets remain on the requested split")

    bundle = load_dnabert2_mlm(device=args.device)
    adapter = Dnabert2Adapter(bundle)
    scorer = ReconstructionScorer(adapter)
    references = _unique_references(matches)
    requests = []
    for reference in references.values():
        example = selected.get(reference.example_id)
        if example is None:
            raise SystemExit(
                f"control {reference.example_id} is not in the selected split collection"
            )
        if reference.interval.end > len(example.sequence):
            raise SystemExit(
                f"site {_request_id(reference)} falls outside its sequence"
            )
        if reference.element_id is not None:
            element = example.element(reference.element_id)
            if (
                element.interval != reference.interval
                or element.site_class != reference.site_class
            ):
                raise SystemExit(
                    f"site {_request_id(reference)} disagrees with its annotated element"
                )
        requests.append(
            ScoringRequest(
                request_id=_request_id(reference),
                sequence=example.sequence,
                site=reference.interval,
            )
        )
    model_scores = {
        score.request_id: score
        for score in scorer.score(requests, batch_size=args.batch_size)
    }
    baselines = [
        NucleotideNGramBaseline(order=order).fit(
            record.sequence for record in background
        )
        for order in (0, 1, 2)
    ]
    site_scores = {}
    raw_rows = []
    for key, reference in references.items():
        example = selected[reference.example_id]
        score = model_scores[_request_id(reference)]
        covered = example.sequence[
            score.mask_plan.covered.start : score.mask_plan.covered.end
        ]
        baseline_scores = {
            f"ngram_order_{model.order}": model.log_probability_per_nt(covered)
            for model in baselines
        }
        site_scores[key] = SiteReconstruction(
            site=reference,
            score_per_covered_nt=score.true_token_log_probability_per_covered_nt,
            homology_cluster=example.homology_cluster,
            baseline_scores_per_nt=baseline_scores,
        )
        raw_rows.append(
            {
                **score.to_dict(),
                "site": reference.to_dict(),
                "homology_cluster": example.homology_cluster,
                "baseline_scores_per_nt": baseline_scores,
            }
        )
    excess = compute_excess_reconstruction(matches, site_scores)
    inference = _inference(
        excess,
        permutations=args.permutations,
        seed=args.seed,
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "split": args.split,
        "preflight_report_id": preflight.report_id,
        "preflight_decision": preflight.decision,
        "n_matched_targets": len(matches),
        "raw_scores": raw_rows,
        "excess_reconstruction": [row.to_dict() for row in excess],
        "site_class_summary": summarize_by_site_class(excess),
        "inference": inference,
        "baseline_models": [model.identity() for model in baselines],
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_atlas_behavioral_screen",
        status="DEVELOPMENT",
        model_identity=adapter.identity(),
        input_paths=(
            args.collection,
            args.matches,
            args.baseline_training,
            args.preflight_config,
            args.preflight_report,
        ),
        parameters={
            "split": args.split,
            "batch_size": args.batch_size,
            "permutations": args.permutations,
            "normalization": "sum_true_token_log_probability / BPE_covered_nucleotides",
            "masking": "simultaneous_minimal_BPE_cover",
            "multiplicity": "Holm across complete site-class x contrast family",
        },
        seeds={"inference": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )


def _unique_references(matches: list) -> dict[tuple[str, int, int], SiteReference]:
    references: dict[tuple[str, int, int], SiteReference] = {}
    for matched in matches:
        all_sites = [matched.target] + [control.site for control in matched.controls]
        for site in all_sites:
            key = _key(site)
            previous = references.get(key)
            if previous is not None and previous.site_class != site.site_class:
                raise ValueError(f"site {key} is assigned conflicting site classes")
            references[key] = site
    return references


def _inference(rows: list, *, permutations: int, seed: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.target.site_class].append(row)
    tests = {}
    raw_pvalues = {}
    for site_class, class_rows in sorted(grouped.items()):
        contrast_names = sorted(
            set(class_rows[0].empirical_advantage_by_family)
            | set(class_rows[0].baseline_advantage)
        )
        for contrast in contrast_names:
            values = [
                (
                    row.empirical_advantage_by_family[contrast]
                    if contrast in row.empirical_advantage_by_family
                    else row.baseline_advantage[contrast]
                )
                for row in class_rows
            ]
            clusters = [row.homology_cluster for row in class_rows]
            observed, null = paired_cluster_sign_permutation(
                values,
                clusters,
                permutations=permutations,
                seed=seed,
            )
            test_id = f"{site_class}|{contrast}"
            pvalue = add_one_permutation_pvalue(
                observed,
                null,
                alternative="greater",
            )
            tests[test_id] = {
                "site_class": site_class,
                "contrast": contrast,
                "observed_cluster_mean": observed,
                "permutations": len(null),
                "p_add_one": pvalue,
            }
            raw_pvalues[test_id] = pvalue
    adjusted = holm_adjust(raw_pvalues)
    for test_id, value in adjusted.items():
        tests[test_id]["p_holm_full_family"] = value
    return {
        "family_size": len(tests),
        "tests": tests,
    }


def _request_id(site: SiteReference) -> str:
    return f"{site.example_id}:{site.interval.start}-{site.interval.end}"


def _key(site: SiteReference) -> tuple[str, int, int]:
    return (site.example_id, site.interval.start, site.interval.end)


if __name__ == "__main__":
    main()
