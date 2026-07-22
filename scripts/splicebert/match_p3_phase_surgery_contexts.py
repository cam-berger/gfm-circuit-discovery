#!/usr/bin/env python
"""Outcome-blind strand-exact matching of CDS and non-CDS phase-surgery cohorts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import DEFAULT_FASTA, DEFAULT_GTF, _dataset  # noqa: E402
from gfmcd.splicing.p3_cohort import sequence_sha256  # noqa: E402


COVARIATE_NAMES = (
    "log_exon_length",
    "donor_partner_distance",
    "donor_position",
    "acceptor_position",
    "far_block_end_to_donor",
    "near_block_end_to_donor",
    "target_block_gc",
    "intron_block_gc",
)


def block_gc(sequence: str, blocks: list[list[int]]) -> float:
    bases = "".join(sequence[int(start):int(end)] for start, end in blocks)
    if not bases:
        raise ValueError("matching block sequence is empty")
    return float(sum(base in "GC" for base in bases) / len(bases))


def record_covariates(record: dict, sequence: str, *, target_context: str) -> np.ndarray:
    target = sorted(record["blocks"][target_context])
    if len(target) != 2:
        raise ValueError("context matching requires exactly two target blocks")
    distances = sorted(
        (float(record["donor_pos"] - int(block[1])) for block in target), reverse=True
    )
    return np.asarray([
        math.log1p(float(record["exon_length"])),
        float(record["donor_partner_distance"]),
        float(record["donor_pos"]),
        float(record["acceptor_pos"]),
        distances[0],
        distances[1],
        block_gc(sequence, target),
        block_gc(sequence, record["blocks"]["upstream_intron"]),
    ])


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    pooled = np.sqrt((left.var(axis=0, ddof=1) + right.var(axis=0, ddof=1)) / 2)
    return np.divide(
        left.mean(axis=0) - right.mean(axis=0),
        pooled,
        out=np.zeros(left.shape[1]),
        where=pooled > 0,
    )


def exact_strand_match(cds_rows: list[dict], non_rows: list[dict]):
    pairs = []
    for strand in ("+", "-"):
        left = [row for row in cds_rows if row["strand"] == strand]
        right = [row for row in non_rows if row["strand"] == strand]
        if not left or not right:
            continue
        combined = np.vstack([row["covariates"] for row in left + right])
        scale = combined.std(axis=0, ddof=1)
        scale[scale <= 0] = 1.0
        left_values = np.vstack([row["covariates"] for row in left]) / scale
        right_values = np.vstack([row["covariates"] for row in right]) / scale
        cost = np.sum((left_values[:, None, :] - right_values[None, :, :]) ** 2, axis=2)
        left_index, right_index = linear_sum_assignment(cost)
        for li, ri in zip(left_index, right_index):
            pairs.append((left[li], right[ri], float(math.sqrt(cost[li, ri]))))
    return sorted(pairs, key=lambda pair: (pair[0]["gene_id"], pair[1]["gene_id"]))


def within_gene_then_strand_match(cds_rows: list[dict], non_rows: list[dict]):
    """Use same-gene context controls first, then match the remaining genes by strand."""

    cds_by_gene = {row["gene_id"]: row for row in cds_rows}
    non_by_gene = {row["gene_id"]: row for row in non_rows}
    shared = sorted(set(cds_by_gene) & set(non_by_gene))
    combined = np.vstack([row["covariates"] for row in cds_rows + non_rows])
    scale = combined.std(axis=0, ddof=1)
    scale[scale <= 0] = 1.0
    direct = []
    for gene in shared:
        left, right = cds_by_gene[gene], non_by_gene[gene]
        distance = float(np.linalg.norm((left["covariates"] - right["covariates"]) / scale))
        direct.append((left, right, distance))
    remaining_left = [row for row in cds_rows if row["gene_id"] not in shared]
    remaining_right = [row for row in non_rows if row["gene_id"] not in shared]
    return sorted(
        direct + exact_strand_match(remaining_left, remaining_right),
        key=lambda pair: (pair[0]["gene_id"], pair[1]["gene_id"]),
    )


def prune_pairs_for_balance(
    pairs: list[tuple[dict, dict, float]], *, max_abs_smd: float, min_pairs: int
):
    """Deterministically retain the largest greedy subset meeting a balance ceiling."""

    if not (0 < max_abs_smd < 1) or min_pairs < 3 or len(pairs) < min_pairs:
        raise ValueError("invalid balance-pruning specification")
    retained = list(range(len(pairs)))

    def balance(indices):
        left = np.vstack([pairs[index][0]["covariates"] for index in indices])
        right = np.vstack([pairs[index][1]["covariates"] for index in indices])
        return standardized_mean_difference(left, right)

    while len(retained) > min_pairs:
        current = balance(retained)
        if float(np.max(np.abs(current))) <= max_abs_smd:
            break
        candidates = []
        for dropped in retained:
            keep = [index for index in retained if index != dropped]
            candidate = balance(keep)
            candidates.append((
                float(np.max(np.abs(candidate))),
                float(np.sum(candidate * candidate)),
                pairs[dropped][0]["gene_id"],
                pairs[dropped][1]["gene_id"],
                dropped,
            ))
        retained.remove(min(candidates)[-1])
    final = balance(retained)
    if float(np.max(np.abs(final))) > max_abs_smd:
        raise RuntimeError(
            f"cannot reach max |SMD| <= {max_abs_smd} while retaining {min_pairs} pairs"
        )
    return [pairs[index] for index in retained], [
        pairs[index] for index in range(len(pairs)) if index not in retained
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cds-cohort", required=True)
    parser.add_argument("--non-cds-cohort", required=True)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--max-abs-smd", type=float, default=0.25)
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument(
        "--out", default="results/phase-lattice/p3_phase_surgery_context_match.json"
    )
    args = parser.parse_args()
    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite P3 context match: {out}")
    cds = json.loads(Path(args.cds_cohort).read_text())
    non = json.loads(Path(args.non_cds_cohort).read_text())
    for package in (cds, non):
        if package.get("schema") != "gfmcd.p3_phase_surgery.cohort.v1":
            raise ValueError("wrong phase-surgery cohort schema")
        if package.get("model_outcomes_read") is not False:
            raise RuntimeError("context matching requires outcome-blind cohorts")
    if set(cds.get("included_roles", ())) != set(non.get("included_roles", ())):
        raise RuntimeError("context packages must contain the same cohort roles")

    source = {example.example_id: example for example in _dataset(args.gtf, args.fasta)}
    prepared = []
    for package, expected_context in ((cds, "cds"), (non, "non_cds_exon")):
        target_context = str(package["config"].get("target_context", "cds"))
        if target_context != expected_context:
            raise RuntimeError(f"expected target context {expected_context}, got {target_context}")
        rows = []
        for record in package["records"]:
            if not record.get("eligible"):
                continue
            example = source.get(record["example_id"])
            if example is None or sequence_sha256(example.sequence) != record["native_sequence_sha256"]:
                raise RuntimeError(f"source reconstruction failed for {record['example_id']}")
            rows.append({
                **record,
                "covariates": record_covariates(
                    record, example.sequence, target_context=target_context
                ),
            })
        prepared.append(rows)
    cds_rows, non_rows = prepared
    initial_pairs = within_gene_then_strand_match(cds_rows, non_rows)
    if len(initial_pairs) < args.min_pairs:
        raise RuntimeError("fewer outcome-blind context matches than the frozen minimum")
    pairs, dropped_pairs = prune_pairs_for_balance(
        initial_pairs, max_abs_smd=args.max_abs_smd, min_pairs=args.min_pairs
    )
    left_all = np.vstack([row["covariates"] for row in cds_rows])
    right_all = np.vstack([row["covariates"] for row in non_rows])
    left_matched = np.vstack([pair[0]["covariates"] for pair in pairs])
    right_matched = np.vstack([pair[1]["covariates"] for pair in pairs])
    pre_smd = standardized_mean_difference(left_all, right_all)
    post_smd = standardized_mean_difference(left_matched, right_matched)
    result = {
        "schema": "gfmcd.p3_phase_surgery.context_match.v1",
        "status": "OUTCOME_BLIND_STRUCTURAL_MATCH",
        "model_outcomes_read": False,
        "cds_cohort_sha256": sha256_file(args.cds_cohort),
        "non_cds_cohort_sha256": sha256_file(args.non_cds_cohort),
        "roles": list(cds.get("included_roles", ())),
        "method": (
            "optimal one-to-one Euclidean assignment after pooled-SD scaling, exact strand; "
            "deterministic maximum-balance pruning"
        ),
        "balance_spec": {
            "max_abs_smd": args.max_abs_smd,
            "minimum_pairs": args.min_pairs,
        },
        "covariates": list(COVARIATE_NAMES),
        "counts": {
            "cds_pool": len(cds_rows),
            "non_cds_pool": len(non_rows),
            "initial_pairs": len(initial_pairs),
            "matched_pairs": len(pairs),
            "pairs_pruned_for_balance": len(dropped_pairs),
            "same_gene_pairs_retained": sum(
                left["gene_id"] == right["gene_id"] for left, right, _distance in pairs
            ),
        },
        "balance": {
            "pre_match_smd": dict(zip(COVARIATE_NAMES, map(float, pre_smd))),
            "post_assignment_pre_pruning_smd": dict(zip(
                COVARIATE_NAMES,
                map(float, standardized_mean_difference(
                    np.vstack([pair[0]["covariates"] for pair in initial_pairs]),
                    np.vstack([pair[1]["covariates"] for pair in initial_pairs]),
                )),
            )),
            "post_match_smd": dict(zip(COVARIATE_NAMES, map(float, post_smd))),
            "max_abs_post_match_smd": float(np.max(np.abs(post_smd))),
            "distance_quantiles": dict(zip(
                ("min", "p25", "median", "p75", "p95", "max"),
                map(float, np.quantile([pair[2] for pair in pairs], (0, .25, .5, .75, .95, 1))),
            )),
        },
        "pairs": [
            {
                "pair_id": f"pair-{index:04d}",
                "strand": left["strand"],
                "same_gene": left["gene_id"] == right["gene_id"],
                "distance": distance,
                "cds_gene_id": left["gene_id"],
                "cds_example_id": left["example_id"],
                "non_cds_gene_id": right["gene_id"],
                "non_cds_example_id": right["example_id"],
                "cds_covariates": list(map(float, left["covariates"])),
                "non_cds_covariates": list(map(float, right["covariates"])),
            }
            for index, (left, right, distance) in enumerate(pairs)
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-context-match-v1",
        sys.argv,
        repo=REPO,
        status="OUTCOME_BLIND_STRUCTURAL_MATCH",
        config={
            "covariates": COVARIATE_NAMES,
            "exact": ["strand"],
            "max_abs_smd": args.max_abs_smd,
            "minimum_pairs": args.min_pairs,
        },
        notes=["No model was loaded for selection; model outcomes were not read."],
    )
    for path in (
        args.cds_cohort,
        args.non_cds_cohort,
        args.gtf,
        args.fasta,
        REPO / "scripts/splicebert/match_p3_phase_surgery_contexts.py",
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({"counts": result["counts"], "balance": result["balance"]}, indent=2))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
