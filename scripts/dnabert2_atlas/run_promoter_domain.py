#!/usr/bin/env python
"""Run the D12 downstream-domain CONFIRMATION (Task 5).

This opens 8 of the 30 D8 tiles -- {21, 22, 23, 25, 26, 27, 28, 29} -- on D8's own
sealed confirmation split to test the four preregistered endpoints E1, E1b, E2, E3
(E4 is descriptive only and can never qualify: it includes tile 26, whose confirmation
value D8 already opened). Tiles {0-20, 24} MUST remain sealed for future work.

Frozen protocol of record:
  studies/dnabert2_reconstruction_atlas/preregistrations/PREREGISTRATION_D12_DOWNSTREAM_DOMAIN.md
  configs/dnabert2_atlas/promoter_domain_v2.frozen.json

This is the confirmatory analogue of scripts/dnabert2_atlas/freeze_promoter_domain_predictions.py
and mirrors the overall structure of scripts/dnabert2_atlas/run_promoter_source_localization.py
(guarded frozen-input loading, scan, write-raw, compute-endpoints, write-result). Endpoint value
extraction dispatches on endpoint NAME, exactly as freeze_promoter_domain_predictions.py does --
duck-typing on config keys is wrong here: E1b_sharp_edge and E3_upstream_anti_evidence both carry a
"tiles" key with different meanings (a pair vs a single tile), and E4_full_domain carries none.

Four non-negotiable guards, each of which fails loudly and is checked before any expensive
model work:
  1. The preflight report must declare "decision": "QUALIFIED".
  2. The frozen development-predictions file must exist; only `predicted_sign` is ever read from
     it -- never `mean` or `ci95` -- so the confirmation cannot be steered by the development
     effect size.
  3. The tiles this run scores (config["tiles_to_score"]) must be disjoint from
     config["tiles_remaining_sealed"]. Any overlap raises SystemExit naming the leaked tile(s).
  4. The output paths (--out, --raw-out, and their manifests) must not already exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.artifact_guard import guard_output_path
from gfmcd.reconstruction_atlas.artifacts import (
    manifest_path_for,
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_domain import (
    TileRow,
    check_memorials,
    contrast_by_locus,
    load_clusters,
    load_tile_rows,
    paired_tile_difference,
    single_tile_by_locus,
)
from gfmcd.reconstruction_atlas.promoter_head_scan import read_jsonl
from gfmcd.reconstruction_atlas.promoter_source_execution import scan_equal_dose_tiles
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    holm_adjust,
    paired_cluster_sign_permutation,
)

Pair = tuple[str, str, str]

# The two candidate heads this D8/D12 study line patches (L11H2, L11H5). This run only
# ever scores the "mask" operation (see operations_by_tile below), so these heads are
# never actually patched here -- but scan_equal_dose_tiles requires a non-empty
# candidate_heads argument regardless (it unconditionally derives an all-layer-heads
# tuple from candidate_heads[0][0]). Hardcoded identically to
# run_promoter_domain_preflight.py and run_promoter_source_localization.py, which this
# script mirrors, rather than config-driven.
CANDIDATE_HEADS: tuple[tuple[int, int], ...] = ((11, 2), (11, 5))

REQUIRED_TRANSFORMERS_VERSION = "4.44.2"

# add_one_permutation_pvalue's "alternative" strings, keyed by the config's own
# per-endpoint "alternative" label.
ALTERNATIVE_TO_PVALUE_MODE = {"one_sided_positive": "greater", "two_sided": "two-sided"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--development-predictions", required=True)
    parser.add_argument("--annotated-cohort", required=True)
    parser.add_argument("--split", choices=("confirmation",), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=32)
    args = parser.parse_args()

    import transformers

    if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers=={transformers.__version__} but this confirmation run is "
            f"pinned to {REQUIRED_TRANSFORMERS_VERSION}. Run under the pinned "
            "environment, e.g.:\n"
            '  uv run --python 3.11 --with "transformers==4.44.2" '
            '--with "tokenizers>=0.19,<0.20" --with "einops>=0.7" --with "numpy<2" '
            '--with "scipy" --with "pyfaidx>=0.8" --with "torch>=2.2" python <script> ...'
        )

    # ---- Guard 4: refuse to overwrite an existing output path. Checked before any
    # model work -- indeed before even reading the config. ----
    for path in (args.out, args.raw_out):
        guard_output_path(path)
        guard_output_path(manifest_path_for(path))

    # ---- Load the config and check the tile seal BEFORE any cross-file hash
    # validation. This ordering matters: it guarantees that a tampered
    # tiles_to_score fails HERE, naming the leaked tile, rather than on some
    # unrelated later consistency check (e.g. a config-hash mismatch against the
    # preflight report, which a tampered copy would also trip). ----
    config = json.loads(Path(args.config).read_text())
    tiles_to_score = sorted(int(value) for value in config["tiles_to_score"])
    tiles_remaining_sealed = sorted(int(value) for value in config["tiles_remaining_sealed"])
    leaked = sorted(set(tiles_to_score) & set(tiles_remaining_sealed))
    if leaked:
        raise SystemExit(
            f"refusing to run: tiles_to_score includes sealed tile(s) {leaked} "
            "from tiles_remaining_sealed -- sealed tiles must never be scored on "
            "the confirmation split"
        )
    if not tiles_to_score:
        raise SystemExit("refusing to run: config['tiles_to_score'] is empty")

    # ---- Now validate the config is genuinely the frozen protocol of record. ----
    config_hash = frozen_config_sha256(args.config)
    if config.get("status") != "FROZEN":
        raise SystemExit(f"refusing to run: config status is not FROZEN ({args.config})")

    # ---- Guard 1: preflight must be QUALIFIED, against this exact config and
    # checkpoint. ----
    preflight = json.loads(Path(args.preflight_report).read_text())
    if preflight.get("decision") != "QUALIFIED":
        raise SystemExit(
            "refusing to run: preflight report does not declare "
            f'"decision": "QUALIFIED" (found {preflight.get("decision")!r}) -- '
            f"{args.preflight_report}"
        )
    if preflight.get("config_sha256") != config_hash:
        raise SystemExit(
            "refusing to run: preflight config_sha256 does not match "
            f"the target config ({args.config})"
        )
    if preflight.get("checkpoint_revision") != config["checkpoint"]["revision"]:
        raise SystemExit(
            "refusing to run: preflight checkpoint_revision does not match "
            "the frozen config's checkpoint.revision"
        )

    # ---- Guard 2: frozen development predictions must exist. Only predicted_sign is
    # ever extracted -- see _load_predicted_signs's docstring. ----
    predicted_signs = _load_predicted_signs(
        args.development_predictions, config["endpoints"].keys()
    )

    print(
        f"confirmation: scoring tiles {tiles_to_score}; sealed tiles remain "
        f"{tiles_remaining_sealed}",
        flush=True,
    )

    # ---- Cohort (no model needed yet). ----
    all_rows = read_jsonl(args.annotated_cohort)
    confirmation_rows = sorted(
        (row for row in all_rows if row["split"] == args.split),
        key=lambda row: str(row["example_id"]),
    )
    expected_pairs_per_split = int(config["cohort"]["pairs_per_split"])
    if len(confirmation_rows) != expected_pairs_per_split:
        raise SystemExit(
            f"refusing to run: expected {expected_pairs_per_split} {args.split} "
            f"pairs in {args.annotated_cohort}, found {len(confirmation_rows)}"
        )
    if args.limit_pairs is not None:
        if args.limit_pairs < 1:
            raise SystemExit("--limit-pairs must be positive")
        confirmation_rows = confirmation_rows[: args.limit_pairs]
    clusters = load_clusters(args.annotated_cohort)
    pairs: list[Pair] = [
        (
            str(row["example_id"]),
            str(row["target"]["clean_sequence"]),
            str(row["control"]["clean_sequence"]),
        )
        for row in confirmation_rows
    ]

    # ---- Model work begins here -- only after all four guards have passed. ----
    print(f"confirmation: loading prom_300_all on {args.device} ...", flush=True)
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    identity = bundle.identity()
    expected_revision = str(config["checkpoint"]["revision"])
    if identity["revision"] != expected_revision:
        raise SystemExit(
            f"loaded checkpoint revision {identity['revision']!r} does not match "
            f"the frozen config revision {expected_revision!r} -- refusing to proceed"
        )
    print(
        f"confirmation: checkpoint OK (revision={identity['revision']}, "
        f"env={identity['env']})",
        flush=True,
    )

    geometry = config["tile_geometry"]
    progress = lambda message: print(f"confirmation: {message}", flush=True)
    tile_rows = scan_equal_dose_tiles(
        bundle,
        pairs,
        tile_width_nt=int(geometry["nominal_tile_width_nt"]),
        token_dose=int(geometry["fixed_BPE_token_dose_per_sequence"]),
        maximum_cover_nt=int(geometry["maximum_actual_cover_nt"]),
        maximum_cover_difference_nt=int(
            geometry["maximum_target_control_cover_length_difference_nt"]
        ),
        candidate_heads=CANDIDATE_HEADS,
        include_all_layer_heads=False,
        include_identity=False,
        batch_pairs=args.batch_pairs,
        tile_indices=tiles_to_score,
        # Restrict every requested tile to the "mask" operation only -- this run
        # never needs candidate_value_swap or candidate_identity (those are D8/
        # preflight concerns), which keeps the pass count to the scale documented
        # in the task brief (~8 tiles x eligible pairs x 2 branches).
        operations_by_tile={tile: ("mask",) for tile in tiles_to_score},
        progress=progress,
    )
    raw_rows = [{"row_type": "tile_effect", **row} for row in tile_rows]

    input_paths = [
        args.config,
        args.preflight_report,
        args.development_predictions,
        args.annotated_cohort,
    ]
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id="dnabert2_promoter_domain_confirmation_raw_v1",
        status="SEALED",
        model_identity=identity,
        input_paths=input_paths,
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "scored_tiles": tiles_to_score,
            "tiles_remaining_sealed": tiles_remaining_sealed,
            "smoke": bool(args.smoke),
            "limit_pairs": args.limit_pairs,
            "n_pairs_scored": len(pairs),
        },
        seeds={"inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(f"confirmation: wrote {len(raw_rows)} raw rows to {raw_path}", flush=True)

    # ---- Recompute endpoints from the artifact just written (not the in-memory
    # list), exactly as freeze_promoter_domain_predictions.py operates on a written
    # raw file -- this also exercises load_tile_rows on this run's own output. ----
    rows = load_tile_rows(raw_path, clusters=clusters, operation="mask")
    if not rows:
        raise SystemExit(
            "no mask tile rows were produced -- refusing to score empty confirmation data"
        )
    memorials = check_memorials(
        rows,
        expected_tokens=int(geometry["fixed_BPE_token_dose_per_sequence"]),
        max_cover_difference_nt=int(
            geometry["maximum_target_control_cover_length_difference_nt"]
        ),
    )
    print(f"confirmation: M3/M4 -- {memorials}", flush=True)

    endpoints = _score_endpoints(rows, config, predicted_signs, smoke=bool(args.smoke))

    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_domain_confirmation_v1",
        "phase": "CONFIRMATION",
        "split": args.split,
        "status": "SEALED",
        "smoke": bool(args.smoke),
        "config_sha256": config_hash,
        "checkpoint_revision": identity["revision"],
        "environment": identity["env"],
        "scored_tiles": tiles_to_score,
        "tiles_remaining_sealed": tiles_remaining_sealed,
        "n_pairs_scored": len(pairs),
        "development_predicted_signs": predicted_signs,
        "memorials": memorials,
        "endpoints": endpoints,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(raw_rows),
        },
        "scope": config["scope"],
    }
    result_path, result_manifest = write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_domain_confirmation_v1",
        status="SEALED",
        model_identity=identity,
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "batch_pairs": args.batch_pairs,
            "scored_tiles": tiles_to_score,
            "smoke": bool(args.smoke),
            "limit_pairs": args.limit_pairs,
        },
        seeds={"inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(f"confirmation: wrote {result_path}", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_predicted_signs(path: str, endpoint_names: Iterable[str]) -> dict[str, int]:
    """Load ONLY each endpoint's ``predicted_sign`` from the frozen development
    predictions file.

    Deliberately never reads or retains ``mean``/``ci95`` from that file: the
    confirmation must not be steered by the development effect size. The raw parsed
    JSON (``raw``, which does contain ``mean``/``ci95``) is a purely local variable
    that goes out of scope when this function returns; only the extracted integer
    signs are handed back to the caller.
    """

    predictions_path = Path(path)
    if not predictions_path.exists():
        raise SystemExit(
            "refusing to run: --development-predictions file does not exist: "
            f"{path}"
        )
    raw = json.loads(predictions_path.read_text())
    if raw.get("status") != "FROZEN" or raw.get("model_outcomes_computed") is not False:
        raise SystemExit(
            "refusing to run: development predictions file is not a frozen, "
            f"model-free prediction record: {path}"
        )
    raw_endpoints = raw["endpoints"]
    signs: dict[str, int] = {}
    for name in endpoint_names:
        entry = raw_endpoints.get(name)
        if entry is None or "predicted_sign" not in entry:
            raise SystemExit(
                "refusing to run: development predictions file is missing "
                f"predicted_sign for endpoint {name!r}"
            )
        signs[name] = int(entry["predicted_sign"])
    return signs


def _endpoint_values(
    name: str, spec: Mapping[str, Any], rows: Sequence[TileRow]
) -> tuple[list[float], list[str]]:
    """Dispatch on endpoint NAME, exactly as freeze_promoter_domain_predictions.py
    does. Duck-typing on config keys is wrong here: E1b_sharp_edge and
    E3_upstream_anti_evidence both carry a "tiles" key with different meanings (a
    pair vs a single tile), and E4_full_domain carries no tile key at all.
    """

    if name == "E1_downstream_edge":
        return contrast_by_locus(
            rows,
            inner_tiles=tuple(spec["inner_tiles"]),
            outer_tiles=tuple(spec["outer_tiles"]),
        )
    if name == "E1b_sharp_edge":
        tile_a, tile_b = (int(index) for index in spec["tiles"])
        return paired_tile_difference(rows, tile_a=tile_a, tile_b=tile_b)
    if name == "E2_tss_polarity":
        return contrast_by_locus(
            rows,
            inner_tiles=tuple(spec["downstream_tiles"]),
            outer_tiles=tuple(spec["upstream_tiles"]),
        )
    if name == "E3_upstream_anti_evidence":
        return single_tile_by_locus(rows, tile_index=int(spec["tiles"][0]))
    if name == "E4_full_domain":
        # Descriptive only; includes tile 26, so it can never qualify.
        return contrast_by_locus(rows, inner_tiles=(25, 26, 27), outer_tiles=(28, 29))
    raise SystemExit(f"unhandled endpoint in config: {name}")


def _endpoint_sesoi(spec: Mapping[str, Any]) -> tuple[float, str]:
    """Return (threshold, kind); kind is "absolute" for E3's two-sided SESOI key,
    "signed" for the one-sided-positive endpoints' plain SESOI key."""

    if "SESOI_logit_margin" in spec:
        return float(spec["SESOI_logit_margin"]), "signed"
    if "SESOI_absolute_logit_margin" in spec:
        return float(spec["SESOI_absolute_logit_margin"]), "absolute"
    raise SystemExit(f"endpoint spec has no recognized SESOI key: {spec}")


def _score_endpoints(
    rows: Sequence[TileRow],
    config: Mapping[str, Any],
    predicted_signs: Mapping[str, int],
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Compute E1-E4. For E1, E1b, E2, E3: cluster_bootstrap_interval for the CI,
    paired_cluster_sign_permutation + add_one_permutation_pvalue for the add-one p,
    then holm_adjust across those four (not E4). E4 is descriptive only and is
    marked unqualified unconditionally. Seeds and draw counts come from
    config["inference"] -- never hardcoded.
    """

    inference = config["inference"]
    draws = int(inference["bootstrap_draws"])
    sign_flip_draws = int(inference["sign_flip_draws"])
    base_seed = int(inference["seed"])
    alpha = float(inference["alpha"])

    endpoint_names = list(config["endpoints"].keys())
    results: dict[str, Any] = {}
    raw_p: dict[str, float] = {}

    for index, name in enumerate(endpoint_names):
        spec = config["endpoints"][name]
        values, value_clusters = _endpoint_values(name, spec, rows)
        entry: dict[str, Any] = {
            "role": spec.get("role"),
            "n_loci": len(values),
            "n_clusters": len(set(value_clusters)),
            "predicted_sign": int(predicted_signs[name]),
        }

        if name == "E4_full_domain":
            if values:
                seed = base_seed ^ ((index + 1) * 0x3333)
                entry["mean"] = sum(values) / len(values)
                entry["ci95"] = list(
                    cluster_bootstrap_interval(
                        values, value_clusters, draws=draws, seed=seed
                    )
                )
            else:
                entry["mean"] = None
                entry["ci95"] = None
            entry["qualified"] = False
            entry["reason"] = "descriptive_only"
            results[name] = entry
            continue

        if not values:
            entry.update(
                {
                    "mean": None,
                    "ci95": None,
                    "raw_add_one_p": None,
                    "holm_adjusted_p": None,
                    "checks": None,
                    "qualified": False,
                    "reason": "no_eligible_loci",
                }
            )
            results[name] = entry
            continue

        sesoi, sesoi_kind = _endpoint_sesoi(spec)
        alternative_label = str(spec["alternative"])
        pvalue_mode = ALTERNATIVE_TO_PVALUE_MODE[alternative_label]

        sign_seed = base_seed ^ ((index + 1) * 0x1111)
        bootstrap_seed = base_seed ^ ((index + 1) * 0x2222)
        point, null = paired_cluster_sign_permutation(
            values, value_clusters, permutations=sign_flip_draws, seed=sign_seed
        )
        raw_pvalue = add_one_permutation_pvalue(point, null, alternative=pvalue_mode)
        ci_low, ci_high = cluster_bootstrap_interval(
            values, value_clusters, draws=draws, seed=bootstrap_seed
        )
        raw_p[name] = raw_pvalue
        entry.update(
            {
                "mean": point,
                "ci95": [ci_low, ci_high],
                "alternative": alternative_label,
                "SESOI": sesoi,
                "SESOI_kind": sesoi_kind,
                "raw_add_one_p": raw_pvalue,
            }
        )
        results[name] = entry

    # Holm across every endpoint that actually produced a p-value above -- i.e.
    # every endpoint except E4_full_domain (which `continue`d before reaching
    # raw_p) and any endpoint with zero eligible loci.
    adjusted = holm_adjust(raw_p) if raw_p else {}
    for name, pvalue in adjusted.items():
        entry = results[name]
        sesoi = float(entry["SESOI"])
        point = float(entry["mean"])
        ci_low, ci_high = entry["ci95"]
        predicted_sign = int(entry["predicted_sign"])
        sign_value = 1 if point > 0 else (-1 if point < 0 else 0)

        meets_sesoi = abs(point) >= sesoi if entry["SESOI_kind"] == "absolute" else point >= sesoi

        if predicted_sign > 0:
            excludes_zero = ci_low > 0
        elif predicted_sign < 0:
            excludes_zero = ci_high < 0
        else:
            excludes_zero = False

        sign_matches = predicted_sign != 0 and sign_value == predicted_sign
        holm_pass = pvalue <= alpha

        checks = {
            "meets_SESOI": meets_sesoi,
            "interval_excludes_zero_on_alternative_side": excludes_zero,
            "holm_p_at_most_alpha": holm_pass,
            "sign_matches_predicted_development_sign": sign_matches,
        }
        entry["holm_adjusted_p"] = pvalue
        entry["checks"] = checks
        if smoke:
            # Smoke runs never qualify anything, regardless of what the (still
            # genuinely computed, above) arithmetic says -- qualification must
            # never be granted on a deliberately truncated sample.
            entry["qualified"] = False
            entry["reason"] = "smoke_run_never_qualifies"
        else:
            entry["qualified"] = all(checks.values())
            if not entry["qualified"]:
                entry["reason"] = "failed_qualification_checks"
    return results


if __name__ == "__main__":
    main()
