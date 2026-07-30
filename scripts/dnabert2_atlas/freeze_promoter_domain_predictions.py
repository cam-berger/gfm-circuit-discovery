"""Freeze D12 development predictions from the existing D8 raw records.

Reads no model. Records the exploratory development estimates and, critically,
their SIGNS, which the confirmation gate requires.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from gfmcd.reconstruction_atlas.promoter_domain import (
    check_memorials,
    contrast_by_locus,
    load_clusters,
    load_tile_rows,
    paired_tile_difference,
    single_tile_by_locus,
)
from gfmcd.reconstruction_atlas.statistics import cluster_bootstrap_interval


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-raw", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing result: {out}")

    config = json.loads(Path(args.config).read_text())
    clusters = load_clusters(args.cohort)
    rows = load_tile_rows(args.development_raw, clusters=clusters, operation="mask")
    if not rows:
        raise SystemExit("no mask tile rows found")

    memorials = check_memorials(
        rows,
        expected_tokens=config["tile_geometry"]["fixed_BPE_token_dose_per_sequence"],
        max_cover_difference_nt=config["tile_geometry"][
            "maximum_target_control_cover_length_difference_nt"
        ],
    )

    seed = int(config["inference"]["seed"])
    draws = int(config["inference"]["bootstrap_draws"])
    endpoints: dict[str, dict] = {}

    # Dispatch on the endpoint NAME, not on which keys happen to be present.
    # Duck-typing breaks here: E1b_sharp_edge and E3_upstream_anti_evidence both
    # carry a "tiles" key with different semantics (a pair vs a single tile), and
    # E4_full_domain carries no tile key at all.
    for name, spec in config["endpoints"].items():
        if name == "E1_downstream_edge":
            values, clusters = contrast_by_locus(
                rows,
                inner_tiles=tuple(spec["inner_tiles"]),
                outer_tiles=tuple(spec["outer_tiles"]),
            )
        elif name == "E1b_sharp_edge":
            tile_a, tile_b = (int(index) for index in spec["tiles"])
            values, clusters = paired_tile_difference(
                rows, tile_a=tile_a, tile_b=tile_b
            )
        elif name == "E2_tss_polarity":
            values, clusters = contrast_by_locus(
                rows,
                inner_tiles=tuple(spec["downstream_tiles"]),
                outer_tiles=tuple(spec["upstream_tiles"]),
            )
        elif name == "E3_upstream_anti_evidence":
            values, clusters = single_tile_by_locus(
                rows, tile_index=int(spec["tiles"][0])
            )
        elif name == "E4_full_domain":
            # Descriptive only; includes tile 26, so it can never qualify.
            values, clusters = contrast_by_locus(
                rows, inner_tiles=(25, 26, 27), outer_tiles=(28, 29)
            )
        else:
            raise SystemExit(f"unhandled endpoint in config: {name}")
        if not values:
            endpoints[name] = {"n_loci": 0, "predicted_sign": 0}
            continue
        point = sum(values) / len(values)
        low, high = cluster_bootstrap_interval(
            values, clusters, draws=draws, seed=seed
        )
        endpoints[name] = {
            "mean": point,
            "ci95": [low, high],
            "n_loci": len(values),
            "n_clusters": len(set(clusters)),
            "predicted_sign": 1 if point > 0 else (-1 if point < 0 else 0),
        }

    report = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_domain_v1",
        "phase": "DEVELOPMENT_PREDICTIONS_FROZEN",
        "status": "FROZEN",
        "model_outcomes_computed": False,
        "boundary": (
            "Exploratory. Derived by re-aggregating D8 development records that had "
            "already been read. These are predictions, not evidence."
        ),
        "memorials": memorials,
        "endpoints": endpoints,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "inputs": {
            "development_raw": args.development_raw,
            "development_raw_sha256": _sha256(Path(args.development_raw)),
            "cohort": args.cohort,
            "cohort_sha256": _sha256(Path(args.cohort)),
            "config": args.config,
            "config_sha256": _sha256(Path(args.config)),
        },
        "code_revision": revision,
        "command": (
            "uv run python scripts/dnabert2_atlas/freeze_promoter_domain_predictions.py "
            f"--development-raw {args.development_raw} --cohort {args.cohort} "
            f"--config {args.config} --out {args.out}"
        ),
        "result_sha256": _sha256(out),
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(endpoints, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
