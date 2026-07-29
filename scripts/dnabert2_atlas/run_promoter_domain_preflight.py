#!/usr/bin/env python
"""Instrument-qualification preflight for the D12 promoter-domain assay.

This is a gate, not a measurement of biology. Before Task 5 may open the sealed
D12 confirmation split (tiles 21,22,23,25,27,28,29), this script proves the
current pinned environment (checkpoint, tokenizer, adapter code, device) still
reproduces values already on the public record for the *same* checkpoint and
cohort at tile 26 (whose confirmation value D8 already opened, so re-scoring it
spends nothing new):

* M1_identity        -- per-example Q/K/V identity-patch reconstruction on the
                         124 recorded ``candidate_identity`` rows. Recomputed
                         via the same explicit-region patch machinery that
                         generated them (``operation="identity"`` is a
                         no-op patch: it recomputes each candidate head's CLS
                         attention output from captured Q/K/V without swapping
                         anything, so it should reproduce the native margin to
                         numerical noise). Gate: max |recomputed| < tolerance.
* M2_record_anchor    -- per-example mask-and-rescore reconstruction on the 124
                         recorded ``mask`` rows, using the exact token ids
                         recorded per row (no retokenization). Gate: max
                         |recomputed - recorded| < tolerance. This is the
                         load-bearing memorial: it is the only one of the five
                         that fails if the checkpoint, tokenizer, cohort, or
                         batching drifted from what D8 recorded.
* M3_cover_balance,
  M4_equal_dose       -- pure-arithmetic geometry memorials over the same
                         recorded rows (gfmcd.reconstruction_atlas.promoter_domain).
* batch_invariance    -- native margins on a fresh, reserved GUE slice (never
                         touched by any prior D8/D12 run) must not depend on
                         batch size.

``decision`` is "QUALIFIED" only if every check that ran passed. Under
``--smoke``, M2 is not run (it is the expensive ~500-forward-pass check) and is
excluded from the decision rule rather than counted as failed; its status is
recorded as ``"SKIPPED_IN_SMOKE"``.

Failure here must not be papered over: per repo convention (README.md,
CLAUDE.md), a failed preflight is recorded as ``INSTRUMENT_UNRESOLVED`` and is
itself a legitimate, immutable result.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    PROMOTER_GUE_DATASET_ID,
    PROMOTER_GUE_DATASET_REVISION,
    explicit_qkv_operation_effects_for_batch,
    load_dnabert2_promoter_classifier,
    tokenize_promoter_pairs,
)
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_domain import (
    check_memorials,
    load_clusters,
    load_tile_rows,
)
from gfmcd.reconstruction_atlas.promoter_head_scan import (
    native_pair_rows,
    read_jsonl,
    select_gue_pairs,
)

Pair = tuple[str, str, str]

# The two candidate heads this whole D8/D12 study line patches (L11H2, L11H5).
# Not config-driven: hardcoded identically in run_promoter_source_localization.py
# and run_promoter_source_localization_preflight.py, which this script mirrors.
CANDIDATE_HEADS: tuple[tuple[int, int], ...] = ((11, 2), (11, 5))

# Prescribed directly by the task brief (not present in the frozen v2 config,
# which has no batch_invariance field).
BATCH_INVARIANCE_TOLERANCE = 1e-5
BATCH_INVARIANCE_MAX_PAIRS = 16

REQUIRED_TRANSFORMERS_VERSION = "4.44.2"
EXPECTED_TILE_INDEX = 26
EXPECTED_MASK_ROWS = 124
EXPECTED_IDENTITY_ROWS = 124


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--gue-test",
        default=None,
        help=(
            "Raw GUE prom_300_all test.csv. Defaults to the pinned "
            f"{PROMOTER_GUE_DATASET_ID}@{PROMOTER_GUE_DATASET_REVISION} local HF cache path."
        ),
    )
    parser.add_argument("--gue-pair-offset", type=int, required=True)
    parser.add_argument("--gue-pairs", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=32)
    args = parser.parse_args()

    import transformers

    if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers=={transformers.__version__} but this preflight is pinned to "
            f"{REQUIRED_TRANSFORMERS_VERSION}. Run under the pinned environment, e.g.:\n"
            '  uv run --python 3.11 --with "transformers==4.44.2" '
            '--with "tokenizers>=0.19,<0.20" --with "einops>=0.7" --with "numpy<2" '
            '--with "scipy" --with "pyfaidx>=0.8" --with "torch>=2.2" python <script> ...'
        )

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    if config.get("status") != "FROZEN":
        raise RuntimeError("promoter-domain preflight requires a FROZEN config")

    gue_test_path = args.gue_test or _resolve_gue_test_csv()
    cohort_path = str(config["cohort"]["annotated_source"])
    confirmation_path = str(config["memorials"]["M2_record_anchor"]["source"])
    expected_checkpoint_revision = str(config["checkpoint"]["revision"])

    print(f"preflight: loading prom_300_all on {args.device} ...", flush=True)
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    identity = bundle.identity()
    if identity["revision"] != expected_checkpoint_revision:
        raise RuntimeError(
            f"loaded checkpoint revision {identity['revision']!r} does not match the "
            f"frozen config revision {expected_checkpoint_revision!r} -- refusing to proceed"
        )
    print(
        "preflight: checkpoint OK "
        f"(revision={identity['revision']}, env={identity['env']})",
        flush=True,
    )

    confirmation_rows = read_jsonl(confirmation_path)
    bad_tile = sorted(
        {int(row["tile_index"]) for row in confirmation_rows} - {EXPECTED_TILE_INDEX}
    )
    if bad_tile:
        raise RuntimeError(
            f"confirmation.raw.jsonl contains tile_index(es) other than "
            f"{EXPECTED_TILE_INDEX}: {bad_tile}"
        )
    mask_rows = [row for row in confirmation_rows if row["operation"] == "mask"]
    identity_rows = [
        row for row in confirmation_rows if row["operation"] == "candidate_identity"
    ]
    if len(mask_rows) != EXPECTED_MASK_ROWS or len(identity_rows) != EXPECTED_IDENTITY_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_MASK_ROWS} mask + {EXPECTED_IDENTITY_ROWS} "
            f"candidate_identity rows, found {len(mask_rows)} mask + "
            f"{len(identity_rows)} candidate_identity -- stop and report, do not proceed"
        )

    cohort_rows = read_jsonl(cohort_path)
    sequences_by_pair: dict[str, tuple[str, str]] = {
        str(row["example_id"]): (
            str(row["target"]["clean_sequence"]),
            str(row["control"]["clean_sequence"]),
        )
        for row in cohort_rows
    }
    clusters = load_clusters(cohort_path)
    needed_pairs = {row["pair_id"] for row in mask_rows + identity_rows}
    missing_pairs = sorted(needed_pairs - set(sequences_by_pair))
    if missing_pairs:
        raise RuntimeError(
            f"cohort is missing {len(missing_pairs)} recorded pair_id(s), e.g. "
            f"{missing_pairs[:5]}"
        )

    # ---- M1: per-example identity-patch reproduction (always runs) ----
    m1_tolerance = float(config["memorials"]["M1_identity"]["tolerance"])
    print(f"preflight: M1 -- rescoring {len(identity_rows)} candidate_identity rows ...", flush=True)
    m1 = _check_m1(
        bundle,
        identity_rows,
        sequences_by_pair,
        tolerance=m1_tolerance,
        batch_pairs=args.batch_pairs,
    )
    print(f"preflight: M1 worst |recomputed| = {m1['worst_absolute_value']:.3e} "
          f"(tolerance {m1_tolerance:.3e}) -> {'PASS' if m1['pass'] else 'FAIL'}", flush=True)

    # ---- M2: per-example record-anchor reproduction (skipped in smoke) ----
    m2_tolerance = float(config["memorials"]["M2_record_anchor"]["maximum_absolute_difference"])
    if args.smoke:
        m2: dict[str, Any] = {
            "status": "SKIPPED_IN_SMOKE",
            "pass": None,
            "tolerance": m2_tolerance,
            "n_rows": len(mask_rows),
        }
        print("preflight: M2 -- SKIPPED_IN_SMOKE", flush=True)
    else:
        print(f"preflight: M2 -- rescoring {len(mask_rows)} mask rows "
              f"(~{4 * len(mask_rows)} forward passes) ...", flush=True)
        m2 = _check_m2(bundle, mask_rows, sequences_by_pair, tolerance=m2_tolerance)
        print(f"preflight: M2 worst |recomputed-recorded| = "
              f"{m2['worst_absolute_difference']:.3e} (tolerance {m2_tolerance:.3e}) "
              f"-> {'PASS' if m2['pass'] else 'FAIL'} (worst pair_id={m2['worst_pair_id']})",
              flush=True)

    # ---- M3/M4: pure-arithmetic geometry memorials over the recorded mask rows ----
    tile_rows = load_tile_rows(confirmation_path, clusters=clusters, operation="mask")
    m3_m4 = check_memorials(
        tile_rows,
        expected_tokens=int(config["memorials"]["M4_equal_dose"]["expected_tokens"]),
        max_cover_difference_nt=int(config["memorials"]["M3_cover_balance"]["tolerance_nt"]),
    )
    print(f"preflight: M3/M4 -- {m3_m4}", flush=True)

    # ---- batch invariance on a fresh, reserved GUE slice ----
    print(
        f"preflight: selecting {args.gue_pairs} reserved GUE pairs at offset "
        f"{args.gue_pair_offset} ...",
        flush=True,
    )
    gue_pairs = select_gue_pairs(
        gue_test_path,
        count=args.gue_pairs,
        offset=args.gue_pair_offset,
        seed=int(config["inference"]["seed"]),
    )
    bi_pairs = gue_pairs[: min(BATCH_INVARIANCE_MAX_PAIRS, len(gue_pairs))]
    batch_of_1 = {
        row["pair_id"]: row
        for row in native_pair_rows(bundle, bi_pairs, batch_pairs=1)
    }
    batch_of_n = {
        row["pair_id"]: row
        for row in native_pair_rows(bundle, bi_pairs, batch_pairs=len(bi_pairs))
    }
    if set(batch_of_1) != set(batch_of_n) or set(batch_of_1) != {pid for pid, _, _ in bi_pairs}:
        raise RuntimeError("batch-invariance pair sets do not agree")
    batch_invariance_max_diff = max(
        abs(float(batch_of_1[pair_id][field]) - float(batch_of_n[pair_id][field]))
        for pair_id in batch_of_1
        for field in ("target_margin", "control_margin")
    )
    batch_invariance_pass = batch_invariance_max_diff < BATCH_INVARIANCE_TOLERANCE
    print(
        f"preflight: batch invariance (n={len(bi_pairs)}, batch 1 vs {len(bi_pairs)}) "
        f"max |diff| = {batch_invariance_max_diff:.3e} "
        f"(tolerance {BATCH_INVARIANCE_TOLERANCE:.3e}) -> "
        f"{'PASS' if batch_invariance_pass else 'FAIL'}",
        flush=True,
    )

    checks: dict[str, bool | None] = {
        "M1_identity": bool(m1["pass"]),
        "M2_record_anchor": None if args.smoke else bool(m2["pass"]),
        "M3_cover_balance": bool(m3_m4["M3_cover_balance"]),
        "M4_equal_dose": bool(m3_m4["M4_equal_dose"]),
        "batch_invariance": bool(batch_invariance_pass),
    }
    evaluated = {name: value for name, value in checks.items() if value is not None}
    qualified = bool(evaluated) and all(evaluated.values())
    decision = "QUALIFIED" if qualified else "INSTRUMENT_UNRESOLVED"
    failing_checks = sorted(name for name, value in evaluated.items() if not value)

    input_paths = (args.config, cohort_path, confirmation_path, gue_test_path)
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_domain_preflight_v1",
        "decision": decision,
        "smoke": bool(args.smoke),
        "config_sha256": config_hash,
        "checkpoint_revision": identity["revision"],
        "environment": identity["env"],
        "input_sha256": {
            "config": sha256_file(args.config),
            "cohort": sha256_file(cohort_path),
            "confirmation_raw": sha256_file(confirmation_path),
            "gue_test": sha256_file(gue_test_path),
        },
        "gue_pair_offset": int(args.gue_pair_offset),
        "gue_pairs_requested": int(args.gue_pairs),
        "gue_pair_ids": [pair_id for pair_id, _, _ in gue_pairs],
        "checks": checks,
        "failing_checks": failing_checks,
        "measurements": {
            "M1_identity": m1,
            "M2_record_anchor": m2,
            "M3_cover_balance": m3_m4["M3_cover_balance"],
            "M4_equal_dose": m3_m4["M4_equal_dose"],
            "batch_invariance": {
                "pass": batch_invariance_pass,
                "tolerance": BATCH_INVARIANCE_TOLERANCE,
                "max_absolute_margin_difference": batch_invariance_max_diff,
                "n_pairs": len(bi_pairs),
                "batch_sizes_compared": [1, len(bi_pairs)],
            },
        },
        "boundary": (
            "Qualifies numerical reproduction of the recorded D8 tile-26 confirmation "
            "memorials plus instrument batch invariance only. Does not open the D12 "
            "sealed confirmation split (tiles 21,22,23,25,27,28,29) and licenses no "
            "biological conclusion."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_domain_preflight_v1",
        status="PREFLIGHT" if qualified else "INSTRUMENT_UNRESOLVED",
        model_identity=identity,
        input_paths=input_paths,
        parameters={
            "config_sha256": config_hash,
            "smoke": bool(args.smoke),
            "batch_pairs": int(args.batch_pairs),
            "candidate_heads": [list(head) for head in CANDIDATE_HEADS],
            "batch_invariance_tolerance": BATCH_INVARIANCE_TOLERANCE,
        },
        seeds={"gue_selection": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _resolve_gue_test_csv() -> str:
    """Locate the pinned GUE prom_300_all test.csv in the local HF cache; never download."""

    owner, name = PROMOTER_GUE_DATASET_ID.split("/", 1)
    pattern = os.path.expanduser(
        f"~/.cache/huggingface/hub/datasets--{owner}--{name}/snapshots/"
        f"{PROMOTER_GUE_DATASET_REVISION}/GUE/prom_300_all/test.csv"
    )
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(
            f"GUE prom_300_all test.csv not found at pinned revision "
            f"{PROMOTER_GUE_DATASET_REVISION} under {pattern}; pass --gue-test explicitly"
        )
    return hits[0]


def _check_m2(
    bundle: Any,
    mask_rows: list[dict[str, Any]],
    sequences_by_pair: dict[str, tuple[str, str]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Per-example mask-and-rescore reproduction of the recorded ``mask`` rows.

    Deliberately unbatched and independent of the tile-geometry scanner: each
    row's own recorded token indices are used directly (no retokenization, no
    recomputed cover geometry), matching the pre-verified reference recipe
    (worst absolute difference 4.3e-06 on a 6-row spot check under this
    pinned environment).
    """

    import torch

    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("promoter tokenizer exposes no mask token")

    def margin(ids: list[int]) -> float:
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(input_ids=t, attention_mask=torch.ones_like(t))
        logits = out.logits[0]
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite M2 logits")
        return float(logits[1] - logits[0])

    worst_absolute_difference = 0.0
    worst_pair_id: str | None = None
    for row_index, row in enumerate(mask_rows):
        target_seq, control_seq = sequences_by_pair[row["pair_id"]]
        tid = tokenizer(target_seq)["input_ids"]
        cid = tokenizer(control_seq)["input_ids"]
        tid_m = list(tid)
        for i in row["target_token_indices"]:
            tid_m[i] = mask_id
        cid_m = list(cid)
        for i in row["control_token_indices"]:
            cid_m[i] = mask_id
        recomputed = (margin(tid) - margin(tid_m)) - (margin(cid) - margin(cid_m))
        recorded = float(row["target_specific_loss"])
        difference = abs(recomputed - recorded)
        if difference > worst_absolute_difference:
            worst_absolute_difference = difference
            worst_pair_id = str(row["pair_id"])
        if (row_index + 1) % 25 == 0:
            print(f"preflight: M2 {row_index + 1}/{len(mask_rows)} rows", flush=True)
    return {
        "status": "RAN",
        "pass": worst_absolute_difference < tolerance,
        "tolerance": tolerance,
        "worst_absolute_difference": worst_absolute_difference,
        "worst_pair_id": worst_pair_id,
        "n_rows": len(mask_rows),
    }


def _check_m1(
    bundle: Any,
    identity_rows: list[dict[str, Any]],
    sequences_by_pair: dict[str, tuple[str, str]],
    *,
    tolerance: float,
    batch_pairs: int,
) -> dict[str, Any]:
    """Per-example Q/K/V identity-patch reproduction of the recorded ``candidate_identity`` rows.

    ``candidate_identity`` rows are NOT token-mask rows: they were generated by
    ``explicit_qkv_operation_effects_for_batch(..., operation="identity")``
    (src/gfmcd/reconstruction_atlas/promoter_source_execution.py,
    scan_equal_dose_tiles with include_identity=True) -- a no-op Q/K/V patch at
    the two candidate heads that recomputes each head's CLS attention output
    from captured Q/K/V without permuting anything. It should reproduce the
    native margin to numerical noise; a real token-mask substitution at the
    same indices would NOT (compare to the paired ``mask`` row at the same
    pair_id/tile, whose effect is large). Reusing the recorded token indices
    directly, exactly as they were captured, rather than recomputing tile
    geometry.
    """

    if batch_pairs < 1:
        raise ValueError("batch_pairs must be positive")
    pairs: list[Pair] = [
        (str(row["pair_id"]), *sequences_by_pair[row["pair_id"]]) for row in identity_rows
    ]
    worst_absolute_value = 0.0
    worst_pair_id: str | None = None
    worst_absolute_difference_vs_recorded = 0.0
    for start in range(0, len(identity_rows), batch_pairs):
        chunk_rows = identity_rows[start : start + batch_pairs]
        chunk_pairs = pairs[start : start + batch_pairs]
        token_indices: list[list[int]] = []
        for row in chunk_rows:
            token_indices.append([int(i) for i in row["target_token_indices"]])
            token_indices.append([int(i) for i in row["control_token_indices"]])
        batch = tokenize_promoter_pairs(bundle, chunk_pairs)
        effects = explicit_qkv_operation_effects_for_batch(
            bundle,
            batch,
            CANDIDATE_HEADS,
            token_indices,
            operation="identity",
        )
        for local_index, row in enumerate(chunk_rows):
            recomputed = float(effects["target_specific_loss"][local_index])
            recorded = float(row["target_specific_loss"])
            if abs(recomputed) > worst_absolute_value:
                worst_absolute_value = abs(recomputed)
                worst_pair_id = str(row["pair_id"])
            worst_absolute_difference_vs_recorded = max(
                worst_absolute_difference_vs_recorded, abs(recomputed - recorded)
            )
        print(
            f"preflight: M1 {min(start + batch_pairs, len(identity_rows))}/"
            f"{len(identity_rows)} rows",
            flush=True,
        )
    return {
        "status": "RAN",
        "pass": worst_absolute_value < tolerance,
        "tolerance": tolerance,
        "worst_absolute_value": worst_absolute_value,
        "worst_pair_id": worst_pair_id,
        "worst_absolute_difference_vs_recorded": worst_absolute_difference_vs_recorded,
        "n_rows": len(identity_rows),
        "candidate_heads": [list(head) for head in CANDIDATE_HEADS],
    }


if __name__ == "__main__":
    main()
