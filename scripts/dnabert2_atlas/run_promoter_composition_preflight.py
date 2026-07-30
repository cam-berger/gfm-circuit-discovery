#!/usr/bin/env python
"""Instrument-qualification preflight for the D13 composition-vs-position assay.

D13's primary claim (P1) is an EXCLUSION: that the target-minus-control GC/CpG
composition differential does not account for the confirmed TSS polarity. An
exclusion is only as good as the instrument's demonstrated ability to detect the
thing being excluded -- see PREREGISTRATION_D13_COMPOSITION_VS_POSITION.md,
"Why the planted positive is load-bearing", which cites the recorded IF30
precedent (promoter_native_source_preflight_v7/v8: INSTRUMENT_UNRESOLVED for a
GC-depletion instrument whose projected MDE never reached an interpretable
precision). This script is a gate, not a measurement of biology. It qualifies
the SAME measurement primitives (enumerate_region_covers / select_matched_pair /
select_unmatched_pair from gfmcd.reconstruction_atlas.promoter_composition, plus
in-place token masking) that a later development-split runner will use, on a
reserved GUE prom_300_all slice disjoint from every prior D8/D11/D12/D13 slice
(D8 used offsets 112-159, D11 960+, D12 992-1023; this preflight is run at
--gue-pair-offset 1056). It never reads or scores the D13 composition cohort
(development or confirmation split) itself.

Config of record: configs/dnabert2_atlas/promoter_composition_v2.frozen.json.
v1 is superseded (see that config's own ``supersession_reason``) and is not
read by this script except insofar as a caller could still point --config at
it; every threshold below is read from whatever FROZEN config is passed, never
hardcoded, except the two calibration numbers noted below that live only in
the config's prose.

Checks, all frozen in the config unless noted:

* M1_identity            -- masking zero tokens must reproduce target_specific_loss
                             at ~0, through the SAME in-place token-substitution
                             function (``_masked_ids`` / ``_target_specific_loss``)
                             the endpoints use. On-path per the config's own note
                             (memorials.M1_identity.note), unlike D12's M1 (a Q/K/V
                             identity patch that never touched the real mask-and-
                             rescore code at all). Scope, stated precisely rather
                             than oversold: with an EMPTY index list the masking
                             loop body never executes, so this specific check can
                             only catch a bug in how an empty list is handled (e.g.
                             an accidental "mask everything" fallback, or an index
                             error) -- it cannot exercise the aliasing/off-by-one
                             bug classes that only a non-empty index list would
                             reveal. Still genuinely on-path and genuinely able to
                             fail for the bug class it covers.
* M2_caliper_respected,
  M3_equal_dose,
  M4_region_containment  -- pure-geometry memorials, recomputed independently
                             from each selected CoverPair/TokenTile's own fields
                             (not trusted from any prior record -- there is none;
                             this is the first forward pass D13 ever runs).
                             ARCHITECTURE-FORCED: select_matched_pair /
                             select_unmatched_pair only ever return a candidate
                             that already clears the caliper/separation filter
                             internally; paired_fixed_dose_tile only ever returns
                             a tile at exactly token_dose adjacent tokens within
                             the length-difference cap or returns None; and
                             enumerate_region_covers only ever yields a cover
                             whose centre already sits inside the requested
                             region. Their PASS is a regression guard on this
                             script's OWN wiring (did it pass the right region
                             bounds / caliper / dose to the right call), not
                             independent evidence about the selected pairs -- the
                             same distinction this repo requires for D12's M1 and
                             for the D13 supply audit's CHECK_FALSIFIABILITY
                             classification (R7/R21;
                             audit_promoter_composition_supply.py).
* planted_positive        -- the deliberately unmatched arm (GC separation >=
                             the config's unmatched_arm_minimum_GC_separation)
                             must beat the matched arm by >= the P2 SESOI, on the
                             SAME reserved loci. The load-bearing sensitivity
                             check described above; genuinely falsifiable (real
                             forward passes, real cover selection).
* permuted_label_fpr      -- see ``_check_permuted_label_fpr`` below for the
                             exact, interpretive resampling design this script
                             uses. The frozen config states the desired OUTCOME
                             check ("permute target/control labels 256 times,
                             center within 0.05 of zero") but not the resampling
                             mechanics, so this is a documented judgment call,
                             not a literal transcription of a fully-specified
                             algorithm.
* projected_MDE           -- 2.80*sd/sqrt(n) at the ALREADY-AUDITED (outcome-
                             blind, Task 4 / Amendment 1) development matched-pair
                             count, sd taken from the config's own
                             supersession_reason (1.2745). Never reruns the model
                             to obtain this number -- it is a real arithmetic
                             gate against a real, independently-committed count.
* batch_invariance        -- native margins on this same reserved slice must not
                             depend on batch size (mirrors
                             run_promoter_domain_preflight.py's own check
                             exactly).

``decision`` is "QUALIFIED" only if every check above passes. A failing planted
positive (or any other check) means this run must record INSTRUMENT_UNRESOLVED
and D13 opens nothing -- per the config's own stop_rules, that is a legitimate,
immutable, publishable result, never something to route around by widening a
caliper or a tolerance.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from gfmcd.adapters.dnabert2 import (
    PROMOTER_GUE_DATASET_ID,
    PROMOTER_GUE_DATASET_REVISION,
    load_dnabert2_promoter_classifier,
    tokenize_promoter_pairs,
)
from gfmcd.artifact_guard import guard_output_path
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import manifest_path_for, write_result_bundle
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_composition import (
    CoverPair,
    enumerate_region_covers,
    select_matched_pair,
    select_unmatched_pair,
)
from gfmcd.reconstruction_atlas.promoter_head_scan import native_pair_rows, select_gue_pairs

Pair = tuple[str, str, str]

REQUIRED_TRANSFORMERS_VERSION = "4.44.2"

# Matches audit_promoter_composition_supply.py's own cover-tiling granularity
# exactly, so this preflight's matched/unmatched availability is comparable to
# the already-committed, qualified Task 4 / Amendment 1 supply numbers rather
# than silently using a different tiling.
NOMINAL_WIDTH_NT = 10

BATCH_INVARIANCE_TOLERANCE = 1e-5
BATCH_INVARIANCE_MAX_PAIRS = 16

# Not their own structured JSON field in the frozen config: calibration.
# permuted_label_fpr.check's prose states both numbers directly ("... permuted
# 256 times ... within 0.05 of zero"), and PREREGISTRATION_D13's own
# "Permuted-label FPR" section repeats them verbatim. Mirrored here as
# constants and asserted against the config's own prose at startup
# (_assert_calibration_prose_matches) -- the same handling
# run_promoter_domain_preflight.py uses for BATCH_INVARIANCE_TOLERANCE /
# BATCH_INVARIANCE_MAX_PAIRS, a brief-prescribed value with no dedicated field.
PERMUTED_LABEL_DRAWS = 256
PERMUTED_LABEL_FPR_TOLERANCE = 0.05

# Amendment 1's measured per-locus SD of the single in-caliper-pair P1
# estimator, on the matched subset of the D12 confirmation data (config
# supersession_reason / PREREGISTRATION_D13 Amendment 1). Also not its own
# JSON field; asserted against the config's own prose at startup.
MATCHED_ARM_SD = 1.2745
MDE_Z_MULTIPLIER = 2.80  # "the standard two-sided, 80%-power form", Amendment 1
MDE_AUDITED_SPLIT = "development"

DEFAULT_SUPPLY_AUDIT_PATH = (
    "results/dnabert2_reconstruction_atlas/promoter_composition_supply_audit_v2.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--supply-audit", default=DEFAULT_SUPPLY_AUDIT_PATH)
    parser.add_argument("--gue-test", default=None)
    parser.add_argument("--gue-pair-offset", type=int, required=True)
    parser.add_argument("--gue-pairs", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=32)
    args = parser.parse_args()

    # Refuse to overwrite an existing output path -- checked before any model work.
    guard_output_path(args.out)
    guard_output_path(manifest_path_for(args.out))

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
        raise RuntimeError("composition preflight requires a FROZEN config")
    _assert_calibration_prose_matches(config)

    supply_audit_path = Path(args.supply_audit)
    supply_audit = json.loads(supply_audit_path.read_text())
    supply_manifest = json.loads(manifest_path_for(supply_audit_path).read_text())
    if supply_audit.get("status") != "QUALIFIED":
        raise RuntimeError(
            f"supply audit {supply_audit_path} is not QUALIFIED "
            f"(status={supply_audit.get('status')!r}) -- cannot trust its matched-pair count"
        )
    supply_manifest_config_hashes = {
        entry["sha256"] for entry in supply_manifest.get("data", [])
    }
    if sha256_file(args.config) not in supply_manifest_config_hashes:
        raise RuntimeError(
            f"supply audit {supply_audit_path} was not built against this exact config "
            f"({args.config}) -- refusing to trust its matched-pair count for the MDE gate"
        )
    audited_matched_pairs = int(
        supply_audit["matching_feasibility"]["by_split"][MDE_AUDITED_SPLIT][
            "matched_pair_frozen_caliper"
        ]
    )

    gue_test_path = args.gue_test or _resolve_gue_test_csv()
    expected_checkpoint_revision = str(config["checkpoint"]["revision"])

    print(f"preflight: loading prom_300_all on {args.device} ...", flush=True)
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    identity = bundle.identity()
    if identity["revision"] != expected_checkpoint_revision:
        raise RuntimeError(
            f"loaded checkpoint revision {identity['revision']!r} does not match the "
            f"frozen config revision {expected_checkpoint_revision!r} -- refusing to proceed"
        )
    tss_index = int(bundle.checkpoint.tss_upstream_nt)
    print(
        "preflight: checkpoint OK "
        f"(revision={identity['revision']}, env={identity['env']}, tss_index={tss_index})",
        flush=True,
    )

    mask_id = bundle.tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("promoter tokenizer exposes no mask token")
    if int(mask_id) != 4:
        raise RuntimeError(f"expected mask_token_id == 4, observed {mask_id}")

    matching_cfg = config["matching"]
    regions_cfg = config["regions"]
    caliper_gc = float(matching_cfg["tight_caliper_GC"])
    caliper_cpg = float(matching_cfg["tight_caliper_CpG"])
    min_unmatched_sep = float(matching_cfg["unmatched_arm_minimum_GC_separation"])
    token_dose = int(matching_cfg["token_dose_per_cover"])
    maximum_cover_nt = int(matching_cfg["maximum_cover_nt"])
    maximum_cover_diff_nt = int(matching_cfg["maximum_cover_length_difference_nt"])
    downstream_region = tuple(int(v) for v in regions_cfg["downstream_relative"])
    upstream_region = tuple(int(v) for v in regions_cfg["upstream_relative"])
    p2_sesoi = float(config["endpoints"]["P2_composition_attributable"]["SESOI_logit_margin"])
    mde_ceiling = float(config["calibration"]["projected_MDE_ceiling"])
    m1_tolerance = float(config["memorials"]["M1_identity"]["tolerance"])
    inference_seed = int(config["inference"]["seed"])

    print(
        f"preflight: selecting {args.gue_pairs} reserved GUE pairs at offset "
        f"{args.gue_pair_offset} ...",
        flush=True,
    )
    gue_pairs = select_gue_pairs(
        gue_test_path, count=args.gue_pairs, offset=args.gue_pair_offset, seed=inference_seed
    )

    # ---- Phase 1: selection -- outcome-blind. No margin/model call anywhere
    # below this point until Phase 2 begins; tokenize_promoter_pairs touches only
    # bundle.tokenizer, never bundle.model. ------------------------------------
    selections = _select_all_loci(
        bundle,
        gue_pairs,
        tss_index=tss_index,
        downstream_region=downstream_region,
        upstream_region=upstream_region,
        nominal_width=NOMINAL_WIDTH_NT,
        token_dose=token_dose,
        maximum_cover_nt=maximum_cover_nt,
        maximum_cover_diff_nt=maximum_cover_diff_nt,
        caliper_gc=caliper_gc,
        caliper_cpg=caliper_cpg,
        min_unmatched_sep=min_unmatched_sep,
        batch_pairs=args.batch_pairs,
    )
    n_matched = sum(1 for s in selections if s["matched"] is not None)
    n_unmatched = sum(1 for s in selections if s["unmatched"] is not None)
    n_matched_swapped = sum(1 for s in selections if s["matched_swapped"] is not None)
    print(
        f"preflight: selection complete (outcome-blind) -- "
        f"{n_matched}/{len(selections)} matched, {n_unmatched}/{len(selections)} unmatched, "
        f"{n_matched_swapped}/{len(selections)} matched_swapped -- entering scoring phase",
        flush=True,
    )

    # ---- Phase 2: scoring. Every masked-margin forward pass this preflight
    # performs happens from here on, consuming only Phase 1's already-selected
    # CoverPair objects; no new selection logic below. --------------------------
    margin, _margin_cache_dict = _margin_cache(bundle)
    locus_results = _score_all_loci(selections, margin=margin, mask_id=int(mask_id))

    m1 = _check_m1(locus_results, tolerance=m1_tolerance)
    print(
        f"preflight: M1 worst |target_specific_loss| = {m1['worst_absolute_value']:.3e} "
        f"(tolerance {m1_tolerance:.3e}) -> {'PASS' if m1['pass'] else 'FAIL'}",
        flush=True,
    )

    m2 = _check_m2(
        locus_results, caliper_gc=caliper_gc, caliper_cpg=caliper_cpg, min_unmatched_sep=min_unmatched_sep
    )
    print(
        f"preflight: M2 checked={m2['checked']} violations={len(m2['violations'])} -> "
        f"{'PASS' if m2['pass'] else 'FAIL'}",
        flush=True,
    )

    m3 = _check_m3(locus_results, token_dose=token_dose, maximum_cover_diff_nt=maximum_cover_diff_nt)
    print(
        f"preflight: M3 checked={m3['checked']} violations={len(m3['violations'])} -> "
        f"{'PASS' if m3['pass'] else 'FAIL'}",
        flush=True,
    )

    m4 = _check_m4(
        locus_results,
        tss_index=tss_index,
        downstream_region=downstream_region,
        upstream_region=upstream_region,
    )
    print(
        f"preflight: M4 checked={m4['checked']} violations={len(m4['violations'])} -> "
        f"{'PASS' if m4['pass'] else 'FAIL'}",
        flush=True,
    )

    planted = _check_planted_positive(locus_results, p2_sesoi=p2_sesoi)
    print(
        f"preflight: planted positive matched={planted['matched_arm_polarity_mean']} "
        f"unmatched={planted['unmatched_arm_polarity_mean']} diff={planted['difference']} "
        f"(n={planted['n_both_arms_loci']}, SESOI {p2_sesoi}) -> "
        f"{'PASS' if planted['pass'] else 'FAIL'}",
        flush=True,
    )

    fpr = _check_permuted_label_fpr(
        locus_results, seed=inference_seed, draws=PERMUTED_LABEL_DRAWS, tolerance=PERMUTED_LABEL_FPR_TOLERANCE
    )
    print(
        f"preflight: permuted-label FPR center={fpr['center']} "
        f"(tolerance {PERMUTED_LABEL_FPR_TOLERANCE}, n={fpr['n_loci_available']}) -> "
        f"{'PASS' if fpr['pass'] else 'FAIL'}",
        flush=True,
    )

    mde = _check_projected_mde(
        audited_matched_pairs=audited_matched_pairs,
        sd=MATCHED_ARM_SD,
        z_multiplier=MDE_Z_MULTIPLIER,
        ceiling=mde_ceiling,
        audited_split=MDE_AUDITED_SPLIT,
        supply_audit_path=str(supply_audit_path),
    )
    print(
        f"preflight: projected MDE = {mde['value']:.4f} (ceiling {mde_ceiling}) at n="
        f"{audited_matched_pairs} ({MDE_AUDITED_SPLIT}) -> {'PASS' if mde['pass'] else 'FAIL'}",
        flush=True,
    )

    bi_pairs = gue_pairs[: min(BATCH_INVARIANCE_MAX_PAIRS, len(gue_pairs))]
    batch_of_1 = {row["pair_id"]: row for row in native_pair_rows(bundle, bi_pairs, batch_pairs=1)}
    batch_of_n = {
        row["pair_id"]: row for row in native_pair_rows(bundle, bi_pairs, batch_pairs=len(bi_pairs))
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

    checks = {
        "M1_identity": bool(m1["pass"]),
        "M2_caliper_respected": bool(m2["pass"]),
        "M3_equal_dose": bool(m3["pass"]),
        "M4_region_containment": bool(m4["pass"]),
        "planted_positive": bool(planted["pass"]),
        "permuted_label_fpr": bool(fpr["pass"]),
        "projected_MDE": bool(mde["pass"]),
        "batch_invariance": bool(batch_invariance_pass),
    }
    qualified = all(checks.values())
    decision = "QUALIFIED" if qualified else "INSTRUMENT_UNRESOLVED"
    failing_checks = sorted(name for name, value in checks.items() if not value)

    assay_version = str(config["assay"]).rsplit("_", 1)[-1]
    by_locus = [
        {
            "pair_id": r["pair_id"],
            "matched_gc_separation": r["matched"].gc_separation if r["matched"] else None,
            "matched_cpg_separation": r["matched"].cpg_separation if r["matched"] else None,
            "matched_polarity": r["matched_polarity"],
            "unmatched_gc_separation": r["unmatched"].gc_separation if r["unmatched"] else None,
            "unmatched_polarity": r["unmatched_polarity"],
            "matched_swapped_gc_separation": (
                r["matched_swapped"].gc_separation if r["matched_swapped"] else None
            ),
            "matched_polarity_swapped": r["matched_polarity_swapped"],
            "zero_mask_target_specific_loss": r["zero_mask_target_specific_loss"],
        }
        for r in locus_results
    ]

    input_paths = (args.config, str(supply_audit_path), gue_test_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": f"promoter_composition_preflight_{assay_version}",
        "decision": decision,
        "smoke": bool(args.smoke),
        "config_sha256": config_hash,
        "checkpoint_revision": identity["revision"],
        "environment": identity["env"],
        "tss_index": tss_index,
        "input_sha256": {
            "config": sha256_file(args.config),
            "supply_audit": sha256_file(str(supply_audit_path)),
            "gue_test": sha256_file(gue_test_path),
        },
        "gue_pair_offset": int(args.gue_pair_offset),
        "gue_pairs_requested": int(args.gue_pairs),
        "gue_pair_ids": [pair_id for pair_id, _, _ in gue_pairs],
        "checks": checks,
        "failing_checks": failing_checks,
        "checks_architecture_forced": {
            "M1_identity": {
                "class": "empirically_falsifiable",
                "why": (
                    "Exercises the same in-place token-substitution function the endpoints "
                    "use, with a boundary-case empty index list. Catches a bug where an empty "
                    "index list is mishandled (e.g. a fallback that masks everything, or an "
                    "index error). Scope is narrower than a general regression guard on the "
                    "masking path: the masking loop body never executes when there is nothing "
                    "to mask, so this specific check cannot exercise the aliasing/off-by-one "
                    "bug classes only a non-empty index list would reveal."
                ),
            },
            "M2_caliper_respected": {
                "class": "architecture_forced",
                "why": (
                    "select_matched_pair/select_unmatched_pair only ever return a candidate "
                    "that already clears the caliper/separation filter internally -- PASS is a "
                    "regression guard on this script's own wiring (did it pass the config's "
                    "actual caliper values), not independent evidence about the selected pairs."
                ),
            },
            "M3_equal_dose": {
                "class": "architecture_forced",
                "why": (
                    "paired_fixed_dose_tile only ever returns a tile at exactly token_dose "
                    "adjacent genomic tokens within maximum_cover_length_difference_nt, or "
                    "returns None -- no cover this script could score can violate M3."
                ),
            },
            "M4_region_containment": {
                "class": "architecture_forced",
                "why": (
                    "enumerate_region_covers only ever yields a cover whose centre already "
                    "sits inside [region_start, region_end] -- no cover this script could "
                    "score can violate M4."
                ),
            },
            "planted_positive": {
                "class": "empirically_falsifiable",
                "why": (
                    "The load-bearing sensitivity check: whether the unmatched arm's polarity "
                    "genuinely exceeds the matched arm's by the P2 SESOI, from real forward "
                    "passes over real (if non-biological) GUE sequences. A sibling instrument "
                    "(IF30) failed an analogous calibration and was retired rather than trusted."
                ),
            },
            "permuted_label_fpr": {
                "class": "empirically_falsifiable",
                "why": (
                    "Re-derives, per locus, a genuinely distinct role-swapped selection+scoring "
                    "outcome (see _check_permuted_label_fpr's docstring) rather than sign-"
                    "flipping an already-fixed number -- enumerate_region_covers' centre/"
                    "containment filter is evaluated against whichever sequence plays "
                    "'target' for a given state, so the swapped-role cover set is not simply "
                    "the negation of the true one and can genuinely differ from it."
                ),
            },
            "projected_MDE": {
                "class": "empirically_falsifiable",
                "why": (
                    "A real arithmetic gate against the already-audited, outcome-blind "
                    "matched-pair count; failed for the superseded v1 config (MDE 0.323 "
                    "against a 0.28 ceiling at development n=122) before Amendment 1 raised "
                    "the raw-pair target."
                ),
            },
            "batch_invariance": {
                "class": "empirically_falsifiable",
                "why": "Would fail on any real batch-dependent numerical drift.",
            },
        },
        "measurements": {
            "M1_identity": m1,
            "M2_caliper_respected": m2,
            "M3_equal_dose": m3,
            "M4_region_containment": m4,
            "planted_positive": planted,
            "permuted_label_fpr": fpr,
            "projected_MDE": mde,
            "batch_invariance": {
                "pass": batch_invariance_pass,
                "tolerance": BATCH_INVARIANCE_TOLERANCE,
                "max_absolute_margin_difference": batch_invariance_max_diff,
                "n_pairs": len(bi_pairs),
                "batch_sizes_compared": [1, len(bi_pairs)],
            },
        },
        "by_locus": by_locus,
        "boundary": (
            "Qualifies the composition measurement pipeline (memorials, planted-positive and "
            "permuted-label-FPR calibration, projected MDE, batch invariance) on a reserved GUE "
            "prom_300_all slice disjoint from every prior D8/D11/D12/D13 slice. Computes no "
            "outcome on the D13 composition cohort (development or confirmation) and licenses "
            "no P1/P2/P3 conclusion; a QUALIFIED decision here means only that development-"
            "split scoring may proceed, per the config's own split_gate."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=f"dnabert2_promoter_composition_preflight_{assay_version}",
        status="PREFLIGHT" if qualified else "INSTRUMENT_UNRESOLVED",
        model_identity=identity,
        input_paths=input_paths,
        parameters={
            "config_sha256": config_hash,
            "smoke": bool(args.smoke),
            "batch_pairs": int(args.batch_pairs),
            "tss_index": tss_index,
            "nominal_width_nt": NOMINAL_WIDTH_NT,
            "permuted_label_draws": PERMUTED_LABEL_DRAWS,
            "permuted_label_fpr_tolerance": PERMUTED_LABEL_FPR_TOLERANCE,
            "matched_arm_sd": MATCHED_ARM_SD,
            "mde_z_multiplier": MDE_Z_MULTIPLIER,
            "audited_matched_pairs": audited_matched_pairs,
            "audited_split": MDE_AUDITED_SPLIT,
            "batch_invariance_tolerance": BATCH_INVARIANCE_TOLERANCE,
        },
        seeds={"gue_selection": inference_seed, "permuted_label_fpr": inference_seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _assert_calibration_prose_matches(config: dict[str, Any]) -> None:
    """Defensive conformance check for the calibration numbers that live only in
    the config's prose, not as their own structured fields (PERMUTED_LABEL_DRAWS,
    PERMUTED_LABEL_FPR_TOLERANCE, MATCHED_ARM_SD, MDE_Z_MULTIPLIER) -- catches a
    future config edit that changes these numbers in prose without updating this
    script's constants to match."""

    fpr_check = str(config["calibration"]["permuted_label_fpr"]["check"])
    if str(PERMUTED_LABEL_DRAWS) not in fpr_check or str(PERMUTED_LABEL_FPR_TOLERANCE) not in fpr_check:
        raise RuntimeError(
            "calibration.permuted_label_fpr.check prose no longer matches this script's "
            f"PERMUTED_LABEL_DRAWS={PERMUTED_LABEL_DRAWS}/"
            f"PERMUTED_LABEL_FPR_TOLERANCE={PERMUTED_LABEL_FPR_TOLERANCE} -- update both"
        )
    supersession = str(config.get("supersession_reason", ""))
    if str(MATCHED_ARM_SD) not in supersession or str(MDE_Z_MULTIPLIER) not in supersession:
        raise RuntimeError(
            f"config supersession_reason no longer mentions sd={MATCHED_ARM_SD} and/or "
            f"z={MDE_Z_MULTIPLIER} -- update MATCHED_ARM_SD/MDE_Z_MULTIPLIER to match the "
            "config's own derivation"
        )


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


def _margin_cache(bundle):
    """A cached single-sequence margin(). Clean margins are reused across M1, the
    matched/unmatched/role-swapped polarity computations, and the permuted-label
    FPR check for the same locus, bounding the total forward-pass count instead of
    recomputing the same clean margin many times over. Caching does not change
    which bugs M1 can or cannot catch (see the M1 scope note in the module
    docstring): a deterministic eval-mode forward pass on identical token ids
    gives an identical result whether or not it is served from cache."""

    import torch

    model, device = bundle.model, bundle.device
    cache: dict[tuple[int, ...], float] = {}

    def margin(ids: list[int]) -> float:
        key = tuple(ids)
        cached = cache.get(key)
        if cached is not None:
            return cached
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(input_ids=t, attention_mask=torch.ones_like(t))
        logits = out.logits[0]
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite preflight logits")
        value = float(logits[1] - logits[0])
        cache[key] = value
        return value

    return margin, cache


def _masked_ids(ids: list[int], indices, mask_id: int) -> list[int]:
    masked = list(ids)
    for index in indices:
        masked[int(index)] = mask_id
    return masked


def _target_specific_loss(margin, mask_id: int, ids_a, indices_a, ids_b, indices_b) -> float:
    masked_a = _masked_ids(ids_a, indices_a, mask_id)
    masked_b = _masked_ids(ids_b, indices_b, mask_id)
    return (margin(ids_a) - margin(masked_a)) - (margin(ids_b) - margin(masked_b))


def _enumerate_both_regions(
    target_offsets,
    control_offsets,
    target_seq: str,
    control_seq: str,
    *,
    tss_index: int,
    downstream_region: tuple[int, int],
    upstream_region: tuple[int, int],
    nominal_width: int,
    token_dose: int,
    maximum_cover_nt: int,
    maximum_cover_diff_nt: int,
):
    downstream = enumerate_region_covers(
        target_offsets,
        control_offsets,
        target_seq,
        control_seq,
        region_start=downstream_region[0],
        region_end=downstream_region[1],
        tss_index=tss_index,
        nominal_width=nominal_width,
        token_dose=token_dose,
        maximum_cover_nt=maximum_cover_nt,
        maximum_cover_length_difference_nt=maximum_cover_diff_nt,
    )
    upstream = enumerate_region_covers(
        target_offsets,
        control_offsets,
        target_seq,
        control_seq,
        region_start=upstream_region[0],
        region_end=upstream_region[1],
        tss_index=tss_index,
        nominal_width=nominal_width,
        token_dose=token_dose,
        maximum_cover_nt=maximum_cover_nt,
        maximum_cover_length_difference_nt=maximum_cover_diff_nt,
    )
    return downstream, upstream


def _select_all_loci(
    bundle,
    pairs: list[Pair],
    *,
    tss_index: int,
    downstream_region: tuple[int, int],
    upstream_region: tuple[int, int],
    nominal_width: int,
    token_dose: int,
    maximum_cover_nt: int,
    maximum_cover_diff_nt: int,
    caliper_gc: float,
    caliper_cpg: float,
    min_unmatched_sep: float,
    batch_pairs: int,
) -> list[dict[str, Any]]:
    """Phase 1 -- outcome-blind. Tokenizes every locus and selects its matched,
    unmatched, and role-swapped cover pairs using only sequence composition and
    BPE token geometry (gfmcd.reconstruction_atlas.promoter_composition's
    committed, tested selectors -- not reimplemented here). tokenize_promoter_pairs
    calls only bundle.tokenizer; bundle.model is never invoked in this function, so
    a reviewer can confirm selection could not have been steered by any model
    score. Every value Phase 2 (_score_all_loci) needs is returned; Phase 2
    performs no further selection.

    The "role-swapped" cover pair (matched_swapped) re-enumerates with target and
    control offsets/sequences exchanged, reusing this locus's own tokenization for
    both roles (never retokenizing) -- see _check_permuted_label_fpr for why this
    is not simply the negation of the true-labeled selection.
    """

    selections: list[dict[str, Any]] = []
    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start : start + batch_pairs]
        batch = tokenize_promoter_pairs(bundle, chunk)
        for index, (pair_id, target_seq, control_seq) in enumerate(chunk):
            target_count = batch.token_counts[2 * index]
            control_count = batch.token_counts[2 * index + 1]
            target_ids = batch.encoded["input_ids"][2 * index, :target_count].tolist()
            control_ids = batch.encoded["input_ids"][2 * index + 1, :control_count].tolist()
            target_offsets = batch.offset_mapping[2 * index, :target_count].tolist()
            control_offsets = batch.offset_mapping[2 * index + 1, :control_count].tolist()

            downstream, upstream = _enumerate_both_regions(
                target_offsets,
                control_offsets,
                target_seq,
                control_seq,
                tss_index=tss_index,
                downstream_region=downstream_region,
                upstream_region=upstream_region,
                nominal_width=nominal_width,
                token_dose=token_dose,
                maximum_cover_nt=maximum_cover_nt,
                maximum_cover_diff_nt=maximum_cover_diff_nt,
            )
            matched = select_matched_pair(downstream, upstream, caliper_gc=caliper_gc, caliper_cpg=caliper_cpg)
            unmatched = select_unmatched_pair(downstream, upstream, minimum_gc_separation=min_unmatched_sep)

            downstream_sw, upstream_sw = _enumerate_both_regions(
                control_offsets,
                target_offsets,
                control_seq,
                target_seq,
                tss_index=tss_index,
                downstream_region=downstream_region,
                upstream_region=upstream_region,
                nominal_width=nominal_width,
                token_dose=token_dose,
                maximum_cover_nt=maximum_cover_nt,
                maximum_cover_diff_nt=maximum_cover_diff_nt,
            )
            matched_swapped = select_matched_pair(
                downstream_sw, upstream_sw, caliper_gc=caliper_gc, caliper_cpg=caliper_cpg
            )

            selections.append(
                {
                    "pair_id": pair_id,
                    "target_ids": target_ids,
                    "control_ids": control_ids,
                    "matched": matched,
                    "unmatched": unmatched,
                    "matched_swapped": matched_swapped,
                }
            )
        print(
            f"preflight: selected covers for {min(start + len(chunk), len(pairs))}/{len(pairs)} loci",
            flush=True,
        )
    return selections


def _polarity(margin, mask_id: int, pair: CoverPair | None, ids_a: list[int], ids_b: list[int]):
    if pair is None:
        return None
    down = _target_specific_loss(
        margin, mask_id, ids_a, pair.downstream.target.token_indices, ids_b, pair.downstream.control.token_indices
    )
    up = _target_specific_loss(
        margin, mask_id, ids_a, pair.upstream.target.token_indices, ids_b, pair.upstream.control.token_indices
    )
    return down - up


def _score_all_loci(selections: list[dict[str, Any]], *, margin, mask_id: int) -> list[dict[str, Any]]:
    """Phase 2 -- scoring only. Consumes Phase 1's already-selected CoverPair
    objects; performs no cover selection of its own. Every masked-margin forward
    pass this preflight makes happens in this function (via margin/
    _target_specific_loss/_polarity)."""

    results: list[dict[str, Any]] = []
    for selection in selections:
        target_ids = selection["target_ids"]
        control_ids = selection["control_ids"]
        matched = selection["matched"]
        unmatched = selection["unmatched"]
        matched_swapped = selection["matched_swapped"]

        matched_polarity = _polarity(margin, mask_id, matched, target_ids, control_ids)
        unmatched_polarity = _polarity(margin, mask_id, unmatched, target_ids, control_ids)
        # Roles exchanged: control_ids plays "target" (ids_a), target_ids plays
        # "control" (ids_b), matching how matched_swapped's own covers were built
        # (target_offsets=control_offsets, control_offsets=target_offsets) in
        # _select_all_loci.
        matched_polarity_swapped = _polarity(margin, mask_id, matched_swapped, control_ids, target_ids)
        zero_mask_tsl = _target_specific_loss(margin, mask_id, target_ids, (), control_ids, ())

        results.append(
            {
                "pair_id": selection["pair_id"],
                "matched": matched,
                "unmatched": unmatched,
                "matched_swapped": matched_swapped,
                "matched_polarity": matched_polarity,
                "unmatched_polarity": unmatched_polarity,
                "matched_polarity_swapped": matched_polarity_swapped,
                "zero_mask_target_specific_loss": zero_mask_tsl,
            }
        )
    return results


def _check_m1(locus_results: list[dict[str, Any]], *, tolerance: float) -> dict[str, Any]:
    worst_value = 0.0
    worst_pair_id: str | None = None
    for row in locus_results:
        value = abs(float(row["zero_mask_target_specific_loss"]))
        if value > worst_value:
            worst_value = value
            worst_pair_id = row["pair_id"]
    return {
        "pass": worst_value < tolerance,
        "tolerance": tolerance,
        "worst_absolute_value": worst_value,
        "worst_pair_id": worst_pair_id,
        "n_rows": len(locus_results),
    }


def _check_m2(
    locus_results: list[dict[str, Any]], *, caliper_gc: float, caliper_cpg: float, min_unmatched_sep: float
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checked = 0
    for row in locus_results:
        matched = row["matched"]
        if matched is not None:
            checked += 1
            if matched.gc_separation > caliper_gc or matched.cpg_separation > caliper_cpg:
                violations.append(
                    {
                        "pair_id": row["pair_id"],
                        "arm": "matched",
                        "gc_separation": matched.gc_separation,
                        "cpg_separation": matched.cpg_separation,
                    }
                )
        matched_sw = row["matched_swapped"]
        if matched_sw is not None:
            checked += 1
            if matched_sw.gc_separation > caliper_gc or matched_sw.cpg_separation > caliper_cpg:
                violations.append(
                    {
                        "pair_id": row["pair_id"],
                        "arm": "matched_swapped",
                        "gc_separation": matched_sw.gc_separation,
                        "cpg_separation": matched_sw.cpg_separation,
                    }
                )
        unmatched = row["unmatched"]
        if unmatched is not None:
            checked += 1
            if unmatched.gc_separation < min_unmatched_sep:
                violations.append(
                    {"pair_id": row["pair_id"], "arm": "unmatched", "gc_separation": unmatched.gc_separation}
                )
    return {
        "pass": checked > 0 and not violations,
        "checked": checked,
        "violations": violations,
        "caliper_gc": caliper_gc,
        "caliper_cpg": caliper_cpg,
        "minimum_unmatched_gc_separation": min_unmatched_sep,
    }


def _check_m3(
    locus_results: list[dict[str, Any]], *, token_dose: int, maximum_cover_diff_nt: int
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checked = 0
    for row in locus_results:
        for arm_name in ("matched", "unmatched", "matched_swapped"):
            pair = row[arm_name]
            if pair is None:
                continue
            for side_name, cover in (("downstream", pair.downstream), ("upstream", pair.upstream)):
                checked += 1
                ok = (
                    len(cover.target.token_indices) == token_dose
                    and len(cover.control.token_indices) == token_dose
                    and abs(cover.target.cover_length - cover.control.cover_length) <= maximum_cover_diff_nt
                )
                if not ok:
                    violations.append(
                        {
                            "pair_id": row["pair_id"],
                            "arm": arm_name,
                            "side": side_name,
                            "target_tokens": len(cover.target.token_indices),
                            "control_tokens": len(cover.control.token_indices),
                            "target_cover_length": cover.target.cover_length,
                            "control_cover_length": cover.control.cover_length,
                        }
                    )
    return {
        "pass": checked > 0 and not violations,
        "checked": checked,
        "violations": violations,
        "token_dose": token_dose,
        "maximum_cover_length_difference_nt": maximum_cover_diff_nt,
    }


def _check_m4(
    locus_results: list[dict[str, Any]],
    *,
    tss_index: int,
    downstream_region: tuple[int, int],
    upstream_region: tuple[int, int],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checked = 0
    for row in locus_results:
        for arm_name in ("matched", "unmatched", "matched_swapped"):
            pair = row[arm_name]
            if pair is None:
                continue
            for side_name, cover, region in (
                ("downstream", pair.downstream, downstream_region),
                ("upstream", pair.upstream, upstream_region),
            ):
                checked += 1
                centre = (cover.target.cover_start + cover.target.cover_end) / 2 - tss_index
                if not (region[0] <= centre <= region[1]):
                    violations.append(
                        {
                            "pair_id": row["pair_id"],
                            "arm": arm_name,
                            "side": side_name,
                            "centre": centre,
                            "region": list(region),
                        }
                    )
    return {
        "pass": checked > 0 and not violations,
        "checked": checked,
        "violations": violations,
        "downstream_region": list(downstream_region),
        "upstream_region": list(upstream_region),
    }


def _check_planted_positive(locus_results: list[dict[str, Any]], *, p2_sesoi: float) -> dict[str, Any]:
    both_arms = [
        row for row in locus_results if row["matched_polarity"] is not None and row["unmatched_polarity"] is not None
    ]
    if not both_arms:
        return {
            "pass": False,
            "matched_arm_polarity_mean": None,
            "unmatched_arm_polarity_mean": None,
            "difference": None,
            "p2_sesoi": p2_sesoi,
            "n_both_arms_loci": 0,
            "both_arms_pair_ids": [],
            "note": "no reserved locus admitted both a matched and an unmatched cover pair",
        }
    matched_mean = mean(row["matched_polarity"] for row in both_arms)
    unmatched_mean = mean(row["unmatched_polarity"] for row in both_arms)
    difference = unmatched_mean - matched_mean
    return {
        "pass": difference >= p2_sesoi,
        "matched_arm_polarity_mean": matched_mean,
        "unmatched_arm_polarity_mean": unmatched_mean,
        "difference": difference,
        "p2_sesoi": p2_sesoi,
        "n_both_arms_loci": len(both_arms),
        "both_arms_pair_ids": [row["pair_id"] for row in both_arms],
    }


def _check_permuted_label_fpr(
    locus_results: list[dict[str, Any]], *, seed: int, draws: int, tolerance: float
) -> dict[str, Any]:
    """Interpretive design, disclosed explicitly: the frozen config states the
    desired OUTCOME check ("with target/control labels permuted 256 times, the
    matched-arm polarity center must fall within 0.05 of zero") but not the
    resampling mechanics, so this is a documented judgment call, not a literal
    transcription of a fully-specified algorithm.

    For each reserved locus, Phase 1/2 already computed TWO independently
    re-derived selection+scoring outcomes: ``matched_polarity`` (target/control in
    their true roles) and ``matched_polarity_swapped`` (roles fully exchanged,
    each sequence using its OWN tokenization -- see _select_all_loci). This is
    deliberately NOT a sign-flip resampling of one fixed number: a per-locus,
    per-draw sign flip of an already-computed vector would be mathematically
    guaranteed to center near zero for ANY input, by the symmetry of a random
    sign draw, regardless of whether the underlying measurement pipeline is
    biased -- an architecture-forced check that could only catch a bug in the
    resampling code itself, never in the measurement. Because
    enumerate_region_covers' centre/containment filter is evaluated against
    whichever sequence plays "target" for a given state, the swapped-role cover
    set can genuinely differ from the true one (different nominal positions can
    pass or fail containment depending on which sequence's own BPE tokenization
    is being checked) -- so matched_polarity_swapped is not simply
    -matched_polarity, and this check can fail on a real pipeline bias.

    Each of the `draws` draws independently, per locus, chooses the true or the
    role-swapped outcome with equal probability (falling back to whichever value
    exists if only one of the two was selectable for that locus). The check is
    the mean, across draws, of each draw's per-locus mean -- "the matched-arm
    polarity center" under label permutation.
    """

    available = [
        row for row in locus_results if row["matched_polarity"] is not None or row["matched_polarity_swapped"] is not None
    ]
    if not available:
        return {
            "pass": False,
            "center": None,
            "tolerance": tolerance,
            "draws": draws,
            "n_loci_available": 0,
            "seed": seed,
            "note": "no reserved locus admitted a matched pair in either labeling",
        }
    rng = random.Random(seed)
    draw_means: list[float] = []
    for _ in range(draws):
        values: list[float] = []
        for row in available:
            true_value = row["matched_polarity"]
            swapped_value = row["matched_polarity_swapped"]
            if true_value is not None and swapped_value is not None:
                values.append(true_value if rng.getrandbits(1) else swapped_value)
            elif true_value is not None:
                values.append(true_value)
            else:
                values.append(swapped_value)
        draw_means.append(mean(values))
    center = mean(draw_means)
    return {
        "pass": abs(center) <= tolerance,
        "center": center,
        "tolerance": tolerance,
        "draws": draws,
        "n_loci_available": len(available),
        "seed": seed,
        "draw_mean_min": min(draw_means),
        "draw_mean_max": max(draw_means),
    }


def _check_projected_mde(
    *,
    audited_matched_pairs: int,
    sd: float,
    z_multiplier: float,
    ceiling: float,
    audited_split: str,
    supply_audit_path: str,
) -> dict[str, Any]:
    if audited_matched_pairs < 1:
        raise ValueError("audited matched-pair count must be positive")
    value = z_multiplier * sd / math.sqrt(audited_matched_pairs)
    return {
        "pass": value <= ceiling,
        "value": value,
        "ceiling": ceiling,
        "matched_arm_sd": sd,
        "z_multiplier": z_multiplier,
        "audited_matched_pairs": audited_matched_pairs,
        "audited_split": audited_split,
        "supply_audit_source": supply_audit_path,
    }


if __name__ == "__main__":
    main()
