#!/usr/bin/env python
"""PBC v2 successor capacity/power ladder -- drives the four model-free analysis layers in
``gfmcd.splicing.pbc_v2_capacity`` (committed at 6d84ee4b, tests green) across the fixed rung
ladder ``N in (600, 660, 729, 780, 819, 864, 909)`` and lands one immutable artifact.

Stage: ``construction``. This runner never imports ``torch`` or a model adapter and never loads a
SpliceBERT checkpoint -- ``model_execution``, ``checkpoint_loaded`` and ``cohort_selection_attempted``
are recorded as honesty fields in the artifact, not merely implied by omission. It performs NO cohort
selection, NO block assembly and NO cell materialisation; the output carries counts, hashes, verdicts
and projections only -- never a gene ID.

**The study directory name ``splicebert_positional_clock`` is PROVISIONAL** and may be renamed before
anything freezes (see ``results/splicebert_positional_clock/RESULTS_INDEX.md``). The artifact this
runner produces lives under ``development/`` for exactly that reason: a later rename costs a move, not
a retraction.

Two inputs, both read-only:

* the hash-pinned candidate census (``candidate_census_v1.json``, 2.34 GB, an INPUT that lives outside
  this repo and is never copied in -- streamed exactly once via ``stream_candidates``);
* the committed capacity certificate (``composition_balanced_cohort_v1.json``) that both supplies the
  ``slot_gene_sets`` filter configuration (mirroring
  ``tests/test_pbc_v2_capacity_census.py::_filters_from_artifact`` -- see ``_filters_from_certificate``
  below) and the seven per-slot ``gene_ids_sha256`` values the streamed pool must reproduce.

This module is read-only with respect to PBC v1: it never writes into
``studies/splicebert_positional_boundary_coherence/`` or
``results/splicebert_positional_boundary_coherence/``; that study is closed.

Before any rung is evaluated, the streamed slot sets are checked against the certificate's seven
``gene_ids_sha256`` hashes via ``verify_slot_hashes`` and the run hard-fails on any mismatch -- a run
that proceeded on an unverified gene pool would be worthless, so this is a gate, not a log line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.pbc_v2_capacity import (  # noqa: E402
    LEGACY_RESIDUAL_VARIANCE_UPPER_BOUND_DEV120,
    SLOT_ORDER,
    allocate_blocks,
    feasibility,
    necessary_counts,
    power_projection,
    rotated_demands,
    slot_gene_sets,
    stream_candidates,
    verify_slot_hashes,
)
from gfmcd.splicing.positional_boundary_coherence import (  # noqa: E402
    DEFAULT_ACCEPTOR_WINDOW,
    DEFAULT_DONOR_WINDOW,
    normalize_chromosome,
)

# ---------------------------------------------------------------------------------------------
# Fixed design constants -- these are the brief's numbers, not tunables discovered at run time.
# ---------------------------------------------------------------------------------------------

SCHEMA = "gfmcd.splicebert_positional_clock.pbc_v2_capacity_ladder.v2"
EXPERIMENT_ID = "splicebert-positional-clock:pbc-v2-capacity-ladder-v1"

LADDER: tuple[int, ...] = (600, 660, 729, 780, 819, 864, 909)
MARGIN_THRESHOLD = 0.25
DEFAULT_PAIR_WEIGHTS: tuple[int, ...] = (1, 1, 1)
DEFAULT_LATTICE = "rotated_two_by_two"

# Recorded verbatim in every artifact so no reader has to infer the decision rule from the code.
SELECTION_RULE = (
    "the largest N whose every slot and every union clears a margin of 0.25 AND whose exact "
    "feasibility passes AND whose Psi verdict is 'powered_for_every_rho'"
)

# A tiny, fully synthetic ladder for --smoke: small enough to be instantaneous, still exercising a
# mix of margin-pass/fail, feasible/infeasible, and Hall-obstruction outcomes.
SMOKE_LADDER: tuple[int, ...] = (6, 9, 12, 15)

DEFAULT_CENSUS_PATH = Path(
    "/Users/cberg/Research/projects/gfm-circuit-discovery-pbc-successor/results/"
    "splicebert_positional_boundary_coherence/development/candidate_census_v1.json"
)
DEFAULT_CERTIFICATE_PATH = (
    REPO / "results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.json"
)
# `--out` has no default (see `parse_args`); the canonical construction-stage path is
# results/splicebert_positional_clock/development/pbc_v2_capacity_ladder_v2.json (v1 is
# superseded -- see results/splicebert_positional_clock/superseded/PBC_V2_CAPACITY_LADDER_V1.SUPERSEDED.md).


# ---------------------------------------------------------------------------------------------
# Pure ladder logic -- no filesystem access, callable directly from tests with injected slot sets.
# ---------------------------------------------------------------------------------------------


def _serialize_allocation(allocation: Mapping[tuple[int, int], int]) -> list[dict]:
    """``allocate_blocks``'s tuple-keyed dict, in a JSON-safe, order-preserving shape."""

    return [
        {"phase_pair": [int(pair[0]), int(pair[1])], "blocks": int(count)}
        for pair, count in allocation.items()
    ]


_NOT_JOINTLY_REALIZABLE_NOTE = (
    "available - demand per slot/union, exactly as necessary_counts' own margin column defines it. "
    "This double-counts genes shared across slots (e.g. the donor_phase_2/acceptor_phase_2 overlap "
    "in the real pool) -- it is NOT a jointly realizable holdout. See `joint_residuals` for the "
    "max-flow-exact figure: the genes actually left over once this rung's demand is satisfied."
)


def _entries_with_holdout(counts: dict) -> dict[str, dict]:
    """``necessary_counts``'s per-slot/union entries, plus the (non-jointly-realizable) residual.

    ``margin`` is already defined once, in ``pbc_v2_capacity._capacity_row``, and is not redefined
    here. ``residual_unique_genes_per_slot_not_jointly_realizable`` is a plain integer count --
    ``available - demand`` -- so the power / capacity-margin / future-holdout trade-off the brief
    asks for is visible in the same table as the margin fraction, without touching how margin
    itself is computed. Per its own name (and `_NOT_JOINTLY_REALIZABLE_NOTE`, carried alongside it
    in the row this feeds), it double-counts genes shared across slots and must not be read as a
    jointly buildable holdout -- that figure is `joint_residuals`.
    """

    entries: dict[str, dict] = {}
    for name, entry in counts["entries"].items():
        row = dict(entry)
        row["residual_unique_genes_per_slot_not_jointly_realizable"] = int(
            entry["available_unique_genes"] - entry["demand_unique_genes"]
        )
        entries[name] = row
    return entries


def _joint_residuals(slot_sets: Mapping[str, set], demands: Mapping[str, int], feas: dict) -> dict:
    """Genuinely joint (max-flow-exact) leftover-gene figures, replacing the double-counted ones.

    ``feas["max_flow"]`` is the number of genes the exact solver actually spent to satisfy this
    rung's demand (every gene->source edge has capacity 1, so a gene is used at most once in the
    flow), so ``|union of the seven real slots| - max_flow`` is the number of genes left unassigned
    *overall* -- correct even when the rung is infeasible, where the naive available-minus-demand
    sum undercounts leftovers by exactly the shortfall (unmet demand leaves its genes unspent too).

    The ``{donor_phase_2, acceptor_phase_2}`` pair gets its own two-slot figure --
    ``|union of the pair| - (demand_d2 + demand_a2)`` -- because that pair is this pool's tightest
    real Hall-type obstruction (see `feasibility()`'s `min_cut_slots` for N in [660, 864]).
    """

    all_union = set().union(*(slot_sets[slot] for slot in SLOT_ORDER))
    pair_union = slot_sets["donor_phase_2"] | slot_sets["acceptor_phase_2"]
    pair_demand = int(demands["donor_phase_2"]) + int(demands["acceptor_phase_2"])
    return {
        "unassigned_overall": len(all_union) - int(feas["max_flow"]),
        "donor_phase_2_acceptor_phase_2_pair": len(pair_union) - pair_demand,
    }


def _rung_row(
    n_blocks: int,
    slot_sets: Mapping[str, set],
    *,
    margin_threshold: float,
    pair_weights: tuple[int, ...],
    lattice: str,
) -> dict:
    """One ladder row: allocation, demand, necessary counts (+ residuals), exact feasibility, power."""

    allocation = allocate_blocks(n_blocks, pair_weights=pair_weights)
    demands = rotated_demands(n_blocks, pair_weights=pair_weights)
    counts = necessary_counts(slot_sets, demands)
    entries = _entries_with_holdout(counts)

    real_demands = {slot: demands[slot] for slot in SLOT_ORDER}
    feas = feasibility(slot_sets, real_demands)
    proj = power_projection(n_blocks, lattice=lattice)
    psi_verdict = proj["psi"]["verdict"]
    joint_residuals = _joint_residuals(slot_sets, demands, feas)

    margin_threshold_met = all(
        entry["margin"] is not None and entry["margin"] >= margin_threshold
        for entry in entries.values()
    )
    selection_candidate = bool(
        margin_threshold_met and feas["feasible"] and psi_verdict == "powered_for_every_rho"
    )

    return {
        "n_blocks": int(n_blocks),
        "block_allocation": _serialize_allocation(allocation),
        "demand": {slot: int(value) for slot, value in demands.items()},
        "necessary_counts": {
            "entries": entries,
            "passed": counts["passed"],
            "failed_constraints": counts["failed_constraints"],
            "residual_unique_genes_per_slot_not_jointly_realizable_note": _NOT_JOINTLY_REALIZABLE_NOTE,
        },
        "joint_residuals": joint_residuals,
        "feasibility": feas,
        "power_projection": proj,
        "psi_verdict": psi_verdict,
        "margin_threshold_met": margin_threshold_met,
        "selection_candidate": selection_candidate,
    }


def _no_feasible_rung_reason(rows: list[dict], *, margin_threshold: float) -> str:
    """Say which condition(s) of the selection rule bound, using only this run's own rows.

    Generic on purpose: it works the same way whether NO rung ever clears margin, NO rung is
    exactly feasible, NO rung is powered, or (the case this ladder's real run actually hits) every
    condition is individually satisfiable somewhere on the ladder but never by the same rung.
    """

    margin_rows = [row for row in rows if row["margin_threshold_met"]]
    feasible_rows = [row for row in rows if row["feasibility"]["feasible"]]
    powered_rows = [row for row in rows if row["psi_verdict"] == "powered_for_every_rho"]

    reasons: list[str] = []
    if not margin_rows:
        reasons.append(f"no ladder rung clears a {margin_threshold} margin on every slot and union")
    if not feasible_rows:
        reasons.append("no ladder rung passes exact feasibility")
    if not powered_rows:
        reasons.append("no ladder rung reaches a powered_for_every_rho Psi verdict")

    if not reasons:
        feasible_max = max(feasible_rows, key=lambda row: row["n_blocks"])
        powered_min = min(powered_rows, key=lambda row: row["n_blocks"])
        powered_min_feas = powered_min["feasibility"]
        if len(margin_rows) == len(rows):
            margin_clause = f"margin >= {margin_threshold} holds on every evaluated rung"
        else:
            margin_ns = sorted(row["n_blocks"] for row in margin_rows)
            margin_clause = (
                f"margin >= {margin_threshold} holds on {len(margin_rows)} of {len(rows)} "
                f"evaluated rungs (N={margin_ns})"
            )
        reasons.append(
            f"{margin_clause}, but exact feasibility and the powered_for_every_rho Psi verdict "
            "are each individually satisfiable on this ladder and never by the same rung: the "
            f"largest exactly-feasible rung is N={feasible_max['n_blocks']} (Psi verdict "
            f"{feasible_max['psi_verdict']!r}); the smallest rung with a powered_for_every_rho "
            f"Psi verdict is N={powered_min['n_blocks']} (exact feasibility "
            f"{'passes' if powered_min_feas['feasible'] else 'fails'}, "
            f"shortfall={powered_min_feas['shortfall']})"
        )
    return "; ".join(reasons)


def build_ladder(
    slot_sets: Mapping[str, set],
    *,
    ladder: tuple[int, ...] = LADDER,
    margin_threshold: float = MARGIN_THRESHOLD,
    pair_weights: tuple[int, ...] = DEFAULT_PAIR_WEIGHTS,
    lattice: str = DEFAULT_LATTICE,
) -> dict:
    """Pure function: one row per rung, plus the top-level selection.

    Takes only the seven real slot -> gene-set mapping; never touches the filesystem, so it is
    directly testable with injected synthetic slot sets. ``feasibility`` is called with the seven
    real slots only (union pseudo-slots are never passed to it, per its own contract).
    """

    rows = [
        _rung_row(
            n_blocks,
            slot_sets,
            margin_threshold=margin_threshold,
            pair_weights=pair_weights,
            lattice=lattice,
        )
        for n_blocks in ladder
    ]

    candidates = [row for row in rows if row["selection_candidate"]]
    if candidates:
        selected_row = max(candidates, key=lambda row: row["n_blocks"])
        selected_rung: int | None = selected_row["n_blocks"]
        no_feasible_rung_reason: str | None = None
    else:
        selected_rung = None
        no_feasible_rung_reason = _no_feasible_rung_reason(rows, margin_threshold=margin_threshold)

    return {
        "rungs": rows,
        "margin_threshold": margin_threshold,
        "pair_weights": list(pair_weights),
        "lattice": lattice,
        "selection_rule": SELECTION_RULE,
        "selected_rung": selected_rung,
        "no_feasible_rung_reason": no_feasible_rung_reason,
    }


def _hash_verification_view(verified: dict[str, bool] | None) -> dict:
    if verified is None:
        return {
            "applicable": False,
            "reason": "smoke mode uses a synthetic pool; there is no certificate to verify against",
        }
    return {
        "applicable": True,
        "per_slot": {slot: bool(ok) for slot, ok in verified.items()},
        "all_verified": all(verified.values()),
    }


def _public_filters_view(filters: Mapping) -> dict:
    """The ``slot_gene_sets`` kwargs actually used, minus anything that could be a gene list.

    Every field here is a count, a chromosome name, or a coordinate -- never a gene ID -- so this
    is safe to embed directly in the artifact.
    """

    return {
        "blacklist_size": len(filters["blacklist"]),
        "allowed_chroms": sorted(filters["allowed_chroms"]),
        "forbidden_chroms": sorted(filters["forbidden_chroms"]),
        "construct_length": int(filters["construct_length"]),
        "donor_pos": int(filters["donor_pos"]),
        "acceptor_pos": int(filters["acceptor_pos"]),
        "donor_window": list(filters["donor_window"]),
        "acceptor_window": list(filters["acceptor_window"]),
        "blacklist_and_forbidden_chroms_note": (
            "both are correctly empty here, not omissions, despite the certificate recording "
            "5,998 blacklist members and 5 forbidden chromosomes: the certificate's blacklist is "
            "built from unversioned gene roots but compared here against this census's versioned "
            "gene_id, so it can never match -- a dead filter by construction, reproduced on "
            "purpose (see slot_gene_sets' docstring); and the census's own allowed_chromosomes "
            "(21 entries) already excludes every forbidden chromosome at build time, so restating "
            "them here would be redundant, not corrective, over this input"
        ),
    }


def assemble_artifact(
    *,
    mode: str,
    slot_sets: Mapping[str, set],
    slot_hash_verification: dict,
    census_receipt: dict,
    certificate_receipt: dict,
    filters_view: dict | None,
    ladder: tuple[int, ...] = LADDER,
    margin_threshold: float = MARGIN_THRESHOLD,
    pair_weights: tuple[int, ...] = DEFAULT_PAIR_WEIGHTS,
    lattice: str = DEFAULT_LATTICE,
) -> dict:
    """Wrap ``build_ladder`` with the honesty fields the brief requires. Pure: no filesystem access.

    ``slot_sets`` themselves are never written anywhere below this call -- only their derived
    counts, margins, hashes and verdicts. If a caller ever finds itself passing a gene ID string
    into one of the ``*_view``/``*_receipt`` arguments, that string will end up in the artifact;
    none of the values built by ``run_real``/``run_smoke`` below do that.
    """

    ladder_result = build_ladder(
        slot_sets,
        ladder=ladder,
        margin_threshold=margin_threshold,
        pair_weights=pair_weights,
        lattice=lattice,
    )
    return {
        "schema": SCHEMA,
        "study_directory": "results/splicebert_positional_clock/",
        "study_directory_provisional": True,
        "study_directory_provisional_note": (
            "the study directory name 'splicebert_positional_clock' is PROVISIONAL and may be "
            "renamed before anything freezes; this artifact is written under development/ "
            "specifically so a later rename costs a move, not a retraction"
        ),
        "stage": "construction",
        "mode": mode,
        "model_execution": False,
        "checkpoint_loaded": None,
        "cohort_selection_attempted": False,
        "census_receipt": census_receipt,
        "certificate_receipt": certificate_receipt,
        "slot_hash_verification": slot_hash_verification,
        "filters": filters_view,
        **ladder_result,
    }


# ---------------------------------------------------------------------------------------------
# Synthetic smoke fixture -- never derived from the real census.
# ---------------------------------------------------------------------------------------------


def _synthetic_smoke_slot_sets() -> dict[str, set[str]]:
    """Small, fully synthetic gene pools sized to exercise a mix of ladder outcomes.

    ``donor_phase_2`` and ``acceptor_phase_2`` deliberately share six IDs, so the smoke ladder also
    exercises the Hall-type joint-oversubscription obstruction that ``feasibility`` (and not plain
    counting) exists to catch -- the same shape of finding the real ladder run produces.
    """

    shared = {f"smoke_shared_{i}" for i in range(6)}
    return {
        "scaffold": {f"smoke_scaffold_{i}" for i in range(60)},
        "donor_phase_0": {f"smoke_d0_{i}" for i in range(24)},
        "donor_phase_1": {f"smoke_d1_{i}" for i in range(18)},
        "donor_phase_2": {f"smoke_d2_{i}" for i in range(10)} | shared,
        "acceptor_phase_0": {f"smoke_a0_{i}" for i in range(24)},
        "acceptor_phase_1": {f"smoke_a1_{i}" for i in range(18)},
        "acceptor_phase_2": {f"smoke_a2_{i}" for i in range(4)} | shared,
    }


# ---------------------------------------------------------------------------------------------
# Certificate -> slot_gene_sets filters. Mirrors tests/test_pbc_v2_capacity_census.py's
# `_filters_from_artifact` exactly (same fields, same two known-inert empty-set filters) so this
# runner and that reproduction test can never silently diverge on how the seven slots are built
# from the same committed certificate. See that test's docstring for why `blacklist=set()` and
# `forbidden_chroms=set()` are correct here, not placeholders.
# ---------------------------------------------------------------------------------------------


def _filters_from_certificate(cohort: dict) -> dict:
    config = cohort["config"]
    allowed_chroms = {normalize_chromosome(chrom) for chrom in config["allowed_chromosomes"]}
    return dict(
        blacklist=set(),
        allowed_chroms=allowed_chroms,
        forbidden_chroms=set(),
        construct_length=int(config["construct_length"]),
        donor_pos=int(config["donor_pos"]),
        acceptor_pos=int(config["acceptor_pos"]),
        donor_window=DEFAULT_DONOR_WINDOW,
        acceptor_window=DEFAULT_ACCEPTOR_WINDOW,
    )


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _repo_relative_or_absolute(path: Path, repo: Path) -> str:
    """Repo-relative string when `path` is under `repo`, else the absolute path unchanged.

    Used for provenance receipts of committed inputs (e.g. the PBC-v1 certificate) so a receipt
    never bakes in a worktree-local absolute path (`.claude/worktrees/<name>/...`) that evaporates
    the moment the branch merges. Inputs that genuinely live outside the repo (the 2.34 GB census)
    are left absolute, since there is no repo-relative path for them to begin with.
    """

    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--census",
        default=str(DEFAULT_CENSUS_PATH),
        help="path to the hash-pinned candidate_census_v1.json (INPUT, outside this repo, never "
        "opened in --smoke mode)",
    )
    ap.add_argument(
        "--certificate",
        default=str(DEFAULT_CERTIFICATE_PATH),
        help="path to the committed composition_balanced_cohort_v1.json capacity certificate",
    )
    # No default: every runner in the PBC family requires `--out` explicitly (see e.g.
    # `scripts/splicebert_positional_boundary_coherence/build_freshness_registry.py`), so a plain
    # rerun with no flags cannot silently target the record -- `tests/test_overwrite_guard.py`'s
    # D1 check enforces that this shape never appears for a new runner. The canonical
    # construction-stage path is `results/splicebert_positional_clock/development/`
    # `pbc_v2_capacity_ladder_v1.json`; see this study's RESULTS_INDEX.md.
    ap.add_argument("--out", required=True, help="output artifact path")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="exercise the demand/feasibility/power layers on synthetic pools only and return "
        "before --census is ever opened",
    )
    ap.add_argument(
        "--overwrite-record",
        action="store_true",
        dest="overwrite_record",
        help="replace an existing artifact of record (results are immutable by default)",
    )
    return ap.parse_args(argv)


def run_smoke(args: argparse.Namespace) -> tuple[Path, Path]:
    """Synthetic-pool-only path. Does not reference ``args.census`` anywhere below."""

    out = guard_output_path(Path(args.out).expanduser().resolve(), overwrite=args.overwrite_record)
    manifest_path = guard_output_path(
        out.with_name(out.stem + ".manifest.json"), overwrite=args.overwrite_record
    )

    slot_sets = _synthetic_smoke_slot_sets()
    empty_receipt = {"mode": "synthetic_smoke_pool", "path": None, "sha256": None, "size_bytes": None}
    artifact = assemble_artifact(
        mode="smoke",
        slot_sets=slot_sets,
        slot_hash_verification=_hash_verification_view(None),
        census_receipt=dict(empty_receipt),
        certificate_receipt=dict(empty_receipt),
        filters_view=None,
        ladder=SMOKE_LADDER,
        margin_threshold=MARGIN_THRESHOLD,
        pair_weights=DEFAULT_PAIR_WEIGHTS,
        lattice=DEFAULT_LATTICE,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        EXPERIMENT_ID,
        sys.argv,
        repo=REPO,
        status="SMOKE_ONLY_NON_AUTHORIZING",
        config={
            "stage": "construction",
            "mode": "smoke",
            "ladder": list(SMOKE_LADDER),
            "margin_threshold": MARGIN_THRESHOLD,
            "pair_weights": list(DEFAULT_PAIR_WEIGHTS),
            "lattice": DEFAULT_LATTICE,
            "checkpoint_loaded": None,
            "selection_rule": SELECTION_RULE,
            "selected_rung": artifact["selected_rung"],
        },
        seeds={},
        notes=[
            "Smoke mode: synthetic gene pools only; --census (if given) was never opened.",
            "Not an authoritative capacity/power record; rerun without --smoke for the real ladder.",
        ],
    )
    manifest.add_input(Path(__file__).resolve())
    manifest.add_input(REPO / "src/gfmcd/splicing/pbc_v2_capacity.py")
    manifest.add_output(out)
    manifest.write(manifest_path)
    return out, manifest_path


def run_real(args: argparse.Namespace) -> tuple[Path, Path]:
    out = guard_output_path(Path(args.out).expanduser().resolve(), overwrite=args.overwrite_record)
    manifest_path = guard_output_path(
        out.with_name(out.stem + ".manifest.json"), overwrite=args.overwrite_record
    )

    census_path = Path(args.census).expanduser().resolve()
    certificate_path = Path(args.certificate).expanduser().resolve()

    cohort = json.loads(certificate_path.read_text())

    # The certificate's sample_size_plan carries LEGACY_RESIDUAL_VARIANCE_UPPER_BOUND_DEV120's
    # value but not its meaning (see that constant's docstring in pbc_v2_capacity.py). Nothing
    # previously checked that the two still agree; assert it and fail loudly on drift rather than
    # silently projecting power off a variance the certificate no longer actually carries.
    observed_variance = cohort["sample_size_plan"]["variance"]
    if observed_variance != LEGACY_RESIDUAL_VARIANCE_UPPER_BOUND_DEV120:
        raise SystemExit(
            "FATAL: PBC-v1 certificate sample_size_plan.variance has drifted from the frozen "
            f"legacy input LEGACY_RESIDUAL_VARIANCE_UPPER_BOUND_DEV120="
            f"{LEGACY_RESIDUAL_VARIANCE_UPPER_BOUND_DEV120}: found {observed_variance} at "
            f"{certificate_path}. Refusing to compute a capacity/power ladder against a variance "
            "source that no longer matches its pinned provenance; no artifact was written."
        )

    filters = _filters_from_certificate(cohort)

    # Stream the 2.34 GB census exactly once; every rung below reuses these same slot sets.
    slot_sets = slot_gene_sets(stream_candidates(census_path), **filters)

    cert_slots = cohort["capacity"]["slots"]
    expected_hashes = {slot: entry["gene_ids_sha256"] for slot, entry in cert_slots.items()}
    verified = verify_slot_hashes(slot_sets, expected_hashes)
    if not all(verified.values()):
        failed = sorted(slot for slot, ok in verified.items() if not ok)
        raise SystemExit(
            "FATAL: slot hash verification failed against the committed certificate for "
            f"{failed}. Refusing to compute the capacity/power ladder on an unverified gene pool; "
            "no artifact was written."
        )

    census_receipt = file_record(census_path)
    census_receipt["mode"] = "file"
    # census_path genuinely lives outside this repo, so it stays absolute; certificate_path lives
    # inside it, so its receipt should not bake in a worktree-local absolute path that evaporates
    # on merge (e.g. `.claude/worktrees/<name>/...`) -- record it repo-relative instead.
    certificate_receipt = file_record(certificate_path)
    certificate_receipt["mode"] = "file"
    certificate_receipt["path"] = _repo_relative_or_absolute(certificate_path, REPO)

    artifact = assemble_artifact(
        mode="real",
        slot_sets=slot_sets,
        slot_hash_verification=_hash_verification_view(verified),
        census_receipt=census_receipt,
        certificate_receipt=certificate_receipt,
        filters_view=_public_filters_view(filters),
        ladder=LADDER,
        margin_threshold=MARGIN_THRESHOLD,
        pair_weights=DEFAULT_PAIR_WEIGHTS,
        lattice=DEFAULT_LATTICE,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        EXPERIMENT_ID,
        sys.argv,
        repo=REPO,
        status="BUILT",
        config={
            "stage": "construction",
            "mode": "real",
            "ladder": list(LADDER),
            "margin_threshold": MARGIN_THRESHOLD,
            "pair_weights": list(DEFAULT_PAIR_WEIGHTS),
            "lattice": DEFAULT_LATTICE,
            "checkpoint_loaded": None,
            "selection_rule": SELECTION_RULE,
            "selected_rung": artifact["selected_rung"],
        },
        seeds={},
        notes=[
            "Model-free construction-stage capacity/power ladder: no checkpoint loaded, no cohort "
            "selected, no block assembled, no cell materialised.",
            "Slot gene sets were verified against the seven certificate gene_ids_sha256 hashes "
            "before any rung was evaluated (hard gate, not a log line); see slot_hash_verification "
            "in the artifact and in this manifest's inputs.",
        ],
    )
    manifest.inputs.append(census_receipt)
    manifest.inputs.append(certificate_receipt)
    manifest.add_input(Path(__file__).resolve())
    manifest.add_input(REPO / "src/gfmcd/splicing/pbc_v2_capacity.py")
    manifest.add_output(out)
    manifest.write(manifest_path)
    return out, manifest_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out, manifest_path = run_smoke(args) if args.smoke else run_real(args)
    print(
        json.dumps(
            {
                "status": "SMOKE_ONLY_NON_AUTHORIZING" if args.smoke else "BUILT",
                "mode": "smoke" if args.smoke else "real",
                "output": str(out),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
