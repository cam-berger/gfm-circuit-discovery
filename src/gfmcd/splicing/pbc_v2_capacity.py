"""Model-free capacity analysis for the PBC v2 successor gene pool -- layer 1.

This module streams the hash-bound candidate census produced by
``scripts/splicebert_positional_boundary_coherence/build_candidate_census.py`` (a 2.34 GB
pretty-printed JSON document, never committed to this repo) into the seven role/phase "slots" the
donor-graph capacity certificate names, and lets a caller verify the resulting gene sets against a
committed certificate's sha256 hashes.

Stage: ``construction``. This module never imports ``torch``, never imports a model adapter, and
never touches a SpliceBERT checkpoint -- it must stay importable without the ``splicebert`` extra.
Later layers (a demand model, an exact max-flow feasibility solver, and a power projection) live in
this same file as additional pure, independent functions; nothing here should come to depend on
model access.

The census file and the ``studies/splicebert_positional_boundary_coherence/`` /
``results/splicebert_positional_boundary_coherence/`` trees are read-only inputs: that study is
closed and this module never writes into either.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from gfmcd.splicing.positional_boundary_coherence import gene_root, sha256_payload

# Identity of the one census file this module is pinned to reproduce against (see the task-1 brief
# and results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.json
# -> candidate_census_receipt.candidate_file). The census itself is a 2.34 GB INPUT that lives
# outside this repo and is never copied in; `stream_candidates` hard-fails rather than silently
# streaming a file that does not match this pin.
_CENSUS_SHA256 = "9be356ac26ff18aa750b0552684d4a94fa6b8963fee01666608b38cc7de25614"
_CENSUS_SIZE_BYTES = 2343604295

# The census is `json.dumps(..., indent=2, sort_keys=True)`-shaped: the `"candidates"` array lives
# at 2-space (depth-1) indent, and each candidate object opens/closes at 4-space (depth-2) indent.
# A later top-level key (`"exclusions"`) is *also* a list of dicts, and its elements sit at the same
# 4-space depth as candidate objects -- so a naive whole-file scan for 4-space `{`/`}` pairs would
# silently pull in ~327k unrelated exclusion records. The array-open/array-close markers below are
# what keep `_iter_candidate_objects` scoped to exactly the candidates array.
_CANDIDATES_ARRAY_OPEN = '  "candidates": ['
_CANDIDATE_OPEN = "    {"
_CANDIDATE_CLOSE_LINES = ("    },", "    }")
_ARRAY_CLOSE_LINES = ("  ],", "  ]")


def _verify_census_checksum(path: Path) -> None:
    """Hard-fail with a clear error unless `path` is byte-identical to the pinned census."""

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"candidate census not readable at {path}: {exc}") from exc
    if size != _CENSUS_SIZE_BYTES:
        raise ValueError(
            f"candidate census size mismatch for {path}: expected {_CENSUS_SIZE_BYTES} bytes, "
            f"found {size} bytes. Refusing to stream an input that does not match the pinned "
            "candidate_census_v1.json this module was audited against."
        )
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != _CENSUS_SHA256:
        raise ValueError(
            f"candidate census sha256 mismatch for {path}: expected {_CENSUS_SHA256}, "
            f"found {digest}. Refusing to stream an input that does not match the pinned "
            "candidate_census_v1.json this module was audited against."
        )


def _iter_candidate_objects(path: Path) -> Iterator[dict]:
    """Yield each candidate dict from the census's `candidates` array, in file order.

    Reads the file one line at a time (never `json.load`s the whole document) and accumulates only
    the lines belonging to the one candidate currently being parsed, so peak memory is bounded by a
    single candidate's text, not the file. See the module-level comment on `_CANDIDATES_ARRAY_OPEN`
    for why this must track the array's own open/close markers rather than scanning indent alone.
    """

    saw_array_open = False
    saw_array_close = False
    in_candidate = False
    buffer: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not saw_array_open:
                if line == _CANDIDATES_ARRAY_OPEN:
                    saw_array_open = True
                continue
            if in_candidate:
                buffer.append(raw_line)
                if line in _CANDIDATE_CLOSE_LINES:
                    text = "".join(buffer).strip()
                    if text.endswith(","):
                        text = text[:-1]
                    yield json.loads(text)
                    buffer = []
                    in_candidate = False
                continue
            if line == _CANDIDATE_OPEN:
                in_candidate = True
                buffer = [raw_line]
                continue
            if line in _ARRAY_CLOSE_LINES:
                saw_array_close = True
                break
            raise ValueError(
                f"{path}: unexpected line inside the candidates array (neither a candidate open "
                f"nor the array close): {line!r}. The census file's format has drifted from what "
                "this streamer was written against."
            )

    if not saw_array_open:
        raise ValueError(
            f'{path}: never found the top-level "candidates": [ array opener. The census file is '
            "malformed, or is not the candidate_census_v1.json format this module expects."
        )
    if in_candidate:
        raise ValueError(
            f"{path}: candidate stream ended in the middle of an object -- the file is truncated."
        )
    if not saw_array_close:
        raise ValueError(
            f"{path}: never found the candidates array's closing ']' -- the file is truncated or "
            "was not fully written."
        )


def stream_candidates(path: str | Path) -> Iterator[dict]:
    """Stream candidate objects out of the hash-bound PBC census without ever loading it whole.

    Verifies the file's sha256 against the pin this module was audited against *before* any
    candidate is yielded, and hard-fails with a clear error on a mismatch. The census is ~2.34 GB of
    pretty-printed JSON; `json.load` on it needs >10 GB of RAM and is forbidden here by design --
    this reads the file one line at a time and `json.loads`\\ s one candidate object at a time.
    """

    resolved = Path(path)
    _verify_census_checksum(resolved)
    return _iter_candidate_objects(resolved)


def slot_gene_sets(
    candidates: Iterable[dict],
    *,
    blacklist: set[str],
    allowed_chroms: set[str],
    forbidden_chroms: set[str],
    construct_length: int,
    donor_pos: int,
    acceptor_pos: int,
    donor_window: tuple,
    acceptor_window: tuple,
) -> dict[str, set[str]]:
    """Bucket eligible candidates into the seven donor-graph capacity slots.

    Mirrors `positional_boundary_coherence.py` lines 2107-2134 exactly: the same three filtering
    passes (generic eligibility, scaffold geometry, role/phase pools), then the same
    gene_root-normalisation `_necessary_capacity_report` applies before hashing. `candidates` is
    consumed exactly once (safe for a one-shot `stream_candidates` iterator); every candidate that
    survives the eligibility filter is held in memory for the rest of the call, matching the
    authority implementation's own memory shape -- this function does not additionally chunk or
    re-stream past that point.

    Returns exactly the seven keys `donor_phase_0..2`, `acceptor_phase_0..2`, `scaffold`, each
    mapped to a set of gene_root-normalised (unversioned) gene IDs. Per the task-1 brief, no public
    function in this module emits a gene list intended for serialisation -- sets are for in-process
    use only; a caller that persists an artifact must reduce them to counts and hashes first.
    """

    # KNOWN QUIRK, REPRODUCED ON PURPOSE -- DO NOT "FIX": the authority builds `blacklist` from
    # `gene_root(...)` (unversioned, e.g. "ENSG00000099194") but compares it here against the
    # versioned `candidate["gene_id"]` (e.g. "ENSG00000099194.8"), so this membership test can never
    # fire. It is a real dead filter in the audited builder, harmless only because freshness was
    # already enforced when the census was built (every candidate carries
    # eligibility_checks.freshness_registry_clear == true). Mirroring it verbatim is required for
    # the committed certificate hashes to reproduce; "fixing" it changes the audited gene pool.
    filtered = [
        candidate
        for candidate in candidates
        if candidate["gene_id"] not in blacklist
        and candidate["chrom"] in allowed_chroms
        and candidate["chrom"] not in forbidden_chroms
    ]
    scaffolds = [
        candidate
        for candidate in filtered
        if candidate["role"] == "scaffold"
        and len(candidate["sequence"]) == int(construct_length)
        and int(candidate["donor_pos"]) == int(donor_pos)
        and int(candidate["acceptor_pos"]) == int(acceptor_pos)
        and tuple(candidate["donor_window"]) == donor_window
        and tuple(candidate["acceptor_window"]) == acceptor_window
    ]
    module_pools = {
        role: {
            phase: [
                candidate
                for candidate in filtered
                if candidate["role"] == role and int(candidate["phase"]) == phase
            ]
            for phase in (0, 1, 2)
        }
        for role in ("donor", "acceptor")
    }
    slot_genes: dict[str, set[str]] = {
        "scaffold": {gene_root(str(candidate["gene_id"])) for candidate in scaffolds},
    }
    for role in ("donor", "acceptor"):
        for phase in (0, 1, 2):
            slot_genes[f"{role}_phase_{phase}"] = {
                gene_root(str(candidate["gene_id"]))
                for candidate in module_pools[role][phase]
            }
    return slot_genes


def verify_slot_hashes(
    slot_sets: Mapping[str, set[str]], expected: Mapping[str, str]
) -> dict[str, bool]:
    """Check each named slot's `sha256_payload(sorted(...))` against an expected hash.

    `expected` maps a slot name to the sha256 hex digest it should reproduce (e.g. read out of a
    committed certificate's `capacity.slots.<slot>.gene_ids_sha256`). Uses `sha256_payload` from
    `positional_boundary_coherence` rather than reimplementing the hash so this can never silently
    drift from the authority's own hashing convention.
    """

    return {
        slot: sha256_payload(sorted(slot_sets[slot])) == expected_hash
        for slot, expected_hash in expected.items()
    }


# ------------------------------------------------------------------------------------------------
# Layer 2 -- rotated 2x2 balanced-incomplete-block demand model
# ------------------------------------------------------------------------------------------------
#
# Each block carries a 2x2 crossed lattice over ONE pair of annotated phases: two donor modules, two
# acceptor modules, one scaffold = five globally unique genes. The pair rotates over the three
# phase pairs so that delta = (p_A - p_D) mod 3 is orthogonal to both donor phase and acceptor
# phase. That orthogonality is a property of the pooled design ACROSS pair types -- it does not
# hold within any single block, so every resampling scheme downstream must stratify on pair type.

PAIR_TYPES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))

SLOT_ORDER: tuple[str, ...] = (
    "scaffold",
    "donor_phase_0",
    "donor_phase_1",
    "donor_phase_2",
    "acceptor_phase_0",
    "acceptor_phase_1",
    "acceptor_phase_2",
)


def allocate_blocks(n_blocks: int, *, pair_weights: tuple[int, ...] = (1, 1, 1)) -> dict:
    """Split `n_blocks` across the three phase pairs by largest remainder.

    Deterministic and exact: the returned counts always sum to `n_blocks`, so a rung is never
    silently rounded up into infeasibility or down into a weaker design. Ties in the remainder are
    broken by pair order, which is fixed by `PAIR_TYPES`.
    """

    if n_blocks < 0:
        raise ValueError(f"n_blocks must be non-negative, got {n_blocks}")
    if len(pair_weights) != len(PAIR_TYPES):
        raise ValueError(f"pair_weights must have {len(PAIR_TYPES)} entries, got {len(pair_weights)}")
    if any(weight < 0 for weight in pair_weights):
        raise ValueError(f"pair_weights must be non-negative, got {pair_weights}")
    total_weight = sum(pair_weights)
    if total_weight <= 0:
        raise ValueError("pair_weights must not sum to zero")

    exact = [n_blocks * weight / total_weight for weight in pair_weights]
    floors = [int(value) for value in exact]
    shortfall = n_blocks - sum(floors)
    order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - floors[i]), i))
    for index in order[:shortfall]:
        floors[index] += 1
    return {pair: count for pair, count in zip(PAIR_TYPES, floors)}


def rotated_demands(n_blocks: int, *, pair_weights: tuple[int, ...] = (1, 1, 1)) -> dict[str, int]:
    """Unique-gene demand per capacity slot, plus the three union families.

    Per block: one donor gene at each of the pair's two phases, one acceptor gene at each, and one
    scaffold -- five globally unique genes. A phase appears in two of the three pair types, so with
    equal weights each donor/acceptor phase slot is demanded by two thirds of the blocks.
    """

    allocation = allocate_blocks(n_blocks, pair_weights=pair_weights)
    demands = {slot: 0 for slot in SLOT_ORDER}
    demands["scaffold"] = n_blocks
    for (phase_a, phase_b), count in allocation.items():
        for phase in (phase_a, phase_b):
            demands[f"donor_phase_{phase}"] += count
            demands[f"acceptor_phase_{phase}"] += count
    demands["donor_all_phases"] = 2 * n_blocks
    demands["acceptor_all_phases"] = 2 * n_blocks
    demands["all_roles"] = 5 * n_blocks
    return demands


def delta_class_phase_counts(
    pair_types: Iterable[tuple[int, int]] = PAIR_TYPES,
    *,
    blocks_per_pair: Mapping[tuple[int, int], int] | None = None,
) -> dict[int, dict[str, dict[int, int]]]:
    """Count donor- and acceptor-phase occurrences within each delta class.

    delta = (p_A - p_D) mod 3. The design's whole justification over a plain {0,1} lattice is that
    every delta class draws equally from all three donor phases and all three acceptor phases; a
    lattice that fails this confounds delta with module identity, which is what makes `C43`'s
    M+/M- decomposition uninterpretable. Returns
    ``{delta: {"donor": {phase: count}, "acceptor": {phase: count}}}`` with an explicit zero for
    every phase that never appears, so a missing phase reads as 0 rather than vanishing.
    """

    pairs = tuple(pair_types)
    weights = blocks_per_pair if blocks_per_pair is not None else {pair: 1 for pair in pairs}
    counts: dict[int, dict[str, dict[int, int]]] = {
        delta: {"donor": {p: 0 for p in (0, 1, 2)}, "acceptor": {p: 0 for p in (0, 1, 2)}}
        for delta in (0, 1, 2)
    }
    for pair in pairs:
        weight = weights.get(pair, 0)
        phase_a, phase_b = pair
        for donor_phase in (phase_a, phase_b):
            for acceptor_phase in (phase_a, phase_b):
                delta = (acceptor_phase - donor_phase) % 3
                counts[delta]["donor"][donor_phase] += weight
                counts[delta]["acceptor"][acceptor_phase] += weight
    return counts


def delta_balance_report(
    pair_types: Iterable[tuple[int, int]] = PAIR_TYPES,
    *,
    blocks_per_pair: Mapping[tuple[int, int], int] | None = None,
) -> dict:
    """Report whether delta is orthogonal to donor and acceptor phase in the pooled design.

    `balanced` is true only when, inside every delta class that occurs at all, all three donor
    phases appear equally often and all three acceptor phases appear equally often.
    """

    counts = delta_class_phase_counts(pair_types, blocks_per_pair=blocks_per_pair)
    per_delta = {}
    for delta, roles in counts.items():
        donor_values = set(roles["donor"].values())
        acceptor_values = set(roles["acceptor"].values())
        occurs = any(value > 0 for value in roles["donor"].values())
        per_delta[delta] = {
            "occurs": occurs,
            "donor_counts": dict(roles["donor"]),
            "acceptor_counts": dict(roles["acceptor"]),
            "donor_balanced": (not occurs) or len(donor_values) == 1,
            "acceptor_balanced": (not occurs) or len(acceptor_values) == 1,
        }
    balanced = all(
        entry["donor_balanced"] and entry["acceptor_balanced"] for entry in per_delta.values()
    )
    return {"balanced": balanced, "per_delta": per_delta}


# ------------------------------------------------------------------------------------------------
# Layer 3 -- exact gene->slot feasibility by max-flow, with a min-cut certificate
# ------------------------------------------------------------------------------------------------
#
# Per-slot and union counts are NECESSARY but not SUFFICIENT. Because a single gene can be eligible
# for several slots and global disjointness lets it fill only one, a set of slots can be jointly
# oversubscribed while every individual slot and every declared union passes. That is a Hall-type
# obstruction, and counting cannot see it.
#
# Exact model: source -> gene (capacity 1, this is global disjointness), gene -> slot (capacity
# infinity for every slot the gene is eligible for), slot -> sink (capacity = that slot's demand).
# Feasible iff max-flow == total demand. Infinite middle edges force every minimum cut onto the
# source and sink edges, which is what makes the cut readable as a Hall violator: the slots left on
# the SINK side are a set B with |N(B)| < demand(B), and the genes on the sink side are exactly
# N(B). Counting alone can never produce that certificate.

_SOURCE = ("__source__", "")
_SINK = ("__sink__", "")


def feasibility(slot_sets: Mapping[str, set[str]], demands: Mapping[str, int]) -> dict:
    """Decide exact constructibility of the per-slot demands against overlapping gene pools.

    Only the slot names present in `demands` are modelled; union-family pseudo-slots such as
    ``donor_all_phases`` must NOT be passed in, because they are aggregates of the real slots and
    would double-count the same genes. Returns a verdict plus, on failure, the min-cut certificate.
    """

    import networkx as nx  # imported lazily so module import stays cheap and dependency-light

    real_slots = [slot for slot in demands if slot in slot_sets]
    unknown = [slot for slot in demands if slot not in slot_sets]
    if unknown:
        raise ValueError(
            f"feasibility() received demands for slots with no gene pool: {sorted(unknown)}. "
            "Union families are aggregates and must not be modelled as slots."
        )
    total_demand = sum(int(demands[slot]) for slot in real_slots)

    graph = nx.DiGraph()
    graph.add_node(_SOURCE)
    graph.add_node(_SINK)
    for slot in real_slots:
        graph.add_edge(("slot", slot), _SINK, capacity=int(demands[slot]))
    for slot in real_slots:
        for gene in slot_sets[slot]:
            gene_node = ("gene", gene)
            if not graph.has_edge(_SOURCE, gene_node):
                graph.add_edge(_SOURCE, gene_node, capacity=1)
            graph.add_edge(gene_node, ("slot", slot), capacity=float("inf"))

    cut_value, (source_side, sink_side) = nx.minimum_cut(graph, _SOURCE, _SINK)
    feasible = cut_value >= total_demand

    violating_slots = sorted(name for kind, name in sink_side if kind == "slot")
    reachable_genes = {name for kind, name in sink_side if kind == "gene"}
    return {
        "feasible": bool(feasible),
        "max_flow": int(cut_value),
        "total_demand": int(total_demand),
        "shortfall": int(max(0, total_demand - cut_value)),
        "min_cut_slots": [] if feasible else violating_slots,
        "min_cut_gene_count": None if feasible else len(reachable_genes),
        "min_cut_demand": None if feasible else sum(int(demands[s]) for s in violating_slots),
        "modelled_slots": sorted(real_slots),
    }


def necessary_counts(
    slot_sets: Mapping[str, set[str]], demands: Mapping[str, int]
) -> dict:
    """The cheap per-slot and union-family necessary conditions, for comparison with the exact flow.

    Kept separate from `feasibility` on purpose: the point of running both is that a disagreement
    between them is exactly a Hall-type obstruction, and it must be REPORTED, never silently
    resolved in favour of whichever verdict is more convenient.
    """

    unions = {
        "donor_all_phases": ("donor_phase_0", "donor_phase_1", "donor_phase_2"),
        "acceptor_all_phases": ("acceptor_phase_0", "acceptor_phase_1", "acceptor_phase_2"),
        "all_roles": SLOT_ORDER,
    }
    entries: dict[str, dict] = {}
    for slot in SLOT_ORDER:
        if slot not in slot_sets:
            continue
        available = len(slot_sets[slot])
        demand = int(demands.get(slot, 0))
        entries[slot] = _capacity_row(available, demand)
    for name, members in unions.items():
        present = [m for m in members if m in slot_sets]
        available = len(set().union(*(slot_sets[m] for m in present))) if present else 0
        entries[name] = _capacity_row(available, int(demands.get(name, 0)))
        entries[name]["source_slots"] = list(present)
    return {
        "entries": entries,
        "passed": all(entry["passed"] for entry in entries.values()),
        "failed_constraints": sorted(k for k, v in entries.items() if not v["passed"]),
    }


def _capacity_row(available: int, demand: int) -> dict:
    """One necessary-condition row. `margin` is defined here, once: (available - demand)/available."""

    return {
        "available_unique_genes": int(available),
        "demand_unique_genes": int(demand),
        "deficit_unique_genes": int(max(0, demand - available)),
        "margin": (available - demand) / available if available else None,
        "passed": demand <= available,
    }


# ------------------------------------------------------------------------------------------------
# Layer 4 -- power projection over a variance split that is NOT identifiable from the record
# ------------------------------------------------------------------------------------------------
#
# Per-block endpoints are DIFFERENCES OF TWO CELL MEANS, so the cell-level variance component enters
# as (1/m + 1/u), not as 1/k. Writing it as 1/k understates how much a 4-cell block loses relative
# to a 9-cell block.
#
# The only permitted legacy input is the single sign-free block SD PBC-v1 already froze and audited
# (sample_size_plan.standard_deviation). No sign, mean, or direction is read from any C43 artifact.
#
# rho -- the fraction of that 9-cell block variance which is gene-level (irreducible by adding
# cells) -- is NOT identifiable from anything committed: the 3,456 per-cell rows behind C43 were
# gitignored at source. Every projection below is therefore a BAND over rho in [0, 1], never a
# point, and the artifact records that explicitly.

LEGACY_BLOCK_SD_9CELL = 0.340503          # PBC-v1 frozen sign-free legacy input
LEGACY_BLOCK_VARIANCE_9CELL = 0.115942293009
SESOI = 0.05
ALPHA = 0.05
POWER = 0.80
_Z = 1.959963985 + 0.841621234            # z_{1-alpha/2} + z_{1-beta}

LATTICE_CELLS = {
    # name: (matched cells, mismatched cells, cells at one delta direction)
    "three_by_three": (3, 6, 3),
    "rotated_two_by_two": (2, 2, 1),
}


def _cell_variance(rho: float) -> float:
    """Cell-level component, pinned by requiring the 3x3 model to reproduce the frozen variance."""

    _check_rho(rho)
    return 2.0 * (1.0 - rho) * LEGACY_BLOCK_VARIANCE_9CELL


def _check_rho(rho: float) -> None:
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1], got {rho}")


def var_psi(matched: int, mismatched: int, rho: float) -> float:
    """Block variance of Psi = mean(matched cells) - mean(mismatched cells)."""

    return rho * LEGACY_BLOCK_VARIANCE_9CELL + _cell_variance(rho) * (1.0 / matched + 1.0 / mismatched)


def var_direction(delta_cells: int, matched: int, rho: float) -> float:
    """Block variance of a directional endpoint M_delta = mean(delta cells) - mean(matched cells)."""

    return rho * LEGACY_BLOCK_VARIANCE_9CELL + _cell_variance(rho) * (1.0 / delta_cells + 1.0 / matched)


def mde80(variance: float, n_blocks: int) -> float:
    """Minimum detectable effect at alpha=0.05 two-sided, power=0.80, for `n_blocks` blocks."""

    if n_blocks <= 0:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")
    return _Z * (variance / n_blocks) ** 0.5


def blocks_for_sesoi(variance: float) -> int:
    """Blocks needed to reach MDE80 <= SESOI at the given per-block variance."""

    import math

    return math.ceil(_Z * _Z * variance / (SESOI ** 2))


def power_projection(n_blocks: int, *, lattice: str = "rotated_two_by_two") -> dict:
    """Full projection for one rung: MDE80 bands over rho, plus the honesty flags.

    Two endpoints deliberately return a status string rather than a number:

    * ``n3`` -- ``N3 = Y00 - Y10`` is a paired within-block difference whose variance depends on an
      unknown correlation between carrier states. That correlation is a property of the dose
      operator, which does not exist yet, so any number here would be invented.
    * ``cross_lineage`` -- the replicate for a corpus intervention is the TRAINING SEED, not the
      evaluation gene. Gene-block bootstraps supply no degrees of freedom for it, and no repeated
      training variance exists anywhere in this record. No choice of `n_blocks` fixes this, because
      blocks are the wrong axis.
    """

    matched, mismatched, delta_cells = LATTICE_CELLS[lattice]
    band = {}
    for label, rho in (("best_case_rho_1", 1.0), ("worst_case_rho_0", 0.0)):
        band[label] = {
            "rho": rho,
            "psi_mde80": mde80(var_psi(matched, mismatched, rho), n_blocks),
            "direction_mde80": mde80(var_direction(delta_cells, matched, rho), n_blocks),
        }
    psi_worst = band["worst_case_rho_0"]["psi_mde80"]
    psi_best = band["best_case_rho_1"]["psi_mde80"]
    dir_worst = band["worst_case_rho_0"]["direction_mde80"]
    dir_best = band["best_case_rho_1"]["direction_mde80"]

    def _verdict(best: float, worst: float) -> str:
        if worst <= SESOI:
            return "powered_for_every_rho"
        if best <= SESOI:
            return "conditional_on_rho"
        return "underpowered_for_every_rho"

    return {
        "n_blocks": int(n_blocks),
        "lattice": lattice,
        "cells": {"matched": matched, "mismatched": mismatched, "per_delta": delta_cells},
        "alpha": ALPHA,
        "power": POWER,
        "sesoi": SESOI,
        "legacy_variance_source": "PBC-v1 composition_balanced_cohort_v1.json sample_size_plan",
        "legacy_block_sd_9cell": LEGACY_BLOCK_SD_9CELL,
        "rho_identifiable": False,
        "rho_unidentifiable_reason": (
            "the 3,456 per-cell rows behind C43 were gitignored at source, so the gene-level vs "
            "cell-level split of the frozen block variance cannot be estimated from any committed "
            "artifact; projections are bands over rho in [0, 1], never points"
        ),
        "band": band,
        "psi": {
            "mde80_best": psi_best,
            "mde80_worst": psi_worst,
            "verdict": _verdict(psi_best, psi_worst),
            "blocks_needed_worst_case": blocks_for_sesoi(var_psi(matched, mismatched, 0.0)),
        },
        "direction": {
            "mde80_best": dir_best,
            "mde80_worst": dir_worst,
            "verdict": _verdict(dir_best, dir_worst),
            "blocks_needed_worst_case": blocks_for_sesoi(
                var_direction(delta_cells, matched, 0.0)
            ),
        },
        "n3": {
            "status": "unprojectable_pending_dose_calibration",
            "reason": (
                "N3 = Y00 - Y10 is a paired within-block difference; its variance depends on an "
                "unknown carrier-state correlation determined by a dose operator that does not "
                "exist yet"
            ),
        },
        "cross_lineage": {
            "projectable": False,
            "replicate": "training_seed",
            "reason": (
                "no repeated-training variance exists anywhere in this record; gene-block "
                "bootstraps give no degrees of freedom for a corpus intervention, and n_blocks is "
                "the wrong axis for it"
            ),
            "resolution": "seed-level pilot: train 2-3 control-arm seeds and measure the seed-level SD of Psi",
        },
    }
