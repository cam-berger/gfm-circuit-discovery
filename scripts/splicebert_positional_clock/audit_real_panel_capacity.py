#!/usr/bin/env python
"""Independently audit ``results/splicebert_positional_clock/real_panel_capacity_v1.json``.

Every quantity reported by ``run_real_panel_capacity.py`` (B1 variance components, B2 exact
max-flow ceilings, B3 power at the measured components, B4 the cross-lineage required
correlation, B5 the exclusion tally) is recomputed here from the same four raw inputs, on a
code path that imports **nothing** from ``run_real_panel_capacity.py`` and **nothing** from
``gfmcd.splicing.pbc_v2_capacity`` or ``gfmcd.splicing.positional_boundary_coherence``. The
census streamer, the slot-bucketing filter, the sha256/canonical-JSON hashing, the sign-free
variance decomposition, the power formulas, the largest-remainder rotated-lattice demand model,
and the max-flow feasibility check are all reimplemented from scratch in this module.

The one exception, made explicit rather than silent: **max-flow itself runs on ``networkx``**,
the same third-party graph library the producer's own ``feasibility()`` helper calls internally.
A bug in ``networkx``'s max-flow implementation would therefore be invisible to this audit --
this is recorded verbatim in the audit artifact's ``independence.shared_primitives_imported``
field. Nothing else is shared.

**Not clean-checkout re-executable.** The 2.34 GB candidate census and the 54 MB `C43` cells file
both live outside this repository and are untracked (see ``census``/``cells`` CLI defaults below).
A fresh checkout without those two local files cannot rerun this audit; it can only be rerun on a
machine that already has them. This is recorded in the audit artifact's
``clean_checkout_reexecutable`` field, per the family's own artifact contract in
``studies/splicebert_positional_boundary_coherence/README.md`` ("An audit that depends on an
uncommitted raw shard must say so and cannot be called clean-checkout re-executable").

Emits ``AUDIT_PASS`` or ``AUDIT_FAIL`` -- never a warning for a mismatch. Stage: ``construction``.
No torch, no checkpoint, no cohort selection; no gene ID is ever read into a field that reaches
the output (every artifact/certificate/census field that carries a gene ID is reduced to a count,
a set cardinality, or a sha256 digest before it is stored anywhere in this module).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402

AUDIT_SCHEMA = "gfmcd.splicebert_positional_clock.real_panel_capacity_audit.v1"
AUDITED_ARTIFACT_SCHEMA = "gfmcd.splicebert_positional_clock.real_panel_capacity.v1"
EXPERIMENT_ID = "splicebert-positional-clock:real-panel-capacity-audit-v1"
PRODUCER_SCRIPT = "scripts/splicebert_positional_clock/run_real_panel_capacity.py"

# ---------------------------------------------------------------------------------------------
# Default paths -- mirror run_real_panel_capacity.py's own defaults for the four raw inputs, plus
# defaults for the artifact/manifest under audit. `--out` has NO default (required, guarded).
# ---------------------------------------------------------------------------------------------

DEFAULT_ARTIFACT_PATH = REPO / "results/splicebert_positional_clock/real_panel_capacity_v1.json"
DEFAULT_ARTIFACT_MANIFEST_PATH = REPO / "results/splicebert_positional_clock/real_panel_capacity_v1.manifest.json"
DEFAULT_CENSUS_PATH = Path(
    "/Users/cberg/Research/projects/gfm-circuit-discovery-pbc-successor/results/"
    "splicebert_positional_boundary_coherence/development/candidate_census_v1.json"
)
DEFAULT_CERTIFICATE_PATH = (
    REPO / "results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.json"
)
DEFAULT_CELLS_PATH = Path(
    "/Users/cberg/Research/projects/gfm-circuit-discovery/results/"
    "splicebert_exon_edge_compatibility/crossed_direct_edge_replication_cells.json"
)
DEFAULT_ANALYSIS_PATH = (
    REPO / "results/splicebert_exon_edge_compatibility/crossed_direct_edge_replication_analysis.json"
)

# ---------------------------------------------------------------------------------------------
# Audited pins -- independently declared by this module, not read from the artifact under audit.
# These are the values a reader independently confirms are correct; a mismatch against any of
# them is a FAIL regardless of what the audited artifact itself claims.
# ---------------------------------------------------------------------------------------------

_CENSUS_SHA256_PIN = "9be356ac26ff18aa750b0552684d4a94fa6b8963fee01666608b38cc7de25614"
_CENSUS_SIZE_PIN = 2343604295
_CELLS_SHA256_PIN = "428ad5d6b357bfba287fcfe3e358b00ab91ddcd748118b83b60b626593c440bf"
_EXPECTED_EXCLUSIONS_TOTAL = 282057

# The two correct required-correlation figures (see B4 below) and the value the documented raw-V9
# bug would have produced instead. If the audited artifact reports the wrong pair, the audit FAILS.
_EXPECTED_REQUIRED_CORRELATION = {544: 0.621, 726: 0.494}
_WRONG_REQUIRED_CORRELATION_RAW_V9 = {544: 0.382, 726: 0.175}
_CORRELATION_TOLERANCE = 0.001

# Frozen module geometry (src/gfmcd/splicing/positional_boundary_coherence.py's
# DEFAULT_DONOR_WINDOW / DEFAULT_ACCEPTOR_WINDOW): numeric design constants, not code: cited here,
# not imported.
DEFAULT_DONOR_WINDOW = (302, 428)
DEFAULT_ACCEPTOR_WINDOW = (462, 602)

ALL_SLOTS: tuple[str, ...] = (
    "scaffold",
    "donor_phase_0",
    "donor_phase_1",
    "donor_phase_2",
    "acceptor_phase_0",
    "acceptor_phase_1",
    "acceptor_phase_2",
)
_PLAIN_PHASE_SLOTS: tuple[str, ...] = (
    "donor_phase_0",
    "donor_phase_1",
    "acceptor_phase_0",
    "acceptor_phase_1",
)
_PAIR_TYPES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
_MARGIN_THRESHOLD = 0.25
_CEILING_SEARCH_UPPER_BOUND = 4000

SESOI = 0.05
_ALPHA_TWO_SIDED = 0.05
_POWER = 0.80

_PHASE_EQUAL_KEY = "upstream_downstream_phase_equal"

# ---------------------------------------------------------------------------------------------
# Hashing / string primitives -- reimplemented, not imported. Each is a few lines; there is no
# advantage to importing gfmcd.splicing.positional_boundary_coherence's versions here.
# ---------------------------------------------------------------------------------------------


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gene_root(gene_id: str) -> str:
    root = str(gene_id).strip().split(".", 1)[0]
    if not root:
        raise ValueError("gene_id is empty")
    return root


def _normalize_chromosome(chrom: str) -> str:
    value = str(chrom).strip()
    if not value:
        raise ValueError("chromosome is empty")
    return value if value.startswith("chr") else f"chr{value}"


def _count_key_occurrences(node: Any, key: str) -> int:
    """Recursively count occurrences of dict key `key` anywhere inside `node`."""
    if isinstance(node, dict):
        count = sum(1 for k in node if k == key)
        for value in node.values():
            count += _count_key_occurrences(value, key)
        return count
    if isinstance(node, list):
        return sum(_count_key_occurrences(item, key) for item in node)
    return 0


# ---------------------------------------------------------------------------------------------
# Census streaming -- an independent, single-pass reader over BOTH top-level arrays
# ("candidates" then "exclusions", in that file order), reimplemented from the documented format
# rather than imported. Bounded memory: only the current object's lines are buffered.
# ---------------------------------------------------------------------------------------------

_OBJ_OPEN = "    {"
_OBJ_CLOSE_LINES = ("    },", "    }")
_ARRAY_CLOSE_LINES = ("  ],", "  ]")
_CANDIDATES_ARRAY_OPEN = '  "candidates": ['
_EXCLUSIONS_ARRAY_OPEN = '  "exclusions": ['


def stream_census_sections(path: Path) -> Iterator[tuple[str, dict]]:
    """Yield ``("candidate", obj)`` for every object in the census's ``candidates`` array, then
    ``("exclusion", obj)`` for every object in its ``exclusions`` array -- a single top-to-bottom
    pass over one open file handle, since the two arrays sit sequentially in the file.

    Assumes the real census's own pretty-printed shape (``json.dumps(..., indent=2,
    sort_keys=True)``), matching ``build_candidate_census.py``'s output format: a genuinely empty
    array collapses to ``"key": []`` on one line under this encoding, which this reader cannot
    open (neither can the producer's own equivalent reader) -- never the case for the real,
    multi-hundred-thousand-row census this module is audited against.
    """
    sections = (("candidate", _CANDIDATES_ARRAY_OPEN), ("exclusion", _EXCLUSIONS_ARRAY_OPEN))
    with path.open("r", encoding="utf-8") as handle:
        for section_name, open_marker in sections:
            found_open = False
            for raw_line in handle:
                if raw_line.rstrip("\n") == open_marker:
                    found_open = True
                    break
            if not found_open:
                raise ValueError(f"{path}: never found {open_marker!r} -- unexpected census format")

            buffer: list[str] = []
            in_object = False
            found_close = False
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if in_object:
                    buffer.append(raw_line)
                    if line in _OBJ_CLOSE_LINES:
                        text = "".join(buffer).strip()
                        if text.endswith(","):
                            text = text[:-1]
                        yield section_name, json.loads(text)
                        buffer = []
                        in_object = False
                    continue
                if line == _OBJ_OPEN:
                    in_object = True
                    buffer = [raw_line]
                    continue
                if line in _ARRAY_CLOSE_LINES:
                    found_close = True
                    break
                raise ValueError(
                    f"{path}: unexpected line inside the {section_name} array (neither an object "
                    f"open nor the array close): {line!r}"
                )
            if in_object:
                raise ValueError(f"{path}: {section_name} stream ended mid-object -- file truncated")
            if not found_close:
                raise ValueError(f"{path}: {section_name} array's closing ']' never found")


# ---------------------------------------------------------------------------------------------
# Layer 1 -- slot gene-set reconstruction + exclusion tally, in one combined pass.
# ---------------------------------------------------------------------------------------------


def independent_recompute_from_census(
    census_path: Path,
    *,
    allowed_chroms: set[str],
    construct_length: int,
    donor_pos: int,
    acceptor_pos: int,
    donor_window: tuple[int, int],
    acceptor_window: tuple[int, int],
) -> dict[str, Any]:
    """One streaming pass producing everything the census-dependent checks need.

    Filters mirror the certificate's own filter configuration exactly (chromosome allowlist,
    scaffold construct geometry, donor/acceptor phase bucketing) -- no blacklist and no forbidden-
    chromosome filter are applied here because ``run_real_panel_capacity.py``'s own
    ``_filters_from_certificate`` never applies them either (blacklist=set(), forbidden_chroms=
    set(), always): freshness was already enforced when the census itself was built, so this
    stage's blacklist/forbidden-chromosome filters are a documented, harmless no-op in the
    authority this audit is checking. Reproducing that -- rather than inventing a stricter filter
    of this audit's own -- is required for the certificate's registered hashes to reproduce.
    """
    slot_genes: dict[str, set[str]] = {slot: set() for slot in ALL_SLOTS}
    n_candidates = 0
    phase_equal_occurrences = 0
    n_exclusions = 0
    sole_reason_counts: Counter[str] = Counter()
    raw_reason_counts: Counter[str] = Counter()

    for kind, obj in stream_census_sections(census_path):
        if kind == "candidate":
            n_candidates += 1
            phase_equal_occurrences += _count_key_occurrences(obj, _PHASE_EQUAL_KEY)
            chrom = obj.get("chrom")
            if chrom not in allowed_chroms:
                continue
            role = obj.get("role")
            gene = _gene_root(str(obj["gene_id"]))
            if role == "scaffold":
                if (
                    len(obj.get("sequence", "")) == construct_length
                    and int(obj.get("donor_pos", -1)) == donor_pos
                    and int(obj.get("acceptor_pos", -1)) == acceptor_pos
                    and tuple(obj.get("donor_window", ())) == donor_window
                    and tuple(obj.get("acceptor_window", ())) == acceptor_window
                ):
                    slot_genes["scaffold"].add(gene)
            elif role in ("donor", "acceptor"):
                phase = int(obj.get("phase", -1))
                if phase in (0, 1, 2):
                    slot_genes[f"{role}_phase_{phase}"].add(gene)
        else:  # exclusion
            n_exclusions += 1
            reasons = obj.get("reasons", [])
            for reason in reasons:
                raw_reason_counts[reason] += 1
            if len(reasons) == 1:
                sole_reason_counts[reasons[0]] += 1

    return {
        "slot_genes": slot_genes,
        "n_candidates": n_candidates,
        "phase_equal_occurrences": phase_equal_occurrences,
        "n_exclusions": n_exclusions,
        "sole_reason_counts": dict(sole_reason_counts),
        "raw_reason_counts": dict(raw_reason_counts),
        "multi_reason_exclusions": n_exclusions - sum(sole_reason_counts.values()),
    }


# ---------------------------------------------------------------------------------------------
# Layer 2 -- exact max-flow feasibility. `networkx` is the one shared low-level primitive: see
# module docstring and `shared_primitives_imported` in the emitted audit artifact.
# ---------------------------------------------------------------------------------------------


def feasibility(slot_sets: Mapping[str, set[str]], demands: Mapping[str, int]) -> dict[str, Any]:
    """Exact constructibility of `demands` against `slot_sets` by max-flow (source -> gene,
    capacity 1; gene -> slot, capacity infinity for every slot the gene is eligible for; slot ->
    sink, capacity = that slot's demand). Feasible iff max_flow == total_demand.
    """
    import networkx as nx  # imported lazily; the one declared shared primitive (see docstring)

    unknown = [slot for slot in demands if slot not in slot_sets]
    if unknown:
        raise ValueError(f"feasibility(): demand for unmodelled slot(s) {sorted(unknown)}")

    graph = nx.DiGraph()
    source, sink = ("__source__",), ("__sink__",)
    graph.add_node(source)
    graph.add_node(sink)
    total_demand = 0
    for slot, demand in demands.items():
        d = int(demand)
        total_demand += d
        graph.add_edge(("slot", slot), sink, capacity=d)
    added_genes: set[tuple[str, str]] = set()
    for slot in demands:
        for gene in slot_sets[slot]:
            gene_node = ("gene", gene)
            if gene_node not in added_genes:
                graph.add_edge(source, gene_node, capacity=1)
                added_genes.add(gene_node)
            graph.add_edge(gene_node, ("slot", slot), capacity=float("inf"))

    max_flow_value = nx.maximum_flow_value(graph, source, sink)
    feasible = max_flow_value >= total_demand
    return {"feasible": bool(feasible), "max_flow": int(max_flow_value), "total_demand": int(total_demand)}


def _largest_n_satisfying(predicate, upper_bound: int) -> int:
    """Largest non-negative N <= `upper_bound` for which `predicate(N)` holds. `predicate` must be
    monotonically non-increasing in N (true for both exact feasibility and a margin fraction,
    since demand only grows with N).
    """
    if not predicate(0):
        return 0
    lo, hi = 0, upper_bound
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if predicate(mid):
            lo = mid
        else:
            hi = mid - 1
    if lo == upper_bound:
        raise ValueError(f"_largest_n_satisfying: predicate still holds at upper_bound={upper_bound}")
    return lo


def plain_lattice_demands(n_blocks: int) -> dict[str, int]:
    demands = {slot: int(n_blocks) for slot in _PLAIN_PHASE_SLOTS}
    demands["scaffold"] = int(n_blocks)
    return demands


def _allocate_blocks_largest_remainder(n_blocks: int) -> dict[tuple[int, int], int]:
    """Split `n_blocks` across the three phase pairs (equal weights) by largest remainder."""
    exact = [n_blocks / 3.0 for _ in _PAIR_TYPES]
    floors = [int(value) for value in exact]
    shortfall = n_blocks - sum(floors)
    order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - floors[i]), i))
    for index in order[:shortfall]:
        floors[index] += 1
    return {pair: count for pair, count in zip(_PAIR_TYPES, floors)}


def rotated_demands(n_blocks: int) -> dict[str, int]:
    """Withdrawn rotated 2x2 lattice's per-slot demand -- for the comparison-only ceiling."""
    allocation = _allocate_blocks_largest_remainder(n_blocks)
    demands = {slot: 0 for slot in ALL_SLOTS}
    demands["scaffold"] = int(n_blocks)
    for (phase_a, phase_b), count in allocation.items():
        for phase in (phase_a, phase_b):
            demands[f"donor_phase_{phase}"] += count
            demands[f"acceptor_phase_{phase}"] += count
    return demands


def exact_plain_ceiling(slot_sets: Mapping[str, set[str]]) -> int:
    return _largest_n_satisfying(
        lambda n: feasibility(slot_sets, plain_lattice_demands(n))["feasible"], _CEILING_SEARCH_UPPER_BOUND
    )


def ceiling_with_assignment_slack(slot_sets: Mapping[str, set[str]], *, union_size: int) -> int:
    def predicate(n: int) -> bool:
        if not feasibility(slot_sets, plain_lattice_demands(n))["feasible"]:
            return False
        if union_size == 0:
            return False
        return (union_size - 4 * n) / union_size >= _MARGIN_THRESHOLD

    return _largest_n_satisfying(predicate, _CEILING_SEARCH_UPPER_BOUND)


def rotated_ceiling(slot_sets: Mapping[str, set[str]]) -> int:
    return _largest_n_satisfying(
        lambda n: feasibility(slot_sets, rotated_demands(n))["feasible"], _CEILING_SEARCH_UPPER_BOUND
    )


# ---------------------------------------------------------------------------------------------
# Layer 3 -- sign-free variance decomposition from the C43 replication cell rows. Never reads,
# computes, or returns a mean, a sign, or a direction of any C43 endpoint -- only variances.
# ---------------------------------------------------------------------------------------------


def compute_variance_components(cell_rows: Iterable[Mapping]) -> dict[str, Any]:
    by_block: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"matched": [], "mismatched": []})
    for row in cell_rows:
        bucket = "matched" if row["compatible"] else "mismatched"
        by_block[row["block_id"]][bucket].append(float(row["effects"]["c_ac"]))

    total_ss = 0.0
    total_df = 0
    psi_b_values: list[float] = []
    n_cell_rows = 0
    for block_id in sorted(by_block):  # deterministic order, independent of file/insertion order
        matched = by_block[block_id]["matched"]
        mismatched = by_block[block_id]["mismatched"]
        n_cell_rows += len(matched) + len(mismatched)
        if len(matched) != 3 or len(mismatched) != 6:
            raise ValueError(
                f"block {block_id!r} has {len(matched)} matched / {len(mismatched)} mismatched "
                "cells; expected exactly 3/6"
            )
        matched_mean = sum(matched) / len(matched)
        mismatched_mean = sum(mismatched) / len(mismatched)
        total_ss += sum((x - matched_mean) ** 2 for x in matched)
        total_ss += sum((x - mismatched_mean) ** 2 for x in mismatched)
        total_df += (len(matched) - 1) + (len(mismatched) - 1)
        psi_b_values.append(matched_mean - mismatched_mean)  # local only: feeds a variance, never returned

    n_blocks = len(psi_b_values)
    sigma_c2 = total_ss / total_df
    grand_mean = sum(psi_b_values) / n_blocks  # local only: feeds the variance sum, never returned
    var_psi_b = sum((p - grand_mean) ** 2 for p in psi_b_values) / (n_blocks - 1)
    block_sd = var_psi_b**0.5
    sigma_u2 = var_psi_b - sigma_c2 * (1.0 / 3.0 + 1.0 / 6.0)
    rho = sigma_u2 / var_psi_b

    return {
        "n_cell_rows": n_cell_rows,
        "n_blocks": n_blocks,
        "pooled_within_block_within_group_variance": sigma_c2,
        "pooled_within_block_within_group_df": total_df,
        "across_block_variance_psi_b": var_psi_b,
        "block_sd": block_sd,
        "sigma_u2": sigma_u2,
        "rho": rho,
    }


# ---------------------------------------------------------------------------------------------
# Layer 4 -- power at the measured components, and the cross-lineage required correlation.
# ---------------------------------------------------------------------------------------------


def _z_critical(*, alpha_two_sided: float = _ALPHA_TWO_SIDED, power: float = _POWER) -> float:
    """z_{1-alpha/2} + z_{1-beta}, derived from scipy's normal quantile function (a generic
    statistics library, not anything the producer script imports -- it hardcoded the equivalent
    literal instead).
    """
    from scipy.stats import norm  # imported lazily; independent numerical derivation of Z

    return float(norm.ppf(1.0 - alpha_two_sided / 2.0) + norm.ppf(power))


def _cell_variance(rho: float, v9: float) -> float:
    return 2.0 * (1.0 - rho) * v9


def var_psi(matched: int, mismatched: int, rho: float, v9: float) -> float:
    return rho * v9 + _cell_variance(rho, v9) * (1.0 / matched + 1.0 / mismatched)


def var_direction(delta_cells: int, matched: int, rho: float, v9: float) -> float:
    return rho * v9 + _cell_variance(rho, v9) * (1.0 / delta_cells + 1.0 / matched)


def blocks_for_sesoi(variance: float, *, z: float, sesoi: float = SESOI) -> int:
    return math.ceil(z * z * variance / (sesoi**2))


def required_cross_lineage_correlation(n_blocks: int, var_psi_block: float, *, z: float, sesoi: float = SESOI) -> float:
    """Minimum same-block L1/L2 correlation r such that Var(Delta-Psi) = 2*Var(Psi_b)*(1-r) reaches
    MDE80 <= SESOI at the given z, `n_blocks` blocks.
    """
    max_var = (sesoi**2) * n_blocks / (z**2)
    return 1.0 - max_var / (2.0 * var_psi_block)


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------


class _CheckRecorder:
    """Accumulates the per-check table. A mismatch is always a fail, never a warning."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, check_id: str, passed: bool, **detail: Any) -> bool:
        entry = {"id": check_id, "passed": bool(passed)}
        entry.update(detail)
        self.checks.append(entry)
        return bool(passed)

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _output_paths(out_path: str | Path, *, overwrite: bool) -> tuple[Path, Path]:
    out = guard_output_path(Path(out_path).expanduser().resolve(), overwrite=overwrite)
    if out.suffix != ".json":
        raise ValueError("audit output must be a .json file")
    manifest = guard_output_path(out.with_name(out.stem + ".manifest.json"), overwrite=overwrite)
    return out, manifest


def run_audit(
    *,
    artifact_path: str | Path,
    artifact_manifest_path: str | Path,
    census_path: str | Path,
    certificate_path: str | Path,
    cells_path: str | Path,
    analysis_path: str | Path,
    out_path: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    out, manifest_path = _output_paths(out_path, overwrite=overwrite)

    artifact_path = Path(artifact_path).expanduser().resolve()
    artifact_manifest_path = Path(artifact_manifest_path).expanduser().resolve()
    census_path = Path(census_path).expanduser().resolve()
    certificate_path = Path(certificate_path).expanduser().resolve()
    cells_path = Path(cells_path).expanduser().resolve()
    analysis_path = Path(analysis_path).expanduser().resolve()

    checks = _CheckRecorder()

    # CORRECTED. This previously reported whether every input EXISTS ON THIS MACHINE, which is
    # a different question -- and in fact it is the condition under which the audit is NOT
    # clean-checkout reproducible, so the field asserted the opposite of the note beside it.
    # "Clean-checkout re-executable" means: could someone who clones this repository, and has
    # nothing else, re-run this audit? They could not. The 2.34 GB candidate census and the 54 MB
    # C43 replication cells file are both untracked and live outside the repository, so they are
    # absent from any clean checkout by construction.
    repo_root = Path(__file__).resolve().parents[2]
    external_inputs = [
        str(p) for p in (census_path, cells_path)
        if repo_root not in Path(p).resolve().parents
    ] + [str(census_path), str(cells_path)]
    external_inputs = sorted(set(external_inputs))
    clean_checkout_reexecutable = False
    clean_checkout_note = (
        "The 2.34 GB candidate census and the 54 MB C43 replication cells file both live outside "
        "this repository and are untracked. This audit is NOT re-executable from a clean checkout; "
        "it only runs on a machine that already has both local files at the paths above (or "
        "supplied via --census/--cells)."
    )

    inputs_present = checks.add(
        "inputs.all_four_raw_inputs_present",
        all(p.exists() for p in (artifact_path, artifact_manifest_path, census_path, certificate_path, cells_path, analysis_path)),
        missing=[str(p) for p in (artifact_path, artifact_manifest_path, census_path, certificate_path, cells_path, analysis_path) if not p.exists()],
    )

    recomputed: dict[str, Any] = {}

    if not inputs_present:
        status = "AUDIT_FAIL"
        audit = _assemble_audit(
            status=status,
            passed=False,
            checks=checks.checks,
            recomputed=recomputed,
            artifact_path=artifact_path,
            artifact_manifest_path=artifact_manifest_path,
            clean_checkout_reexecutable=clean_checkout_reexecutable,
            external_inputs=external_inputs,
            clean_checkout_note=clean_checkout_note,
        )
        _write_audit(out, manifest_path, audit, inputs=[artifact_path, artifact_manifest_path, census_path, certificate_path, cells_path, analysis_path, Path(__file__).resolve()], overwrite=overwrite)
        return out, manifest_path, audit

    artifact = _read_json(artifact_path)
    artifact_manifest = _read_json(artifact_manifest_path)
    certificate = _read_json(certificate_path)
    analysis_doc = _read_json(analysis_path)

    if artifact.get("schema") != AUDITED_ARTIFACT_SCHEMA:
        checks.add("input_identity.audited_artifact_schema", False, expected=AUDITED_ARTIFACT_SCHEMA, observed=artifact.get("schema"))

    # ---- 7. Chronology and hashes: the artifact's own sha256 matches its sibling manifest. ----
    observed_artifact_sha256 = _sha256_file(artifact_path)
    manifest_outputs = artifact_manifest.get("outputs") or []
    manifest_output_sha256 = manifest_outputs[0].get("sha256") if manifest_outputs else None
    checks.add(
        "chronology.artifact_sha256_matches_sibling_manifest",
        observed_artifact_sha256 == manifest_output_sha256,
        observed=observed_artifact_sha256,
        manifest_recorded=manifest_output_sha256,
    )
    created_at_raw = artifact_manifest.get("created_at_utc")
    chronology_ok = False
    try:
        parsed_created_at = datetime.fromisoformat(str(created_at_raw))
        if parsed_created_at.tzinfo is None:
            parsed_created_at = parsed_created_at.replace(tzinfo=timezone.utc)
        chronology_ok = parsed_created_at <= datetime.now(timezone.utc)
    except (TypeError, ValueError):
        chronology_ok = False
    checks.add("chronology.manifest_created_at_utc_not_in_the_future", chronology_ok, created_at_utc=created_at_raw)

    # ---- 1. Input identity. ----
    census_size = census_path.stat().st_size
    checks.add("input_identity.census_size_matches_pin", census_size == _CENSUS_SIZE_PIN, expected=_CENSUS_SIZE_PIN, observed=census_size)
    census_sha256 = _sha256_file(census_path)
    checks.add("input_identity.census_sha256_matches_pin", census_sha256 == _CENSUS_SHA256_PIN, expected=_CENSUS_SHA256_PIN, observed=census_sha256)
    reported_census_sha256 = (artifact.get("census_receipt") or {}).get("sha256")
    checks.add("input_identity.census_sha256_matches_artifact_receipt", census_sha256 == reported_census_sha256, observed=census_sha256, artifact_reports=reported_census_sha256)

    certificate_sha256 = _sha256_file(certificate_path)
    reported_certificate_sha256 = (artifact.get("certificate_receipt") or {}).get("sha256")
    checks.add(
        "input_identity.certificate_sha256_matches_artifact_receipt",
        certificate_sha256 == reported_certificate_sha256,
        observed=certificate_sha256,
        artifact_reports=reported_certificate_sha256,
    )

    cells_sha256 = _sha256_file(cells_path)
    checks.add("input_identity.cells_sha256_matches_pin", cells_sha256 == _CELLS_SHA256_PIN, expected=_CELLS_SHA256_PIN, observed=cells_sha256)
    analysis_cells_sha256 = analysis_doc.get("cells_result_sha256")
    checks.add(
        "input_identity.cells_sha256_equals_analysis_cells_result_sha256",
        cells_sha256 == analysis_cells_sha256 == _CELLS_SHA256_PIN,
        observed_cells_sha256=cells_sha256,
        analysis_cells_result_sha256=analysis_cells_sha256,
        note="the ONLY thing binding the untracked C43 cells file into the record",
    )
    reported_cells_sha256 = (artifact.get("cells_receipt") or {}).get("sha256")
    checks.add("input_identity.cells_sha256_matches_artifact_receipt", cells_sha256 == reported_cells_sha256, observed=cells_sha256, artifact_reports=reported_cells_sha256)

    analysis_sha256 = _sha256_file(analysis_path)
    reported_analysis_sha256 = (artifact.get("analysis_receipt") or {}).get("sha256")
    checks.add(
        "input_identity.analysis_sha256_matches_artifact_receipt",
        analysis_sha256 == reported_analysis_sha256,
        observed=analysis_sha256,
        artifact_reports=reported_analysis_sha256,
    )

    # ---- Filters from the certificate (mirrors the same config keys the producer reads). ----
    config = certificate["config"]
    allowed_chroms = {_normalize_chromosome(c) for c in config["allowed_chromosomes"]}
    construct_length = int(config["construct_length"])
    donor_pos = int(config["donor_pos"])
    acceptor_pos = int(config["acceptor_pos"])

    # ---- One combined streaming pass: slot gene sets + exclusion tally + phase-equal tally. ----
    combined = independent_recompute_from_census(
        census_path,
        allowed_chroms=allowed_chroms,
        construct_length=construct_length,
        donor_pos=donor_pos,
        acceptor_pos=acceptor_pos,
        donor_window=DEFAULT_DONOR_WINDOW,
        acceptor_window=DEFAULT_ACCEPTOR_WINDOW,
    )
    slot_genes = combined["slot_genes"]

    # ---- 2. Seven slot gene-set hashes reproduce. ----
    cert_slots = certificate["capacity"]["slots"]
    slot_hashes = {}
    for slot in ALL_SLOTS:
        recomputed_hash = _sha256_payload(sorted(slot_genes[slot]))
        slot_hashes[slot] = recomputed_hash
        expected_hash = cert_slots[slot]["gene_ids_sha256"]
        checks.add(f"slot_hashes.{slot}", recomputed_hash == expected_hash, observed=recomputed_hash, certificate=expected_hash)
    reported_slot_verification = artifact.get("slot_hash_verification") or {}
    checks.add(
        "slot_hashes.artifact_claims_all_verified",
        reported_slot_verification.get("all_verified") is True
        and all(reported_slot_verification.get("per_slot", {}).get(slot) is True for slot in ALL_SLOTS),
        artifact_slot_hash_verification=reported_slot_verification,
    )

    # ---- 4. Ceilings by an independent max-flow call. ----
    union4 = len(set().union(*(slot_genes[s] for s in _PLAIN_PHASE_SLOTS)))
    exact_ceiling = exact_plain_ceiling(slot_genes)
    slack_ceiling = ceiling_with_assignment_slack(slot_genes, union_size=union4)
    rotated_ceil = rotated_ceiling(slot_genes)
    intersections = {phase: len(slot_genes[f"donor_phase_{phase}"] & slot_genes[f"acceptor_phase_{phase}"]) for phase in (0, 1, 2)}

    b2 = artifact["b2_real_panel_ceilings"]
    checks.add("ceilings.union_of_four_phase01_module_slots", union4 == b2["union_of_four_phase01_module_slots"], observed=union4, artifact_reports=b2["union_of_four_phase01_module_slots"])
    checks.add("ceilings.exact_ceiling", exact_ceiling == b2["exact_ceiling"], observed=exact_ceiling, artifact_reports=b2["exact_ceiling"])
    checks.add(
        "ceilings.ceiling_with_25pct_assignment_slack",
        slack_ceiling == b2["ceiling_with_25pct_assignment_slack"],
        observed=slack_ceiling,
        artifact_reports=b2["ceiling_with_25pct_assignment_slack"],
    )
    checks.add(
        "ceilings.rotated_lattice_ceiling_for_comparison",
        rotated_ceil == b2["rotated_lattice_ceiling_for_comparison"],
        observed=rotated_ceil,
        artifact_reports=b2["rotated_lattice_ceiling_for_comparison"],
    )
    for phase in (0, 1, 2):
        reported = b2["donor_acceptor_phase_intersections"][str(phase)]
        checks.add(
            f"ceilings.donor_acceptor_phase_intersection_{phase}",
            intersections[phase] == reported,
            observed=intersections[phase],
            artifact_reports=reported,
        )

    # ---- 6. The exclusion tally totals 282,057. ----
    n_exclusions = combined["n_exclusions"]
    checks.add("exclusions.total_matches_pin", n_exclusions == _EXPECTED_EXCLUSIONS_TOTAL, expected=_EXPECTED_EXCLUSIONS_TOTAL, observed=n_exclusions)
    b5 = artifact["b5_filter_relaxation_closed"]
    checks.add("exclusions.total_matches_artifact", n_exclusions == b5["census_exclusions_total"], observed=n_exclusions, artifact_reports=b5["census_exclusions_total"])
    checks.add(
        "exclusions.sole_reason_counts_match_artifact",
        combined["sole_reason_counts"] == b5["sole_reason_counts"],
        observed=combined["sole_reason_counts"],
        artifact_reports=b5["sole_reason_counts"],
    )
    checks.add(
        "exclusions.multi_reason_total_matches_artifact",
        combined["multi_reason_exclusions"] == b5["multi_reason_exclusions"],
        observed=combined["multi_reason_exclusions"],
        artifact_reports=b5["multi_reason_exclusions"],
    )
    phase_equal_occurrences = combined["phase_equal_occurrences"]
    phase_equal_candidates = phase_equal_occurrences // 2
    reported_phase_equal = b5["upstream_downstream_phase_equal"]
    checks.add(
        "exclusions.phase_equal_occurrences_match_artifact",
        phase_equal_occurrences == reported_phase_equal["occurrences"],
        observed=phase_equal_occurrences,
        artifact_reports=reported_phase_equal["occurrences"],
    )
    checks.add(
        "exclusions.phase_equal_occurrences_is_exactly_2x_candidates",
        phase_equal_occurrences == 2 * phase_equal_candidates and phase_equal_candidates == combined["n_candidates"],
        occurrences=phase_equal_occurrences,
        candidates_from_occurrences=phase_equal_candidates,
        n_candidates_streamed=combined["n_candidates"],
    )
    checks.add(
        "exclusions.phase_equal_candidates_match_artifact",
        phase_equal_candidates == reported_phase_equal["candidates"],
        observed=phase_equal_candidates,
        artifact_reports=reported_phase_equal["candidates"],
    )
    checks.add(
        "exclusions.phase_equal_never_a_raw_exclusion_reason",
        combined["raw_reason_counts"].get(_PHASE_EQUAL_KEY, 0) == 0,
        observed=combined["raw_reason_counts"].get(_PHASE_EQUAL_KEY, 0),
    )

    # ---- 3. Variance components (sign-free: variances only, never a mean/sign/direction). ----
    cells_doc = _read_json(cells_path)
    b1_recomputed = compute_variance_components(cells_doc["cell_rows"])
    b1_reported = artifact["b1_measured_variance_components"]
    for field in ("n_cell_rows", "n_blocks", "pooled_within_block_within_group_df"):
        checks.add(f"variance.{field}_matches_artifact", b1_recomputed[field] == b1_reported[field], observed=b1_recomputed[field], artifact_reports=b1_reported[field])
    for field in (
        "pooled_within_block_within_group_variance",
        "across_block_variance_psi_b",
        "block_sd",
        "sigma_u2",
        "rho",
    ):
        observed = b1_recomputed[field]
        reported = b1_reported[field]
        checks.add(
            f"variance.{field}_matches_artifact",
            math.isclose(observed, reported, rel_tol=1e-9, abs_tol=1e-12),
            observed=observed,
            artifact_reports=reported,
        )

    rho = b1_recomputed["rho"]
    v9 = b1_recomputed["across_block_variance_psi_b"]

    # ---- 5a. Power figures. ----
    z = _z_critical()
    b3_reported = artifact["b3_power_at_measured_components"]
    power_recomputed = {
        "psi_plain_2x2": blocks_for_sesoi(var_psi(2, 2, rho, v9), z=z),
        "m_pm_plain_2x2": blocks_for_sesoi(var_direction(1, 2, rho, v9), z=z),
        "psi_full_3x3": blocks_for_sesoi(var_psi(3, 6, rho, v9), z=z),
        "m_pm_full_3x3": blocks_for_sesoi(var_direction(3, 3, rho, v9), z=z),
    }
    for key, observed in power_recomputed.items():
        reported = b3_reported[key]["blocks_needed"]
        checks.add(f"power.{key}_blocks_needed_matches_artifact", observed == reported, observed=observed, artifact_reports=reported)

    # ---- 5b. Cross-lineage: the plain {0,1} lattice variance (2 - rho) * V9, NOT raw V9. ----
    b4_reported = artifact["b4_cross_lineage_reframe"]
    var_psi_plain_correct = (2.0 - rho) * v9  # the correct per-block variance for m=2, u=2
    var_psi_plain_wrong = v9  # what the raw 3x3 V9 (the documented bug) would give instead
    checks.add(
        "cross_lineage.var_psi_per_block_used_is_plain_lattice_not_raw_v9",
        math.isclose(b4_reported["var_psi_per_block_used"], var_psi_plain_correct, rel_tol=1e-9),
        observed_correct_formula=var_psi_plain_correct,
        artifact_reports=b4_reported["var_psi_per_block_used"],
        raw_v9_bug_would_give=var_psi_plain_wrong,
    )

    usable_ceiling = slack_ceiling
    absolute_ceiling = exact_ceiling
    required_correlation_recomputed: dict[int, float] = {}
    for n_blocks in (usable_ceiling, absolute_ceiling):
        r_correct = required_cross_lineage_correlation(n_blocks, var_psi_plain_correct, z=z)
        r_wrong = required_cross_lineage_correlation(n_blocks, var_psi_plain_wrong, z=z)
        required_correlation_recomputed[n_blocks] = r_correct
        reported_row = b4_reported["required_same_block_correlation"].get(str(n_blocks), {})
        reported_r = reported_row.get("required_same_block_correlation_min")
        checks.add(
            f"cross_lineage.required_correlation_{n_blocks}_matches_recomputed_correct_formula",
            reported_r is not None and math.isclose(reported_r, r_correct, rel_tol=1e-6),
            observed_correct=r_correct,
            artifact_reports=reported_r,
        )
        expected_pin = _EXPECTED_REQUIRED_CORRELATION.get(n_blocks)
        if expected_pin is not None:
            checks.add(
                f"cross_lineage.required_correlation_{n_blocks}_matches_expected_pin",
                reported_r is not None and abs(reported_r - expected_pin) <= _CORRELATION_TOLERANCE,
                expected_pin=expected_pin,
                artifact_reports=reported_r,
            )
        wrong_pin = _WRONG_REQUIRED_CORRELATION_RAW_V9.get(n_blocks)
        if wrong_pin is not None:
            checks.add(
                f"cross_lineage.required_correlation_{n_blocks}_is_not_the_raw_v9_bug_value",
                reported_r is None or abs(reported_r - wrong_pin) > _CORRELATION_TOLERANCE,
                raw_v9_bug_value=wrong_pin,
                artifact_reports=reported_r,
                recomputed_raw_v9_value=r_wrong,
            )
        # n_blocks used above must themselves match the artifact's own ceiling labels.
        checks.add(
            f"cross_lineage.n_blocks_{n_blocks}_matches_artifact_ceiling_label",
            str(n_blocks) in b4_reported["required_same_block_correlation"],
            n_blocks=n_blocks,
        )

    recomputed = {
        "b1_variance_components": b1_recomputed,
        "b2_ceilings": {
            "union_of_four_phase01_module_slots": union4,
            "exact_ceiling": exact_ceiling,
            "ceiling_with_25pct_assignment_slack": slack_ceiling,
            "rotated_lattice_ceiling_for_comparison": rotated_ceil,
            "donor_acceptor_phase_intersections": {str(k): v for k, v in intersections.items()},
        },
        "b3_power_at_measured_components": power_recomputed,
        "b4_cross_lineage": {
            "z_critical": z,
            "var_psi_per_block_used": var_psi_plain_correct,
            "var_psi_per_block_raw_v9_alternative": var_psi_plain_wrong,
            "required_same_block_correlation": {str(k): v for k, v in required_correlation_recomputed.items()},
        },
        "b5_exclusions": {
            "total": n_exclusions,
            "sole_reason_counts": combined["sole_reason_counts"],
            "multi_reason_exclusions": combined["multi_reason_exclusions"],
            "upstream_downstream_phase_equal": {"occurrences": phase_equal_occurrences, "candidates": phase_equal_candidates},
        },
        "n_candidates_streamed": combined["n_candidates"],
        "slot_gene_set_sizes": {slot: len(genes) for slot, genes in slot_genes.items()},
        "slot_hashes": slot_hashes,
    }

    status = "AUDIT_PASS" if checks.all_passed else "AUDIT_FAIL"
    audit = _assemble_audit(
        status=status,
        passed=checks.all_passed,
        checks=checks.checks,
        recomputed=recomputed,
        artifact_path=artifact_path,
        artifact_manifest_path=artifact_manifest_path,
        clean_checkout_reexecutable=clean_checkout_reexecutable,
        external_inputs=external_inputs,
        clean_checkout_note=clean_checkout_note,
    )
    _write_audit(
        out,
        manifest_path,
        audit,
        inputs=[artifact_path, artifact_manifest_path, census_path, certificate_path, cells_path, analysis_path, Path(__file__).resolve()],
        overwrite=overwrite,
    )
    return out, manifest_path, audit


def _assemble_audit(
    *,
    status: str,
    passed: bool,
    checks: list[dict[str, Any]],
    recomputed: dict[str, Any],
    artifact_path: Path,
    artifact_manifest_path: Path,
    clean_checkout_reexecutable: bool,
    external_inputs: list[str],
    clean_checkout_note: str,
) -> dict[str, Any]:
    return {
        "schema": AUDIT_SCHEMA,
        "status": status,
        "passed": bool(passed),
        "stage": "construction",
        "audited_artifact": {
            "path": _repo_relative_or_absolute(artifact_path),
            "schema": AUDITED_ARTIFACT_SCHEMA,
        },
        "audited_artifact_manifest": {"path": _repo_relative_or_absolute(artifact_manifest_path)},
        "producer_script": PRODUCER_SCRIPT,
        "independence": {
            "producer_verdict_functions_imported": [],
            "shared_primitives_imported": [
                "networkx (nx.maximum_flow_value / nx.DiGraph -- the max-flow solver backing both "
                "this audit's feasibility() and the producer's own feasibility() helper; a bug in "
                "networkx's max-flow implementation would be invisible to this audit)"
            ],
            "note": (
                "This module imports nothing from scripts/splicebert_positional_clock/"
                "run_real_panel_capacity.py, gfmcd.splicing.pbc_v2_capacity, or "
                "gfmcd.splicing.positional_boundary_coherence. Every reported quantity -- the "
                "census streamer, slot-bucketing filter, sha256/canonical-JSON hashing, sign-free "
                "variance decomposition, power formulas, rotated-lattice demand model, and exact "
                "max-flow feasibility -- is reimplemented from scratch on this file's own code "
                "path. The one declared exception is networkx itself, listed above."
            ),
        },
        "clean_checkout_reexecutable": bool(clean_checkout_reexecutable),
        "clean_checkout_blocking_inputs": external_inputs,
        "clean_checkout_note": clean_checkout_note,
        "checks": checks,
        "counts": {"total_checks": len(checks), "passed_checks": sum(1 for c in checks if c["passed"]), "failed_checks": sum(1 for c in checks if not c["passed"])},
        "recomputed": recomputed,
        "model_execution": False,
        "checkpoint_loaded": None,
        "cohort_selection_attempted": False,
    }


def _repo_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _write_audit(out: Path, manifest_path: Path, audit: dict[str, Any], *, inputs: list[Path], overwrite: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        EXPERIMENT_ID,
        sys.argv,
        repo=REPO,
        status=audit["status"],
        config={"stage": "construction", "checkpoint_loaded": None},
        seeds={},
        notes=[
            "Independent audit: recomputes every B1-B5 quantity in real_panel_capacity_v1.json "
            "from the same four raw inputs, on its own code path.",
            "Imports no verdict function from run_real_panel_capacity.py or from any "
            "gfmcd.splicing module; the one shared low-level primitive (networkx's max-flow "
            "solver) is declared in the audit artifact's independence.shared_primitives_imported.",
            "Not clean-checkout re-executable: the census and C43 cells file are untracked and "
            "live outside this repository.",
        ],
    )
    for path in inputs:
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", default=str(DEFAULT_ARTIFACT_PATH), help="path to real_panel_capacity_v1.json under audit")
    ap.add_argument("--artifact-manifest", default=str(DEFAULT_ARTIFACT_MANIFEST_PATH), dest="artifact_manifest", help="path to real_panel_capacity_v1.manifest.json")
    ap.add_argument("--census", default=str(DEFAULT_CENSUS_PATH), help="path to candidate_census_v1.json (untracked, outside this repo)")
    ap.add_argument("--certificate", default=str(DEFAULT_CERTIFICATE_PATH), help="path to composition_balanced_cohort_v1.json")
    ap.add_argument("--cells", default=str(DEFAULT_CELLS_PATH), help="path to crossed_direct_edge_replication_cells.json (untracked, outside this repo)")
    ap.add_argument("--analysis", default=str(DEFAULT_ANALYSIS_PATH), help="path to crossed_direct_edge_replication_analysis.json")
    ap.add_argument("--out", required=True, help="audit output artifact path (required; no default)")
    ap.add_argument(
        "--overwrite-record",
        action="store_true",
        dest="overwrite_record",
        help="replace an existing audit artifact of record (results are immutable by default)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out, manifest_path, audit = run_audit(
        artifact_path=args.artifact,
        artifact_manifest_path=args.artifact_manifest,
        census_path=args.census,
        certificate_path=args.certificate,
        cells_path=args.cells,
        analysis_path=args.analysis,
        out_path=args.out,
        overwrite=args.overwrite_record,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "passed": audit["passed"],
                "counts": audit["counts"],
                "output": str(out),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not audit["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
