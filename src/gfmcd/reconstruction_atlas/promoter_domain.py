"""Endpoint estimators for the D12 downstream-domain assay.

Pure functions over per-locus tile rows. No model, no I/O beyond reading a
raw JSONL emitted by the source-localization runner.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

__all__ = [
    "TileRow",
    "check_memorials",
    "contrast_by_locus",
    "load_clusters",
    "paired_tile_difference",
    "load_tile_rows",
    "single_tile_by_locus",
]


@dataclass(frozen=True)
class TileRow:
    pair_id: str
    cluster: str
    tile_index: int
    target_specific_loss: float
    target_cover_length: int
    control_cover_length: int
    token_count: int


def load_clusters(cohort_path: str | Path) -> dict[str, str]:
    """Map example_id -> homology_cluster.

    The tile rows do not carry the cluster; it lives only in the cohort. Every
    clustered statistic depends on this join being correct.
    """

    clusters: dict[str, str] = {}
    with Path(cohort_path).open() as handle:
        for line in handle:
            record = json.loads(line)
            clusters[str(record["example_id"])] = str(record["homology_cluster"])
    return clusters


def load_tile_rows(
    path: str | Path,
    *,
    clusters: Mapping[str, str],
    operation: str = "mask",
) -> list[TileRow]:
    rows: list[TileRow] = []
    with Path(path).open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("row_type") != "tile_effect":
                continue
            if record.get("operation") != operation:
                continue
            pair_id = str(record["pair_id"])
            if pair_id not in clusters:
                raise KeyError(
                    f"pair_id {pair_id!r} missing from the cohort cluster map; "
                    "refusing to fall back to a per-example cluster"
                )
            rows.append(
                TileRow(
                    pair_id=pair_id,
                    cluster=clusters[pair_id],
                    tile_index=int(record["tile_index"]),
                    target_specific_loss=float(record["target_specific_loss"]),
                    target_cover_length=int(record["target_cover_end"])
                    - int(record["target_cover_start"]),
                    control_cover_length=int(record["control_cover_end"])
                    - int(record["control_cover_start"]),
                    token_count=len(record["target_token_indices"]),
                )
            )
    return rows


def _by_pair(rows: Iterable[TileRow]) -> dict[str, dict[int, TileRow]]:
    grouped: dict[str, dict[int, TileRow]] = defaultdict(dict)
    for row in rows:
        grouped[row.pair_id][row.tile_index] = row
    return grouped


def contrast_by_locus(
    rows: Sequence[TileRow],
    *,
    inner_tiles: Sequence[int],
    outer_tiles: Sequence[int],
    require_all: bool = False,
) -> tuple[list[float], list[str]]:
    """Per-locus mean(inner) - mean(outer).

    With ``require_all=False`` (the config-v2 rule) a locus contributes if at
    least one inner and at least one outer tile are present, and the means are
    taken over whichever are available. ``require_all=True`` restores the v1
    rule, which demanded every named tile and collapsed the usable cohort.
    """

    values: list[float] = []
    clusters: list[str] = []
    for _pair_id, tiles in sorted(_by_pair(rows).items()):
        inner_present = [index for index in inner_tiles if index in tiles]
        outer_present = [index for index in outer_tiles if index in tiles]
        if require_all:
            if len(inner_present) != len(inner_tiles):
                continue
            if len(outer_present) != len(outer_tiles):
                continue
        elif not inner_present or not outer_present:
            continue
        inner = mean(tiles[index].target_specific_loss for index in inner_present)
        outer = mean(tiles[index].target_specific_loss for index in outer_present)
        values.append(inner - outer)
        clusters.append(tiles[inner_present[0]].cluster)
    return values, clusters


def paired_tile_difference(
    rows: Sequence[TileRow], *, tile_a: int, tile_b: int
) -> tuple[list[float], list[str]]:
    """Per-locus target_specific_loss(tile_a) - target_specific_loss(tile_b).

    Both tiles are required: this endpoint's whole point is that it needs only
    two tiles per locus, which is why it retains far more loci than a four-tile
    contrast.
    """

    values: list[float] = []
    clusters: list[str] = []
    for _pair_id, tiles in sorted(_by_pair(rows).items()):
        if tile_a not in tiles or tile_b not in tiles:
            continue
        values.append(
            tiles[tile_a].target_specific_loss - tiles[tile_b].target_specific_loss
        )
        clusters.append(tiles[tile_a].cluster)
    return values, clusters


def single_tile_by_locus(
    rows: Sequence[TileRow], *, tile_index: int
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    clusters: list[str] = []
    for _pair_id, tiles in sorted(_by_pair(rows).items()):
        row = tiles.get(tile_index)
        if row is None:
            continue
        values.append(row.target_specific_loss)
        clusters.append(row.cluster)
    return values, clusters


def check_memorials(
    rows: Sequence[TileRow],
    *,
    expected_tokens: int,
    max_cover_difference_nt: int,
) -> dict[str, bool]:
    """Evaluate the on-path memorials M3 (cover-length balance) and M4 (equal
    token dose) over ``rows``.

    Raises ``ValueError`` if ``rows`` is empty. Python's ``all()`` is
    vacuously true on an empty sequence, so without this guard both
    memorials would silently report "passed" having measured zero rows —
    indistinguishable from a real pass, and this function gates a run that
    opens sealed data. A memorial that cannot fail is not a memorial.
    """

    if not rows:
        raise ValueError(
            "check_memorials cannot be evaluated on zero rows: an empty "
            "sequence would vacuously report both memorials as passed "
            "under all(), which would silently mask that nothing was "
            "measured"
        )
    balanced = all(
        abs(row.target_cover_length - row.control_cover_length)
        <= max_cover_difference_nt
        for row in rows
    )
    dosed = all(row.token_count == expected_tokens for row in rows)
    return {"M3_cover_balance": balanced, "M4_equal_dose": dosed}
