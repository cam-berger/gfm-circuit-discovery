#!/usr/bin/env python
"""Independently audit the complete historical replay before authorizing CC.

This auditor reads data, reconstructs requests, and performs CPU arithmetic. It
does not import the production measurement/statistics modules or archived code.
The command accepts exactly the pinned 384-block C43 reference, never a subset.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
from itertools import permutations
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402

SCHEMA = "gfmcd.splicebert_register_operator_replication.fidelity.v1"
AUDIT_SCHEMA = "gfmcd.splicebert_register_operator_replication.fidelity_audit.v1"
RAW_TOLERANCE = 1e-4
LOG_PROBABILITY_TOLERANCE = 1e-4
REPEAT_TOLERANCE = 1e-8
INFERENCE_TOLERANCE = 1e-10
BLOCKS = 384
IDS = {"CLS": 2, "SEP": 3, "MASK": 4, "A": 6, "C": 7, "G": 8, "T": 9}
PROBES = {
    "donor": (64, 96, 128, 160, 192, 224, 256, 288, 320, 344, 360, 376),
    "acceptor": (528, 544, 560, 576, 608, 640, 672, 704, 736, 768, 800, 832),
}
PINS = {
    "cohort": "478838e64a3a3eff2bca5868212852a5b833d0ccf74ff720bb15cd00cf6fd5ca",
    "cells": "428ad5d6b357bfba287fcfe3e358b00ab91ddcd748118b83b60b626593c440bf",
    "native": "e67649a908789a26acb2086819ddd98eb9f374cd7e29db4749a4d57af665e79d",
    "audit": "b2b15d99a87ac758a98fb82eeef1ba322a2813fd7133c119945ef1eee73113ac",
    "fasta": "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51",
}
DIRECTIONS = ("m0_minus_plus", "m0_minus_minus", "donor_phase_0_excess",
              "donor_phase_1_excess", "donor_phase_2_excess")
MDE_MULTIPLIER = 1.96 + 0.8416212335729143


def digest(value: Any) -> str:
    """The masked-input contract: UTF-8 JSON with compact separators."""
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Checks:
    def __init__(self) -> None:
        self.total = 0
        self.failed = 0
        self.failures: list[str] = []
        self.maxima: dict[str, float] = {}

    def check(self, condition: bool, label: str) -> None:
        self.total += 1
        if not condition:
            self.failed += 1
            if len(self.failures) < 50:
                self.failures.append(label)

    def numeric(self, actual: Any, expected: float, tolerance: float, label: str,
                family: str) -> None:
        valid = isinstance(actual, (int, float)) and not isinstance(actual, bool)
        valid = valid and math.isfinite(float(actual)) and math.isfinite(float(expected))
        if valid:
            difference = abs(float(actual) - float(expected))
            self.maxima[family] = max(self.maxima.get(family, 0.0), difference)
            self.check(difference <= tolerance, label)
        else:
            self.check(False, label + ": missing or non-finite")

    def result(self) -> dict[str, Any]:
        return {"passed": self.failed == 0, "checks": self.total,
                "failed_checks": self.failed, "first_failures": self.failures,
                "maximum_absolute_discrepancies": self.maxima}


@dataclass(frozen=True)
class ExpectedRequest:
    request_id: str
    sequence: str
    position: int
    true_base: str
    linear_margin: float | None = None
    log_probability: float | None = None
    correct: bool | None = None


class IndexedFasta:
    """Small independent read-only FASTA-index reader, without model libraries."""
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("rb")
        self.index = {}
        for line in Path(str(path) + ".fai").read_text().splitlines():
            fields = line.split("\t")
            self.index[fields[0]] = tuple(map(int, fields[1:5]))

    def close(self) -> None:
        self.handle.close()

    def fetch(self, chrom: str, start: int, end: int, strand: str) -> str:
        length, offset, bases, width = self.index[chrom]
        require(0 <= start < end <= length and strand in {"+", "-"}, "invalid genomic slice")
        left = offset + (start // bases) * width + start % bases
        right = offset + ((end - 1) // bases) * width + (end - 1) % bases + 1
        self.handle.seek(left)
        sequence = self.handle.read(right - left).replace(b"\n", b"").decode().upper()
        require(len(sequence) == end - start and not set(sequence) - set("ACGT"),
                "genomic slice is not exact ACGT")
        return sequence if strand == "+" else sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def expected_requests(cohort: Mapping[str, Any], cells: Mapping[str, Any],
                      native: Mapping[str, Any], fetch: Callable[[str, int, int, str], str],
                      *, expected_blocks: int = BLOCKS) -> dict[str, list[ExpectedRequest]]:
    """Reconstruct every historical state/probe independently of the producer."""
    blocks = cohort["blocks"]
    require(len(blocks) == expected_blocks, "historical cohort block count differs")
    require(len({b["block_id"] for b in blocks}) == expected_blocks, "duplicate historical block")
    raw_cells = cells["cell_rows"]
    raw_native = native["native_rows"]
    require(len(raw_cells) == 9 * expected_blocks, "historical cell coverage differs")
    require(len(raw_native) == 6 * expected_blocks, "historical native coverage differs")
    cell_map = {(r["block_id"], r["p"], r["q"]): r for r in raw_cells}
    native_map = {(r["block_id"], r["source_role"], r["phase"]): r for r in raw_native}
    require(len(cell_map) == len(raw_cells) and len(native_map) == len(raw_native),
            "duplicate historical raw row")
    outputs: dict[str, dict[str, ExpectedRequest]] = {"cell": {}, "native": {}}

    def add(stage: str, request: ExpectedRequest) -> None:
        require(len(request.sequence) == 1022 and request.sequence[request.position] == request.true_base,
                "reference request does not mask its stated base")
        old = outputs[stage].get(request.request_id)
        require(old is None or old == request, "conflicting historical source probe")
        outputs[stage][request.request_id] = request

    def fragment(record: Mapping[str, Any]) -> str:
        sequence = fetch(record["chrom"], record["genomic_start"], record["genomic_end"], record["strand"])
        require(sequence_digest(sequence) == record["sequence_sha256"], "fragment hash differs")
        return sequence

    def window(record: Mapping[str, Any]) -> str:
        sequence = fetch(record["chrom"], record["window_start"], record["window_end"], record["strand"])
        require(sequence_digest(sequence) == record["sequence_sha256"], "native window hash differs")
        return sequence

    for block in blocks:
        bid = block["block_id"]
        bridge = fragment(block["bridge"])
        require(len(bridge) == 24 and bridge[0] == "A" and bridge[-1] == "T"
                and bridge[11:13] == "AG" and bridge.count("AG") == 1, "bridge geometry differs")
        donors, acceptors, source_windows = {}, {}, {}
        for role, records in (("donor", block["donors"]), ("acceptor", block["acceptors"])):
            require(set(records) == {"0", "1", "2"}, "missing source phase")
            for phase in range(3):
                source = records[str(phase)]
                require(source["boundary_phase"] == phase, "source phase differs")
                nat = window(source["native_window"])
                row = native_map[(bid, role, phase)]
                require(row["source_id"] == source["source_id"] and row["chrom"] == block["chrom"]
                        and row["strand"] == block["strand"], "native source identity differs")
                dpos = int(source["native_donor_pos"])
                require(dpos == 512 - int(source["intron_length"]) and nat[dpos:dpos + 2] == "GT"
                        and nat[510:512] == "AG", "native motif/mask geometry differs")
                require(abs(row["c_ac"] - (row["donor_margin_intact"] - row["donor_margin_true_acceptor_ac"])) <= 1e-12,
                        "historical native subtraction differs")
                key = f"native:{bid}:{role}:p{phase}:{source['source_id']}"
                for state, sequence, position, field in (
                    ("donor_intact", nat, dpos, "donor_margin_intact"),
                    ("donor_true_acceptor_ac", nat[:510] + "AC" + nat[512:], dpos, "donor_margin_true_acceptor_ac"),
                    ("acceptor_intact", nat, 511, "acceptor_margin_intact"),
                ):
                    add("native", ExpectedRequest(f"{key}:{state}", sequence, position, "G", float(row[field])))
                if role == "donor":
                    donors[phase] = fragment(source["donor_fragment"])
                    boundary = int(source["donor_boundary"])
                    start = boundary - 392 if source["strand"] == "+" else boundary + 392 - 1022
                    aligned = fetch(source["chrom"], start, start + 1022, source["strand"])
                    require(aligned[:424] == donors[phase], "donor protected source alignment differs")
                    source_windows[(role, phase)] = aligned
                else:
                    acceptors[phase] = fragment(source["acceptor_fragment"])
                    require(nat[448:] == acceptors[phase], "acceptor protected source alignment differs")
                    source_windows[(role, phase)] = nat
        public = {(r["p"], r["q"]): r for r in block["cells"]}
        require(len(public) == 9, "historical construction lacks full lattice")
        for p in range(3):
            for q in range(3):
                sequence = donors[p] + bridge + acceptors[q]
                require(sequence_digest(sequence) == public[(p, q)]["sequence_sha256"], "cell sequence hash differs")
                require(sequence[392:394] == "GT" and sequence[435:437] == "AG"
                        and sequence[510:512] == "AG", "cell site geometry differs")
                require(all(sequence[k-1:k+1] not in {"AG", "GT", "GC"} for k in (424, 448)),
                        "cell seam creates motif")
                row = cell_map[(bid, p, q)]
                require(row["donor_phase"] == p and row["acceptor_phase"] == q
                        and row["phase_gap"] == (q - p) % 3 and row["compatible"] is (p == q),
                        "historical cell phase labels differ")
                require(row["chrom"] == block["chrom"] and row["strand"] == block["strand"]
                        and row["donor_source_id"] == block["donors"][str(p)]["source_id"]
                        and row["acceptor_source_id"] == block["acceptors"][str(q)]["source_id"],
                        "historical cell source identity differs")
                key = f"{bid}:p{p}:q{q}"
                margins = row["donor_margins"]
                for state, seq, field in (
                    ("donor_intact", sequence, "intact"),
                    ("donor_true_acceptor_ac", sequence[:510] + "AC" + sequence[512:], "true_acceptor_ac"),
                    ("donor_true_acceptor_at", sequence[:510] + "AT" + sequence[512:], "true_acceptor_at"),
                    ("donor_bridge_ac", sequence[:435] + "AC" + sequence[437:], "bridge_ac"),
                ):
                    add("cell", ExpectedRequest(f"{key}:{state}", seq, 392, "G", float(margins[field])))
                add("cell", ExpectedRequest(f"{key}:acceptor_intact", sequence, 511, "G", float(row["acceptor_margin_intact"])))
                for effect, field in (("c_ac", "true_acceptor_ac"), ("c_at", "true_acceptor_at"), ("n_bridge", "bridge_ac")):
                    require(abs(row["effects"][effect] - (margins["intact"] - margins[field])) <= 1e-12,
                            "historical cell subtraction differs")
                probes = row["protected_probes"]
                expected_positions = [(module, position) for module, positions in PROBES.items() for position in positions]
                require([(v["module"], v["position"]) for v in probes] == expected_positions,
                        "historical protected probe coverage/order differs")
                pending_sources = []
                for probe in probes:
                    module, pos = probe["module"], probe["position"]
                    phase = p if module == "donor" else q
                    source_id = row[f"{module}_source_id"]
                    construct_id = f"{key}:health:{module}:{pos}:construct"
                    source_id_key = f"source:{module}:{source_id}:health:{pos}"
                    require(probe["construct_request_id"] == construct_id and probe["source_request_id"] == source_id_key,
                            "historical probe request identity differs")
                    require(probe["true_base"] == sequence[pos], "historical protected base differs")
                    require(type(probe["construct_correct"]) is bool and type(probe["source_correct"]) is bool,
                            "historical accuracy flag is not boolean")
                    add("cell", ExpectedRequest(construct_id, sequence, pos, probe["true_base"],
                        log_probability=float(probe["construct_true_base_log_probability"]), correct=probe["construct_correct"]))
                    pending_sources.append(ExpectedRequest(source_id_key, source_windows[(module, phase)], pos,
                        probe["true_base"], log_probability=float(probe["source_true_base_log_probability"]), correct=probe["source_correct"]))
                for request in pending_sources:
                    add("cell", request)
    require(len(outputs["cell"]) == 333 * expected_blocks and len(outputs["native"]) == 18 * expected_blocks,
            "historical unique-request counts differ")
    return {stage: list(values.values()) for stage, values in outputs.items()}


def audit_records(payload: Mapping[str, Any], expected: Mapping[str, Sequence[ExpectedRequest]],
                  *, cohort_sha256: str, freeze_sha256: str) -> dict[str, Any]:
    """Compare every request, including fields not present in historical summaries."""
    checks = Checks()
    checks.check(payload.get("schema") == SCHEMA and payload.get("stage") == "fidelity", "fidelity header")
    checks.check(payload.get("status") == "COMPLETE", "fidelity completed without instrument error")
    checks.check(type(payload.get("batch_size")) is int and payload["batch_size"] == 16,
                 "fidelity frozen batch size")
    checks.check(payload.get("cohort_sha256") == cohort_sha256, "fidelity cohort binding")
    checks.check(payload.get("freeze_sha256") == freeze_sha256, "fidelity freeze binding")
    checks.check(payload.get("tokenizer_ids") == IDS, "exact frozen tokenizer IDs")
    attention_hash = digest([1] * 1024)
    sequence_hashes: dict[str, str] = {}

    def record(row: Mapping[str, Any], request: ExpectedRequest) -> None:
        key = request.request_id
        seq = request.sequence
        if seq not in sequence_hashes:
            sequence_hashes[seq] = sequence_digest(seq)
        input_ids = [IDS["CLS"], *(IDS[base] for base in seq), IDS["SEP"]]
        input_ids[request.position + 1] = IDS["MASK"]
        for field, value in {
            "request_id": key, "sequence_sha256": sequence_hashes[seq], "position": request.position,
            "token_position": request.position + 1, "true_base": request.true_base,
            "input_ids_sha256": digest(input_ids), "attention_mask_sha256": attention_hash,
            "token_count": 1024,
        }.items():
            checks.check(row.get(field) == value, f"{key}:{field}")
        checks.check(row.get("all_forward_logits_finite") is True, f"{key}:all forward logits finite")
        values = row.get("masked_logits")
        valid = isinstance(values, list) and len(values) == 10 and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values)
        checks.check(valid, f"{key}:finite full ten-token logits")
        if valid:
            logits = np.asarray(values, dtype=np.float32)
            base = logits[[IDS[b] for b in "ACGT"]]
            base_values = row.get("base_logits")
            checks.check(isinstance(base_values, list) and len(base_values) == 4
                         and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                 and math.isfinite(x) for x in base_values)
                         and base_values == [values[IDS[b]] for b in "ACGT"],
                         f"{key}:four-base logits exactly match full vocabulary")
            target = logits[IDS[request.true_base]]
            linear = np.float32(target - np.float32(np.float32(base.sum(dtype=np.float32) - target) / np.float32(3)))
            others = [logits[IDS[b]] for b in "ACGT" if b != request.true_base]
            maximum = np.float32(target - max(others))
            logits64 = np.asarray(values, dtype=np.float64)
            shift = float(logits64.max())
            logp = float(logits64[IDS[request.true_base]] - shift - np.log(np.exp(logits64 - shift).sum()))
            checks.numeric(row.get("linear_margin"), float(linear), RAW_TOLERANCE, f"{key}:linear from logits", "linear_from_logits")
            checks.numeric(row.get("max_margin"), float(maximum), RAW_TOLERANCE, f"{key}:max from logits", "max_from_logits")
            checks.numeric(row.get("true_base_log_probability"), logp, LOG_PROBABILITY_TOLERANCE,
                           f"{key}:full-vocabulary log probability", "log_probability_from_logits")
            checks.check(type(row.get("correct")) is bool and row["correct"] == (int(np.argmax(logits64)) == IDS[request.true_base]),
                         f"{key}:full-vocabulary accuracy")
        if request.linear_margin is not None:
            checks.numeric(row.get("linear_margin"), request.linear_margin, RAW_TOLERANCE,
                           f"{key}:historical margin", "historical_linear_margin")
        if request.log_probability is not None:
            checks.numeric(row.get("true_base_log_probability"), request.log_probability, LOG_PROBABILITY_TOLERANCE,
                           f"{key}:historical protected log probability", "historical_log_probability")
            checks.check(type(row.get("correct")) is bool and row["correct"] is request.correct,
                         f"{key}:historical protected accuracy")

    counts = {}
    for stage in ("cell", "native"):
        rows = payload.get(f"{stage}_requests", [])
        wanted = list(expected[stage])
        valid_rows = isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows)
        checks.check(valid_rows, f"{stage}:request record schema")
        if not valid_rows:
            continue
        expected_ids = [r.request_id for r in wanted]
        actual_ids = [r.get("request_id") for r in rows]
        checks.check(actual_ids == expected_ids, f"{stage}:complete exact request order/coverage; no CC or subset")
        counts[stage] = {"expected": len(wanted), "observed": len(rows)}
        by_id = {r.get("request_id"): r for r in rows if isinstance(r.get("request_id"), str)}
        checks.check(len(by_id) == len(rows), f"{stage}:unique request IDs")
        for request in wanted:
            if request.request_id in by_id:
                record(by_id[request.request_id], request)
        repeats = payload.get(f"{stage}_repeats", [])
        repeat_ids = expected_ids[:16]
        repeats_valid = isinstance(repeats, list) and all(isinstance(r, Mapping) for r in repeats)
        checks.check(repeats_valid and [r.get("request_id") for r in repeats] == repeat_ids,
                     f"{stage}:exact first sixteen repeated requests")
        if repeats_valid:
            expected_by_id = {r.request_id: r for r in wanted[:16]}
            for row in repeats:
                key = row.get("request_id")
                if key not in expected_by_id or key not in by_id:
                    continue
                record(row, expected_by_id[key])
                baseline = by_id[key]
                left, right = row.get("masked_logits"), baseline.get("masked_logits")
                checks.check(isinstance(left, list) and isinstance(right, list) and len(left) == len(right) == 10,
                             f"{key}:repeat logit shape")
                if isinstance(left, list) and isinstance(right, list):
                    for j, (a, b) in enumerate(zip(left, right)):
                        if isinstance(b, (int, float)):
                            checks.numeric(a, b, REPEAT_TOLERANCE, f"{key}:repeat logit {j}", "repeat_logits")
                for field in ("linear_margin", "max_margin", "true_base_log_probability"):
                    b = baseline.get(field)
                    if isinstance(b, (int, float)):
                        checks.numeric(row.get(field), b, REPEAT_TOLERANCE, f"{key}:repeat {field}", "repeat_" + field)
                checks.check(row.get("correct") is baseline.get("correct"), f"{key}:repeat exact accuracy")
    return {**checks.result(), "request_counts": counts}


def _endpoints(matrix: Sequence[Sequence[float]]) -> dict[str, float]:
    a = np.asarray(matrix, dtype=float)
    m0 = float(np.mean([a[p, p] for p in range(3)]))
    plus = float(np.mean([a[p, (p + 1) % 3] for p in range(3)]))
    minus = float(np.mean([a[p, (p - 1) % 3] for p in range(3)]))
    out = {"m0": m0, "m_plus": plus, "m_minus": minus, "psi_match": m0 - .5*(plus + minus),
           "m0_minus_plus": m0 - plus, "m0_minus_minus": m0 - minus, "pooled": float(a.mean())}
    for p in range(3):
        out[f"donor_phase_{p}_excess"] = float(a[p,p] - .5*(a[p,(p+1)%3] + a[p,(p-1)%3]))
        out[f"acceptor_phase_{p}_excess"] = float(a[p,p] - .5*(a[(p+1)%3,p] + a[(p-1)%3,p]))
    return out


def historical_inference(cells: Mapping[str, Any], cohort: Mapping[str, Any]) -> dict[str, Any]:
    """Independent transcription of C43 section 5 and every target diagnostic.

    Operations/ordering follow the frozen definition so bit identity is testable.
    Input matrices are rebuilt from raw margins, never the stored endpoint table.
    """
    grouped: dict[str, dict[tuple[int, int], Mapping[str, Any]]] = defaultdict(dict)
    for row in cells["cell_rows"]:
        key = (row["p"], row["q"])
        require(key not in grouped[row["block_id"]], "duplicate inference cell")
        grouped[row["block_id"]][key] = row
    block_ids = sorted(grouped)
    require(len(block_ids) == BLOCKS and all(len(x) == 9 for x in grouped.values()), "inference requires all384")
    matrices, ac, at, strata = [], [], [], []
    for bid in block_ids:
        rows = grouped[bid]
        strata.append(f"{rows[(0,0)]['chrom']}:{rows[(0,0)]['strand']}")
        def matrix(arm: str) -> list[list[float]]:
            return [[float(rows[(p,q)]["donor_margins"]["intact"])
                     - float(rows[(p,q)]["donor_margins"][f"true_acceptor_{arm}"])
                     for q in range(3)] for p in range(3)]
        matrices.append(matrix("ac"))
        ac.append(_endpoints(matrices[-1]))
        at.append(_endpoints(matrix("at")))
    labels = np.asarray(strata, dtype=object)
    groups = [np.flatnonzero(labels == label) for label in sorted(set(strata))]
    require(all(len(g) >= 2 for g in groups), "singleton historical bootstrap stratum")
    rng = np.random.default_rng(20260803)
    indices = np.concatenate([rng.choice(g, size=(20_000,len(g)), replace=True) for g in groups], axis=1)

    def summary(values: Sequence[float]) -> dict[str, Any]:
        a = np.asarray(values, dtype=float)
        draws = a[indices].mean(axis=1)
        sd = float(draws.std(ddof=1))
        return {"mean": float(a.mean()), "ci95": np.quantile(draws, (.025,.975)).tolist(),
                "bootstrap_sd": sd, "mde80_unadjusted": float(MDE_MULTIPLIER * sd)}

    def simultaneous(names: Sequence[str], values: Any) -> dict[str, Any]:
        a = np.asarray(values, dtype=float)
        means = a.mean(axis=0)
        draws = a[indices].mean(axis=1)
        se = draws.std(axis=0, ddof=1)
        active = se > 0
        critical = float(np.quantile(np.abs((draws[:,active]-means[active])/se[active]).max(axis=1), .95)) if active.any() else 0.0
        lo, hi = means - critical*se, means + critical*se
        return {"method": "max-absolute-studentized whole-block bootstrap", "critical_value": critical,
                "endpoints": {name: {"mean": float(means[i]), "simultaneous_ci95": [float(lo[i]),float(hi[i])],
                                     "simultaneous_lower95": float(lo[i])} for i,name in enumerate(names)}}

    psi = np.asarray([r["psi_match"] for r in ac], dtype=float)
    primary = summary(psi)
    centered = psi.copy()
    for label in sorted(set(strata)):
        selected = labels == label
        centered[selected] -= centered[selected].mean()
    primary["mde80"] = float(MDE_MULTIPLIER * centered[indices].mean(axis=1).std(ddof=1))
    choices = tuple(permutations(range(3)))
    controls = np.empty((99, BLOCKS), dtype=float)
    recorded_family = cohort["pseudo_phase_family"]["roles"]["replication"]["assignments_by_seed"]
    for s, seed in enumerate(range(2026080401, 2026080500)):
        order = sorted(range(BLOCKS), key=lambda i: hashlib.sha256(
            f"crossed-phase-null-v1\0{seed}\0{block_ids[i]}".encode()).hexdigest())
        assignments = [choices[0]] * BLOCKS
        for rank, i in enumerate(order):
            assignments[i] = choices[rank // 64]
        recorded = {r["block_id"]: tuple(r["acceptor_label_permutation"])
                    for r in recorded_family[str(seed)]["assignments"]}
        require(recorded == dict(zip(block_ids, assignments)), "historical pseudo assignment differs")
        for i, (matrix, perm) in enumerate(zip(matrices, assignments)):
            a = np.asarray(matrix, dtype=float)
            diagonal = [a[p,q] for p in range(3) for q in range(3) if perm[q] == p]
            off = [a[p,q] for p in range(3) for q in range(3) if perm[q] != p]
            controls[s,i] = float(np.mean(diagonal) - np.mean(off))
    means = controls.mean(axis=1)
    paired = psi - controls.mean(axis=0)
    leave_one = [float((psi.sum()-psi[i])/(BLOCKS-1)) for i in range(BLOCKS)]
    chroms = np.asarray([s.split(":",1)[0] for s in strata])
    strands = np.asarray([s.rsplit(":",1)[-1] for s in strata])
    stop_by_id = {b["block_id"]: b["stop_diagnostics"] for b in cohort["blocks"]}
    require(set(stop_by_id) == set(block_ids), "historical stop block coverage differs")
    stop_matrices = np.asarray([stop_by_id[bid]["stop_count_matrix"] for bid in block_ids], dtype=float)
    require(stop_matrices.shape == (BLOCKS,3,3) and np.isfinite(stop_matrices).all()
            and np.equal(stop_matrices,np.floor(stop_matrices)).all()
            and np.all((stop_matrices >= 0) & (stop_matrices <= 12)), "historical stop matrices differ")
    omega = np.asarray([float(stop_by_id[bid]["omega_mismatch_minus_diagonal"]) for bid in block_ids])
    reconstructed_omega = stop_matrices[:,~np.eye(3,dtype=bool)].mean(axis=1) - np.diagonal(
        stop_matrices,axis1=1,axis2=2).mean(axis=1)
    require(np.isfinite(omega).all() and np.all(np.abs(omega-reconstructed_omega) <= 1e-12),
            "historical stop-load contrasts do not reconstruct")
    cutpoint = float(cohort["stop_diagnostics"]["replication_median_omega_cutpoint"])
    require(cutpoint == float(np.median(omega)), "historical stop cutpoint differs")
    low = omega <= cutpoint
    require(low.any() and (~low).any(), "historical stop split is empty")

    def average_ranks(values: np.ndarray) -> np.ndarray:
        unique, inverse, counts = np.unique(values,return_inverse=True,return_counts=True)
        del unique
        ends = np.cumsum(counts)
        starts = ends-counts
        return (0.5*(starts+ends-1)+1.0)[inverse]

    ranked_omega, ranked_psi = average_ranks(omega), average_ranks(psi)
    rho = (None if ranked_omega.std() == 0 or ranked_psi.std() == 0
           else float(np.corrcoef(ranked_omega,ranked_psi)[0,1]))
    return {
        "n_blocks": BLOCKS, "primary_psi": primary,
        "directions": simultaneous(DIRECTIONS, [[r[n] for n in DIRECTIONS] for r in ac]),
        "mismatch_equivalence": simultaneous(("m_plus","m_minus"), [[r["m_plus"],r["m_minus"]] for r in ac]),
        "operator_sensitivity": simultaneous(("psi_ac_minus_at",), [[a["psi_match"]-b["psi_match"]] for a,b in zip(ac,at)]),
        "matching_means": {n: float(np.mean([r[n] for r in ac])) for n in ("m0","m_plus","m_minus","pooled")},
        "pseudo_phase": {"seeds": list(range(2026080401,2026080500)), "control_means": means.tolist(),
            "maximum_control": float(means.max()), "target_exceeds_all": bool(psi.mean() > means.max()),
            "exact_add_one_one_sided_p": float((1+int(np.sum(means >= psi.mean())))/100),
            "target_minus_mean_control": {"mean": float(paired.mean()), "lower95": float(np.quantile(paired[indices].mean(axis=1),.05))}},
        "sensitivity": {"leave_one_block": {"minimum": min(leave_one), "maximum": max(leave_one)},
            "leave_one_chromosome": {c: float(psi[chroms != c].mean()) for c in sorted(set(chroms))},
            "leave_one_strand": {s: float(psi[strands != s].mean()) for s in sorted(set(strands))}},
        "stop_count_mean_3x3": stop_matrices.mean(axis=0).tolist(),
        "stop_sensitivity": {"cutpoint":cutpoint,"spearman_rho_omega_psi":rho,
            "low_or_equal":{"n":int(low.sum()),"mean_psi":float(psi[low].mean())},
            "high":{"n":int((~low).sum()),"mean_psi":float(psi[~low].mean())}},
        "block_endpoints": [{"block_id": bid, "stratum": stratum, "ac": a, "at": b,
                             "psi_ac_minus_at": float(a["psi_match"]-b["psi_match"])}
                            for bid,stratum,a,b in zip(block_ids,strata,ac,at)],
    }


def compare_inference(actual: Any, expected: Any) -> dict[str, Any]:
    checks = Checks()
    identical = True
    def compare(a: Any, b: Any, path: str) -> None:
        nonlocal identical
        if isinstance(b, dict):
            same_shape = isinstance(a, dict) and set(a) == set(b)
            identical = identical and same_shape
            checks.check(same_shape, path + ":mapping fields")
            if isinstance(a, dict):
                for key in b:
                    if key in a:
                        compare(a[key], b[key], path + "." + key)
        elif isinstance(b, list):
            same_shape = isinstance(a, list) and len(a) == len(b)
            identical = identical and same_shape
            checks.check(same_shape, path + ":list shape")
            if isinstance(a, list):
                for i,(left,right) in enumerate(zip(a,b)):
                    compare(left,right,f"{path}[{i}]")
        elif type(b) is bool or isinstance(b, (str, int)):
            identical = identical and type(a) is type(b) and a == b
            checks.check(type(a) is type(b) and a == b, path + ":exact discrete value")
        elif isinstance(b, float):
            identical = identical and type(a) is float and struct.pack("!d",a) == struct.pack("!d",b)
            checks.numeric(a,b,max(INFERENCE_TOLERANCE, INFERENCE_TOLERANCE*abs(b)),path,"inference")
        else:
            identical = identical and type(a) is type(b) and a == b
            checks.check(a == b, path)
    compare(actual,expected,"C43")
    return {**checks.result(), "bitwise_identical": identical}


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    data = json.loads(raw)
    require(isinstance(data, dict), f"JSON object required: {path}")
    return data, hashlib.sha256(raw).hexdigest()


def write_exclusive(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("fidelity", "cohort", "fasta", "cells", "native", "historical-audit", "freeze", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    for path in (args.out,args.out.with_suffix(".manifest.json")):
        if path.is_symlink():
            raise FileExistsError(path)
    out = args.out.resolve()
    manifest_path = out.with_suffix(".manifest.json")
    for path in (out,manifest_path):
        guard_output_path(path)
        if path.is_symlink():
            raise FileExistsError(path)
    paths = {name: getattr(args,name.replace("-","_")).resolve() for name in (
        "fidelity", "cohort", "fasta", "cells", "native", "historical-audit", "freeze")}
    report: dict[str, Any] = {"schema": AUDIT_SCHEMA, "status": "FAIL", "passed": False,
        "authorized_for_cc": False, "model_loaded": False, "expected_blocks": BLOCKS,
        "auditor_sha256": sha256_file(Path(__file__)), "tolerances": {
            "raw_margin_abs": RAW_TOLERANCE, "protected_log_probability_abs": LOG_PROBABILITY_TOLERANCE,
            "repeat_abs": REPEAT_TOLERANCE, "historical_inference_abs_rel": INFERENCE_TOLERANCE}}
    try:
        report["fidelity_sha256"] = sha256_file(paths["fidelity"])
        report["freeze_sha256"] = sha256_file(paths["freeze"])
        payload, raw_fidelity_hash = read_json(paths["fidelity"])
        report["fidelity_uncompressed_sha256"] = raw_fidelity_hash
        sources = {}
        for name in ("cohort","cells","native","historical-audit"):
            data, observed = read_json(paths[name])
            pin = PINS["audit" if name == "historical-audit" else name]
            require(observed == pin, f"historical {name} bytes do not match immutable pin")
            sources[name] = data
        require(sha256_file(paths["fasta"]) == PINS["fasta"], "FASTA differs from historical pin")
        require(sources["cells"]["cohort_sha256"] == sources["native"]["cohort_sha256"] == PINS["cohort"],
                "historical source cohort binding differs")
        fasta = IndexedFasta(paths["fasta"])
        try:
            wanted = expected_requests(sources["cohort"],sources["cells"],sources["native"],fasta.fetch)
        finally:
            fasta.close()
        report["raw_measurement"] = audit_records(payload,wanted,cohort_sha256=PINS["cohort"],
                                                  freeze_sha256=report["freeze_sha256"])
        target = historical_inference(sources["cells"],sources["cohort"])
        reference = sources["historical-audit"]["recomputed"]["target"]
        report["historical_inference"] = compare_inference(target,reference)
        report["historical_target_recomputed"] = target
        report["reference_pins"] = PINS
        report["passed"] = report["raw_measurement"]["passed"] and report["historical_inference"]["passed"]
        report["authorized_for_cc"] = report["passed"]
        report["status"] = "PASS" if report["passed"] else "FAIL"
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        report["fatal_audit_error"] = f"{type(exc).__name__}: {exc}"
    write_exclusive(out,report)
    manifest = ExperimentManifest.create("splicebert-register-operator-replication:fidelity-audit",
        sys.argv if argv is None else [str(Path(__file__)),*argv],repo=REPO,status=report["status"],
        config={"expected_blocks": BLOCKS,"tolerances":report["tolerances"],"model_loaded":False},
        seeds={"historical_bootstrap":20260803,"pseudo_first":2026080401,"pseudo_last":2026080499})
    for path in (*paths.values(),Path(str(paths["fasta"])+".fai"),Path(__file__).resolve(),REPO/"src/gfmcd/provenance.py"):
        manifest.add_input(path)
    manifest.add_output(out)
    write_exclusive(manifest_path,manifest.to_dict())
    print(json.dumps({"status":report["status"],"authorized_for_cc":report["authorized_for_cc"],
                      "out":str(out),"fatal_audit_error":report.get("fatal_audit_error")},sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
