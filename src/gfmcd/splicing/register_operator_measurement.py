"""One scoring/reduction path for real requests and raw-logit calibration fixtures."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Callable

import numpy as np
import torch

from gfmcd.splicing.register_operator_construction import PROBES, Request, sha

SCHEMA = "gfmcd.register_operator.score_payload.v1"
CELL_STATES = ("intact", "cc", "ac", "at", "bridge_ac", "acceptor_intact")
NATIVE_STATES = ("intact", "cc", "ac", "acceptor_intact")
BASES = "ACGT"


def nucleotide_margins(logits, target: int = 2) -> dict[str, np.ndarray]:
    """CPU float32 arithmetic preserves the historical four-base sum order."""
    values = torch.as_tensor(np.asarray(logits, dtype=np.float32))
    if values.shape[-1] != 4 or not bool(torch.isfinite(values).all()):
        raise ValueError("nucleotide logits must be finite A,C,G,T vectors")
    actual = values[..., target]
    linear = actual - (values.sum(dim=-1) - actual) / 3.0
    maximum = actual - values[..., [i for i in range(4) if i != target]].max(dim=-1).values
    return {"linear": linear.numpy().astype(float), "max": maximum.numpy().astype(float)}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def serialize_score_payload(payload: dict) -> bytes:
    validate_score_payload(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=_json_default).encode()


def deserialize_score_payload(raw: bytes) -> dict:
    payload = json.loads(raw)
    validate_score_payload(payload)
    return payload


def validate_score_payload(payload: dict) -> None:
    if payload.get("schema") != SCHEMA or type(payload.get("synthetic")) is not bool:
        raise ValueError("unknown raw score schema/origin")
    ids = payload["block_ids"]
    n = len(ids)
    if not n or len(set(ids)) != n or len(payload["strata"]) != n:
        raise ValueError("invalid block identity/strata")
    phase = np.asarray(payload["cell_phases"])
    expected = np.tile(np.array([(p, q) for p in range(3) for q in range(3)]), (n, 1, 1))
    if phase.shape != (n, 9, 2) or not np.array_equal(phase, expected):
        raise ValueError("missing, duplicate, or misassigned cell phase")
    for name, shape in (("cell_logits", (n, 9, 6, 4)), ("native_logits", (n, 6, 4, 4))):
        x = np.asarray(payload[name], dtype=float)
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError(f"invalid {name}")
    health = np.asarray(payload["health_logits"], dtype=float)
    hc = 1 if payload.get("health_repeat_cells") == 9 else 9
    if hc == 1 and not payload["synthetic"]:
        raise ValueError("real model health probes cannot be broadcast")
    if health.ndim != 5 or health.shape[:4] != (n, hc, 24, 2) or not np.isfinite(health).all():
        raise ValueError("invalid full-vocabulary protected logits")
    base_ids = payload["health_base_ids"]
    if isinstance(base_ids, dict):
        base_ids = [base_ids[b] for b in BASES]
    if len(base_ids) != 4 or len(set(base_ids)) != 4 or min(base_ids) < 0 or max(base_ids) >= health.shape[-1]:
        raise ValueError("invalid full-vocabulary nucleotide map")
    truth = np.asarray(payload["health_true_base_ids"])
    if truth.shape != (n, hc, 24) or not np.isin(truth, range(4)).all():
        raise ValueError("invalid protected true nucleotide IDs")
    if not isinstance(payload["prerequisites"], dict) or not isinstance(payload["numerics"], dict):
        raise ValueError("missing execution prerequisites")


def reduce_score_payload(payload: dict) -> dict:
    validate_score_payload(payload)
    cell = nucleotide_margins(payload["cell_logits"])
    native = nucleotide_margins(payload["native_logits"])
    n = len(payload["block_ids"])
    result = {"block_ids": list(payload["block_ids"]), "strata": list(payload["strata"]),
              "synthetic": payload["synthetic"], "matrices": {}, "native": {},
              "phase_blind": {}, "prerequisites": payload["prerequisites"],
              "numerics": payload["numerics"]}
    for margin in ("max", "linear"):
        for j, operator in enumerate(("cc", "ac", "at"), 1):
            result["matrices"][f"{operator}_{margin}"] = (cell[margin][:, :, 0] - cell[margin][:, :, j]).reshape(n, 3, 3)
        result["phase_blind"][f"bridge_{margin}"] = (cell[margin][:, :, 0] - cell[margin][:, :, 4]).mean(axis=1)
        result["phase_blind"][f"donor_{margin}"] = cell[margin][:, :, 0].mean(axis=1)
        result["phase_blind"][f"acceptor_{margin}"] = cell[margin][:, :, 5].mean(axis=1)
        result["native"][f"cc_{margin}"] = (native[margin][:, :, 0] - native[margin][:, :, 1]).mean(axis=1)
        result["native"][f"ac_{margin}"] = (native[margin][:, :, 0] - native[margin][:, :, 2]).mean(axis=1)
        result["native"][f"donor_{margin}"] = native[margin][:, :, 0].mean(axis=1)
        result["native"][f"acceptor_{margin}"] = native[margin][:, :, 3].mean(axis=1)
    h = torch.as_tensor(np.asarray(payload["health_logits"], dtype=np.float32))
    base_ids = payload["health_base_ids"]
    if isinstance(base_ids, dict):
        base_ids = [base_ids[b] for b in BASES]
    target = np.array(base_ids, dtype=int)[np.asarray(payload["health_true_base_ids"], dtype=int)]
    targets = torch.from_numpy(np.repeat(target[..., None, None], 2, axis=-2))
    lp = torch.log_softmax(h, dim=-1).gather(-1, targets).squeeze(-1).numpy().astype(float)
    correct = h.argmax(dim=-1).numpy() == target[..., None]
    repeat = 9 if payload.get("health_repeat_cells") == 9 else 1
    result["health"] = {
        "logp_delta": (lp[..., 0] - lp[..., 1]).mean(axis=(1, 2)),
        "construct_correct": correct[..., 0].sum(axis=(1, 2)) * repeat,
        "source_correct": correct[..., 1].sum(axis=(1, 2)) * repeat,
        "probe_pairs": np.full(n, 216, dtype=int),
    }
    # Inspectable max competitors, including genuine ties, for each state.
    raw = np.asarray(payload["cell_logits"], dtype=np.float32)
    rivals = raw[..., [0, 1, 3]]
    is_max = rivals == rivals.max(axis=-1, keepdims=True)
    result["competitors"] = {
        state: {"maximizer_counts_AC_T": is_max[:, :, i].sum(axis=(0, 1)).tolist(),
                "tied_requests": int((is_max[:, :, i].sum(axis=-1) > 1).sum())}
        for i, state in enumerate(CELL_STATES)
    }
    return result


def token_digest(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def score_requests(adapter, requests: list[Request], *, batch_size: int = 16,
                   progress: Callable | None = None, output_records: list | None = None) -> list[dict]:
    """Mask actual tokenized inputs, retain complete masked-vocabulary logits."""
    if batch_size != 16 or len({r.request_id for r in requests}) != len(requests):
        raise ValueError("frozen batch size or unique request contract violated")
    base_ids = [int(adapter.tokenizer.convert_tokens_to_ids(b)) for b in BASES]
    mask_id = int(adapter.tokenizer.mask_token_id)
    output = [] if output_records is None else output_records
    if output:
        raise ValueError("score output accumulator must start empty")
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(requests), batch_size):
            rows = requests[start:start + batch_size]
            encoded = adapter.tokenize([r.sequence for r in rows])
            if encoded.input_ids.shape != (len(rows), 1024):
                raise ValueError("sequence does not tokenize to exactly1024 tokens")
            positions = [int(adapter.nucleotide_to_token(r.position)) for r in rows]
            before = encoded.input_ids.detach().cpu().tolist()
            for j, (r, pos) in enumerate(zip(rows, positions)):
                if pos != r.position + 1 or before[j][pos] != base_ids[BASES.index(r.true_base)]:
                    raise ValueError("masked nucleotide/token mapping mismatch")
                if before[j].count(mask_id) != 0:
                    raise ValueError("input already contains a mask")
                encoded.input_ids[j, pos] = mask_id
            actual_ids = encoded.input_ids.detach().cpu().tolist()
            actual_attention = encoded.attention_mask.detach().cpu().tolist()
            if any(sum(a) != 1024 or a != [1] * 1024 for a in actual_attention):
                raise ValueError("attention mask differs from frozen full-length input")
            raw = adapter.model(input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask, return_dict=True).logits.float()
            finite = bool(torch.isfinite(raw).all().item())
            if not finite:
                raise ValueError("nonfinite model forward")
            selected = raw[torch.arange(len(rows), device=raw.device),
                           torch.tensor(positions, device=raw.device)].detach().cpu()
            base = selected[:, base_ids]
            target_ids = torch.tensor([base_ids[BASES.index(r.true_base)] for r in rows])
            true_logits = selected.gather(1, target_ids[:, None]).squeeze(1)
            linear = true_logits - (base.sum(dim=1) - true_logits) / 3.0
            competitors = base.clone()
            competitors[torch.arange(len(rows)), [BASES.index(r.true_base) for r in rows]] = -torch.inf
            maxmargin = true_logits - competitors.max(dim=1).values
            logp = torch.log_softmax(selected, dim=-1).gather(1, target_ids[:, None]).squeeze(1)
            predicted = selected.argmax(dim=-1)
            for j, r in enumerate(rows):
                output.append({"request_id": r.request_id, "sequence_sha256": sha(r.sequence),
                    "position": r.position, "token_position": positions[j], "true_base": r.true_base,
                    "input_ids_sha256": token_digest(actual_ids[j]),
                    "attention_mask_sha256": token_digest(actual_attention[j]), "token_count": 1024,
                    "masked_logits": selected[j].tolist(), "base_logits": base[j].tolist(),
                    "linear_margin": float(linear[j]), "max_margin": float(maxmargin[j]),
                    "true_base_log_probability": float(logp[j]), "correct": bool(predicted[j] == target_ids[j]),
                    "all_forward_logits_finite": finite})
            if progress is not None and (start == 0 or (start // batch_size + 1) % 128 == 0 or start + batch_size >= len(requests)):
                progress({"scored": len(output), "total": len(requests), "seconds": time.monotonic() - started})
    return output


def repeat_difference(first: list[dict], repeated: list[dict]) -> dict:
    if [r["request_id"] for r in first] != [r["request_id"] for r in repeated]:
        raise ValueError("repeat request identity differs")
    diffs = {field: max(abs(float(a[field]) - float(b[field])) for a, b in zip(first, repeated))
             for field in ("linear_margin", "max_margin", "true_base_log_probability")}
    diffs["masked_logits"] = max(float(np.max(np.abs(np.array(a["masked_logits"]) - b["masked_logits"])))
                                   for a, b in zip(first, repeated))
    return {"maximum_absolute_differences": diffs,
            "accuracy_identical": all(a["correct"] == b["correct"] for a, b in zip(first, repeated)),
            "maximum": max(diffs.values()), "threshold": 1e-8,
            "passed": max(diffs.values()) <= 1e-8 and all(a["correct"] == b["correct"] for a, b in zip(first, repeated))}


def requests_to_score_payload(raw: dict, blocks: list[dict], *, prerequisites: dict) -> dict:
    """Rebuild the analysis tensor from audited request identities, never summaries."""
    records = raw["cell_requests"] + raw["native_requests"]
    by_id = {r["request_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate raw request identity")
    base_ids = [raw["tokenizer_ids"][b] for b in BASES]
    cells, native, health, truth = [], [], [], []
    consumed = set()

    def get(key: str, *, full: bool = False):
        consumed.add(key)
        row = by_id[key]
        logits = row["masked_logits"]
        return logits if full else [logits[i] for i in base_ids]

    for block in blocks:
        cr, hr, tr, nr = [], [], [], []
        for p in range(3):
            for q in range(3):
                key = f"{block['block_id']}:p{p}:q{q}"
                states = ("donor_intact", "donor_true_acceptor_cc", "donor_true_acceptor_ac",
                          "donor_true_acceptor_at", "donor_bridge_ac", "acceptor_intact")
                cr.append([get(key + ":" + s) for s in states])
                probes, bases = [], []
                for role, phase in (("donor", p), ("acceptor", q)):
                    source = block[role + "s"][str(phase)]["source_id"]
                    for pos in PROBES[role]:
                        ck = f"{key}:health:{role}:{pos}:construct"
                        sk = f"source:{role}:{source}:health:{pos}"
                        if by_id[ck]["true_base"] != by_id[sk]["true_base"]:
                            raise ValueError("protected true bases disagree")
                        probes.append([get(ck, full=True), get(sk, full=True)])
                        bases.append(BASES.index(by_id[ck]["true_base"]))
                hr.append(probes)
                tr.append(bases)
        for role in ("donor", "acceptor"):
            for p in range(3):
                source = block[role + "s"][str(p)]["source_id"]
                key = f"native:{block['block_id']}:{role}:p{p}:{source}"
                nr.append([get(key + ":" + s) for s in ("donor_intact", "donor_true_acceptor_cc",
                          "donor_true_acceptor_ac", "acceptor_intact")])
        cells.append(cr); native.append(nr); health.append(hr); truth.append(tr)
    if consumed != set(by_id):
        raise ValueError("unused or unaccounted raw requests")
    result = {"schema": SCHEMA, "synthetic": False, "block_ids": [b["block_id"] for b in blocks],
              "strata": [f"{b['chrom']}:{b['strand']}" for b in blocks],
              "cell_phases": [[(p, q) for p in range(3) for q in range(3)] for b in blocks],
              "cell_logits": cells, "native_logits": native, "health_logits": health,
              "health_true_base_ids": truth, "health_base_ids": base_ids,
              "prerequisites": prerequisites,
              "numerics": {"all_finite": all(r["all_forward_logits_finite"] for r in records),
                  "token_count": 1024, "mask_mapping_valid": True,
                  "repeat_max_abs": max(raw["determinism"][s]["maximum"] for s in ("native", "cells"))}}
    validate_score_payload(result)
    return result
