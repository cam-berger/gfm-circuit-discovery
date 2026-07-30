#!/usr/bin/env python
"""D16 — did fine-tuning move the [CLS] attention, or only the readout?

Measures [CLS]-row attention band mass for all 144 heads on BOTH checkpoints over the same cohort,
subtracts the common ALiBi-only positional prior, and compares.

Both checkpoints share the vendored MosaicBERT encoder and (verified at load, and again at runtime
here) an identical BPE, so the offsets and therefore the prior are common to both and cancel in any
difference.

Protocol: studies/dnabert2_reconstruction_atlas/PREREGISTRATION_D16_FINETUNING_ATTENTION_SHIFT.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    load_dnabert2_mlm,
    load_dnabert2_promoter_classifier,
)
from gfmcd.adapters.dnabert2.promoter_head_atlas import (
    PROMOTER_SOURCE_BANDS,
    _token_band_allocations,
    tokenize_promoter_pairs,
)
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "data/processed/dnabert2_atlas/promoter_source_localization_v1/cohort.jsonl"
ATLAS_DEV = REPO / "results/dnabert2_reconstruction_atlas/promoter_head_atlas_development_v1.json"

BAND_ORDER = ["special", *PROMOTER_SOURCE_BANDS]
TSS_INDEX = 249
CARRIERS = ("L11H2", "L11H5")
READER = "L9H7"
D5_HEADS = ("L2H2", "L4H6", "L4H7", "L9H7", "L10H1", "L10H5", "L11H2", "L11H5")
PRIOR_THRESHOLD = 0.10  # frozen in prereg §4


def alibi_slopes(n_heads: int) -> list[float]:
    def power_of_two(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return power_of_two(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return power_of_two(closest) + alibi_slopes(2 * closest)[0::2][: n_heads - closest]


def capture_bands(model, config, batch, device) -> dict[tuple[int, int], dict[str, list[float]]]:
    """[CLS]-row attention band mass per head. Works for the MLM and the classifier alike:
    both wrap the same vendored encoder, and only a forward pass is needed to fire the hooks."""
    import torch

    qkv_by_layer: dict[int, Any] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            qkv_by_layer[layer] = output.detach()
            return output

        return hook

    try:
        for layer, block in enumerate(model.bert.encoder.layer):
            handles.append(block.attention.self.Wqkv.register_forward_hook(make_hook(layer)))
        with torch.no_grad():
            model(**{name: tensor.to(device) for name, tensor in batch.encoded.items()})
    finally:
        for handle in handles:
            handle.remove()

    n_layers = int(config.num_hidden_layers)
    if set(qkv_by_layer) != set(range(n_layers)):
        raise RuntimeError("failed to capture every QKV projection")
    n_heads = int(config.num_attention_heads)
    head_size = int(config.hidden_size) // n_heads
    output = {
        (layer, head): {name: [] for name in BAND_ORDER}
        for layer in range(n_layers)
        for head in range(n_heads)
    }
    offsets = batch.offset_mapping.cpu()
    spans, cursor = [], 0
    for count in batch.token_counts:
        spans.append((cursor, cursor + count))
        cursor += count
    alibi = model.bert.encoder.alibi
    for layer in range(n_layers):
        qkv_all = qkv_by_layer[layer]
        for seq_index, ((start, end), count) in enumerate(zip(spans, batch.token_counts)):
            qkv = qkv_all[start:end].view(count, 3, n_heads, head_size)
            scores = torch.einsum("hd,shd->hs", qkv[0, 0], qkv[:, 1]) / math.sqrt(head_size)
            scores = scores + alibi[0, :, 0, :count].to(device=scores.device, dtype=scores.dtype)
            probabilities = torch.softmax(scores, dim=1)
            allocations = _token_band_allocations(
                offsets[seq_index, :count].tolist(),
                tss_index=TSS_INDEX,
                bands=PROMOTER_SOURCE_BANDS,
            ).to(probabilities.device)
            masses = probabilities @ allocations
            # M4: architecture-forced normalisation check, flagged not counted
            if not torch.allclose(
                masses.sum(dim=1), torch.ones(n_heads, device=masses.device), atol=1e-5, rtol=0
            ):
                raise RuntimeError("CLS attention band mass does not sum to one")
            for head in range(n_heads):
                for band_index, name in enumerate(BAND_ORDER):
                    output[(layer, head)][name].append(float(masses[head, band_index].cpu()))
    return output


def run_model(model, config, device, pairs, batch_pairs: int, label: str):
    """Mean band mass per head over the cohort, separately for target and control sequences."""
    totals: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: {f"{role}_{name}": [] for role in ("target", "control") for name in BAND_ORDER}
    )
    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start : start + batch_pairs]
        batch = tokenize_promoter_pairs_for(model, chunk)
        captured = capture_bands(model, config, batch, device)
        for head, bands in captured.items():
            for name in BAND_ORDER:
                values = bands[name]
                totals[head][f"target_{name}"].extend(values[0::2])
                totals[head][f"control_{name}"].extend(values[1::2])
        print(f"[D16:{label}] {min(start + batch_pairs, len(pairs))}/{len(pairs)}", flush=True)
    return {
        f"L{layer}H{head}": {k: statistics.mean(v) for k, v in bands.items() if v}
        for (layer, head), bands in totals.items()
    }


_TOKENIZER_BUNDLE: Any = None


def tokenize_promoter_pairs_for(_model, chunk):
    """Both checkpoints tokenize identically (M1 proves it on this cohort), so one tokenization
    is shared -- which also guarantees the two models see byte-identical inputs."""
    return tokenize_promoter_pairs(_TOKENIZER_BUNDLE, chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-pairs", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    global _TOKENIZER_BUNDLE

    rows = [json.loads(line) for line in COHORT.open()]
    if args.limit:
        rows = rows[: args.limit]
    pairs = [
        (r["example_id"], r["target"]["clean_sequence"], r["control"]["clean_sequence"])
        for r in rows
    ]

    classifier = load_dnabert2_promoter_classifier()
    base = load_dnabert2_mlm()
    _TOKENIZER_BUNDLE = classifier

    import torch

    base_device = str(next(base.model.parameters()).device)

    # ------------------------------------------------------------------ M1: runtime tokenization
    mismatched_ids, mismatched_offsets = 0, 0
    for _, target, control in pairs:
        for sequence in (target, control):
            a = classifier.tokenizer(sequence, return_offsets_mapping=True, add_special_tokens=True)
            b = base.tokenizer(sequence, return_offsets_mapping=True, add_special_tokens=True)
            if a["input_ids"] != b["input_ids"]:
                mismatched_ids += 1
            if [tuple(x) for x in a["offset_mapping"]] != [tuple(x) for x in b["offset_mapping"]]:
                mismatched_offsets += 1
    memorials: dict[str, Any] = {
        "M1_runtime_tokenization_identity": {
            "sequences_checked": 2 * len(pairs),
            "input_id_mismatches": mismatched_ids,
            "offset_mismatches": mismatched_offsets,
            "passed": mismatched_ids == 0 and mismatched_offsets == 0,
            "note": "load-bearing: if this fails the band allocation differs between models and "
                    "no comparison is valid",
        },
        "M4_band_mass_normalisation": {
            "architecture_forced": True,
            "note": "asserted inside the capture (masses sum to 1 within 1e-5); flagged, not "
                    "counted as evidence",
        },
        "M5_base_checkpoint_identity": {
            "has_learned_position_table": base.provenance.get("has_learned_position_table"),
            "mlm_head_complete": base.provenance.get("mlm_head_complete"),
            "passed": base.provenance.get("mlm_head_complete") is True,
            "note": "the absent position table is an architecture fact and is flagged; the MLM "
                    "head presence is a real checkpoint check",
        },
    }
    if not memorials["M1_runtime_tokenization_identity"]["passed"]:
        raise RuntimeError("M1 failed: the two checkpoints do not tokenize this cohort identically")

    # ------------------------------------------------------------------ measure both models
    slopes = alibi_slopes(12)
    measured = {
        "fine_tuned": run_model(
            classifier.model, classifier.config, classifier.device, pairs, args.batch_pairs, "ft"
        ),
        "base": run_model(base.model, base.config, base_device, pairs, args.batch_pairs, "base"),
    }

    # ------------------------------------------------------------------ M2: batch invariance
    probe = pairs[:16]
    worst = 0.0
    for name, (model, config, device) in {
        "fine_tuned": (classifier.model, classifier.config, classifier.device),
        "base": (base.model, base.config, base_device),
    }.items():
        a = run_model(model, config, device, probe, 16, f"{name}:probeA")
        b = run_model(model, config, device, probe, 4, f"{name}:probeB")
        for head in a:
            for key in a[head]:
                worst = max(worst, abs(a[head][key] - b[head][key]))
    memorials["M2_batch_invariance"] = {
        "worst_absolute_difference": worst,
        "tolerance": 1e-5,
        "passed": worst < 1e-5,
        "probe_loci": len(probe),
    }

    # ------------------------------------------------------------------ prior and excess
    prior_samples: dict[int, dict[str, list[float]]] = {h: defaultdict(list) for h in range(12)}
    for _, target, _control in pairs:
        encoded = classifier.tokenizer(
            target, return_offsets_mapping=True, add_special_tokens=True
        )
        offsets = encoded["offset_mapping"]
        allocation = _token_band_allocations(
            [list(o) for o in offsets], tss_index=TSS_INDEX, bands=PROMOTER_SOURCE_BANDS
        ).tolist()
        for head, slope in enumerate(slopes):
            weights = [math.exp(-slope * j) for j in range(len(offsets))]
            total = sum(weights)
            for band_index, name in enumerate(BAND_ORDER):
                prior_samples[head][name].append(
                    sum(weights[j] * allocation[j][band_index] for j in range(len(offsets))) / total
                )
    prior = {h: {k: statistics.mean(v) for k, v in prior_samples[h].items()} for h in range(12)}

    per_head: dict[str, Any] = {}
    for label in measured["base"]:
        head_index = int(label.split("H")[1])
        record: dict[str, Any] = {
            "layer": int(label.split("H")[0][1:]),
            "head_index": head_index,
            "slope": slopes[head_index],
        }
        for model_name in ("base", "fine_tuned"):
            for name in BAND_ORDER:
                value = measured[model_name][label][f"target_{name}"]
                record[f"{model_name}_measured_{name}"] = value
                record[f"{model_name}_excess_{name}"] = value - prior[head_index][name]
        for name in BAND_ORDER:
            record[f"delta_excess_{name}"] = (
                record[f"fine_tuned_excess_{name}"] - record[f"base_excess_{name}"]
            )
        per_head[label] = record

    # ------------------------------------------------------------------ frozen §4 verdict
    def excess(label: str, model_name: str, band: str = "far_upstream") -> float:
        return per_head[label][f"{model_name}_excess_{band}"]

    carrier_base = {h: excess(h, "base") for h in CARRIERS}
    carrier_ft = {h: excess(h, "fine_tuned") for h in CARRIERS}
    if all(abs(v) < PRIOR_THRESHOLD for v in carrier_base.values()):
        verdict = "ALREADY_AT_PRIOR"
    elif any(abs(v) >= PRIOR_THRESHOLD for v in carrier_base.values()) and all(
        abs(carrier_ft[h]) < abs(carrier_base[h]) for h in CARRIERS if abs(carrier_base[h]) >= PRIOR_THRESHOLD
    ):
        verdict = "MOVED_ONTO_PRIOR"
    else:
        verdict = "OTHER_SEE_TABLE"

    labels = sorted(per_head)
    base_excess = [per_head[k]["base_excess_far_upstream"] for k in labels]
    ft_excess = [per_head[k]["fine_tuned_excess_far_upstream"] for k in labels]

    def pearson(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
        return num / den if den else 0.0

    payload: dict[str, Any] = {
        "study": "dnabert2_reconstruction_atlas",
        "experiment": "D16_finetuning_attention_shift",
        "n_loci": len(pairs),
        "checkpoints": {
            "base": {"hf_id": base.provenance["hf_id"], "revision": base.provenance["revision"]},
            "fine_tuned": {
                "hf_id": classifier.checkpoint.hf_id,
                "revision": classifier.checkpoint.revision,
            },
        },
        "memorials": memorials,
        "frozen_verdict": verdict,
        "carriers": {
            h: {
                "base_excess_far_upstream": carrier_base[h],
                "fine_tuned_excess_far_upstream": carrier_ft[h],
                "delta": carrier_ft[h] - carrier_base[h],
                "threshold": PRIOR_THRESHOLD,
            }
            for h in CARRIERS
        },
        "reader_L9H7": {
            band: {
                "base_excess": excess(READER, "base", band),
                "fine_tuned_excess": excess(READER, "fine_tuned", band),
                "delta": excess(READER, "fine_tuned", band) - excess(READER, "base", band),
            }
            for band in BAND_ORDER
        },
        "population": {
            "mean_abs_excess_far_upstream_base": statistics.mean(abs(v) for v in base_excess),
            "mean_abs_excess_far_upstream_fine_tuned": statistics.mean(abs(v) for v in ft_excess),
            "pearson_base_vs_fine_tuned": pearson(base_excess, ft_excess),
            "mean_abs_delta_excess_far_upstream": statistics.mean(
                abs(a - b) for a, b in zip(ft_excess, base_excess)
            ),
            "largest_absolute_shifts": sorted(
                (
                    {"head": k, "delta_excess_far_upstream": per_head[k]["delta_excess_far_upstream"]}
                    for k in labels
                ),
                key=lambda d: -abs(d["delta_excess_far_upstream"]),
            )[:10],
        },
        "per_head": per_head,
    }

    # M3 cross-cohort consistency, reported not gated
    atlas = json.loads(ATLAS_DEV.read_text())
    committed = {
        s["head_label"]: (s.get("attention") or {})
        for s in atlas["head_summaries"]
        if s["head_label"] in D5_HEADS
    }
    memorials["M3_cross_cohort_consistency"] = {
        "gated": False,
        "note": "D5 measured these on a DIFFERENT cohort; agreement is expected to be close but "
                "not exact. A gross divergence would indicate a capture bug.",
        "far_upstream_here_vs_D5": {
            h: [
                round(per_head[h]["fine_tuned_measured_far_upstream"], 4),
                round(committed[h]["far_upstream"]["target_mean"], 4),
            ]
            for h in D5_HEADS
            if h in committed and "far_upstream" in committed[h]
        },
    }

    write_result_bundle(
        args.out,
        payload,
        experiment_id="D16_finetuning_attention_shift",
        status="EXPLORATORY",
        model_identity={
            "base": base.provenance["hf_id"] + "@" + base.provenance["revision"],
            "fine_tuned": classifier.checkpoint.hf_id + "@" + classifier.checkpoint.revision,
        },
        input_paths=[COHORT, ATLAS_DEV],
        parameters={"n_loci": len(pairs), "tss_index": TSS_INDEX, "threshold": PRIOR_THRESHOLD},
        seeds={},
        command=sys.argv,
        repo=REPO,
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "per_head"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
