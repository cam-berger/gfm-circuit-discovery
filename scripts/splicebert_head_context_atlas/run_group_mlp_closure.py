#!/usr/bin/env python
"""Outcome-aware diagnostic closure of the donor-side 96-head screen.

This study answers two questions that independent single-head effects cannot:

1. What happens when ranked head sets are patched *jointly* (with all interactions
   preserved by the model)?
2. Do any of the six MLP residual writes mediate the same perturbation response?

The historical ``head_screen_v1`` acceptor rows used the wrong masked coordinate.
Accordingly this closure is intentionally donor-only (``alt5ss_donor``), where the
screen coordinate was correct.  It is EXPLORATORY and outcome-aware: rankings come
from the train-only screen reanalysis and no sealed chromosome is touched.

Effect convention is the same scrub-to-corrupt convention as the screen:

    exact effect = native-present margin - margin(native input with selected write(s)
                   replaced by their motif-destroyed-source values)

No single-head effects are summed.  ``C`` and ``D`` use the exact same ranked sets so
their magnitude contrast is paired and does not optimize a separate circuit per arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_all_head_screen import (  # noqa: E402
    BASES,
    ScreenModel,
    collateral_health,
    gate_destroy_real_bases,
    load_cohort,
    metrics_from_logits,
    target_base_position,
)
from gfmcd.circuits.hf_bert import capture_head_outputs, patch_heads  # noqa: E402
from gfmcd.splicing.circuit import capture_writes, clamp_write  # noqa: E402
from gfmcd.splicing.competition_seq import build_window  # noqa: E402
from gfmcd.splicing.coupling import destroy_acceptor, destroy_donor  # noqa: E402
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402
from gfmcd.splicing.specificity import find_matched_decoy  # noqa: E402


ALL_HEADS = tuple((layer, head) for layer in range(6) for head in range(16))
DEFAULT_GROUP_SIZES = (1, 3, 5, 10, 20, 40, 64, 96)
DEFAULT_RANDOM_SIZES = (5, 20, 40)
CONDITIONS = {
    "B": ("partner_win", "partner_is_donor"),
    "C": ("competitor_win", "competitor_is_donor"),
    "D": ("decoy_win", "competitor_is_donor"),
}


def parse_ints(text: str) -> tuple[int, ...]:
    vals = tuple(sorted(set(int(x.strip()) for x in text.split(",") if x.strip())))
    if not vals or vals[0] < 1 or vals[-1] > 96:
        raise ValueError("head-set sizes must be unique integers in [1, 96]")
    return vals


def parse_head(label: str) -> tuple[int, int]:
    if not label.startswith("L") or "H" not in label:
        raise ValueError(f"invalid head label {label!r}")
    layer, head = label[1:].split("H", 1)
    ans = (int(layer), int(head))
    if ans not in ALL_HEADS:
        raise ValueError(f"head outside SpliceBERT's 6x16 grid: {label!r}")
    return ans


def head_label(head: tuple[int, int]) -> str:
    return f"L{head[0]}H{head[1]}"


def load_rankings(path: Path) -> dict[str, tuple[tuple[int, int], ...]]:
    """Freeze two post-hoc rankings from the train-only reanalysis.

    B uses absolute donor-side signed effects. C/D share one ranking by the paired
    per-head magnitude excess ``E|C_h|-E|D_h|``.  This is deliberately a diagnostic
    ranking for the nonadditivity puzzle, not a new confirmation selection rule.
    """
    data = json.loads(path.read_text())
    directional = data["directional_B"]
    cd = data["C_minus_D_magnitude"]
    per_head = data.get("per_head_train_only", {})

    def cd_score(head: tuple[int, int]) -> float:
        label = head_label(head)
        if label in cd:
            return float(cd[label]["mean"])
        # The reanalysis stores cluster CIs only for claim-bearing candidates, but
        # retains all 96 raw train-only magnitude means.  The ranking needs the full
        # permutation, so reconstruct the same point estimand for the remaining heads.
        return (
            float(per_head[f"C|{label}"]["magnitude_mean"])
            - float(per_head[f"D|{label}"]["magnitude_mean"])
        )

    b = sorted(ALL_HEADS, key=lambda h: (-abs(float(directional[head_label(h)]["donor_mean"])), h))
    shared = sorted(ALL_HEADS, key=lambda h: (-cd_score(h), h))
    for name, rank in {"B_donor_abs": b, "CD_magnitude_excess": shared}.items():
        if len(rank) != 96 or set(rank) != set(ALL_HEADS):
            raise ValueError(f"{name} ranking is not a permutation of all 96 heads")
    return {"B": tuple(b), "CD": tuple(shared)}


def layer_matched_random_sets(
    selected: Iterable[tuple[int, int]], *, n: int, seed: int
) -> list[tuple[tuple[int, int], ...]]:
    """Return distinct random sets preserving the selected set's layer counts.

    Individual selected heads may occur in a null set; only the exact selected set is
    rejected.  That keeps every layer-count stratum feasible without pretending the
    null comes from a different architecture.
    """
    selected = tuple(sorted(selected))
    counts = Counter(layer for layer, _ in selected)
    if not selected or any(k > 16 for k in counts.values()):
        raise ValueError("invalid selected head set")
    # No alternative exists when every layer is saturated.
    total_alternatives = math.prod(math.comb(16, k) for k in counts.values()) - 1
    if total_alternatives <= 0 or n <= 0:
        return []
    target = min(n, total_alternatives)
    rng = random.Random(seed)
    out: set[tuple[tuple[int, int], ...]] = set()
    while len(out) < target:
        draw = tuple(sorted(
            (layer, head)
            for layer, k in sorted(counts.items())
            for head in rng.sample(range(16), k)
        ))
        if draw != selected:
            out.add(draw)
    return sorted(out)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_donor_windows(cfg: dict, *, n: int, per_gene_cap: int, min_separation: int):
    """Build a deterministic, train-only, donor-target cohort with matched decoys."""
    exs, ccfg, m5, m3 = load_cohort(cfg, min_separation, unlock_sealed=False)
    exs = [e for e in exs if e.split == "train" and e.site_type == "alt5ss_donor"]
    genome = PyfaidxGenome(cfg["fasta"])
    windows, decoys, per_gene = [], [], defaultdict(int)
    for ex in exs:
        if len(windows) >= n:
            break
        if per_gene[ex.gene_id] >= per_gene_cap:
            continue
        window = build_window(ex, genome, ccfg)
        if window is None:
            continue
        decoy, _ = find_matched_decoy(window, m5, m3)
        if decoy is None:
            continue
        if window.chrom in {"chr8", "chr9"} or window.split != "train":
            raise RuntimeError("sealed/non-train example reached the closure cohort")
        windows.append(window)
        decoys.append(int(decoy))
        per_gene[ex.gene_id] += 1
    return windows, decoys


def destroyed_sequences(windows, decoys, condition: str) -> list[str]:
    pos_attr, donor_attr = CONDITIONS[condition]
    out = []
    for window, decoy in zip(windows, decoys):
        pos = decoy if pos_attr == "decoy_win" else getattr(window, pos_attr)
        is_donor = getattr(window, donor_attr)
        variants = (destroy_donor if is_donor else destroy_acceptor)(window.seq, pos)
        damaged = variants[0]
        ok, msg = gate_destroy_real_bases(window.seq, damaged, pos, is_donor)
        if not ok:
            raise RuntimeError(f"invalid {condition} corruption for {window.example_id}: {msg}")
        out.append(damaged)
    return out


def score_logits(model: ScreenModel, logits: torch.Tensor, windows) -> tuple[np.ndarray, np.ndarray]:
    """Target margin plus the historical unmasked collateral proxy, per example."""
    margins, collateral = [], []
    for j, window in enumerate(windows):
        pos = target_base_position(window)
        native_id = model.base_ids[BASES.index(window.seq[pos])]
        margins.append(metrics_from_logits(
            logits[j, pos + model.tok_offset], native_id, model.base_ids
        )["margin"])
        probes = [q for q in range(30, len(window.seq) - 30, 137)][:6]
        native = [model.base_ids[BASES.index(window.seq[q])] for q in probes]
        collateral.append(collateral_health(
            logits[j], probes, native, model.base_ids, model.tok_offset
        ))
    return np.asarray(margins), np.asarray(collateral)


def gene_boot(rows: list[dict], value_key: str, *, seed: int, n_boot: int = 4000) -> dict:
    by_gene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if math.isfinite(value):
            by_gene[row["gene_id"]].append(value)
    genes = sorted(by_gene)
    if not genes:
        return {"n": 0, "n_genes": 0, "mean": None, "ci95": [None, None]}
    gene_means = np.asarray([np.mean(by_gene[g]) for g in genes], dtype=float)
    point = float(gene_means.mean())
    if len(genes) == 1:
        return {"n": len(rows), "n_genes": 1, "mean": point, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(genes), size=(n_boot, len(genes)))
    boot = gene_means[idx].mean(axis=1)
    return {
        "n": len(rows), "n_genes": len(genes), "mean": point,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["intervention_id"])].append(row)
    summary = {}
    for (condition, iid), block in sorted(grouped.items()):
        key = f"{condition}|{iid}"
        summary[key] = {
            "condition": condition,
            "intervention_type": block[0]["intervention_type"],
            "intervention_id": iid,
            "size": block[0]["size"],
            "effect_signed": gene_boot(block, "effect_margin", seed=stable_seed(key, "signed")),
            "effect_magnitude": gene_boot(block, "effect_magnitude", seed=stable_seed(key, "magnitude")),
            "collateral_proxy_signed": gene_boot(block, "effect_collateral", seed=stable_seed(key, "collateral")),
            "behavior_signed": gene_boot(block, "behavioral_effect", seed=stable_seed(key, "behavior")),
            "behavior_magnitude": gene_boot(block, "behavioral_magnitude", seed=stable_seed(key, "behavior_abs")),
        }
        behavior_mean = summary[key]["behavior_signed"]["mean"]
        effect_mean = summary[key]["effect_signed"]["mean"]
        summary[key]["signed_effect_over_behavior"] = (
            effect_mean / behavior_mean
            if behavior_mean is not None and abs(behavior_mean) >= 0.02 else None
        )

    # Paired C-D magnitude contrast for identically defined ranked groups / MLPs.
    by_key = {(r["example_id"], r["condition"], r["intervention_id"]): r for r in rows}
    contrasts = {}
    for (_condition, iid), cblock in grouped.items():
        if _condition != "C":
            continue
        pairs = []
        for c in cblock:
            d = by_key.get((c["example_id"], "D", iid))
            if d is None:
                continue
            pairs.append({
                "gene_id": c["gene_id"],
                "C_minus_D_abs_exact_effect": c["effect_magnitude"] - d["effect_magnitude"],
            })
        if pairs:
            contrasts[iid] = gene_boot(
                pairs, "C_minus_D_abs_exact_effect", seed=stable_seed(iid, "CD")
            )
    return {"per_intervention": summary, "paired_C_minus_D_magnitude": contrasts}


def intervention_row(window, condition: str, iid: str, itype: str, size: int,
                     base_margin: float, corrupt_margin: float, patched_margin: float,
                     base_collateral: float, patched_collateral: float) -> dict:
    effect = float(base_margin - patched_margin)
    behavior = float(base_margin - corrupt_margin)
    return {
        "example_id": window.example_id,
        "gene_id": window.gene_id,
        "chrom": window.chrom,
        "split": window.split,
        "site_type": window.site_type,
        "condition": condition,
        "intervention_type": itype,
        "intervention_id": iid,
        "size": size,
        "baseline_margin": float(base_margin),
        "counterfactual_margin": float(corrupt_margin),
        "patched_margin": float(patched_margin),
        "behavioral_effect": behavior,
        "behavioral_magnitude": abs(behavior),
        "effect_margin": effect,
        "effect_magnitude": abs(effect),
        "effect_collateral": float(base_collateral - patched_collateral),
    }


def run_closure(model: ScreenModel, windows, decoys, rankings, *, group_sizes,
                random_sizes, n_random: int, batch_size: int, self_test: bool,
                conditions: tuple[str, ...] = ("B", "C", "D"),
                include_mlp: bool = True) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    gates: list[dict] = []
    mlp_blocks = tuple((layer, "mlp") for layer in range(model.a.num_layers))
    for start in range(0, len(windows), batch_size):
        batch_w = windows[start:start + batch_size]
        batch_d = decoys[start:start + batch_size]
        seqs = [w.seq for w in batch_w]
        targets = [target_base_position(w) for w in batch_w]
        if any(w.seq[p] != "G" for w, p in zip(batch_w, targets)):
            raise RuntimeError("target-coordinate gate failed: native masked base is not G")
        enc_present = model.enc(seqs, targets)
        base_logits = model.logits(enc_present)
        base_margin, base_collateral = score_logits(model, base_logits, batch_w)

        if self_test and start == 0:
            present_mlp: dict = {}
            if include_mlp:
                present_heads = capture_head_outputs(
                    model.a,
                    lambda: present_mlp.update(capture_writes(
                        model.a, mlp_blocks, lambda: model.logits(enc_present)
                    )),
                )
            else:
                present_heads = capture_head_outputs(model.a, lambda: model.logits(enc_present))
            with patch_heads(model.a, present_heads, ALL_HEADS):
                noop_heads = model.logits(enc_present)
            gate_values = {
                "source_to_source_all_heads": float((noop_heads - base_logits).abs().max()),
            }
            if include_mlp:
                with clamp_write(model.a, mlp_blocks, present_mlp):
                    noop_mlp = model.logits(enc_present)
                gate_values["source_to_source_all_mlps"] = float((noop_mlp - base_logits).abs().max())
            for name, value in gate_values.items():
                gates.append({"gate": name, "value": value, "pass": value < 1e-3})
                if value >= 1e-3:
                    raise RuntimeError(f"{name} failed: max difference {value}")

        for condition in conditions:
            damaged = destroyed_sequences(batch_w, batch_d, condition)
            enc_corrupt = model.enc(damaged, targets)
            corrupt_logits = model.logits(enc_corrupt)
            corrupt_margin, _ = score_logits(model, corrupt_logits, batch_w)
            mlp_cache: dict = {}
            if include_mlp:
                head_cache = capture_head_outputs(
                    model.a,
                    lambda: mlp_cache.update(capture_writes(
                        model.a, mlp_blocks, lambda: model.logits(enc_corrupt)
                    )),
                )
            else:
                head_cache = capture_head_outputs(model.a, lambda: model.logits(enc_corrupt))

            ranking = rankings["B"] if condition == "B" else rankings["CD"]
            for size in group_sizes:
                heads = ranking[:size]
                with patch_heads(model.a, head_cache, heads):
                    patched_logits = model.logits(enc_present)
                patched_margin, patched_collateral = score_logits(model, patched_logits, batch_w)
                iid = f"ranked_heads_k{size:02d}"
                for j, window in enumerate(batch_w):
                    rows.append(intervention_row(
                        window, condition, iid, "joint_ranked_head_set", size,
                        base_margin[j], corrupt_margin[j], patched_margin[j],
                        base_collateral[j], patched_collateral[j],
                    ))

                if size in random_sizes:
                    random_sets = layer_matched_random_sets(
                        heads,
                        n=n_random,
                        seed=stable_seed("B" if condition == "B" else "CD", size, "random-head-null"),
                    )
                    for draw, random_heads in enumerate(random_sets):
                        with patch_heads(model.a, head_cache, random_heads):
                            random_logits = model.logits(enc_present)
                        random_margin, random_collateral = score_logits(model, random_logits, batch_w)
                        rid = f"layer_matched_random_k{size:02d}_d{draw:02d}"
                        for j, window in enumerate(batch_w):
                            rows.append(intervention_row(
                                window, condition, rid, "layer_matched_random_head_set", size,
                                base_margin[j], corrupt_margin[j], random_margin[j],
                                base_collateral[j], random_collateral[j],
                            ))

            if include_mlp:
                for layer in range(model.a.num_layers):
                    block = ((layer, "mlp"),)
                    with clamp_write(model.a, block, mlp_cache):
                        patched_logits = model.logits(enc_present)
                    patched_margin, patched_collateral = score_logits(model, patched_logits, batch_w)
                    iid = f"mlp_L{layer}"
                    for j, window in enumerate(batch_w):
                        rows.append(intervention_row(
                            window, condition, iid, "single_mlp_write", 1,
                            base_margin[j], corrupt_margin[j], patched_margin[j],
                            base_collateral[j], patched_collateral[j],
                        ))

                with clamp_write(model.a, mlp_blocks, mlp_cache):
                    all_mlp_logits = model.logits(enc_present)
                all_mlp_margin, all_mlp_collateral = score_logits(model, all_mlp_logits, batch_w)
                for j, window in enumerate(batch_w):
                    rows.append(intervention_row(
                        window, condition, "mlp_all6", "joint_all_mlp_writes", 6,
                        base_margin[j], corrupt_margin[j], all_mlp_margin[j],
                        base_collateral[j], all_mlp_collateral[j],
                    ))

        print(f"[closure] {min(start + batch_size, len(windows))}/{len(windows)}", file=sys.stderr)
    return rows, gates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ranking", required=True, help="train-only head_effect_reanalysis.json")
    ap.add_argument("--out", default=str(
        REPO / "results/splicebert_head_context_atlas/group_mlp_closure_v1.json"
    ))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--per-gene-cap", type=int, default=1)
    ap.add_argument("--min-separation", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--group-sizes", default=",".join(map(str, DEFAULT_GROUP_SIZES)))
    ap.add_argument("--random-sizes", default=",".join(map(str, DEFAULT_RANDOM_SIZES)))
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--conditions", default="B,C,D", help="comma-separated subset of B,C,D")
    ap.add_argument("--skip-mlp", action="store_true", help="run only exact joint head-set curves")
    args = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    group_sizes = parse_ints(args.group_sizes)
    random_sizes = parse_ints(args.random_sizes) if args.random_sizes else ()
    if not set(random_sizes).issubset(group_sizes):
        raise ValueError("random-control sizes must be a subset of ranked group sizes")
    conditions = tuple(x.strip() for x in args.conditions.split(",") if x.strip())
    if not conditions or not set(conditions).issubset(CONDITIONS):
        raise ValueError("--conditions must be a non-empty subset of B,C,D")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refuse to overwrite immutable result {out}")
    cfg_path = Path(args.config)
    ranking_path = Path(args.ranking)
    cfg = json.loads(cfg_path.read_text())
    rankings = load_rankings(ranking_path)

    from gfmcd.splicing.circuit import load_adapter, pick_device
    device = pick_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = load_adapter(cfg["model"], device, eager=False)
    model = ScreenModel(adapter, batch_size=args.batch)
    windows, decoys = build_donor_windows(
        cfg, n=args.n, per_gene_cap=args.per_gene_cap, min_separation=args.min_separation
    )
    if not windows:
        raise RuntimeError("no eligible donor windows")
    print(f"[closure] {len(windows)} train-only donor windows on {device}", file=sys.stderr)
    t0 = time.time()
    rows, gates = run_closure(
        model, windows, decoys, rankings,
        group_sizes=group_sizes, random_sizes=random_sizes,
        n_random=args.n_random, batch_size=args.batch, self_test=args.self_test,
        conditions=conditions, include_mlp=not args.skip_mlp,
    )
    report = {
        "schema": "head_context_atlas.group_mlp_closure/1",
        "status": "EXPLORATORY_OUTCOME_AWARE_DIAGNOSTIC",
        "scope": "train-only alt5ss_donor; chr8/9 sealed; one canonical destroy variant",
        "coordinate_note": (
            "donor-only because historical head_screen_v1 acceptor rows masked the wrong coordinate; "
            "donor rows and donor-derived rankings are valid"
        ),
        "n_examples": len(windows),
        "n_genes": len({w.gene_id for w in windows}),
        "group_sizes": list(group_sizes),
        "random_sizes": list(random_sizes),
        "n_random_per_size": args.n_random,
        "conditions": list(conditions),
        "include_mlp": not args.skip_mlp,
        "rankings": {k: [head_label(h) for h in v] for k, v in rankings.items()},
        "estimands": {
            "B": "signed exact joint effect; donor targets only",
            "C_D": "magnitude of exact joint effect; same CD-ranked set in both arms",
            "C_minus_D": "paired E[|joint C effect|-|joint D effect|]",
            "MLP": "exact scrub-to-corrupt residual-write patch, each MLP and all six jointly",
            "collateral": "unmasked true-base logprob proxy; not proper masked-MLM health",
        },
        "mechanical_gates": gates,
        "summary": summarize(rows),
        "rows": rows,
        "provenance": {
            "config": str(cfg_path.resolve()),
            "ranking": str(ranking_path.resolve()),
            "device": device,
            "precision": "float32 logits; TF32/AMP off",
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "seconds": time.time() - t0,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    try:
        from gfmcd.provenance import ExperimentManifest
        manifest = ExperimentManifest.create(
            experiment_id="head_context_atlas_group_mlp_closure_v1",
            command=["python"] + sys.argv,
            status="EXPLORATORY_OUTCOME_AWARE_DIAGNOSTIC",
        )
        manifest.add_input(str(cfg_path))
        manifest.add_input(str(ranking_path))
        manifest.add_output(str(out))
        manifest.write(str(out.with_suffix(".manifest.json")))
    except Exception as exc:  # pragma: no cover - best-effort provenance wrapper
        print(f"WARN manifest: {exc}", file=sys.stderr)
    print(f"[closure] wrote {out} ({len(rows)} rows, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
