#!/usr/bin/env python
"""Benchmark the all-head screen on 8-16 examples: throughput, memory, projected full-run cost.

Does NOT continue to the full run. Report is for Cameron's GO decision after review.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for run_all_head_screen
from run_all_head_screen import ScreenModel, screen_chunk, load_cohort  # noqa: E402
from gfmcd.splicing.competition_seq import build_window
from gfmcd.splicing.specificity import find_matched_decoy
from gfmcd.splicing.dataset import PyfaidxGenome
from gfmcd.splicing.circuit import load_adapter, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--full-n", type=int, default=4864, help="projected full discovery example count (usable genes)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = json.loads(Path(args.config).read_text())
    dev = pick_device(None)
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else dev
    mp = cfg["model"] if Path(cfg["model"]).is_absolute() else str(REPO / cfg["model"])
    adapter = load_adapter(mp, dev, eager=False)
    model = ScreenModel(adapter, batch_size=args.batch)
    exs, ccfg, m5, m3 = load_cohort(cfg, 30, unlock_sealed=False)
    genome = PyfaidxGenome(cfg["fasta"])

    windows, decoys, per_gene = [], [], defaultdict(int)
    for e in exs:
        if len(windows) >= args.n:
            break
        if per_gene[e.gene_id] >= 3:
            continue
        w = build_window(e, genome, ccfg)
        if w is None:
            continue
        dq, _ = find_matched_decoy(w, m5, m3)
        if dq is None:
            continue
        windows.append(w); decoys.append(dq); per_gene[e.gene_id] += 1

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    recs, gates = screen_chunk(model, windows[:args.batch], decoys[:args.batch], self_test=True)
    dt = time.time() - t0
    peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else None

    per_ex = dt / max(len(windows[:args.batch]), 1)
    shard_bytes = len(json.dumps(next(iter(recs.values()))))
    proj_hours = per_ex * args.full_n / 3600
    proj_storage_mb = shard_bytes * args.full_n / 1e6
    report = {
        "device": dev, "gpu": gpu, "batch": args.batch, "benchmarked_examples": len(windows[:args.batch]),
        "peak_gpu_mem_gb": peak_mem_gb, "seconds_per_example": round(per_ex, 3),
        "examples_per_hour": round(3600 / per_ex, 1),
        "seconds_per_head_intervention": round(per_ex / (4 * 96), 5),
        "projected_full_run_gpu_hours": round(proj_hours, 2),
        "projected_full_run_n": args.full_n,
        "shard_bytes": shard_bytes, "projected_storage_mb": round(proj_storage_mb, 1),
        "gate_failures": [g for g in gates if not g[1]],
        "recommended_batch": args.batch, "recommended_head_chunk": 96,
        "precision": "float32; TF32/AMP off",
        "STOP": "Do NOT continue to the full run without Cameron's explicit GO after reviewing this projection.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
