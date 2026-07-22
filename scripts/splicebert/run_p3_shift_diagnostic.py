#!/usr/bin/env python
"""Exploratory current-use diagnostic for the SpliceBERT period-3 carrier.

See EXPLORATORY_P3_SHIFT_DIAGNOSTIC_V1.md. This runner deliberately does not use the word
"origin" in its verdicts: a frozen checkpoint can expose current use, not formation pressure.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ACC_WINDOW_OFFSETS,
    CORRUPT_ALT,
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    gboot,
    load_adapter,
    load_examples,
    pick_device,
)
from gfmcd.splicing.coupling import CouplingExample, substitute  # noqa: E402
from gfmcd.splicing.dataset import PyfaidxGenome, reverse_complement  # noqa: E402
from gfmcd.splicing.phase import example_phase, load_cds_frames  # noqa: E402

BAND = tuple(range(339, 346))
SHIFTS = (0, 1, 2, 3)
KINDS = ("window_translation", "intron_insert", "exon_insert", "upstream_intron_sham")
MOTIFS = ("AG", "GT", "GC")
READER = (3, 1)
CHANCE = 0.25
HEALTH_HEADROOM_FLOOR = 0.50


def motif_counts(seq: str) -> dict[str, int]:
    return {m: sum(seq[i:i + 2] == m for i in range(len(seq) - 1)) for m in MOTIFS}


def choose_insertion(seq: str, pos: int, size: int, radius: int = 18) -> str:
    """Choose a deterministic local-composition match that creates no local splice motif."""
    if size == 0:
        return ""
    lo, hi = max(0, pos - radius), min(len(seq), pos + radius)
    original = seq[lo:hi]
    target = Counter(original)
    total = max(len(original), 1)
    best: tuple[float, str] | None = None
    for tup in itertools.product("ACGT", repeat=size):
        ins = "".join(tup)
        edited = seq[lo:pos] + ins + seq[pos:hi]
        before, after = motif_counts(original), motif_counts(edited)
        if any(after[m] > before[m] for m in MOTIFS):
            continue
        got = Counter(ins)
        composition_error = sum(abs(got[b] / size - target[b] / total) for b in "ACGT")
        # Prefer native-looking local continuations when composition ties; lexical is final tie-break.
        continuation_error = float(ins[0] != seq[pos - 1]) + float(ins[-1] != seq[pos])
        score = composition_error + 0.05 * continuation_error
        cand = (score, ins)
        if best is None or cand < best:
            best = cand
    if best is None:
        raise ValueError(f"no no-new-motif insertion at {pos} size={size}")
    return best[1]


def _shift_tuple(values: tuple[int, ...], pos: int, size: int) -> tuple[int, ...]:
    return tuple(v + size if v >= pos else v for v in values if v < 1024 - size)


def insertion_variant(x: CouplingExample, kind: str, size: int) -> CouplingExample:
    if size == 0:
        return x
    if x.next_acceptor_pos is None:
        raise ValueError("missing partner acceptor")
    gap = x.next_acceptor_pos - x.donor_pos
    if kind == "intron_insert":
        pos = x.donor_pos + gap // 2
    elif kind == "exon_insert":
        pos = x.donor_pos - 30
    elif kind == "upstream_intron_sham":
        pos = x.acceptor_pos - 30
    else:
        raise ValueError(kind)
    if not (12 <= pos < len(x.sequence) - 40):
        raise ValueError(f"unsafe insertion coordinate {pos}")
    ins = choose_insertion(x.sequence, pos, size)
    local_lo, local_hi = max(0, pos - 18), min(len(x.sequence), pos + 18)
    before = motif_counts(x.sequence[local_lo:local_hi])
    sequence = x.sequence[:pos] + ins + x.sequence[pos:len(x.sequence) - size]
    after = motif_counts(sequence[local_lo:local_hi + size])
    if any(after[m] > before[m] for m in MOTIFS):
        raise AssertionError("insertion created a local splice motif")

    def moved(v: int | None) -> int | None:
        return None if v is None else v + size if v >= pos else v

    md = dict(x.metadata)
    md["p3_shift_variant"] = {"kind": kind, "size": size, "insert_pos": pos, "insert": ins}
    return replace(
        x,
        sequence=sequence,
        acceptor_pos=moved(x.acceptor_pos),
        donor_pos=moved(x.donor_pos),
        next_acceptor_pos=moved(x.next_acceptor_pos),
        other_ag=_shift_tuple(x.other_ag, pos, size),
        other_gt=_shift_tuple(x.other_gt, pos, size),
        real_acc=_shift_tuple(x.real_acc, pos, size),
        real_don=_shift_tuple(x.real_don, pos, size),
        metadata=md,
    )


def translation_variant(x: CouplingExample, size: int, genome: PyfaidxGenome) -> CouplingExample:
    """Translate the same natural genomic content right by `size` absolute token positions."""
    if size == 0:
        return x
    ws, we = x.metadata["genomic_window_start"], x.metadata["genomic_window_end"]
    if x.strand == "+":
        nws, nwe = ws - size, we - size
    else:
        nws, nwe = ws + size, we + size
    seq = genome.fetch(x.chrom, nws, nwe)
    if x.strand == "-":
        seq = reverse_complement(seq)
    seq = seq.upper().replace("U", "T")
    if len(seq) != len(x.sequence):
        raise ValueError("natural translation changed window length")
    # All but the newly introduced left flank and dropped right flank must be byte-identical.
    if seq[size:] != x.sequence[:-size]:
        raise AssertionError("natural-flank translation is not a pure content translation")
    md = dict(x.metadata)
    md.update({"genomic_window_start": nws, "genomic_window_end": nwe,
               "p3_shift_variant": {"kind": "window_translation", "size": size}})
    return replace(
        x,
        sequence=seq,
        acceptor_pos=x.acceptor_pos + size,
        donor_pos=x.donor_pos + size,
        next_acceptor_pos=x.next_acceptor_pos + size if x.next_acceptor_pos is not None else None,
        other_ag=tuple(v + size for v in x.other_ag if v + size < len(seq)),
        other_gt=tuple(v + size for v in x.other_gt if v + size < len(seq)),
        real_acc=tuple(v + size for v in x.real_acc if v + size < len(seq)),
        real_don=tuple(v + size for v in x.real_don if v + size < len(seq)),
        metadata=md,
    )


def build_variant(x: CouplingExample, kind: str, size: int,
                  genome: PyfaidxGenome) -> CouplingExample:
    if kind == "window_translation":
        return translation_variant(x, size, genome)
    return insertion_variant(x, kind, size)


def validate_variant(base: CouplingExample, variant: CouplingExample, kind: str, size: int) -> None:
    if len(variant.sequence) != len(base.sequence) or len(variant.sequence) != 1024:
        raise AssertionError("variant length invariant failed")
    if variant.sequence[variant.donor_pos:variant.donor_pos + 2] != "GT":
        raise AssertionError("donor motif changed")
    a = variant.next_acceptor_pos
    if a is None or variant.sequence[a - 2:a] != "AG":
        raise AssertionError("partner acceptor motif changed")
    if size == 0 and variant != base:
        raise AssertionError("zero-shift constructor is not the identity")
    # The readout and partner-site local contexts must be identical after coordinate adjustment.
    for p0, p1 in ((base.donor_pos, variant.donor_pos),
                   (base.next_acceptor_pos, variant.next_acceptor_pos)):
        if base.sequence[p0 - 8:p0 + 8] != variant.sequence[p1 - 8:p1 + 8]:
            raise AssertionError(f"{kind}/{size} altered a protected site context")


def make_deleted(P0: torch.Tensor) -> torch.Tensor:
    P = P0.detach().float().cpu().numpy()
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    F[list(BAND)] = 0.0
    return torch.tensor(np.fft.irfft(F, n=P.shape[0], axis=0) + mu,
                        dtype=P0.dtype, device=P0.device)


@contextmanager
def position_state(embedding, weight: torch.Tensor):
    original = embedding.weight.detach().clone()
    with torch.no_grad():
        embedding.weight.copy_(weight)
    try:
        yield
    finally:
        with torch.no_grad():
            embedding.weight.copy_(original)


def attention_metrics(row: np.ndarray, donor: int, acceptor: int) -> dict:
    """Metrics from L3H1's masked-donor query; all indices here are nucleotide indices."""
    lo, hi = donor + 3, acceptor + 1
    if hi <= lo:
        raise ValueError("empty donor-to-acceptor attention span")
    keys = np.arange(lo, hi)
    vals = np.asarray(row[lo:hi], dtype=float)
    if vals.size == 0 or not np.isfinite(vals).all() or vals.sum() <= 0:
        raise ValueError("dead attention row")
    order = np.argsort(vals)[::-1][:5]
    peak = int(keys[order[0]])
    span_mass = float(vals.sum())

    def masses(origin: int) -> list[float]:
        raw = np.array([vals[(keys - origin) % 3 == r].sum() for r in range(3)])
        return (raw / max(raw.sum(), 1e-12)).tolist()

    acc_idx = np.array([acceptor + k for k in ACC_WINDOW_OFFSETS])
    return {
        "peak_abs": peak,
        "peak_weight": float(vals[order[0]]),
        "top5_abs": [int(keys[i]) for i in order],
        "top5_weight": [float(vals[i]) for i in order],
        "peak_rel_donor": peak - donor,
        "peak_rel_acceptor": peak - acceptor,
        "peak_abs_mod3": peak % 3,
        "span_mass": span_mass,
        "acceptor_mass": float(row[acc_idx].sum()),
        "mass_abs_mod3": masses(0),
        "mass_rel_donor_mod3": masses(donor),
        "mass_rel_acceptor_mod3": masses(acceptor),
    }


def evaluate_cell(ad, task: CouplingTask, examples: list[CouplingExample]) -> dict[str, list]:
    data: dict[str, list] = {
        "clean_margin": [], "corrupt_margin": [], "coupling": [],
        "peak_abs": [], "peak_weight": [], "peak_rel_donor": [],
        "peak_rel_acceptor": [], "peak_abs_mod3": [], "span_mass": [],
        "acceptor_mass": [], "mass_rel_donor_mod3": [], "mass_rel_acceptor_mod3": [],
        "mass_abs_mod3": [],
        "top5_abs": [], "top5_weight": [],
    }
    with torch.no_grad():
        for i in range(0, len(examples), task.batch):
            ch = examples[i:i + task.batch]
            clean, corrupt, donors = task.pair(ch)
            out = ad.model(**clean, output_attentions=True)
            mc = task.margin(out.logits.float(), donors)
            mk = task.margin(ad.model(**corrupt).logits.float(), donors)
            A = out.attentions[READER[0]][:, READER[1]].float().cpu().numpy()
            for j, x in enumerate(ch):
                q = x.donor_pos + task.off
                # Remove special tokens so attention_metrics works in nucleotide coordinates.
                row = A[j, q, task.off:task.off + len(x.sequence)]
                am = attention_metrics(row, x.donor_pos, x.next_acceptor_pos)
                data["clean_margin"].append(float(mc[j]))
                data["corrupt_margin"].append(float(mk[j]))
                data["coupling"].append(float(mc[j] - mk[j]))
                for key, value in am.items():
                    data[key].append(value)
    return data


def health(ad, task: CouplingTask, examples: list[CouplingExample], n_probe: int, seed: int) -> float:
    return task.mlm_health(examples, n_probe=n_probe, seed=seed)


def arr(cells: dict, state: str, kind: str, shift: int, metric: str) -> np.ndarray:
    return np.asarray(cells[state][kind][str(shift)][metric], dtype=float)


def contrast_summary(cells: dict, genes: list[str], state: str, kind: str,
                     metric: str) -> dict:
    y = {d: arr(cells, state, kind, d, metric) for d in SHIFTS}
    saw = (y[0] + y[3]) / 2 - (y[1] + y[2]) / 2
    return {"mean": float(saw.mean()), "ci": list(gboot(saw, genes)), "per_example": saw.tolist()}


def peak_tracking(cells: dict, kind: str, state: str) -> dict:
    p0 = arr(cells, state, kind, 0, "peak_abs").astype(int)
    rd0 = arr(cells, state, kind, 0, "peak_rel_donor")
    ra0 = arr(cells, state, kind, 0, "peak_rel_acceptor")
    out = {}
    for d in SHIFTS[1:]:
        p = arr(cells, state, kind, d, "peak_abs").astype(int)
        rd = arr(cells, state, kind, d, "peak_rel_donor")
        ra = arr(cells, state, kind, d, "peak_rel_acceptor")
        md = arr(cells, state, kind, d, "mass_rel_donor_mod3")
        ma = arr(cells, state, kind, d, "mass_rel_acceptor_mod3")
        out[str(d)] = {
            "absolute_exact": float(np.mean(p == p0)),
            "absolute_within1": float(np.mean(np.abs(p - p0) <= 1)),
            "absolute_mod3_same": float(np.mean((p % 3) == (p0 % 3))),
            "donor_relative_exact": float(np.mean(rd == rd0)),
            "donor_relative_within1": float(np.mean(np.abs(rd - rd0) <= 1)),
            "acceptor_relative_exact": float(np.mean(ra == ra0)),
            "acceptor_relative_within1": float(np.mean(np.abs(ra - ra0) <= 1)),
            "mean_peak_shift": float(np.mean(p - p0)),
            "mean_mass_rel_donor_mod3": md.mean(axis=0).tolist(),
            "mean_mass_rel_acceptor_mod3": ma.mean(axis=0).tolist(),
            "dominant_donor_residue_counts": np.bincount(md.argmax(axis=1), minlength=3).tolist(),
            "dominant_acceptor_residue_counts": np.bincount(ma.argmax(axis=1), minlength=3).tolist(),
        }
    md0 = arr(cells, state, kind, 0, "mass_rel_donor_mod3")
    ma0 = arr(cells, state, kind, 0, "mass_rel_acceptor_mod3")
    out["0_baseline"] = {
        "mean_mass_rel_donor_mod3": md0.mean(axis=0).tolist(),
        "mean_mass_rel_acceptor_mod3": ma0.mean(axis=0).tolist(),
        "dominant_donor_residue_counts": np.bincount(md0.argmax(axis=1), minlength=3).tolist(),
        "dominant_acceptor_residue_counts": np.bincount(ma0.argmax(axis=1), minlength=3).tolist(),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--health-n", type=int, default=8)
    ap.add_argument("--health-probes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/phase-lattice/p3_shift_diagnostic_v1.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    t0 = time.time()

    device = pick_device(args.device)
    ad = load_adapter(args.model, device, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    genome = PyfaidxGenome(args.fasta)
    emb = ad.model.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    Pdel = make_deleted(P0)
    reconstruction_error = float((Pdel + (P0 - Pdel) - P0).abs().max())

    pool_n = max(args.n * 8, 64)
    pool, _ = load_examples(args.gtf, args.fasta, args.split, pool_n, task=task,
                            tag=f"p3-shift-{args.split}-{args.seed}", min_effect=None)
    chroms = {x.chrom if x.chrom.startswith("chr") else f"chr{x.chrom}" for x in pool}
    cds = load_cds_frames(args.gtf, allowed_chromosomes=chroms)
    cohort: list[CouplingExample] = []
    variants: dict[str, dict[str, list[CouplingExample]]] = {
        k: {str(d): [] for d in SHIFTS} for k in KINDS
    }
    skipped = Counter()
    for x in pool:
        if x.gene_type != "protein_coding" or example_phase(x, cds) is None:
            skipped["not_unambiguous_cds"] += 1
            continue
        if x.exon_length < 70 or x.next_acceptor_pos - x.donor_pos < 80:
            skipped["geometry"] += 1
            continue
        try:
            built = {k: {str(d): build_variant(x, k, d, genome) for d in SHIFTS} for k in KINDS}
            for k in KINDS:
                for d in SHIFTS:
                    validate_variant(x, built[k][str(d)], k, d)
        except (AssertionError, ValueError, KeyError) as exc:
            skipped[type(exc).__name__ + ":" + str(exc)] += 1
            continue
        cohort.append(x)
        for k in KINDS:
            for d in SHIFTS:
                variants[k][str(d)].append(built[k][str(d)])
        if len(cohort) >= (6 if args.smoke else args.n):
            break
    if len(cohort) < (6 if args.smoke else args.n):
        raise RuntimeError(f"only {len(cohort)} eligible examples; skips={dict(skipped)}")
    genes = [x.gene_id for x in cohort]

    print(f"P3 shift diagnostic: n={len(cohort)} genes={len(set(genes))} device={device}")
    print(f"carrier reconstruction error={reconstruction_error:.2e}")
    print(f"skips={dict(skipped)}")

    cells: dict = {"intact": {}, "deleted": {}}
    health_rows: dict = {"intact": {}, "deleted": {}}
    for state, weight in (("intact", P0), ("deleted", Pdel)):
        cells[state] = {k: {} for k in KINDS}
        health_rows[state] = {k: {} for k in KINDS}
        with position_state(emb, weight):
            for kind in KINDS:
                for d in SHIFTS:
                    print(f"  {state:7s} {kind:20s} d={d}", end="\r", flush=True)
                    cell = evaluate_cell(ad, task, variants[kind][str(d)])
                    cells[state][kind][str(d)] = cell
                    hset = variants[kind][str(d)][:min(args.health_n, len(cohort))]
                    health_rows[state][kind][str(d)] = health(
                        ad, task, hset, args.health_probes, args.seed + d)
    print()

    continuous_metrics = ("coupling", "clean_margin", "acceptor_mass", "span_mass", "peak_weight")
    contrasts: dict = {}
    carrier_specific_per_example: dict = {}
    for kind in KINDS:
        contrasts[kind] = {}
        carrier_specific_per_example[kind] = {}
        for metric in continuous_metrics:
            intact = contrast_summary(cells, genes, "intact", kind, metric)
            deleted = contrast_summary(cells, genes, "deleted", kind, metric)
            interaction = (np.asarray(intact["per_example"]) - np.asarray(deleted["per_example"]))
            carrier_specific_per_example[kind][metric] = interaction
            contrasts[kind][metric] = {
                "inphase_minus_offphase_intact": {k: v for k, v in intact.items() if k != "per_example"},
                "inphase_minus_offphase_deleted": {k: v for k, v in deleted.items() if k != "per_example"},
                "carrier_specific": {"mean": float(interaction.mean()),
                                     "ci": list(gboot(interaction, genes))},
            }

    location_specific: dict = {}
    for kind in ("intron_insert", "exon_insert"):
        location_specific[kind] = {}
        for metric in continuous_metrics:
            vals = (carrier_specific_per_example[kind][metric]
                    - carrier_specific_per_example["upstream_intron_sham"][metric])
            location_specific[kind][metric] = {
                "mean": float(vals.mean()), "ci": list(gboot(vals, genes))
            }

    tracking = {state: {kind: peak_tracking(cells, kind, state) for kind in KINDS}
                for state in ("intact", "deleted")}

    health_values = [v for s in health_rows.values() for k in s.values() for v in k.values()]
    health_retained = {k: {} for k in KINDS}
    for k in KINDS:
        for d in SHIFTS:
            intact_h = health_rows["intact"][k][str(d)]
            deleted_h = health_rows["deleted"][k][str(d)]
            health_retained[k][str(d)] = ((deleted_h - CHANCE)
                                          / max(intact_h - CHANCE, 1e-12))
    gates = {
        "carrier_reconstruction_lt_1e6": reconstruction_error < 1e-6,
        "all_lengths_1024": all(len(x.sequence) == 1024 for k in KINDS for d in SHIFTS
                                for x in variants[k][str(d)]),
        "all_donor_gt": all(x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
                            for k in KINDS for d in SHIFTS for x in variants[k][str(d)]),
        "all_acceptor_ag": all(x.sequence[x.next_acceptor_pos - 2:x.next_acceptor_pos] == "AG"
                               for k in KINDS for d in SHIFTS for x in variants[k][str(d)]),
        "attention_live": all(np.isfinite(cells[s][k][str(d)]["peak_weight"]).all()
                              for s in cells for k in KINDS for d in SHIFTS),
        "health_above_chance": min(health_values) > CHANCE,
        "deleted_health_headroom_retained_ge_0_5": (
            min(v for k in health_retained.values() for v in k.values()) >= HEALTH_HEADROOM_FLOOR
        ),
    }
    gates["all_pass"] = all(gates.values())

    result = {
        "status": "EXPLORATORY_CURRENT_USE_DIAGNOSTIC",
        "interpretation_boundary": (
            "Frozen-checkpoint shifts test present-day positional/frame sensitivity; they do not "
            "identify why training formed the carrier."
        ),
        "config": {"split": args.split, "n": len(cohort), "n_genes": len(set(genes)),
                   "batch": args.batch, "health_n": args.health_n,
                   "health_probes": args.health_probes, "seed": args.seed,
                   "device": device, "band": list(BAND), "reader": list(READER),
                   "kinds": list(KINDS), "shifts": list(SHIFTS), "smoke": args.smoke},
        "cohort": {"example_ids": [x.example_id for x in cohort], "genes": genes,
                   "intron_phase": [example_phase(x, cds) for x in cohort],
                   "skipped": dict(skipped)},
        "gates": gates,
        "health_top1": health_rows,
        "deleted_health_headroom_retained": health_retained,
        "contrasts": contrasts,
        "location_specific_target_minus_upstream_intron_sham": location_specific,
        "peak_tracking": tracking,
        "cells": cells,
        "runtime_seconds": time.time() - t0,
    }
    out = REPO / args.out
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    manifest_path = out.with_suffix(".manifest.json")
    manifest = ExperimentManifest.create(
        f"splicebert:{out.stem}",
        sys.argv,
        repo=REPO,
        status="EXPLORATORY",
        config=result["config"],
        seeds={"cohort_and_health": args.seed},
        notes=[result["interpretation_boundary"],
               "GTF/FASTA/checkpoint hashes are anchored by data/external/splicebert_resources_v50.sha256.",
               f"model_weight_sha256={sha256_file(REPO / args.model / 'pytorch_model.bin')}"])
    manifest.add_input(REPO / "data/external/splicebert_resources_v50.sha256")
    manifest.add_input(REPO / args.model / "config.json")
    manifest.add_input(REPO / "studies/splicebert_partner_selection/preregistrations/EXPLORATORY_P3_SHIFT_DIAGNOSTIC_V1.md")
    manifest.add_output(out)
    manifest.write(manifest_path)

    print(json.dumps({"gates": gates, "contrasts": contrasts,
                      "location_specific_target_minus_upstream_intron_sham": location_specific,
                      "peak_tracking": tracking}, indent=2))
    print(f"wrote {out}\nwrote {manifest_path}\nruntime={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
