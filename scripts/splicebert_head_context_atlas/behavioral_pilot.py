#!/usr/bin/env python
"""STAGE 1 preflight — behavioral competition gate on DISCOVERY genes (sealed chr8/9 untouched).

Answers, on real data, the three questions the pre-compute gates need:
  * MF1: does the behavioral competition effect M_comp EXIST above the floor (CI excludes 0, |point|>=SESOI)?
         If bounded-null -> INSTRUMENT-LIMITED / no-behavior-to-attribute (terminal), coheres with IC1/IC2.
  * MF2: what is the real SD(M_comp) (example- and gene-level) -> the honest anchor for the power gate.
  * Gate B (positive control): partner-coupling (destroy the shared partner) must be materially positive,
         so a M_comp null is EVIDENCED (instrument works), not a default.

Definitions (target t, shared intron-mate partner p, competitor c; all destroyed with REAL bases):
  M_comp = margin(t | c present) - margin(t | c destroyed)      [competition dependence, either sign]
  B      = margin(t | p present) - margin(t | p destroyed)      [known coupling; positive control]
  sham   = margin(t | present)   - margin(t | matched-distance non-site 2bp flipped)  [generic perturbation]

DISCOVERY split only (train = autosomes ∉ {chr8,9,11,16,19}). No sealed data, no head interventions.
Deterministic (PYTHONHASHSEED=0 + sha256-seeded sampling). Writes results/.../behavioral_pilot_v1.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.gtf import load_transcripts  # noqa: E402
from gfmcd.splicing.competition import (  # noqa: E402
    CompetitionConfig, build_competition_index, enumerate_competition_examples)
from gfmcd.splicing.competition_seq import build_window  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    destroy_donor, destroy_acceptor, destroy_donor_clean, destroy_acceptor_clean,
    perturb_random,
)

USE_CLEAN_ALT = False   # R44 --clean-alt; default preserves committed behaviour
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402

DEFAULT_GTF: str | None = None  # supply via --gtf
DEFAULT_FASTA: str | None = None  # supply via --fasta
DEFAULT_MODEL = "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
STD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def _order_key(eid: str) -> int:
    return int.from_bytes(hashlib.sha256(eid.encode()).digest()[:8], "big")


def _destroy(seq, pos, is_donor):
    if USE_CLEAN_ALT:
        # single motif-clean construct replaces the contaminated 4-alt mean (R44)
        return [(destroy_donor_clean if is_donor else destroy_acceptor_clean)(seq, pos)]
    return destroy_donor(seq, pos) if is_donor else destroy_acceptor(seq, pos)


def gene_clustered_boot(pairs, sesoi, n_boot=4000, seed=0):
    """pairs: list[(gene, value)]. Returns dict with mean, gene-clustered 95% CI, SDs, CI class."""
    vals = np.array([v for _, v in pairs], float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3:
        return {"n": int(len(vals)), "mean": float("nan"), "ci": [None, None], "class": "inconclusive"}
    by_gene = defaultdict(list)
    for g, v in pairs:
        if not math.isnan(v):
            by_gene[g].append(v)
    genes = sorted(by_gene)
    gene_means = np.array([np.mean(by_gene[g]) for g in genes])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    G = len(genes)
    for b in range(n_boot):
        idx = rng.integers(0, G, G)
        boot[b] = np.mean([np.mean(by_gene[genes[i]]) for i in idx])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    point = float(np.mean(gene_means))
    excl0 = (lo > 0) or (hi < 0)
    within = (lo > -sesoi) and (hi < sesoi)
    if excl0 and abs(point) >= sesoi:
        cls = "material"
    elif excl0:
        cls = "small-but-detectable"
    elif within:
        cls = "bounded-null"
    else:
        cls = "inconclusive"
    return {"n": int(len(vals)), "n_genes": G, "mean": point, "ci": [lo, hi],
            "sd_example": float(np.std(vals, ddof=1)), "sd_gene": float(np.std(gene_means, ddof=1)),
            "class": cls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=300, help="target successful windows")
    ap.add_argument("--per-gene-cap", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sesoi", type=float, default=0.05)
    ap.add_argument("--clean-alt", action="store_true", help="R44: destroy with MOTIF_CLEAN_ALT")
    ap.add_argument("--out", default=str(REPO / "results/splicebert_head_context_atlas/behavioral_pilot_v1.json"))
    args = ap.parse_args()
    global USE_CLEAN_ALT
    USE_CLEAN_ALT = bool(args.clean_alt)
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("WARN: PYTHONHASHSEED != 0", file=sys.stderr)

    cfg = CompetitionConfig()
    print("[pilot] loading transcripts + building competition index ...", file=sys.stderr)
    txs = load_transcripts(args.gtf, transcript_types=("protein_coding",), allowed_chromosomes=STD_CHROMS)
    idx = build_competition_index(txs)
    exs = [e for e in enumerate_competition_examples(idx, cfg) if e.split == "train" and e.usable]
    exs.sort(key=lambda e: _order_key(e.example_id))  # deterministic, label-independent order
    print(f"[pilot] {len(exs)} usable discovery examples; building windows to reach n={args.n}", file=sys.stderr)

    genome = PyfaidxGenome(args.fasta)
    windows, per_gene = [], defaultdict(int)
    for e in exs:
        if len([w for w in windows if w.site_type == e.site_type]) >= args.n // 2 + args.n % 2 \
           and len(windows) >= args.n:
            break
        if per_gene[e.gene_id] >= args.per_gene_cap:
            continue
        w = build_window(e, genome, cfg)
        if w is None:
            continue
        windows.append(w)
        per_gene[e.gene_id] += 1
        if len(windows) >= args.n:
            break
    print(f"[pilot] built {len(windows)} windows across {len(per_gene)} genes", file=sys.stderr)

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from gfmcd.splicing.coupling import CouplingScorer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    tok = AutoTokenizer.from_pretrained(mp)
    mdl = AutoModelForMaskedLM.from_pretrained(mp).to(dev)
    scorer = CouplingScorer(mdl, tok, device=dev, batch_size=args.batch)
    print(f"[pilot] scoring on {dev} ...", file=sys.stderr)

    def score(items):
        """items: list[(seq, target_win, is_donor)] -> np.array of margins, order preserved."""
        out = np.empty(len(items))
        for is_don in (True, False):
            sel = [(i, s, p) for i, (s, p, d) in enumerate(items) if d == is_don]
            if not sel:
                continue
            seqs = [s for _, s, _ in sel]
            pos = [p for _, _, p in sel]
            m = scorer.donor_margin(seqs, pos) if is_don else scorer.acceptor_margin(seqs, pos)
            for (i, _, _), v in zip(sel, m):
                out[i] = v
        return out

    present = score([(w.seq, w.target_win, w.target_is_donor) for w in windows])

    def destroyed_mean(site_attr, is_donor_attr):
        flat, groups = [], []
        for w in windows:
            pos = getattr(w, site_attr)
            variants = _destroy(w.seq, pos, getattr(w, is_donor_attr))
            start = len(flat)
            flat += [(v, w.target_win, w.target_is_donor) for v in variants]
            groups.append((start, len(flat)))
        m = score(flat)
        return np.array([m[a:b].mean() for a, b in groups])

    comp_destroyed = destroyed_mean("competitor_win", "competitor_is_donor")
    partner_destroyed = destroyed_mean("partner_win", "partner_is_donor")

    # sham: mirror competitor across target (matched distance, opposite side), 2bp real flip at a non-site
    sham_items, sham_valid = [], []
    for w in windows:
        sp = 2 * w.target_win - w.competitor_win
        ok = (4 <= sp < len(w.seq) - 4 and abs(sp - w.target_win) >= 3 and abs(sp - w.partner_win) >= 3
              and w.seq[sp:sp + 2] not in ("GT", "GC") and w.seq[sp - 2:sp] != "AG")
        sham_valid.append(ok)
        sham_items.append((perturb_random(w.seq, sp, 2) if ok else w.seq, w.target_win, w.target_is_donor))
    sham_scored = score(sham_items)

    M_comp = present - comp_destroyed
    B = present - partner_destroyed
    sham = np.where(sham_valid, present - sham_scored, np.nan)

    def by_type(arr, st=None):
        return [(w.gene_id, arr[i]) for i, w in enumerate(windows) if st is None or w.site_type == st]

    report = {}
    for st in (None, "alt5ss_donor", "alt3ss_acceptor"):
        key = st or "BOTH"
        report[key] = {
            "M_comp": gene_clustered_boot(by_type(M_comp, st), args.sesoi),
            "M_comp_abs": gene_clustered_boot([(g, abs(v)) for g, v in by_type(M_comp, st)], args.sesoi),
            "B_partner_coupling": gene_clustered_boot(by_type(B, st), args.sesoi),
            "sham": gene_clustered_boot(by_type(sham, st), args.sesoi),
            "M_comp_minus_sham": gene_clustered_boot(
                [(w.gene_id, M_comp[i] - sham[i]) for i, w in enumerate(windows)
                 if (st is None or w.site_type == st) and not np.isnan(sham[i])], args.sesoi),
        }

    # projected per-head + group MDE using the REAL discovery SD of M_comp
    both = report["BOTH"]["M_comp"]
    sd_ex, sd_ge = both.get("sd_example"), both.get("sd_gene")
    n_sealed_genes, n_sealed_ex = 444, 3014  # from census_v1 (BOTH usable, chr8/9)
    def mde(sd, n, k=96):
        z = 3.48 + 0.84  # ~Bonferroni(96) two-sided .05 + power .8 upper bound
        return z * sd / math.sqrt(n) if (sd and n) else None
    power = {
        "sd_example_Mcomp": sd_ex, "sd_gene_Mcomp": sd_ge,
        "note": "E_ih (head ablation of M_comp) has SD >= SD(M_comp); this is a LOWER bound on E_ih MDE",
        "MDE_perhead_sealed_genes_Mcompsd": mde(sd_ge, n_sealed_genes),
        "MDE_perhead_sealed_examples_Mcompsd": mde(sd_ex, n_sealed_ex),
        "sesoi_c": args.sesoi,
    }

    out = {"schema": "head_context_atlas.behavioral_pilot/1",
           "split": "discovery (train, ∉ chr8/9/11/16/19)", "n_windows": len(windows),
           "n_genes": len(per_gene), "sesoi_c_candidate": args.sesoi,
           "report": report, "power_projection": power,
           "interpretation_rules": {
               "MF1_gate": "M_comp material or small-but-detectable => behavior exists; bounded-null/inconclusive => INSTRUMENT-LIMITED",
               "B_gate": "B must be material (positive) or the instrument is not eliciting coupling"}}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        raise FileExistsError(f"refuse to overwrite {outp}")
    outp.write_text(json.dumps(out, indent=2, sort_keys=True))
    try:
        from gfmcd.provenance import ExperimentManifest
        m = ExperimentManifest.create(experiment_id="head_context_atlas_behavioral_pilot_v1",
                                      command=["python"] + sys.argv, status="EXPLORATORY")
        m.add_input(args.gtf); m.add_input(args.fasta); m.add_output(str(outp))
        m.write(str(outp.with_suffix(".manifest.json")))
    except Exception as e:
        print(f"WARN manifest: {e}", file=sys.stderr)

    print("\n================ BEHAVIORAL COMPETITION PILOT (discovery) ================")
    for key in ("BOTH", "alt5ss_donor", "alt3ss_acceptor"):
        r = report[key]
        print(f"\n[{key}]  n={r['M_comp']['n']} genes={r['M_comp'].get('n_genes')}")
        for name in ("B_partner_coupling", "M_comp", "M_comp_abs", "sham", "M_comp_minus_sham"):
            s = r[name]
            ci = s["ci"]
            cistr = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci[0] is not None else "[--]"
            print(f"    {name:20s} mean={s['mean']:+.4f} CI95={cistr}  class={s['class']}"
                  + (f"  sd_ex={s.get('sd_example',float('nan')):.3f} sd_gene={s.get('sd_gene',float('nan')):.3f}"
                     if 'sd_example' in s else ""))
    print("\n---- projected MDE from REAL discovery SD(M_comp) (LOWER bound on E_ih MDE) ----")
    print(f"    SD(M_comp) example={sd_ex} gene={sd_ge}")
    print(f"    MDE/head sealed genes={power['MDE_perhead_sealed_genes_Mcompsd']}  "
          f"examples={power['MDE_perhead_sealed_examples_Mcompsd']}  vs SESOI-C {args.sesoi}")
    print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
