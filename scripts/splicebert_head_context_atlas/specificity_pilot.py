#!/usr/bin/env python
"""STAGE 1b — the DECISIVE competition-specificity test (MF6), on DISCOVERY genes.

The behavioral pilot showed |M_comp| is material (~0.11 nats) but not distinguishable from a crude sham.
This test asks the R22-safe question with the REAL control:

    Does destroying the true COMPETITOR move the target margin MORE than destroying a real,
    MaxEnt-strength-and-distance-matched NON-competitor site of the same type?

  M_comp  = margin(t | competitor present) - margin(t | competitor destroyed)     [real bases]
  M_decoy = margin(t | decoy present)      - margin(t | matched-decoy destroyed)   [real bases]
  specificity = |M_comp| - |M_decoy|   (paired, gene-clustered)

specificity material(+) => competition-specific behavior exists; bounded-null/inconclusive => the
competition magnitude effect is generic local perturbation => INSTRUMENT-LIMITED (coheres with IC1/IC2).

Discovery split only. No sealed data, no head interventions. Deterministic.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.gtf import load_transcripts
from gfmcd.splicing.competition import CompetitionConfig, build_competition_index, enumerate_competition_examples
from gfmcd.splicing.competition_seq import build_window
from gfmcd.splicing.coupling import (destroy_donor, destroy_acceptor,
                                     destroy_donor_clean, destroy_acceptor_clean)

USE_CLEAN_ALT = False   # R44 --clean-alt; default preserves committed behaviour
from gfmcd.splicing.dataset import PyfaidxGenome
from gfmcd.splicing.specificity import find_matched_decoy  # shared: MaxEnt-matched decoy selector

DEFAULT_GTF: str | None = None  # supply via --gtf
DEFAULT_FASTA: str | None = None  # supply via --fasta
DEFAULT_MODEL = "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
STD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def gene_boot(pairs, sesoi, n_boot=4000, seed=0):
    vals = np.array([v for _, v in pairs if not math.isnan(v)], float)
    if len(vals) < 3:
        return {"n": int(len(vals)), "mean": float("nan"), "ci": [None, None], "class": "inconclusive"}
    by = defaultdict(list)
    for g, v in pairs:
        if not math.isnan(v):
            by[g].append(v)
    genes = sorted(by); gm = np.array([np.mean(by[g]) for g in genes])
    rng = np.random.default_rng(seed); G = len(genes); b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, G, G); b[i] = np.mean([np.mean(by[genes[j]]) for j in idx])
    lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)); pt = float(np.mean(gm))
    excl0 = lo > 0 or hi < 0; within = lo > -sesoi and hi < sesoi
    cls = ("material" if excl0 and abs(pt) >= sesoi else "small-but-detectable" if excl0
           else "bounded-null" if within else "inconclusive")
    return {"n": int(len(vals)), "n_genes": G, "mean": pt, "ci": [lo, hi],
            "sd_example": float(np.std(vals, ddof=1)), "class": cls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--per-gene-cap", type=int, default=3); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sesoi", type=float, default=0.05)
    ap.add_argument("--clean-alt", action="store_true", help="R44: destroy with MOTIF_CLEAN_ALT")
    ap.add_argument("--out", default=str(REPO / "results/splicebert_head_context_atlas/specificity_pilot_v1.json"))
    args = ap.parse_args()
    global USE_CLEAN_ALT
    USE_CLEAN_ALT = bool(args.clean_alt)
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("WARN PYTHONHASHSEED!=0", file=sys.stderr)

    from maxentpy.maxent import load_matrix5, load_matrix3
    m5, m3 = load_matrix5(), load_matrix3()
    cfg = CompetitionConfig()
    print("[spec] index ...", file=sys.stderr)
    txs = load_transcripts(args.gtf, transcript_types=("protein_coding",), allowed_chromosomes=STD_CHROMS)
    idx = build_competition_index(txs)
    exs = [e for e in enumerate_competition_examples(idx, cfg) if e.split == "train" and e.usable]
    exs.sort(key=lambda e: int.from_bytes(hashlib.sha256(e.example_id.encode()).digest()[:8], "big"))
    genome = PyfaidxGenome(args.fasta)

    windows, decoys, strengths, per_gene, n_no_decoy = [], [], [], defaultdict(int), 0
    for e in exs:
        if len(windows) >= args.n:
            break
        if per_gene[e.gene_id] >= args.per_gene_cap:
            continue
        w = build_window(e, genome, cfg)
        if w is None:
            continue
        dq, cs = find_matched_decoy(w, m5, m3)
        if dq is None:
            n_no_decoy += 1
            continue
        windows.append(w); decoys.append(dq); strengths.append(cs); per_gene[e.gene_id] += 1
    print(f"[spec] {len(windows)} windows w/ matched decoy ({n_no_decoy} lacked one) across {len(per_gene)} genes",
          file=sys.stderr)

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from gfmcd.splicing.coupling import CouplingScorer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    tok = AutoTokenizer.from_pretrained(mp); mdl = AutoModelForMaskedLM.from_pretrained(mp).to(dev)
    scorer = CouplingScorer(mdl, tok, device=dev, batch_size=args.batch)
    print(f"[spec] scoring on {dev} ...", file=sys.stderr)

    def score(items):
        out = np.empty(len(items))
        for isd in (True, False):
            sel = [(i, s, p) for i, (s, p, d) in enumerate(items) if d == isd]
            if not sel: continue
            m = (scorer.donor_margin if isd else scorer.acceptor_margin)([s for _, s, _ in sel], [p for _, _, p in sel])
            for (i, _, _), v in zip(sel, m): out[i] = v
        return out

    present = score([(w.seq, w.target_win, w.target_is_donor) for w in windows])

    def destroyed_mean(pos_of, is_donor_of):
        flat, groups = [], []
        for k, w in enumerate(windows):
            pos = pos_of(k, w); isd = is_donor_of(w)
            if USE_CLEAN_ALT:
                variants = [(destroy_donor_clean if isd else destroy_acceptor_clean)(w.seq, pos)]
            else:
                variants = destroy_donor(w.seq, pos) if isd else destroy_acceptor(w.seq, pos)
            a = len(flat); flat += [(v, w.target_win, w.target_is_donor) for v in variants]; groups.append((a, len(flat)))
        m = score(flat); return np.array([m[a:b].mean() for a, b in groups])

    comp_destroyed = destroyed_mean(lambda k, w: w.competitor_win, lambda w: w.competitor_is_donor)
    decoy_destroyed = destroyed_mean(lambda k, w: decoys[k], lambda w: w.competitor_is_donor)
    M_comp = present - comp_destroyed
    M_decoy = present - decoy_destroyed
    spec = np.abs(M_comp) - np.abs(M_decoy)

    def rows(arr, st=None):
        return [(w.gene_id, arr[i]) for i, w in enumerate(windows) if st is None or w.site_type == st]

    report = {}
    for st in (None, "alt5ss_donor", "alt3ss_acceptor"):
        key = st or "BOTH"
        report[key] = {
            "abs_M_comp": gene_boot([(g, abs(v)) for g, v in rows(M_comp, st)], args.sesoi),
            "abs_M_decoy": gene_boot([(g, abs(v)) for g, v in rows(M_decoy, st)], args.sesoi),
            "specificity_absMcomp_minus_absMdecoy": gene_boot(rows(spec, st), args.sesoi),
        }
    out = {"schema": "head_context_atlas.specificity_pilot/1", "split": "discovery",
           "n_windows": len(windows), "n_genes": len(per_gene), "n_lacked_decoy": n_no_decoy,
           "sesoi_c": args.sesoi, "report": report,
           "gate": "specificity material(+) => competition-specific; else INSTRUMENT-LIMITED (generic perturbation)"}
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists(): raise FileExistsError(f"refuse to overwrite {outp}")
    outp.write_text(json.dumps(out, indent=2, sort_keys=True))
    try:
        from gfmcd.provenance import ExperimentManifest
        m = ExperimentManifest.create(experiment_id="head_context_atlas_specificity_pilot_v1",
                                      command=["python"] + sys.argv, status="EXPLORATORY")
        m.add_input(args.gtf); m.add_input(args.fasta); m.add_output(str(outp))
        m.write(str(outp.with_suffix(".manifest.json")))
    except Exception as e:
        print(f"WARN manifest: {e}", file=sys.stderr)

    print("\n============= COMPETITION-SPECIFICITY TEST (discovery, real matched decoy) =============")
    for key in ("BOTH", "alt5ss_donor", "alt3ss_acceptor"):
        r = report[key]; print(f"\n[{key}] n={r['abs_M_comp']['n']} genes={r['abs_M_comp'].get('n_genes')}")
        for name in ("abs_M_comp", "abs_M_decoy", "specificity_absMcomp_minus_absMdecoy"):
            s = r[name]; ci = s["ci"]
            cistr = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci[0] is not None else "[--]"
            print(f"    {name:38s} mean={s['mean']:+.4f} CI95={cistr} class={s['class']}")
    print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
