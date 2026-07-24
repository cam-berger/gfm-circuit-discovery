#!/usr/bin/env python
"""Per-example covariate join for the head-screen matrix (NO GPU / NO model).

Recomputes, for every example_id present in head_effect_matrix.jsonl, the ontology covariates that the
screen did not export: MaxEnt strength (target/partner/competitor), local+window composition, coding
context (CDS/exon membership from the GTF), and packet geometry (within-window positions, separation,
period-3 phase). Distance/topology/cluster_size are already in the matrix.

Determinism: the cohort is regenerated with the SAME loader/config as the run, then intersected with the
matrix's example_ids (the memorial: every matrix example must be reproducible). Run with `uv run python`
and PYTHONHASHSEED=0 (historical runs used the conda env gfm-thrust1).

Out: example_covariates.jsonl (one row per example_id), joinable to the matrix by example_id.
"""
from __future__ import annotations

import argparse, gzip, json, math, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.gtf import load_transcripts
from gfmcd.splicing.competition import CompetitionConfig, build_competition_index, enumerate_competition_examples
from gfmcd.splicing.competition_seq import build_window
from gfmcd.splicing.specificity import don_score, acc_score
from gfmcd.splicing.dataset import PyfaidxGenome

SEALED = {"chr8", "chr9"}
VAL = {"chr11", "chr16", "chr19"}


def split_for(chrom):
    return "val" if chrom in VAL else ("test" if chrom in SEALED else "train")


def gc(s):
    return (s.count("G") + s.count("C")) / len(s) if s else float("nan")


def strength(seq, pos, is_donor, m5, m3):
    return don_score(seq, pos, m5) if is_donor else acc_score(seq, pos, m3)


def parse_gtf_intervals(gtf_path, keep_genes):
    """gene_id -> {'CDS': [(s,e)...], 'exon': [(s,e)...]}  (1-based inclusive, from the GTF)."""
    iv = defaultdict(lambda: {"CDS": [], "exon": []})
    op = gzip.open if str(gtf_path).endswith(".gz") else open
    with op(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t")
            if len(p) < 9 or p[2] not in ("CDS", "exon"):
                continue
            attrs = p[8]
            k = attrs.find('gene_id "')
            if k < 0:
                continue
            gid = attrs[k + 9: attrs.find('"', k + 9)]
            if gid not in keep_genes:
                continue
            iv[gid][p[2]].append((int(p[3]), int(p[4])))
    return iv


def in_intervals(pos, ivs):
    return any(s - 1 <= pos <= e for s, e in ivs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(Path.home() / "Downloads/head_effect_matrix.jsonl"))
    ap.add_argument("--config", default=str(REPO / "studies/splicebert_head_context_atlas/colab_package/config.discovery.json"))
    ap.add_argument("--out", default=str(Path.home() / "Downloads/example_covariates.jsonl"))
    ap.add_argument("--min-separation", type=int, default=30)
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    # 1. target example_ids from the matrix
    want = set()
    with open(args.matrix) as f:
        for line in f:
            want.add(json.loads(line)["example_id"])
    print(f"[cov] matrix has {len(want)} distinct example_ids", file=sys.stderr)

    # 2. regenerate the cohort (same loader/config as the run) and index by example_id
    from maxentpy.maxent import load_matrix5, load_matrix3
    m5, m3 = load_matrix5(), load_matrix3()
    ccfg = CompetitionConfig(min_separation=args.min_separation)
    txs = load_transcripts(cfg["gtf"], transcript_types=("protein_coding",),
                           allowed_chromosomes={f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"})
    idx = build_competition_index(txs)
    by_id = {e.example_id: e for e in enumerate_competition_examples(idx, ccfg) if e.usable}
    print(f"[cov] regenerated {len(by_id)} usable examples", file=sys.stderr)

    genome = PyfaidxGenome(cfg["fasta"])
    genes = {by_id[i].gene_id for i in want if i in by_id}
    print(f"[cov] parsing GTF CDS/exon for {len(genes)} genes ...", file=sys.stderr)
    iv = parse_gtf_intervals(cfg["gtf"], genes)

    # 3. covariates per example
    missing, built = [], 0
    with open(args.out, "w") as out:
        for eid in sorted(want):
            ex = by_id.get(eid)
            if ex is None:
                missing.append(eid); continue
            w = build_window(ex, genome, ccfg)
            if w is None:
                missing.append(eid); continue
            seq = w.seq; N = len(seq)
            t_str = strength(seq, w.target_win, w.target_is_donor, m5, m3)
            p_str = strength(seq, w.partner_win, w.partner_is_donor, m5, m3)
            c_str = strength(seq, w.competitor_win, w.competitor_is_donor, m5, m3)
            gcds = iv.get(ex.gene_id, {"CDS": [], "exon": []})
            in_cds = in_intervals(ex.target, gcds["CDS"])
            in_exon = in_intervals(ex.target, gcds["exon"])
            region = "CDS" if in_cds else ("UTR_exon" if in_exon else "intron_or_other")
            comp_between = (min(w.target_win, w.partner_win) < w.competitor_win < max(w.target_win, w.partner_win))
            row = {
                "example_id": eid, "gene_id": ex.gene_id, "chrom": w.chrom, "strand": w.strand,
                "split": split_for(w.chrom), "site_type": w.site_type,
                # strength
                "target_strength": t_str, "partner_strength": p_str, "competitor_strength": c_str,
                "target_minus_competitor_strength": (t_str - c_str) if not (math.isnan(t_str) or math.isnan(c_str)) else None,
                # composition
                "gc_window": gc(seq), "gc_target_local50": gc(seq[max(0, w.target_win - 25):w.target_win + 25]),
                "gc_competitor_local50": gc(seq[max(0, w.competitor_win - 25):w.competitor_win + 25]),
                # coding context
                "target_in_cds": in_cds, "target_in_exon": in_exon, "coding_region": region,
                # packet geometry
                "target_win": w.target_win, "partner_win": w.partner_win, "competitor_win": w.competitor_win,
                "target_center_offset": w.target_win - N // 2,
                "target_competitor_distance": w.target_competitor_distance,
                "target_partner_distance": w.target_partner_distance,
                "phase_tc_mod3": w.target_competitor_distance % 3,
                "competitor_between_target_partner": comp_between,
                "cluster_size": w.cluster_size, "n_usable_competitors": w.n_usable_competitors,
                "target_is_donor": w.target_is_donor, "competitor_is_donor": w.competitor_is_donor,
            }
            out.write(json.dumps(row, sort_keys=True) + "\n")
            built += 1

    print(f"[cov] wrote {built} rows -> {args.out}")
    print(f"[MEMORIAL] example_id match: built {built}/{len(want)} ; missing {len(missing)}")
    if missing:
        print("  first missing:", missing[:5])


if __name__ == "__main__":
    main()
