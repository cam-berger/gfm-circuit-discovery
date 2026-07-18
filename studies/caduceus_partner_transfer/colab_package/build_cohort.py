"""LOCAL cohort + construct builder (no model; run before packaging so Colab only does forward passes).

Reuses the proven SpliceBERT eligibility machinery (gfmcd.splicing.circuit.eligible/_dataset) to emit the
Caduceus T1 tagged construct set as JSONL. Every construct is a FIXED 1024 nt sequence + a readout position;
the GPU harness (caduceus_t1.py) batches them equal-length and reads donor margins.

Tags per example: clean, partner_cc, donor_dead, nonpartner_cc, clean_other, partner_cc_other, lr_block@<d>.
Controls: nonpartner = distance-matched SAME-SIDE (downstream) real non-partner AG from `other_ag`, with the
partner `next_acceptor_pos` EXCLUDED (B-3) and a per-example common-support caliper (A3 Berkson: drop + report).
other-donor = distance-matched real `other_gt`. Fresh Caduceus discovery/confirm CHROMOSOME split.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import eligible, _dataset, DEFAULT_GTF, DEFAULT_FASTA  # noqa: E402
from gfmcd.splicing.coupling import substitute                                     # noqa: E402

# fresh Caduceus split (SpliceBERT's chr8/9 seal does NOT bind Caduceus); >=3-4 confirmatory chromosomes
DISCOVERY = {"chr2", "chr3", "chr4", "chr5", "chr6", "chr7"}
CONFIRM = {"chr8", "chr9", "chr11", "chr16"}
LR_OFFSETS = (10, 30, 60, 100, 150, 200)


def scramble_window(seq, start, width, rng):
    w = list(seq[start:start + width]); rng.shuffle(w)
    return seq[:start] + "".join(w) + seq[start + width:]


def build(args):
    import numpy as np
    rng = np.random.default_rng(0)
    ds = _dataset(args.gtf, args.fasta)
    tr = []                                                   # ALL eligible exons; Caduceus re-splits by chrom
    seen = set()
    for sp in ("train", "val", "test"):
        for x in eligible(ds, sp):
            if x.example_id not in seen:
                seen.add(x.example_id); tr.append(x)
    out = []
    kept = dropped = 0
    for x in tr:
        chrom = x.chrom
        split = "discovery" if chrom in DISCOVERY else ("confirm" if chrom in CONFIRM else None)
        if split is None:
            continue
        seq, d, a = x.sequence, x.donor_pos, x.next_acceptor_pos
        if a is None or len(seq) != args.window:
            continue
        d_part = a - d                                        # signed downstream distance to partner
        # --- distance-matched SAME-SIDE non-partner AG (exclude partner; caliper) ---
        cand = sorted([p for p in x.other_ag if p != a and p > d and abs((p - d) - d_part) <= args.caliper],
                      key=lambda p: abs((p - d) - d_part))
        if not cand:
            dropped += 1; continue                            # common-support fail (report retention)
        npg = cand[0]
        npg2 = cand[1] if len(cand) > 1 else None             # 2nd non-partner => GENUINE permuted-partner null (both non-partners)
        # --- distance-matched other real donor GT ---
        gts = [p for p in x.other_gt if p != d]
        if not gts:
            dropped += 1; continue
        d2 = min(gts, key=lambda p: abs(abs(p - d) - d_part))
        kept += 1
        partner_cc = substitute(seq, a - 2, "CC")             # partner AG -> CC
        strand = getattr(x, "strand", "+")
        rec = {"ex_id": x.example_id, "gene": x.gene_id, "chrom": chrom, "split": split, "strand": strand,
               "d": d, "a": a, "npg": npg, "npg2": npg2, "d2": d2, "d_part": int(d_part),
               "constructs": {
                   "clean": [seq, d],
                   "partner_cc": [partner_cc, d],
                   "donor_dead": [substitute(seq, d, "AA"), d],       # GT -> AA
                   "nonpartner_cc": [substitute(seq, npg - 2, "CC"), d],
                   "clean_other": [seq, d2],
                   "partner_cc_other": [partner_cc, d2],
               }}
        if npg2 is not None:
            rec["constructs"]["nonpartner2_cc"] = [substitute(seq, npg2 - 2, "CC"), d]
        for off in LR_OFFSETS:
            s = d + off
            if 0 <= s and s + 15 < len(seq):
                rec["constructs"][f"lr_block@{off}"] = [scramble_window(seq, s, 15, rng), d]
        out.append(rec)
        if args.limit and kept >= args.limit:
            break
    retention = kept / (kept + dropped) if (kept + dropped) else 0.0
    minus = sum(r.get("strand") == "-" for r in out)
    minus_share = minus / len(out) if out else 0.0
    meta = {"n_examples": len(out), "kept": kept, "dropped_no_control": dropped,
            "retention_fraction": round(retention, 3), "caliper": args.caliper,
            "minus_strand_share": round(minus_share, 3),
            "both_strands_ok": bool(0.3 <= minus_share <= 0.7),
            "n_with_permuted_null_npg2": sum("nonpartner2_cc" in r["constructs"] for r in out),
            "discovery_chroms": sorted(DISCOVERY), "confirm_chroms": sorted(CONFIRM),
            "lr_offsets": list(LR_OFFSETS), "window": args.window,
            "n_discovery": sum(r["split"] == "discovery" for r in out),
            "n_confirm": sum(r["split"] == "confirm" for r in out),
            "n_genes_confirm": len({r["gene"] for r in out if r["split"] == "confirm"})}
    Path(args.out).write_text("\n".join(json.dumps(r) for r in out))
    Path(args.meta).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"wrote {len(out)} examples -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--caliper", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="cohort_constructs.jsonl")
    ap.add_argument("--meta", default="cohort_meta.json")
    build(ap.parse_args())
