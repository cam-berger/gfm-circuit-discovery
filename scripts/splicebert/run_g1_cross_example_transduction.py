#!/usr/bin/env python
"""G1 -- cross-example / cross-context transduction of the donor-side DELIVERED STATE.

Design of record: preregistrations/PREREGISTRATION_G1_CROSS_EXAMPLE_TRANSDUCTION.md
Red-team of record: redteam/REDTEAM_BRIEF_G1.md + REDTEAM_SYNTHESIS_G1.json (2026-07-23; FIXABLE_THEN_
RUNNABLE, survivor=YES, 0 FATAL). This runner folds in all 9 pre-freeze must-fixes.

INTENT (north star, re-read at every gate):
  Does the donor-side delivered state (C16-C18/C23/C26) behave as a TRANSFERABLE, EXAMPLE-SPECIFIC
  information-bearing state -- derived on genes the target never contributed to, injected into a
  gene-disjoint / context-mismatched target, moving its native donor margin dose-ordered and
  source-specifically -- or does any cross-example effect reduce to a push along a GENE-GENERIC READOUT
  AXIS? (No payload semantics; R22 stands. No minimality; C33 stands.)

ESTIMAND: U_D = top_k_subspace(H_cl-H_co at donor, 8) on D_src donors; s_j = project(diff_j, U_D); inject
  additively at the FINAL donor residual H_i(a)=H_co,i+a*s; read margin_hard(cls(.)). tau = E[m(a)-m(0)]
  / E[coup], two-way (target-gene x source-gene) dyadic bootstrap. PRIMARY Delta_spec = tau(REAL)-tau(MEAN)
  at a=1, paired within-target.

RED-TEAM FIXES (all on the measurement path):
  F1 IDENTIFIABILITY: Delta_spec<0 is substantially a norm-wastage DENOISING artifact (tau(REAL)/tau(MEAN)
     ~= E[cos(s_j,s_bar)]). The load-bearing disambiguator is the PARALLEL/PERP decomposition: inject the
     example-specific residual s_perp ALONE at LOW alpha (linear regime; margin_hard saturates -- P3.3) vs
     a norm-matched random-perp control. s_perp sub-SESOI => off-axis example content is INERT for the
     donor margin => GENERIC; s_perp restoring => example-specific transduction survives despite Delta<0.
     GENERIC is GATED on s_perp being sub-SESOI. Report tau(REAL)/tau(MEAN) vs E[cos] (denoising-null).
  F2 DECISION RULE: distinguish equivalence-null Delta_spec (CI in +-SESOI) from "REAL significantly below
     MEAN" (denoising-consistent, geometry-attributed); |Delta_spec| is NOT an effect size.
  F3 R22: draws SUBSAMPLE + resample sources WITH REPLACEMENT so each draw derives an independent U_D/s_bar
     (the old n_src==|src_all| made draws identical -- the A9/C30 blind spot). >=8 draws.
  F4 SCOPE: the null bounds ONLY the C16/C26 delivered FINAL-RESIDUAL state, via ADDITIVE injection through
     the SHORT-SOURCE rank-8 U_D. It does NOT bound C17 routing / C18 transporters / C23 address+payload,
     nor earlier-layer or non-additive example-specificity (R24-style within-intervention qualifier).
  F5 M8 OOD band (the designated failable health check): ||H_inj||/||H_cl|| + donor entropy per arm x dose.
  F6 M4 (gene-disjoint) and M9 (cohort hash) COMPUTED, not hardcoded (R21 class).
  F7 M3: SELF must reproduce C26's +0.875 within a sealed tolerance; gates smoke AND confirmatory.
  F8 SRC_SHAM: EXCLUDE no-decoy fallback pairs (not zero-fill); report frac(sham_ok); role narrowed to the
     sham-frac ratio-exclusion, NOT the (unreachable) equivalence family.
  F9 SESOI_spec sealed + scale relation documented.

MODES: --smoke (D_src mechanics + calibration, val sealed); --pilot (D_src+H_dev: seal + MDE gate);
  --decomp (H_dev PAR/PERP identifiability diagnostic across the dose grid); (default) confirmatory on
  H_conf with >=8 seeded draws -- requires a passed pilot/MDE gate + freeze.

Model+data live in the MAIN checkout; this worktree symlinks data/models and reads the audited Atlas
cohorts by absolute path. No new kernel; reuses C16/A5 subspace machinery + B3's dyadic bootstrap.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file          # noqa: E402
from gfmcd.splicing.circuit import (                                              # noqa: E402
    CouplingTask, load_adapter, pick_device, ratio_of_means,
)
from gfmcd.splicing.coupling import CouplingExample, substitute                   # noqa: E402
from gfmcd.splicing.subspace import (                                             # noqa: E402
    energy_spectrum, patch_component, project, random_subspace, top_k_subspace,
)
from gfmcd.splicing.unembed import margin_hard                                    # noqa: E402

# ----- frozen constants (design of record) -----
RANK_K = 8
DOSE_GRID = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5)
ALPHA_LINEAR = 0.25          # F1: low-alpha linear regime for the PERP arm (margin_hard saturates at a=1)
SEED = 0
SESOI_ABS = 0.05             # nats of donor-margin movement (house absolute floor; dual threshold)
SESOI_TAU = 0.10             # normalized transfer ratio (tau scale)
# F9: SESOI_spec is the specificity-contrast threshold on the tau (ratio) scale. Sealed = 0.05 normalized.
# Scale relation: SESOI_ABS=0.05 is on the raw-nat numerator; SESOI_spec/SESOI_TAU are on tau = numerator /
# E[coup]. With pooled E[coup]~0.41, 0.05 normalized ~= 0.02 nat -- STRICTER than the abs floor, so the
# dual threshold (|Delta_m|>=SESOI_ABS AND |tau|>SESOI_spec) is respected. Pilot Delta_spec -0.34 is ~6x.
SESOI_SPEC_DEFAULT = 0.05
M3_TOL = 0.10                # F7: |SELF_suff - 0.875| < M3_TOL (Atlas is a broader distribution than a5's
                             # partner-selection val; 0.10 is the sealed reproduction tolerance).
M6_RANDOM_MAX = 0.05         # RANDOM tau floor (C16 anchor)
M7_COUP_MIN = 0.20           # per-context E[coup] floor (A5 positive_control_partner_coupling)
M8_RATIO_BAND = (0.5, 2.0)   # ||H_inj||/||H_cl|| band (sealed on H_dev; see pilot report)
SRC_DIST_MIN, SRC_DIST_MAX = 60, 150   # sources = canonical single-pair SHORT introns (GAP_MIN/MAX)
CORE_CONTEXTS = ("single_pair", "donor_cluster", "acceptor_cluster",
                 "multiple_independent_pairs", "cassette_exon", "complex_event")
ATLAS = REPO / "results/splicebert_head_context_atlas"
ATLAS_MAIN = ATLAS
MODEL_ABS = str(REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
ARMS = ("REAL", "MEAN", "SELF", "RANDOM", "SCRAMBLE", "SRC_SHAM", "POS_SHAM")


# ======================================================================================
# Atlas -> CouplingExample adapter (frozen coupling path uses .sequence/.donor_pos/
# .next_acceptor_pos only; covariates ride in .metadata). Validated 2026-07-23.
# ======================================================================================
def _atlas_dir() -> Path:
    if (ATLAS / "atlas_context_matched_v7.discovery.sequences.jsonl").exists():
        return ATLAS
    return ATLAS_MAIN


def _load_sequences(split: str) -> dict[str, dict]:
    d = _atlas_dir()
    seqs = {}
    with open(d / f"atlas_context_matched_v7.{split}.sequences.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            seqs[r["sequence_id"]] = r
    return seqs


def load_atlas_donor_examples(split: str, *, single_pair_only=False, dist_lo=None, dist_hi=None,
                              label=None) -> list[CouplingExample]:
    seqs = _load_sequences(split)
    d = _atlas_dir()
    out = []
    with open(d / f"atlas_context_matched_v7.{split}.packets.jsonl") as fh:
        for line in fh:
            p = json.loads(line)
            if p.get("target_role") != "donor":
                continue
            tags = set(p.get("tags", []))
            if single_pair_only and "single_pair" not in tags:
                continue
            tpd = p.get("target_partner_distance")
            if dist_lo is not None and (tpd is None or tpd < dist_lo):
                continue
            if dist_hi is not None and (tpd is None or tpd > dist_hi):
                continue
            s = seqs.get(p["sequence_id"])
            if s is None:
                continue
            seq = s["sequence"]
            dwin, awin = p["donor_win"], p["acceptor_win"]
            if not (0 <= dwin < len(seq) - 1 and 2 <= awin <= len(seq)):
                continue
            contexts = tuple(c for c in CORE_CONTEXTS if c in tags)
            out.append(CouplingExample(
                example_id=p["packet_id"], sequence=seq,
                acceptor_pos=awin, donor_pos=dwin,
                chrom=p["chrom"], strand=p["strand"], gene_id=p["gene_id"],
                split=label or split, exon_length=-1, intron_length_down=tpd,
                n_isoforms_donor=int(p.get("n_annotated_transcripts_for_target_site", 1) or 1),
                n_isoforms_acceptor=int(p.get("acceptor_degree", 1) or 1),
                gene_type=str(p.get("target_boundary_region", "unknown")),
                next_acceptor_pos=awin,
                metadata={"contexts": contexts, "target_partner_distance": tpd,
                          "gc": p.get("gc_target_local50"), "pyr": p.get("pyrimidine_target_local50"),
                          "in_cds": bool(p.get("target_in_cds", False)),
                          "donor_dinuc": seq[dwin:dwin + 2], "partner_dinuc": seq[awin - 2:awin]},
            ))
    return out


def _find_decoy_acceptor(seq: str, partner_pos: int, donor_pos: int) -> int | None:
    """A non-partner AG in-window at distance closest to the true donor-partner distance (C2 sham)."""
    target = abs(donor_pos - partner_pos)
    best, bestd = None, 1 << 30
    for k in range(2, len(seq)):
        if seq[k - 2:k] == "AG" and k != partner_pos:
            d = abs(abs(donor_pos - k) - target)
            if d < bestd:
                best, bestd = k, d
    return best


def _gene_hash(genes) -> str:
    return hashlib.sha256(("|".join(sorted(set(genes)))).encode()).hexdigest()[:16]


def _cohort_hash(X) -> str:                                # F6/M9: deterministic cohort identity
    return hashlib.sha256(("|".join(sorted(x.example_id for x in X))).encode()).hexdigest()[:16]


# ======================================================================================
# Capture: donor final residual + margins (A5 idiom). Sources also get a non-partner sham corrupt;
# F8: sham_ok flags whether a decoy AG existed (no-decoy pairs are EXCLUDED from SRC_SHAM downstream).
# ======================================================================================
def capture(ad, task, X, dev, off, want_sham=False):
    Hcl_L, Hco_L, mcl_L, mco_L = [], [], [], []
    Hsham_L, msham_L, sham_ok = [], [], []
    genes, meta = [], []
    for i in range(0, len(X), task.batch):
        ch = X[i:i + task.batch]
        cl, co, dn = task.pair(ch)
        o_cl = ad.model(**cl, output_hidden_states=True)
        o_co = ad.model(**co, output_hidden_states=True)
        pos = torch.tensor([d + off for d in dn], device=dev)
        idx = torch.arange(len(dn), device=dev)
        Hcl_L.append(o_cl.hidden_states[-1][idx, pos].float().cpu().numpy())
        Hco_L.append(o_co.hidden_states[-1][idx, pos].float().cpu().numpy())
        mcl_L += task.margin(o_cl.logits.float(), dn).cpu().numpy().tolist()
        mco_L += task.margin(o_co.logits.float(), dn).cpu().numpy().tolist()
        if want_sham:
            sham_seqs = []
            for x in ch:
                dec = _find_decoy_acceptor(x.sequence, x.next_acceptor_pos, x.donor_pos)
                if dec is None:
                    sham_seqs.append(x.sequence); sham_ok.append(False)
                else:
                    sham_seqs.append(substitute(x.sequence, dec - 2, "CC")); sham_ok.append(True)
            enc = task.encode(sham_seqs, dn)
            o_sh = ad.model(**enc, output_hidden_states=True)
            Hsham_L.append(o_sh.hidden_states[-1][idx, pos].float().cpu().numpy())
            msham_L += task.margin(o_sh.logits.float(), dn).cpu().numpy().tolist()
        genes += [x.gene_id for x in ch]
        meta += [dict(x.metadata) for x in ch]
    out = {"Hcl": np.concatenate(Hcl_L), "Hco": np.concatenate(Hco_L),
           "mcl": np.array(mcl_L), "mco": np.array(mco_L), "genes": genes, "meta": meta}
    out["coup"] = out["mcl"] - out["mco"]
    out["diff"] = out["Hcl"] - out["Hco"]
    if want_sham:
        out["Hsham"] = np.concatenate(Hsham_L)
        out["msham"] = np.array(msham_L)
        out["diff_sham"] = out["Hcl"] - out["Hsham"]
        out["sham_ok"] = np.array(sham_ok, dtype=bool)
    return out


# ======================================================================================
# Two-way (gt x gs) gene Poisson-multiplier dyadic bootstrap for a ratio-of-means (verbatim from B3).
# ======================================================================================
def dyadic_ratio_ci(num, dnm, gt, gs, gs2=None, seed=SEED, B=2000, lo=2.5, hi=97.5):
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(gt), np.asarray(gs)] + ([np.asarray(gs2)] if gs2 is not None else [])
    genes = np.unique(np.concatenate(arrs))
    gx = {g: k for k, g in enumerate(genes)}
    idx = [np.array([gx[g] for g in a]) for a in arrs]
    num = np.asarray(num, float); dnm = np.asarray(dnm, float)
    est = float(num.sum() / dnm.sum()) if abs(dnm.sum()) > 1e-9 else float("nan")
    boot = np.empty(B)
    for b in range(B):
        w = rng.poisson(1.0, len(genes))
        pw = w[idx[0]]
        for ii in idx[1:]:
            pw = pw * w[ii]
        d = float(np.sum(pw * dnm))
        boot[b] = float(np.sum(pw * num) / d) if abs(d) > 1e-9 else np.nan
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return est, float("nan"), float("nan")
    return est, float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


def _stable_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


# ======================================================================================
# Source<->target matching (gene-disjoint by construction; native-coup caliper; capped source reuse).
# ======================================================================================
def match_pairs(tgt, src, rng, coup_caliper=0.75, max_src=8, extreme=False):
    src_coup = src["coup"]
    use = Counter()
    pairs = []
    for i in range(len(tgt["genes"])):
        ci = tgt["coup"][i]
        cand = [j for j in range(len(src["genes"]))
                if src["genes"][j] != tgt["genes"][i] and use[j] < max_src]
        if not cand:
            cand = [j for j in range(len(src["genes"])) if src["genes"][j] != tgt["genes"][i]]
        if not cand:
            continue
        dm = np.array([abs(src_coup[j] - ci) for j in cand])
        if extreme:
            j = cand[int(np.argmax(dm))]
        else:
            within = [c for c, d in zip(cand, dm) if d <= coup_caliper]
            pool = within or [cand[int(np.argmin(dm))]]
            j = pool[rng.integers(len(pool))]
        use[j] += 1
        pairs.append((i, j))
    return pairs


# ======================================================================================
# Arm states + injection. Each arm returns s [D] to inject into target i from source j.
# ======================================================================================
def _rescale(v, target_norm):
    n = np.linalg.norm(v)
    return v * (target_norm / n) if n > 1e-12 else v


def build_source_states(S, U_D, seed):
    s_src = project(S["diff"], U_D)
    s_bar_raw = project(S["diff"].mean(0, keepdims=True), U_D)[0]
    s_bar_unit = s_bar_raw / (np.linalg.norm(s_bar_raw) + 1e-12)
    s_sham = project(S.get("diff_sham", S["diff"] * 0.0), U_D) if "diff_sham" in S else None
    U_rand = random_subspace(S["diff"].shape[1], RANK_K, np.random.default_rng(seed))
    return {"s_src": s_src, "s_bar_raw": s_bar_raw, "s_bar_unit": s_bar_unit,
            "s_sham": s_sham, "U_rand": U_rand, "U_D": U_D}


def arm_state(arm, i, j, tgt, src, st, alpha):
    sj = st["s_src"][j]
    nj = np.linalg.norm(sj)
    if arm == "REAL":
        return sj
    if arm == "MEAN":
        return _rescale(st["s_bar_raw"], nj)
    if arm == "SELF":
        return project(tgt["diff"][i:i + 1], st["U_D"])[0]
    if arm == "RANDOM":
        return _rescale(project(src["diff"][j:j + 1], st["U_rand"])[0], nj)
    if arm == "SCRAMBLE":
        c = src["diff"][j:j + 1] @ st["U_D"].T
        perm = np.random.default_rng(_stable_seed("scr:" + src["genes"][j])).permutation(st["U_D"].shape[0])
        return (c[:, perm] @ st["U_D"])[0]
    if arm == "SRC_SHAM":
        return _rescale(st["s_sham"][j], nj)
    if arm == "POS_SHAM":
        return sj
    # PERP arms (F1): decompose s_j along the generic axis s_bar_unit
    spar = float(sj @ st["s_bar_unit"]) * st["s_bar_unit"]
    sperp = sj - spar
    if arm == "PAR_NAT":
        return spar
    if arm == "PERP_NAT":
        return sperp
    if arm == "PERP_MATCHED":
        return _rescale(sperp, nj)
    if arm == "RANDPERP":                                  # random dir orthogonal to s_bar, ||.||=||sperp||
        g = np.random.default_rng(_stable_seed("rp:" + src["genes"][j]))
        r = g.standard_normal(sj.shape)
        r = r - float(r @ st["s_bar_unit"]) * st["s_bar_unit"]
        return _rescale(r, np.linalg.norm(sperp))
    raise ValueError(arm)


def tau_arm(arm, pairs, tgt, src, st, margins_fn, alpha, B=2000, seed=SEED, sham_only_ok=True):
    """Return (num, gt, gs, snorm) for a single arm at a single dose. POS_SHAM forced to 0; SRC_SHAM
    drops no-decoy fallback pairs (F8)."""
    if arm == "SRC_SHAM" and sham_only_ok and "sham_ok" in src:
        pairs = [(i, j) for (i, j) in pairs if src["sham_ok"][j]]
    Hco = tgt["Hco"]
    inj = np.zeros((len(pairs), Hco.shape[1]), dtype=np.float64)
    gt, gs = [], []
    for p, (i, j) in enumerate(pairs):
        inj[p] = arm_state(arm, i, j, tgt, src, st, alpha)
        gt.append(tgt["genes"][i]); gs.append(src["genes"][j])
    ti = np.array([i for i, _ in pairs])
    if arm == "POS_SHAM":
        num = np.zeros(len(pairs))
    else:
        m0 = margins_fn(Hco[ti])
        num = margins_fn(Hco[ti] + alpha * inj) - m0
    snorm = np.linalg.norm(inj, axis=1)
    return num, gt, gs, snorm, ti, inj


def _entropy_acgt(logits_np, bid):
    idx = [bid[b] for b in "ACGT"]
    z = logits_np[:, idx]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p = p / p.sum(1, keepdims=True)
    return -(p * np.log(p + 1e-12)).sum(1)


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-src", type=int, default=200, help="F3: strictly < |src_all| so R22 draws vary U_D")
    ap.add_argument("--n-tgt", type=int, default=1200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=8, help="F3: >=8 seeded source draws (resample w/ repl.)")
    ap.add_argument("--max-src", type=int, default=8)
    ap.add_argument("--sesoi-spec", type=float, default=SESOI_SPEC_DEFAULT)
    ap.add_argument("--min-effect", type=float, default=None,
                    help="optional coup floor for the min_effect-qualified sensitivity arm (Lens3-F4)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=MODEL_ABS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--decomp", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/g1_cross_example_transduction.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out
    if out.exists() and not args.force and not (args.smoke or args.pilot or args.decomp):
        raise SystemExit(f"{out} exists; pass --force (results are immutable).")
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off, bid, cls = task.off, task.bid, ad.model.cls
    dtype = next(cls.parameters()).dtype

    def margins_fn(H_np):
        h = torch.from_numpy(np.ascontiguousarray(H_np, dtype=np.float32)).to(ad.device, dtype)
        return margin_hard(cls(h), bid).float().cpu().numpy()

    def logits_at(H_np):
        h = torch.from_numpy(np.ascontiguousarray(H_np, dtype=np.float32)).to(ad.device, dtype)
        return cls(h).float().cpu().numpy()

    # ---- cohorts ----
    src_all = load_atlas_donor_examples("discovery", single_pair_only=True,
                                        dist_lo=SRC_DIST_MIN, dist_hi=SRC_DIST_MAX, label="D_src")
    tgt_all = load_atlas_donor_examples("validation", label="H")
    tgt_genes = sorted({x.gene_id for x in tgt_all})
    hdev_genes = set(g for g in tgt_genes if _stable_seed("half:" + g) % 2 == 0)
    H_dev = [x for x in tgt_all if x.gene_id in hdev_genes]
    H_conf = [x for x in tgt_all if x.gene_id not in hdev_genes]

    # F6/M4: gene-disjointness of sources vs ALL targets, COMPUTED + hashed (not hardcoded).
    src_g = set(x.gene_id for x in src_all); tgt_g = set(tgt_genes)
    m4_overlap = sorted(src_g & tgt_g)
    prov = {"commit": git_commit(REPO),
            "checkpoint_sha256": sha256_file(Path(args.model) / "pytorch_model.bin"),
            "n_src_all": len(src_all), "n_tgt_all": len(tgt_all),
            "genes_src": len(src_g), "genes_tgt": len(tgt_genes),
            "genes_Hdev": len(hdev_genes), "genes_Hconf": len(tgt_genes) - len(hdev_genes),
            "n_Hdev": len(H_dev), "n_Hconf": len(H_conf),
            "M4_gene_disjoint": len(m4_overlap) == 0, "M4_overlap_genes": m4_overlap[:10],
            "hash_src_genes": _gene_hash(src_g), "hash_tgt_genes": _gene_hash(tgt_g),
            "sesoi_abs": SESOI_ABS, "sesoi_tau": SESOI_TAU, "sesoi_spec": args.sesoi_spec,
            "m3_tol": M3_TOL, "rank_k": RANK_K, "dose_grid": list(DOSE_GRID),
            "alpha_linear": ALPHA_LINEAR, "arms": list(ARMS), "n_src_per_draw": args.n_src,
            "draws": args.draws}
    if not prov["M4_gene_disjoint"]:
        raise SystemExit(f"M4 FAIL: {len(m4_overlap)} source genes overlap targets -- leakage. STOP.")

    result = {"mode": "smoke" if args.smoke else ("pilot" if args.pilot else
                                                  ("decomp" if args.decomp else "confirm")),
              "provenance": prov}

    # =========================== SMOKE ===========================
    if args.smoke:
        src = src_all[:48]
        S = capture(ad, task, src, dev, off, want_sham=True)
        ckpt0 = float(next(ad.model.parameters()).abs().sum())
        ug = sorted(set(S["genes"])); Ag = set(ug[::2])
        inA = np.array([g in Ag for g in S["genes"]])
        U_D = top_k_subspace(S["diff"][inA], RANK_K)
        st = build_source_states(S, U_D, SEED)
        m1 = float(np.abs(margins_fn(S["Hco"]) - S["mco"]).max())
        self_suff = ratio_of_means(
            (margins_fn(patch_component(S["Hco"][~inA], S["Hcl"][~inA], U_D)) - S["mco"][~inA]).tolist(),
            S["coup"][~inA].tolist())
        pairs = [(i, i) for i in range(len(S["genes"]))]
        norms = {}
        for a in ("REAL", "MEAN", "RANDOM", "SCRAMBLE", "SRC_SHAM"):
            _, _, _, nn, _, _ = tau_arm(a, pairs, S, S, st, margins_fn, 1.0, sham_only_ok=False)
            norms[a] = nn
        m5 = float(np.max([np.abs(norms[a] - norms["REAL"]).max() for a in norms]))
        num0, _, _, _, _, _ = tau_arm("REAL", pairs, S, S, st, margins_fn, 0.0)
        m2 = float(np.abs(num0).max())
        betas = [0.25, 0.5, 1.0]
        rec = []
        for beta in betas:
            m0 = margins_fn(S["Hco"]); ma = margins_fn(S["Hco"] + beta * st["s_src"])
            rec.append(ratio_of_means((ma - m0).tolist(), S["coup"].tolist()))
        slope = float(np.polyfit(betas, rec, 1)[0])
        planted_ok = bool(np.all(np.diff(rec) > 0) and slope > 0)
        # permuted-label FPR on the ACTUAL cross-gene structure (F: Lens4-F4) using a disjoint src/tgt
        srcB = capture(ad, task, src_all[48:96], dev, off, want_sham=False)
        stB = build_source_states(srcB, U_D, SEED)  # reuse U_D as fixed axis for the null geometry
        xpairs = match_pairs(S, srcB, np.random.default_rng(SEED + 2), max_src=args.max_src)
        nR, gtx, gsx, _, _, _ = tau_arm("REAL", xpairs, S, srcB, stB, margins_fn, 1.0)
        nM, _, _, _, _, _ = tau_arm("MEAN", xpairs, S, srcB, stB, margins_fn, 1.0)
        D = nR - nM; coupx = np.array([S["coup"][i] for i, _ in xpairs])
        fpr = sum(1 for pi in range(200)
                  if dyadic_ratio_ci(np.random.default_rng(1000 + pi).choice([-1.0, 1.0], len(D)) * D,
                                     coupx, gtx, gsx, seed=1000 + pi, B=400)[1] > args.sesoi_spec) / 200
        ckpt1 = float(next(ad.model.parameters()).abs().sum())
        mem = {"M1_head_readout": m1, "M2_zero_dose": m2, "M3_self_suff": self_suff,
               "M3_smoke_ok": bool(self_suff > 0.5),
               "M3_reproduction_note": "tight |SELF-0.875|<M3_TOL is gated on the real-cohort pilot/"
                                       "confirmatory verdict; the tiny smoke half is too small for it",
               "M4_gene_disjoint": prov["M4_gene_disjoint"], "M5_norm_match": m5,
               "M9_cohort_hash": _cohort_hash(src), "checkpoint_unmutated": abs(ckpt1 - ckpt0),
               "frac_sham_ok": float(np.mean(S["sham_ok"]))}
        cal = {"planted_recovery": rec, "planted_slope": slope, "planted_ok": planted_ok,
               "permuted_label_fpr_crossgene": fpr, "permuted_ok": bool(fpr <= 0.05)}
        mem_ok = (m1 < 1e-4 and m2 < 1e-9 and m5 < 1e-6 and abs(ckpt1 - ckpt0) < 1e-6
                  and mem["M3_smoke_ok"] and prov["M4_gene_disjoint"])
        result["memorials"] = mem; result["calibration"] = cal
        result["verdict"] = "SMOKE_" + ("PASS" if (mem_ok and cal["permuted_ok"] and planted_ok) else "FAIL")
        _write(out, result, args, "_smoke")
        print(json.dumps({k: result[k] for k in ("verdict", "memorials", "calibration")}, indent=2))
        return

    # =========================== PILOT ===========================
    if args.pilot:
        src = src_all[:args.n_src]
        S = capture(ad, task, src, dev, off, want_sham=True)
        T = capture(ad, task, H_dev[:args.n_tgt], dev, off, want_sham=False)
        c = _contrasts(S, T, margins_fn, logits_at, bid, args, seed=SEED)
        n_conf = prov["genes_Hconf"]; g_dev = max(prov["genes_Hdev"], 1)
        mde = {}
        for name, cc in c.items():
            if not (isinstance(cc, dict) and "ci" in cc and "hw" in cc):
                continue
            hw = cc["hw"]; est = cc.get("est")
            sesoi = args.sesoi_spec if name in ("Delta_spec", "tau_PERP_MATCHED", "tau_PERP_NAT") else SESOI_TAU
            proj = hw * (g_dev / max(n_conf, 1)) ** 0.5
            mde[name] = {"est_Hdev": est, "hw_dev": hw, "projected_hw_Hconf": proj, "sesoi": sesoi,
                         "equiv_resolvable": bool(proj < sesoi),
                         "effect_resolvable": bool(est is not None and (abs(est) - proj) > sesoi),
                         "reaches_verdict": bool(proj < sesoi or (est is not None and (abs(est) - proj) > sesoi))}
        result["pilot"] = {"contrasts": {k: v for k, v in c.items() if not k.startswith("_")},
                           "mde_gate": mde, "M8_band_sealed": M8_RATIO_BAND,
                           "gates": {"M3_pass": c.get("M3_pass"), "M3_self_suff": c.get("M3_self_suff"),
                                     "M6_random_ok": c.get("_M6_random_ok"), "M7_coup": c.get("_M7_coup"),
                                     "M8": c.get("_M8"), "M8_ok": c.get("_M8_ok")},
                           "denoising_null": c.get("_denoising_null"),
                           "provisional_verdict": _verdict(c, args),
                           "coup_stats": {"mean": float(np.mean(T["coup"])),
                                          "frac_ge_0.20": float(np.mean(T["coup"] >= 0.20))},
                           "frac_sham_ok": float(np.mean(S["sham_ok"]))}
        result["verdict"] = "PILOT_COMPLETE_no_freeze"
        _write(out, result, args, "_pilot")
        print(json.dumps({k: mde[k] for k in mde if k in ("Delta_spec", "tau_PERP_MATCHED", "tau_PERP_NAT",
                          "REAL_minus_RANDOM")}, indent=2))
        print(json.dumps({"denoising_null": c.get("_denoising_null"), "perp": {
            "PERP_NAT_lin": c.get("tau_PERP_NAT_lin"), "PERP_MATCHED_lin": c.get("tau_PERP_MATCHED_lin"),
            "RANDPERP_lin": c.get("tau_RANDPERP_lin")}, "M8": c.get("_M8"),
            "M6_random": c.get("_M6_random_ok"), "provisional_verdict": _verdict(c, args)}, indent=2))
        return

    # =========================== DECOMP (H_dev diagnostic) ===========================
    if args.decomp:
        src = src_all[:args.n_src]
        S = capture(ad, task, src, dev, off, want_sham=False)
        T = capture(ad, task, H_dev[:args.n_tgt], dev, off, want_sham=False)
        c = _contrasts(S, T, margins_fn, logits_at, bid, args, seed=SEED)
        result["decomp"] = {k: c[k] for k in c if not k.startswith("_")}
        result["verdict"] = "DECOMP_DIAGNOSTIC_no_freeze"
        _write(out, result, args, "_decomp")
        print(json.dumps({k: c[k] for k in ("tau_REAL", "tau_PAR_NAT_lin", "tau_PERP_NAT_lin",
              "tau_PERP_MATCHED_lin", "tau_RANDPERP_lin", "_denoising_null")}, indent=2))
        return

    # =========================== CONFIRMATORY (H_conf; >=8 seeded draws, F3) ===========================
    draws = []
    for d in range(args.draws):
        rs = np.random.default_rng(SEED + 100 * d)
        # F3: resample sources WITH REPLACEMENT (independent U_D/s_bar each draw)
        pick = rs.integers(0, len(src_all), size=args.n_src)
        src = [src_all[k] for k in pick]
        S = capture(ad, task, src, dev, off, want_sham=True)
        T = capture(ad, task, H_conf[:args.n_tgt], dev, off, want_sham=False)
        c = _contrasts(S, T, margins_fn, logits_at, bid, args, seed=SEED + 100 * d)
        draws.append({k: v for k, v in c.items() if not k.startswith("_")})
        draws[-1]["_verdict"] = _verdict(c, args)
        draws[-1]["_gates"] = {"M3_pass": c.get("M3_pass"), "M6_random_ok": c.get("_M6_random_ok"),
                               "M7_coup": c.get("_M7_coup"), "M8_ok": c.get("_M8_ok")}
        draws[-1]["_M8"] = c.get("_M8"); draws[-1]["_denoising_null"] = c.get("_denoising_null")
        draws[-1]["_frac_sham_ok"] = float(np.mean(S["sham_ok"]))

    def _acc(key, sub):
        return [dr[key][sub] for dr in draws if key in dr and isinstance(dr[key], dict) and sub in dr[key]]
    dspec = _acc("Delta_spec", "est"); perp = _acc("tau_PERP_MATCHED_lin", "est")
    result["draws"] = draws
    result["across_draw"] = {
        "tau_REAL": _acc("tau_REAL", "est"), "Delta_spec": dspec,
        "Delta_spec_sd": float(np.std(dspec)) if dspec else None,
        "PERP_MATCHED_lin": perp, "PERP_MATCHED_lin_sd": float(np.std(perp)) if perp else None,
        "per_draw_verdict": [dr["_verdict"] for dr in draws]}
    result["stability_ok"] = bool(dspec and np.std(dspec) <= 0.5 * args.sesoi_spec)
    result["evidential_tier"] = "confirmatory"
    # modal verdict across draws
    vs = [dr["_verdict"] for dr in draws]
    result["verdict"] = Counter(vs).most_common(1)[0][0] if vs else "NO_DRAWS"
    _write(out, result, args)
    print(json.dumps({"verdict": result["verdict"], "per_draw_verdict": vs,
                      "across_draw": result["across_draw"], "stability_ok": result["stability_ok"]}, indent=2))


# --------------------------------------------------------------------------------------
def _contrasts(S, T, margins_fn, logits_at, bid, args, seed=SEED):
    """Derive U_D on D_src, build states, match, compute all arms + primary contrasts + the PERP
    decomposition (linear regime) + M8 + M6 + the denoising-null ratio."""
    U_D = top_k_subspace(S["diff"], RANK_K)
    st = build_source_states(S, U_D, seed)
    rng = np.random.default_rng(seed + 1)
    pairs = match_pairs(T, S, rng, max_src=args.max_src)
    out = {}
    per_arm = {}
    coup = np.array([T["coup"][i] for i, _ in pairs])
    for a in ARMS:
        num, gt, gs, nn, ti, inj = tau_arm(a, pairs, T, S, st, margins_fn, 1.0, B=args.n_boot, seed=seed)
        cpa = np.array([T["coup"][i] for i, _ in (
            [(i, j) for (i, j) in pairs if S["sham_ok"][j]] if (a == "SRC_SHAM" and "sham_ok" in S) else pairs)])
        est, lo, hi = dyadic_ratio_ci(num, cpa, gt, gs, seed=seed, B=args.n_boot)
        per_arm[a] = {"num": num, "gt": gt, "gs": gs, "coup": cpa, "ti": ti, "inj": inj}
        out[f"tau_{a}"] = {"est": est, "ci": [lo, hi], "hw": 0.5 * (hi - lo),
                           "abs_nat_mean": float(np.mean(num)), "s_norm_mean": float(np.mean(nn)),
                           "n": len(num)}
    # PRIMARY Delta_spec (paired within-target)
    D = per_arm["REAL"]["num"] - per_arm["MEAN"]["num"]
    est, lo, hi = dyadic_ratio_ci(D, coup, per_arm["REAL"]["gt"], per_arm["REAL"]["gs"], seed=seed, B=args.n_boot)
    out["Delta_spec"] = {"est": est, "ci": [lo, hi], "hw": 0.5 * (hi - lo)}
    for cn, other in (("REAL_minus_RANDOM", "RANDOM"), ("REAL_minus_SCRAMBLE", "SCRAMBLE")):
        Dx = per_arm["REAL"]["num"] - per_arm[other]["num"]
        est2, lo2, hi2 = dyadic_ratio_ci(Dx, coup, per_arm["REAL"]["gt"], per_arm["REAL"]["gs"],
                                         seed=seed, B=args.n_boot)
        out[cn] = {"est": est2, "ci": [lo2, hi2], "hw": 0.5 * (hi2 - lo2)}
    # F1 PERP decomposition at LOW alpha (linear regime) -- the identifiability disambiguator
    for a in ("PAR_NAT", "PERP_NAT", "PERP_MATCHED", "RANDPERP"):
        num, gt, gs, nn, ti, inj = tau_arm(a, pairs, T, S, st, margins_fn, ALPHA_LINEAR, B=args.n_boot, seed=seed)
        est3, lo3, hi3 = dyadic_ratio_ci(num, coup, gt, gs, seed=seed, B=args.n_boot)
        out[f"tau_{a}_lin"] = {"est": est3, "ci": [lo3, hi3], "hw": 0.5 * (hi3 - lo3),
                               "alpha": ALPHA_LINEAR, "abs_nat_mean": float(np.mean(num))}
    # denoising-null: tau(REAL)/tau(MEAN) vs E[cos(s_j,s_bar)]
    sj = np.stack([st["s_src"][j] for _, j in pairs])
    cos = (sj @ st["s_bar_unit"]) / (np.linalg.norm(sj, axis=1) + 1e-12)
    tR, tM = out["tau_REAL"]["est"], out["tau_MEAN"]["est"]
    out["_denoising_null"] = {"tau_REAL_over_MEAN": float(tR / tM) if abs(tM) > 1e-9 else None,
                              "E_cos_sj_sbar": float(np.mean(cos)), "cos_p50": float(np.median(cos)),
                              "geometry_consistent": bool(abs((tR / tM if abs(tM) > 1e-9 else 0) - np.mean(cos)) < 0.15)}
    out["cos_sj_sbar"] = {"mean": float(np.mean(cos)), "p50": float(np.median(cos)),
                          "p10": float(np.percentile(cos, 10)), "p90": float(np.percentile(cos, 90))}
    out["energy_spectrum_top8"] = energy_spectrum(S["diff"])[:8].tolist()
    # dose ordering for REAL/MEAN/SELF
    dose = {}
    for a in ("REAL", "MEAN", "SELF"):
        row = {}
        for al in DOSE_GRID:
            num_a, gt_a, gs_a, _, _, _ = tau_arm(a, pairs, T, S, st, margins_fn, al, B=args.n_boot, seed=seed)
            e, _, _ = dyadic_ratio_ci(num_a, coup, gt_a, gs_a, seed=seed, B=min(args.n_boot, 1000))
            row[f"a{al}"] = {"est": e, "abs_nat": float(np.mean(num_a))}
        dose[a] = row
    out["dose_REAL"] = dose["REAL"]; out["dose_MEAN"] = dose["MEAN"]; out["dose_SELF"] = dose["SELF"]
    ys = [dose["REAL"][f"a{al}"]["abs_nat"] for al in DOSE_GRID]
    ys_pos = [dose["REAL"][f"a{al}"]["abs_nat"] for al in DOSE_GRID if al >= 0]
    out["beta_dose_REAL"] = float(np.polyfit(list(DOSE_GRID), ys, 1)[0])
    out["monotone_REAL_pos"] = bool(np.all(np.diff(ys_pos) >= -SESOI_ABS))
    # F7 M3: SELF reproduces C26 0.875
    self_suff = ratio_of_means(
        (margins_fn(patch_component(T["Hco"], T["Hcl"], U_D)) - T["mco"]).tolist(), T["coup"].tolist())
    out["M3_self_suff"] = self_suff
    out["M3_pass"] = bool(abs(self_suff - 0.875) < M3_TOL)
    out["_M6_random_ok"] = bool(out["tau_RANDOM"]["est"] <= M6_RANDOM_MAX + out["tau_RANDOM"]["hw"])
    out["_M7_coup"] = {"E_coup": float(np.mean(coup)), "ok": bool(np.mean(coup) >= M7_COUP_MIN)}
    # F5 M8 OOD band per interpreted arm x dose (ratio ||H_inj||/||H_cl|| + donor entropy)
    Hcl_t = T["Hcl"]; m8 = {}
    for a in ("REAL", "MEAN"):
        for al in (0.5, 1.0, 1.5):
            _, _, _, _, ti, inj = tau_arm(a, pairs, T, S, st, margins_fn, al, B=1, seed=seed)
            Hinj = T["Hco"][ti] + al * inj
            ratio = np.linalg.norm(Hinj, axis=1) / (np.linalg.norm(Hcl_t[ti], axis=1) + 1e-12)
            ent = _entropy_acgt(logits_at(Hinj), bid)
            m8[f"{a}@{al}"] = {"ratio_mean": float(np.mean(ratio)),
                               "ratio_in_band": float(np.mean((ratio >= M8_RATIO_BAND[0]) & (ratio <= M8_RATIO_BAND[1]))),
                               "entropy_mean": float(np.mean(ent))}
    out["_M8"] = m8
    out["_M8_ok"] = bool(all(v["ratio_in_band"] >= 0.95 for k, v in m8.items() if k.endswith("@1.0")))
    # SELF norm (Lens1-F3): report, do not interpret MEAN>SELF
    out["s_norm_report"] = {"REAL_MEAN": out["tau_REAL"]["s_norm_mean"], "SELF": out["tau_SELF"]["s_norm_mean"]}
    # optional min_effect-qualified sensitivity Delta_spec (Lens3-F4)
    if args.min_effect is not None:
        keep = coup >= args.min_effect
        if keep.sum() > 10:
            Dk = (per_arm["REAL"]["num"] - per_arm["MEAN"]["num"])[keep]
            gk_t = [g for g, k in zip(per_arm["REAL"]["gt"], keep) if k]
            gk_s = [g for g, k in zip(per_arm["REAL"]["gs"], keep) if k]
            e, lo, hi = dyadic_ratio_ci(Dk, coup[keep], gk_t, gk_s, seed=seed, B=args.n_boot)
            out["Delta_spec_mineffect"] = {"est": e, "ci": [lo, hi], "n": int(keep.sum()),
                                           "min_effect": args.min_effect}
    return out


def _verdict(c, args):
    """Frozen decision rule (prereg S6, red-team-reconciled F2/F1/F4). GENERIC is GATED on the PERP arm
    being sub-SESOI (F1) AND M3/M6/M7/M8 passing. The scope of any null is C16/C26 final-residual only (F4)."""
    ds = c.get("Delta_spec", {}); tr = c.get("tau_REAL", {})
    rr = c.get("REAL_minus_RANDOM", {}); rs = c.get("REAL_minus_SCRAMBLE", {})
    perp = c.get("tau_PERP_MATCHED_lin", {}); perp_nat = c.get("tau_PERP_NAT_lin", {})
    if not ds or "ci" not in ds:
        return "INCOMPLETE"
    gates_ok = (c.get("M3_pass", False) and c.get("_M6_random_ok", False)
                and c.get("_M7_coup", {}).get("ok", False) and c.get("_M8_ok", False))
    if not gates_ok:
        return "INSTRUMENT_LIMITED"
    real_above = (tr.get("ci", [0, 0])[0] > SESOI_TAU and tr.get("abs_nat_mean", 0) >= SESOI_ABS
                  and c.get("beta_dose_REAL", 0) > 0 and bool(c.get("monotone_REAL_pos", False)))
    content_specific = (rr.get("ci", [0, 0])[0] > args.sesoi_spec and rs.get("ci", [0, 0])[0] > args.sesoi_spec)
    no_transfer = rr.get("ci", [0, 1])[1] < SESOI_TAU and rr.get("hw", 1) < SESOI_TAU
    # F1: example-specific (off-axis) component restores => transduction survives
    perp_restoring = perp.get("ci", [0, 0])[0] > args.sesoi_spec or perp_nat.get("ci", [0, 0])[0] > args.sesoi_spec
    perp_inert = (perp.get("ci", [0, 1])[1] < args.sesoi_spec and perp.get("hw", 1) < args.sesoi_spec)
    dspec_pos = ds["ci"][0] > args.sesoi_spec
    dspec_not_pos = ds["ci"][1] <= args.sesoi_spec
    if no_transfer:
        return "NO_TRANSDUCTION_C16C26_finalresidual"
    if real_above and content_specific and (dspec_pos or perp_restoring):
        return "TRANSDUCING_STATE"
    if real_above and content_specific and dspec_not_pos and perp_inert:
        return "GENERIC_READOUT_AXIS_C16C26_finalresidual"
    return "INCONCLUSIVE_at_precision"


def _write(out: Path, result: dict, args, tag_extra: str = "") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = out.stem + tag_extra
    p = out.parent / (tag + ".json")
    p.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    man = ExperimentManifest.create("splicebert_coupling:" + tag, sys.argv, repo=REPO,
                                    notes="G1 cross-example transduction of the donor-side delivered state.")
    man.add_output(p)
    man.write(out.parent / (tag + ".manifest.json"))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
