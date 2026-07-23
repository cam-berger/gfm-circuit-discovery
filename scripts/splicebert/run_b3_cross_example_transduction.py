#!/usr/bin/env python
"""B3 -- cross-example transduction of the acceptor payload (donor circuit generalization).

Frozen design of record:
  studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_B3_CROSS_EXAMPLE_TRANSDUCTION.md
This runner implements the RED-TEAM v2 (the v1 in that prereg must NOT be run; its two fatals are
fixed here). The scientific question: is the donor circuit's acceptor payload a *transferable*
information-bearing state, or does intra-example clean/corrupt patching only work within a matched
pair? A positive result promotes the mechanism from "established on the tested distribution" to
"interventionally general", which is the single largest rejection-risk for the donor-circuit paper.

WHAT THIS MEASURES (v2, must-fixes folded in)
  Object (MF1): the natural acceptor-payload edit delivered through the TARGET's own clean address,
    exactly A9's cross-example interchange -- NOT a Pi_U final-residual patch (C24: the payload is
    NOT concentrated in U_info; L3H1 converts a broad acceptor-local value into a compact donor-side
    object, so the injectable object is the value-path payload dr = r3(source) - r3(target), delivered
    dV = W_V dr, written at the donor via head_output, read out at the masked-donor margin).
  Estimand (MF2 -- close the U_info backdoor): transfer is
        M_xfer = margin(inject CLEAN-source payload) - margin(inject CORRUPT-source payload),
    both cross-example into the SAME target-corrupt run. Clean vs corrupt source differ ONLY in
    partner-intactness, so a generic clean-donor offset cannot ride through (that was the v1 fatal:
    clean-source > EMPTY admits any clean occupancy; clean-source > corrupt-source does not).
  Recovery ratio (R20-safe, ratio-of-means, never mean-of-ratios):
        R_xfer = <M_xfer> / <M_self>,  M_self = margin(inject target's OWN clean payload) - corrupt-source.
  Dose (MF4 -- the confirmable endpoint is ONE-SIDED POSITIVE): scale the payload edit by
    alpha in {0, 0.5, 1.0, 1.5}; require beta_dose > 0 with a monotone grid, gene-dyadic CI-low > 0.
  Uncertainty (MF5): a NET-NEW two-way source-gene x target-gene Poisson-multiplier dyadic bootstrap
    for the ratio-of-means (each gene is reused as source-for-many and target-for-many). One-way
    gboot_ratio is reported alongside but is NOT the inferential gate.

CONTROLS (a POSITIVE call requires ALL of these, each enforced in the verdict formula)
  - matched cohort: source!=target GENE (gene-disjoint), matched on strand, coding status, and
    target-partner distance (|Δpdist|<=200), source chosen nearest in native donor margin (caliper).
  - margin-MISMATCHED stratum (MF2 confound gate): a source matched on strand/coding/pdist but FAR in
    native margin; the matched primary transfer CI-low must EXCEED the mismatched stratum CI-high, so
    the effect is not a margin-gap artifact.
  - scrambled-source: permute the source payload's HIDDEN-FEATURE COORDINATES (norm/per-position
    energy preserved, content destroyed) -> a content-specific transfer must collapse (2026-07-23
    redesign; the shipped position-permutation was too weak and never fired).
  - non-target-position: deliver at a decoy donor -> must collapse.
  - permuted-pairing dyadic FPR: randomize source<->target assignment -> R_xfer CI must include 0.
  - health gate: masked-MLM health retained under injection (no generic damage).

MODES
  --smoke  : mechanics memorials + calibration only, tiny-n. NO verdict. Run this FIRST on the main
             checkout (model+data present) to shake out any signature drift before spending MPS.
  --pilot  : MF3 -- measure the ACTUAL cross-example numerator <M_xfer> and its per-gene dyadic
             half-width on the DISCOVERY split, then STOP. Re-derive power from the piloted numerator
             (never from R_self=+0.875 or the rank-1 overlap 0.970). No freeze, no confirmatory read.
  (default): full confirmatory run on the requested split (val by default; chr8/9 sealed, NOT opened).

Model/data are required (SpliceBERT.1024nt + hg38 + GENCODE v50). This worktree may lack them; run
from the main checkout or symlink data/models in (gitignored). No new kernel; reuses A9/C16 machinery.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest                                # noqa: E402
from gfmcd.circuits.hf_bert import patch_heads                                 # noqa: E402
from gfmcd.splicing.circuit import (                                           # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask,
    gboot_ratio, load_adapter, load_examples, matched_decoy_donor, pick_device,
)
from gfmcd.splicing.coupling import substitute                                 # noqa: E402
from gfmcd.splicing.payload import deliver, head_output, value_weight_head     # noqa: E402

L3, H3 = 3, 1
ACC_OFF = (-3, -2, -1, 0)          # four-token acceptor window (C24 payload locus)
CORRUPT_ALT = "CC"                 # partner AG -> CC, the C1 corruption (matches CouplingTask.pair)
SESOI_R = 0.20                     # smallest transfer ratio worth calling positive (prereg S7)
SESOI_M = 0.02                     # smallest margin-nat effect (matches B1/C35 value scale)
DOSE_GRID = (0.0, 0.5, 1.0, 1.5)
SEED = 0


# --------------------------------------------------------------------------------------
# Verified model idioms, replicated verbatim from run_a9_payload_semantics.py (working code).
# --------------------------------------------------------------------------------------
def caprun(ad, inp, dtok, want_attn):
    """Capture (output, layer-3 residual, L3H1 value head, L3H1 attention row at the donor)."""
    vbox = {}
    hh = ad.attention_self_modules[L3].value.register_forward_hook(
        lambda _m, _i, o: vbox.__setitem__("v", o.detach()))
    try:
        o = ad.model(**inp, output_hidden_states=True, output_attentions=want_attn)
    finally:
        hh.remove()
    Dh = ad.hidden_size // ad.num_heads
    vhead = vbox["v"][:, :, H3 * Dh:(H3 + 1) * Dh]
    B = inp["input_ids"].shape[0]
    ap = o.attentions[L3][torch.arange(B, device=dtok.device), H3, dtok, :] if want_attn else None
    return o, o.hidden_states[L3], vhead, ap


@contextmanager
def inject_capture(ad, corr, dtok, recon):
    """Inject recon [B,Dh] as L3H1's donor-token output; yield the patched forward output."""
    B, Dh = recon.shape
    S = corr["input_ids"].shape[1]
    src = torch.zeros(B, ad.num_heads, S, Dh, device=recon.device, dtype=recon.dtype)
    b = torch.arange(B, device=recon.device)
    src[b, H3, dtok] = recon
    mask = torch.zeros(B, S, dtype=torch.bool, device=recon.device); mask[b, dtok] = True
    with patch_heads(ad, {L3: src}, [(L3, H3)], token_mask=mask):
        yield ad.model(**corr, output_hidden_states=True)


# --------------------------------------------------------------------------------------
# Per-example capture: everything needed to re-inject any source payload into any target.
# --------------------------------------------------------------------------------------
def capture_examples(ad, task, X, dev, off, Wv):
    Dh = ad.hidden_size // ad.num_heads
    E = []
    lin_err = 0.0
    for i in range(0, len(X), task.batch):
        ch = X[i:i + task.batch]
        cl, co, dn = task.pair(ch)
        B = cl["input_ids"].shape[0]
        dtok = torch.tensor([d + off for d in dn], device=dev)
        accwin = torch.tensor([[x.next_acceptor_pos + off + k for k in ACC_OFF] for x in ch], device=dev)
        bidx = torch.arange(B, device=dev)[:, None].expand(B, 4)
        oc, r3_cl, V_cl, Ap = caprun(ad, cl, dtok, True)
        oo, r3_co, V_co, Ap_co = caprun(ad, co, dtok, True)  # corrupt attn A- captured for recon memorial
        base = torch.einsum("bs,bsd->bd", Ap, V_co)          # CLEAN address A+ . corrupt values V- (MF1 object)
        base_corr = torch.einsum("bs,bsd->bd", Ap_co, V_co)  # NATURAL corrupt donor output A- . V- (== m_co)
        # value-linearity memorial: delivered value edit must equal W_V (residual edit)
        dVt = (V_cl - V_co)[bidx, accwin]
        dVl = deliver(Wv, (r3_cl - r3_co)[bidx, accwin])
        lin_err = max(lin_err, float((dVt - dVl).abs().max()))
        m_cl = task.margin(oc.logits.float(), dn)
        m_co = task.margin(oo.logits.float(), dn)
        for bj, x in enumerate(ch):
            E.append({
                "r3cl": r3_cl[bj, accwin[bj]].cpu(),        # [4,H] source clean payload donor
                "r3co": r3_co[bj, accwin[bj]].cpu(),        # [4,H] its own corrupt (partner destroyed)
                "Ap": Ap[bj, accwin[bj]].cpu(),             # [4] address weights at the acceptor window
                "base": base[bj].cpu(), "base_corr": base_corr[bj].cpu(), "dtok": int(dtok[bj]),
                "co_ids": co["input_ids"][bj].cpu(),
                "attn_mask": cl["attention_mask"][bj].cpu() if "attention_mask" in cl else None,
                "m_cl": float(m_cl[bj]), "m_co": float(m_co[bj]),
                "gene": x.gene_id, "strand": x.strand,
                "coding": int(getattr(x, "is_coding", getattr(x, "coding", 0)) or 0),
                "pdist": abs(x.next_acceptor_pos - x.donor_pos),
                "decoy": matched_decoy_donor(x),
            })
        print(f"  capture {min(i + task.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()
    return E, lin_err


# --------------------------------------------------------------------------------------
# Cross-example matching: source != target gene, native-margin caliper, strand+coding exact.
# --------------------------------------------------------------------------------------
def match_pairs(E, rng, margin_caliper=0.75, pdist_caliper=200, mismatch=False,
                scramble=False, decoy=False):
    """Return a list of matched cross-example controls, gene-disjoint.

    Candidates are constrained on strand, coding status, and target-partner distance
    (|Δpdist| ≤ pdist_caliper); the source is then chosen by native donor margin: nearest
    within the margin caliper for the primary/scramble/decoy cohorts, or the MAXIMALLY
    mismatched candidate for the MF2 margin-mismatched stratum (mismatch=True)."""
    order = list(range(len(E)))
    pairs = []
    for i in order:
        cands = [j for j in order
                 if E[j]["gene"] != E[i]["gene"]
                 and E[j]["strand"] == E[i]["strand"]
                 and E[j]["coding"] == E[i]["coding"]
                 and abs(E[j]["pdist"] - E[i]["pdist"]) <= pdist_caliper]
        if not cands:
            continue
        dm = np.array([abs(E[j]["m_cl"] - E[i]["m_cl"]) for j in cands])
        if mismatch:                                   # MF2 margin-mismatched stratum: FAR in margin
            j = cands[int(np.argmax(dm))]
        else:                                          # primary: matched within caliper, nearest
            within = [c for c, d in zip(cands, dm) if d <= margin_caliper]
            pool = within or [cands[int(np.argmin(dm))]]
            j = pool[rng.integers(len(pool))]
        pairs.append({"t": i, "s": j, "scramble": scramble, "decoy": decoy})
    return pairs


def _stable_seed(key: str) -> int:
    """Deterministic 32-bit seed from a string (feature-permutation reproducibility)."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def _payload_edit(E, p, dev, alpha):
    """dr [4,H] = alpha * (source_clean_payload - target_corrupt_payload) at target's window.

    Scramble control (pre-committed redesign, 2026-07-23): the prereg N-scr asks whether transfer is
    specific to the payload's *content* vs any equal-norm state of matched geometry. The shipped
    version permuted the 4 WINDOW POSITIONS -- 4 adjacent acceptor tokens barely differ, so it left the
    payload essentially intact (empirically R_scr~=0.978, no collapse) AND never actually fired (its
    scramble_rng argument was never passed). The corrected control permutes the HIDDEN-FEATURE
    COORDINATES of the source payload with a fixed per-source permutation, applied IDENTICALLY to all
    four window rows. This preserves the payload's norm and per-position energy exactly (so the
    delivered ||dV|| and the injection geometry are matched) while destroying the coordinate structure
    that carries partner-specific content. A content-specific transfer must COLLAPSE under it; a
    generic "needs some acceptor-like state here" effect will survive -- which is the alternative
    hypothesis MF2 also probes, so the two controls agree by construction on a NULL. Deterministic per
    source gene (no run-to-run drift); the same permutation is reused across the dose grid."""
    src = E[p["s"]]["r3cl"].clone()                    # [4, H]
    if p.get("scramble"):
        g = np.random.default_rng(_stable_seed(E[p["s"]]["gene"]))
        perm = torch.as_tensor(g.permutation(src.shape[1]), dtype=torch.long)  # feature-dim perm
        src = src[:, perm]                             # SAME coordinate perm on every window row
    dr = (src - E[p["t"]]["r3co"]) * alpha
    return dr.to(dev)


def _corrupt_source_edit(E, p, dev, alpha):
    """MF2 control: inject the SOURCE's OWN corrupt payload (partner destroyed) -> dr differs from
    clean-source only in partner-intactness."""
    dr = (E[p["s"]]["r3co"] - E[p["t"]]["r3co"]) * alpha
    return dr.to(dev)


def inject_margins(ad, task, E, pairs, dev, off, Wv, edit_fn, alpha):
    """Deliver edit_fn's payload through each target's clean address; return per-pair donor margins."""
    out, genes_t, genes_s = [], [], []
    B = task.batch
    for s in range(0, len(pairs), B):
        chunk = pairs[s:s + B]
        tidx = [p["t"] for p in chunk]
        corr = {"input_ids": torch.stack([E[i]["co_ids"] for i in tidx]).to(dev)}
        if E[0]["attn_mask"] is not None:
            corr["attention_mask"] = torch.stack([E[i]["attn_mask"] for i in tidx]).to(dev)
        dtok_use = [(E[p["t"]]["decoy"] + off) if p.get("decoy") and E[p["t"]]["decoy"] is not None
                    else E[p["t"]]["dtok"] for p in chunk]
        dtok = torch.tensor(dtok_use, device=dev)
        base = torch.stack([E[i]["base"] for i in tidx]).to(dev)
        Ap = torch.stack([E[i]["Ap"] for i in tidx]).to(dev)
        dr = torch.stack([edit_fn(E, p, dev, alpha) for p in chunk])   # [b,4,H]
        dV = deliver(Wv, dr)                                           # [b,4,Dh]
        od = head_output(base, Ap, dV)                                 # [b,Dh]
        with inject_capture(ad, corr, dtok, od) as o_inj:
            m = task.margin(o_inj.logits.float(), [E[p["t"]]["dtok"] - off for p in chunk])
        out += m.cpu().tolist()
        genes_t += [E[p["t"]]["gene"] for p in chunk]
        genes_s += [E[p["s"]]["gene"] for p in chunk]
    return np.asarray(out), genes_t, genes_s


# --------------------------------------------------------------------------------------
# MF5: two-way source-gene x target-gene Poisson-multiplier dyadic bootstrap for a ratio of means.
# --------------------------------------------------------------------------------------
def dyadic_ratio_ci(num, dnm, gt, gs, seed=SEED, B=2000, lo=2.5, hi=97.5):
    rng = np.random.default_rng(seed)
    genes = np.unique(np.concatenate([np.asarray(gt), np.asarray(gs)]))
    gx = {g: k for k, g in enumerate(genes)}
    ti = np.array([gx[g] for g in gt]); si = np.array([gx[g] for g in gs])
    num = np.asarray(num, float); dnm = np.asarray(dnm, float)
    est = float(num.sum() / dnm.sum()) if abs(dnm.sum()) > 1e-9 else float("nan")
    boot = np.empty(B)
    for b in range(B):
        w = rng.poisson(1.0, len(genes))
        pw = w[ti] * w[si]                          # two-way multiplier: reused as source AND target
        d = float(np.sum(pw * dnm))
        boot[b] = float(np.sum(pw * num) / d) if abs(d) > 1e-9 else np.nan
    boot = boot[~np.isnan(boot)]
    return est, float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/b3_cross_example_transduction.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out
    if out.exists() and not args.force and not args.smoke:
        raise SystemExit(f"{out} exists; pass --force to overwrite (results are immutable).")
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off
    Wv = value_weight_head(ad).to(dev)
    rng = np.random.default_rng(SEED)

    # PILOT/CONFIRM draw from val; PILOT is analytically a DISCOVERY-split sizing exercise (MF3).
    n = min(args.n, 24) if args.smoke else args.n
    split = "train" if args.pilot else args.split
    X, _ = load_examples(args.gtf, args.fasta, split, n, task=task,
                         tag=f"b3-{split}", min_effect=None)
    E, lin_err = capture_examples(ad, task, X, dev, off, Wv)
    if len(E) < 16:
        raise RuntimeError(f"only {len(E)} scorable examples; raise --n.")

    # ---- mechanics memorials (must pass at smoke) ----
    ckpt0 = float(next(ad.model.parameters()).abs().sum())
    pairs = match_pairs(E, rng)
    m_co_t = np.array([E[p["t"]]["m_co"] for p in pairs])
    # recon-fidelity memorial (R-b3fix1): the OLD "empty->corrupt" memorial injected the CLEAN-address
    # object A+ . V- (base) and required it to equal the natural corrupt margin m_co. That identity is
    # INCOMPATIBLE with MF1: the estimand deliberately delivers through the target's CLEAN address A+,
    # but the natural corrupt margin is produced by the CORRUPT address A- (corruption reallocates
    # L3H1's attention row, |A+ - A-| ~ 0.95, per C35). So A+ . V- != m_co by design. The genuine
    # plumbing identity is A- . V- == m_co (exact); we verify that here. value_linearity remains the
    # on-path memorial for the delivered edit.
    recon_err = 0.0
    for s0 in range(0, len(E), task.batch):
        chunk = E[s0:s0 + task.batch]
        corr = {"input_ids": torch.stack([e["co_ids"] for e in chunk]).to(dev)}
        if chunk[0]["attn_mask"] is not None:
            corr["attention_mask"] = torch.stack([e["attn_mask"] for e in chunk]).to(dev)
        dtok = torch.tensor([e["dtok"] for e in chunk], device=dev)
        od = torch.stack([e["base_corr"] for e in chunk]).to(dev)
        with inject_capture(ad, corr, dtok, od) as o_inj:
            m = task.margin(o_inj.logits.float(), [e["dtok"] - off for e in chunk]).cpu().numpy()
        recon_err = max(recon_err, float(np.abs(m - np.array([e["m_co"] for e in chunk])).max()))
    ckpt1 = float(next(ad.model.parameters()).abs().sum())
    memorials = {
        "value_linearity": lin_err,                         # deliver(W_V, dr) == true value edit
        "recon_fidelity_gap": recon_err,                    # inject(A- . V-) == m_co (natural donor output)
        "checkpoint_unmutated": abs(ckpt1 - ckpt0),
        "n_examples": len(E), "n_pairs": len(pairs),
    }
    mem_ok = (lin_err < 1e-3) and (recon_err < 1e-2) and (abs(ckpt1 - ckpt0) < 1e-6)

    # ---- calibration: planted-positive (self-transfer recovers) + sign-permutation FPR ----
    # (R-b3fix2) The OLD FPR null used match_pairs(SEED+7) -- a second *matched* cohort (verified: ~54%
    # identical sources, all gene-disjoint AND margin-matched). Re-matching preserves the per-source
    # clean-vs-corrupt contrast, so it can NEVER null a clean-source - corrupt-source difference
    # estimand -> permuted_includes_0 was structurally False. A valid dyadic FPR null instead removes
    # the SIGNED structure of the per-pair contrast (Rademacher sign flip on xfer); the ratio-of-means
    # dyadic CI must then cover 0. Specificity (that the effect needs the real payload/position) is
    # still enforced downstream by scramble + decoy + the MF2 margin-mismatched stratum.
    m_self, _, _ = inject_margins(ad, task, E, [{"t": i, "s": i} for i in range(len(E))],
                                  dev, off, Wv, _payload_edit, 1.0)
    self_gain = float(np.mean(m_self - np.array([E[i]["m_co"] for i in range(len(E))])))
    m_pair_cl, gpt, gps = inject_margins(ad, task, E, pairs, dev, off, Wv, _payload_edit, 1.0)
    m_pair_co, _, _ = inject_margins(ad, task, E, pairs, dev, off, Wv, _corrupt_source_edit, 1.0)
    xfer_real = m_pair_cl - m_pair_co
    denom_c = np.maximum(m_self - m_co_t[:len(m_self)], 1e-6)[:len(xfer_real)]
    signs = np.random.default_rng(SEED + 7).choice([-1.0, 1.0], size=len(xfer_real))
    _, plo, phi = dyadic_ratio_ci(signs * xfer_real, denom_c, gpt[:len(xfer_real)], gps[:len(xfer_real)],
                                  B=min(args.n_boot, 500))
    calib = {"planted_positive_self_gain": self_gain,
             "permuted_fpr_ci": [plo, phi], "permuted_includes_0": bool(plo <= 0 <= phi)}
    calib_ok = (self_gain > SESOI_M) and calib["permuted_includes_0"]

    result = {"mode": "smoke" if args.smoke else ("pilot" if args.pilot else "confirm"),
              "split": split, "memorials": memorials, "memorials_pass": mem_ok,
              "calibration": calib, "calibration_pass": calib_ok,
              "sesoi_R": SESOI_R, "sesoi_M": SESOI_M, "dose_grid": list(DOSE_GRID)}

    if args.smoke:
        result["verdict"] = "SMOKE_" + ("PASS" if (mem_ok and calib_ok) else "FAIL")
        _write(out, result, args, smoke=True)
        print(json.dumps({k: result[k] for k in ("verdict", "memorials", "calibration")}, indent=2))
        return

    # ---- primary transfer + dose (MF2 estimand: clean-source minus corrupt-source) ----
    self_num = m_self - m_co_t[:len(m_self)]                 # denominator: own-payload gain
    dose = {}
    for a in DOSE_GRID:
        mc, gt, gs = inject_margins(ad, task, E, pairs, dev, off, Wv, _payload_edit, a)
        mk, _, _ = inject_margins(ad, task, E, pairs, dev, off, Wv, _corrupt_source_edit, a)
        xfer = mc - mk                                      # clean-source > corrupt-source (backdoor-safe)
        est, clo, chi = dyadic_ratio_ci(xfer, self_num[:len(xfer)], gt, gs, B=args.n_boot)
        dose[f"alpha_{a}"] = {"M_xfer_mean": float(np.mean(xfer)),
                              "R_xfer": est, "dyadic_ci": [clo, chi],
                              "one_way_gboot": list(gboot_ratio(xfer, self_num[:len(xfer)], gt,
                                                                n=args.n_boot))}
    # one-sided positive dose slope over the grid (MF4)
    xs = np.array(DOSE_GRID); ys = np.array([dose[f"alpha_{a}"]["M_xfer_mean"] for a in DOSE_GRID])
    beta = float(np.polyfit(xs, ys, 1)[0])
    monotone = bool(np.all(np.diff(ys) >= -SESOI_M))
    result["dose"] = dose; result["beta_dose"] = beta; result["monotone"] = monotone

    if args.pilot:
        # MF3: report the numerator + dyadic HW; re-derive the N needed for HW(R_xfer) < SESOI_R.
        hw = 0.5 * (dose["alpha_1.0"]["dyadic_ci"][1] - dose["alpha_1.0"]["dyadic_ci"][0])
        result["pilot"] = {"M_xfer_at_1": dose["alpha_1.0"]["M_xfer_mean"],
                           "R_xfer_at_1": dose["alpha_1.0"]["R_xfer"],
                           "dyadic_half_width": hw,
                           "n_pairs_used": len(pairs),
                           "N_req_for_SESOI": (None if hw <= 0 else
                                               int(np.ceil(len(pairs) * (hw / SESOI_R) ** 2)))}
        result["verdict"] = "PILOT_COMPLETE_no_freeze"
        _write(out, result, args)
        print(json.dumps(result["pilot"], indent=2))
        return

    # ---- specificity controls (scramble/decoy must COLLAPSE for a POSITIVE) ----
    sp = {}
    for name, kw in (("scramble", {"scramble": True}), ("decoy", {"decoy": True})):
        cp = match_pairs(E, np.random.default_rng(SEED + 3), **kw)
        mc, gt, gs = inject_margins(ad, task, E, cp, dev, off, Wv, _payload_edit, 1.0)
        mk, _, _ = inject_margins(ad, task, E, cp, dev, off, Wv, _corrupt_source_edit, 1.0)
        est, clo, chi = dyadic_ratio_ci(mc - mk, self_num[:len(mc)], gt, gs, B=args.n_boot)
        sp[name] = {"R_xfer": est, "dyadic_ci": [clo, chi], "collapses": bool(clo <= SESOI_R)}
    result["specificity"] = sp

    # ---- MF2 margin-mismatched stratum (confound gate): a source matched on strand/coding/pdist
    # but FAR in native margin. If the "transfer" were a margin-gap artifact rather than
    # partner-specific payload, this stratum would reproduce (or exceed) the matched transfer.
    # POSITIVE requires the matched primary to NOT be dominated by the mismatched stratum.
    mm = match_pairs(E, np.random.default_rng(SEED + 5), mismatch=True)
    mmc, mgt, mgs = inject_margins(ad, task, E, mm, dev, off, Wv, _payload_edit, 1.0)
    mmk, _, _ = inject_margins(ad, task, E, mm, dev, off, Wv, _corrupt_source_edit, 1.0)
    mm_est, mm_lo, mm_hi = dyadic_ratio_ci(mmc - mmk, self_num[:len(mmc)], mgt, mgs, B=args.n_boot)
    prim = dose["alpha_1.0"]
    mm_ok = bool(prim["dyadic_ci"][0] > mm_hi)     # matched primary CI-low exceeds mismatched CI-high
    result["margin_mismatched_stratum"] = {"R_xfer": mm_est, "dyadic_ci": [mm_lo, mm_hi],
                                            "matched_exceeds_mismatched": mm_ok,
                                            "note": "MF2 confound gate: transfer must not be a margin-gap artifact."}

    # ---- health gate ----
    result["mlm_health"] = float(task.mlm_health(X[:min(40, len(X))]))

    positive = (prim["dyadic_ci"][0] > SESOI_R and beta > 0 and monotone
                and all(sp[c]["collapses"] for c in sp)
                and mm_ok
                and calib["permuted_includes_0"] and result["mlm_health"] > 0.8)
    result["verdict"] = ("POSITIVE_transfer" if positive
                         else ("NULL_no_transfer" if prim["dyadic_ci"][1] < SESOI_R
                               else "INCONCLUSIVE_underpowered"))
    result["memorials_pass"] = mem_ok; result["calibration_pass"] = calib_ok
    _write(out, result, args)
    print(json.dumps({k: result.get(k) for k in
                      ("verdict", "beta_dose", "monotone", "specificity", "mlm_health")}, indent=2))


def _write(out: Path, result: dict, args, smoke: bool = False) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = out.stem + ("_smoke" if smoke else "")
    p = out.parent / (tag + ".json")
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    man = ExperimentManifest.create(tag, sys.argv, repo=REPO,
                                    notes="B3 cross-example transduction (red-team v2).")
    for src in (args.gtf, args.fasta):
        try:
            man.add_input(src)
        except OSError:
            pass
    man.add_output(p)
    man.write(out.parent / (tag + ".manifest.json"))


if __name__ == "__main__":
    main()
