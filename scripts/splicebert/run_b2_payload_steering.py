#!/usr/bin/env python
"""B2 -- payload semantics via site-choice steering (the C30 headline limitation).

Frozen design of record:
  studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_B2_PAYLOAD_STEERING.md
  redteam synthesis: studies/splicebert_partner_selection/redteam/REDTEAM_SYNTHESIS_B2.md
This runner implements the red-team's runnable remnant. Read the disposition before running:

  The SEMANTICS arm that would actually *name* the payload (close C30) is PRE-DECLARED possibly
  NON-IDENTIFIABLE on a frozen checkpoint at MPS budget -- a diff-in-means choice direction between
  two DIFFERENT-POSITION acceptors is dominated by learned position-embedding geometry, which is the
  R20 trap ("looked like identity, was geometry") re-entering at the steering layer. Therefore B2 is
  staged behind a STRICT FIREWALL and only the first two rungs are confirmable here:

    EXISTENCE  (B2.1): is there a causal choice DIRECTION? Dose-graded residual injection
                       u_choice at the less-preferred site raises its native margin B_j(alpha).
    SPECIFICITY(B2.2): does it move the INTENDED paired site, partner-specifically (not a decoy,
                       not a random equal-norm direction)?
    SEMANTICS  (B2.3): does an EXTERNAL variable (local MaxEnt V2b) predict the per-example dose
                       response? --> the C30 target. RUN ONLY IF --semantics AND existence+
                       specificity both passed; reported with the pre-declared non-identifiability
                       caveat and NEVER promoted to a MaxEnt label without the external predictor
                       clearing its dyadic CI. Labels are model-native behaviour, never MaxEnt.

  The firewall is the whole point: a direction that moves behaviour is NOT necessarily the concept
  direction. Existence/specificity establish only that a causal partner-specific choice axis exists;
  they do not name it. If you skip the firewall you re-earn R20/R22.

DATA PRIMITIVE: alt-5'SS donor / alt-3'SS acceptor competition CHOICE SETS (two same-type sites
  competing for one shared partner) -- the direct analogue of the unresolved payload. Model-native
  labels: the preferred site is the one with the higher native margin B; steering targets the lower.

MODES
  --smoke      : mechanics memorials (M-alpha0 zero-dose identity, M-dose monotonicity on a PLANTED
                 direction, checkpoint-unmutated) + label-generalization sanity, tiny-n. NO verdict.
  --pilot      : EXISTENCE numerator + its dyadic half-width on DISCOVERY, then STOP (size the run).
  (default)    : EXISTENCE + SPECIFICITY confirmatory (val; chr8/9 sealed, NOT opened).
  --semantics  : additionally run B2.3 -- ONLY meaningful if existence+specificity pass; otherwise it
                 self-reports NON_IDENTIFIABLE per the pre-declaration.

Model/data required (SpliceBERT.1024nt + hg38 + GENCODE v50); run from the main checkout or symlink
data/models in (gitignored). Reuses competition_seq/behavioral_pilot idioms + a general residual
injection hook (patch_heads token_mask on the residual add); no new kernel.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest                                # noqa: E402
from gfmcd.splicing.gtf import load_transcripts                                # noqa: E402
from gfmcd.splicing.competition import (                                       # noqa: E402
    CompetitionConfig, build_competition_index, enumerate_competition_examples)
from gfmcd.splicing.competition_seq import build_window                        # noqa: E402
from gfmcd.splicing.dataset import PyfaidxGenome                               # noqa: E402
from gfmcd.splicing.circuit import (                                           # noqa: E402
    DEFAULT_FASTA, DEFAULT_MODEL, load_adapter, pick_device)

DEFAULT_GTF = "/Users/cberg/Research/data/genomes/gencode.v50.annotation.gtf.gz"
L_INJECT = 3                       # residual layer for u_choice injection (L3 = payload/address locus)
ACC_OFF = (-3, -2, -1, 0)          # four-token site window
DOSE_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
SESOI_B = 0.05                     # smallest native-margin choice shift worth calling causal
SEED = 0
STD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def _order_key(eid: str) -> int:
    return int.from_bytes(hashlib.sha256(eid.encode()).digest()[:8], "big")


# --------------------------------------------------------------------------------------
# Model-native site margin: mask the core base of a donor (G of GT) or acceptor (G of AG),
# score logit(true) - max(others). Mirrors CouplingTask.margin / behavioral_pilot scoring.
# --------------------------------------------------------------------------------------
class SiteScorer:
    def __init__(self, ad, batch=8):
        self.ad = ad; self.batch = batch
        self.tok = ad.tokenizer; self.off = ad.prefix_tokens
        self.mask_id = self.tok.mask_token_id
        self.bid = {b: self.tok.convert_tokens_to_ids(b) for b in "ACGT"}

    def _encode(self, seqs, core_pos):
        enc = self.tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True).to(self.ad.device)
        for j, c in enumerate(core_pos):
            enc["input_ids"][j, c + self.off] = self.mask_id
        return enc

    def margin(self, logits, core_pos, true_base="G"):
        i = torch.arange(len(core_pos), device=logits.device)
        p = torch.tensor([c + self.off for c in core_pos], device=logits.device)
        v = logits[i, p]
        others = [b for b in "ACGT" if b != true_base]
        alt = torch.stack([v[:, self.bid[b]] for b in others], -1).max(-1).values
        return v[:, self.bid[true_base]] - alt

    def score(self, seqs, core_pos, ctx=None):
        ctx = ctx or _noop
        out = []
        with torch.no_grad():
            for s in range(0, len(seqs), self.batch):
                sub, cp = seqs[s:s + self.batch], core_pos[s:s + self.batch]
                enc = self._encode(sub, cp)
                with ctx():
                    out += self.margin(self.ad.model(**enc).logits.float(), cp).tolist()
        return np.asarray(out)


@contextmanager
def _noop():
    yield


@contextmanager
def inject_residual(ad, layer, token_idx, vec):
    """Add `vec` [B,H] to the layer-`layer` residual output at `token_idx` [B] for the forward pass.
    General dose hook: alpha=0 (vec all-zero) is an exact identity (M-alpha0)."""
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        b = torch.arange(h.shape[0], device=h.device)
        h[b, token_idx] = h[b, token_idx] + vec.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    handle = ad.encoder_layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------------------
# Choice-set construction: build a windowed competition example, locate the two same-type
# sites (target/competitor) and read each one's native margin -> model-native preferred label.
# --------------------------------------------------------------------------------------
def load_choice_sets(gtf, fasta, split, n, cfg, seed=SEED):
    tx = [t for t in load_transcripts(gtf) if t.chrom in STD_CHROMS]
    idx = build_competition_index(tx)
    exs = [e for e in enumerate_competition_examples(idx, cfg)
           if e.usable and not e.is_tandem and e.split == split]
    exs.sort(key=lambda e: _order_key(e.example_id))
    genome = PyfaidxGenome(fasta)
    sets = []
    for e in exs:
        w = build_window(e, genome, cfg, offset_seed=seed)
        if w is None:
            continue
        sets.append(w)
        if len(sets) >= n:
            break
    return sets


def _core(win_pos, is_donor):
    """Core masked base: donor = the G at GT (win_pos), acceptor = the G at AG (win_pos-1)."""
    return win_pos if is_donor else win_pos - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--semantics", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/b2_payload_steering.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out
    if out.exists() and not args.force and not args.smoke:
        raise SystemExit(f"{out} exists; pass --force to overwrite (results are immutable).")
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    sc = SiteScorer(ad, batch=args.batch)
    off = sc.off; H = ad.hidden_size
    rng = np.random.default_rng(SEED)
    cfg = CompetitionConfig()

    split = "train" if args.pilot else args.split
    n = min(args.n, 24) if args.smoke else args.n
    sets = load_choice_sets(args.gtf, args.fasta, split, n, cfg)
    if len(sets) < 12:
        raise RuntimeError(f"only {len(sets)} choice sets on {split}; raise --n.")

    # ---- per-set: native margins at target & competitor -> model-native preferred label + L3 state ----
    def capture(sets):
        rows = []
        for w in sets:
            is_d = w.target_is_donor
            tb = "G"                                        # both donor GT and acceptor AG mask a G
            tcore = _core(w.target_win, is_d)
            ccore = _core(w.competitor_win, w.competitor_is_donor)
            enc = sc._encode([w.seq, w.seq], [tcore, ccore])
            o = ad.model(**enc, output_hidden_states=True)
            mt = float(sc.margin(o.logits.float()[:1], [tcore], tb)[0])
            mc = float(sc.margin(o.logits.float()[1:], [ccore], tb)[0])
            # L3 residual state at each site's window (the payload locus), for the choice direction
            r3 = o.hidden_states[L_INJECT]
            twin = [tcore + off + k for k in ACC_OFF]
            cwin = [ccore + off + k for k in ACC_OFF]
            rows.append({"w": w, "is_d": is_d, "tcore": tcore, "ccore": ccore,
                         "mt": mt, "mc": mc, "gene": w.gene_id,
                         "r_t": r3[0, twin].mean(0).cpu(), "r_c": r3[1, cwin].mean(0).cpu()})
        return rows

    R = capture(sets)
    # model-native preference: preferred = higher native margin; steer the LESS-preferred toward it
    for r in R:
        r["pref_is_target"] = r["mt"] >= r["mc"]
        r["r_pref"] = r["r_t"] if r["pref_is_target"] else r["r_c"]
        r["r_less"] = r["r_c"] if r["pref_is_target"] else r["r_t"]
        r["m_less"] = min(r["mt"], r["mc"])
        r["core_less"] = r["ccore"] if r["pref_is_target"] else r["tcore"]

    # ---- choice direction: cross-fitted, gene-disjoint diff-in-means (pref - less) ----
    genes = sorted({r["gene"] for r in R})
    fold = {g: (_order_key(g) % 2) for g in genes}
    def choice_dir(train_rows):
        d = torch.stack([r["r_pref"] - r["r_less"] for r in train_rows]).mean(0)
        return d / (d.norm() + 1e-8)
    u0 = {f: choice_dir([r for r in R if fold[r["gene"]] != f]) for f in (0, 1)}
    gap = float(torch.stack([(r["r_pref"] - r["r_less"]).norm() for r in R]).mean())  # norm reference

    # ---- memorials (must pass at smoke) ----
    ckpt0 = float(next(ad.model.parameters()).abs().sum())
    # M-alpha0: alpha=0 injection reproduces the un-injected margin exactly
    w0 = R[0]; z = torch.zeros(1, H, device=dev)
    tok0 = torch.tensor([w0["core_less"] + off], device=dev)
    m_base = sc.margin(ad.model(**sc._encode([w0["w"].seq], [w0["core_less"]])).logits.float(),
                       [w0["core_less"]])[0]
    with inject_residual(ad, L_INJECT, tok0, z):
        m_inj0 = sc.margin(ad.model(**sc._encode([w0["w"].seq], [w0["core_less"]])).logits.float(),
                           [w0["core_less"]])[0]
    alpha0_gap = float(abs(m_base - m_inj0))
    # M-dose on a PLANTED direction: inject the exact (r_pref - r_less) full-state -> B_less must rise
    planted = []
    for r in R[:min(8, len(R))]:
        d = (r["r_pref"] - r["r_less"]).to(dev)
        tok = torch.tensor([r["core_less"] + off], device=dev)
        vals = []
        for a in DOSE_GRID:
            with inject_residual(ad, L_INJECT, tok, a * d[None, :]):
                vals.append(float(sc.margin(
                    ad.model(**sc._encode([r["w"].seq], [r["core_less"]])).logits.float(),
                    [r["core_less"]])[0]))
        planted.append(vals[-1] - vals[0])
    ckpt1 = float(next(ad.model.parameters()).abs().sum())
    m_dose_gain = float(np.mean(planted))
    memorials = {"alpha0_gap": alpha0_gap, "m_dose_planted_gain": m_dose_gain,
                 "checkpoint_unmutated": abs(ckpt1 - ckpt0),
                 "n_sets": len(R), "n_genes": len(genes), "state_gap_norm": gap}
    mem_ok = (alpha0_gap < 1e-4) and (m_dose_gain > SESOI_B) and (abs(ckpt1 - ckpt0) < 1e-6)

    # M-label: discovery B-ordering predicts confirm-split ordering (labels are model-native, not MaxEnt)
    d_rows = [r for r in R if fold[r["gene"]] == 0]; c_rows = [r for r in R if fold[r["gene"]] == 1]
    label_ok = None
    if d_rows and c_rows:
        sign_agree = np.mean([(r["mt"] >= r["mc"]) for r in c_rows])   # confirm split consistency proxy
        label_ok = float(sign_agree)

    result = {"mode": "smoke" if args.smoke else ("pilot" if args.pilot else "confirm"),
              "split": split, "layer": L_INJECT, "dose_grid": list(DOSE_GRID), "sesoi_B": SESOI_B,
              "memorials": memorials, "memorials_pass": mem_ok, "label_generalization": label_ok,
              "firewall": "EXISTENCE->SPECIFICITY->SEMANTICS; semantics pre-declared possibly non-identifiable (R20)"}

    if args.smoke:
        result["verdict"] = "SMOKE_" + ("PASS" if mem_ok else "FAIL")
        _write(out, result, args, smoke=True)
        print(json.dumps({k: result[k] for k in ("verdict", "memorials", "label_generalization")}, indent=2))
        return

    # ---- B2.1 EXISTENCE: dose-slope of B_less under the cross-fitted choice direction ----
    def dose_curve(rows, direction_fn, scale=1.0):
        per = defaultdict(list); slopes = []
        for r in rows:
            u = direction_fn(r).to(dev) * scale
            tok = torch.tensor([r["core_less"] + off], device=dev)
            vals = []
            for a in DOSE_GRID:
                with inject_residual(ad, L_INJECT, tok, a * gap * u[None, :]):
                    vals.append(float(sc.margin(
                        ad.model(**sc._encode([r["w"].seq], [r["core_less"]])).logits.float(),
                        [r["core_less"]])[0]))
            slopes.append(np.polyfit(DOSE_GRID, vals, 1)[0]); per[r["gene"]].append(vals[-1] - vals[0])
        return per, np.asarray(slopes)

    def u_for(r):
        return u0[fold[r["gene"]]]                          # cross-fitted: use the OTHER fold's direction

    per_exist, slopes = dose_curve(R, u_for)
    exist = _gene_ci([(g, v) for g, vs in per_exist.items() for v in vs], SESOI_B, args.n_boot)
    result["existence"] = {**exist, "mean_slope": float(np.mean(slopes))}

    if args.pilot:
        hw = 0.5 * (exist["ci"][1] - exist["ci"][0])
        result["pilot"] = {"exist_gain": exist["mean"], "dyadic_half_width": hw,
                           "n_sets": len(R),
                           "N_req_for_SESOI": (None if hw <= 0 else
                                               int(np.ceil(len(R) * (hw / SESOI_B) ** 2)))}
        result["verdict"] = "PILOT_COMPLETE_no_freeze"
        _write(out, result, args)
        print(json.dumps(result["pilot"], indent=2))
        return

    exist_pos = (exist["ci"][0] > 0 and exist["mean"] >= SESOI_B)

    # ---- B2.2 SPECIFICITY: random-direction control + decoy-position control must NOT reproduce it ----
    def rand_dir(r):
        g = np.random.default_rng(_order_key(r["gene"]))
        v = torch.tensor(g.standard_normal(H), dtype=torch.float32)
        return v / (v.norm() + 1e-8)
    per_rand, _ = dose_curve(R, rand_dir)
    rand = _gene_ci([(g, v) for g, vs in per_rand.items() for v in vs], SESOI_B, args.n_boot)
    # decoy position: inject the choice direction at a non-site token (partner window) -> must collapse
    def u_decoy(r):
        return u0[fold[r["gene"]]]
    per_dec = defaultdict(list)
    for r in R:
        u = u_decoy(r).to(dev)
        tok = torch.tensor([r["w"].partner_win + off], device=dev)
        vals = []
        for a in (DOSE_GRID[0], DOSE_GRID[-1]):
            with inject_residual(ad, L_INJECT, tok, a * gap * u[None, :]):
                vals.append(float(sc.margin(
                    ad.model(**sc._encode([r["w"].seq], [r["core_less"]])).logits.float(),
                    [r["core_less"]])[0]))
        per_dec[r["gene"]].append(vals[-1] - vals[0])
    decoy = _gene_ci([(g, v) for g, vs in per_dec.items() for v in vs], SESOI_B, args.n_boot)
    result["specificity"] = {"random_direction": rand, "decoy_position": decoy,
                             "random_collapses": bool(rand["ci"][1] < SESOI_B),
                             "decoy_collapses": bool(decoy["ci"][1] < SESOI_B)}
    spec_ok = result["specificity"]["random_collapses"] and result["specificity"]["decoy_collapses"]

    result["existence_positive"] = exist_pos
    result["specificity_pass"] = spec_ok
    if not (exist_pos and spec_ok):
        result["verdict"] = ("NULL_no_choice_direction" if exist["ci"][1] < SESOI_B
                             else "INCONCLUSIVE_or_nonspecific")
    else:
        result["verdict"] = "EXISTENCE_SPECIFICITY_POSITIVE"

    # ---- B2.3 SEMANTICS (only if requested AND the core passed): pre-declared possibly non-identifiable ----
    if args.semantics:
        if not (exist_pos and spec_ok):
            result["semantics"] = {"status": "NOT_RUN_core_failed"}
        else:
            # per-example steering gain vs local MaxEnt V2b (external predictor). If the choice
            # direction is a position/distance axis (R20), V2b will NOT predict the gain.
            try:
                from maxentpy.maxent import load_matrix3, score3            # noqa: E402
                M3 = load_matrix3()
            except Exception:
                M3 = None
            gains, v2b, gt, gs = [], [], [], []
            for r in R:
                u = u_for(r).to(dev); tok = torch.tensor([r["core_less"] + off], device=dev)
                with inject_residual(ad, L_INJECT, tok, DOSE_GRID[-1] * gap * u[None, :]):
                    g = float(sc.margin(ad.model(**sc._encode([r["w"].seq], [r["core_less"]])).logits.float(),
                                        [r["core_less"]])[0]) - r["m_less"]
                a = r["core_less"]
                w23 = r["w"].seq[a - 20:a + 3]
                sv = (score3(w23, matrix=M3) if (M3 is not None and len(w23) == 23
                      and set(w23) <= set("ACGT")) else float("nan"))
                gains.append(g); v2b.append(sv); gt.append(r["gene"]); gs.append(r["gene"])
            gains = np.asarray(gains); v2b = np.asarray(v2b)
            ok = ~np.isnan(v2b)
            corr = float(np.corrcoef(gains[ok], v2b[ok])[0, 1]) if ok.sum() > 3 else float("nan")
            result["semantics"] = {
                "status": "REPORTED_with_nonidentifiability_caveat",
                "external_var": "V2b_local_MaxEnt",
                "corr_gain_vs_maxent": corr, "n_scored": int(ok.sum()),
                "caveat": ("A nonzero correlation does NOT license a MaxEnt label: the choice "
                           "direction may be a position/distance axis (R20). Naming requires the "
                           "external predictor's dyadic CI to exclude 0 AND a matched position control."),
            }

    result["memorials_pass"] = mem_ok
    _write(out, result, args)
    print(json.dumps({k: result.get(k) for k in
                      ("verdict", "existence", "specificity", "semantics")}, indent=2))


def _gene_ci(pairs, sesoi, n_boot):
    """Gene-clustered bootstrap CI + effect class (mirrors behavioral_pilot.gene_clustered_boot)."""
    by = defaultdict(list)
    for g, v in pairs:
        if not (isinstance(v, float) and np.isnan(v)):
            by[g].append(v)
    genes = sorted(by)
    if len(genes) < 3:
        return {"mean": float("nan"), "ci": [None, None], "class": "inconclusive", "n_genes": len(genes)}
    gm = np.array([np.mean(by[g]) for g in genes])
    rng = np.random.default_rng(SEED)
    boot = np.array([np.mean(gm[rng.integers(0, len(genes), len(genes))]) for _ in range(n_boot)])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    point = float(np.mean(gm))
    cls = ("material" if (lo > 0 or hi < 0) and abs(point) >= sesoi
           else "small-but-detectable" if (lo > 0 or hi < 0)
           else "bounded-null" if (lo > -sesoi and hi < sesoi) else "inconclusive")
    return {"mean": point, "ci": [lo, hi], "class": cls, "n_genes": len(genes)}


def _write(out: Path, result: dict, args, smoke: bool = False) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = out.stem + ("_smoke" if smoke else "")
    p = out.parent / (tag + ".json")
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    man = ExperimentManifest.create(tag, sys.argv, repo=REPO,
                                    notes="B2 payload site-choice steering (existence/specificity; semantics firewalled).")
    for src in (args.gtf, args.fasta):
        try:
            man.add_input(src)
        except OSError:
            pass
    man.add_output(p)
    man.write(out.parent / (tag + ".manifest.json"))


if __name__ == "__main__":
    main()
