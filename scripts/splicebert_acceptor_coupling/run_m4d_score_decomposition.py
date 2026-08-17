#!/usr/bin/env python
"""M4d -- is M4c's class asymmetry a real score shift, or just softmax curvature?

M4c ablated each early head and re-read how much of ``L3H1``'s donor attention MASS survived,
finding the minor class losing 20-38% against canonical's 0-9%.  That was taken as evidence
those heads carry class-specific content.  It does not establish it.  The two arms start from
different baselines (``GC`` 0.1702, ``GT`` 0.4669) and mass is a softmax output, so the SAME
change in the underlying attention scores produces different fractional mass changes purely
because the arms sit at different points on the curve.  Call that the headroom account.

Working on the log-odds scale removes the confound exactly.  With ``D`` the donor's two key
columns and ``V`` every valid column,

    p = sum_D exp(s) / sum_V exp(s) = sigmoid( LSE(s_D) - LSE(s_{V\\D}) ) = sigmoid(m)

so the score margin ``m`` is an exact reparameterisation of the mass, and an ablation's effect
is the additive shift ``delta_m``.  Two questions then separate cleanly:

    PRIMARY    does delta_m itself differ by class, or only the baseline it acts on?
               contrast delta_m(GC) - delta_m(GT), gene-clustered, per matched pair

    ADJUDICATE give the canonical arm the MINOR arm's per-pair margin shifts and re-read its
               fractional drop.  What that recovers is score shift; what it leaves behind is
               curvature.  The two are additive in the log ratio (``score_margin``).

A bound is available before the run and is asserted as a memorial: at these baselines the
curvature-only fractional-drop ratio is at most ``(1-p_GC)/(1-p_GT) = 1.556``, approached as
the shift goes to zero and falling from there.  M4c's largest observed ratio was 17.4.  So the
headroom account is already quantitatively insufficient by an order of magnitude, and the run
exists to confirm that on per-example data and to deliver the positive result.

SECONDARY, and the reason pre-softmax scores are needed at all: ``delta_m = delta(LSE donor) -
delta(LSE others)``.  The softmax is shift-invariant, so probabilities pin down only that
difference.  Capturing the scores separates "the donor's key got worse" from "every competitor
got better" -- the first says the ablated head helped BUILD the donor's representation, which
is the M4c reading; the second says it changed the query or raised the noise floor.

Development tier, train split; chr8/9 never read and the validation reserve is not spent.
Descriptive geometry plus an intervention on attention only -- no margin, route, mediation or
biological claim follows.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.circuits.hf_bert import capture_attention_scores, edit_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _dataset, eligible,
    load_adapter, pick_device,
)
from gfmcd.splicing.coupling import planted_donor_motifs  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    COMPARATOR_MOTIF, MATCHED_OPERATOR, TREATMENT_MOTIF, assert_seal_intact,
    flank_and_support_key, match_comparators, train_chromosomes,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout  # noqa: E402
from gfmcd.splicing.score_margin import (  # noqa: E402
    decompose_asymmetry, fractional_drop, margin_from_scores, sigmoid,
)

TAG = "m4d-score-decomposition-v1"
READER_LAYER, READER_HEAD = 3, 1
SEED, N_BOOT = 0, 2000

#: M4c's recorded baselines, re-derived here and required to agree. A drifted cohort would
#: otherwise produce a decomposition of a different experiment's asymmetry.
M4C_BASELINE = {"GC": 0.17015116387624504, "GT": 0.46690675535802706}
#: Tight on purpose. The cohort is deterministic and mean-ablation is deterministic, so the
#: only admissible gap is the ~1e-6 between the model's own softmax and this runner's
#: recomputation of it. Anything larger means a different cohort, not a different rounding.
M4C_BASELINE_TOL = 1e-4

PROBES: list[tuple[int, int, str]] = [
    (0, 4, "class-specific"), (0, 5, "class-specific"), (0, 14, "class-specific"),
    (0, 3, "class-specific"), (1, 13, "class-specific"), (1, 6, "class-specific"),
    (1, 12, "shared load-bearing"), (2, 15, "shared load-bearing"),
    (0, 0, "contrast, inert"), (5, 7, "contrast, inert, late"),
]


def read_rows(ad, task, rows, off, batch, ctx_factory):
    """Per-example ``(mass, margin, donor_lse, other_lse)`` for L3H1's donor attention."""
    mass, margin, donor_lse, other_lse = [], [], [], []
    for i in range(0, len(rows), batch):
        b = rows[i:i + batch]
        pos = [acceptor_readout(x) for x in b]
        enc = task.encode([x.sequence for x in b], pos)
        with capture_attention_scores(ad, READER_LAYER, head=READER_HEAD) as box:
            with torch.no_grad(), ctx_factory():
                ad.model(**enc)
        scores = box.require().float().cpu().numpy()
        lens = enc["attention_mask"].sum(-1).tolist()
        for j, x in enumerate(b):
            r, d = pos[j] + off, x.donor_pos + off
            m, t, o = margin_from_scores(scores[j, r], d, int(lens[j]), width=2)
            margin.append(m)
            donor_lse.append(t)
            other_lse.append(o)
            mass.append(float(sigmoid(m)))
    return (np.asarray(mass), np.asarray(margin),
            np.asarray(donor_lse), np.asarray(other_lse))


def gene_index(genes):
    by: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        by.setdefault(str(g), []).append(i)
    return by, list(by)


def gboot_mean(v, genes, *, n_draws=N_BOOT, seed=SEED):
    by, keys = gene_index(genes)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.integers(0, len(keys), len(keys))
        draws[i] = np.mean(v[[r for j in pick for r in by[keys[j]]]])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(v)), float(lo), float(hi)


def gboot_components(base, abl, dm_gc, genes, *, n_draws=N_BOOT, seed=SEED):
    """Gene-clustered CIs on the curvature / score-shift split.

    The whole decomposition is recomputed inside each draw rather than resampling a
    precomputed statistic, so the interval carries the covariance between the three
    fractional drops instead of assuming them independent.
    """
    by, keys = gene_index(genes)
    rng = np.random.default_rng(seed)
    curv, shift = [], []
    for _ in range(n_draws):
        pick = rng.integers(0, len(keys), len(keys))
        idx = [r for j in pick for r in by[keys[j]]]
        try:
            f_t = fractional_drop(base["GC"][idx], abl["GC"][idx])
            f_c = fractional_drop(base["GT"][idx], abl["GT"][idx])
            f_s = fractional_drop(base["GT"][idx],
                                  sigmoid(base["m_GT"][idx] - dm_gc[idx]))
        except ValueError:
            continue
        out = decompose_asymmetry(f_t, f_c, f_s)
        if out.verdict == "UNDEFINED":
            continue
        curv.append(out.curvature)
        shift.append(out.score_shift)
    if len(curv) < 0.5 * n_draws:
        return None                      # too many draws undefined to quote an interval
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]  # noqa: E731
    return {"curvature_ci": q(curv), "score_shift_ci": q(shift),
            "n_usable_draws": len(curv)}


def memorial_softmax_identity(ad, task, rows, off, batch):
    """sigma(margin) must reproduce the model's OWN reported attention mass.

    Two independent paths: the pre-hook recomputation this runner reads, and the model's
    ``output_attentions``.  If they disagree, every margin below is measuring something other
    than the attention the model actually used.  On the measurement path, and able to fail.
    """
    b = rows[:batch]
    pos = [acceptor_readout(x) for x in b]
    enc = task.encode([x.sequence for x in b], pos)
    with torch.no_grad():
        reported = ad.model(**enc, output_attentions=True).attentions[READER_LAYER]
    with capture_attention_scores(ad, READER_LAYER, head=READER_HEAD) as box:
        with torch.no_grad():
            ad.model(**enc)
    scores = box.require().float().cpu().numpy()
    lens = enc["attention_mask"].sum(-1).tolist()
    worst = 0.0
    for j, x in enumerate(b):
        r, d = pos[j] + off, x.donor_pos + off
        m, _, _ = margin_from_scores(scores[j, r], d, int(lens[j]), width=2)
        truth = float(reported[j, READER_HEAD, r, d:d + 2].sum())
        worst = max(worst, abs(float(sigmoid(m)) - truth))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="memorials only, on a handful of pairs; writes nothing")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None and not a.smoke:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    ad = load_adapter(a.model, pick_device(a.device), eager=True)
    task = CouplingTask(ad, batch=a.batch)
    off = ad.prefix_tokens

    D = _dataset(a.gtf, a.fasta, allowed_chromosomes=train_chromosomes())
    n_rows = assert_seal_intact(D)
    pool = lambda m: [x for x in eligible(D, "train", donor_motifs=(m,))  # noqa: E731
                      if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)]
    coh = match_comparators(pool(TREATMENT_MOTIF), pool(COMPARATOR_MOTIF), k=1,
                            key_fn=flank_and_support_key)
    gc, gt = coh.treatment, coh.comparators
    if a.smoke:
        gc, gt = gc[:a.batch * 2], gt[:a.batch * 2]
    genes = [x.gene_id for x in gc]
    print(f"pairs={len(gc)} genes={len(set(genes))} rows_scanned={n_rows} "
          f"reader=L{READER_LAYER}H{READER_HEAD}", flush=True)

    memorials: dict[str, dict] = {}

    worst = memorial_softmax_identity(ad, task, gc, off, a.batch)
    memorials["softmax_identity"] = {
        "statistic": "max |sigmoid(margin) - reported donor mass|",
        "value": worst, "threshold": 1e-5, "pass": bool(worst < 1e-5),
        "forced_by_architecture": False,
    }
    print(f"[memorial] softmax identity  max|diff| = {worst:.3e}  "
          f"{'PASS' if worst < 1e-5 else 'FAIL'}", flush=True)

    base = {}
    b_mass, b_margin, b_don, b_oth = {}, {}, {}, {}
    for arm, rows in (("GC", gc), ("GT", gt)):
        m, mg, dn, ot = read_rows(ad, task, rows, off, a.batch, nullcontext)
        b_mass[arm], b_margin[arm], b_don[arm], b_oth[arm] = m, mg, dn, ot
    base = {"GC": b_mass["GC"], "GT": b_mass["GT"],
            "m_GC": b_margin["GC"], "m_GT": b_margin["GT"]}

    p_gc, p_gt = float(b_mass["GC"].mean()), float(b_mass["GT"].mean())
    print(f"baseline L3H1 donor mass: GC {p_gc:.4f}  GT {p_gt:.4f}  ratio {p_gc/p_gt:.4f}")
    print(f"baseline score margin:    GC {b_margin['GC'].mean():+.4f}  "
          f"GT {b_margin['GT'].mean():+.4f}", flush=True)

    if not a.smoke:
        drift = max(abs(p_gc - M4C_BASELINE["GC"]), abs(p_gt - M4C_BASELINE["GT"]))
        memorials["m4c_baseline_agreement"] = {
            "statistic": "max |this baseline mass - M4c's recorded baseline|",
            "value": drift, "threshold": M4C_BASELINE_TOL,
            "pass": bool(drift < M4C_BASELINE_TOL), "forced_by_architecture": False,
        }
        print(f"[memorial] M4c baseline agreement  max|diff| = {drift:.4e}  "
              f"{'PASS' if drift < M4C_BASELINE_TOL else 'FAIL'}", flush=True)

    # The curvature ceiling, evaluated at THIS run's baselines rather than quoted from the
    # design note.  sup over shift is the shift->0 limit, (1-p_GC)/(1-p_GT).
    ceiling = (1.0 - p_gc) / (1.0 - p_gt)
    memorials["curvature_ceiling"] = {
        "statistic": "sup over all shifts of the curvature-only fractional-drop ratio",
        "value": float(ceiling), "note": "(1-p_GC)/(1-p_GT); attained as shift -> 0",
        "forced_by_architecture": True,
        "comment": "arithmetic given the two baselines, not an empirical result",
    }
    print(f"[curvature ceiling] a shared margin shift can produce at most "
          f"{ceiling:.4f}x  (M4c observed up to 17.4x)", flush=True)

    # Smoke runs the FULL probe path on two heads rather than stopping at the standalone
    # memorials: the decomposition, the bootstrap and the LSE identity all live inside that
    # loop, and a gate that skips them tests only the parts that were never in doubt.
    probes = [PROBES[0], PROBES[-1]] if a.smoke else PROBES

    res = {}
    print()
    print(f"{'probe':>7} {'kind':>20} {'dm GC':>8} {'dm GT':>8} {'diff':>8} "
          f"{'95% CI':>18} {'F_GC':>7} {'F_GT':>7} {'F_swap':>7} {'verdict':>22}")
    for L, H, kind in probes:
        f = lambda l=L, h=H: edit_heads(ad, [(l, h)], mode="mean")  # noqa: E731
        abl_mass, abl_margin, abl_don, abl_oth = {}, {}, {}, {}
        for arm, rows in (("GC", gc), ("GT", gt)):
            m, mg, dn, ot = read_rows(ad, task, rows, off, a.batch, f)
            abl_mass[arm], abl_margin[arm] = m, mg
            abl_don[arm], abl_oth[arm] = dn, ot

        dm_gc = b_margin["GC"] - abl_margin["GC"]
        dm_gt = b_margin["GT"] - abl_margin["GT"]
        e, lo, hi = gboot_mean(dm_gc - dm_gt, genes)

        # WHERE in the score row the shift lives.  A uniform rise or fall across the whole row
        # is invisible to the softmax, so only the DIFFERENCE of these two is behavioural --
        # which is exactly why they are reported separately here and their difference is
        # checked against the margin shift below.
        lse = {}
        worst_identity = 0.0
        for arm in ("GC", "GT"):
            d_don = b_don[arm] - abl_don[arm]
            d_oth = b_oth[arm] - abl_oth[arm]
            d_m = b_margin[arm] - abl_margin[arm]
            worst_identity = max(worst_identity,
                                 float(np.abs((d_don - d_oth) - d_m).max()))
            lse[arm] = {"donor": float(d_don.mean()), "other": float(d_oth.mean()),
                        "donor_share_of_margin_shift":
                            float(d_don.mean() / d_m.mean()) if abs(d_m.mean()) > 1e-9
                            else None}
        identity_ok = worst_identity < 1e-6
        memorials.setdefault("lse_decomposition_identity", {
            "statistic": "max |(d_donor_lse - d_other_lse) - d_margin| over probes and arms",
            "value": 0.0, "threshold": 1e-6, "pass": True,
            "forced_by_architecture": False,
        })
        rec = memorials["lse_decomposition_identity"]
        rec["value"] = max(rec["value"], worst_identity)
        rec["pass"] = bool(rec["pass"] and identity_ok)

        f_gc = fractional_drop(b_mass["GC"], abl_mass["GC"])
        f_gt = fractional_drop(b_mass["GT"], abl_mass["GT"])
        f_swap = fractional_drop(b_mass["GT"], sigmoid(b_margin["GT"] - dm_gc))
        split = decompose_asymmetry(f_gc, f_gt, f_swap)
        cis = gboot_components(base, abl_mass, dm_gc, genes)

        res[f"L{L}H{H}"] = {
            "kind": kind,
            "margin_shift": {"GC": float(dm_gc.mean()), "GT": float(dm_gt.mean()),
                             "difference": e, "ci": [lo, hi]},
            "fractional_drop": {"GC": f_gc, "GT": f_gt, "GT_under_GC_shifts": f_swap},
            "decomposition": split.to_dict(),
            "decomposition_ci": cis,
            "lse_shift": lse,
        }
        print(f"L{L}H{H:<2} {kind:>20} {dm_gc.mean():>+8.4f} {dm_gt.mean():>+8.4f} "
              f"{e:>+8.4f} [{lo:>+.3f},{hi:>+.3f}] {f_gc:>+7.3f} {f_gt:>+7.3f} "
              f"{f_swap:>+7.3f} {split.verdict:>22}", flush=True)

    # L5H7 sits DOWNSTREAM of layer 3 and cannot reach its scores. Forced by the architecture,
    # so it is no evidence of anything -- but it is a live plumbing check: a non-zero value
    # here would mean the ablation context and the capture are not composing as intended.
    lse_rec = memorials.get("lse_decomposition_identity")
    if lse_rec is not None:
        print(f"\n[memorial] LSE decomposition identity  max|diff| = {lse_rec['value']:.3e}  "
              f"{'PASS' if lse_rec['pass'] else 'FAIL'}", flush=True)

    late = res.get("L5H7")
    if late is not None:
        leak = max(abs(late["margin_shift"]["GC"]), abs(late["margin_shift"]["GT"]))
        memorials["downstream_head_is_inert"] = {
            "statistic": "max |margin shift| from ablating L5H7, downstream of the reader",
            "value": float(leak), "threshold": 1e-6, "pass": bool(leak < 1e-6),
            "forced_by_architecture": True,
            "comment": "plumbing check only; cannot support a scientific claim",
        }
        print(f"\n[memorial] downstream head inert  max|shift| = {leak:.3e}  "
              f"{'PASS' if leak < 1e-6 else 'FAIL'}", flush=True)

    failed = sorted(k for k, v in memorials.items() if not v.get("pass", True))
    if a.smoke:
        print(f"\nSMOKE {'PASS' if not failed else 'FAIL ' + str(failed)} -- nothing written")
        raise SystemExit(1 if failed else 0)
    if failed:
        raise SystemExit(f"MEMORIAL FAILURE {failed} -- no artifact written, no verdict read")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "gfmcd.m4d.score_decomposition/1",
        "tier": "development", "split": "train", "attention_impl": "eager",
        "sealed_chr8_chr9_opened": False, "validation_reserve_spent": False,
        "reader": f"L{READER_LAYER}H{READER_HEAD}",
        "n_pairs": len(gc), "n_genes": len(set(genes)),
        "baseline": {"mass": {"GC": p_gc, "GT": p_gt},
                     "margin": {"GC": float(b_margin["GC"].mean()),
                                "GT": float(b_margin["GT"].mean())}},
        "memorials": memorials,
        "probes": res,
        "licenses": ("Development-tier attention geometry. Establishes whether M4c's "
                     "class-differential head effects survive removal of the softmax "
                     "curvature confound. No margin, route, mediation or biological claim "
                     "follows, and nothing here is confirmatory."),
    }, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:m4d_score_decomposition", sys.argv,
        config={"n_pairs": len(gc), "reader": f"L{READER_LAYER}H{READER_HEAD}",
                "n_boot": N_BOOT},
        seeds=dict(bootstrap=SEED), status="DEVELOPMENT",
        notes=["Train split only; chr8/9 not opened; validation reserve not spent.",
               "Pre-softmax scores read through a forward pre-hook; the model stays in "
               "genuine eager mode and is not perturbed by the measurement."])
    for f_ in (a.model, a.gtf, a.fasta):
        try:
            man.add_input(f_)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f_}")
    man.add_output(out)
    man.write(out.with_suffix(".manifest.json"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
