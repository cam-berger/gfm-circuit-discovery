"""Phase C, tests T1 and T2 as registered in `docs/PREREGISTRATION.md` (2026-07-09 b).

T1  anchor dependence as a function of distance from the start codon (16-nt bins), decided by BIC
    between a constant, a slope in log d, and a sinusoid.  H3 -> flat.  H1 -> slope.  H2 -> periodic.
T2  the same far-field anchor dependence, split by whether the anchor-free pooled oracle already
    recovers the frame at that position.  H3 -> the effect lives on the oracle's errors.

Every number carries its paired decoy-Met control: an in-frame ATG >=30 nt downstream is replaced by
the same CTC, so the edit type is held fixed and only the *position* varies.

    python scripts/phase_c_relay.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from train_cell import pool  # noqa: E402

from gfmcd.data.genome import Split  # noqa: E402
from gfmcd.models.gfm_nano import GFMNano, NanoConfig  # noqa: E402
from gfmcd.oracle import oracle_correct  # noqa: E402
from gfmcd.paths import project_paths  # noqa: E402
from gfmcd.signatures import ablate, downstream_met, start_index  # noqa: E402

OUT = project_paths().results / "reading-frame/fig_data"
IGNORE = -100
N = 800
FAR = 64
BIN = 16

MODELS = [
    ("C · real CDS, easy detector", "cellC_s0_step8000.pt", "C"),
    ("D · real CDS, natural", "cellD_s0_step8000.pt", "D"),
    ("A · uniform, easy detector", "nano_uniform_step8400.pt", "A"),  # the counter, as the H1 reference
    ("B · uniform, hard detector", "cellB_s1_step24000.pt", "B"),
]


def load(name: str, device: str) -> GFMNano:
    blob = torch.load(project_paths().checkpoints / name, map_location=device, weights_only=False)
    m = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    m.load_state_dict(blob["state"])
    return m


@torch.no_grad()
def per_position(model, tok, aa, device):
    """(n, w) correct-flag and (n, w) cross-entropy, position by position."""
    t, y = torch.from_numpy(tok).long(), torch.from_numpy(aa).long()
    corr, loss = [], []
    for i in range(0, len(t), 64):
        lg = model(t[i : i + 64].to(device)).cpu().float()
        yy = y[i : i + 64]
        corr.append(lg.argmax(-1) == yy)
        loss.append(F.cross_entropy(lg.transpose(1, 2), yy.clamp(min=0), reduction="none"))
    return torch.cat(corr).numpy(), torch.cat(loss).numpy()


def bic(y: np.ndarray, yhat: np.ndarray, k: int) -> float:
    n = len(y)
    rss = float(((y - yhat) ** 2).sum())
    return n * np.log(max(rss, 1e-12) / n) + k * np.log(n)


# The registered statistic used 16-nt bins and returned a fitted period of 17 nt -- a period at the
# sampling interval, which is aliased and unidentifiable. Δ(d) is now measured at 1-nt resolution and
# the period search is floored at 8 nt (4x Nyquist). The periodic term is tested *on top of* the trend,
# so curvature in log d cannot masquerade as an oscillation.
PERIOD_MIN, PERIOD_MAX = 8.0, 120.0


def fit_t1(d: np.ndarray, delta: np.ndarray) -> dict:
    """M0 const; M1 + slope in log2 d; M2 + sinusoid at a fitted period. Registered decision rule."""
    logd = np.log2(d)
    b_const = bic(delta, np.full_like(delta, delta.mean()), 1)

    X = np.stack([np.ones_like(logd), logd], 1)
    coef = np.linalg.lstsq(X, delta, rcond=None)[0]
    trend = X @ coef
    b_slope = bic(delta, trend, 2)

    best = (np.inf, None, None)
    for period in np.arange(PERIOD_MIN, PERIOD_MAX + 0.5, 0.5):
        S = np.concatenate([X, np.stack([np.sin(2 * np.pi * d / period), np.cos(2 * np.pi * d / period)], 1)], 1)
        c = np.linalg.lstsq(S, delta, rcond=None)[0]
        b = bic(delta, S @ c, 5)  # 4 linear + the fitted period
        if b < best[0]:
            best = (b, period, float(np.hypot(c[2], c[3])))
    b_per, period, amp = best

    return dict(n_points=len(d), mean=float(delta.mean()), slope_per_log2=float(coef[1]),
                bic_const=b_const, bic_slope=b_slope, bic_periodic=b_per,
                d_bic_slope=b_const - b_slope, d_bic_periodic=b_slope - b_per,
                best_period=float(period), amplitude=amp)


def verdict(f: dict) -> str:
    """d_bic_periodic is now periodic-vs-trend; d_bic_slope is trend-vs-constant."""
    if f["d_bic_periodic"] > 6:
        return "H2 re-sync chain (periodic on top of trend)"
    if f["d_bic_slope"] > 6:
        return "H1 counter (slope)" if f["slope_per_log2"] >= 0 else "undecided (negative slope)"
    if abs(f["slope_per_log2"]) < 0.02:
        return "H3 anchor-corrected pooling (flat)"
    return "undecided"


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    for key, ckpt, cell in MODELS:
        model = load(ckpt, device)
        tok, aa, ph = pool(cell, N, model.cfg.max_len, Split.VAL, 4242)
        start = start_index(ph)
        met = downstream_met(tok, start)
        have = met >= 0

        c_int, l_int = per_position(model, tok, aa, device)
        c_abl, l_abl = per_position(model, ablate(tok, start), aa, device)
        c_ctl, l_ctl = per_position(model, ablate(tok, np.where(have, met, -1)), aa, device)

        body = (ph >= 0) & (aa != IGNORE)
        pos = np.arange(tok.shape[1])[None, :]
        d = pos - start[:, None]

        # An edit is only a control on the rows that actually have a downstream Met, and only
        # downstream of it -- upstream of the decoy nothing was changed.
        ctl_ok = have[:, None] & (pos >= met[:, None]) & body

        # --- T1: delta(d) at 1-nt resolution, paired control subtracted at the same distance ---
        curve = []
        for dd in range(4, 256):
            sel = body & (d == dd)
            csel = ctl_ok & (d == dd)
            if sel.sum() < 200 or csel.sum() < 200:
                continue
            d_abl = float(c_int[sel].mean() - c_abl[sel].mean())
            d_ctl = float(c_int[csel].mean() - c_ctl[csel].mean())
            curve.append(dict(d=dd, n=int(sel.sum()),
                              delta=d_abl - d_ctl, delta_raw=d_abl, control=d_ctl,
                              dloss=float(l_abl[sel].mean() - l_int[sel].mean())))
        fit = fit_t1(np.array([r["d"] for r in curve], dtype=float),
                     np.array([r["delta"] for r in curve]))
        fit["verdict"] = verdict(fit)

        # --- T2: far field, split by the anchor-free oracle -----------------------------------
        far = body & (d >= FAR)
        cfar = ctl_ok & (d >= FAR)
        ok = oracle_correct(tok, ph)

        t2 = {}
        for name, m, cm in (("all", far, cfar),
                            ("oracle-right", far & ok, cfar & ok),
                            ("oracle-wrong", far & ~ok, cfar & ~ok)):
            d_abl = float(c_int[m].mean() - c_abl[m].mean())
            d_ctl = float(c_int[cm].mean() - c_ctl[cm].mean()) if cm.sum() > 100 else float("nan")
            t2[name] = dict(n=int(m.sum()), frac=float(m.sum() / far.sum()),
                            intact=float(c_int[m].mean()), ablated=float(c_abl[m].mean()),
                            delta_raw=d_abl, control=d_ctl, delta=d_abl - d_ctl,
                            dloss=float(l_abl[m].mean() - l_int[m].mean()))
        r, w = t2["oracle-right"]["delta"], t2["oracle-wrong"]["delta"]
        t2["ratio_wrong_over_right"] = float(w / r) if r > 1e-6 else float("inf")
        t2["h3_strong"] = bool(r < 0.10 and t2["ratio_wrong_over_right"] >= 3)

        results[key] = dict(ckpt=ckpt, cell=cell, t1_curve=curve, t1_fit=fit, t2=t2)

        print(f"\n=== {key}")
        print(f"  T1  mean Δ {fit['mean']:.4f}   slope/log2 {fit['slope_per_log2']:+.4f}   "
              f"ΔBIC slope {fit['d_bic_slope']:.1f}  periodic {fit['d_bic_periodic']:.1f} "
              f"(T={fit['best_period']:.0f} nt, amp {fit['amplitude']:.4f})")
        print(f"      -> {fit['verdict']}")
        print(f"  T2  {'subset':>14}{'frac':>8}{'intact':>9}{'ablated':>9}{'ctl':>9}{'Δ':>9}{'Δloss':>9}")
        for name in ("all", "oracle-right", "oracle-wrong"):
            s = t2[name]
            print(f"      {name:>14}{s['frac']:>8.3f}{s['intact']:>9.4f}{s['ablated']:>9.4f}"
                  f"{s['control']:>9.4f}{s['delta']:>9.4f}{s['dloss']:>9.4f}")
        print(f"      wrong/right = {t2['ratio_wrong_over_right']:.2f}   H3-strong: {t2['h3_strong']}")

    (OUT / "phase_c_t1_t2.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}/phase_c_t1_t2.json")


if __name__ == "__main__":
    main()
