#!/usr/bin/env python
"""AR1 -- is the donor coupling an artifact of the single hardcoded alt ``CORRUPT_ALT="CC"``?

Every confirmatory and mechanistic result in the donor-circuit arc corrupts the partner
acceptor with ONE dinucleotide.  Nothing in ``results/`` has ever reported a per-alt effect.
This sweeps the full dinucleotide panel at both sites, on a per-example motif-hygiene
filter, and applies a frozen +-25% EQUIVALENCE rule -- so "the alts agree" is a licensed
statement rather than an accepted null.

Three arms (prereg SS2); ACC is the operator the whole arc actually uses:

    ACC   partner acceptor AG->z at [p-2,p)   readout = masked donor G at d
    DONM  donor GT->z at [d,d+2)              readout = masked donor G at d    (d masked!)
    DONA  donor GT->z at [d,d+2)              readout = masked acceptor G at p-1

DONM is partly DEGENERATE BY CONSTRUCTION and that is the point: masking d hides z[0], so
alts sharing z[1] are byte-identical and z[1]=="T" is byte-identical to clean.  Recorded as
FORCED memorials (R7/R21), never as findings.

Prereg (FROZEN): studies/splicebert_partner_selection/preregistrations/
                 PREREGISTRATION_AR1_CORRUPTION_OPERATOR_SENSITIVITY.md
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, eligible,
    load_adapter, pick_device,
)
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, DONOR_ALTS, CouplingConfig, stable_rng, substitute,
)

BASES = "ACGT"
ALTS = [a + b for a in BASES for b in BASES]
ARMS = ("ACC", "DONM", "DONA")
NATIVE = {"ACC": "AG", "DONM": "GT", "DONA": "GT"}

# --- frozen decision constants (prereg SS4/SS6) -----------------------------------------
DELTA_LO, DELTA_HI = 0.75, 1.33      # +-25% relative equivalence bound on rho(z)
ADMISSIBLE_FRAC = 0.95               # alt joins the verdict panel only above this
MIN_SUPPORT = 100                    # common-support floor (prereg SS3.1)
# Calibration plants (AMENDMENT_AR1_A1.md): (gate name, planted TRUE ratio, requirement).
# The headline gate guards FALSE EQUIVALENCE, not detection at the bound.
PLANTS = (
    ("sensitivity_no_false_equivalence", 1.35, "NOT_EQUIVALENT"),
    ("sensitivity_detect", 1.60, "DIFFERENT"),
    ("specificity_planted_1.00", 1.00, "EQUIVALENT"),
)
FPR_TARGET, N_PERM = 0.05, 200
N_BOOT = 2000
N_BOOT_PERM = 400                    # reduced draws inside the permutation loop; reported
HALFWIDTH_TARGET = 0.15              # power target for the train pilot (prereg SS7)


# ========================================================================================
# Motif hygiene -- a deterministic function of the reference genome (prereg SS1)
# ========================================================================================
def registers(seq: str, q: int, alt: str, *, masked_first: bool = False) -> dict[str, str]:
    """The three overlapping dinucleotide registers after writing ``alt`` at [q-2, q).

    ``masked_first`` (DONM): the model never sees ``alt[0]`` because that index is the
    masked readout position, so registers touching it cannot form a motif.
    """
    if masked_first:
        return {"C": alt[1] + seq[q]}
    return {"A": seq[q - 3] + alt[0], "B": alt, "C": alt[1] + seq[q]}


def admissible(seq: str, q: int, alt: str, native: str, *, masked_first: bool = False) -> bool:
    """No register may be AG (spurious acceptor) or GT/GC (spurious donor)."""
    if alt == native:
        return False
    r = registers(seq, q, alt, masked_first=masked_first)
    return not any(v == "AG" or v in ("GT", "GC") for v in r.values())


def arm_locus(x, arm: str) -> tuple[int, int]:
    """(substitution end index q, so the pair sits at [q-2,q); readout position)."""
    if arm == "ACC":
        return x.next_acceptor_pos, x.donor_pos
    if arm == "DONM":
        return x.donor_pos + 2, x.donor_pos
    return x.donor_pos + 2, x.next_acceptor_pos - 1        # DONA


# ========================================================================================
# Scoring
# ========================================================================================
def score_all(task: CouplingTask, X, batch: int, verbose: bool = True):
    """out[(i, arm, alt)] = margin (alt "__clean__" for the unperturbed sequence)."""
    import torch

    memo: dict[str, list] = defaultdict(list)
    jobs: list[tuple[int, str, str, str, int]] = []
    for i, x in enumerate(X):
        if x.sequence[x.donor_pos:x.donor_pos + 2] != "GT":
            memo["readout_anchor"].append(f"{x.example_id}:donor_not_GT")
        if x.sequence[x.next_acceptor_pos - 2:x.next_acceptor_pos] != "AG":
            memo["readout_anchor"].append(f"{x.example_id}:acceptor_not_AG")
        for arm in ARMS:
            q, rd = arm_locus(x, arm)
            jobs.append((i, arm, "__clean__", x.sequence, rd))
            for alt in ALTS:
                s = substitute(x.sequence, q - 2, alt)
                diff = [k for k in range(len(s)) if s[k] != x.sequence[k]]
                if any(k < q - 2 or k >= q for k in diff) or s[q - 2:q] != alt:
                    memo["variant_construction_exact"].append(f"{x.example_id}/{arm}/{alt}")
                built = registers(s, q, s[q - 2:q], masked_first=(arm == "DONM"))
                frozen = registers(x.sequence, q, alt, masked_first=(arm == "DONM"))
                if built != frozen:
                    memo["motif_hygiene_recomputed"].append(f"{x.example_id}/{arm}/{alt}")
                jobs.append((i, arm, alt, s, rd))

    out: dict[tuple[int, str, str], float] = {}
    with torch.no_grad():
        for b in range(0, len(jobs), batch):
            ch = jobs[b:b + batch]
            enc = task.encode([j[3] for j in ch], [j[4] for j in ch])
            m = task.margin(task.ad.model(**enc).logits.float(), [j[4] for j in ch])
            for j, v in zip(ch, m.tolist()):
                out[(j[0], j[1], j[2])] = float(v)
            if verbose and b % (batch * 100) == 0:
                print(f"    {b:,}/{len(jobs):,} forwards", end="\r", flush=True)
    if verbose:
        print(f"    {len(jobs):,}/{len(jobs):,} forwards done   ")

    # batch_invariance -- rescore a slice at different offsets; on-path, can fail
    probe = jobs[:min(len(jobs), batch)][::-1]
    if probe:
        with torch.no_grad():
            enc = task.encode([j[3] for j in probe], [j[4] for j in probe])
            m = task.margin(task.ad.model(**enc).logits.float(), [j[4] for j in probe])
        for j, v in zip(probe, m.tolist()):
            if abs(float(v) - out[(j[0], j[1], j[2])]) > 0:
                memo["batch_invariance"].append(f"{j[0]}/{j[1]}/{j[2]}")

    for i in range(len(X)):
        # identity_noop -- FORCED (byte-identical input); plumbing only, R7/R21
        for arm in ARMS:
            if abs(out[(i, arm, NATIVE[arm])] - out[(i, arm, "__clean__")]) > 0:
                memo["identity_noop"].append(f"{i}/{arm}")
        # masked_position_invisibility -- FORCED by masking d under DONM
        for alt in ALTS:
            if any(abs(out[(i, "DONM", z)] - out[(i, "DONM", alt)]) > 0
                   for z in ALTS if z[1] == alt[1]):
                memo["masked_position_invisibility"].append(f"{i}/{alt}:twin")
            if alt[1] == "T" and abs(out[(i, "DONM", alt)] - out[(i, "DONM", "__clean__")]) > 0:
                memo["masked_position_invisibility"].append(f"{i}/{alt}:clean")
    return out, memo


# ========================================================================================
# Gene-clustered bootstrap -- vectorised over draws (the permutation gate calls it ~9000x)
# ========================================================================================
def _gene_index(genes) -> tuple[np.ndarray, int]:
    order: dict = {}
    return np.array([order.setdefault(g, len(order)) for g in genes]), len(order)


def boot_ratio(num: np.ndarray, dnm: np.ndarray, genes, n_boot: int, seed: int = 0):
    """CI for a RATIO OF MEANS on gene-clustered resamples.  Never mean-of-ratios."""
    gi, G = _gene_index(genes)
    if G == 0:
        return (float("nan"), float("nan"))
    sn = np.bincount(gi, weights=num, minlength=G)
    sd = np.bincount(gi, weights=dnm, minlength=G)
    pick = np.random.default_rng(seed).integers(0, G, size=(n_boot, G))
    d = sd[pick].sum(1)
    r = np.where(np.abs(d) > 1e-9, sn[pick].sum(1) / np.where(d == 0, 1, d), np.nan)
    return float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


def boot_mean(vals: np.ndarray, genes, n_boot: int, seed: int = 0):
    gi, G = _gene_index(genes)
    if G == 0:
        return (float("nan"), float("nan"))
    s = np.bincount(gi, weights=vals, minlength=G)
    c = np.bincount(gi, minlength=G).astype(float)
    pick = np.random.default_rng(seed).integers(0, G, size=(n_boot, G))
    m = s[pick].sum(1) / c[pick].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_did(cc: np.ndarray, ac: np.ndarray, is_g: np.ndarray, genes,
             n_boot: int, seed: int = 0):
    """CI for [d(CC|G)-d(CC|!G)] - [d(AC|G)-d(AC|!G)] on gene-clustered resamples."""
    gi, G = _gene_index(genes)
    cols = []
    for v, m in ((cc, is_g), (cc, ~is_g), (ac, is_g), (ac, ~is_g)):
        cols.append((np.bincount(gi, weights=v * m, minlength=G),
                     np.bincount(gi, weights=m.astype(float), minlength=G)))
    pick = np.random.default_rng(seed).integers(0, G, size=(n_boot, G))
    means = []
    for s, c in cols:
        cs = c[pick].sum(1)
        means.append(np.where(cs > 0, s[pick].sum(1) / np.where(cs == 0, 1, cs), np.nan))
    d = (means[0] - means[1]) - (means[2] - means[3])
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


# ========================================================================================
# Inference
# ========================================================================================
def visible_construct(x, arm: str, alt: str) -> str:
    """The sequence AS THE MODEL SEES IT: substituted, with the readout position masked.

    AMENDMENT_AR1_A2 -- byte-identity of twins and no-ops is VERIFIED on this string rather
    than inferred from the masking argument.
    """
    q, rd = arm_locus(x, arm)
    s = substitute(x.sequence, q - 2, alt)
    return s[:rd] + "#" + s[rd + 1:]


def prep_arm(X, arm: str) -> dict:
    """Admissibility, common support, gene labels, and the A2 panel reduction."""
    q = [arm_locus(x, arm)[0] for x in X]
    mf = arm == "DONM"
    adm = {z: np.array([admissible(X[i].sequence, q[i], z, NATIVE[arm], masked_first=mf)
                        for i in range(len(X))]) for z in ALTS}
    cc = adm[CORRUPT_ALT]
    support = {z: np.flatnonzero(adm[z] & cc) for z in ALTS}

    # --- AMENDMENT_AR1_A2: reduce to distinct realised constructs -----------------------
    clean_sig = tuple(visible_construct(x, arm, NATIVE[arm]) for x in X)
    sig = {z: tuple(visible_construct(x, arm, z) for x in X) for z in ALTS}
    noop = sorted(z for z in ALTS if z != NATIVE[arm] and sig[z] == clean_sig)
    # A3: the representative is the INCUMBENT where the group contains it -- a lexicographic
    # rule elected CC's twin and left the panel comparing the reference with itself.
    rep: dict[tuple, str] = {}
    for z in sorted(ALTS):
        if z == NATIVE[arm] or z in noop:
            continue
        if sig[z] not in rep or z == CORRUPT_ALT:
            rep[sig[z]] = z
    twins = {z: rep[sig[z]] for z in sorted(ALTS)
             if z != NATIVE[arm] and z not in noop and rep[sig[z]] != z}
    # A3: an alt byte-identical to the REFERENCE construct yields rho == 1 by construction.
    ref_twins = sorted(z for z in ALTS if z != CORRUPT_ALT and z not in noop
                       and z != NATIVE[arm] and sig[z] == sig[CORRUPT_ALT])
    return {"adm": adm, "frac": {z: float(adm[z].mean()) for z in ALTS},
            "support": support, "genes": [x.gene_id for x in X],
            "excluded_noop": noop, "deduplicated_twins": twins,
            "reference_twins": ref_twins,
            "eligible": {z for z in ALTS if z != NATIVE[arm] and z not in noop
                         and z not in twins and z not in ref_twins}}


def verdict_for(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return "INCONCLUSIVE"
    if lo >= DELTA_LO and hi <= DELTA_HI:
        return "EQUIVALENT"
    if hi < DELTA_LO or lo > DELTA_HI:
        return "DIFFERENT"
    return "INCONCLUSIVE"


def deltas(X, out, arm: str, z: str) -> np.ndarray:
    return np.array([out[(i, arm, "__clean__")] - out[(i, arm, z)] for i in range(len(X))])


def analyse_arm(X, out, arm: str, pre: dict, *, scale: dict | None = None,
                n_boot: int = N_BOOT, seed: int = 0, full: bool = True) -> dict:
    """Per-alt equivalence against CC on common support (prereg SS3.1/SS4)."""
    scale = scale or {}
    D = {z: deltas(X, out, arm, z) * scale.get(z, 1.0) for z in ALTS}
    res: dict = {}
    for z in ALTS:
        if z == NATIVE[arm]:
            continue
        idx = pre["support"][z]
        row: dict = {"admissible_frac": pre["frac"][z], "n_common_support": int(len(idx)),
                     "in_panel": z in (ACCEPTOR_ALTS if arm == "ACC" else DONOR_ALTS),
                     "gc_content": sum(b in "GC" for b in z)}
        if len(idx) < MIN_SUPPORT:
            row["status"] = "INSUFFICIENT_SUPPORT"
            res[z] = row
            continue
        dz, dcc = D[z][idx], D[CORRUPT_ALT][idx]
        g = [pre["genes"][i] for i in idx]
        m_z, m_cc = float(dz.mean()), float(dcc.mean())
        ci = boot_ratio(dz, dcc, g, n_boot, seed)
        row["status"] = verdict_for(ci)
        row["rho"] = m_z / m_cc if abs(m_cc) > 1e-9 else float("nan")
        row["rho_ci"] = list(ci)
        if full:
            w = np.sort(dz)
            cut = max(1, int(0.05 * len(w)))
            row.update(
                mean_delta=m_z, mean_delta_cc=m_cc,
                ci_halfwidth=(ci[1] - ci[0]) / 2,
                den_clear_of_zero=bool(abs(m_cc) > 1e-9),
                paired_diff=float((dz - dcc).mean()),
                paired_diff_ci=list(boot_mean(dz - dcc, g, n_boot, seed)),
                winsorized_delta=float(np.clip(w, w[cut], w[-cut]).mean()),
                tail_share_top_decile=float(np.sort(np.abs(w))[-max(1, len(w) // 10):].sum()
                                            / max(np.abs(w).sum(), 1e-9)),
                n_genes=len(set(g)))
        res[z] = row

    # A2: only distinct realised constructs may carry a verdict -- a no-op or a byte-identical
    # twin returns an architecture-forced number, not a measurement (R7/R21).
    panel = sorted(z for z, r in res.items()
                   if r["admissible_frac"] >= ADMISSIBLE_FRAC
                   and r["status"] != "INSUFFICIENT_SUPPORT" and z != CORRUPT_ALT
                   and z in pre["eligible"])
    sts = [res[z]["status"] for z in panel]
    # A3: an empty panel is NO_ADMISSIBLE_PANEL, never ALT_INVARIANT -- equivalence is a
    # claim, and it requires something to have been compared.
    verdict = ("NO_ADMISSIBLE_PANEL" if not panel
               else "NOT_INVARIANT" if "DIFFERENT" in sts
               else "ALT_INVARIANT" if all(s == "EQUIVALENT" for s in sts)
               else "INCONCLUSIVE")
    for z, r in res.items():
        if z in pre["excluded_noop"]:
            r["excluded"] = "NO_OP -- visible construct byte-identical to clean (forced)"
        elif z in pre["reference_twins"]:
            r["excluded"] = "REFERENCE_TWIN -- byte-identical to CC; rho == 1 forced"
        elif z in pre["deduplicated_twins"]:
            r["excluded"] = f"TWIN of {pre['deduplicated_twins'][z]} (byte-identical, forced)"
    return {"alts": res, "verdict_panel": panel, "verdict": verdict,
            "excluded_noop": pre["excluded_noop"],
            "deduplicated_twins": pre["deduplicated_twins"],
            "reference_twins": pre["reference_twins"],
            "contaminated": sorted(z for z, r in res.items()
                                   if r["admissible_frac"] < ADMISSIBLE_FRAC)}


def calibrate(X, out, arm: str, pre: dict, seed: int = 0) -> dict:
    """Calibration gates, respecified by AMENDMENT_AR1_A1.md.

    The plant is rescaled so its TRUE ratio is ``tau`` by construction, independent of the
    unmeasured ``rho_true(z)`` -- the original prereg SS6 plant scaled by tau directly, which
    tests ``tau * rho_true(z)``, an unknown.  It keeps a real alt's per-example noise: a
    synthetic ``delta_CC * tau`` has a bootstrap ratio of exactly tau in every draw and so
    could not fail (R7/R21).

    The headline gate guards FALSE EQUIVALENCE -- the error that would let "the alts agree"
    be asserted wrongly -- not positive detection at the bound.
    """
    ref = analyse_arm(X, out, arm, pre, n_boot=N_BOOT_PERM, seed=seed, full=False)
    panel = ref["verdict_panel"]
    gates: dict = {"noise_donor_alt": panel[0] if panel else None,
                   "n_boot_in_gates": N_BOOT, "n_boot_in_permutation": N_BOOT_PERM}

    if not panel:
        # A3: no panel -> nothing to plant into.  NOT_APPLICABLE, not FAIL.
        for g, _t, _w in PLANTS:
            gates[g] = {"verdict": "NO_PANEL", "PASS": True, "applicable": False,
                        "note": "NOT_APPLICABLE: no alt cleared admissibility + common "
                                "support, so there is nothing to plant into (A3)"}
    else:
        z = panel[0]
        idx = pre["support"][z]
        dz, dcc = deltas(X, out, arm, z)[idx], deltas(X, out, arm, CORRUPT_ALT)[idx]
        g_lbl = [pre["genes"][i] for i in idx]
        rho_true = float(dz.mean()) / float(dcc.mean())
        for name, tau, want in PLANTS:
            if abs(rho_true) < 1e-9:
                gates[name] = {"verdict": "UNDEFINED", "PASS": False}
                continue
            ci = boot_ratio(dz * (tau / rho_true), dcc, g_lbl, N_BOOT, seed)
            v = verdict_for(ci)
            gates[name] = {"planted_tau": tau, "verdict": v, "rho_ci": list(ci),
                           "requirement": want,
                           "PASS": (v != "EQUIVALENT") if want == "NOT_EQUIVALENT"
                           else (v == want)}
        gates["rho_true_of_noise_donor"] = rho_true

    rng = stable_rng(f"ar1-perm-{arm}", 0)
    kw = dict(n_boot=N_BOOT_PERM, seed=seed, full=False)
    fires = 0
    for _ in range(N_PERM):
        perm = dict(out)
        for i in range(len(X)):
            vals = [out[(i, arm, z2)] for z2 in ALTS]
            rng.shuffle(vals)
            for z2, v in zip(ALTS, vals):
                perm[(i, arm, z2)] = v
        if analyse_arm(X, perm, arm, pre, **kw)["verdict"] == "NOT_INVARIANT":
            fires += 1
    fpr = fires / N_PERM
    gates["permuted_label_fpr"] = {"fpr": fpr, "target": FPR_TARGET, "n_perm": N_PERM,
                                   "PASS": fpr <= FPR_TARGET}
    return gates


def did_probe(X, out, seed: int = 0) -> dict:
    """DONA donor-side confound probe: does planting GC/GT at [d-1,d] rescue the donor?

    DiD = [d(CC|G) - d(CC|!G)] - [d(AC|G) - d(AC|!G)];  AC plants nothing in either
    stratum, so it absorbs any intrinsic strength difference between the strata.
    """
    cc, ac = deltas(X, out, "DONA", "CC"), deltas(X, out, "DONA", "AC")
    is_g = np.array([x.sequence[x.donor_pos - 1] == "G" for x in X])
    genes = [x.gene_id for x in X]
    point = ((cc[is_g].mean() - cc[~is_g].mean()) - (ac[is_g].mean() - ac[~is_g].mean())
             if is_g.any() and (~is_g).any() else float("nan"))
    ci = boot_did(cc, ac, is_g, genes, N_BOOT, seed)
    bound = 0.25 * abs(float(ac.mean()))
    status = ("CONFIRMED" if ci[1] < 0
              else "REFUTED" if (ci[0] >= -bound and ci[1] <= bound) else "INCONCLUSIVE")
    return {"DiD": float(point), "ci": list(ci), "n_G": int(is_g.sum()),
            "n_notG": int((~is_g).sum()), "mean_delta_AC": float(ac.mean()),
            "mean_delta_CC": float(cc.mean()), "refute_bound": bound, "status": status,
            "predicted_sign": "negative (CC drop SMALLER where a GC/GT is planted)"}


# ========================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--smoke", action="store_true", help="memorials + calibration, no verdict")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--from-scores", default=None,
                    help="re-analyse a persisted ar1_scores_*.json (no model run)")
    ap.add_argument("--overwrite-record", action="store_true")
    args = ap.parse_args()

    # A smoke whose calibration gates cannot fire tests nothing (R7/R21).  Drop the
    # common-support floor so the DECISION MACHINERY is actually exercised at tiny n; the
    # relaxed value is recorded in provenance and a smoke never emits a verdict anyway.
    global MIN_SUPPORT
    if args.smoke:
        MIN_SUPPORT = 2

    dev = args.device or pick_device()
    print(f"AR1 corruption-operator sensitivity | split={args.split} device={dev}")

    from gfmcd.splicing.circuit import _dataset
    X = eligible(_dataset(args.gtf, args.fasta), args.split)
    stable_rng("ar1", CouplingConfig().seed).shuffle(X)

    if args.from_scores:
        # Re-analysis path (AMENDMENT_AR1_A2): a panel-rule change must never cost a re-run.
        blob = json.loads(Path(args.from_scores).read_text())
        ids = [e["example_id"] for e in blob["examples"]]
        by_id = {x.example_id: x for x in X}
        missing = [e for e in ids if e not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} scored examples absent from the cohort; "
                             "the split or GTF does not match the score file")
        X = [by_id[e] for e in ids]
        out = {(int(r["i"]), r["arm"], r["alt"]): float(r["margin"]) for r in blob["scores"]}
        memo_raw = {k: list(v) for k, v in blob["memorials_raw"].items()}
        print(f"  re-analysing {len(X)} scored examples from {args.from_scores}")
    else:
        X = X[: (8 if args.smoke else args.n)]
        print(f"  {len(X)} eligible exons, {len(set(x.gene_id for x in X))} genes")
        ad = load_adapter(args.model, dev)
        task = CouplingTask(ad, batch=args.batch)
        out, memo_raw = score_all(task, X, args.batch)

        # Persist raw scores so a re-analysis never needs a re-run (AMENDMENT_AR1_A1).
        stag = args.tag or ("smoke" if args.smoke else args.split)
        sp = REPO / f"results/splicebert_coupling/development/ar1_scores_{stag}.json"
        sp = guard_output_path(sp, overwrite=args.overwrite_record)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({
            "scores": [{"i": i, "arm": a, "alt": z, "margin": v}
                       for (i, a, z), v in out.items()],
            "examples": [{"i": i, "example_id": x.example_id, "gene_id": x.gene_id,
                          "chrom": x.chrom, "strand": x.strand, "donor_pos": x.donor_pos,
                          "next_acceptor_pos": x.next_acceptor_pos,
                          "up_donor_base": x.sequence[x.donor_pos - 1],
                          "first_exon_base": x.sequence[x.next_acceptor_pos]}
                         for i, x in enumerate(X)],
            "memorials_raw": {k: v for k, v in memo_raw.items()},
        }, indent=1))
        print(f"  scores -> {sp}")

    forced = {"identity_noop", "masked_position_invisibility"}
    memorials = {k: {"n_fail": len(memo_raw.get(k, [])), "PASS": not memo_raw.get(k),
                     "forced": k in forced, "examples": memo_raw.get(k, [])[:5],
                     **({"note": "FORCED by construction: plumbing/architecture only (R7/R21)"}
                        if k in forced else {})}
                 for k in ("variant_construction_exact", "motif_hygiene_recomputed",
                           "readout_anchor", "batch_invariance", "identity_noop",
                           "masked_position_invisibility")}
    hard_fail = [k for k, v in memorials.items() if not v["PASS"] and not v["forced"]]

    pre = {arm: prep_arm(X, arm) for arm in ARMS}
    res: dict = {"memorials": memorials, "smoke": bool(args.smoke),
                 "equivalence_bound": [DELTA_LO, DELTA_HI], "arms": None}
    print("  calibrating (planted-difference, planted-null, permuted-label FPR) ...")
    res["calibration"] = {arm: calibrate(X, out, arm, pre[arm]) for arm in ARMS}
    gate_names = [p[0] for p in PLANTS] + ["permuted_label_fpr"]
    # A3: gates are scored PER ARM -- one arm's failure never voids another.
    arm_gates = {arm: all(res["calibration"][arm][g]["PASS"] for g in gate_names)
                 for arm in ARMS}

    # The descriptive table is ALWAYS computed and stored -- the first pilot dropped it on a
    # gate failure and left no way to inspect rho without re-scoring (AMENDMENT_AR1_A1).
    # When gates fail it is stored under `arms_descriptive` and carries a refusal label; the
    # interpretable `arms` key is populated only when every gate passes.
    tbl = {arm: analyse_arm(X, out, arm, pre[arm]) for arm in ARMS}
    probe = did_probe(X, out)
    base_ok = not hard_fail and not args.smoke
    licensed = {arm: base_ok and arm_gates[arm] for arm in ARMS}
    res["arm_gates_pass"] = arm_gates

    verdicts, keep, drop = {}, {}, {}
    for arm in ARMS:
        if licensed[arm]:
            verdicts[arm] = tbl[arm]["verdict"]
            keep[arm] = tbl[arm]
        else:
            verdicts[arm] = ("HALTED -- memorial" if hard_fail else
                             "SMOKE" if args.smoke else "GATES_FAILED")
            tbl[arm]["verdict"] = "NOT_LICENSED -- descriptive only, no interpretation"
            drop[arm] = tbl[arm]
    if keep:
        res["arms"] = keep
    if drop:
        res["arms_descriptive"] = drop
    # The DONA-only probe follows DONA's licence.
    if licensed["DONA"]:
        res["donor_confound_probe"] = probe
    else:
        probe["status"] = "NOT_LICENSED -- descriptive only, no interpretation"
        res["donor_confound_probe_descriptive"] = probe

    if hard_fail:
        res["verdict"] = f"HALTED -- non-forced memorial failed: {hard_fail}"
    elif args.smoke:
        res["verdict"] = "SMOKE -- mechanics and calibration only, no interpretation"
    else:
        res["verdict"] = verdicts

    res["provenance"] = {
        "split": args.split, "n": len(X), "n_genes": len(set(x.gene_id for x in X)),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "model": args.model,
        "gtf": args.gtf, "n_boot": N_BOOT, "n_boot_in_gates": N_BOOT_PERM,
        "delta": [DELTA_LO, DELTA_HI], "admissible_frac": ADMISSIBLE_FRAC,
        "min_support": MIN_SUPPORT, "halfwidth_target": HALFWIDTH_TARGET,
        "min_support_relaxed_for_smoke": bool(args.smoke),
        "prereg": "studies/splicebert_partner_selection/preregistrations/"
                  "PREREGISTRATION_AR1_CORRUPTION_OPERATOR_SENSITIVITY.md",
    }

    report(res)
    tag = args.tag or ("smoke" if args.smoke else args.split)
    canonical = args.split == "val" and not args.smoke and not args.tag
    p = REPO / ("results/splicebert_coupling/ar1_corruption_operator_sensitivity.json"
                if canonical else f"results/splicebert_coupling/development/ar1_{tag}.json")
    p = guard_output_path(p, overwrite=args.overwrite_record)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=float))
    ExperimentManifest.create(
        "splicebert_coupling:ar1_corruption_operator_sensitivity", sys.argv,
        config=res["provenance"],
        status=("SMOKE" if args.smoke else "CONFIRMATORY" if canonical else "EXPLORATORY"),
    ).write(p.with_suffix(".manifest.json"))
    print(f"\n-> {p}")


def report(res: dict) -> None:
    print("\n" + "=" * 96)
    for k, v in res["memorials"].items():
        print(f"  {'PASS' if v['PASS'] else 'FAIL'}  {k}"
              f"{' [FORCED]' if v['forced'] else ''}  n_fail={v['n_fail']}")
    for arm, c in res["calibration"].items():
        print(f"\n  calibration {arm} (noise donor {c['noise_donor_alt']}, "
              f"rho_true={c.get('rho_true_of_noise_donor', float('nan')):.3f}):")
        for g in [p[0] for p in PLANTS] + ["permuted_label_fpr"]:
            print(f"    {'PASS' if c[g]['PASS'] else 'FAIL'}  {g:<34} "
                  f"{c[g].get('verdict') if 'verdict' in c[g] else c[g]['fpr']}")
    for arm, a in (res.get("arms") or res.get("arms_descriptive") or {}).items():
        print(f"\n  --- {arm} ---  verdict={a['verdict']}  panel={a['verdict_panel']}")
        if a.get("excluded_noop") or a.get("deduplicated_twins"):
            print(f"    A2 reduction: no-ops={a['excluded_noop']}  "
                  f"twins={a['deduplicated_twins']}")
        print(f"    {'alt':>4} {'pnl':>4} {'adm%':>6} {'n':>5} {'meanD':>8} {'rho':>7} "
              f"{'95% CI':>18}  status")
        for z, r in sorted(a["alts"].items()):
            pn = "*" if r["in_panel"] else "-"
            if r.get("status") == "INSUFFICIENT_SUPPORT":
                print(f"    {z:>4} {pn:>4} {r['admissible_frac']:>5.1%} "
                      f"{r['n_common_support']:>5} {'-':>8} {'-':>7} {'-':>18}  "
                      f"INSUFFICIENT_SUPPORT")
                continue
            print(f"    {z:>4} {pn:>4} {r['admissible_frac']:>5.1%} {r['n_common_support']:>5} "
                  f"{r['mean_delta']:>+8.3f} {r['rho']:>7.3f} "
                  f"[{r['rho_ci'][0]:>7.3f},{r['rho_ci'][1]:>7.3f}]  {r['status']}")
    d = res.get("donor_confound_probe") or res.get("donor_confound_probe_descriptive")
    if d:
        print(f"\n  donor GC-planting DiD = {d['DiD']:+.3f} "
              f"[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]  n_G={d['n_G']} n_notG={d['n_notG']}"
              f"  -> {d['status']}")
    print(f"\n  VERDICT: {res['verdict']}")
    print("=" * 96)


if __name__ == "__main__":
    main()
