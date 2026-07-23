"""B5 — HCA18 context-signal power adjudication (PREREGISTRATION_B5_CONTEXT_SIGNAL_POWER.md, FROZEN v2).

Deterministic, model-free power arithmetic over three committed JSON artifacts. Answers: for each of the
20 HCA18 context-associated discovery signals, what EFFECTIVE matched-pair sample size N_req would reach the
frozen precision gate MDE80 <= 0.05, and does N_req exceed the achievable supply (led by the hard
one-case-per-gene exact-other-tag support gene cap)?  Because N_req rests on a selected, single-draw se, the
arithmetic can only license CLOSE-AS-UNDERPOWERED (never a reachable/redesign disposition) on the real signals.

Reuses the repo's own power identity verbatim:
    Z_POWER_80 = float(norm.ppf(.975) + norm.ppf(.80))          # analyze_atlas_discovery.py
    mde80      = Z_POWER_80 * se_gene_dyadic
    n_eff      = Kish effective size (gene-dyadic weights)
    => mde80 ∝ n_eff^(-1/2), so  N_req = n_eff * (mde80 / tau)^2   with tau = 0.05.

No model forward pass. Gates: Global PASS = AND(M1..M9, PP1..PP4, N1..N3). --smoke runs the arithmetic-only
memorials + calibration and emits NO ceiling estimate and NO verdict.

Usage:
    python scripts/splicebert_head_context_atlas/analyze_b5_context_signal_power.py --smoke
    python scripts/splicebert_head_context_atlas/analyze_b5_context_signal_power.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from scipy.stats import norm  # same dependency the producing script used (exact constant match, M3)

# --- repo import shim so gfmcd is importable regardless of install mode ---
REPO_ROOT = Path(__file__).resolve().parents[2]
_src = REPO_ROOT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
from gfmcd.provenance import ExperimentManifest  # noqa: E402

# --------------------------------------------------------------------------------------------------
# Frozen constants (PREREGISTRATION_B5 §1, §6)
# --------------------------------------------------------------------------------------------------
TAU = 0.05                                            # the frozen precision gate MDE80 <= 0.05
Z_POWER_80 = float(norm.ppf(0.975) + norm.ppf(0.80))  # reused verbatim; NOT a rounded 2.8016 literal (M3)
MARGINAL_BAND = 1.5                                   # [1.0x, 1.5x] common-support band (§5, reported)

RESULTS_DIR = REPO_ROOT / "results" / "splicebert_head_context_atlas"
CONTEXT_JSONL = RESULTS_DIR / "atlas_discovery_analysis_v1.context.jsonl"
MATCHED_SUMMARY = RESULTS_DIR / "atlas_context_matched_v7.summary.json"
ATLAS_CENSUS = RESULTS_DIR / "atlas_context_census_v1.json"

# The three context cells the 20 signals live in (category | target_role -> matched-cohort contrast key).
CELLS = ("acceptor_cluster|acceptor", "donor_cluster|donor", "cassette_exon|donor")

# Expected artifact facts (M1/M5) — verified against the committed JSON this session.
EXP_N_SIGNALS = 20
EXP_CLASS = {"material_suppress": 17, "material_support": 3}
EXP_CATEGORY = {"acceptor_cluster": 11, "donor_cluster": 6, "cassette_exon": 3}
EXP_COMPONENT_TYPE = {"mlp_write": 16, "attention_head": 4}
EXP_SUPPLY_GENES = {"acceptor_cluster|acceptor": 1982, "donor_cluster|donor": 1780, "cassette_exon|donor": 2672}
EXP_N_MATCHED = {"acceptor_cluster|acceptor": 640, "donor_cluster|donor": 653, "cassette_exon|donor": 641}


# --------------------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------------------
def load_context_rows() -> list[dict]:
    rows = [json.loads(l) for l in CONTEXT_JSONL.read_text().splitlines() if l.strip()]
    ctx = [r for r in rows if r.get("context_associated") is True]
    return ctx


def load_supply() -> dict[str, dict]:
    s = json.loads(MATCHED_SUMMARY.read_text())
    disc = s["summaries"]["atlas_discovery"]
    out = {}
    for cell in CELLS:
        c = disc[cell]
        out[cell] = {
            "genes": c["n_case_genes_in_exact_other-tag_support"],
            "n_matched": c["n_matched"],
            "n_cases_sampled": c["n_cases_sampled"],
            "n_case_packets_support": c.get("n_case_packets_in_exact_other-tag_support"),
        }
    return out


def load_census_fraction() -> float:
    return float(json.loads(ATLAS_CENSUS.read_text())["junction_window_fit_fraction"])


# --------------------------------------------------------------------------------------------------
# Core power arithmetic (E1 demand, E2 supply)  — PREREGISTRATION_B5 §1
# --------------------------------------------------------------------------------------------------
def n_req_effective(n_eff: float, mde80: float, tau: float = TAU) -> float:
    """E1: effective matched pairs needed to reach the gate, holding observed dispersion fixed."""
    return n_eff * (mde80 / tau) ** 2


def cell_supply(cell: str, rows_in_cell: list[dict], supply: dict) -> dict:
    """E2: hard (yield=1.0) and realistic effective ceilings, unit-matched to effective pairs via DE."""
    g = supply[cell]["genes"]
    r = supply[cell]["n_matched"] / supply[cell]["n_cases_sampled"]  # observed yield ~0.80
    n_eff = rows_in_cell[0]["n_effective_pairs"]
    n_pairs = rows_in_cell[0]["n_pairs"]
    de = n_eff / n_pairs  # per-cell design factor (M9)
    return {
        "genes": g,
        "yield": r,
        "n_eff": n_eff,
        "n_pairs": n_pairs,
        "design_effect": de,
        "n_endpoint_genes": rows_in_cell[0].get("n_endpoint_genes"),
        "ceil_hard_eff": g * de,          # yield=1.0, the LEAD ceiling (most generous supply)
        "ceil_real_eff": g * r * de,      # observed yield, the realistic ceiling
    }


def signal_table(ctx: list[dict], supply: dict) -> list[dict]:
    by_cell: dict[str, list[dict]] = {cell: [] for cell in CELLS}
    for r in ctx:
        by_cell[r["contrast"]].append(r)
    cellsup = {cell: cell_supply(cell, by_cell[cell], supply) for cell in CELLS}
    out = []
    for r in ctx:
        cell = r["contrast"]
        cs = cellsup[cell]
        nreq = n_req_effective(r["n_effective_pairs"], r["mde80"])
        ceil_hard = cs["ceil_hard_eff"]
        closed = nreq > ceil_hard  # closed iff (lower-bound) demand exceeds even the hard-cap supply
        out.append({
            "component_id": r["component_id"],
            "contrast": cell,
            "category": r["category"],
            "classification": r["classification"],
            "component_type": r["component_type"],
            "mde80": r["mde80"],
            "n_effective_pairs": r["n_effective_pairs"],
            "n_pairs": r["n_pairs"],
            "multiplier": (r["mde80"] / TAU) ** 2,
            "n_req_effective": nreq,
            "ceil_hard_eff": ceil_hard,
            "ceil_real_eff": cs["ceil_real_eff"],
            "closed": closed,                                  # licensable (CLOSE)
            "reachable_diag": (not closed),                    # DIAGNOSTIC only (barred from disposition, §5/§6)
            "margin_over_hard_cap": nreq / ceil_hard,          # >1 => closed; reported (thin margins honest)
        })
    return out, cellsup


# --------------------------------------------------------------------------------------------------
# Memorials M1..M9  — PREREGISTRATION_B5 §3
# --------------------------------------------------------------------------------------------------
def run_memorials(ctx, supply, cellsup) -> list[dict]:
    m = []

    def add(name, passed, detail, forced=False):
        m.append({"name": name, "pass": bool(passed), "forced": forced, "detail": detail})

    from collections import Counter
    n = len(ctx)
    cls = Counter(r["classification"] for r in ctx)
    cat = Counter(r["category"] for r in ctx)
    ctype = Counter(r["component_type"] for r in ctx)
    add("M1_signal_counts",
        n == EXP_N_SIGNALS and dict(cls) == EXP_CLASS and dict(cat) == EXP_CATEGORY and dict(ctype) == EXP_COMPONENT_TYPE,
        {"n": n, "class": dict(cls), "category": dict(cat), "component_type": dict(ctype)})

    add("M2_gate_frozen", all(r["tau"] == TAU for r in ctx),
        {"tau_values": sorted(set(r["tau"] for r in ctx))})

    m3_bad = [r["component_id"] for r in ctx if abs(Z_POWER_80 * r["se_gene_dyadic"] - r["mde80"]) >= 1e-9]
    add("M3_precision_identity_roundtrip", not m3_bad,
        {"Z_POWER_80": Z_POWER_80, "n_fail_ge_1e-9": len(m3_bad), "worst_abs_err":
         max(abs(Z_POWER_80 * r["se_gene_dyadic"] - r["mde80"]) for r in ctx)})

    m4_bad = []
    for r in ctx:
        nreq = n_req_effective(r["n_effective_pairs"], r["mde80"])
        proj = r["mde80"] * math.sqrt(r["n_effective_pairs"] / nreq)
        if abs(proj - TAU) >= 1e-9:
            m4_bad.append(r["component_id"])
    add("M4_required_N_backout_consistency", not m4_bad,
        {"n_fail": len(m4_bad), "note": "FORCED algebraic identity of N_req def (R7/R21) — reported, not leaned on"},
        forced=True)

    n_matched_ok = all(supply[c]["n_matched"] == EXP_N_MATCHED[c] for c in CELLS)
    genes_ok = all(supply[c]["genes"] == EXP_SUPPLY_GENES[c] for c in CELLS)
    sampled_ok = all(supply[c]["n_cases_sampled"] == 800 for c in CELLS)
    add("M5_supply_counts_real", n_matched_ok and genes_ok and sampled_ok,
        {"n_matched": {c: supply[c]["n_matched"] for c in CELLS},
         "genes": {c: supply[c]["genes"] for c in CELLS},
         "n_cases_sampled": {c: supply[c]["n_cases_sampled"] for c in CELLS}})

    m6_ok = all(cellsup[c]["ceil_real_eff"] <= cellsup[c]["ceil_hard_eff"] + 1e-9
                and cellsup[c]["ceil_hard_eff"] <= cellsup[c]["genes"] + 1e-9 for c in CELLS)
    add("M6_ceiling_monotone_bounded", m6_ok,
        {c: {"real": cellsup[c]["ceil_real_eff"], "hard": cellsup[c]["ceil_hard_eff"], "genes": cellsup[c]["genes"]}
         for c in CELLS})

    # M7 (forced): permuting mean_difference must not change any N_req (the formula never reads it).
    base = {r["component_id"] + "|" + r["contrast"]: n_req_effective(r["n_effective_pairs"], r["mde80"]) for r in ctx}
    md = [r["mean_difference"] for r in ctx]
    shuffled = md[1:] + md[:1]  # deterministic rotation (no RNG)
    perm = {}
    for r, fake_md in zip(ctx, shuffled):
        _ = fake_md  # explicitly unused by the demand formula
        perm[r["component_id"] + "|" + r["contrast"]] = n_req_effective(r["n_effective_pairs"], r["mde80"])
    add("M7_effect_size_excluded", base == perm,
        {"note": "near-FORCED (R7): demand depends only on (se, tau); reported, not leaned on"}, forced=True)

    # M8 determinism: recompute the signal table twice, compare.
    t1, _ = signal_table(ctx, supply)
    t2, _ = signal_table(ctx, supply)
    add("M8_determinism", t1 == t2, {"note": "PYTHONHASHSEED pinned by caller; recompute identical"})

    de_ok = all(0.0 < cellsup[c]["design_effect"] <= 1.0 for c in CELLS)
    add("M9_design_effect_visible", de_ok,
        {c: {"design_effect": cellsup[c]["design_effect"], "n_endpoint_genes": cellsup[c]["n_endpoint_genes"]}
         for c in CELLS})
    return m


# --------------------------------------------------------------------------------------------------
# Calibration PP1..PP4, N1..N3  — PREREGISTRATION_B5 §4
# --------------------------------------------------------------------------------------------------
def classify_synthetic(mde80: float, n_eff: float, ceil_hard_eff: float) -> dict:
    """Route a synthetic row through the exact reachability logic used for real signals."""
    nreq = n_req_effective(n_eff, mde80)
    already_powered = nreq <= n_eff
    reachable = nreq <= ceil_hard_eff
    via_ceiling = reachable and not already_powered  # reachable only through the ceiling comparison
    return {"mde80": mde80, "n_req": nreq, "already_powered": already_powered,
            "reachable": reachable, "via_ceiling_channel": via_ceiling, "ceil_hard_eff": ceil_hard_eff}


def run_calibration(ctx, supply, cellsup, table) -> list[dict]:
    cal = []

    def add(name, passed, detail):
        cal.append({"name": name, "pass": bool(passed), "detail": detail})

    ref = cellsup["acceptor_cluster|acceptor"]
    n_eff_ref = ref["n_eff"]
    ceil_ref = ref["ceil_hard_eff"]  # ~1925

    # PP1: already-powered (mde80=0.04 < gate) -> reachable via already-powered shortcut.
    pp1 = classify_synthetic(0.04, n_eff_ref, ceil_ref)
    add("PP1_already_powered", pp1["reachable"] and pp1["already_powered"], pp1)

    # PP2: exact boundary mde80=0.05 -> N_req == n_eff (multiplier 1.0).
    pp2 = classify_synthetic(0.05, n_eff_ref, ceil_ref)
    add("PP2_boundary", abs(pp2["n_req"] - n_eff_ref) < 1e-6 and pp2["reachable"], pp2)

    # PP3: CEILING-CHANNEL reachable — target N_req in (n_eff, ceil): pick ~1200.
    mde_pp3 = TAU * math.sqrt(1200.0 / n_eff_ref)
    pp3 = classify_synthetic(mde_pp3, n_eff_ref, ceil_ref)
    add("PP3_ceiling_channel_reachable",
        pp3["reachable"] and pp3["via_ceiling_channel"] and (n_eff_ref < pp3["n_req"] < ceil_ref), pp3)

    # PP4: CEILING-CHANNEL not-reachable — target N_req just above ceil (~2000 > ~1925).
    mde_pp4 = TAU * math.sqrt(2000.0 / n_eff_ref)
    pp4 = classify_synthetic(mde_pp4, n_eff_ref, ceil_ref)
    add("PP4_ceiling_channel_not_reachable", (not pp4["reachable"]) and pp4["n_req"] > ceil_ref, pp4)

    # N1 (effect-permutation): closed/reachable flags unchanged under mean_difference permutation.
    base_flags = {t["component_id"] + "|" + t["contrast"]: t["closed"] for t in table}
    md = [r["mean_difference"] for r in ctx]
    shuffled = md[1:] + md[:1]
    perm_flags = {}
    for r, fake in zip(ctx, shuffled):
        _ = fake
        nreq = n_req_effective(r["n_effective_pairs"], r["mde80"])
        perm_flags[r["component_id"] + "|" + r["contrast"]] = nreq > cellsup[r["contrast"]]["ceil_hard_eff"]
    add("N1_effect_permutation_no_flip", base_flags == perm_flags, {"n": len(base_flags)})

    # N2 (ceiling provenance + load-bearing): each signal uses its OWN cell ceiling; correct pairing => all
    # closed; and the ceiling is load-bearing (>=1 closed-flag would change under a cross-cell mis-pairing).
    provenance_ok = all(abs(t["ceil_hard_eff"] - cellsup[t["contrast"]]["ceil_hard_eff"]) < 1e-9 for t in table)
    all_closed_correct = all(t["closed"] for t in table)
    # sensitivity: assign every signal the MAX hard ceiling across cells; does any flip open?
    max_ceil = max(cellsup[c]["ceil_hard_eff"] for c in CELLS)
    n_flip = sum(1 for t in table if t["closed"] and not (t["n_req_effective"] > max_ceil))
    add("N2_ceiling_provenance_load_bearing",
        provenance_ok and all_closed_correct and n_flip >= 1,
        {"provenance_ok": provenance_ok, "all_closed_correct": all_closed_correct,
         "n_flip_under_max_ceiling": n_flip, "max_ceil": max_ceil,
         "note": "ceiling is load-bearing: a wrong (too-generous) ceiling WOULD open >=1 signal"})

    # N3 (gate-relaxation tripwire): no context row may carry tau != 0.05.
    add("N3_gate_relaxation_tripwire", all(r["tau"] == TAU for r in ctx),
        {"note": "aborts on any tau != 0.05"})
    return cal


# --------------------------------------------------------------------------------------------------
# Verdict  — PREREGISTRATION_B5 §6
# --------------------------------------------------------------------------------------------------
def decide(table) -> dict:
    all_closed = all(t["closed"] for t in table)
    verdict = "CLOSE_AS_UNDERPOWERED" if all_closed else "NOT_ALL_CLOSED_INVESTIGATE"
    per_cell_min = {}
    for cell in CELLS:
        cell_rows = [t for t in table if t["contrast"] == cell]
        mn = min(cell_rows, key=lambda t: t["n_req_effective"])
        per_cell_min[cell] = {
            "min_component": mn["component_id"], "min_n_req_effective": mn["n_req_effective"],
            "ceil_hard_eff": mn["ceil_hard_eff"], "margin_over_hard_cap": mn["margin_over_hard_cap"],
        }
    return {"verdict": verdict, "all_closed": all_closed, "n_signals": len(table),
            "n_closed": sum(1 for t in table if t["closed"]), "per_cell_min": per_cell_min}


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------
def gates_pass(memorials, calibration) -> bool:
    return all(m["pass"] for m in memorials) and all(c["pass"] for c in calibration)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="arithmetic memorials + calibration only; no ceiling estimate, no verdict")
    ap.add_argument("--out", default=str(RESULTS_DIR / "b5_context_signal_power_v1.json"))
    args = ap.parse_args()

    ctx = load_context_rows()
    supply = load_supply()
    census_fraction = load_census_fraction()
    table, cellsup = signal_table(ctx, supply)

    memorials = run_memorials(ctx, supply, cellsup)
    calibration = run_calibration(ctx, supply, cellsup, table)
    passed = gates_pass(memorials, calibration)

    print(f"Z_POWER_80 = {Z_POWER_80!r}")
    print("--- memorials ---")
    for m in memorials:
        print(f"  [{'PASS' if m['pass'] else 'FAIL'}]{' (forced)' if m['forced'] else ''} {m['name']}")
    print("--- calibration ---")
    for c in calibration:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}")

    if args.smoke:
        print(f"\n--smoke Global gate: {'PASS' if passed else 'FAIL'} (no verdict emitted)")
        return 0 if passed else 1

    if not passed:
        print("\nGlobal PASS = FALSE -> run STOPS before writing a verdict.")
        return 1

    verdict = decide(table)
    print("\n--- per-cell minimum demand vs hard one-per-gene supply cap ---")
    for cell, v in verdict["per_cell_min"].items():
        print(f"  {cell}: min N_req_eff={v['min_n_req_effective']:.0f} ({v['min_component']}) "
              f"vs hard-cap {v['ceil_hard_eff']:.0f}  => {v['margin_over_hard_cap']:.3f}x over")
    print(f"\nVERDICT: {verdict['verdict']}  ({verdict['n_closed']}/{verdict['n_signals']} closed)")

    payload = {
        "experiment_id": "b5_context_signal_power_v1",
        "preregistration": "studies/splicebert_head_context_atlas/PREREGISTRATION_B5_CONTEXT_SIGNAL_POWER.md",
        "redteam_synthesis": "studies/splicebert_head_context_atlas/REDTEAM_SYNTHESIS_B5.md",
        "gate": {"tau": TAU, "Z_POWER_80": Z_POWER_80},
        "atlas_junction_window_fit_fraction": census_fraction,
        "verdict": verdict,
        "cell_supply": cellsup,
        "signals": table,
        "memorials": memorials,
        "calibration": calibration,
        "global_pass": passed,
        "notes": [
            "Model-free deterministic power arithmetic; zero forward passes.",
            "N_req is a SELECTED single-draw LOWER BOUND (R22) -> reachability is one-sided-falsifiable; the "
            "arithmetic can only license CLOSE, never a reachable/redesign disposition (see prereg §5/§6).",
            "Pooled-taxonomy reachability is NON-IDENTIFIABLE-AT-DESIGN (removed): a fine signal's se cannot be "
            "applied to the sign-heterogeneous, cross-role, gene-double-counting pooled cohort (R22).",
            "Supply wall = exact-other-tag support GENE pool (one-case-per-gene sampler), downstream of the "
            "~29% atlas junction-window-fit; demand and supply unit-matched to effective pairs via DE.",
            "HCA18 boundary honored: the 20 signals are retained as under-powered leads, NOT erased and NOT "
            "validated.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        experiment_id="b5_context_signal_power_v1",
        command=["python"] + sys.argv,
        repo=REPO_ROOT,
        status=verdict["verdict"],
        config={"tau": TAU, "Z_POWER_80": Z_POWER_80, "marginal_band": MARGINAL_BAND, "cells": list(CELLS)},
        seeds={"PYTHONHASHSEED": 0},
        notes=["B5 HCA18 context-signal power adjudication; deterministic, model-free (prereg v2 FROZEN)."],
    )
    for inp in (CONTEXT_JSONL, MATCHED_SUMMARY, ATLAS_CENSUS):
        manifest.add_input(inp)
    manifest.add_output(out_path)
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest.write(manifest_path)

    print(f"\nwrote {out_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
