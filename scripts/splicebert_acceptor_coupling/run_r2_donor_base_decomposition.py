#!/usr/bin/env python
"""R2 — which base of the donor GT carries the reverse coupling?

WHY THIS EXISTS.  AC2 originally read "the reverse operator's effect is governed by whether
`seq[d+1]` is rewritten, and by nothing else", from a nine-alt panel.  That panel could not
identify the claim: across all nine, `z[1]=="T"` held **if and only if** `edit_size==1`, so
"keeps the T" and "changes one base" were perfectly confounded -- and, worse, every alt that
rewrote the T *also* destroyed the G, so "rewrites the T" and "destroys both bases" were the
same set.  The panel contained two of the three reachable cells and neither of the two
alternative readings could be separated.

The missing cell is an alt that destroys the T while KEEPING the G, i.e. `z[0]=="G"`:

    GA   G kept, T -> A     admissible where seq[d+2] != "G"   (~37% of val donors)
    GG   G kept, T -> G     admissible where seq[d+2] != "T"   (~84% of val donors)
    GC   excluded -- "GC" IS the non-canonical U2 donor, 0% admissible by R44's own rule

PRE-STATED READINGS (written before the arm ran; see CORRECTIONS.md ACR4):
    rho(T-only) ~ 0.33  -> the two bases contribute additively (0.66 + 0.33 ~ 1.00)
    rho(T-only) ~ 0.99  -> destroying the T alone suffices; the G is redundant here
    rho(T-only) ~ 0.00  -> the T carries nothing; the 0.67 -> 1.00 gap is edit size after all,
                           and AC2's "governed by seq[d+1]" clause is refuted

VERIFICATION TIER.  Validation split only; chr8/9 not opened.  This arm exists to CORRECT a
claim, not to establish a new one -- it licenses the wording of AC2 and nothing beyond it.

    uv run python scripts/splicebert_acceptor_coupling/run_r2_donor_base_decomposition.py \
        --out results/splicebert_acceptor_coupling/r2_donor_base_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot_ratio, load_adapter,
    load_examples, matched_decoy_acceptor, matched_decoy_donor, pick_device,
)
from gfmcd.splicing.reverse_coupling import (  # noqa: E402
    acceptor_readout, destroy_donor, donor_alt_admissible, edit_size, margins_at,
)

REFERENCE_ALT = "AC"          # destroys both bases; R1b's operator, so rho is on R1b's scale
ARMS = ("AC", "AT", "GA", "GG")
DRAW_TAG = "r1b-d0"           # the R1b primary cohort, so numbers are directly comparable
MIN_N = 50
N_BOOT = 2000
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=False)
    task = CouplingTask(ad, batch=args.batch)
    X, _ = load_examples(args.gtf, args.fasta, "val", args.n, task=task, tag=DRAW_TAG,
                         min_effect=None)

    dd = [matched_decoy_donor(x) for x in X]
    da = [matched_decoy_acceptor(x, 40) for x in X]
    P = [X[i] for i in range(len(X)) if dd[i] is not None and da[i] is not None]
    g = [x.gene_id for x in P]
    a_pos = [acceptor_readout(x) for x in P]

    a_clean = margins_at(task, a_pos, [x.sequence for x in P])
    ref = a_clean - margins_at(task, a_pos, [destroy_donor(x, REFERENCE_ALT) for x in P])

    arms = {}
    for z in ARMS:
        adm = [i for i, x in enumerate(P) if donor_alt_admissible(x, z)]
        if len(adm) < MIN_N:
            arms[z] = dict(n=len(adm), status="INSUFFICIENT_SUPPORT")
            continue
        v = a_clean[adm] - margins_at(task, [a_pos[i] for i in adm],
                                      [destroy_donor(P[i], z) for i in adm])
        r = ref[adm]
        lo, hi = gboot_ratio(v, r, [g[i] for i in adm], n=N_BOOT, seed=SEED)
        arms[z] = dict(n=len(adm), admissible_frac=round(len(adm) / len(P), 3),
                       edit_size=edit_size("GT", z),
                       destroys="".join(b for b, m in zip("GT", z) if b != m) or "(none)",
                       keeps_G_at_d=(z[0] == "G"), keeps_T_at_d_plus_1=(z[1] == "T"),
                       mean=float(v.mean()), rho=float(v.sum() / r.sum()),
                       rho_ci=[float(lo), float(hi)])
        a = arms[z]
        print(f"  {z}: destroys {a['destroys']:<3} edit={a['edit_size']} n={a['n']:4d} "
              f"rho={a['rho']:.3f} [{a['rho_ci'][0]:.3f},{a['rho_ci'][1]:.3f}]", flush=True)

    g_only = arms.get("AT", {}).get("rho")
    t_only = [arms[z]["rho"] for z in ("GA", "GG") if "rho" in arms.get(z, {})]
    additivity = None
    if g_only is not None and t_only:
        additivity = dict(
            g_only=g_only, t_only_mean=sum(t_only) / len(t_only), both=1.0,
            sum_of_singles=g_only + sum(t_only) / len(t_only),
            NOTE="both-bases is the reference and is 1.0 by construction; sum_of_singles > 1 "
                 "means the two single-base destructions are sub-additive (saturating)")

    R = dict(
        experiment="R2_donor_base_decomposition", tier="VERIFICATION", confirmatory=False,
        split="val", sealed_split_opened=False, purpose="identify AC2; licenses AC2's wording only",
        cohort=dict(n_paired=len(P), n_genes=len(set(g)), draw_tag=DRAW_TAG, device=dev,
                    reference_alt=REFERENCE_ALT),
        reference_mean=float(ref.mean()), arms=arms, additivity=additivity,
        REFLECTION=dict(
            identification="The nine-alt panel behind AC2's original wording contained only two "
                           "of three cells: 'destroy both' and 'destroy G only'. Every alt that "
                           "rewrote the T also destroyed the G, and z[1]=='T' held iff "
                           "edit_size==1, so three readings were indistinguishable. GA/GG supply "
                           "the 'destroy T only' cell and separate them.",
            could_not_have_failed=[
                "AC is the reference; its rho is 1.000 by construction and is not evidence",
                "GC is excluded because it IS a non-canonical U2 donor -- 0% admissible under "
                "R44's own rule, not a power failure",
            ],
            residual_confound="GA and GG are the only admissible T-only alts and both RETAIN the "
                              "G at d. A 'destroy T, also destroy G' cell is by definition the "
                              "both-bases arm, so retaining G is not separable from destroying T "
                              "alone. GA and GG additionally sit on different admissibility "
                              "sub-cohorts (37% vs 84%) and are reported separately for that "
                              "reason; their agreement is the only cross-check available.",
        ),
    )

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(R, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r2_donor_base_decomposition", sys.argv,
        config=R["cohort"], seeds=dict(bootstrap=SEED), status="VERIFICATION",
        notes=["Validation split only; sealed chr8/9 not opened.",
               "Corrects AC2 (see CORRECTIONS.md ACR4); licenses no new claim."])
    for f in (args.model, args.gtf, args.fasta):
        try:
            man.add_input(f)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f}")
    man.add_output(p)
    man.write(p.with_suffix(".manifest.json"))
    if additivity:
        print(f"\n  G-only {additivity['g_only']:.3f} + T-only {additivity['t_only_mean']:.3f} "
              f"= {additivity['sum_of_singles']:.3f} against both-bases 1.000 (sub-additive)")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
