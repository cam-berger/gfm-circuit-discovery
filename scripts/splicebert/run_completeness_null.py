#!/usr/bin/env python
"""SIZE-MATCHED NULLS for the completeness ablations.  Without these the ablations mean nothing.

run_faithfulness.py found that ablating all 32 layer-4/5 heads leaves only 21% of the coupling,
which reads as "the downstream stack is required to read L3H1's output".  But ablating 32 of 96
heads damages the model no matter WHICH 32 you pick.  The claim is only meaningful against the
distribution of 32-head ablations -- so build it.

For each proposed set S (15, 32 and 47 heads), draw ``--n-null`` random head sets of |S| from the
same universe (always EXCLUDING the circuit C, which must stay intact in every draw, exactly as it
does in the real ablation) and compare:

    coupling(ablate S)   vs   the distribution of coupling(ablate a random set of the same size)

If coupling(S) sits deep in the lower tail, the downstream stack is SPECIFICALLY required.
If it sits in the bulk, we have measured collateral damage and must say so.

MLM health is recorded for every draw, because a random set that happens to wreck the model is
not a fair comparison either -- the honest statistic is the coupling loss AT MATCHED HEALTH.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.rigor import empirical_upper_p_value  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CIRCUIT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask, ablate,
    load_adapter, load_examples, pick_device,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/completeness_null.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev)
    task = CouplingTask(ad, batch=args.batch)
    L3 = READER[0]
    CIRC = set(CIRCUIT)

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"compnull-{args.split}", min_effect=0.3)
    base = float(task.coupling(X).mean())
    h_base = task.mlm_health(X)
    print(f"{len(X)} {args.split} exons   baseline coupling {base:+.3f}   "
          f"MLM health {h_base:.4f}   device={dev}")

    SETS = {
        "layer-3 others (15)": [(L3, h) for h in range(ad.num_heads) if (L3, h) not in CIRC],
        "layer-4+5 (32)": [(l, h) for l in range(L3 + 1, ad.num_layers)
                           for h in range(ad.num_heads)],
    }
    SETS["L3-others + L4 + L5 (47)"] = SETS["layer-3 others (15)"] + SETS["layer-4+5 (32)"]

    # the universe a random draw may touch: every head EXCEPT the circuit (which the real
    # ablations also leave intact).  A null that is allowed to delete L3H1 is not matched.
    universe = [(l, h) for l in range(ad.num_layers) for h in range(ad.num_heads)
                if (l, h) not in CIRC]

    R: dict = {"split": args.split, "n": len(X), "baseline_coupling": base,
               "baseline_health": h_base, "sets": {}}
    for name, S in SETS.items():
        k = len(S)
        obs = float(task.coupling(X, ablate(ad, S)).mean())
        obs_h = task.mlm_health(X, ablate(ad, S))
        rng = random.Random(hash(name) & 0xFFFF)
        null_c, null_h = [], []
        for i in range(args.n_null):
            draw = rng.sample(universe, k)
            null_c.append(float(task.coupling(X, ablate(ad, draw)).mean()))
            null_h.append(task.mlm_health(X, ablate(ad, draw)))
            if i % 20 == 0:
                print(f"  {name}: null {i}/{args.n_null}", end="\r", flush=True)
        null_c = np.asarray(null_c)
        null_h = np.asarray(null_h)
        # p = P(a random size-matched set destroys AT LEAST as much coupling)
        p = empirical_upper_p_value(base - obs, (base - null_c).tolist())
        # health-matched comparison: only nulls that damaged the LM at least as much
        sick = null_h <= obs_h
        matched = null_c[sick]
        print(f"\n  {name:28s}  k={k}")
        print(f"    observed   coupling {obs:+.3f} ({100*obs/base:5.1f}% retained)"
              f"   health {obs_h:.4f}")
        print(f"    random     coupling {null_c.mean():+.3f} +- {null_c.std():.3f}"
              f"   (min {null_c.min():+.3f})   health {null_h.mean():.4f}")
        print(f"    => add-one p = {p:.4f}   "
              f"({int((null_c <= obs).sum())}/{args.n_null} random sets did as much damage)")
        if sick.sum() >= 5:
            print(f"    HEALTH-MATCHED ({int(sick.sum())} nulls with health <= {obs_h:.3f}): "
                  f"their coupling {matched.mean():+.3f} vs observed {obs:+.3f}")
        else:
            print(f"    HEALTH-MATCHED: only {int(sick.sum())} random sets damaged the LM as "
                  "much -- the observed set is unusually GENTLE on the LM for its size.")
        R["sets"][name] = dict(
            k=k, observed=obs, observed_health=obs_h, frac_retained=float(obs / base),
            null_mean=float(null_c.mean()), null_std=float(null_c.std()),
            null_min=float(null_c.min()), null_health_mean=float(null_h.mean()),
            p_add_one=float(p), n_null=args.n_null,
            n_random_at_least_as_damaging=int((null_c <= obs).sum()),
            n_health_matched=int(sick.sum()),
            health_matched_coupling=float(matched.mean()) if sick.sum() else None,
        )

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:completeness_null", sys.argv,
        config={"split": args.split, "n": len(X), "n_null": args.n_null},
        notes=["Random draws EXCLUDE the circuit, exactly as the real ablations do.",
               "Reports both the raw null and a health-matched comparison: a large ablation "
               "damages the LM whatever it deletes, and that damage must be priced in."],
    ).write(out.parent / "completeness_null.manifest.json")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
