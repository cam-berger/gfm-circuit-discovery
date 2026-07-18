#!/usr/bin/env python
"""v8 determinism test: prove the IC1 shuffler + eligibility are PYTHONHASHSEED-INDEPENDENT.

Root cause of the v6/v7 cohort drift (255-261): `set(seq)` char-set iteration in dinuc_shuffle is
hashseed-dependent. Fixed to `sorted(set(seq))`. This test spawns child processes under several
PYTHONHASHSEED values and asserts IDENTICAL shuffle sequences, eligible-cohort membership, and counts.
Run: `python scripts/splicebert/test_ic1_determinism.py`  (parent orchestrates the children).
"""
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/splicebert")

if os.environ.get("IC1_DETERM_CHILD"):
    import numpy as np
    from run_ic1_pilot import dinuc_shuffle, anchored_shuffle, unconstrained_shuffle
    from run_ic1_pilot3 import conservative_scramble
    from run_ic1_distal_intron_necessity import gen_arms, dose_ok, shash, elig_window, N_SEED
    from gfmcd.splicing.circuit import eligible, _dataset, DEFAULT_GTF, DEFAULT_FASTA

    W = "ACGTGCATTTACGGATCCGATTACAGGCATATATCGCGCGATTACGGTACAGTAG"     # fixed test window
    shufs = "|".join([dinuc_shuffle(W, np.random.default_rng(0)),
                      anchored_shuffle(W, np.random.default_rng(1)),
                      unconstrained_shuffle(W, np.random.default_rng(2)),
                      str(conservative_scramble(W, np.random.default_rng(3), 400))])
    print("SHUF", hashlib.sha256(shufs.encode()).hexdigest()[:16])

    tl = eligible(_dataset(DEFAULT_GTF, DEFAULT_FASTA), "val")
    seen, dd = set(), []
    for x in tl:
        if x.gene_id not in seen:
            seen.add(x.gene_id); dd.append(x)
    members = []
    for x in dd[:400]:                                              # capped for test speed
        ok, lo, hi = elig_window(x, 70)
        if not ok:
            continue
        Wx = x.sequence[lo:hi + 1]
        if not dose_ok(Wx, [anchored_shuffle(Wx, np.random.default_rng(shash(x.gene_id) * 7 + s))
                            for s in range(N_SEED)]):
            continue
        if gen_arms(x, lo, hi, Wx) is None:
            continue
        members.append(x.gene_id)
    print("NELIG", len(members))
    print("MEMBERS", hashlib.sha256("|".join(members).encode()).hexdigest()[:16])
    sys.exit(0)

env = dict(os.environ); env["IC1_DETERM_CHILD"] = "1"
results = {}
for hs in ("0", "1", "2", "17"):
    env["PYTHONHASHSEED"] = hs
    out = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                         capture_output=True, text=True)
    d = {}
    for line in out.stdout.strip().splitlines():
        p = line.split(maxsplit=1)
        if p and p[0] in ("SHUF", "NELIG", "MEMBERS"):
            d[p[0]] = p[1]
    results[hs] = d
    print(f"HASHSEED={hs}: {d}")
    if not d:
        print("child stderr:", out.stderr[-500:])

base = results["0"]
passed = all(results[hs] == base for hs in results) and bool(base)
print("=" * 60)
print(f"DETERMINISM_PASS = {passed}  "
      f"(shuffles + n_elig + membership identical across PYTHONHASHSEED {list(results)})")
sys.exit(0 if passed else 1)
