"""ED1 memorial `determinism_hashseed` — cohort, sham choice and matching must not depend
on PYTHONHASHSEED.

This is the IC1-Amendment-A2 bug class: ``for x in set(...)`` over strings iterates in
PYTHONHASHSEED-dependent order, which is deterministic *within* a process and varying
*across* processes, with no visible random call anywhere.  It silently made an IC1 cohort
non-reproducible (255-261 introns).  ED1 selects shams by sorting and matches in
``example_id`` order, so it should be immune — this test is what makes "should" checkable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDS = ("0", "1", "17")

_PROBE = r"""
import json, sys
sys.path.insert(0, %(src)r)
from gfmcd.splicing.coupling import CouplingExample
from gfmcd.splicing.exon_intron import (
    ShamPolicy, architecture_cell, axis_spec, match_pairs, sham_positions, EXACT_KEYS,
    dyadic_bootstrap, signflip_p,
)

# --- deterministic synthetic cohort (no genome/annotation needed) --------------------
import random
rng = random.Random(12345)          # explicit stream: unaffected by PYTHONHASHSEED
out = {"sham": [], "match": None, "boot": None}
examples = []
for i in range(40):
    seq = "".join(rng.choice("ACGT") for _ in range(1024))
    a, d = 200, 320
    na = 520
    seq = seq[:a-2] + "AG" + seq[a:]
    seq = seq[:d] + "GT" + seq[d+2:]
    seq = seq[:na-2] + "AG" + seq[na:]
    other = tuple(p for p in range(4, 1020)
                  if seq[p-2:p] == "AG" and abs(p-a) > 10 and abs(p-d) > 10 and abs(p-na) > 10)
    examples.append(CouplingExample(
        example_id=f"ex{i:03d}", sequence=seq, acceptor_pos=a, donor_pos=d,
        chrom="chr1", strand="+", gene_id=f"g{i%%7}", split="train",
        exon_length=d-a, intron_length_down=na-d, n_isoforms_donor=5,
        n_isoforms_acceptor=5, gene_type="protein_coding",
        intron_length_up=150 if (i // 2) %% 2 else 900, other_ag=other,
        next_acceptor_pos=na))
for x in examples:
    out["sham"].append([sham_positions(x, "exon", ShamPolicy()),
                        sham_positions(x, "intron", ShamPolicy())])

# --- matching on a cohort with genuinely varied covariates ---------------------------
rows = []
for i, x in enumerate(examples):
    rows.append(dict(
        example_id=x.example_id, gene=x.gene_id, gene_type=x.gene_type,
        up_band="S" if x.intron_length_up <= 200 else "L",
        dn_band="S" if i %% 2 else "L",
        L_exon=100 + (i %% 5), gc_exon=0.5 + 0.001*(i %% 9), gc_intron=0.5 + 0.001*(i %% 6),
        s_acc=4.0 + 0.01*(i %% 11), s_next=5.0 + 0.01*(i %% 7), s_don=4.0 + 0.01*(i %% 13),
        n_iso_don=5 + i %% 3, n_iso_acc=5 + i %% 4, donor_pos=320 + i, acc_pos=200 + i,
        L_up=x.intron_length_up, L_down=x.intron_length_down,
        n_real_acc=i %% 3, n_real_don=i %% 2))
cov, cal = axis_spec("down")
pairs = match_pairs(rows, [r["dn_band"] == "L" for r in rows],
                    covariates=cov, calipers=cal, exact=EXACT_KEYS["down"])
out["match"] = [(rows[i]["example_id"], rows[j]["example_id"]) for i, j in pairs]
out["cells"] = [architecture_cell(x) for x in examples]

# --- inference must be seeded, not hash-ordered --------------------------------------
vals = [0.1 * ((i %% 7) - 3) for i in range(40)]
ga = [f"g{i%%7}" for i in range(40)]
gb = [f"g{(i+3)%%7}" for i in range(40)]
out["boot"] = list(dyadic_bootstrap(vals, ga, gb, n_boot=200, seed=0))
out["flip"] = signflip_p(vals, n_perm=200, seed=0)
print(json.dumps(out))
"""


def _run(seed: str) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    src = str(REPO / "src")
    r = subprocess.run([sys.executable, "-c", _PROBE % {"src": src}],
                       capture_output=True, text=True, env=env, cwd=str(REPO))
    if r.returncode != 0:
        pytest.fail(f"probe failed at PYTHONHASHSEED={seed}:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout)


def test_ed1_pipeline_is_hashseed_independent():
    ref = _run(SEEDS[0])
    assert ref["match"], "matching produced no pairs — the probe cohort is degenerate"
    assert any(s[0] or s[1] for s in ref["sham"]), "no shams found — probe is degenerate"
    for seed in SEEDS[1:]:
        got = _run(seed)
        assert got["sham"] == ref["sham"], f"sham selection varies at PYTHONHASHSEED={seed}"
        assert got["match"] == ref["match"], f"matching varies at PYTHONHASHSEED={seed}"
        assert got["cells"] == ref["cells"], f"architecture cells vary at PYTHONHASHSEED={seed}"
        assert got["boot"] == ref["boot"], f"dyadic bootstrap varies at PYTHONHASHSEED={seed}"
        assert got["flip"] == ref["flip"], f"sign-flip p varies at PYTHONHASHSEED={seed}"
