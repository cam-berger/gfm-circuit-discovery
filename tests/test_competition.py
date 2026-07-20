"""Cohort-builder correctness + determinism for the Head-Context Atlas (intron-mate competition).

Locks the corrected intron-mate definitions (NOT the exon-mate `attested` map) and the geometry gate,
and asserts hash-seed determinism (the IC1/shuffle-reproducibility house rule).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.gtf import ExonRecord, TranscriptRecord
from gfmcd.splicing.competition import (
    CompetitionConfig, build_competition_index, enumerate_competition_examples, iter_junctions, split_for,
)


def _exon(chrom, start, end, strand, gene, tx, tags=()):
    return ExonRecord(chrom, start, end, strand, gene, tx, None, "protein_coding", "protein_coding", tags)


def _fixture():
    # Gene S (short intron ~300bp) on chr8 (sealed): usable competition (both arms).
    sA = TranscriptRecord("s1", "S", "chr8", "+", "protein_coding", ("MANE_Select",),
        (_exon("chr8", 1000, 1100, "+", "S", "s1", ("MANE_Select",)),
         _exon("chr8", 1400, 1500, "+", "S", "s1", ("MANE_Select",))))
    sB = TranscriptRecord("s2", "S", "chr8", "+", "protein_coding", (),
        (_exon("chr8", 1000, 1160, "+", "S", "s2"), _exon("chr8", 1400, 1500, "+", "S", "s2")))
    sC = TranscriptRecord("s3", "S", "chr8", "+", "protein_coding", (),
        (_exon("chr8", 1000, 1100, "+", "S", "s3"), _exon("chr8", 1460, 1560, "+", "S", "s3")))
    # Gene G (3.9kb intron): competition exists but not window-constructible.
    g1 = TranscriptRecord("t1", "G", "chr1", "+", "protein_coding", (),
        (_exon("chr1", 1000, 1100, "+", "G", "t1"), _exon("chr1", 5000, 5100, "+", "G", "t1")))
    g2 = TranscriptRecord("t2", "G", "chr1", "+", "protein_coding", (),
        (_exon("chr1", 1000, 1300, "+", "G", "t2"), _exon("chr1", 5000, 5100, "+", "G", "t2")))
    return [sA, sB, sC, g1, g2]


def test_intron_mate_junctions():
    js = list(iter_junctions(_fixture()))
    # short-intron gene S: donor 1100->acc 1400 (MANE), donor 1160->acc 1400, donor 1100->acc 1460
    s = {(j.donor, j.acceptor) for j in js if j.gene_id == "S"}
    assert (1100, 1400) in s and (1160, 1400) in s and (1100, 1460) in s
    assert any(j.is_mane and (j.donor, j.acceptor) == (1100, 1400) for j in js)


def test_both_competition_types_and_geometry_gate():
    idx = build_competition_index(_fixture())
    exs = enumerate_competition_examples(idx, CompetitionConfig(min_separation=30))
    assert any(e.site_type == "alt5ss_donor" for e in exs)
    assert any(e.site_type == "alt3ss_acceptor" for e in exs)
    short = [e for e in exs if e.gene_id == "S"]
    assert short and all(e.usable for e in short), [(e.site_type, e.usable, e.span) for e in short]
    assert all(e.split == "test" for e in short)  # chr8 sealed
    longg = [e for e in exs if e.gene_id == "G"]
    assert longg and all(not e.in_window for e in longg)  # 3.9kb intron out of window


def test_min_separation_excludes_tandem():
    # donors 1100 & 1112 (12nt apart) share acceptor 1400 -> tandem, excluded at min_sep=30
    tx1 = TranscriptRecord("a1", "T", "chr2", "+", "protein_coding", (),
        (_exon("chr2", 1000, 1100, "+", "T", "a1"), _exon("chr2", 1400, 1500, "+", "T", "a1")))
    tx2 = TranscriptRecord("a2", "T", "chr2", "+", "protein_coding", (),
        (_exon("chr2", 1000, 1112, "+", "T", "a2"), _exon("chr2", 1400, 1500, "+", "T", "a2")))
    exs = enumerate_competition_examples(build_competition_index([tx1, tx2]),
                                         CompetitionConfig(min_separation=30))
    donor_exs = [e for e in exs if e.site_type == "alt5ss_donor"]
    assert donor_exs and all(e.is_tandem and not e.usable for e in donor_exs)


def test_split_assignment():
    assert split_for("chr8") == "test" and split_for("chr9") == "test"
    assert split_for("chr11") == "val" and split_for("chr16") == "val" and split_for("chr19") == "val"
    assert split_for("chr1") == "train" and split_for("7") == "train"


def test_inprocess_determinism():
    idx = build_competition_index(_fixture())
    a = [e.example_id for e in enumerate_competition_examples(idx, CompetitionConfig())]
    b = [e.example_id for e in enumerate_competition_examples(build_competition_index(_fixture()), CompetitionConfig())]
    assert a == b and a == sorted(a, key=a.index)  # stable, reproducible order


_CROSS_SEED_SNIPPET = """
import sys
sys.path.insert(0, r"{src}")
from gfmcd.splicing.gtf import ExonRecord, TranscriptRecord
from gfmcd.splicing.competition import build_competition_index, enumerate_competition_examples, CompetitionConfig
def ex(c,s,e,st,g,t): return ExonRecord(c,s,e,st,g,t,None,"protein_coding","protein_coding",())
txs=[TranscriptRecord(f"t{{i}}","S","chr8","+","protein_coding",(),
       (ex("chr8",1000,1100+10*i,"+","S",f"t{{i}}"), ex("chr8",1400,1500,"+","S",f"t{{i}}"))) for i in range(6)]
exs=enumerate_competition_examples(build_competition_index(txs), CompetitionConfig())
print("|".join(e.example_id for e in exs))
"""


def test_cross_hashseed_determinism(tmp_path):
    """Set iteration must not depend on PYTHONHASHSEED (sorted(set) discipline)."""
    snippet = _CROSS_SEED_SNIPPET.format(src=str(REPO / "src"))
    f = tmp_path / "snip.py"
    f.write_text(snippet)
    outs = []
    for seed in ("0", "1", "2", "17"):
        env = {"PYTHONHASHSEED": seed, "PATH": __import__("os").environ.get("PATH", "")}
        r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert len(set(outs)) == 1, f"cohort order varied across PYTHONHASHSEED: {outs}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
