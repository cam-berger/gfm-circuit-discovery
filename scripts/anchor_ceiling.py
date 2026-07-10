"""How well can ANY sequence-only model find the translation start? That bounds the circuit.

A counter is only as good as its anchor. If the start codon cannot be located from sequence, a
"find the landmark and count" circuit cannot beat a bag of motifs no matter what the objective is,
and mu* does not exist on natural real CDS.

Two accuracies matter, and only one of them governs the circuit:

    exact   predicted start == true start
    frame   predicted start == true start (mod 3)   <-- this one. An in-frame decoy is harmless.

Features are strictly sequence-derived. Positional features are excluded on purpose: our sampler
places the true start at a uniformly random offset <= 163, which is an artifact of window
construction, not biology, and a model with position embeddings could exploit it.

    python scripts/anchor_ceiling.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyfaidx import Fasta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from gfmcd.data.genome import SPLIT_CHROMS, Split, atg_index, parse_gencode_cds, sample_anchored_windows, split_by_anchor
from gfmcd.paths import project_paths

DATA = project_paths().references
A, C, G, T = 0, 1, 2, 3
STOPS = {(T, A, A), (T, A, G), (T, G, A)}
CTX = 12          # +-12 nt of one-hot context around each candidate ATG
ORF_SCAN = 90     # nt downstream to scan for an in-frame stop


def candidates(tok: np.ndarray) -> list[int]:
    w = len(tok)
    return [p for p in range(w - 2) if (tok[p], tok[p + 1], tok[p + 2]) == (A, T, G)]


def features(tok: np.ndarray, p: int, all_atgs: list[int]) -> np.ndarray:
    w = len(tok)
    ctx = np.zeros((2 * CTX + 1, 4))
    for j, off in enumerate(range(-CTX, CTX + 1)):
        q = p + off
        if 0 <= q < w and tok[q] < 4:
            ctx[j, tok[q]] = 1.0

    # ORF integrity: an authentic start has no in-frame stop close downstream.
    stops = 0
    orf_len = ORF_SCAN
    for k in range(p + 3, min(p + 3 + ORF_SCAN, w - 2), 3):
        if (tok[k], tok[k + 1], tok[k + 2]) in STOPS:
            stops += 1
            orf_len = min(orf_len, k - p)
            break

    n_before = sum(1 for q in all_atgs if q < p)
    return np.concatenate([ctx.ravel(), [stops, orf_len / ORF_SCAN, n_before, float(n_before == 0)]])


def build(n: int, split: Split, seed: int):
    fasta = Fasta(str(DATA / "hg38.fa"), as_raw=True, sequence_always_upper=True)
    cds = parse_gencode_cds(DATA / "gencode.v50.basic.annotation.gtf.gz", chroms=SPLIT_CHROMS[split], min_len=150)
    starts, _ = split_by_anchor(cds, fasta)
    tok, lab = sample_anchored_windows(starts, fasta, n=n, window=256, seed=seed)
    true = atg_index(tok, lab)

    X, y, group = [], [], []
    for r in range(len(tok)):
        cands = candidates(tok[r])
        if true[r] not in cands:
            continue
        for p in cands:
            X.append(features(tok[r], p, cands))
            y.append(int(p == true[r]))
            group.append((r, p, true[r]))
    return np.array(X), np.array(y), group


def evaluate(scores: np.ndarray, group: list) -> dict:
    by_win: dict[int, list] = {}
    for s, (r, p, t) in zip(scores, group):
        by_win.setdefault(r, []).append((s, p, t))

    exact = frame = n = 0
    exact_amb = frame_amb = n_amb = 0
    for r, rows in by_win.items():
        rows.sort(reverse=True)
        s, pred, t = rows[0]
        ambiguous = len(rows) > 1 and any(p < t for _, p, _ in rows)  # an upstream decoy exists
        exact += pred == t
        frame += (pred - t) % 3 == 0
        n += 1
        if ambiguous:
            exact_amb += pred == t
            frame_amb += (pred - t) % 3 == 0
            n_amb += 1
    return dict(exact=exact / n, frame=frame / n, n=n,
                exact_amb=exact_amb / max(n_amb, 1), frame_amb=frame_amb / max(n_amb, 1), n_amb=n_amb)


def main() -> None:
    Xtr, ytr, _ = build(6000, Split.TRAIN, 5)
    Xte, yte, gte = build(2000, Split.VAL, 6)
    print(f"train candidates {Xtr.shape}   val candidates {Xte.shape}\n")

    for name, model in [
        ("logistic regression", LogisticRegression(max_iter=2000, C=1.0)),
        ("gradient boosting", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1)),
    ]:
        model.fit(Xtr, ytr)
        s = model.predict_proba(Xte)[:, 1]
        r = evaluate(s, gte)
        print(f"{name}")
        print(f"  exact start accuracy            {r['exact']:.4f}")
        print(f"  FRAME accuracy (mod 3)          {r['frame']:.4f}   <- governs the circuit")
        print(f"  ...on windows with an upstream ATG ({r['n_amb']}/{r['n']}):")
        print(f"     exact {r['exact_amb']:.4f}   frame {r['frame_amb']:.4f}\n")

    # what the frame accuracy implies for the circuit
    model.fit(Xtr, ytr)
    d = evaluate(model.predict_proba(Xte)[:, 1], gte)["frame"]
    circuit = d + (1 - d) * 0.07  # wrong frame still hits the right amino acid ~7% of the time
    print(f"implied circuit ceiling on real CDS: {circuit:.4f}   (bag the model reaches: ~0.80)")
    print(f"the circuit buys: {circuit - 0.80:+.4f}")


if __name__ == "__main__":
    main()
