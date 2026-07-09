"""Phase C, test T3: relocate the anchor instead of deleting it.

Registered in `docs/PREREGISTRATION.md` (2026-07-09 b) as the control that decides whether D33's whole
ablation column measures anchor use or distribution shift.

Deletion removes the ATG. Relocation keeps an ATG -- same token, same local context, one or two bases
over -- but makes it disagree with the local codon structure about the frame:

    intact      ...  A T G | C x x | ...          anchor and content agree
    shift+1     ...  C A T G | x x  ...           anchor says frame+1, content still says frame+0
    shift+2     ...  C T A T G | x  ...           anchor says frame+2
    shift+3     ...  C T C A T G | ...            anchor moved, frame unchanged (positional control)

Then read the model's far-field output against two label sets computed on the *same edited sequence*:
the frame the content implies, and the frame the relocated anchor implies.

    follows the anchor           -> the anchor is a frame source        (H1 counter, H2 chain)
    follows the content          -> the anchor is a gate, not a source  (registered outcome (ii))
    follows neither (chance)     -> the edit is an OOD shock; D33's ablation column means nothing

    python scripts/phase_c_t3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
from phase_c_relay import MODELS, load, per_position  # noqa: E402
from train_cell import pool  # noqa: E402

from gfmcd.data.genome import IGNORE, Split, amino_acid_labels  # noqa: E402
from gfmcd.signatures import ATG, NON_START, ablate, start_index  # noqa: E402

OUT = Path("results/fig_data")
N = 800
FAR = 64


def relocate(tok: np.ndarray, start: np.ndarray, offset: int) -> np.ndarray:
    """Destroy the ATG at `start`, write one at `start + offset`. Touches offset + 3 bases."""
    out = tok.copy()
    for r, s in enumerate(start):
        out[r, s : s + 3] = NON_START
        out[r, s + offset : s + offset + 3] = ATG
    return out


def frame_labels(tok: np.ndarray, body: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Amino-acid labels for the frame implied by `anchor`, on the edited sequence."""
    pos = np.arange(tok.shape[1])[None, :]
    ph = np.where(body, (pos - anchor[:, None]) % 3, IGNORE)
    return amino_acid_labels(tok, ph)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    for key, ckpt, cell in MODELS:
        model = load(ckpt, device)
        tok, aa, ph = pool(cell, N, model.cfg.max_len, Split.VAL, 4242)
        start = start_index(ph)
        body = ph >= 0
        pos = np.arange(tok.shape[1])[None, :]
        far = body & (pos - start[:, None] >= FAR) & (aa != IGNORE)

        rows = {}
        c_int, _ = per_position(model, tok, aa, device)
        rows["intact"] = dict(content=float(c_int[far].mean()), anchor=float(c_int[far].mean()))

        abl = ablate(tok, start)
        c_abl, _ = per_position(model, abl, frame_labels(abl, body, start), device)
        rows["deleted"] = dict(content=float(c_abl[far].mean()), anchor=float("nan"))

        for off in (1, 2, 3):
            ed = relocate(tok, start, off)
            aa_content = frame_labels(ed, body, start)          # frame the local codons still carry
            aa_anchor = frame_labels(ed, body, start + off)     # frame the relocated ATG announces
            pred, _ = per_position(model, ed, aa_content, device)
            pred_a, _ = per_position(model, ed, aa_anchor, device)
            m = far & (aa_content != IGNORE) & (aa_anchor != IGNORE)
            rows[f"shift+{off}"] = dict(content=float(pred[m].mean()), anchor=float(pred_a[m].mean()),
                                        n=int(m.sum()))

        results[key] = rows
        print(f"\n=== {key}   (far field, >= {FAR} nt from the start)")
        print(f"{'condition':>12}{'follows content':>18}{'follows anchor':>17}")
        for name, r in rows.items():
            a = "     --" if np.isnan(r["anchor"]) else f"{r['anchor']:>17.4f}"
            print(f"{name:>12}{r['content']:>18.4f}{a}")

    (OUT / "phase_c_t3.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}/phase_c_t3.json")


if __name__ == "__main__":
    main()
