#!/usr/bin/env python
"""Derive exact absolute x%3 attention summaries from a P3 shift diagnostic artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import DEFAULT_FASTA, DEFAULT_GTF, _dataset  # noqa: E402


def absolute_mass(relative_mass: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Convert mass indexed by (x-origin)%3 to mass indexed by absolute x%3."""
    relative_mass = np.asarray(relative_mass, dtype=float)
    origin = np.asarray(origin, dtype=int)
    out = np.empty_like(relative_mass)
    for i, o in enumerate(origin):
        for absolute_class in range(3):
            out[i, absolute_class] = relative_mass[i, (absolute_class - o) % 3]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/phase-lattice/p3_shift_diagnostic_v2.json")
    ap.add_argument("--out", default="results/phase-lattice/p3_shift_diagnostic_v2.mod3.json")
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    args = ap.parse_args()

    source = REPO / args.input
    result = json.loads(source.read_text())
    examples = {x.example_id: x for x in _dataset(args.gtf, args.fasta)}
    ids = result["cohort"]["example_ids"]
    if any(i not in examples for i in ids):
        raise RuntimeError("source cohort cannot be reconstructed from the pinned data")
    base_donors = np.asarray([examples[i].donor_pos for i in ids], dtype=int)

    summary: dict = {}
    for state in ("intact", "deleted"):
        summary[state] = {}
        for kind in result["config"]["kinds"]:
            summary[state][kind] = {}
            peak0 = np.asarray(result["cells"][state][kind]["0"]["peak_abs_mod3"], dtype=int)
            for shift in result["config"]["shifts"]:
                # Intron insertion leaves the donor fixed; every other operation translates it.
                donors = base_donors if kind == "intron_insert" else base_donors + shift
                cell = result["cells"][state][kind][str(shift)]
                rel = np.asarray(cell["mass_rel_donor_mod3"], dtype=float)
                absolute = absolute_mass(rel, donors)
                peaks = np.asarray(cell["peak_abs_mod3"], dtype=int)
                delta = (peaks - peak0) % 3
                summary[state][kind][str(shift)] = {
                    "definition": "class = nucleotide_index x modulo 3",
                    "mean_attention_mass_x_mod3": absolute.mean(axis=0).tolist(),
                    "dominant_x_mod3_counts": np.bincount(absolute.argmax(axis=1), minlength=3).tolist(),
                    "peak_x_mod3_counts": np.bincount(peaks, minlength=3).tolist(),
                    "peak_class_change_from_d0_counts": np.bincount(delta, minlength=3).tolist(),
                }

    out_data = {
        "status": "DERIVED_FROM_EXPLORATORY_V2",
        "source": args.input,
        "convention": "absolute period-3 class is exactly x%3: 0,1,2,0,1,2,...",
        "n": len(ids),
        "summary": summary,
    }
    out = REPO / args.out
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable analysis: {out}")
    out.write_text(json.dumps(out_data, indent=2) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert:p3_shift_diagnostic_v2_mod3",
        sys.argv,
        repo=REPO,
        status="EXPLORATORY_DERIVED",
        config={"source": args.input, "n": len(ids)},
        notes=[out_data["convention"], "No model forward pass; derived from recorded attention masses."])
    manifest.add_input(source)
    manifest.add_output(out)
    manifest.write(out.with_suffix(".manifest.json"))
    print(json.dumps(out_data, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
