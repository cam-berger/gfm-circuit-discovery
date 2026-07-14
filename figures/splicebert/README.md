# SpliceBERT figures — current record

Start with [`EVIDENCE.md`](EVIDENCE.md), which keeps position-table, QK, attention,
OV/mediation, and behavioral evidence separate.

## Publication set

The four publication figures read one generated data package,
`results/phase-lattice/fig_report_data.json`. That package is built by
`scripts/splicebert/make_fig_report_data.py` and checks recomputed quantities against the
recorded runs before it writes.

| Figure | Script | Main point |
|---|---|---|
| `fig1_summary` | `make_fig1_summary.py` | Two positional carriers: one with a causal role, one with an unknown role |
| `fig2_oscillators` | `make_fig2_oscillators.py` | Carrier geometry, spectrum, orthogonality, and sub-bin wavelength |
| `fig3_clock_circuit` | `make_fig3_clock_circuit.py` | Period-3 deletion, dose response, and mediation |
| `fig4_carrier_circuit` | `make_fig4_carrier_circuit.py` | The ≈10.63 post-softmax attention circuit and matched human-only null |

Each figure has a caption beside it. The narrative and limitations belong in the caption rather
than being compressed into the artwork.

## Three live claims

1. **SpliceBERT contains a donor–acceptor information-routing circuit.**
2. **A learned period-3 positional clock helps that circuit address coding sequence in codon
   phase.** These are two parts of one mechanism.
3. **A separate ≈10.63 positional carrier forms a genuine attention circuit, but its downstream
   use, origin, and biological meaning remain unknown.**

The period-3 and ≈10.63 carrier planes are orthogonal. The period-3 clock has a demonstrated causal
role in coupling. The ≈10.63 carrier stops at attention; the figures deliberately draw no arrow
from it to splicing, chromatin, translation, or another behavior.

## Retired language

- Use **≈10.63 carrier**, not “10.67 phase.” A carrier plane contains every phase; a head selects a
  direction through QK.
- “10.67” was the label of a discrete Fourier bin, not a wavelength measurement.
- “10.58 ± 0.35” is not a measured constant and is not the carrier's explanation.
- The nucleosomal signal is ≈10.298, not ≈10.63.
- The proposed human-only ≈10.2 chromatin circuit failed at post-softmax attention.

Superseded and retracted hand assets and earlier publication drafts are retained in Git history,
not beside the figures of record.

## Statistical conventions

- Exact add-one permutation p-values; with 400 permutations the floor is 1/401 ≈ 0.0025.
- BH-FDR is applied across the full scanned head family.
- Mediation nulls are matched to intervention size.
- Preregistered and exploratory results are labeled separately in the result record.

## Detailed provenance

- `studies/splicebert/README.md` — concise worked example
- `studies/splicebert_partner_selection/CLAIMS.md` — claims, corrections, and retractions
- `results/splicebert_coupling/` — donor–acceptor circuit results
- `results/phase-lattice/RESULTS_INDEX.md` — positional clock and carrier results
