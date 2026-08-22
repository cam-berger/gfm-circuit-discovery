# SpliceBERT figures — current record

Start with [`EVIDENCE.md`](EVIDENCE.md), which keeps position-table, QK, attention,
OV/mediation, and behavioral evidence separate.

## Publication set

The first three publication figures read one generated data package,
`results/phase-lattice/fig_report_data.json`. That package is built by
`scripts/splicebert/make_fig_report_data.py` and checks recomputed quantities against the
recorded runs before it writes.

| Figure | Script | Main point |
|---|---|---|
| `fig1_summary` | `make_fig1_summary.py` | Two positional carriers: one with a causal role, one with an unknown role |
| `fig2_oscillators` | `make_fig2_oscillators.py` | Carrier geometry, spectrum, orthogonality, and sub-bin wavelength |
| `fig3_clock_circuit` | `make_fig3_clock_circuit.py` | Period-3 deletion, dose response, and mediation |
| `fig5_p3_phase_surgery_confirmation` | `make_fig5_p3_phase_surgery_confirmation.py` | Clean phase-aligned P3-use confirmation, learned-direction controls, and inconclusive coding boundary |

Each figure has a caption beside it. The narrative and limitations belong in the caption rather
than being compressed into the artwork.

## Supporting figures

These are additions to the set above; they change nothing in it. Each reads a manifest-stamped
artifact (or, for the methods figure, the two method documents) and writes a provenance sidecar.

| Figure | Script | Main point |
|---|---|---|
| `fig_evidence_ladder` | `make_fig_evidence_ladder.py` | The methods figure: seven-stage workflow, evidence ladder, tool-to-question map, rendered from `docs/METHODOLOGY.md` and `docs/EVIDENCE_LADDER.md` |
| `fig_generator_graph` | `make_fig_generator_graph.py` | The A10/C31 payload generator as a directed graph, laid out by Graphviz from `a10_generator_edges.json` |
| `fig_carrier_heatmap` | `make_fig_carrier_heatmap.py` | 2D companion to the 3D carrier surface: same function, same constants, quantitatively readable |

### Donor-circuit figures generated in this folder

`make_fig1_coupling.py`, `make_fig2_mechanism.py` and `make_fig3_circuit.py` read
`results/splicebert_coupling/*.json` and write `fig1_coupling`, `fig2_mechanism` and `fig3_circuit`
as PDF+PNG with provenance. They compute nothing; every number is in an artifact. The canonical
phase-lattice set listed above is unaffected by them.

The older unprovenanced `paper/fig2_circuit_schematic` and its hand-packaged `viz/`
inputs were removed in the 2026-08-22 evidence cleanup. The current
`fig_full_circuit` and `workshop_edits/fig_full_circuit_edit` are generated directly
from recorded result artifacts. The former hand-authored `fig_full_circuit.gv` was
also removed; it was not the source of the current rendered figure.

## Style

`figstyle.py` is the one style module: fonts, the validated palette as named tokens, DPI, and the
helper that writes PDF **and** PNG beside a provenance sidecar. Every `make_fig*.py` in this study
imports it, including the ones under `scripts/splicebert/`.

The palette was **computed, not chosen** — every pair of colours that appears together in a panel
was checked for separation under simulated deuteranopia, protanopia and tritanopia. `figstyle.py`
therefore *re-exports* that one definition from `scripts/splicebert/figstyle.py` rather than copying
the values into a second file. The colour thread it enforces: **period-3 = blue, ≈10.63 = orange**,
donor–acceptor coupling = purple, nucleosomal ≈10.298 = red, deleted/failed = slate. Methods figures
use none of those hues, so the thread stays reserved for results.

Three matplotlib blues are still in use across the set (the validated palette, the Okabe-Ito tokens
in `src/gfmcd/figures/style.py`, and the confirmation figure's own five colours). All three are now
named in one place; unifying them would restyle published panels, so it is left to the owner and
recorded in `docs/REVIEW_OPEN_QUESTIONS.md`.

## External dependency: Graphviz

`make_fig_generator_graph.py` writes `fig_generator_graph.gv` with no dependencies — the DOT source
is the data-derived, reproducible artifact. Rendering it to PDF/PNG needs Graphviz's `dot` on
`PATH`:

```
brew install graphviz        # macOS
apt-get install graphviz     # Debian/Ubuntu
```

**`dot` is not currently installed in this environment**, so only the `.gv` source and its
provenance sidecar are committed. The script reports this and stops; it does not fall back to a
hand-placed drawing.

## Three live claims

1. **SpliceBERT contains a donor–acceptor information-routing circuit.**
2. **A learned period-3 positional carrier acts as a distributed relative-positional routing coordinate
   supporting that circuit.** C34/C35 refine the earlier carrier figures; its present-day use is supported,
   while C39/C40 associate its comparative origin with multi-species pretraining. Coding specificity and
   training dynamics remain open.
3. **A separate ≈10.63 positional carrier forms a genuine attention circuit. Its donor-side arms were
   measured and returned boundaries, not positives; its general downstream use and biological meaning
   remain unresolved; C39/C40 provide comparative training-lineage association, not causal origin.**

The period-3 and ≈10.63 carrier planes are orthogonal. The period-3 carrier has a demonstrated causal
role in coupling. The ≈10.63 carrier reaches attention and goes no further on the evidence: its donor-side
OV and behavioural arms returned a sub-SESOI non-detection and a small suppressive bound (C37→C12, R31), so
the figures deliberately draw no arrow from it to splicing, chromatin, translation, or another behavior.

## Retired language

- Use **≈10.63 carrier**, not “10.67 phase.” A carrier plane contains every phase; a head selects a
  direction through QK.
- “10.67” was the label of a discrete Fourier bin, not a wavelength measurement.
- “10.58 ± 0.35” is not a measured constant and is not the carrier's explanation.
- The nucleosomal signal is ≈10.298, not ≈10.63.
- The proposed human-only ≈10.2 chromatin circuit failed at post-softmax attention.

Superseded phase figures and the stale `fig4_carrier_circuit` were removed from the active tree. Git
history preserves their evolution. The figures in the publication table above are the complete
canonical phase-lattice set.

## Statistical conventions

- Exact add-one permutation p-values; with 400 permutations the floor is 1/401 ≈ 0.0025.
- BH-FDR is applied across the full scanned head family.
- Mediation nulls are matched to intervention size.
- Preregistered and exploratory results are labeled separately in the result record.

## Detailed provenance

- `../../MECHANISM.md` — the one-file mechanism account (replaces the former `studies/splicebert/` overview)
- `studies/splicebert_partner_selection/CLAIMS.md` — claims, corrections, and retractions
- `studies/splicebert_partner_selection/CORRECTIONS.md` — full correction and supersession history
- `../../MECHANISM.md` — current mechanism including post-figure C34/C35 refinements
- `results/splicebert_coupling/` — donor–acceptor circuit results
- `results/phase-lattice/RESULTS_INDEX.md` — positional carrier results
- [`HAND_ASSETS.md`](HAND_ASSETS.md) — **T3 retirement record.** Hand-authored artwork removed from the
  active tree, and why the donor–acceptor binding illustration must not be recovered from Git history and
  cited: the content-addressed read it depicts is retracted as R15.
