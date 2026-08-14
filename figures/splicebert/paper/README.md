# Paper figures — donor–acceptor circuit (NeurIPS-grade)

> Figure provenance for the sequence→head→output circuit figures, 2026-07-24.
> Every value is licensed from the frozen record; nothing here is a new claim.

One publication figure (vector PDF + 300-dpi PNG), driven by
[`../viz/circuit_data.json`](../viz/circuit_data.json) — values from `CLAIMS.md` and frozen artifacts.

## Figures

- **`fig2_circuit_schematic.*`** — the standalone front-to-back circuit schematic, matplotlib-native vector.

## Removed on 2026-07-31 — not regenerable

`fig1_hero.*` and `fig3_real_attention.*` were removed, along with the `make_paper_figures.py` stub that
claimed to build them. Both depended on `../viz/live_attention.json` — a single live SpliceBERT.1024nt
forward pass used as a visualization input — and **that file was never committed**. Neither figure could be
rebuilt from the repository, neither carried a `.provenance.json`, and neither had a `MANIFEST` row, so both
violated rule 1 of [`../README.md`](../README.md). The stub generator contained no executable statements.

If a hero figure is wanted for submission it must be built from real code against committed artifacts, with a
provenance sidecar, exactly like `fig1_coupling` and `fig5_p3_phase_surgery_confirmation`. The data-derived
[`../fig_full_circuit.png`](../fig_full_circuit.png) already covers the front-to-back schematic and does have
declared provenance.

## Values shown (all licensed)

- Payload heads (C31, held-out N): L0H7 0.348, L0H12 0.160, L0H10 0.116 (dominant); L0H8/H13/H11 marginal.
- MLP chain (C31): MLP0 0.354, MLP1 0.434, MLP2 0.880. Serial; N are conditional, non-additive.
- L3H1 restore recovers 0.83 of coupling (C17); U_info rank-8 recovers ≈0.92 vs random ≤0.05 (C16).
- Readout cascade cumulative (readout_path.json): 0.023 → 0.181 → 0.258 → 0.445 → 0.506 → 0.818.
- Behavior: paired-acceptor corruption +0.872 [0.771, 0.969] (C1).

## What the figures must not assert

The N values are conditional causal necessities on held-out splits: they do **not** sum and are not additive
fractions; the chain is strictly serial. This is a **structured top-ranked core** inside an irreducibly
distributed computation, **not** a minimal or complete circuit. Attention magnitude is observational; head
necessity is established by intervention (panels c/d), which is why the brightest non-circuit head (L5H2) is
not part of the circuit.
