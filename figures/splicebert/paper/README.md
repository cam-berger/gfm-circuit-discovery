# Paper figures — donor–acceptor circuit (NeurIPS-grade)

> **Tier: T2 — provenance.** Figure provenance for the sequence→head→output circuit figures, 2026-07-24.
> Every value is licensed from the frozen record; nothing here is a new claim.

Three publication figures (vector PDF + 300-dpi PNG), all driven by
[`../viz/circuit_data.json`](../viz/circuit_data.json) (values from `CLAIMS.md` + frozen artifacts) and, for the
real-attention panels, a live SpliceBERT.1024nt forward pass (`../viz/live_attention.json`).

## Figures

- **`fig1_hero.*`** — the multi-panel Figure 1. (a) front-to-back circuit schematic; (b) real attention from a
  live forward pass, all 6×16 heads scored by donor→acceptor read, circuit heads ringed; (c) payload-branch
  conditional necessity (C31) with 95% CIs; (d) the readout cascade (distributed chain: 0.023 → 0.818 as blocks
  are released).
- **`fig2_circuit_schematic.*`** — the standalone front-to-back schematic (Figure 2 / hero), matplotlib-native
  vector.
- **`fig3_real_attention.*`** — the real-attention evidence plate: 96-head address grid + per-head attention
  maps (L3H1, L0H7, L2H15) from the live forward pass.

## Provenance of the live attention

`../viz/live_attention.json` is one CPU forward pass of **SpliceBERT.1024nt** (6 layers × 16 heads, eager
attention) on a canonical short-intron donor–acceptor example (intron 106 nt, within the licensed 60–150 nt
regime). It is a **visualization input only** — not a claim-ledger result and not registered in `MANIFEST`.

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
