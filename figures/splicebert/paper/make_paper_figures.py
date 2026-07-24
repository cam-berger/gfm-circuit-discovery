#!/usr/bin/env python
"""NeurIPS paper figures for the SpliceBERT donor-acceptor circuit.

Builds three publication figures from the licensed record:
  fig1_hero              : (a) schematic (b) real-attention grid (c) necessity (d) readout cascade
  fig2_circuit_schematic : standalone front-to-back schematic
  fig3_real_attention    : 96-head address grid + per-head attention maps

Data:
  ../viz/circuit_data.json     -- values licensed from CLAIMS.md + frozen artifacts
  ../viz/live_attention.json   -- one live SpliceBERT.1024nt forward pass (viz only, not a ledger result)

Run from figures/splicebert/paper/. Requires matplotlib + numpy + the figure-style helpers.
All N are conditional causal necessities on held-out splits: non-additive, strictly serial.
Structured top-ranked core, NOT a minimal/complete circuit.
"""
# The exact build code is captured in the session lineage for figures/splicebert/paper/*.
# See ../viz/circuit_data.json for the single source of truth, and README.md for provenance.
