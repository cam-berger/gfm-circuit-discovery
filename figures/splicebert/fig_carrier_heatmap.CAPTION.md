# fig_carrier_heatmap — the two-carrier surface, readable off the page

A 2D companion to the 3D render in `phase_graphs_3D/carrier-wave-interaction-paper.svg`. Same
function, same constants, no recomputation: `joint_weight` and its parameters are imported from
`phase_graphs_3D/generate_carrier_paper_figure.py`, so the two figures cannot drift apart. The 3D
view stays the better object for a talk; this one is the object a reader can put a number on.

**A.** The unnormalized weight `exp(C₃ + C₁₀)` over the two carrier coordinates, with contours. The
white/orange line is the physical trajectory `x = y` — the only line in this plane that a real
sequence position travels.

**B.** The same surface read along that trajectory and factored. Because `exp(a + b) = exp(a)·exp(b)`,
the period-3 factor (blue) and the ≈10.63-nt factor (orange) separate **exactly**; their product
(ink) is what the model sees. Reading B rather than the 3D surface is how one recovers the numbers:
the period-3 factor dominates the fast structure, the ≈10.63 factor slowly modulates its envelope.

**This is geometry, not a measurement.** The figure draws an analytic two-carrier model, not measured
attention. The period-3 period is 3 by construction; the long period is the **recorded** carrier
wavelength `10.62994` nt, and the generator refuses to draw if it disagrees with
`results/phase-lattice/fig_report_data.json → wavelengths.carrier.T`. The amplitudes
(`A_period-3 = 1.0`, `A_10.63 = 0.578`) are the ones set in the 3D generator and are **not** measured
quantities.

**No evidence level is claimed by this drawing.** The ≈10.63 carrier is established at Level A
(attention) with **OV/mediation and behaviour OPEN**; the period-3 carrier has causal support as a
distributed relative-positional routing coordinate. Neither status is changed, and the figure draws
no arrow from either carrier to splicing, chromatin, or translation. The nucleosomal signal is
≈10.298, **0.33 nt away from the model's 10.63 and excluded by a powered refutation** — this figure
is not about it.

**Sources:** `phase_graphs_3D/generate_carrier_paper_figure.py` (the surface definition) and
`results/phase-lattice/fig_report_data.json` (the recorded carrier wavelength), both hashed in
`fig_carrier_heatmap.provenance.json`. Generator: `make_fig_carrier_heatmap.py`.
