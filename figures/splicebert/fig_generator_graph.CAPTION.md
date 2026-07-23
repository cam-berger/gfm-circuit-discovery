# fig_generator_graph — the A10 payload generator, laid out by Graphviz from the edge artifact

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

The directed graph of the acceptor-local payload generator (A10 / **C31**, Directed-circuit level,
held-out): six load-bearing layer-0 attention heads feed **MLP0**, which **mediates** their
contribution, into the strictly serial backbone **MLP0 → MLP1 → MLP2**.

**Not hand-placed.** Nodes, edges, weights and every printed number are read from
`results/splicebert_coupling/a10_generator_edges.json`; Graphviz does the layout. The head set is
taken from the artifact's own `load_bearing_H` flags and the script refuses to draw if that set
disagrees with the recorded `L0_star_heldout`, if the artifact is not marked confirmatory, if its
preflight did not pass, or if the serial chain is not confirmed.

**Values.** `N` is a conditional causal necessity on the held-out split with a gene-bootstrap 95% CI:
L0H7 `+0.348 [0.300,0.399]` (dominant; replicates its discovery value `+0.374`), L0H12
`+0.160 [0.129,0.192]`, L0H10 `+0.116 [0.090,0.142]`, with L0H8, L0H11 and L0H13 marginal but with
CI-low above zero. MLP0 `+0.354 [0.292,0.417]`, MLP1 `+0.434 [0.365,0.507]`, MLP2
`+0.880 [0.809,0.953]`. Mediation: layer-0 heads alone recover `0.136`; adding MLP0 takes it to
`0.356`. Serial-chain evidence: each MLP is necessary on held-out and none is sufficient alone
(MLP0 only `0.015`, MLP1 only `0.010`, MLP0+MLP1 `0.005`). n = 220 pairs across 196 genes; generator
ceiling `0.282`.

**The dashed grey link between L0H12 and L0H10** is the recorded interaction `−0.126` on the
**discovery** split: the two heads are partially redundant and substitutable. It is drawn as an
undirected annotation, not as a causal edge.

**What this graph must not be read as.** The `N` values are **conditional causal effects. They do not
sum and they are not additive fractions of the effect.** The two dashed grey boxes at the ends are
scope context, not measured nodes. The artifact measures the **value path with the address held
clean**, so the graph shows how the acceptor-local payload is *built*, not how it is addressed or
delivered, and it says nothing about the payload's semantic identity — that remains **unresolved**
(C30 as bounded by **R22**; the "splice-strength" naming is withdrawn). The on-path memorial for this
experiment is `value_linearity = 2.86e-6`; the `empty→corrupt` / `full→clean` covers are
**R21-vestigial** (embed-clamp-dominated) and are not evidence of node-cover completeness.

**Dependency.** Rendering PDF/PNG requires Graphviz's `dot` on `PATH`
(`brew install graphviz` / `apt-get install graphviz`). Without it the generator writes only
`fig_generator_graph.gv`, which is the complete data-derived artifact; it does not substitute a
hand-drawn imitation.

**Sources:** `results/splicebert_coupling/a10_generator_edges.json` (hashed in
`fig_generator_graph.provenance.json`). Generator: `make_fig_generator_graph.py`.
