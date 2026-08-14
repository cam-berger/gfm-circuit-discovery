# Figure 1 — L3H1 selectively carries partner-specific coupling

**A.** Corrupting the annotated same-intron acceptor strongly changes the masked-donor
margin in the intact model. The same-exon, unpaired, other-donor, acceptor–acceptor,
and matched-random controls are smaller. Ablating L3H1 selectively removes the
same-intron effect. Points are means and intervals are the recorded confidence
intervals.

**B.** L3H1 ranks first in the held-out exhaustive 96-head denoising scan. The panel
shows the eight highest-ranked heads and their recovered coupling.

**C.** L3H1 has opposite signed effects when the paired acceptor is intact versus
corrupted. The partner-consistency interaction is +1.110 [1.018, 1.205].

All axes report raw logit differences in logit units. The figure establishes
model-native partner-specific coupling and causal localization to L3H1; it does not
by itself identify an in-vivo biological function.

**Sources:** `figures/splicebert/data/pair_categories.json` and
`results/splicebert_coupling/confirmation.json`. Exact source hashes are recorded in
the adjacent provenance file.
