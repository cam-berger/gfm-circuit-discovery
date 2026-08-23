# Figure 1 — Partner-specific coupling and load-bearing attention heads

**A.** Corrupting the annotated same-intron acceptor strongly changes the masked-donor
margin in the intact model. Same-exon, unpaired, other-donor, acceptor–acceptor, and
matched-random controls are smaller. Ablating L3H1 selectively reduces the same-intron
effect. Points are means and intervals are the recorded confidence intervals.

**B.** The corrected exhaustive validation scan (160 examples, 146 genes) ranks L3H1
and L2H15 first by partner-specific loss under mean ablation. The scan nominates
candidates; sealed necessity is tested separately.

**C.** On the sealed conditional mechanism cohort (150 examples), ablating L3H1 and
L2H15 together removes 81.6% of coupling while retaining 99.2% of the non-site MLM
health measure. Error bars are 95% confidence intervals for raw coupling removed.

All axes report raw pre-softmax logit differences. The figure establishes model-native
partner-specific coupling and component necessity; it does not identify an in-vivo
biological function or a complete circuit.

**Sources:** `figures/splicebert/data/pair_categories.json`, the corrected
`results/splicebert_coupling/development/ablation_scan.json`, and
`results/splicebert_coupling/sealed_test.json`. Exact hashes are recorded in the
adjacent provenance file.
