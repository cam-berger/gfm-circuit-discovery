# QUESTION — splicebert_partner_choice

**Target computation (1).** Partner-aware splice-site choice: given a splice site, which *partner* site
pairs with it across an intron/exon — a relation between two positions, not a property of either.

**Competence behavior (2).** Under matched local motif strength, does SpliceBERT prefer the true
partner over decoys whose local cues are matched? (zero-shot nucleotide dependency; frozen-backbone
readout; fine-tuned — the three levels in `INTERVENTIONS.md`).

**Why it may be shortcut-proof.** Isolated splice-site motif detection is a local PWM scan (Tier 0 —
a pooled shortcut suffices). *Exon definition* (summing donor+acceptor strengths across the exon) is
the relational residual: composition can be held fixed while the pairing changes. The registered
shortcut class and its ceiling are in `BASELINES.md`; only a positive gap justifies a trace.

**Status: PROPOSED.** Requires the `models` extra (SpliceBERT weights), a real GENCODE dataset, and
owner sign-off before any run (TO_FIX.md §18–26). Firewalls: SpliceBERT is pretrained on the whole
human genome (no holdout; mechanistic claims only) and sense-strand only (no cross-strand claims).
