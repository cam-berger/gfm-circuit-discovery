# CUE_CONFLICTS — <study name>

**Cue-conflict generator (5).** Construct sequence pairs with matched local/pooled cues but a
conflicting relation, so in-distribution attribution cannot pass a pooled statistic off as a circuit.
Freeze these before the full evaluation.

Each conflict returns: clean sequence · corrupt/conflict sequence · intended relational target ·
shortcut target · edit locations · matching diagnostics · score mask.

List every conflict and the mechanism it separates (e.g. true-vs-decoy, partner-swap, one-sided
weakening, spacing perturbation, matched-irrelevant-edit control).
