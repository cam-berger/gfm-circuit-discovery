# The falsifiers

These scripts **killed the v1/v2 partner-selection task**. They are not utilities; they are the
evidence behind retractions **R2** and **R3** in
`studies/splicebert_partner_selection/CLAIMS.md`, and the reason the coupling study exists.

| script | what it proved |
|---|---|
| `oracle_audit.py` | What PMR does an **oracle site detector** get on the conflict set? Answer: **1.000.** The task never required partner information. → **R2** |
| `oracle_audit2.py` | Follow-up: a **geometry oracle reading zero nucleotides** also scores 1.000, because `dataset.py:135` pinned the acceptor to a constant position. → **R2** |
| `fair_pwm.py` | Is the registered PWM a strawman? A fair PWM scores **0.687** vs 0.392 chance — so "pick the partner" had collapsed into "pick the better donor". → **R3** |
| `../audit_site_detection_sufficiency.py` | Is the task solved by pure site detection + a length prior? |
| `../audit_partner_feasibility.py` | Are clean negatives even available in a real genome? |

They lived loose in a scratch directory, outside version control, while the retractions they
justify were being cited as settled. **A falsifier is provenance. Version it like one.**

They hardcode absolute paths to the old working copy and are kept for the record, not to be
re-run.
