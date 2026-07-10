# Studies

Each study is one worked instance of the five-pillar protocol on a single target computation.
`docs/RESULTS.md` stays the global source of truth; each study's `RESULTS.md` rolls up into it.

| study | target computation | status | dir |
|---|---|---|---|
| reading_frame | anchor-relative modulo-3 frame (counter-vs-lookup open) | active | `docs/studies/reading_frame/` |
| repeat_copy | prefix-keyed copy / induction (weight-1 control pending) | active | `docs/studies/repeat_copy/` |
| splicebert_partner_choice | partner-aware splice-site relation in pretrained SpliceBERT | proposed | `docs/studies/splicebert_partner_choice/` |

## Old → new organization

Legacy per-track docs are **not** moved yet (to avoid breaking links/notebooks); this index maps them.
`git mv` into the per-study dirs happens only after every reference is updated (TO_FIX.md §17).

- reading frame → narrative in `docs/RESULTS.md`, `docs/REC.md`, `docs/PLAN_reading_frame.md`,
  `docs/RELAY_TRACE.md`; new: `docs/studies/reading_frame/COUNTER_VS_LOOKUP.md`.
- repeat-copy → `docs/PLAN_repeat_copy.md`, archived findings at
  `.archive/branches-codex-repeat-induction/codex-repeat-induction/`.

## Every study must specify (TO_FIX.md §17)

1. target computation · 2. model behavior demonstrating competence · 3. registered shortcut class ·
4. shortcut ceilings · 5. cue-conflict generator · 6. scalar causal metric · 7. intervention targets ·
8. necessity test · 9. sufficiency/mediation test · 10. replication plan ·
11. external biological validation · 12. explicit failure interpretations.

Start a new study by copying `docs/studies/TEMPLATE/`.
