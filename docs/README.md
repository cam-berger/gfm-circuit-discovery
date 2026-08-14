# Method documentation

These documents define the research program independently of any one model.

- [METHODOLOGY.md](METHODOLOGY.md) — the end-to-end discovery workflow and tool families.
- [EVIDENCE_LADDER.md](EVIDENCE_LADDER.md) — what each experiment licenses, and where claims must stop.
- [NOMENCLATURE.md](NOMENCLATURE.md) — the controlled vocabulary. Read before writing any prose that
  names a carrier wavelength or the acceptor payload; it lists the retired terms that must not reappear.
- [PREREG_CONFORMANCE.md](PREREG_CONFORMANCE.md) — where executed code departed from its frozen protocol.
  Read before treating any preregistered gate as having been enforced.
- [BIOLOGICAL_BRIDGES.md](BIOLOGICAL_BRIDGES.md) — what would be needed to connect a model mechanism to biology.
- [TRANSFER_GUIDE.md](TRANSFER_GUIDE.md) — how to port the approach to another GFM.
- [ROADMAP.md](ROADMAP.md) — open problems, and what is deferred or stopped.
- [REFERENCES.md](REFERENCES.md) — external literature, with verification status.
- [REPO_SCRUB_PROCEDURE.md](REPO_SCRUB_PROCEDURE.md) — how to audit this repository without damaging the record.
- [THIRD_PARTY_ASSETS.md](THIRD_PARTY_ASSETS.md) — external models, genomes, and annotation, with licences.
- [../MECHANISM.md](../MECHANISM.md) — one-file current SpliceBERT mechanism, scope, and open bridges.
- [../REPRODUCE.md](../REPRODUCE.md) — canonical environment and claim-to-artifact reproduction map.

Three point-in-time records are also kept here and are **not** current guidance:
[REVIEW_2026-07-22.md](REVIEW_2026-07-22.md) (pinned by `scripts/verify_manifest.py`, which cites its §4
as the manifest specification, and by `results/MANIFEST.json`),
[REVIEW_2026-08-11_DNABERT_PR40.md](REVIEW_2026-08-11_DNABERT_PR40.md) (the integration review that gave
the DNABERT-2 promoter arc its `DBP*` and `DBPR*` ledgers — read the ledgers, not this, for what is
claimed), and [REVIEW_OPEN_QUESTIONS.md](REVIEW_OPEN_QUESTIONS.md) (pinned by the CI workflow and
`tests/test_manifest.py`) — the latter still lists **live owner decisions**, so it is stale as narrative
but open as a to-do list. One advisory sits alongside them:
[MECH_INTERP_GAP_ANALYSIS.md](MECH_INTERP_GAP_ANALYSIS.md), a method-coverage survey ported from the
`jeyashree-ajay` branch and recorded as a correction in `CORRECTIONS.md`; re-read `CLAIMS.md` and
`ROADMAP.md` before acting on anything in it.

The two 2026-07-23 B2/B3 disposition notes this section used to name **no longer exist in the tree**; their
dispositions are carried by C30 in `studies/splicebert_partner_selection/CLAIMS.md` and by the B2/B3
red-team syntheses under `studies/splicebert_partner_selection/redteam/`.

The detailed SpliceBERT record lives under
[`studies/splicebert_partner_selection/`](../studies/splicebert_partner_selection/README.md); generated
evidence summaries live under [`figures/splicebert/`](../figures/splicebert/README.md).

Protocol-vs-code conformance for every experiment family is recorded in
[`PREREG_CONFORMANCE.md`](PREREG_CONFORMANCE.md) — 188 deviations across 25 families, only one of which
(R28) had been logged before the audit.
