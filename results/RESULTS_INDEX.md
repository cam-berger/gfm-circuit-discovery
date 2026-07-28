# Results tree — index

> **Tier: T1 — canonical.** Current, authoritative. Read this.

`MANIFEST.csv` and `MANIFEST.json` are the **machine-readable entry point** to this tree. They carry
one row per tracked file under `results/`, with the column set fixed by
[`docs/REVIEW_2026-07-22.md`](../docs/REVIEW_2026-07-22.md) §4 (fix F2):

```
claim_id,status,evidence_level,polarity,artifact,script,manifest,figure
```

- `claim_id` — the claim(s) the artifact supports, as recorded in the owning study's `CLAIMS.md`.
- `status` — organizational label, see below.
- `evidence_level` — the ladder rung the artifact speaks to (`Representation`, `QK`, `Attention`,
  `OV/mediation`, `Behaviour`, `Biology`, `Directed circuit`); blank where the file is not itself
  evidence.
- `polarity` — the *direction* of the evidence, so a null can never be silently re-read as a positive:
  `positive` (504), `null` (19), `bounded` (56), `refutation` (4), `na` (292, non-evidence rows).
  `tests/test_manifest.py` pins the non-positive polarities; adding a claim whose result is a null or a
  bound without setting this field is what that guard exists to catch.
- `artifact` — the file this row describes, repository-relative.
- `script` — the runner that produced it.
- `manifest` — its sibling `*.manifest.json` provenance record, where one exists.
- `figure` — the published figure(s) that consume it; `;`-separated when there is more than one.

`MANIFEST.json` additionally carries the Phase-A `safe_to_move` flag and a one-line `rationale` per
row. Those two fields are working metadata for the results reorganization, not part of the eight
columns above (the seven specified by review §4 plus the later `polarity` addition), and the CSV omits
them.

## What `status` means

| status | meaning |
|---|---|
| `canonical` | cited by a `CLAIMS.md` ledger, the figure pipeline, or a memorial test |
| `development` | smoke, preflight, calibration, or development-split output, kept for provenance |
| `superseded` | replaced by a later version of the same artifact, retained rather than deleted |
| `infrastructure` | index, directory-level provenance, or export file; not evidence for a claim |

`status` is an **organizational** label and carries no evidence weight. The evidence a result
licenses is given by `evidence_level` and, authoritatively, by the claim ledgers — never by this
index. Evidence never promotes itself between rungs.

## Integrity check

`scripts/verify_manifest.py` verifies that every tracked file under `results/` has exactly one
manifest row, that every `artifact` path exists, that every non-empty `script` / `manifest` /
`figure` path exists, that `status` is one of the four legal values, and that the CSV and JSON
serializations agree. `tests/test_manifest.py` runs it under pytest.

```bash
uv run python scripts/verify_manifest.py
```

Whenever a file under `results/` is added, removed, or moved, the manifest must be updated in the
same change; the verifier fails otherwise.

## Directories

| Directory | Index | Claim ledger |
|---|---|---|
| `phase-lattice/` | [`RESULTS_INDEX.md`](phase-lattice/RESULTS_INDEX.md) | `studies/splicebert_partner_selection/CLAIMS.md` |
| `splicebert_coupling/` | [`RESULTS_INDEX.md`](splicebert_coupling/RESULTS_INDEX.md) | `studies/splicebert_partner_selection/CLAIMS.md` |
| `splicebert_head_context_atlas/` | [`RESULTS_INDEX.md`](splicebert_head_context_atlas/RESULTS_INDEX.md) | `studies/splicebert_head_context_atlas/CLAIMS.md` |
| `splicebert_visualizer/` | [`RESULTS_INDEX.md`](splicebert_visualizer/RESULTS_INDEX.md) | none — export bundle, not a claim artifact |
| `origin-siblings/` | [`RESULTS_INDEX.md`](origin-siblings/RESULTS_INDEX.md) | `studies/splicebert_partner_selection/CLAIMS.md` (C39) |
| `origin-crossmodel/` | [`README.md`](origin-crossmodel/README.md) | `studies/splicebert_partner_selection/CLAIMS.md` (C40) |
| `dnabert2_reconstruction_atlas/` | [`RESULTS.md`](../studies/dnabert2_reconstruction_atlas/RESULTS.md) | [`CLAIMS.md`](../studies/dnabert2_reconstruction_atlas/CLAIMS.md) (D1–D5, IF1–IF24) |

**Layout convention.** Most trees keep their **canonical artifacts at the directory root**, with
`development/` and `superseded/` subtrees beside them — the `canonical/`-subdirectory half of the 2026-07-22
reorg was deliberately not applied, because frozen preregistrations and read-only ledgers name canonical
artifacts by their current root path (see [`docs/REVIEW_OPEN_QUESTIONS.md`](../docs/REVIEW_OPEN_QUESTIONS.md) §C1).
The two newest origin trees (`origin-siblings/`, `origin-crossmodel/`) are the exception: they additionally
use a `canonical/` subdirectory. `results/MANIFEST.csv` classifies every artifact regardless of layout.

`p3_carrier_locus.json` and `p3_av_decomposition.json` sit at the root of this tree and carry C34 and
C35 respectively; both paths are quoted verbatim in `CLAIMS.md`, `CORRECTIONS.md` and `REPRODUCE.md`.

The repository-level reproduction map is [`REPRODUCE.md`](../REPRODUCE.md).
