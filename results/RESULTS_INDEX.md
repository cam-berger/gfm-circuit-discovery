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
  `positive`, `null`, `bounded`, `refutation`, and `na` (non-evidence rows).
  `tests/test_manifest.py` pins the non-positive polarities; adding a claim whose result is a null or a
  bound without setting this field is what that guard exists to catch. For the current tallies run
  `uv run python scripts/verify_manifest.py`, which prints them — they are deliberately not restated
  here, because the counts written into this file had already drifted from the manifest once.
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

## `claim_id` values that are not claim IDs

Most rows carry a ledger claim ID (`C*`, `HCA*`, `GFS*`, `IC1`, `ED1`). Six values do not resolve to a
ledger row, because they name an **experiment arc** rather than a claim. They are listed here so a
reader of the manifest is not left chasing an identifier that does not exist. `tests/test_ledger_consistency.py`
only validates the `C*`/`HCA*`/`GFS*` forms, so these are not caught automatically.

(A seventh, `ED1`, *was* an unresolved identifier until 2026-07-31: four canonical rows carried it with
no defining row anywhere. It now has one — see the last section of the donor-circuit ledger.)

| `claim_id` | What it is | Where the claim actually lives |
|---|---|---|
| `A4` | The first address-geometry design, superseded before it produced a claim. | Redesigned as A11 → **C32**. |
| `A9` | Preflight for the payload-semantics arc. | The arc's surviving result is **C30**; its naming was withdrawn by **R22**. |
| `R22` | `a9c_sweep_seeded_draws.json` is evidence *for a retraction*, not for a claim. | `CORRECTIONS.md` R22. |
| `IC2` | Acceptor-competition feasibility and calibration for an arc retired before its confirmatory run. | No claim; the premise was refuted by Atlas HCA4/HCA10. |
| `P3-shift-diagnostic` | An exploratory period-3 shift diagnostic, outcome-aware; never promoted. | No claim. The carrier claims are C8, C34–C36. |
| `HCA7-HCA10` | A range label on one closure-audit artifact. | The individual rows HCA7…HCA10 exist in the Atlas ledger. |

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

> **Three Atlas claims are backed by artifacts that are not in this repository, and only one is
> retrievable.** HCA20, HCA21 and HCA22 have no `MANIFEST.csv` row because there is no file here to index.
> **HCA21** supplies working Drive URLs and a bundle SHA-256 in the Atlas `RESULTS.md`, so it is
> verifiable off-repo. **HCA20** names only a bare Drive folder and **HCA22** names a separate
> `gfm-visualizer` repository whose location appears nowhere here — neither is retrievable from this
> checkout. The promise that every claim resolves to a committed artifact holds for every other row in
> every ledger.

## Directories

| Directory | Index | Claim ledger |
|---|---|---|
| `phase-lattice/` | [`RESULTS_INDEX.md`](phase-lattice/RESULTS_INDEX.md) | `studies/splicebert_partner_selection/CLAIMS.md` |
| `splicebert_coupling/` | [`RESULTS_INDEX.md`](splicebert_coupling/RESULTS_INDEX.md) | `studies/splicebert_partner_selection/CLAIMS.md` |
| `splicebert_head_context_atlas/` | [`RESULTS_INDEX.md`](splicebert_head_context_atlas/RESULTS_INDEX.md) | `studies/splicebert_head_context_atlas/CLAIMS.md` |
| `origin-siblings/` | [`RESULTS_INDEX.md`](origin-siblings/RESULTS_INDEX.md) | `studies/splicebert_partner_selection/CLAIMS.md` (C39) |
| `origin-crossmodel/` | [`README.md`](origin-crossmodel/README.md) | `studies/splicebert_partner_selection/CLAIMS.md` (C40) |
| `splicebert_gtex_finetune_shift/` | [`RESULTS_INDEX.md`](splicebert_gtex_finetune_shift/RESULTS_INDEX.md) | `studies/splicebert_gtex_finetune_shift/CLAIMS.md` (GFS1–GFS34) |
| `splicebert_acceptor_coupling/` | no per-directory index; the five artifacts and their manifests are the `MANIFEST.csv` rows | `studies/splicebert_acceptor_coupling/CLAIMS.md` (AC1–AC4) |
| `splicebert_exon_edge_compatibility/` | [archive record](../studies/splicebert_exon_edge_compatibility/README.md) | `studies/splicebert_partner_selection/CLAIMS.md` (C43, C44) |

The `dnabert2_reconstruction_atlas/` and `splicebert_visualizer/` trees were removed from the working
tree in the 2026-07-31 cleanup along with their studies; both remain in Git history. Neither controlled
a claim in a surviving ledger.

`splicebert_exon_edge_compatibility/` is different: it is a **consolidation remnant, not a full tree**.
Twelve artifacts (4.0 MB) were carried on 2026-08-04 for C43 and C44; the other ~99 MB of that study's
results — including the 48 MB and 37 MB crossed cohorts — stayed on branch `codex/intron-exon-phase` at
commit `fd26cc1b`, together with every runner that produced them. Consequently these twelve rows carry a
**blank `script` column** by design: each sibling `*.manifest.json` pins its archived runner by path and
sha256, so provenance is intact and the code is recoverable, but it is not in this tree. Both audits were
already un-re-runnable on the source branch, whose own `.gitignore` excluded the raw rows they recompute
from; the controlling numbers are re-derivable from per-block endpoint tables inside the artifacts. See
[`CORRECTIONS.md` R45](../studies/splicebert_partner_selection/CORRECTIONS.md).

**Layout convention.** Most trees keep their **canonical artifacts at the directory root**, with
`development/` and `superseded/` subtrees beside them — the `canonical/`-subdirectory half of the 2026-07-22
reorg was deliberately not applied, because frozen preregistrations and read-only ledgers name canonical
artifacts by their current root path (see [`docs/REVIEW_OPEN_QUESTIONS.md`](../docs/REVIEW_OPEN_QUESTIONS.md) §C1).
The two newest origin trees (`origin-siblings/`, `origin-crossmodel/`) are the exception: they additionally
use a `canonical/` subdirectory. `results/MANIFEST.csv` classifies every artifact regardless of layout.

`p3_carrier_locus.json` and `p3_av_decomposition.json` sit at the root of this tree and carry C34 and
C35 respectively; both paths are quoted verbatim in `CLAIMS.md`, `CORRECTIONS.md` and `REPRODUCE.md`.

The repository-level reproduction map is [`REPRODUCE.md`](../REPRODUCE.md).
