# Tier-2 verification report — F1–F4 integrity gates

> **Tier: T2 — provenance.** Author: Tier-2 formalization session (frame 0a2ba5cc), 2026-07-23.
> Controlling brief: [`TIER2_HANDOFF_2026-07-23.md`](TIER2_HANDOFF_2026-07-23.md) §5.G.
> Baseline commit (pre-work): `0bdaa1c`. Branch: `formalization-handoff-2026-07-23` (**not pushed**).

This report records the four adversarial verification gates run after the C1 reorg, the B2/B4
fidelity passes, the D1 figures, and the E1 tiering. It reads no scientific value it did not already
verify; it is an integrity audit only.

## Scope of Tier-2 changes verified

Versus baseline `0bdaa1c`, git records **229 renames (all R100 — pure, zero content change on move)**
and **93 modified files**, fully reconciled:

| Bucket | Count | What |
|---|---:|---|
| C1 renames | 229 | dev/superseded artifacts + `.manifest.json` siblings → `development/`/`superseded/` subdirs |
| E1 tier stamps | 70 | one `> **Tier: …**` blockquote each, inserted after the H1 (69 stamp-only + 1 also ref-rewritten) |
| C1 reference rewrites | 18 | script `--out`/`--scan`/input-path defaults + `docs/REVIEW_OPEN_QUESTIONS.md` (moved-path rewrite) |
| MANIFEST regen | 2 | `results/MANIFEST.{csv,json}` — path columns only |
| D1 figure regen | 4 | `fig_carrier_heatmap.{png,pdf}` (binary), 2 `*.provenance.json` (corrective path/commit) |

(70 + 18 overlap by 1 on `REVIEW_OPEN_QUESTIONS.md`; 69 + 18 + 2 + 4 = 93.)

Untracked new files (D1 deliverables, not yet committed): `figures/splicebert/fig_generator_graph.{gv,pdf,png}`
+ its provenance, and `docs/REFERENCE_MAP.json`. No `.claude/worktrees/` path is in the change set
(the parallel Tier-1 track's worktree was not touched).

---

## F1 — Scientific-record integrity diff — **PASS**

Assertion: no numeric value, claim text, evidence level, C/R/HCA-number, statistical result, or
verdict changed anywhere in the record.

- **Renames are pure.** All 229 are `R100` — the artifact bytes are unchanged on relocation; no result
  was split from its `.manifest.json` sibling (checked: 0 orphans, 0 cross-dir splits).
- **Record-class files (40 changed) are purely additive.** For every `CLAIMS.md`, `CORRECTIONS.md`,
  `DECISIONS_A0.md`, `RESULTS_VERIFICATION.md`, `MECHANISM.md`, `FINDINGS.md`, and `preregistrations/*.md`
  touched, the only change is the tier-stamp blockquote: `git diff --numstat` = **2 added / 0 deleted**
  per file. Zero non-blank existing lines removed; zero non-stamp additions.
- **Digit guard.** No line removed from any record file contained a decimal value, `C<n>`, `R<n>`,
  `HCA<n>`, or `z=<n>` token.
- **MANIFEST scientific columns unchanged.** Only the `artifact` (229) and `manifest` (123) path cells
  were remapped (352 rewrites, CSV == JSON); `status`/`evidence_level`/`polarity`/`claim_id` and the
  `row_count`/`status_counts`/`polarity_counts` summaries are identical to baseline (770 rows;
  527 canonical / 99 development / 132 superseded / 12 infrastructure).
- **Figure provenance diffs are corrective, not substantive.** The two `*.provenance.json` changes
  replace a stale `.claude/worktrees/repo-structure-review-938072/…` source path and an old
  `git_commit` with the canonical repo-root path and baseline commit; **every source `sha256` is
  unchanged**, so the figures were regenerated from bit-identical inputs.

## F2 — Link & path integrity — **PASS**

Checklist: `docs/REFERENCE_MAP.json` (229 moving artifacts, 299 in-prose references).

- **229/229** moved artifacts present at their new path; **229/229** old paths now absent.
- **26/26** rewritten navigational targets (script defaults) resolve on disk.
- **215/215** markdown relative links across live docs resolve — 0 broken.
- **0** lingering old-path references in 428 editable files scanned — the rewrite is complete, not
  best-effort.
- The **272 frozen-provenance references** (251 inside `.manifest.json` `command`/`config` arrays,
  15 result-JSON internal path fields, 6 read-only preregistration planned-output mentions) were left
  intact **by policy**: they are historical records of *what was run* (immutable under Global
  Constraints 2/3), not live navigational links. All 119 distinct old paths they mention correspond to
  genuinely-moved artifacts — expected-stale-by-design, no phantom targets.

## F3 — Citation integrity — **PASS (7/7)**

Each `docs/REFERENCES.md` anchor independently re-verified to exist and to support its exact attached
statement; none retracted.

| Citation | Identifier | Verified via | Verdict |
|---|---|---|---|
| Mueller et al., MIB | arXiv:2504.13151 (ICML 2025 / PMLR v267) | arXiv API (title, 23 authors, 2025-04-17) | PASS |
| Geiger et al., Causal Abstraction | arXiv:2301.04709 (JMLR 26(83) 2025) | arXiv API (title, 11 authors, 2023-01-11) | PASS |
| Chan et al., Causal Scrubbing | AI Alignment Forum 2022 | cited-as-such (not peer-reviewed; no numeric claim rides on it) | PASS |
| Zhang & Nanda | arXiv:2309.16042 (ICLR 2024) | arXiv API (title, authors, 2023-09-27) | PASS |
| Heimersheim & Nanda | arXiv:2404.15255 (2024) | arXiv API (title, authors, 2024-04-23) | PASS |
| Conmy et al., ACDC | arXiv:2304.14997 (NeurIPS 2023) | arXiv API (title, 5 authors, 2023-04-28) | PASS |
| Syed et al., EAP | arXiv:2310.10348 / DOI 10.18653/v1/2024.blackboxnlp-1.25 | arXiv API **and** CrossRef DOI (BlackboxNLP 2024, not retracted) | PASS |

Two documented caveats are honored in the prose: Chan et al. is marked not-peer-reviewed at every use,
and the "noising = necessity / denoising = sufficiency, non-symmetric" wording is **not** attached to
Heimersheim & Nanda (per the REFERENCES.md scope caveat). Three venue landing pages (JMLR v26,
PMLR v267, the Chan AF URL) are outside the network allowlist and could not be re-fetched here; each
was attested by the 2026-07-22 review and carries no numeric claim. **No unsupported citation; no
`REVIEW_OPEN_QUESTIONS.md` flag needed.**

## F4 — Test / CI integrity — **PASS**

- **Full suite green:** `uv run --extra splicebert --extra dev pytest -q` → **305 passed, 35 skipped,
  0 failed** (identical to the pre-work baseline).
- **`tests/test_manifest.py`:** 33 passed.
- **Manifest guard "test the test":** removing one canonical row from a copy of `MANIFEST.{csv,json}`
  made `scripts/verify_manifest.py` **fail** (exit 1: "769 rows / 770 tracked files"); restoring the
  row returned it to exit 0. The guard is protective, not vacuous.

---

## Deliverables produced this session

1. `docs/REFERENCE_MAP.json` — A4 path-reference index (229 artifacts, 299 references, C1 rewrite policy).
2. Reorganized `results/` tree — 229 files under `development/`/`superseded/`, references rewritten,
   MANIFEST regenerated, suite green (C1).
3. B2/B4 fidelity passes — audited; `METHODOLOGY.md`/`EVIDENCE_LADDER.md`/`BIOLOGICAL_BRIDGES.md`
   already conform to review §6/§7 (12/12 checklist items); additive-by-default rule left them unedited.
4. `figures/splicebert/fig_generator_graph.{gv,pdf,png}` (graphviz, data-derived, `.gv` reproduces
   bit-identically from `a10_generator_edges.json`); ladder-hero SVG and 2D carrier-heatmap confirmed
   against their recompute/memorial guards (D1).
5. All 70 repo-wide unstamped docs tier-stamped (E1); recursive scan returns zero.
6. This F1–F4 report.

## Owner-only items deliberately not done (per brief)

Branch push; `master`/other-worktree reconciliation; running G1/any experiment, inference, ACDC/EAP,
or GPU/MPS job. The G1 pilot artifacts under `.claude/worktrees/tier1-handoff-agent-brief-ff45d6/`
belong to the separate Tier-1 session and were not touched.
