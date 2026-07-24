# ULTRAREVIEW — gfm-circuit-discovery Publication-Readiness Report

> **Tier: T2 — provenance.** Dated review/handoff record; not current canonical guidance.

> **Execution note (2026-07-24).** This is the review of record from a 38-agent multi-agent ultrareview
> (5 ground-truth sweeps → 14 region/formalization reviewers → 18-finding adversarial verification →
> synthesis). It was produced read-only. The **Tier-A** mechanical items were then applied on branch
> `ultrareview-cleanup-2026-07-24` (stray-file removal, dead-module deletion, `conda`→`uv` env-command
> reconciliation, this doc). Three items this report lists under Tier A were **reclassified as owner
> decisions and deferred to the Tier-B review**, because the repo already logs them as owner rulings in
> `docs/REVIEW_OPEN_QUESTIONS.md` (the 3 nomenclature live-prose slips: stamp-T3 vs reword) and because
> additively editing frozen records was previously reverted as BLOCKER-1. Branch reconciliation is
> prepared separately as a merge plan for owner approval. `RESULTS_INDEX`/`MANIFEST` registration of the
> untracked `human_only_full_spectrum.json` is deferred to Tier B (the master `MANIFEST.csv/json` is
> hand-maintained and enforces one row per tracked result).

Repo: `/Users/cberg/Research/projects/gfm-circuit-discovery`
Branch (actual HEAD): `viz-circuit-sequence-to-output` @ `c75cf52` (2026-07-23)
Review date: 2026-07-24 · Synthesis of 14 region reviewers (R1–R10, F1–F4) + 5 ground-truth sweeps (GT1–GT5)
Priorities: readability · reproducibility · documentation · formalization · + AGGREGATE NOMENCLATURE · REMOVE IRRELEVANT REFERENCES

---

## 1. Executive summary & overall verdict

**Verdict: NEAR-READY. Pass conditional on a bounded, mostly-mechanical cleanup pass. No scientific defect was found; every high-severity issue is a bookkeeping / provenance / staleness mismatch, not a wrong result.**

This is an unusually disciplined pre-publication repository. The ground-truth sweeps confirm the substrate is sound:

- **Repro stack is complete and GREEN.** `pyproject.toml` (4 extras) + committed `uv.lock` + `.python-version 3.13` + T1 `REPRODUCE.md`. Test suite runs **305 passed / 35 skipped / 0 failed / 0 errors** (107s); all 35 skips are data-gated/optional-dep, none are breakage. `MANIFEST.json` (770 rows) resolves 770/770 artifacts, 755/755 scripts, 521/521 manifests — **0 dangling paths**. `verify_manifest.py` passes. (GT4)
- **Links are clean.** 0 broken markdown links across all 162 tracked `.md` files; `REFERENCE_MAP.json` 229/229 moves applied cleanly; the self-contained dashboard has 0 external links. (GT1)
- **No code lands on a wrong number.** No code symbol contains any retired number or `splice_strength`; carrier symbols correctly use `PERIOD_1063`/`PERIOD_C10`/`T_P10`/`BAND_P10`. (GT3)
- **Determinism is asserted AND implemented** (`PYTHONHASHSEED=0` hard-fails in ~35 atlas scripts; `sorted(set())`/SHA-256 discipline). (GT4)
- **Zero TODO/FIXME/XXX/HACK** markers in any tracked file. (GT5)

Against that baseline, the findings cluster into a few recurring, fixable themes. **Calibration note: several reviewers' initial "critical/high" severities were downgraded on adversarial verification** — e.g. the branch-drift "critical" is a high, and three "orphaned/dead-code" highs (figstyle, verify_uniform, P3-manifest) are genuinely low once you see they are inert. The honest picture is **0 critical, 11 high, 50 medium, 30 low (91 findings; 0 REFUTED)**, and the highs are dominated by four correlated bookkeeping gaps that a focused day resolves.

### The four things that actually gate submission

1. **Branch drift (submission hazard).** Actual HEAD `viz-circuit-sequence-to-output` is named in **no** doc; four *different* branches are each claimed "current" across README/CLAUDE/ROADMAP/dashboard, and the default remote is `master`. An external reviewer landing on the repo link sees a branch none of the docs point to. → Merge the record to `master` (or reconcile pointers) before sharing.
2. **Branch-local ledger incompleteness (C37–C40).** This working branch's `CLAIMS.md`/`CORRECTIONS.md` cite C37 as established but carry no C37 row (8 C37 rows live in `MANIFEST.json`), and C37–C40 are missing from one or both ledgers. Several "high" R7/F1 findings collapse to *this single fact*: the branch is behind `master`. → Reconcile the branch, and most of R7/F1 resolves at once.
3. **Results-tree index gaps + one uncommitted canonical result.** `results/RESULTS_INDEX.md` omits the two newest tracked canonical dirs (`origin-crossmodel/`, `origin-siblings/`), two layout conventions coexist undocumented, and one canonical-labeled result (`origin-siblings/canonical/human_only_full_spectrum.json`) is git-**untracked** and in no manifest — a genuine provenance hole.
4. **Catalogue staleness.** `scripts/README.md` and the study `REPRODUCIBILITY.md` claim to be the "current record" but omit the entire current arc (A0–A12, P3 phase-surgery, IC1, B2/B3, siblings, carrier-ov) — the majority of live claims. A reviewer using the catalogue as the map is misdirected.

Everything else is medium/low polish: env-command inconsistency (`conda` vs `uv`), spent handoffs reading as live, 3 known nomenclature slips already logged as open items, ~8 stray untracked files, and a set of formalization gaps (prereg↔code) that are honest-to-log rather than must-fix-before-submit.

---

## 2. Findings by priority (deduplicated)

Severity shown is the **corrected** (post-verification) severity. `verdict` = adversarial-verification status where one ran. Findings reported by multiple reviewers are merged; the merge key is noted.

### 2.1 REPRODUCIBILITY

| Sev | File:line | Issue | Recommendation | Verdict / merge |
|---|---|---|---|---|
| **high** | README.md:9-11, CLAUDE.md:209, docs/ROADMAP.md:90, dashboard:310/688 | **Branch drift.** Actual HEAD `viz-circuit-sequence-to-output` named nowhere; 4 branches each claimed "current"; default remote `master`. | Merge record to `master` (or update all pointers to the real branch); remove live-branch strings from T1 docs. | CONFIRMED · merges R1-01 + R2-04 + GT5 |
| **high** | results/RESULTS_INDEX.md:56-61 | Top Directories table omits tracked canonical dirs `origin-crossmodel/` (11 rows) & `origin-siblings/` (9 rows) carrying C39/C40. | Add both rows with claim-ledger mapping. | R10-02 |
| **high** | results/RESULTS_INDEX.md:56-61 + tree | Two canonical layouts coexist undocumented: majority "canonical-at-root", `origin-*` use `canonical/` subdir. | Pick one repo-wide and state it in RESULTS_INDEX; or document the exception + rationale. | R10-01 |
| **high** | results/origin-siblings/canonical/human_only_full_spectrum.json | Canonical-labeled result (+.manifest) is git-**UNTRACKED**, in no MANIFEST or index; cited in TIER2_HANDOFF:255. verify_manifest passes only because it checks tracked files. | Commit both + register MANIFEST row + RESULTS_INDEX entry, or remove from `canonical/`. | CONFIRMED · merges R10-03 + GT2 #3/#4 |
| **high** | figures/splicebert/paper/fig1_hero.png (+fig2,fig3) | Paper figure set has no `.provenance.json` sidecars and 0 MANIFEST figure rows (violates figures/README rule #1); depends on untracked `viz/live_attention.json`. | Emit provenance sidecars (sha256 of inputs), commit/regenerate the live-attention input, add manifest rows or document as a declared exception. | R10-04 |
| **high** | studies/.../REPRODUCIBILITY.md:5,41-98 | Claim→script→artifact map omits A0–A12 (C27–C33), P3 (C34–C36), IC1, B2/B3, carrier-ov, siblings — the majority of current claims. | Add rows mapping each current claim to runner + indexed artifact. | CONFIRMED(med) · related R5-08 |
| med | CLAUDE.md:192 | Live run command `conda run -n gfm-thrust1 ...` contradicts uv-canonical REPRODUCE.md/README/AGENTS. | Rewrite to `uv run python ...`; demote gfm-thrust1 to a provenance parenthetical. | CONFIRMED · merges R1-02 + R5-09 + F4-01 + GT4 |
| med | scripts/.../build_example_covariates.py:10 | Docstring "Run under gfm-thrust1" contradicts uv workflow. | Change to `uv run python ...` (PYTHONHASHSEED=0); keep conda only in provenance. | R6-03 |
| med | src/gfmcd/splicing/dataset.py:73-74 vs coupling/competition/partner_data | chr19 chromosome split hard-coded in 4 modules with **two partitions** (dataset.py: chr19=SEALED TEST; others: chr19=DISCOVERY/VAL); no canonical definition. | Hoist one canonical split constant + `split_for`; import everywhere; document dataset.py's v1 split as historical. | CONFIRMED(med) · R4-02 |
| med | src/gfmcd/splicing/specificity.py:20-31 vs splice_landscape.py:117-151 | MaxEnt 3'/5' scoring implemented twice with **different invalid sentinels** (nan vs -100.0) — pooled scores would diverge. | Extract one shared MaxEnt-scorer helper (one invalid convention). | R4-04 |
| med | REPRODUCE.md:49,56-61 | C34/C35 row points to "runners named in the P3 manifest" but no P3 manifest exists; carrier files at results/ root are absent from phase-lattice/RESULTS_INDEX (the section reviewers are sent to). | Name the two concrete scripts; cross-ref the root-level P3 files. | CONFIRMED · merges F4-03 + F4-04 |
| med | tests/ (C34–C36) | ~20 P3 tests guard only analysis logic; none pins the headline numbers (C35 +0.284, C36 Ψ_C=+0.141). | Add skipif-gated memorial pinning the C34–C36 confirmation-artifact numbers. | R9-04 |
| med | tests/ (C38/C39/C40) | Recent live claims have **no** test coverage, unlike every prior C1–C33 family. | Add skipif-gated memorial tests loading C38/C39/C40 artifacts & pinning numbers/polarity. | PARTIAL · R9-01 |
| med | scripts/crossmodel/test_carrier_spectrum.py, scripts/splicebert/test_ic1_determinism.py | `test_*` gate files live outside `testpaths=["tests"]` — canonical pytest never runs them (incl. the C40 carrier-band amplitude precondition). | Wire into tests/ via thin importing wrappers, or document exact standalone invocation in REPRODUCE.md. | merges R9-03 + R6-05 |
| med | tests/test_circuit_memorials.py:26-27 (+test_partner_data.py) | Strongest behavioural memorials gate on hardcoded `/Users/cberg` genome paths → silently SKIP off-machine; reviewer gets zero signal. | Read genome from `GFMCD_GENOME_DIR` env var (default current path); document in REPRODUCE.md. | R9-05 |
| low | scripts/README.md / studies REPRODUCIBILITY.md:50 | "Circuit faithfulness" row maps to old run_faithfulness.py; current C27/C33 faithfulness is run_a6/run_a12. | Clarify canonical artifact or list both with claim IDs. | R5-08 |

### 2.2 DOCUMENTATION

| Sev | File:line | Issue | Recommendation | Verdict / merge |
|---|---|---|---|---|
| **high** | scripts/README.md:3-5 | T1 catalogue lists ~35/104 scripts and omits the entire current record (A-series, P3, IC1/IC2, B2/B3, siblings, carrier-ov) while claiming to be "only the active current scientific record". | Regenerate the index (or generate it programmatically); fix the mislabelling. | merges R5-01 + R6-02 |
| **high** | docs/REVIEW_OPEN_QUESTIONS.md:37,43-46,104-143 | States the development/superseded reorg was reverted and F1 "DROPPED", but it was re-applied as C1 and is live (99 files development/, 132 superseded/). | Append reconciliation note (F1 re-applied as C1) or retract the stale RESOLVED section. | R2-02 |
| **high** | studies/.../preregistrations/README.md:41,43 | B1(v2)/B3 index rows say "pending smoke gate"/"NOT survivor" but both executed (C37/C38) with artifacts + MANIFEST rows. | Update to "executed" with C37/C38 pointers at branch reconciliation. | R7-03 |
| med | studies/.../CORRECTIONS.md:6,20-57 & CLAIMS.md:62-63 | Self-described "complete claim ledger" omits C37–C40 that CLAIMS carries; C39/C40 lack prereg pointer + ledger row. | Add the four rows (or soften "complete"); document C39/C40 as unpreregistered natural experiments with manifest pointer. | merges R7-04 + R7-05 |
| med | studies/.../preregistrations/README.md:10-46 | Index omits two protocols present on disk (G1_CROSS_EXAMPLE_TRANSDUCTION, EXPLORATORY_P3_SHIFT_DIAGNOSTIC_V1). | Add both rows (G1=draft/not-frozen; V1=exploratory, superseded). | R7-08 |
| med | studies/.../P3_SHIFT_DIAGNOSTIC_V2_REPORT.md (+V1 prereg) | Orphaned exploratory report pair, unreferenced by README/prereg index. | Link from README/FINDINGS §5 as exploratory adjunct to C36, or move to archive/. | R7-07 |
| med | docs/HANDOFF_B2_B3_EXECUTION.md:14-15,31 | Spent handoff reads as live ("NEITHER HAS BEEN RUN") but both drivers ran; B2 retired. | Demote to T3/archive with "SUPERSEDED by B2_B3_EXECUTION_DISPOSITION; B2 retired". | R2-03 |
| med | docs/ (top level) | Doc sprawl: ~9 dated T2 execution/handoff/review records at top of docs/; remediation they drive is complete (F1–F4 PASS). | Move spent set under docs/archive/ or stamp T3; keep TIER1_HANDOFF until G1 dispositioned. | R2-05 |
| med | README.md:86-96 | Repository map omits live packages `crossmodel/` (C39/C40) and `figures/`. | Add both rows. | R3-02 |
| med | scripts/splicebert_head_context_atlas/ | ~40-file atlas dir has no README/index; head_screen vs atlas_ families unordered. | Add a stage-pipeline README; link from scripts/README. | R6-04 |
| med | REPRODUCE.md (C34/C35) | See §2.1 F4-03/04 (dangling P3 manifest pointer). | — | (in repro table) |
| low | README.md:113 vs CLAUDE.md:3/AGENTS.md:3 | CLAUDE.md & AGENTS.md self-declare T1 but README's T1 inventory omits them. | Add both to README's T1 enumeration. | R1-04 |
| low | README.md:106 | "Roughly 130 markdown documents" but 162 tracked (~25% low). | "over 160" or drop the count. | R1-06 |
| low | CLAUDE.md:62-160 | ~90-line dated "Current direction (in progress)" live-status block inside a T1 "authoritative" doc (fastest-ageing content). | Compress to a pointer into CLAIMS/CORRECTIONS/DECISIONS; move running status to a T2 status doc. | R1-05 |
| low | src/gfmcd/rigor.py:15-23 | Docstring shows kit-authoring/vendoring instructions; real callers use `from gfmcd.rigor import`. | Fix example import; keep VENDORED provenance note. | R3-04 |
| low | src/gfmcd/crossmodel/__init__.py:8-22 | `__all__`/docstring omit live public submodule `reader_scan`. | Mention reader_scan; optionally surface its fns. | R3-05 |
| low | studies/codon_phase_lattice/PREREGISTRATION_PHASE_SHIFT.md:3 | Stamped T2 inside a T3-historical study (inconsistent with caduceus DRAFT precedent). | Demote to T3 or add "superseded, retained for provenance". | R8-03 |
| low | studies/TEMPLATE_REAL_GFM_STUDY/RESULTS_TEMPLATE.md | Template ships RESULTS.md + RESULTS_TEMPLATE.md but README lists only the former. | Reference the template's per-run role, or fold into RESULTS.md. | R8-05 |
| low | figures/README.md:27-29 | Studies list omits tracked `figures/origin-crossmodel/`. | Add the row. | R10-08 |
| low | RESULTS_VERIFICATION.md:192 | Cites "CLAIMS.md (uncommitted edit)" no longer in the tree (protected by historical banner). | Optional note it is now committed; arguably a non-finding. | R1-08 / GT5 |
| low | F1-01 (C37 cited but no ledger row) | See §2.4 formalization — resolved by branch reconciliation. | — | merges R7-01 |
| low | studies/.../DECISIONS duplication | METHOD_CHARACTERIZATION §3 & LITERATURE_BIOFM_INTERP duplicate the same 4-source write-up. | Fold into one doc, or reduce LITERATURE to annotated-bib half. | R2-08 |

### 2.3 READABILITY

| Sev | File:line | Issue | Recommendation | Verdict / merge |
|---|---|---|---|---|
| med | src/gfmcd/splicing/competition.py:42-56 | Chromosome split represented in 3 shapes across sibling cohort modules (constants+free-fn vs frozen-dataclass fields). | Consolidate on the canonical split (R4-02) — one overridable definition. | R4-03 |
| med | scripts/splicebert/run_a4_address_geometry.py:1-12 (+run_localization, run_faithfulness) | Superseded runners lack a supersession banner; run_a4 presents as live though A11 redesigned it. | Add "SUPERSEDED by <runner>; retained for provenance" banner. | R5-04 |
| med | scripts/splicebert/run_ic1_...py:26-29 | 24 scripts import frozen helpers from sibling run_ scripts via `sys.path.insert` — hidden, non-relocatable dependency graph. | Migrate frozen helpers into src/gfmcd/ (frozen tag), or add "imports frozen helpers from: ..." header. | R5-05 |
| low | src/gfmcd/ (working tree) | Six `__pycache__`-only dirs for deleted modules (data, dicts, lab, models, nulls, tracer) mislead a browser into thinking gfmcd.tracer/lab exist. | `find src/gfmcd -type d -name __pycache__ -exec rm -rf {} +` (git history untouched). | R3-06 |
| low | src/gfmcd/circuits/attribution.py:25 | Imports underscore-private `_context_from_output`/`_split_context` from sibling hf_bert. | Promote to shared non-underscore module, or export intentionally. Optional. | R3-07 |
| low | src/gfmcd/splicing/p3_stats.py:27-71 vs p3_closure.py:560-594 | Gene-first paired modulo-3 contrast implemented twice over different schemas. | Optionally factor a shared core with schema adapter. Low priority (both tested). | R4-05 |
| low | results/phase-lattice/.../smoke_v1/ + results/reading-frame/figures/ | Empty untracked smoke dir; 6 untracked SVGs from retired reading-frame experiment in a committed-tree location. | Remove empty dir; move/gitignore the SVGs (history untouched). | R10-09 |

### 2.4 FORMALIZATION (do methods follow implementation?)

| Sev | File:line | Issue | Recommendation | Verdict / merge |
|---|---|---|---|---|
| **high** | docs/METHODOLOGY.md:128-130 | States corruption is a "length-preserving **one-base** edit ... differing at a single base" citing perturb.py, but live corruption is a **TWO-base** substitution (acceptor AG→CC) in circuit.py/coupling.py; the cited perturb.py `ablate_true_site` has **zero callers**. Science holds (still in-distribution/minimal); wording & citation wrong. | Rewrite to "length-preserving dinucleotide substitution (AG→CC)"; cite CORRUPT_ALT (circuit.py) + destroy_acceptor/destroy_donor (coupling.py). | CONFIRMED · F3-1 |
| med | studies/.../CLAIMS.md:35,61 | C12 "bounded by C37" but no C37 row on this branch (8 C37 rows in MANIFEST). | Bring C37 row into branch CLAIMS/CORRECTIONS, or reconcile to master. | CONFIRMED(med) · merges R7-01 + F1-01 |
| med | docs/BIOLOGICAL_BRIDGES.md:113,179,242 + NOMENCLATURE.md:26,35 | Two T1 docs still say carrier origin is "deferred/unproven/unknown" and "No frozen-checkpoint experiment can settle origin" — contradicts newer T1 docs + ledger (origin ESTABLISHED via C39/C40, R25). | Update to "corpus-attributed origin (C39/C40); only training dynamics deferred"; delete the "No frozen-checkpoint..." sentence at :179. | CONFIRMED · R2-01 |
| med | scripts/splicebert/run_p3_carrier_locus.py:210-252 (+run_p3_av_decomposition.py) | Neither P3 confirmatory script implements the frozen §8 "Global PASS = AND(...); Run STOPS on failure" hard gate — both always write results. This is why the R24 M1 failure (5.5%>5%) was reported instead of halting. | Implement the AND-gate halt (mirror A6/A9/IC1 preflight), or amend prereg §8 to record the gate is advisory. | CONFIRMED(med) · F2-01 |
| med | src/gfmcd/config.py:1-9,64-89 | Presents as an *enforced* over-claim "firewall" (raises), but assertions are called only by tests/test_config.py — never on the measurement path. | Wire assertions into adapter-load/experiment preamble, or soften docstring to "available guard, call explicitly". | R3-03 |
| med | run_p3_carrier_locus.py:187 | P3.2 omits prereg's energy-matched inert E-band memorial (M2/N1) + size-matched null-head restorations (M6/N3). | Add the E arm + ≥400 null-head restorations, or amend prereg to scope P3.2 to N2 only. | F2-02 |
| med | run_p3_av_decomposition.py:130-140 | P3.3 computes only margin cells; omits required pre-saturation U_info-projection readout + R23 E-band factorial. | Compute the U_info readout + E-band factorial, or amend prereg to record both deferred. | F2-03 |
| med | run_a9_payload_semantics.py:352 | A9 equivalence gate tests only MEDIAN transfer, not frozen "entire distribution within [0.80,1.20] at margin AND U_info". | Evaluate whole-distribution at both stages (uinfo array already captured), or amend §4. | F2-05 |
| med | run_a9b_...composition.py:569 | A9b verdict redefines frozen §6 "correct sign" (match A9 held-out direction) to D/H internal consistency — changed in code, not a logged amendment. | Use `sign_matches_a9` (already at L429), or log an explicit pre-run amendment. | F2-07 |
| med | run_ic1_...py:240 | IC1's preregistered `identity_noop_exact` memorial is hardcoded `True` ("by construction") — can never fail (R7/R21 unfireable-memorial pathology). | Actually run an identity permutation and assert ΔC==0 in the preflight, or amend §5. | F2-08 |
| med | run_a6_circuit_faithfulness.py:322 | A6 frozen §4 "health under F(C) scrub ≥ 0.90×clean" is neither implemented nor formally retracted (superseded by §2 memorial reframe, but §4 never updated). | Strike the health-multiplier clause consistently with the §2 amendment. | CONFIRMED(high→) · F2-11 |
| med | src/gfmcd/splicing/perturb.py:37-56 | The genuine one-base `ablate_true_site` METHODOLOGY cites has no callers; other perturb.py helpers used only by tests. | After F3-1 citation fix, wire it as the real corruption or retire perturb.py (decide library vs fixture). | F3-2 |
| low | run_p3_carrier_locus.py:224 | P3.2 monotonicity is non-strict (+0.05 slack), not anchored empty~0/all~1, vs frozen M5″ (strictly increasing 0→1). | Tighten to M5″, or amend M5″ to the implemented definition. | F2-04 |
| low | run_a9_...py:368-371 | A9 contrastive verdict never enforces frozen "reciprocal (i←j and j←i opposite-signed)". | Add reciprocity check, or amend prereg to say it's implicit in signed-transfer correlation. | F2-06 |
| low | run_ic1_...py:4,31 | Docstring labels protocol "(v6)" but code implements v8 (A1+A2 amendments). | Update docstring + "FROZEN v6 constants" comment to v8. | F2-09 |
| low | run_ic1_...py:267-274 | IC1 preflight omits several §5 HARD gates (protected_bit_identical, length_position_preserved, whole_intron_dinuc_preserved, perturbation_dose). | Surface as reported preflight checks, or amend §5 to record they're enforced by eligibility filter + unit tests. | F2-10 |
| low | src/gfmcd/splicing/metrics.py:98-107 | Canonical add-one p-value helper `matched_random_pvalue` is uncalled; formula reimplemented inline in ≥5 scripts (drift risk). | Route ad-hoc sites through the helper, or delete it. Not a methodology violation. | F3-3 |
| low | studies/.../CLAIMS.md:61 | C38 controlling-artifact cell names a `_n260` file that doesn't exist (n=260 run is the base file). | Rewrite as "(powered; + base [n=260], `_pilot`, `_smoke`)". | F1-02 |

---

## 3. AGGREGATE NOMENCLATURE — concrete plan

Ground truth (GT3): the controlled vocabulary in `docs/NOMENCLATURE.md` is **well-enforced**. There are **exactly three genuine live-prose violations, zero code-symbol violations**, and all three are already logged as unresolved open items in `docs/REVIEW_OPEN_QUESTIONS.md`. The remaining hits are legitimate retirement records, grid arithmetic, refutation experiments, or frozen preregs — **leave them alone** (NOMENCLATURE explicitly says "retirement records are not to be scrubbed").

### 3a. FIX — genuine live-prose violations (3)

| # | File:line | Current text | Fix to |
|---|---|---|---|
| 1 | figures/splicebert/fig_two_carriers.CAPTION.md:71 | "The carrier is ~10.67" | "the ≈10.63 carrier" (retired DFT-bin label used as the carrier's wavelength identity) |
| 2 | figures/splicebert/fig_two_carriers.CAPTION.md:67 | "at 10.67 it is not even a local maximum" | reference the ≈10.63 carrier band (retired 10.67 as wavelength) |
| 3 | data/NUCLEOSOME_MAP.md:38 | "attenuate the 10.67-nt signal" | "the ≈10.298-nt nucleosomal signal" — **double defect**: retired 10.67 AND conflates the model carrier number with the biological nucleosomal signal |

Also on fig_two_carriers.CAPTION.md (same cluster, sub-threshold but worth doing while editing): line 72 quotes the stage-1 value as "period 10.6667" as a measurement; the caption uses "10.69" as a wavelength ~7× (same DFT-bin-label anti-pattern; "10.69" is not on the explicit retired list but is flag-adjacent). After fixing 1–3, tick the three items closed in REVIEW_OPEN_QUESTIONS §1/§2.

### 3b. Standing-phrase completeness (medium/low — wording, not misleading)

- **SP1 (circuit first-use qualifier):** SATISFIED in README:22-23 and MECHANISM:57-60. GAPS (unqualified first use, verbatim phrase absent) in **CLAUDE.md:21, REPRODUCE.md:44, RESULTS_VERIFICATION.md:1/14, AGENTS.md:28**. RESULTS_VERIFICATION (titled "Circuit Discovery", reader-facing, no qualifier anywhere) is the strongest concern — add the standing phrase or a one-line pointer to the README/NOMENCLATURE definition at first use. (R1-07)
- **SP2 (≈10.63 + ≈10.298 same-sentence "0.33 nt / powered refutation"):** NO hard violation — every co-occurrence asserts distinctness, so no skimmer concludes "found nucleosome positioning." Verbatim qualifier merely missing at README:42, fig_two_carriers:63, RESULTS_INDEX:122, studies/splicebert/README:39. Add the verbatim qualifier for consistency (low). (R10-06 overlaps.)

### 3c. LEAVE (do NOT scrub) — legitimate uses confirmed by GT3

- All 10.67/10.6875/10.6667 hits used as **transparent grid arithmetic** tied to "≈10.63 carrier" (e.g. `PERIOD_1063 = 1026/96`), refutation experiments, test fixtures, and retirement records — ~40 sites. Clean.
- "10.58" — **no live violation**; every hit is vocab/retirement/explicit refutation of Kogan.
- "splice-site strength" — **no live violation**; withdrawn (R22) everywhere in canonical prose; present-tense survivals are only in DECISIONS_A0 log + frozen preregs (which also record R22); the external-MaxEnt-covariate use (`acceptor_motif_score()`) is the LIVE permitted use.
- bare "10.2" — **no violation**; all qualified "human-only ≈10.2" or the disambiguated 10.2–10.5 literature range.
- **No code-symbol violation** anywhere.

### 3d. ONTOLOGY.md ↔ NOMENCLATURE.md — reconcile by cross-link, do NOT merge (R8-01)

Different scopes: `docs/NOMENCLATURE.md` is the controlling authority for retired/withdrawn model-term status; `studies/splicebert_head_context_atlas/ONTOLOGY.md §H` is the atlas biological-annotation vocabulary but governs model-derived terms (≈10.63 carrier, U_info, R22) with no cross-reference. **Action:** ONTOLOGY §H points to NOMENCLATURE.md as the authority for retired/withdrawn model-term status; add ONTOLOGY to NOMENCLATURE's authoritative-detail list as the atlas vocabulary. No content merge.

---

## 4. REMOVE IRRELEVANT REFERENCES — removal/relocation list

Ground truth (GT1/GT2): the repo is remarkably clean — 0 broken markdown links, 0 tracked orphans in results/ or top-level, no stray tracked figures. The genuine items:

| Verdict | Path / reference | One-line justification |
|---|---|---|
| **DELETE** | figures/military-operations-to-object-graph.png (untracked, 2 MB) | Off-domain agent/architecture diagram; token "military" appears nowhere; 0 refs. Not a genomics artifact. (GT2 #1 · R10-07) |
| **DELETE** | plan_origin-evidence-for-the-period-3-10-63-c_df5fc482.json (untracked, root) | Orchestration planning scaffold dropped at repo root; doc hits are the authoring-session frame only, not file refs. (GT2 #2) |
| **DELETE** | figures/splicebert/paper/fig_L3H1_detail.{png,pdf} (untracked) | Not in the documented paper set (fig1_hero/fig2/fig3), not emitted by the tracked generator, 0 refs. (GT2 #5-6 · R10-07) |
| **DELETE** | figures/splicebert/paper/fig_attention_patterns.{png,pdf} (untracked) | Same — exploratory extra render, 0 refs. (GT2 #7-8 · R10-07) |
| **DELETE** | figures/splicebert/viz/_view_001_fig3_real_attention.png, _view_002_... (untracked) | Auto-generated snapshot captures, 0 refs. (GT2 #9-10 · R10-07) |
| **DELETE** | src/gfmcd/figstyle.py (tracked) | Dead, off-domain (grokking `fig_lambda_grok`, ANCHORED/RELAY roles), redundant with the actually-imported figures/style.py; no importer. History retains it. (CONFIRMED, downgraded to low · R3-01) |
| **DELETE** | scripts/verify_uniform.py (tracked) | Orphaned + import-broken; references retired GFM-nano; cannot run; referenced nowhere. History retains it. (CONFIRMED, downgraded to low · R6-01) |
| **DELETE (local)** | Six src/gfmcd/__pycache__-only dirs (data, dicts, lab, models, nulls, tracer) | Stale bytecode for deleted modules; misleads browser into thinking gfmcd.tracer/lab exist. Not committed. (R3-06) |
| **DELETE (local)** | results/phase-lattice/.../p3_phase_surgery_confirmation_smoke_v1/ (empty untracked) + results/reading-frame/figures/ 6 untracked SVGs | Working-tree clutter from retired reading-frame experiment. History untouched. (R10-09) |
| **COMMIT** | results/origin-siblings/canonical/human_only_full_spectrum.json + .manifest.json (untracked) | Genuine canonical C39-extension result (artifact 7bf703d7), cited in TIER2_HANDOFF, siblings all tracked — but uncommitted and in no MANIFEST/index. Commit + register, or remove from canonical/. (CONFIRMED · R10-03/GT2 #3-4) |
| **RELOCATE** | docs/superpowers/plans/2026-07-22-publication-readiness.md, 2026-07-14-donor-logit-anchor.md | Superpowers-skill agent-workflow scaffolding, not scientific/method record. Move to docs/archive/ or gitignore before external review. (R2-06) |
| **RELOCATE** | docs/HANDOFF_B2_B3_EXECUTION.md, ~8 other spent dated T2 handoff/execution/verification records | Remediation they drive is complete (F1–F4 PASS); they read as live. Move to docs/archive/ or stamp T3. (R2-03 · R2-05) |
| **RELOCATE** | studies/.../RUNBOOK.md | Reads as active but all 7 scripts it commands are absent from the tree; superseded by REPRODUCIBILITY.md. Delete or demote-with-banner. (CONFIRMED · R7-02) |
| **RELOCATE** | studies/.../B1_EXECUTION_HANDOFF.md | Spent "for whoever runs B1" guide; B1 executed (C37), 0 inbound refs, historical integration checklist. (R7-06) |
| **FIX-REF** | figures/splicebert/fig_phase_circuit/clock/wave/two_carriers.CAPTION.md:6 "Built by:" | Cite generator scripts (make_fig_phase_circuit/clock/wave/two_carriers.py) that don't exist. Restore real generator names, or state "preserved historical assets, no live generator". (CONFIRMED · R10-05/GT1) |
| **FIX-REF** | figures/splicebert/fig_two_carriers.CAPTION.md:24,34,71 | Cites claim IDs D22/D23/D24/D25 absent from live CLAIMS.md. Annotate as retired D-series. (R10-06) |
| **KEEP** | viz/live_attention.json (untracked); caduceus_t1_colab.zip, mock_caduceus.py, 3 figure generators, 2 viz HTML, HAND_ASSETS.md (tracked) | Regenerable viz input / legitimate build tooling / test double — GT2 verified all legitimate, not stray. |
| **KEEP** | docs/REFERENCES.md external works (Elhage 2021, Kendiukhov/Kadem&Zheng/Consens/Rood 2026) | Disciplined non-ledger use; no change now. **Confirm the 4 external arXiv/bioRxiv IDs resolve** and add verified REFERENCES entries only if any enters a manuscript. (R2-07) |

---

## 5. Staged remediation plan

### Tier A — safe mechanical (execute immediately; low risk)

| Item | Effort | Risk | Findings |
|---|---|---|---|
| Delete 8 stray untracked files + local __pycache__ shells + empty smoke dir + reading-frame SVGs | 10 min | none (history untouched) | GT2, R10-07, R10-09, R3-06 |
| Delete dead tracked modules `src/gfmcd/figstyle.py`, `scripts/verify_uniform.py` | 10 min | very low (no importers; verified) | R3-01, R6-01 |
| Fix env command: CLAUDE.md:192 + build_example_covariates.py:10 → `uv run`; demote gfm-thrust1 to provenance | 15 min | none | R1-02, R5-09, F4-01, R6-03 |
| Reconcile branch pointers: **merge record to `master`** (or update README:9-11, CLAUDE:209, ROADMAP:90, dashboard:310/688 to the real branch) | 30–60 min | low-medium (git op; decide merge vs pointer-fix with owner) | R1-01, R2-04, GT5 |
| Fix count/inventory nits: README:106 (md count), README:113 (T1 list), README:86-96 (add crossmodel/figures), figures/README (origin-crossmodel row) | 20 min | none | R1-04, R1-06, R3-02, R10-08 |
| Fix 3 nomenclature live-prose violations (fig_two_carriers:67/71, NUCLEOSOME_MAP:38) + close REVIEW_OPEN_QUESTIONS §1/§2 | 20 min | none (verbatim swap to ≈10.63/≈10.298) | GT3, R10-06 |
| Fix dangling caption "Built by" + D-series IDs on fig captions | 20 min | none | R10-05, R10-06 |
| Commit + register the untracked canonical result (human_only_full_spectrum) OR remove from canonical/ | 15 min | low (owner confirms canonical intent) | R10-03, GT2 |
| Tier stamps: demote spent handoffs (HANDOFF_B2_B3, RUNBOOK, B1_EXECUTION_HANDOFF, phase-shift prereg) to T3/archive; relocate superpowers plans | 30 min | low | R2-03/05/06, R7-02/06, R8-03 |
| Prune 2.6 GB `.claude/worktrees` (2 detached-HEAD orphans) | 5 min | none (local only) | GT5 |

### Tier B — judgment edits (owner sign-off before executing)

| Item | Effort | Risk | Findings |
|---|---|---|---|
| **Reconcile branch ledger C37–C40** into this branch's CLAIMS/CORRECTIONS (or merge master first — resolves ~6 findings at once) | 1–2 h | medium (correctness of ledger rows) | R7-01/03/04/05, F1-01, F2 C12 |
| Regenerate scripts/README.md catalogue + study REPRODUCIBILITY.md claim→script→artifact map to cover the current arc (A0–A12, P3, IC1, B2/B3, siblings, carrier-ov) | 2–3 h | medium (accuracy of mapping) | R5-01, R5-02, R6-02, R6-04 |
| Results-tree convention: pick canonical-at-root vs canonical/ subdir repo-wide; add origin-*/rows to RESULTS_INDEX | 1–2 h | medium (file moves) | R10-01, R10-02 |
| Paper-figure provenance: emit sidecars + commit/regenerate live-attention input + manifest rows or documented exception | 1–2 h | medium | R10-04 |
| Nomenclature standing-phrase completeness (SP1 in CLAUDE/REPRODUCE/RESULTS_VERIFICATION/AGENTS; SP2 verbatim qualifier in 4 spots) | 45 min | low | R1-07, GT3 |
| BIOLOGICAL_BRIDGES/NOMENCLATURE origin-status update (deferred → corpus-attributed C39/C40); ONTOLOGY↔NOMENCLATURE cross-link | 45 min | low-medium (must match ledger) | R2-01, R8-01 |
| REVIEW_OPEN_QUESTIONS reconciliation (F1 re-applied as C1); refresh stale line anchors; segment_attribution version-label fix (v1/v2/v3) | 1 h | low | R2-02, R2-09, R8-02 |
| Doc consolidation: fold METHOD_CHARACTERIZATION §3 / LITERATURE_BIOFM duplicate; compress CLAUDE.md live-status block | 1 h | low | R2-08, R1-05 |
| Config firewall: wire assertions into measurement path OR soften "enforced" docstring | 1 h | medium (behavioral if wired) | R3-03 |

### Tier C — larger formalization / science items

| Item | Effort | Risk | Findings |
|---|---|---|---|
| METHODOLOGY.md corruption description: one-base → dinucleotide AG→CC; re-cite circuit.py/coupling.py; decide perturb.py fate | 1 h | low (wording only; science holds) | F3-1, F3-2 |
| Canonical chromosome-split constant: hoist one definition + `split_for`, import in coupling/competition/partner_data; document dataset.py v1 divergence | 2–3 h | **medium-high** (touches the discovery/val/test partition — verify no result shifts) | R4-02, R4-03 |
| Shared MaxEnt-scorer helper (one invalid-window convention) across specificity.py / splice_landscape.py | 2 h | medium (numeric — re-verify pooled scores) | R4-04 |
| Test coverage: add skipif-gated memorials for C34–C36 and C38/C39/C40 numbers/polarity; add C38/C40 to test_manifest polarity-pin (**R9-02: relabelling C40 null→positive currently PASSES** — the one confirmed test-integrity gap) | 3–4 h | low-medium | R9-01, R9-02, R9-04 |
| Off-machine memorials: `GFMCD_GENOME_DIR` env var + document; wire out-of-tree `test_*` gates into tests/ | 2 h | low | R9-03, R9-05, R6-05 |
| P3 prereg↔code conformance: implement the §8 AND-gate halt (or amend to advisory); reconcile A6 §4 health clause; A9/A9b/IC1 gate reconciliations | 4–6 h | medium (prereg discipline; several are amend-the-log, not change-the-code) | F2-01/02/03/05/07/08/11 |
| P3 manifest sidecars (p3_carrier_locus / p3_av_decomposition) + populate MANIFEST manifest field; fix REPRODUCE C34/C35 pointer | 1 h | low (provenance already inline) | F4-02, F4-03 |
| Retire/label orphaned v1/v2 splicing scaffolding (partner_readout, rivals, scoring, zero_shot, training) with a T3 banner — **do not auto-delete** (may be intentional negative-result scaffolding) | 1 h | low | R4-01 |
| Add per-script prereg/claim headers to ~45 run_ scripts lacking them; supersession banners on superseded runners | 2–3 h | low | R5-03, R5-04 |

---

## 6. False positives ruled out (so the owner can trust the rest)

**No finding was REFUTED on adversarial verification** — the confirmed set is solid. But several reviewer severities were **downgraded**, and the GT sweeps ruled out whole classes of would-be problems. Trust-calibration notes:

- **Severity downgrades (initial → corrected):** branch drift critical→**high**; figstyle.py, verify_uniform.py, F4-02 P3-manifest, R3-01 all high→**low** (genuinely inert/mitigated); R4-02 chr19 split, R5-02, R7-01/02, R8-02, R9-01, F2-01, F4-01, F2-11 high→**medium**. These are real but not blockers.
- **R9-01 is PARTIAL, not full:** C38/C39/C40 lack *memorial* tests, but they are not wholly untested — treat as a coverage gap, not zero coverage.
- **Nomenclature is NOT broadly broken:** GT3 confirmed only **3** genuine live-prose violations (all already logged, all in T2/data docs), **0** code-symbol violations. "10.58", "splice-site strength", and bare "10.2" have **no** live violations — every hit is a retirement record, refutation, grid arithmetic, or correctly-qualified closed term. Do not mass-scrub.
- **Links & orphans are clean:** 0 broken markdown links (GT1); 0 tracked orphans in results/ or top-level; no stray *tracked* figures (GT2). The "irrelevant references" surface is small and untracked-only (plus 2 dead tracked scripts).
- **Repro is GREEN:** suite 305 passed / 0 failed; MANIFEST 0 dangling; verify_manifest passes; uv.lock committed (GT4). The reproducibility findings are about *documentation of* the path (env command, catalogue coverage, index gaps), not about the path being broken.
- **Confirmed-legit, flagged-but-KEEP:** viz/live_attention.json, caduceus_t1_colab.zip, mock_caduceus.py, the 3 figure generators, 2 viz HTML pages — all verified legitimate build tooling / test doubles, not orphans.

**Bottom line:** publication-ready after Tier A (a focused half-day) plus the two Tier-B reconciliations that matter most — **merge/reconcile the branch ledger (C37–C40)** and **regenerate the script/results catalogues + results index**. Tier C is honest formalization polish that strengthens the record but does not gate an initial external review, with one exception worth doing early: the **R9-02 polarity-pin gap** (C40 null→positive would currently pass the suite).
