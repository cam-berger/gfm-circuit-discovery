# Decision log — Head–Context Atlas

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Design decisions and post-diagnostic pivots. Preregistered confirmatory choices live in
`PREREGISTRATION_C_COMPETITION.md`; this log records rationale and the red-team pivot.

## 2026-07-18 — Estimand framing: C primary, A census, B control
- **Trigger:** scoping the atlas against the closed C1–C35 record.
- **Evidence:** A (general reconstruction) and B (partner coupling) largely re-derive established
  territory; C (competitor-conditioned selection) is the genuinely new claim.
- **Decision:** C is the sole primary confirmatory estimand; A is a descriptive all-head prevalence
  census (methodological fix to the "DAG-frequency ⇒ common head" fallacy, not a hypothesis test); B is
  the outcome-blind positive-control gate (recover L3H1/L2H15 coupling).
- **Alternatives:** all-three co-primary (weaker family, stakes claims on closed territory); A-primary
  (weakest novelty).

## 2026-07-18 — Session stop-line: scaffold + census + outcome-blind preflight
- **Decision:** this session builds the scaffold, frozen ontology, annotation-only census, and an
  outcome-blind preflight that gates BEFORE any primary is computed; sealed chr8/9 stays untouched.
  Preserves confirmatory status per IC1/R22 ("seeing the outcome forecloses confirmatory status").

## 2026-07-18 — C counterfactual: destroy competitor + matched decoy-destroy
- **Decision:** `M_comp = margin(t | competitor present) − margin(t | competitor destroyed w/ REAL bases)`,
  target+partner held intact. Two matched IUT controls: (1) matched decoy-destroy (same target, destroy a
  matched non-competitor) → competitor-specificity; (2) matched single-site (isolated matched target) →
  competition-context vs generic reconstruction. QK/OV separation = C35-style A-vs-V **main effects**,
  never an A×V interaction (R23).

## 2026-07-18 — Census substrate: comprehensive GTF, defer RNA-seq
- **Trigger:** the only on-disk GTF was `gencode.v50.basic`, which drops minor alternative isoforms —
  biasing the primary competition cohort.
- **Decision:** download + sha256-pin the **comprehensive** GENCODE v50 GTF (`83fba3e9…13a453`). Evidence
  tier v1 = annotation-derived only. **Four corrections (Cam):** (1) `annotation_isoform_support` →
  `n_annotated_transcripts_for_site` + `n_annotated_transcripts_for_junction`; (2) recount3/intropolis (v2)
  can populate `junction_observed`+`expressed_unannotated`, but `activated_cryptic` needs
  condition-/variant-linked evidence (→ v3); (3) preflight positive is coupling **B** > 0.05 (single-pair
  has no C); (4) SESOI-C calibrated WITHIN the C intervention/units — the null envelope is detectability,
  not the meaningfulness floor; L0H10's +0.116 retired as a cross-estimand import.

## 2026-07-18 — Confirmatory core (frozen): HC1 existence, HC2/HC3 secondary
- **Decision:** HC1 primary = existence via **max-T FWER omnibus / 96 heads → BH-FDR to enumerate →
  per-head IUT** (floor ∧ > decoy-destroy ∧ > single-site) ∧ stable across all seeded chr8/9 draws ∧
  common-support+health. HC2 (L3H1 pre-frozen) = sham-adjusted QK route redistribution + OV
  **equivalence-bound** stability + importance-matched A×V null. HC3 = one global recruitment-burden stat +
  one ordered slope over `cluster_size`/`local_degree`. `{HC2,HC3}` a separate BH family gated on HC1.

## 2026-07-18 — Compute doctrine: Colab CUDA, all 96 heads exact
- **Decision (Cam):** confirmatory venue = Colab CUDA on Algoverse credits; evaluate **all 96 heads exact**
  (completeness, not a discovery shortlist). Per-`(target,competitor,decoy,condition,head)` immutable tuple
  cache so the seeded draws re-index one table (no N× forwards). Resumable, chunked, outcome-silent,
  float32 + TF32/AMP off until an agreement gate, no attention-matrix capture except L3H1/HC2. Credits buy
  measurement completeness, **not** more hypotheses.

## 2026-07-18 — Pre-compute 6-lens design red-team → NO_GO_REDESIGN; decision = fix-in-place
- **Trigger:** house pattern (IC1 5-lens, P3 4-lens) — adversarial design red-team before any compute.
- **Evidence:** 3/6 lenses FATAL (`REDTEAM_SYNTHESIS_C_6lens.md`). Four convergent, code-grounded reasons:
  (1) `E_ih` bounded by likely-null `M_comp` (IC1 ΔC_specific CI incl 0; IC2 β_transport SD 0.774
  unreachable); (2) third-order contrast underpowered; (3) single-head existence contradicts C27/C33
  distribution; (4) cohort built on the exon-mate `attested` map, not the intron-mate partner + collider
  selection. Synthesis: **NO_GO_REDESIGN**, recommended demotion to a cheap-MPS behavioral program.
- **Decision (Cam):** **fix-in-place**, still targeting the Colab confirmatory run — apply all must-fixes
  (MF1–MF11) and keep the paid run on the table, with a **hard pre-compute power/behavior gate baked into
  the Colab runner** (cheap cohort census + MDE first; STOP → INSTRUMENT-LIMITED before the bulk spend).
- **Effect on prereg:** cohort switched to **intron-mate** (`competition.py`, not `attested`); intervention
  → **scrub-to-corrupt** (not mean-ablation); estimand → **group/faithfulness-size** (per-head null
  expected); decoy → real strength-matched to the competitor; collider-aware between-cohort matching;
  channel-faithful calibration; SLOT positive control; pooled seeded draws with a stability-SD trigger.
- **Files:** `DESIGN_DRAFT.md`, `REDTEAM_SYNTHESIS_C_6lens.md`, `src/gfmcd/splicing/competition.py`.

## 2026-07-18 — STAGE 0 census result: cohort exists but short-intron-biased; power anchor-dependent
- **Evidence (`results/.../census_v1.json`):** intron-mate usable competition = 3,014 sealed examples /
  444 genes (32,054 / 4,864 discovery) — ~200× the red-team's ~15-loci worst case. But only ~10% of
  competition targets fit a 1024-nt window (median intron ≈3.8 kb) ⇒ short-intron biased. Projected MDE is
  licensable under IC1's first-order SD (0.033<0.05) but NOT under IC2's DiD SD (0.061>0.05); since `E_ih`
  is third-order, the pessimistic anchor governs.
- **Decision:** resolve the power/behavior question empirically with a discovery-side behavioral pilot
  (measures real SD + tests `M_comp` existence + B) before configuring the Colab run.

## 2026-07-18 — STAGE 1 behavioral pilot outcome (n=400, 370 discovery genes)
- **Evidence (`results/.../behavioral_pilot_v1.json`):**
  - **Gate B (partner coupling): +0.175, CI [+0.124,+0.229], `material`** — instrument works; a `M_comp`
    null would be EVIDENCED, not defaulted. (alt5ss +0.108, alt3ss +0.244, both material.)
  - **`M_comp` signed: +0.006, CI [−0.026,+0.040], `bounded-null`** — no systematic directional effect.
  - **`|M_comp|`: +0.111, CI [+0.082,+0.142], `material`** — a competitor DOES move the target margin
    (~0.11 nats), sign-heterogeneously. So MF1's "no behavior to attribute" is NOT cleanly satisfied —
    there is a magnitude behavior.
  - **`M_comp − sham`: −0.015, CI [−0.067,+0.036], `inconclusive`** — the crude matched-distance 2-bp sham
    moves the margin comparably; **competition-specificity is UNPROVEN** (the R22/MF6 concern, live).
  - **SD(M_comp) ≈ 0.31** (gene 0.317) — between IC1 (0.162) and IC2 (0.774); the n=8 smoke's 0.017 was a
    low-variance fluke. Projected per-head MDE at sealed n ≈ 0.024 (examples) / 0.065 (genes) — straddles
    SESOI 0.05; `E_ih` (third-order) is a lower bound, so per-head confirmation is marginal → the
    group/faithfulness estimand (MF3) is the right primary.
- **Verdict:** behavioral gate is **AMBIGUOUS** — a material magnitude effect (encouraging) with UNPROVEN
  competition-specificity (concerning). The crude sham is a weak control; the decisive test is `M_comp` vs a
  **real, strength-matched, non-competitor matched-decoy-destroy** (MF6). That specificity pilot is the
  next MPS gate and a hard precondition for any Colab spend.
- **Effect on prereg:** the behavioral gate is split into (i) B positive control [PASS], (ii) `|M_comp|`
  existence [PASS], (iii) **competition-specificity `M_comp − matched-decoy-destroy` [UNRESOLVED → REQUIRED
  before confirmatory]**. INSTRUMENT-LIMITED if (iii) is bounded-null/inconclusive with the real decoy.

## 2026-07-18 — STAGE 1b specificity verdict: INSTRUMENT-LIMITED, Colab NOT licensed
- **Evidence (`results/.../specificity_pilot_v1.json`, n=300 / 274 discovery genes):** with a REAL
  MaxEnt-strength+distance-matched non-competitor decoy — `|M_comp|` = +0.088 [0.061,0.122] material;
  `|M_decoy|` = **+0.135** [0.073,0.212] material; **specificity `|M_comp|−|M_decoy|` = −0.047
  [−0.118,+0.011], inconclusive (negative)**. Both arms agree (alt5ss −0.015, alt3ss −0.074). Destroying a
  matched non-competitor moves the target margin **as much or more** than destroying the true competitor.
- **Verdict (frozen decision rule, prereg §2/§9):** the competition magnitude effect is **NOT
  competition-specific** — it is generic proximal same-type-motif perturbation ⇒ **INSTRUMENT-LIMITED /
  no competition-specific behavior to attribute**. Coheres with IC1/IC2.
- **Decision:** the pre-compute specificity gate baked into the fix-in-place plan has FAILED ⇒ the Colab
  confirmatory head-atlas run is **NOT licensed**; **Algoverse credits are not spent**. This is the credit
  protection working as designed. HCA4 → null; HCA6 → not-licensed; terminal claim HCA7 recorded.
- **Next (Cam's call):** write up the negative result (extends the IC arc: competition is not a
  competition-specific driver of SpliceBERT reconstruction at the ≤1 kb intron-mate scale); OR probe a
  variant (e.g. a different behavioral readout, cassette-exon inclusion, or a group-level readout on the
  discovery split) as a NEW prereg. The census + pilots stand as the visualizable deliverable.

## 2026-07-19 — Second mechanical audit: acceptor rows withdrawn

- **Trigger:** exact group/MLP closure required reusing the target-coordinate contract. The screen runner
  masked `target_win` for both donors and acceptors, while the frozen acceptor MLM readout masks
  `target_win-1`.
- **Decision:** withdraw every acceptor-side 96-head result, all donor-versus-acceptor head comparisons,
  and pooled profiles that mix the two. Preserve the valid donor rows and the independently correct
  behavioral pilots. Add one coordinate function plus a regression test before any new screen.
- **Evidence:** `AMENDMENT_HEAD_SCREEN.md` §6 and `head_effect_donor_valid_reanalysis_v1.json` (3,549
  valid train donor examples / 2,244 genes). The donor positive result strengthens: L3H1 B +0.114
  [0.098,0.130]. The reciprocal acceptor computation is now **untested**, not null.

## 2026-07-19 — Exact donor-only group/MLP closure; competition branch stops

- **Design:** outcome-aware discovery diagnostic on 400 train donor genes; exact joint scrub-to-corrupt
  patches at k={1,3,5,10,20,40,64,96}; layer-matched random groups; each MLP and all six MLP writes;
  no summed single-head approximation; chr8/9 excluded.
- **Correction within closure:** v1 B/MLP results are valid, but its C/D rank inherited pooled invalid
  acceptor rows. Derive a donor-only rank and rerun only C/D joint groups as v2; preserve v1 unchanged.
- **Result:** B is concentrated (L3H1 ≈84% and top 10 ≈99% of the signed behavioral mean). C/D are the
  same broad response family. A small post-hoc top-three C>D exact-magnitude excess survives, but k≥5,
  all-head behavior, and all MLP contrasts are inconclusive.
- **Decision:** no competition-specific component/module claim, no chr8/9 run, and no further Colab spend
  on this branch. Record the top-three set as a hypothesis-generating atlas candidate only.

## 2026-07-19 — Return to the original Head–Context Atlas

- **Decision (Cam):** finish the competition branch, then restore the original program: compare all 96
  heads and six MLPs across overlapping biological sequence categories to distinguish general-use from
  context-associated components; validate shortlists with exact group/QK/OV/MLP/DAG interventions;
  publish validated artifacts through `gfm-visualizer`.
- **Persistent scope:** competition C is one secondary atlas cell, never the organizing question.
  Experiments remain in `gfm-circuit-discovery`; visualization remains in `gfm-visualizer`.
- **Next:** multi-label atlas census and matched packet construction, followed by a corrected donor/acceptor
  coordinate + masked-MLM-health preflight. See `RESEARCH_SPINE.md`.

## 2026-07-19 — Wider atlas taxonomy: overlap is preserved; controls are not relaxed

- **Trigger:** matching a six-family atlas exposed a real distinction between biological taxonomy and
  statistical identifiability. Cam authorized loosening category/cohort generation if useful.
- **Decision:** use that flexibility to make the ontology more honest, not to make a gate easier to pass.
  Core labels remain overlapping/hierarchical; `single_pair` is explicitly packet-scoped; donor clusters
  are tested on donor targets and acceptor clusters on acceptor targets; cassette/complex labels may
  overlap. Reverse shared-partner roles require separately named future tags.
- **Controls retained:** case/control pairs still require exact equality of all non-focal core tags and
  coarse biological strata, different genes, capped control reuse, propensity/caliper matching on
  continuous confounds, and maximum absolute numeric SMD ≤0.10. No failed contrast is rescued by
  weakening these controls.
- **Pilot record:** matching v1–v6 remain immutable failed/superseded designs (numeric imbalance,
  support separation, over-fragmentation, or thin validation strata). v7 lowers only the
  internal-validation feasibility minimum to 80; it does not loosen balance. All ten role-valid v7
  contrasts pass in both splits.
- **Precision boundary:** the 107/130-pair validation single-pair arms are exploratory until an
  outcome-specific MDE is available. A smaller balanced cohort can test directional replication but
  cannot license an equivalence null by declaration.
- **Audit:** `atlas_context_matched_v7.audit.json` independently checks joins, tags, roles, genes, splits,
  sequence coordinates/motifs, reuse, balance, hashes, and leakage: **PREFLIGHT_PASS, 0 failures**.
- **Next:** discovery-only all-96-head/all-six-MLP mechanical model preflight. chr8/9 remains untouched.

## 2026-07-19 — Wider-atlas model instrument passes; Colab package frozen

- **Preflight:** choose one donor matched pair and one acceptor matched pair that jointly cover all six
  overlapping labels. Evaluate case and control packets under A (zero head/MLP output) and B
  (partner-destroyed source scrub), plus four genuinely masked non-site probes. This is a plumbing test,
  not a rank screen.
- **Result:** 4 packets × 102 components × 2 estimands = 816 rows; 0 failures. Corrected donor/acceptor
  masks, deterministic repeat, tokenizer, finite outputs, and complete grids all pass. All-head and
  all-MLP source-to-source identity errors are exactly 0. Masked-health accuracy = 0.688 and mean
  four-base log-probability = −0.937, both above frozen floors.
- **Independent bulk-path check:** the resumable runner re-evaluates the same packets through its own
  shard path. Across A/B target effects and A/B masked-health effects, all 2,448 comparisons reproduce
  the preflight exactly (maximum absolute difference 0.0). Shard validation and gzip merge pass.
- **Compute design:** evaluate each of 11,807 unique packets once; recover the 6,487 context contrasts by
  pair-table join. Run a deterministic 512-packet masked-health panel first, then the full target matrix.
  This avoids duplicating packets across overlapping labels and cuts collateral-health compute to the
  amount needed for a component-level gate.
- **Package:** self-contained 80,245,583-byte Colab zip, SHA-256
  `4bf5696d640b40d6a34dafd1461a5a547227cb3f851babda0399f2fddeadc58f`; checkpoint and discovery
  sequences bundled; no GTF/FASTA download; no validation/sealed files. Notebook enforces verify → tests
  → cross-device reference smoke → benchmark → health → explicit GO → target → validate → merge.
- **Claim boundary:** HCA12 is instrument/deployment evidence only. No preflight value ranks a component
  or enters the wider-atlas biological analysis.

## 2026-07-19 — Discovery analysis frozen before full outcomes; independent smoke audit passes

- **Freeze:** `ANALYSIS_PLAN_V1.md` pins A/B definitions, `tau=0.05` nats, gene-clustered general-use
  estimates, two-endpoint gene-dyadic matched differences, separate A/B BH families, fixed gene-hash
  stability halves, and deterministic shortlist caps (four general; eight context).
- **Smoke-only amendment A1:** after viewing only the four-packet/two-pair plumbing smoke, clarify that
  generic masked-MLM effects annotate broad use rather than silently excluding it; require unique
  unordered endpoint-gene dyads for the frozen variance formula; encode empty halves as strict JSON
  `null`; require an independent full-analysis audit. No effect threshold, correction family, stability
  count, ranking, or shortlist rule changed.
- **Verification:** analysis smoke v2 writes 408 general and 408 context rows over all 102 components,
  selects zero candidates by construction, contains no `NaN`/`Infinity`, and passes the independent
  estimator/join/hash/split/selection audit with 0 failures. The earlier failed verifier audit is retained:
  it incorrectly rejected the expected empty smoke half and prompted the strict-null fix.
- **Boundary:** HCA13 establishes only that the analysis instrument works. The full discovery matrix has
  not been read here; no head is called general, context-associated, or mechanistically validated.

## 2026-07-19 — Generic mechanism rules frozen; QK/V source bridge is exact

- **Plan:** before the full discovery matrix, freeze the candidate-specific handoff: deterministic
  importance-matched controls and representative packets; exact groups; B-specific QK/V/QKV source
  patches; honest A-only matched-control interchange semantics; MLP/write-level path controls; Holm
  whole-endpoint and within-endpoint max-|t| families; evidence-graded DAG export.
- **Seal:** the package builder reads only audited discovery artifacts and records internal-validation
  filenames without opening, hashing, counting, or sampling them. The smoke package correctly contains
  zero candidates/groups and passes an independent seal audit with `validation_loaded=false`.
- **QK/V preflight:** on corrected donor and acceptor packets, fixed heads L0H0/L3H1/L5H15 span all
  architectural depths. Native QKV→native, corrupt QKV→whole-head, and whole-head→independent-preflight
  comparisons are exactly 0.0 for target and masked-health forwards; 14/14 gates pass. The first artifact
  audit's only failures were path-string normalization (relative versus absolute); it is preserved and the
  corrected independent audit passes.
- **Boundary:** HCA14 is plumbing/sealing evidence. QK/V values from these four packets are not ranks and
  cannot name routing or writing. Candidate-specific packaging waits for the full audited shortlist;
  internal validation remains untouched.

## 2026-07-19 — Whole/group validation executor and inference are sealed end to end

- **Access-order correction:** the row label is `atlas_internal_validation`, but the physical artifacts
  are `.validation.*`. Package v3 seals those real filenames as portable repository-relative references.
  The executor verifies package + independent audit before resolving, hashing, counting, or opening them.
- **Amendment A1:** freeze a 256-packet masked-health precondition, exact candidate/control/group
  interventions, shared-computation reuse, target scopes, the primary whole-candidate family, Holm
  detection plus conservative Bonferroni simultaneous intervals, familywise MDEs, the L3H1/B donor
  control, and progression rules. Internal gene-hash validation remains distinct from chr8/9, which stays
  sealed.
- **Execution smoke:** four discovery packets exercise A/B attention heads, A/B MLP writes, mixed
  head+MLP groups, health, target, and an immutable no-op resume. There are 11 semantic endpoints but ten
  physical computations; aliases agree exactly.
- **Independent audits:** singleton target and health effects reproduce the earlier 102-component model
  preflight with maximum absolute error 0.0. A separate inference verifier reconstructs all eight primary
  smoke endpoints, 13 diagnostics, clustered/dyadic statistics, multiplicity, intervals, classifications,
  progression, positive control, and hashes with zero failures.
- **Boundary:** HCA15 is instrument evidence only. No full discovery outcome or internal-validation model
  outcome informed these rules; no component advances or is exported until the full audited sequence is
  complete.

## 2026-07-19 — Conditional QK/OV/MLP/path stage sealed and independently reconstructed

- **Amendment A2:** only whole-effect survivors advance. B attention uses the same-packet
  partner-destroyed state for population QK and V/OV interventions; QKV is an exact whole-operation
  bridge and never an interaction/binding statistic. A attention has no population QK/V decomposition.
  Every eligible later MLP and validated same-group upstream attention path is fixed before stage-two
  forwards.
- **Family:** each validated whole endpoint receives one joint family containing its frozen exact-group
  effect and conditional QK/V/MLP-path members. Inference uses joint gene-clustered or inverse-reuse
  unique-dyad covariance and Gaussian max-|t| (200,000 draws, seed 0), with simultaneous intervals,
  familywise MDE, and the same 0.05-nat material floor.
- **Mechanical fixes before retention:** candidate IDs were added to whole-reference keys; every B QKV
  candidate now receives its own bridge/identity gate even after partial resume; the executor hash and
  complete candidate grid enter the contract; stage-one package identity is rechecked at execution. A
  final resume audit caught that the old no-op still rewrote manifests; v4 leaves all shard and manifest
  hashes byte-identical while reporting the no-op only on stdout.
- **Smoke:** three fixed candidates exercise B QK/V, A/B downstream MLP recovery, and a B upstream
  attention→MLP path across both roles. All independent whole references, native identities, QKV bridges,
  and derived arithmetic have maximum absolute error 0.0. The independent inference verifier rebuilds
  six endpoint families, 34 family members, 56 diagnostics, covariance, max-|t| draws, classifications,
  relations, and hashes with zero failures.
- **Boundary:** HCA16 is instrument evidence only. Smoke classifications are not scientific. No function
  name, representative DAG, or visualizer export is permitted until the full discovery, whole validation,
  conditional inference, and both independent audits pass in order.

## 2026-07-19 — Representative DAG handoff made evidence-graded and fail-closed

- **Graph semantics:** case graphs retain nucleotide-indexed activation×Jacobian screening paths across
  all six layers and explicit remainder nodes. Matched negative-context graphs are separate and carry
  `negative_control` evidence. Neither becomes an exact causal graph.
- **Population overlay:** whole-component, exact-group, QK, V/OV, and MLP-path evidence lives in a
  separate overlay copied from the independently audited simultaneous-family result. Component-level QK
  support never upgrades a particular nucleotide edge. Edge width uses absolute signed effect; every
  interval and classification remains attached.
- **A boundary:** matched-control QK/V state transplantation runs only on the frozen representative and
  exports exact margin changes plus position maps as `representative_only_descriptive_no_population_label`.
  It cannot validate or decompose A.
- **Compatibility fix:** the older screening exporter depended on the removed Transformers
  `transpose_for_scores` helper. It now uses an explicit `[B,S,H,Dh]→[B,H,S,Dh]` reshape, preserving the
  same attention math across installed versions.
- **Smoke/audit:** two donor representatives (fixed A and B attention candidates) each include a case and
  matched negative graph. Each graph has 193 nodes and 215 edges; correctness gates pass, remainder mass
  reconstructs, every evidence label is legal, and the independent audit reports zero failures.
- **Boundary:** HCA17 is instrument evidence. The bundle is structurally `SMOKE_NOT_EXPORTABLE` with
  `export_eligible=false`. Only a full post-stage-two bundle plus zero-failure independent DAG audit may be
  adapted into `gfm-visualizer`.

## 2026-07-19 — Full wider-atlas discovery and whole-component validation

- **Discovery execution:** the complete Colab matrix contains 11,807 target packets, 512 masked-health
  packets, all 96 heads plus six MLP writes under A and B, and all 6,487 frozen matched pairs. Every shard,
  grid, split, hash, and independent analysis reconstruction gate passes.
- **Discovery result:** four candidates pass the frozen general-use shortlist: A/L2H15, A/L3H1,
  A/L1H12, and B/L2MLP. Twenty context rows are material, BH-significant, and cross-half stable: 11 in
  acceptor clusters, six in donor clusters, and three in donor-side cassette exons; MLP-write effects
  dominate. None passes the separate frozen precision requirement (`MDE80≤0.05`; range 0.090–0.299),
  so none advances to mechanism validation. This is an instrument-limited shortlist, not an absence of
  context-selective computation.
- **Seal order:** the candidate-specific mechanism package was built and independently audited before the
  gene-disjoint internal-validation outcomes were opened. The package hash is
  `48ac6b1791f2ddcd8f9ce54918afbd0f6067d87b1fa1d88ab535105fc28144cf`.
- **Whole validation:** all four candidates validate in both donor and acceptor reconstruction; all eight
  FWER intervals clear the 0.05-nat material floor and no endpoint carries a broad masked-MLM health flag.
  L3H1/B also passes its prespecified positive-control diagnostic.
- **Boundary:** the validated statement is component-level general use. It does not assign a biological
  function or decompose A into QK/OV. Conditional labels and visualizer export still require the audited
  stage-two and representative-DAG sequence.

## 2026-07-19 — Stage 2 Colab packaging defect reproduced and repaired before target execution

- **Observed failures:** the first notebook could reach its target cell without defining `stage2_out`
  after a runtime/cell-state mismatch. The attempted v2 notebook then failed at import with
  `ModuleNotFoundError: run_atlas_mechanism_validation`.
- **Root cause:** `gfm_atlas_stage2_colab_overlay_v1.zip` packaged the Stage 2 entry point but omitted a
  direct sibling module introduced after the discovery bundle was frozen. Adding the repository root to
  `PYTHONPATH` could not make an absent file importable.
- **Reproduction:** a clean extraction of the original discovery bundle plus overlay v1 reproduces the
  same import failure before any model forward or target outcome read.
- **Fix:** versioned overlay v2 carries the complete sibling-import closure and an internal per-file hash
  manifest. The self-contained v3 notebook redefines fixed paths in every execution cell, verifies the
  base and overlay bytes, runs an import preflight, seeds only the 256 audited health shards, and resumes
  target outputs in a separate Drive directory.
- **Verification:** the clean combined extraction now imports the Stage 2 executor successfully;
  `tests/test_stage2_colab_overlay.py` rebuilds the archive, checks recursive sibling imports, verifies
  all 256 health shards, and rejects any target shard/manifest in the overlay.
- **Boundary:** this is an execution-packaging repair only. No Stage 2 target outcome or mechanism label
  was read, and no frozen estimand, package, threshold, cohort, or analysis rule changed.

## 2026-07-19 — Stage 2 Colab runtime-reference closure repaired before target execution

- **Observed failure:** after the import repair, the target process stopped before model loading because
  `atlas_mechanism_validation_v1_merged.target_rows.jsonl.gz` was absent from the overlay. Zero of 4,019
  target shards existed; no Stage 2 forward outcome was computed.
- **Root cause:** overlay v2 tested Python sibling-import closure but not data paths dereferenced from the
  audited Stage 2 package at execution. The runner requires the frozen whole-validation target rows to
  enforce its whole-stage reference bridge; later validation/merge steps also require the corresponding
  health-row and pair tables.
- **Fix:** versioned overlay v3 resolves all `whole_validation_row_references` directly from the audited
  package and refuses to build unless every file matches its sealed size and SHA-256. The notebook now
  streams child stdout/stderr and persists `stage2_target_runtime.log` in Drive.
- **Verification:** `tests/test_stage2_colab_overlay.py` rebuilds the archive and checks every declared
  runtime reference byte-for-byte in addition to recursive sibling imports, the 256 health shards, and
  exclusion of Stage 2 target shards/manifests. Bundle SHA-256 is
  `883a7ed39e68afe09d5b37ab4630306531c1ec49160aef00092fad431d2aabc4`.
- **Boundary:** this is a second execution-packaging correction. It changes no candidate, endpoint,
  estimand, cohort, threshold, inference rule, or scientific claim.

## 2026-07-19 — A3 numerical recovery admitted the completed Stage 2 forwards without rewriting them

- **Trigger:** all 4,019 target shards were written, but the immutable target manifest failed exactly one
  gate: cross-stage whole-effect agreement was `2.9802322387695312e-05` nats against a `1e-6` tolerance.
  Every other runtime gate passed. Conditional QK/V/MLP families had not been merged, classified, or read.
- **Diagnosis:** Stage 1 and Stage 2 used different deterministic packet-rank namespaces, changing CUDA
  batch composition. The 32,152 reference discrepancies had median `1.4305e-06`, p90 `4.7684e-06`, p99
  `9.3579e-06`, and maximum `2.9802e-05`; none exceeded A2's already-frozen `1e-4` numerical-identity
  scale. This is float32 cross-batch drift, not an intervention or coordinate change.
- **Amendment:** A3 admits only the preserved one-gate failure, requires the independently reconstructed
  maximum to lie in `(1e-6, 1e-4]`, and rechecks every shard hash, endpoint, bridge, health, arithmetic,
  split, and provenance invariant. A second failure remains non-waivable. The original failed target
  manifest is untouched.
- **Portability correction:** the first A3 audit remains preserved as `validation_audit.a3.json` (FAIL).
  It found three absent producer-Mac absolute paths, not scientific failures. The corrected auditor may
  relocate a missing `{results,data,studies}/...` suffix inside the Colab repository only when the original
  SHA-256 matches. `validation_audit.a3_v2.json` passes with zero failures.
- **Prevention:** future Stage 2 executions reuse the whole-validation `mechanism-validation-order`
  namespace; a regression test pins it. This future code change does not alter the completed A3 artifacts.

## 2026-07-19 — Stage 2 conditional result closes the general-mechanism branch without function naming

- **Execution/inference:** the A3 audit, merge, 200,000-draw max-|t| analysis, and independent inference
  reconstruction all pass. The result contains four candidates, eight donor/acceptor endpoint families,
  32 family members, and 64 diagnostics. Twenty-nine members are material-support, three are
  detected-subfloor, none licenses a bounded null, and none carries a broad masked-MLM health flag.
- **A-head fan-out:** L2H15 has material candidate-minus-control MLP recovery at L2–L5 in donors and
  acceptors. L3H1 has material recovery at donor L3–L5 and acceptor L3/L5; acceptor L4 is detected but
  subfloor. L1H12 has material recovery at donor L1–L5 and acceptor L2/L3/L5; acceptor L1/L4 are
  detected but subfloor.
- **Exact group:** jointly zeroing the frozen A group `{L1H12,L2H15,L3H1}` produces +1.762 nats in donors
  and +0.874 in acceptors under simultaneous intervals. This establishes a strong group relation, not
  synergy or an additive decomposition; the nested incremental contrasts needed for serial-versus-parallel
  structure were not in the frozen family.
- **B write boundary:** L2MLP's partner-conditioned relation is material in donors (+0.344) and acceptors
  (+0.320). Its upstream attention path remains `upstream_path_instrument_limited`: the frozen validated
  B-attention set is empty. Importing L3H1 from A after seeing the result is prohibited.
- **Licensed language:** the three A heads participate in overlapping downstream MLP recovery paths, and
  L2MLP is a general partner-conditioned write. No population A QK/V label, unique serial circuit,
  L3H1→L2MLP bridge, binding statement, or biological head/MLP function is licensed.

## 2026-07-19 — Full representative handoff passes the export boundary

- **Bundle:** eight frozen representatives cover donor and acceptor views for A/L2H15, A/L3H1,
  A/L1H12, and B/L2MLP. Each includes the native sequence, matched negative sequence, annotations,
  six-layer activation×Jacobian screening graph, correctness gates, explicit hidden-remainder report,
  and a separate population-evidence overlay copied from the audited Stage 2 family.
- **A localization:** the three A-head representatives include matched-control QK and V/OV state maps
  labeled `representative_only_descriptive_no_population_label`; they do not become population QK/V
  estimates.
- **Audit:** `atlas_representative_dags_v1_a3_v2.audit.json` passes with zero failures,
  `n_representatives=8`, and `export_eligible=true`. Bundle SHA-256 is
  `fa9d974efa4ca1da36fadf23d9385efe566ff83dec48e5891493b29efdb5d241`.
- **Boundary:** this is an export candidate, not a feature-complete attribution graph. Screening and
  remainder edges remain candidate-generating; only the separate exact/MLP overlay carries audited
  causal evidence. Visualizer ingestion must preserve that separation and every provenance hash.

## 2026-07-19 — HCA21 visualizer transport closes fail-closed

- **Lossless handoff:** the study-side exporter accepts only the full non-smoke bundle, its matching
  source manifest, and a zero-failure independent source audit. All representative records are preserved
  exactly after JSON decoding; the transport adds no mechanism label or derived outcome.
- **Independent transport audit:** the exported evidence contains the exact eight relation/role cells,
  16 measured 1024-nt graphs, 2,458 `screening` routes, 992 `remainder` routes, and 37 separately displayed
  component/path population relations. The audit verifies graph acyclicity, root connectivity, coordinate
  alignment, two measured trace metrics, simultaneous intervals, case/control evidence labels, and
  explicit omitted mass with zero failures.
- **Fail-closed viewer:** the browser recomputes the evidence and manifest hashes, pins source bundle
  SHA-256 `fa9d974e…d241`, rejects a standalone Atlas JSON without its sidecars, and refuses smoke, stale,
  mutated, or evidence-upgraded payloads. The production build repeats the file/hash/audit checks.
- **Usability correction:** the default threshold for each measured graph/metric is the largest preset
  that still retains at least one original-nucleotide source. This prevents a clean-looking graph from
  silently ending before the genome; users can still tighten or loosen the display threshold.
- **Boundary:** HCA22 validates evidence transport, not a new model mechanism. The measured DAGs expose
  target log-probability and entropy only; synthetic `U_info`/`U_read` lenses are not retrofitted. Case
  population interventions remain visually separate from representative screening and matched-negative
  graphs.
