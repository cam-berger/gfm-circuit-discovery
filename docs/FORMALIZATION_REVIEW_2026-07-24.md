# Formalization review — publication readiness, second pass

> **Tier: T2 — provenance.** Dated review record; not current canonical guidance.

Repo: `gfm-circuit-discovery` · Reviewed at `a8aa70d` (worktree of `master`, tree clean)
Date: 2026-07-24 · Supersedes nothing; **complements** `ULTRAREVIEW_2026-07-24.md` by reviewing the tree
*after* that report's Tier-A/Tier-B items landed.
Priorities per owner brief: **readability · reproducibility · documentation · formalization**, plus
**aggregate nomenclature** and **remove irrelevant references**.

---

## 0. Provenance and confidence of this report — read first

**COMPLETE RUN (2026-07-25).** After two spend-limit interruptions the sweep finished: **382/382 agents,
0 errors**, 8.8M + 19.5M + 6.8M subagent tokens across three resumes, ~100 minutes of wall clock.

| | Planned | Completed |
|---|---|---|
| Ground-truth sweeps | 6 | **6** |
| Perspective/region reviewers | 17 | **17** |
| Prereg↔code formalization audits | 25 | **25** |
| Adversarial verification agents | ~374 | **374** |
| Dedup + editorial synthesis | 10 | **10** |

**Result: 374 findings → 329 survived verification, 45 refuted.**
Corrected severity: **5 critical · 38 high · 112 medium · 174 low.**
By category: formalization 99 · evidence-discipline 64 · reproducibility 63 · documentation 47 ·
references 22 · nomenclature 21 · structure 13 · readability 0.

Calibration you should apply when reading:

1. **Verifiers were told to default to REFUTED** and to confirm only what they personally reproduced at
   HEAD. The verdict split — **274 PARTIAL, 46 REFUTED, 20 CONFIRMED** across the first 340 — means the
   typical finding is *real but materially narrower* than the reporting agent framed it. Severities in
   this report are the **corrected** ones; the raw agent severities were roughly 4× more alarming.
2. **Some REFUTED verdicts are "fixed during this session," not "never real."** Tier A landed between
   the reviewers and the verifiers, so the R-number pointers, fig-1's 40–44%, the MANIFEST schema, the
   prereg-index rows, `config.py`, and the missing dependency come back refuted because they were fixed.
   That is the cache behaving correctly, not the agents contradicting themselves.
3. **Ten findings were hand-verified by direct measurement** during this session (§2) — independent of
   the agent verifiers. All ten confirmed, with numbers matching the reporting agent exactly except one
   (the Level-A head count, corrected 5→6 by the memorial written for it).

**Artifacts** (outside the repo, at `~/Research/projects/gfm-circuit-discovery-review-2026-07-24/`):
`findings_final_329.json`, `cat_formalization.md`, `cat_evidence-discipline.md`, `cat_reproducibility.md`,
`cat_documentation.md`, `cat_references.md`, `cat_nomenclature.md`, `cat_structure.md`,
`COMPLETENESS_CRITIC.md`, `EDITORIAL_DECISION_fragment.md`.

> **One synthesis output was lost.** The editorial-decision agent's return was captured as a ~4.4K tail
> fragment beginning mid-word; its appendix checklist and closing summary survive, its verdict section and
> tiered roadmap do not. §1 and §5 below are therefore **my** synthesis over the seven complete category
> reports, not that agent's. Everything else is verbatim agent output.

---

## 1. Decision

**Major Revision — not ready for external artifact review.** (Verdict unchanged after the complete run.
It was reached on five hand-verified criticals and a large unverified tail; the tail turned out to be
milder than filed — 174 of 329 survivors are `low` — but the completed run added **two new criticals**
(§2.9, §2.10) and closed the formalization question unfavourably, so the balance is the same.)

**The single highest-leverage deliverable is `docs/PREREG_CONFORMANCE.md`** — a clause-by-clause table of
what each frozen protocol specified, what the runner implemented, and whether any claim's wording depends
on the gap. It does not require new compute, it answers the reviewer's hardest question directly, and it
converts ~30 scattered undisclosed deviations into one disclosed, auditable surface. Ship that, batch the
R-numbers once, and this artifact stops asking a reviewer to take its discipline on trust.

**The failure mode to design against is propagation, not intent.** The calibration anchor: C37's missing
ledger row was logged by the prior review, marked "resolved by branch reconciliation", the reconciliation
merged — and `grep -c C37 CORRECTIONS.md` still returns **0**. Remediation here fails because the same
fact lives in four to eight hand-maintained surfaces and a fix lands in one. **After every edit, grep the
repo for the string you just changed and confirm no sibling copy still carries the old version.**

This is not a reversal of the prior verdict. The prior review examined *bookkeeping*, found it near-ready,
and was right. This pass examined *the load-bearing joins* — number-in-prose vs. number-in-artifact,
preregistered gate vs. executed gate, hash pin vs. current file, documented command vs. runnable command —
and those joins are where the defects are. Three of them would be found by a competent reviewer in the
first afternoon and would cost the project credibility disproportionate to their size:

- a **wrong headline number** carried in a T1 ledger and propagated to five documents (§2.1);
- a **frozen, hash-sealed evidence package that no longer verifies**, whose committed audit still reads
  `PASS`, broken by the repo's own tier-stamping pass (§2.3–2.4);
- **preregistered gates that are hardcoded to pass** while the artifact of record advertises them as
  satisfied (§3.4).

None of these is a wrong *result*. All are integrity-of-record defects, which in a project whose entire
selling point is evidence discipline are more damaging than an ordinary bug. The good news is that the
underlying science survives every one: the corrections are to wording, provenance, and enforcement, not
to findings.

---

## 2. Hand-verified findings (measured this session — treat as CONFIRMED)

### 2.1 `C25`/`C21`: "≈66% of coupling" is a unit error — the value is ≈48%  · CRITICAL

`CORRECTIONS.md:44` (C21) and `:48` (C25) both state **"L2H15 ≈66% of coupling"**. Measured from the
controlling artifact `results/splicebert_coupling/l2_accounting.json`:

```
baseline_coupling        = 1.3840273753801982
coupling_L2H15_ablated   = 0.7229255515336991
coupling_supplied_by_L2H15 = 0.6611018248399099   # = baseline - ablated, an ABSOLUTE DIFFERENCE
0.6611 / 1.3840          = 0.4777                 # the actual fraction
```

`0.661` is a margin difference in nats being read as a fraction. Independent corroboration:
`faithfulness.json → necessity.L2H15.frac_lost = 0.5061`. The correct statement is **+0.661 absolute,
≈48% of the conditional baseline** (0.51 by the faithfulness route).

This is the single most damaging finding in the report: it is a wrong number, in a T1 canonical ledger,
anchoring a named claim, propagated into `DECISIONS_A0.md` in at least four places.
**Fix must be a numbered correction (R29), not a silent edit** — the ledger and the frozen A3 prereg
record the old value.

### 2.2 Figure 1 of the publication set prints 40–44% where the data says 27–44%  · CRITICAL

`scripts/splicebert/make_fig1_summary.py:159` → `("Kept after carrier deleted", "40-44%")`, a literal.
Recomputed from the declared source `results/phase-lattice/fig_report_data.json`, the 8 `LEVEL_A` heads:

```
kept_carrier = [0.2695, 0.3377, 0.3673, 0.3888, 0.3904, 0.3955, 0.4349, 0.4397]   →  27-44%
```

**Six of the eight heads fall below 40%** (the reporting agent said five; the memorial test written
for this correction caught the miscount). The sibling figure has it right
(`fig4_carrier_circuit.CAPTION.md:38` → `27–44%`), as does `figures/splicebert/EVIDENCE.md:44`.
`fig1_summary.CAPTION.md:24` repeats the wrong value. The adjacent hard-coded rows (89–97%, 93–107%)
**do** match the artifact, so this is one retyping error, not a systematic offset — but Figure 1 and
Figure 4 currently disagree about the same eight heads.

### 2.3 The tier-stamping pass broke the C36 frozen package; its audit still says PASS  · CRITICAL

`results/phase-lattice/p3_phase_surgery_confirmation_package_v1.json` binds 43 files by SHA-256.
Re-running the project's own gate at HEAD:

```
MATCH 24 · MISSING 16 (6 model files, expected) · CHANGED 3
```

13 non-model failures — 10 relocated development artifacts, plus **3 content-hash changes**:

| File | Stored | Now |
|---|---|---|
| `preregistrations/PREREGISTRATION_P3_CLOSURE.md` | `81186a642a…` | `098ba72548…` |
| `redteam/REDTEAM_SYNTHESIS_P3_CLOSURE.md` | `17c0afeb8a…` | `5afdd87136…` |
| `scripts/splicebert/freeze_p3_phase_surgery_confirmation.py` | `763a04d142…` | `efdd624740…` |

The committed sibling audit still records `"status": "PASS"`. Cause: commit `c38e1ba`, whose own message
asserts *"read-only ledgers/preregs touched only by additive tier stamps"* — the additive stamp is exactly
what broke the seal. **An additive edit to a hash-sealed document is not additive.**

### 2.4 The same pass invalidated every Head–Context Atlas protocol pin  · CRITICAL

Four hash-pinned atlas protocol documents now carry a tier banner at line 3, and **not one stored pin
matches**:

| Document | Stored pin | Current hash |
|---|---|---|
| `ANALYSIS_PLAN_V1.md` | `21e696e3…` | `1687ef17…` |
| `ANALYSIS_AMENDMENT_A1.md` | `00008afa…` | `dd10e50c…` |
| `MECHANISM_VALIDATION_AMENDMENT_A1.md` | `9c60a92c…` | `a0e4e868…` |
| `MECHANISM_VALIDATION_AMENDMENT_A2.md` | `cf118b61…` | `968e3b42…` |

29 pin occurrences across the atlas result tree. Two live gates assert them —
`build_atlas_mechanism_package.py:295-298` raises `RuntimeError("analysis-plan hash drift")`, and
`validate_atlas_discovery_analysis.py` does the same — so **the atlas package is currently unbuildable**.
The suite stays green because no test exercises the real files.

### 2.5 Reproducing any claim by the documented command destroys the record  · CRITICAL

`README.md:152` — *"no new run may silently overwrite the scientific record."*
`REPRODUCE.md:81` — *"Write a new immutable artifact; do not overwrite the controlling result."*

Measured: **56 runners** default `--out` to a path listed as an artifact in `results/MANIFEST.csv`;
`grep -rc 'out).exists()' scripts/` → **0 files**; `src/gfmcd/provenance.py`'s write path has no
existence check. The two `--force` flags that exist gate the *preflight*, not the write. The stated
invariant is documentation only — nothing enforces it.

### 2.6 One of 93 canonical manifests has a recoverable code revision  · HIGH

`REPRODUCE.md:34-36` makes matching *"the … code revision of the controlling artifact"* the comparability
standard. Measured over the 93 manifests referenced by canonical `MANIFEST.csv` rows:

```
66  revision recorded but git_dirty = true
21  no revision field of any kind
 5  revision with no dirty flag
 1  CLEAN and recoverable  →  p3_phase_surgery_confirmation_v1/confirmation_analysis.manifest.json (C36)
```

Byte-exact code recovery is impossible for 92 of 93 canonical claims. This is not fixable by backfill —
it must be **disclosed** and enforced going forward.

### 2.9 The ≈10.298 nucleosomal comparator is measured in MOUSE, and no live document says so · CRITICAL

Found only by the completeness critic, after every region reviewer missed it: **nobody checked organism
provenance.** The ≈10.298 nt dyad-anchored periodicity — the project's single powered *biological*
result, and the instrument that stops a reader concluding "you found nucleosome positioning" — comes from:

```
scripts/splicebert/run_dyad_real.py:2   "Voong chemical-cleavage map, mouse ESC, mm9."
results/phase-lattice/dyad_real.json    "build": "mm9",  discovery [chr1,chr2] / test [chr3,chr4]   (MOUSE chromosomes)
data/NUCLEOSOME_MAP.md:12               Voong et al. 2016 — "…Mouse Embryonic Stem Cells…"
```

Measured: `git grep -lie 'mouse|mm9|mESC|Voong' -- '*.md'` returns **exactly one file**,
`data/NUCLEOSOME_MAP.md`. Zero hits in `README.md`, `MECHANISM.md`, `NOMENCLATURE.md`,
`BIOLOGICAL_BRIDGES.md`, `EVIDENCE_LADDER.md`, `RESULTS_VERIFICATION.md`, `CLAIMS.md` — or in either
prior review.

Two consequences, both in T1:

- **A wrong sentence.** `README.md:44` reads *"absent in human-only and in **the human genome's** ≈10.63
  spectrum; **genome's real period is nucleosomal ≈10.298 nt**"* — attributing a mouse measurement to the
  human genome inside one clause.
- **The governance instrument encodes the omission.** `NOMENCLATURE.md:27` defines the term as measured
  *"in DNA, not in the model"* with no species, and the mandated standing phrase at `:46-48` omits it too.

The refutation's force rests on precision — 0.33 nt apart, carrier fitted to ±0.005 — which silently
assumes the helical repeat transfers across species *and* cell type at better than 0.33 nt. **The
refutation may well survive** (helical periodicity is broadly conserved); what is unsupported is the
*unqualified* transfer. This is the highest-severity item in the report because it is the one place where
a live T1 statement is simply false, in the project's flagship biological claim.

### 2.10 C39's canonical artifact carries a hand-written verdict that overrides the runner · CRITICAL

`results/origin-siblings/canonical/sibling_carrier_bandtransfer.json` contains:

```
period3.healthy      = False
p_did = p_exon       = 0.0769          (> 0.05)
verdict              = "USED_coding_specific"    <-- hand-written
verdict_mechanical   = "VOID_unhealthy"          <-- what the runner emitted
verdict_note         = <350-word justification>
```

`run_sibling_carrier_bandtransfer.py:222` can only emit `VOID_unhealthy` (when `healthy` is false) or,
with the health gate corrected, `NOT_USED` (since `p=0.077 > 0.05`). It cannot emit the value that shipped.

**State this fairly: it is not concealed.** The `verdict_note` is transparent and substantively reasoned —
it argues the mechanical gate wrongly used `min(exon,intron)` headroom when exon headroom *is* the outcome
(the R13 error), gives the corrected non-outcome health channel (intron prediction retains 87.6%), reports
disjoint gene-grouped CIs (exon [0.153,0.188] vs intron [0.021,0.044]), and notes `p=0.077` is the 1/13
floor at `n_control=12`, not a weak effect. The science is probably right.

The defects are governance, and they are real:
1. The adjudication was written **into the immutable artifact**, overwriting the machine field, rather
   than into a sibling adjudication record. A re-run now disagrees with the committed file.
2. **No R-number logs it.** The repo's own rule is that a changed verdict is a numbered correction.
3. **Three surfaces disagree on status:** the sibling manifest says `"status": "EXPLORATORY"`,
   `MANIFEST.csv:756` says `canonical / Behaviour / positive`, and `CLAIMS.md:62` calls C39 *established*.

Same pattern at `a11_address_geometry.json` (machine `ABSOLUTE` preserved as `verdict_raw_script`) and
`superseded/a9_payload_semantics.json`. **Do not edit any of the three artifacts** — the remedy is one
owner-approved R-entry covering all three plus the status mismatch, and a `verdict_machine` +
sibling `.adjudication.json` convention going forward.

### 2.7 The correction ledger runs to R28; five documents say it stops at R22/R24  · HIGH

`CORRECTIONS.md` carries R25–R28 as `##` sections (lines 103, 138, 149, 162) while R1–R24 are table rows —
two formats in one file, which is why the pointers were never updated. Stale: `CLAUDE.md:54` ("R1–R22"),
`CLAUDE.md:204`, `README.md:28` (study), `CLAIMS.md:7`, `FINDINGS.md:6` (all "R1–R24"). Correct:
`REPRODUCIBILITY.md:7`, `scripts/README.md:11`.

Worse than the pointer drift: **R26 and R27 are cited by exactly one file each — their own definition.**
R27 records that A9b's primary DML nuisance was switched from the preregistered GBR to linear Ridge; the
live C30 ledger row discloses none of it. R28 records that C41's planted-positive gate was applied as a
point check with the planted-confound arm not run — while `CLAIMS.md:64` still labels C41 *confirmatory*.

### 2.8 Two hard dependencies are in neither `pyproject.toml` nor `uv.lock`  · HIGH

`maxentpy` — **11** importers, including the A9/C30 payload-semantics runner.
`multimolecule` — **3** importers, all three C40 cross-model runners.

`uv sync --extra splicebert --extra dev` produces an environment that cannot run these. The C30 and C40
families are not reproducible from the documented setup.

---

## 3. Root causes (the 269 findings collapse to seven)

Fixing the cause is worth more than fixing the symptoms. Ranked by consequence.

### RC1 — Additive edits to hash-sealed documents (§2.3, §2.4)
The tier-stamp program treated "additive" as "safe." For a document whose SHA-256 is pinned elsewhere,
it is not. Blast radius: C36's package + 29 atlas pins + 2 unbuildable gates + 2 committed audits now
lying. **Also note `docs/REVIEW_OPEN_QUESTIONS.md:18-25` asserts these stamps were reverted — HEAD does
not contain that revert.** The governance memo is wrong about its own repo.

### RC2 — Prose numbers drifted from artifact numbers (§2.1, §2.2)
No mechanism ties a number printed in a doc or a figure to the artifact it came from. `RESULTS_VERIFICATION.md`
claims "52/52 exact, no discrepancy" — two of its own rows are reported as falsifying that. Every
hard-coded number in a figure generator or a caption is an unguarded copy.

### RC3 — Preregistered gates that cannot fail, and verdicts redefined in code
The project's own R7/R21/R23 pathology, still live. Reported instances (**unverified**, from the 13
completed formalization audits):

| Family | Defect |
|---|---|
| A1 | `full_matches_A0` preflight hardcoded to PASS, no comparison, counted in the global AND |
| A9 | `maxent_pinned` PASS is literal `True`; sha256 is a hardcoded string never computed |
| A9 | V4 composition arm mathematically void — caliper unreachable given L1-normalized inputs |
| A9b | `self_interchange_identity` is an algebraic identity; A9's fireable positive control dropped |
| A9b | planted-positive gate computed on GBR, not the Ridge estimator that produced the reported β |
| A0 | preregistered stage-localization permutation null never computed (`empirical_upper_p_value` imported, never called) |
| G1 (C41) | frozen M6 stopping rule not implemented; runner hardcodes its own tier as "confirmatory" |
| C37 | a memorial inside the frozen Global-PASS AND-gate is hardcoded `True` and written to the artifact |

If these hold, **the artifacts of record advertise gates that were never enforced.** This is the finding
class most likely to be raised by a hostile reviewer, and the 12 unaudited families likely contain more.

### RC4 — The ledger outran the narrative
C37–C41 and R25–R28 exist in `CORRECTIONS.md`/`MANIFEST` but the T1 narrative still describes the
pre-C38 world. `MECHANISM.md` presents cross-example transduction as future work (C38/C41 answered it);
`docs/ROADMAP.md:60` calls the flagship question untested; `studies/splicebert/README.md` and the study
`README.md` still list settled questions as open; `EVIDENCE_LADDER.md:36` marks the ≈10.63 carrier's
OV and Behaviour cells `?` (defined by that same document as *"intervention not run"*) although both ran;
`CLAUDE.md:148` still asserts carrier origin is unanswerable on a frozen checkpoint — the exact
overstatement R25 corrected. `project_status_dashboard.html` omits C39/C40/C41 and R25–R28 entirely.

### RC5 — The reproduction path is not executable by a stranger
Beyond §2.5/2.6/2.8: no acquisition path for the model or genome/annotation anywhere in the tree; the
canonical command `uv run pytest` drops the extras (the repo's own CI file documents this); `data/` is
undocumented and its two in-tree specs disagree (`configs/…/base.json` says
`data/external/gencode/v50/GRCh38.primary_assembly.genome.fa`, while 62 manifests record
`/Users/cberg/Research/data/genomes/hg38.fa`); `sealed_test.manifest.json` — provenance for the headline
+0.87 result — records **no checkpoint hash, no data hash, no seed, no revision**; 13 canonical artifacts
have no manifest at all; ~32 memorial tests are permanently inert off-machine.

### RC6 — Uncontrolled vocabulary in the controlled-vocabulary docs
Directly on your "aggregate nomenclature" ask. The *retired-term* discipline is genuinely well enforced —
do not mass-scrub. The failure is elsewhere:
- **"specificity"** names ~6 different estimands across live T1 prose, one of them a statistical FPR.
- **"CLOSED"** is defined incompatibly by the two T1 vocabularies (tested-and-failed in `NOMENCLATURE.md`;
  no-further-evidence-expected in `ONTOLOGY.md`).
- **"circuit"** is defined in `NOMENCLATURE.md` by a property (necessary *and* sufficient-when-injected)
  that is false for two of the three objects the repo calls circuits.
- The **evidence-level column** of both ledgers uses ~38 free-form strings where the ladder defines 6 rungs.
- Units: C36's Ψ_C is labelled "nats" in `BIOLOGICAL_BRIDGES.md`; `NOMENCLATURE.md` names it a logit-difference.

### RC7 — Residue that misleads a reader about what is live
`src/gfmcd/splicing/rivals.py` — 972 lines of retracted v1/v2 scaffolding, no banner, zero importers, in
the active tree while a T1 doc says such code lives only in Git history. `src/gfmcd/config.py` names
`yangheng/SpliceBERT-510nt` — a mirror of a *different* checkpoint this repo never loads. Superseded
runners without supersession banners. `handoff/verification_state.json`, untiered at repo root, with
evidence assertions the ledger contradicts. Two 2026-07-24 method docs prescribe "the DNABERT2 port" as
the next transfer milestone, contradicting the T1 roadmap (Caduceus) — and one asserts the Caduceus study
is *running* when its own banner says deferred.

---

## 4. Answers to the four named questions

**READABILITY — blocked, and measurably so.** One sweep reports only **86 of 166** tracked documents are
reachable from `README.md` by any chain of links, and that the complete evidence chain for **C41, the
newest established claim, is in the unreachable half**. The tier system is sound; the navigation is not.
A reviewer with 90 minutes will land on documents that describe a superseded state (RC4) with nothing
telling them so.

**REPRODUCIBILITY — no.** A stranger cannot obtain the model or the genome (no acquisition path), cannot
build a working environment (two undeclared dependencies), and if they succeed anyway, the documented
command overwrites the record they were comparing against. See RC5.

**DOCUMENTATION — complete but not current, and ambiguous about authority.** The corpus is unusually
thorough; the defects are staleness (RC4) and unresolved authority conflicts — `README.md` files
`CORRECTIONS.md` as T2 while the file self-stamps T1 and the same README makes it the tie-breaker over T2;
`EVIDENCE_LADDER.md` (T1) delegates authority over values to a T2 document.

**FORMALIZATION — now fully measured, and it is the real gap. The answer is: no, not consistently.**

All 25 families audited. 99 surviving findings, the largest category in the report. The headline number:

> **The stated methods and the implementation diverge in roughly thirty preregistered clauses, exactly
> one of which (R28) has ever been logged.**

The root cause is structural, not sloppiness: **the decision layer of every experiment lives in the
runner, not in the protocol, and nothing checks the two against each other.** There is *no prereg→code
conformance artifact anywhere in the repo*. Concretely, five recurring shapes:

1. **Unpreregistered operational constants.** Where a frozen prereg states a rule in words — "above the
   95th-percentile null", "≈ 0", "significant", "systematic/monotonic" — the runner supplies a number
   that was never preregistered. Once this changes a published boolean: `run_a6_circuit_faithfulness.py:320`
   applies a floor `max(add_p95, 0.05)` where the prereg says the 95th percentile; with
   `addition_null_p95=0.0022`, **five nodes exceed the preregistered threshold** yet
   `completeness_incomplete_nodes=[]` and `verdict.complete=true`.
2. **Preregistered arms never executed and nowhere logged** (~30 instances). Sharpest: B1's OV arm
   substituted a coupling-margin value-main for the frozen `U_info`/`S_ov` endpoint — zero hits for
   `U_info` in the driver — while live prose says "no carrier-specific write onto the donor readout".
   `DONOR_ANCHOR` §4.1's "≥30 nt from *any* annotated splice site" screen checks only donor and mate;
   **69 of 250 validation controls violate it**. A10's sufficiency confirm never ran. G1's mandatory
   balance table is absent.
3. **Preflight gates that cannot fail.** `checkpoint_recorded {"PASS": True}` appears in **16** runners;
   A1's `full_matches_A0` is literal `True`; A5's `disjoint` is `X∩(Y−X)=∅`; P3's `carrier_reconstruction`
   is `(a−b)+b−a`. Live prose then credits "preflight 8/8", "9/9", "eleven instrument gates".
4. **Hand-written verdicts in committed results** — three artifacts (§2.10).
5. **The method docs describe the v1 pipeline.** `METHODOLOGY.md:128-133` still calls corruption a
   "one-base edit" citing a module with zero callers (live corruption is the 2-base AG→CC in
   `circuit.py:66`), and names the wrong estimand. And the inference vocabulary that carries every null —
   SESOI, equivalence, TOST, dyadic, max-T — appears **zero times** across all four method-level docs,
   while the ledger reports results in exactly that vocabulary.

**Nothing here indicates a wrong scientific result.** C39's substantive numbers survive its broken gate;
every P3 SESOI comparison is far from its threshold. What is at risk is the *legibility of the
discipline* — precisely the thing this artifact exists to demonstrate.

---

## 5. Staged plan

**Guardrails, non-negotiable.** Do not edit frozen preregistrations, claim/correction ledgers,
`.manifest.json` files, or committed figures of record in place. Where one is wrong, the fix is a
**numbered correction (R-number) plus an additive amendment record**. And per §2.3/§2.4: **an additive
edit to a hash-pinned document is a breaking edit** — check for stored pins before touching any file.

### Phase 0 — the false statements — **COMPLETE 2026-07-25**

Seven items verified, each by an independent agent plus an adversarial cross-check, then applied serially
by hand with a propagation grep after every edit. Suite **294 → 311 passed / 46 skipped / 0 failed**.
Deny-list diff empty throughout: no frozen preregistration, decision log, `.manifest.json`, result JSON, or
figure raster was touched. `CLAIMS.md` / `CORRECTIONS.md` were edited **only** through numbered corrections.

| Item | Outcome |
|---|---|
| `METHODOLOGY.md` corruption / metric / steering clauses | Fixed — **no R-number**; corrects a description of live method, changes no recorded value |
| C25/C21 "≈66% of coupling" | **R29** — unit error; +0.661 nats ≈ **47.8%** of the conditional baseline |
| ≈10.298 species + "absent" over-claim | **R30** — mouse mm9 (Voong 2016); "absent" → "not detectable above the 5–10% floor" |
| Ladder `?` cells + B1's OV endpoint | **R31** — merged; the same evidence mis-stated in opposite directions |
| C41 tier | **R32** — verification tier, not confirmatory |
| C39 verdict + status trilemma | **R33** — verification level; hand-written adjudication disclosed |
| A6 completeness gate | **DROPPED** — cross-check returned ACTUALLY_CORRECT (see below) |

**R31 is the scientifically load-bearing one.** `EVIDENCE_LADDER.md:36` marked the ≈10.63 carrier's
OV/Behaviour cells `?`, which that document's own legend defines as *"the defining intervention has not been
run"*, and `figures/splicebert/EVIDENCE.md:45` said so in prose — while `CLAIMS.md` C12 simultaneously
claimed the carrier *"does not write, at the donor readout or generally."* Both wrong, oppositely: the arms
ran (C37), but the **preregistered OV endpoint was never computed** — `run_carrier_ov_av.py` has zero hits
for `U_info`, `S_ov`, or `suff_recovery`. What was measured is a coupling-margin value main, sub-SESOI at
deep readers with its sensitivity calibration deferred. C12 is now scoped to that, the B1 **A-mixed branch
is recorded as unexcluded**, and the ladder gained a **`bounded`** marker so a measured boundary is no longer
recorded as an absence.

**A6 was dropped on cross-check, and the reasoning is worth keeping.** The runner really does apply an
unpreregistered floor `max(add_p95, 0.05)` — 22.7× the stored band — forcing `verdict.complete=true` while
five nodes exceed the preregistered threshold. But **C27's prose cites the magnitude, not the boolean**, and
the artifact's own REFLECTION states the same scope correction, so prose and artifact agree: this is a
runner-vs-prereg conformance defect (Phase 1), not a false statement. It is also threshold-independent —
max add-one gain is 0.498% of the residual, and the best single addition moves `F` from 0.0054 to 0.0104,
about 1% of the way to the preregistered `F ≥ 0.80` bar.

**The propagation grep caught a site all eleven agents missed:**
`docs/CARRIER_ORIGIN_SYNTHESIS_2026-07-23.md:56`, T1 canonical, carrying the same "absent" over-claim.
Every R-entry now names both its corrected sites and the ones deliberately left as dated record.

### Phase 2 — enforcement — **COMPLETE 2026-07-25**

Three guards, 22 tests, all mutation-verified. Every convention below was previously stated in prose and
enforced by nothing.

| Guard | What it does | Ledger |
|---|---|---|
| `tests/test_hash_pin_ledger.py` (5) | Recomputes the C36 package's 43 bound records and the 4 Atlas protocol pins. Asserts the broken set is **exactly** `tests/data/hash_pin_ledger.json` (17 entries, generated by measurement) — new drift fails, **and a repaired pin fails**, forcing the ledger to be pruned. | **R34** |
| `tests/test_ledger_consistency.py` (6) | Every cited `C`/`R` ID resolves; no document understates the correction range; MANIFEST claim IDs resolve; polarity vocabulary is closed. Kills the C37/R28 propagation class. | — |
| `tests/test_overwrite_guard.py` (11) | `guard_output_path` + `ExperimentManifest.write` refuse to clobber (keyword-only opt-in, no env escape hatch). The 56 runners that default `--out` at the record are **inventoried, not edited**, and the inventory cannot grow. | **R35** |

**The ledger test immediately caught its author.** Adding R29–R34 made nine `R1–R28` / `R1–R24` pointers
stale across six files — two of which this same session had set during Tier A. That is the propagation
failure mode, detected mechanically within minutes of the guard existing.

**Judgment call recorded:** the overwrite guard was applied to the library, **not** to the 56 runners. Their
producing commits are pinned by the manifests beside their results; inserting a guard call would
desynchronise a pinned provenance chain to fix a documentation defect. See R35.

### Tier A — mechanical, safe, parallelizable — **EXECUTED 2026-07-24**

Suite before: **294 passed / 46 skipped / 0 failed**. After: **300 / 46 / 0** (+6 = the new fig-1
memorial). `uv sync --locked` (CI's install path) verified passing; `uv lock --check` consistent.
A post-edit pin sweep over the exact changed set confirmed **no changed file is bound by any stored
SHA-256 pin**, and no frozen preregistration, claim/correction ledger, `.manifest.json`, or figure
raster was touched.

| # | Item | Status | Files |
|---|---|---|---|
| A1 | R1–R24 → R1–R28 pointers, plus a one-line gloss of what R25–R28 actually say | **done** (`CLAIMS.md:7` **deferred** — deny-listed ledger, needs a numbered correction) | `CLAUDE.md:54,204`, study `README.md`, `FINDINGS.md` |
| A2 | fig1 retention ranges now derived from the artifact; caption corrected to 27–44%; memorial added | **done** | `make_fig1_summary.py`, `fig1_summary.CAPTION.md`, `tests/test_fig1_summary_ranges.py` |
| A3 | `maxentpy` declared as a `maxent` extra pinned to commit `b5e69aad`; relocked (2 additions, no version bumps) | **done** — `multimolecule` deliberately **not** added: it is a separate `gfmcd-mm` env by design | `pyproject.toml`, `uv.lock` |
| A4 | `REPRODUCE.md` §1.1 (the two undeclared deps), §2.1 (data layout + the conflicting specs), §2.2 (code-revision + overwrite limits); extras-on-`uv run` fix | **done** | `REPRODUCE.md` |
| A5 | Repository map extended with `scripts/crossmodel/`, `results/origin-*`, `tests/`, `configs/`, `data/`, `handoff/`, `logs/` | **done** | `README.md` |
| A6 | MANIFEST schema 7 → 8 columns; `polarity` documented with its vocabulary and counts | **done** | `results/RESULTS_INDEX.md` |
| A7 | G1 (→C41, frozen) and EXPLORATORY_P3_SHIFT_DIAGNOSTIC_V1 rows added; B1v2 and B3 corrected to executed, pre-run status retained | **done** | `preregistrations/README.md` |
| A8 | Banners on `rivals.py`, `partner_readout.py`, `run_a4_address_geometry.py` (superseded by A11) and a scope note on `run_faithfulness.py` | **done** — `run_faithfulness.py` is **not** superseded: it controls canonical C7, so it got a disambiguation note, not a supersession banner | `src/gfmcd/splicing/`, `scripts/splicebert/` |
| A9 | `config.py` reference id corrected from the 510nt community mirror to the official repo, with a note that the measurement path uses the local 1024nt archive | **done** | `src/gfmcd/config.py` |

**Corrected during execution:** the reporting agent's "five of eight Level-A heads below 40%" is wrong —
it is **six of eight**. The memorial written for A2 caught it. `README.md`'s "over 160 markdown documents"
was checked and is accurate (166 tracked), so no change was needed there.

**Not done, deliberately:** `fig1_summary.svg`/`.png` still render 40–44%. They are figures of record
cited by `results/MANIFEST.csv`; regenerating them overwrites committed rasters and their provenance,
which is the failure mode reverted as BLOCKER-2 in `REVIEW_OPEN_QUESTIONS.md`. The caption now states
this explicitly. Regeneration is an owner-approved step.

### Tier B — judgment, owner sign-off (1–2 days)
| # | Item |
|---|---|
| B1 | **R29: the C25/C21 unit error.** Numbered correction restating +0.661 absolute / ≈48% conditional; propagate to `DECISIONS_A0.md`; leave the frozen prereg untouched and reference the R-number. **Also fold in `CLAIMS.md:7`'s R1–R24 → R1–R28 pointer**, deferred from Tier A because the ledger is deny-listed |
| B2 | **Hash-seal reconciliation. Owner decision taken 2026-07-24: LOG AN AMENDMENT, do not revert the banners.** Add an amendment record giving the pre-stamp and post-stamp digest of each of the 7 affected files (4 atlas protocols + `PREREGISTRATION_P3_CLOSURE.md`, `REDTEAM_SYNTHESIS_P3_CLOSURE.md`, `freeze_p3_phase_surgery_confirmation.py`), stating that `git show c38e1ba` proves the diffs are banner-only and no scientific content changed. Re-run `audit_p3_phase_surgery_confirmation_package.py` and the atlas validator; commit the new audits *alongside* — never overwriting — the 2026-07-21 PASS audits. Add a test that recomputes both pin sets so the drift fails CI instead of sitting undetected |
| B3 | **Correct `REVIEW_OPEN_QUESTIONS.md`** — it asserts reverts HEAD does not contain |
| B4 | **RC4 currency sweep:** bring `MECHANISM.md`, `ROADMAP.md`, `CLAUDE.md:148`, `EVIDENCE_LADDER.md:36`, both study READMEs, `figures/README.md`, and the dashboard up to C38–C41/R25–R28 |
| B5 | **C41 tier reconciliation** — `CLAIMS.md:64` says confirmatory; R28 says verification tier. One of them must move |
| B6 | **Disclose the code-revision reality** (§2.6) in `REPRODUCE.md`; add a `git_dirty` gate to `verify_manifest.py` for *new* canonical rows |
| B7 | **Overwrite guard** in `src/gfmcd/provenance.py` — refuse to write an existing path without `--overwrite-record`; all 56 runners inherit it |

### Tier C — formalization completion (the item that actually gates submission)
| # | Item |
|---|---|
| C1 | **Finish the 12 unaudited families** (P3-role, P3-closure, IC1, IC2, B1, B2/B3/G1, A12, DONOR, SEALED-COUPLING, PERIOD3, ATLAS-B4/B5/C, CADUCEUS). Until this runs, prereg↔code conformance is unknown for half the record |
| C2 | **Adjudicate every RC3 gate.** Each is either *fix the code* or *log an amendment*. Do not leave a third state |
| C3 | **Nomenclature aggregation (RC6):** disambiguate "specificity"; reconcile "CLOSED"; repair the "circuit" definition; put the evidence-level column on a controlled 6-value enum; fix the Ψ_C units |
| C4 | **Navigability:** make every claim's evidence chain reachable from `README.md`; C41 first |
| C5 | **Test integrity:** pin headline numbers per claim; make the ~32 inert memorials run via `GFMCD_GENOME_DIR`; close the C41/C12 polarity gap |

---

## 6. What this review did *not* find (trust calibration)

- **No wrong scientific result.** Every defect is wording, provenance, enforcement, or currency.
- **Retired-term discipline holds.** The nomenclature problem is homonyms and missing definitions,
  not leakage of retired terms. **Do not mass-scrub.**
- **The test suite is genuinely green** — 294 passed / 46 skipped / 0 failed at `a8aa70d`. (Note
  `ULTRAREVIEW_2026-07-24.md:30` records a different baseline, 305/35; the two should be reconciled.)
- **The working tree is clean** and the branch-drift hazard from the prior review is resolved.
- **`MANIFEST` path integrity holds** — the prior review's 0-dangling result still stands.
