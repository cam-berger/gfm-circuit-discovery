# B5 — The 20 context signals (HCA18): power, not method — 5-lens design red-team synthesis (2026-07-22)

**Reviews:** `PREREGISTRATION_B5_CONTEXT_SIGNAL_POWER.md` v1 (DRAFT for pre-compute red-team). Five independent
adversarial lenses, run BEFORE any compute (the branch schedules **zero model forward passes** — pure
arithmetic over committed JSON). **Outcome: the design is NOT freezable as written, but it is cheaply
fixable, and its terminal verdict is robust.** No lens returned `fatal`; four returned `fixable` and one
`sound`. The convergent problem is a single **ill-posed pooled-taxonomy escape hatch** that could license an
invalid disposition, plus an internal §5/§6 contradiction on the one marginal signal and a mis-cited census.
Once the pooled branch is removed and the verdict is decided on the fine regime only, the licensed answer is
**CLOSE-AS-UNDERPOWERED** — every one of the 20 signals demands more effective matched pairs than exist even
at the hard one-case-per-gene supply cap.

| Lens | Verdict | Headline |
|---|---|---|
| Identifiability | fixable | Pooled E2 branch is **definitionally ill-posed** (sign-heterogeneous, cross-role, gene-double-counting cohort; reuses a fine signal's se); fine regime CLOSE is robust |
| Power / inference | fixable | Frozen rule is **self-contradictory on acceptor L2H4** (satisfies CLOSE *and* INCONCLUSIVE); N_req is a lower bound, not "conservative"; pooled branch statistically invalid |
| Adjustment-set completeness | fixable | Prereg cites the **WRONG census** as its supply wall (competition-cohort `census_v1.json` 444 genes, not the atlas `atlas_context_census_v1.json` 29% window-fit); N_req is a one-sided-falsifiable lower bound |
| Calibration / intent | fixable | **Dead planted-positive** — PP1/PP2 are already-powered, so the ceiling-comparison code path (the branch's actual product) is never exercised; unit mismatch decision-relevant for the tightest cell |
| Feasibility / power / MPS | sound | GO — every input verified present locally, zero forward passes, MPS wall is the COMPUTED object of study, not a constraint on B5 |

---

## The convergent MAJOR (independently verified against code + data): the pooled-taxonomy branch is ill-posed

Three lenses (Identifiability, Power, Calibration) converge on the same defect, and it is the one place the
frozen decision rule could license a scientifically invalid disposition.

**What the prereg does (§1 E2 pooled, §6 POOLED-ONLY REDESIGN, L96–99/211–213/251–253/262–263):** for a
FINE signal it computes `reachable_pooled := N_req(signal) ≤ 6,487`, using the fine signal's own
`se`/`mde80`, and routes any signal reachable only under this bound to a "POOLED-ONLY REDESIGN." The
pre-registered expectation (L262–263) names cassette L4MLP (`N_req 4,897 < 6,487`) as the nearest-reachable.

**Why it is invalid (verified):**
- The pooled ceiling `6,487` is `selected.atlas_discovery.n_pairs` — the **sum of matched pairs across all 10
  contrasts** (verified: `atlas_context_matched_v7.summary.json` `selected.atlas_discovery = {n_genes:5089,
  n_pairs:6487}`), spanning **both acceptor and donor roles** and **double-counting genes** that appear in
  multiple contrasts. It is not a coherent one-per-gene donor-side supply.
- Those pairs measure a **sign-heterogeneous** mix: 17 material_suppress (`mean_difference` negative) + 3
  material_support (cassette, positive) — verified `classification Counter{material_suppress:17,
  material_support:3}`. Pooling to a "donor-side" cell merges the
  +support cassette signals with the −suppress donor signals; opposite signs partially cancel, so the pooled
  mean does **not** estimate the cassette effect.
- Applying the **fine cassette signal's** `mde80 = 0.1399` to a 6,487-pair cohort that measures a *different,
  partly-cancelling* effect is a definitional/unit entanglement: the pooled effect's gene-dyadic `se` is
  **UNMEASURED**, and **R22 forbids re-drawing it here** (that would change the very numbers under
  adjudication). Pooling buys pairs by *dropping* the exact-other-tag constraint — precisely the specificity
  HCA18 forbids erasing — so a pooled "reachable" verdict cannot recover the fine cassette specificity being
  validated.

**Consequence:** as written, cassette L4MLP (`N_req 4,897 < 6,487`) satisfies the pooled-reachability
condition and would be routed to a POOLED-ONLY REDESIGN — a "worth-powering" disposition for a design that in
principle cannot power a cassette-*specific* signal. That is a scientifically hollow REDESIGN standing in for
the honest CLOSE/DEFER. **This is the R23/R15-class trap in arithmetic form:** the estimand silently changes
under the pooling operation, and the decision rule does not notice.

**Fix (must-fix #1):** delete the pooled-reachability arithmetic that reuses a fine signal's `se`. Adjudicate
on the **fine regime only** (where CLOSE is robust). Mark the pooled regime **NON-IDENTIFIABLE-AT-DESIGN**:
coarse pooled reachability requires a *fresh* measurement of the pooled effect's gene-dyadic `se`, which B5
cannot supply and R22 forbids re-drawing. Route the 3 cassette signals to CLOSE (or explicit DEFER), never to
a pooled redesign.

---

## Additional MAJORs (fixable)

### MAJOR-A — the frozen rule is self-contradictory on the one marginal signal (Power, Feasibility lenses)
The best-powered signal, **acceptor L2H4** (`mde80 0.0901`, `n_eff 621.6`), has `N_req = 2,019`
(verified). Against the fine 0.80-yield ceiling `1,982×0.80 = 1,586` it is **1.27×** over → inside the §5
marginal band `[1.0×,1.5×]` → §5 says **NON-IDENTIFIABLE / defer to achieved precision**. But §6's CLOSE
condition (`N_req > N_ceil for every signal`) and the §6 pre-registered expectation ("best case … 1.27× over
the ceiling … pointing to CLOSE") **both fire on the same numbers**. The four §6 verdict clauses are not
mutually exclusive and no precedence is given, so the crux signal satisfies CLOSE and INCONCLUSIVE
simultaneously. A "frozen decision rule" that resolves the crux two ways is not frozen.

**Fix (must-fix #2):** add explicit precedence AND lead the supply story with the **hard one-case-per-gene
gene cap** (verified below), not the 0.80-yield estimate. Against the hard cap `1,982` (yield = 1.0, the
absolute upper bound), acceptor L2H4 is `2,019/1,982 = 1.019×` over — CLOSE holds, but by a thin, honestly
stated **1.9% margin**. Restate the frozen expectation as "CLOSE for all 20 at the hard gene cap (donor
3.8×, cassette 1.83×, acceptor 1.02×); the acceptor cell's margin is thin and unit-matching only widens it."

### MAJOR-B — the prereg cites the WRONG census as its load-bearing supply wall (Adjustment-set lens)
§0/§1(L110–111)/§5/§7(L277–279) invoke "HCA1's ~10% window-fit ceiling / 3,014 usable sealed exons / 444
genes" from `census_v1.json`. **Verified: that artifact is the intron-mate splice-COMPETITION cohort's
sealed-test split** (`census_v1.json` estimand = "C — intron-mate splice competition (alt-5'SS donor +
alt-3'SS acceptor)"; `BOTH|test`: `n_genes_usable=444`, `n_usable=3014`, `n_targets=29030`). The 20 signals
live in the **wider atlas cohort** (`acceptor_cluster`/`donor_cluster`/`cassette_exon`), which has its **own
census** `atlas_context_census_v1.json`: **`junction_window_fit_fraction = 0.293`** (~29%, not 10%),
acceptor_cluster `n_genes ≈ 10,901`. The 444 figure is doubly wrong (wrong cohort AND smallest split). The
verdict survives — the real binding constraint is the exact-other-tag case-gene support pool, which the fine
ceiling already uses — but a frozen prereg must not cite a false, irrelevant supply constraint.

**Fix (must-fix #3):** delete all `census_v1.json` / 10% / 3,014 / 444-gene language; re-anchor on
`atlas_context_census_v1.json` (29% atlas window-fit, already baked into the support pool) and state the
exact-other-tag support pool is the true wall.

### MAJOR-C — dead planted-positive: the sensitivity control never travels the branch's product channel (Calibration lens)
The branch's non-trivial output is the ceiling comparison `reachable := N_req ≤ N_ceiling`. But PP1
(`mde80=0.04`) and PP2 (`mde80=0.05`) are **already-powered**: both give `N_req ≤ n_eff (622) < every ceiling`.
For PP1/PP2 the reachable verdict is emitted via the already-powered shortcut, and the **ceiling-comparison
code path is never exercised** in the reachable-but-not-already-powered regime. A bug in the ceiling
comparison (wrong regime, wrong support column, off-by-yield) passes PP1/PP2/N1/N2/N3 undetected. **This is
the P3.4 "dead planted-positive" archetype in arithmetic form** (the sensitivity control does not travel the
channel whose output is the branch's product).

**Fix (must-fix #4):** add **PP3** with `se` chosen so `N_req ∈ (n_eff, N_ceil)` — the pipeline must emit
reachable **via the ceiling comparison** — and **PP4** with `N_req` just above `N_ceil` — must emit NOT
reachable. Keep PP1/PP2 as already-powered boundary checks but do not treat them as the reachability
sensitivity control.

### MAJOR-D — unit mismatch: effective-pair demand vs raw-gene supply (Identifiability, Power, Calibration, Feasibility — 4 lenses)
`N_req` is in Kish **EFFECTIVE** pairs (`n_effective_pairs 621.6 < n_pairs 640`); the E2 ceiling is **RAW**
matched pairs (gene pool × yield). Genes ≠ packets ≠ effective pairs. The gene-vs-packet choice is
decision-relevant: if the samplable unit were packets (`n_case_packets_in_exact_other-tag_support = 3,015`),
the acceptor ceiling would be `3,015×0.80 = 2,412 > N_req 2,019` and acceptor would flip to reachable-marginal.
**Verified refutation:** the cohort sampler is strictly one-case-per-gene — `select_one_per_gene`
(`build_atlas_matched_cohorts.py:203–213`, called `:254`) skips any gene already seen — so the **gene pool
(1,982), not the packet pool (3,015), is the binding constraint**. The gene currency is correct; the prereg
just never says why, and never unit-matches the two sides.

**Fix (must-fix #5):** justify GENES as the supply currency (gene-dyadic SE ⇒ independent genes bound
precision) and express demand and supply in one unit — deflate the ceiling by the ~0.97 effective/raw factor,
or state `N_req` in raw pairs. Direction is monotone: unit-matching (effective ceiling < raw ceiling)
**strengthens CLOSE**.

---

## Minors (fixable, mostly wording / memorial hygiene)

- **M-count narrative error:** §"What HCA18 established" (L36–37) says "17 of 20 are mlp_write." Verified
  artifact: **16 mlp_write + 4 attention_head** rows (L2H4, L2H15, and L3H1 under two estimands = 3 distinct
  heads, L3H1 twice). Correct to 16/4 and add the component_type split to M1 so the memorial can fail on it.
- **Rounded-constant trap (M3):** stored `mde80` used `float(norm.ppf(.975)+norm.ppf(.80)) = 2.8015852…`; a
  script hard-coding the displayed `2.8016` round-trips to only ~1e-5 (|Δ|=1.48e-5), **failing M3's <1e-9
  tolerance**. State in §2 that the net-new script must reuse `float(norm.ppf(...))` verbatim. (M3 correctly
  catches this — it is a latent implementation trap, not a logic error.)
- **M4/M7 are FORCED, relabel per R7/R21:** M4 (`mde80·sqrt(n_eff/N_req)=0.05000`) is an exact algebraic
  identity of the `N_req` definition — fully forced, a code-duplication check, not a physics check (prereg
  labels it "partly forced"). M7 (permute `mean_difference` → `N_req` unchanged) is near-forced (the demand
  formula never reads `mean_difference`). The genuine failable on-path anchor is **M3**; lean memorial weight
  there.
- **N_req is a LOWER BOUND, not "conservative demand" (R22):** (a) the observed `se` is **selected** — the 20
  signals passed `q_bh ≤ 0.05` AND cross-half stability (`classify()`, `analyze_atlas_discovery.py:159–165`),
  a winner's-curse on precision → observed `mde80` optimistically small → `N_req` biased low; (b) `N_req`
  scales as observed `se²`, a single-draw point estimate with ~50% chance the true `se` is smaller. So
  reachability is **one-sided-falsifiable**: the arithmetic can only license CLOSE (rule OUT), never a
  "reachable"/redesign. State this in §5 and bar any reachable/redesign disposition on a single selected
  discovery `se`. (This also independently kills any residual temptation toward the pooled redesign.)
- **Gene-clustered vs independent-pairs scaling:** the back-out uses `mde80 ∝ n_effective^(−1/2)`, but the SE
  is a two-way gene-dyadic sandwich whose floor is set by independent GENES. As the case pool scales toward
  the fixed gene pool with `max_control_reuse=2`, the design effect rises → true `mde80` falls *slower* than
  n^(−1/2) → true `N_req` exceeds the computed value. Again strengthens CLOSE, weakens any reachable claim.
  Add a memorial reporting the current design effect (`n_pairs/n_effective`) and endpoint-gene count so the
  extrapolation regime is visible.

---

## Feasibility & MPS (lens 5) — GO, genuinely the tractable branch

- **Zero model forward passes.** B5 is deterministic arithmetic over four committed JSONs, runs in seconds on
  CPU. The MPS/0.02-nat power wall that stalled P3.4 is **the computed object of study here, not a constraint
  on B5** — the opposite of the P3.4 fatal.
- **Every input verified present locally with the cited values:**
  `atlas_discovery_analysis_v1.context.jsonl` (exactly 20 `context_associated==true` rows; `mde80` spread
  0.0901→0.2992; `tau==0.05` on every row; `se_gene_dyadic`, `n_effective_pairs`, `n_pairs`, `mean_difference`
  present); `atlas_context_matched_v7.summary.json` discovery side (`n_matched` 640/653/641,
  `n_cases_sampled` 800, `n_case_genes_in_exact_other-tag_support` 1982/1780/2672,
  `n_case_packets_in_exact_other-tag_support` 3015/2681/5103, `max_control_reuse` 2); the sampler's
  one-per-gene enforcement (`build_atlas_matched_cohorts.py:203–213,254`). No Drive artifact, no CUDA, no
  cohort rebuild.
- The CLOSE-AS-UNDERPOWERED direction is **numerically robust** even at the theoretical yield=1.0 hard gene
  cap: donor `6,790/1,780 = 3.8×`, cassette `4,897/2,672 = 1.83×`, acceptor `2,019/1,982 = 1.02×` — all over.

---

## Independent verification (my checks, against code + data this session)

- **20 rows / 17 suppress / 3 support / 11-6-3 categories / 16 mlp_write + 4 attention_head** — confirmed from
  `context.jsonl`. Prereg's "17 mlp_write" is wrong (16/4). ✓
- **N_req** acceptor 2,019 / donor 6,790 / cassette 4,897 — reproduced exactly. ✓
- **Hard one-per-gene cap is binding, not the packet pool** — `select_one_per_gene` skips seen `gene_id`
  (build script L205–213); packet pool 3,015 does not drive the acceptor ceiling. ✓
- **Wrong census** — `census_v1.json` is the intron-mate competition cohort (estimand string verified);
  atlas census `atlas_context_census_v1.json` gives `junction_window_fit_fraction=0.293`. ✓
- **Pooled 6,487** = `selected.atlas_discovery.n_pairs` (cross-contrast, cross-role sum; `n_genes=5089`). ✓

---

## RECOMMENDATION (to Cam) — disposition: **FIXABLE_THEN_RUNNABLE → terminal CLOSE-AS-UNDERPOWERED**

The red-team did its job: **do not freeze B5 as written, but the fixes are cheap, pre-run, and do not touch
the science.** No lens found a fatal; the branch is the most tractable of the five (zero forward passes, all
artifacts local, feasibility lens GO). The one place the design could have licensed an *invalid* disposition
— the pooled-taxonomy escape hatch that would have routed the 3 cassette signals to a hollow "worth-powering"
redesign — must be removed; the pooled reachability estimand is definitionally ill-posed (sign-heterogeneous,
cross-role, gene-double-counting cohort whose `se` is unmeasured and R22-locked). Adjudicated on the **fine
regime only**, with demand and supply unit-matched to independent genes and the verdict led by the **hard
one-case-per-gene gene cap**, the answer is unambiguous: **all 20 HCA18 context signals are under-powered
leads** — each demands more effective matched pairs than exist even at the absolute supply cap (donor 3.8×,
cassette 1.83×, acceptor 1.02× over). The single marginal signal (acceptor L2H4, 1.9% over the hard cap) must
be reported honestly at its thin margin — unit-matching to effective pairs only widens it — not narrated as a
clean close while §5 calls it non-identifiable.

Because `N_req` rests on a **selected, single-draw** `se`, reachability is **one-sided-falsifiable**: the
arithmetic can only rule OUT (license CLOSE), never license a "reachable"/redesign on this discovery `se`
(R22). So there is no scoped-redesign survivor hiding in these numbers — the honest terminal disposition of
HCA18 is **CLOSE-AS-UNDERPOWERED**, exactly HCA18's own boundary ("do not erase, do not validate; a fresh,
better-powered design is required"). B5 should be **executed this session** as the corrected deterministic
power-calc, emitting `b5_context_signal_power_v1.json` + manifest with memorials/calibration (incl. the new
ceiling-channel planted-positives) passing, to freeze that CLOSE record. Any future WH4/WH5 context validation
must *increase supply* (more window-fitting genes / relaxed span at fixed specificity) or accept a coarser
same-sign, same-role pooled estimand with a **freshly measured** pooled `se` — neither of which B5 can
manufacture.

### Must-fix before freeze
1. **Remove the pooled-reachability branch** (reuses a fine signal's `se` on a sign-heterogeneous, cross-role,
   gene-double-counting 6,487-pair cohort; R22 forbids re-drawing the pooled `se`). Mark pooled
   NON-IDENTIFIABLE-AT-DESIGN; route the 3 cassette signals to CLOSE/DEFER, never a pooled redesign.
2. **Resolve §5/§6 precedence** and lead with the hard one-per-gene gene cap; restate the frozen expectation
   as CLOSE-at-hard-cap for all 20, with acceptor L2H4's ~1.9% margin stated honestly (not a clean close while
   §5 marks it non-identifiable).
3. **Fix the census provenance** — delete `census_v1.json`/10%/3,014/444-gene; re-anchor on
   `atlas_context_census_v1.json` (29% atlas window-fit) + the exact-other-tag support pool.
4. **Add ceiling-channel planted-positives** PP3 (`N_req ∈ (n_eff, N_ceil)` → reachable) and PP4
   (`N_req` just above `N_ceil` → not reachable); the current PP1/PP2 never exercise the reachability code.
5. **Unit-match demand and supply** to independent genes (justify gene currency via one-per-gene sampler;
   deflate ceiling by ~0.97 effective/raw or state `N_req` in raw pairs).
6. **Minors:** correct 16/4 component_type count + add to M1; reuse `float(norm.ppf(...))` not `2.8016`;
   relabel M4/M7 as forced (lean on M3); state `N_req` is a selected, single-draw **lower bound** →
   reachability one-sided-falsifiable (bars any reachable/redesign disposition); add a design-effect memorial.
