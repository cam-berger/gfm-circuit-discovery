# B3 CROSS-EXAMPLE TRANSDUCTION — 5-lens design red-team synthesis (2026-07-22)

**Reviews:** `../preregistrations/PREREGISTRATION_B3_CROSS_EXAMPLE_TRANSDUCTION.md` v1 (DRAFT, not
frozen). Five independent adversarial lenses, run BEFORE any confirmatory compute (formalize-audit-verify
step 2). **Outcome: the design is NOT freezable as written — but the scientific target IS identifiable and
MPS-feasible after a bounded REDESIGN of the primary estimand object plus a mandatory discovery pilot.**
Two lenses returned `fatal`; the two fatals are *convergent* (three lenses independently flag the same
object/space mismatch) and both are fixable — this is a pre-run catch of the R7/R15/R23 class, not a P3.4-style
non-identifiable wall.

**Disposition: FIXABLE_THEN_RUNNABLE. survivor = FALSE** (do NOT execute this session): the primary
injected object must be re-specified (it is currently physically impossible), a load-bearing content control
must be added, the dyadic ratio-of-means inference is net-new (not "reused"), and confirmatory feasibility is
*unestablished* until a discovery pilot measures the actual cross-example numerator. Retain v1 as design-of-record.

| Lens | Verdict | Headline |
|---|---|---|
| Identifiability / estimand well-posedness | **fatal** | Primary estimand injects `Π_U[final donor residual]` **through the L3H1 head-output path** — a **shape/space contradiction** (`[8,512]` U vs `[B,32]` head output); the "validated U_info ⇒ lower-risk-than-B2" thesis is false either way (code-verified) |
| Confound / backdoor | **fatal** | A **generic clean-donor offset inside U_info** rides along and is blocked by **none** of the four controls; margin-matching + rank-1 overlap 0.970 force source≈target ⇒ "transfer" ≈ near-self-injection |
| Power / inference (statistics) | fixable | Feasibility argument imports the **ceiling (`R_self`=+0.875) as the effect**; Endpoint B is the P3.4 wide-CI-null relapse; **dyadic ratio CI is net-new, not reused** |
| Calibration / intent | fixable | **No planted-positive travels B3's actual `M_xfer` channel**; M3 self-ceiling is asserted, not demonstrated, for the U-projected L3 path |
| Feasibility / power / MPS | fixable | Cohort sized on **pre-gate** atlas counts (620→153 after short-intron gate); kernel composes and ran in this regime; no CUDA/Drive wall |

---

## The two convergent FATALs (independently verified against code + data)

### FATAL-1 — The primary estimand's injected object is physically incompatible with the cited machinery (identifiability + confound + calibration lenses, unanimous)

- **Intent (§0, §1):** promote `U_info` from within-example mediator to a *transferable* direction, and the
  headline justification (intro L9–11, B3.0) is "the transplanted object is the **already-validated `U_info`**,
  so B3 is **lower interpretive-risk than B2**."
- **Estimand as written (§1 L78–81):** inject `Π_U[r_src]` — "the `U_info` projection of the source's clean
  **FINAL donor residual** — at the target donor token **through the target's held-clean address**, exactly as
  in C16/A8 (`payload.head_output` / `inject_capture` / `patch_heads`)."
- **Why fatal (two different objects at two different layers; code-verified):**
  - `U_info` is `top_k_subspace` of the **clean−corrupt delta of `hidden_states[-1]`** at the donor →
    `U` is `[k=8, H=512]` in the **layer-6 final residual** (`run_a9_payload_semantics.py:110-112`;
    `subspace.top_k_subspace`).
  - The cited `deliver`/`head_output`/`inject_capture`/`patch_heads` path edits a **512-d layer-3 acceptor-window
    residual** `dr[b,4,512]` → `deliver(Wv,dr)` → `[b,4,32]` → `head_output` → a **32-d L3H1 head output**,
    injected donor-only (`payload.py:40-70`; `run_a9:223-227`; `patch_heads` patches head outputs of dim
    `Dh=32`, `hf_bert.py:143-182`).
  - A `[8,512]` final-residual direction **cannot be injected as an `[B,32]` L3H1 head output** — a literal shape
    contradiction. The only 512-d slot in the cited path is the **acceptor-window residual**, which is the
    **acceptor PAYLOAD**, not `U_info`. C24 (established) makes them explicitly distinct:
    the payload "is localized to the four-token acceptor window and is **not concentrated in the donor-derived
    rank-8 `U_info`**; L3H1 **converts** representations" (`CLAIMS.md` C24). L3H1 is the *converter between* the
    two objects — they cannot be the same injected vector.
- **The two coherent repairs both break the study's thesis:**
  - **(A) Transplant the acceptor payload cross-example** — what the machinery literally does (A9 generalized:
    `dr = E[j]["r3cl"] − r3co`, the full L3 acceptor-window delta, injected at L3H1; `run_a9:215-227`, and note
    A9 does **NOT** project onto `U` — `U` is only a readout metric there, `run_a9:231`). Then B3 is an
    **A9-generalization study injecting the R22-WITHDRAWN payload** whose semantic identity is unresolved; the
    "validated-object, lower-risk-than-B2" framing is **false**, and *projecting* that payload onto `U_info`
    (`Π_U`) would discard most of it by C24 → a **near-null by construction**.
  - **(B) Patch the final-residual `U_info` component directly** (as C16/`run_a5` actually does:
    `patch_component` on `hidden_states[-1]`). Then it is **not the cited machinery**, it **bypasses the whole
    circuit** (address, L3H1, delivery), and it is **architecture-FORCED**: the margin is `cls(final donor
    residual)` *exactly* (`run_donor_anchor.py:6,112-116`, the G0 anchor), so injecting a clean-`U` component into
    the corrupt final residual moves the readout by a linear-forced amount (R7/R21 class) — it validates nothing
    about *transfer*.
- **Fix (redesign, not footnote):** pick ONE object and rewrite §1/§2 to match. The recommended repair is **(A)
  the honest A9-generalization** — transplant the acceptor payload cross-example, **DROP `Π_U`** from the injection
  path, **DROP** the "`U_info`/lower-risk-than-B2/validated-object" framing, and OWN the R22 encumbrance (B3's
  claim becomes *transferability of the delivered state*, not its semantics — which is legitimate and does not
  need R22 resolved). Delete every claim that C16/A9 already validated this specific transplant operation —
  neither did. **The interchange test survives the redesign; the "U_info study" headline does not.**

### FATAL-2 — A generic clean-donor offset inside U_info rides along, blocked by none of the four controls (confound lens; code+data-verified)

- **The backdoor:** `U_info` IS, by construction, the axis along which the margin moves clean→corrupt
  (`top_k_subspace` of the clean−corrupt donor delta, `run_a9:110`). ANY real clean donor's residual sits high on
  the clean side of that axis, so injecting a matched source's clean component pushes the target margin up
  **regardless of whether partner-SPECIFIC information transfers**. Two facts make this near-certain, not
  hypothetical: (a) §5 matches source↔target on native margin/strength (`target_strength`), which the prereg
  itself admits forces "`U_info` components near-identical" (L205-207); (b) C26's rank-1 cross-gene subspace
  overlap is **0.970** (`a5_uinfo_replication.json`; prereg L22-23) — the signature of a dominant SHARED
  direction, not per-gene partner content.
- **None of the four controls blocks it:** `N-rand` (wrong subspace), `N-scr` (wrong norm — see MAJOR-1),
  `N-pos` (wrong token) — not one isolates "the partner-intact-*specific* direction within `U` at matched norm"
  from "generic clean-`U` occupancy." **M4** (`‖Π_U[r_src] − Π_U[r_tgt]‖` CI-excludes-0) does **not** block it:
  two vectors dominated by the same shared offset still differ by residual noise, so M4 fires positive trivially.
  ⇒ the modal Endpoint-A "interchange of a partner-specific causal variable" is confounded with "the model has a
  generic clean-donor axis" — a much weaker claim.
- **Fix (add the load-bearing control the guards name):** inject the **SAME source's CORRUPT-run donor
  projection** at matched norm (same gene, same composition, same absolute position, differing ONLY in
  partner-intactness) and require **clean-source > corrupt-source** as the transfer estimand — **not
  clean-source > empty**. If clean-source ≈ corrupt-source, report "generic clean-side occupancy transfers"
  (Endpoint-C-adjacent), NOT interchange of a partner-specific variable. Also add a deliberately
  **margin-mismatched stratum** so the transferred component cannot be the shared offset.

---

## Additional MAJORS (fixable, but load-bearing)

- **[power] The feasibility argument conflates the DENOMINATOR (ceiling) with the EFFECT.** §7 cites "C16
  cross-gene sufficiency +0.875 ⇒ the transfer effect is big" — but +0.875 is `R_self`, the *ceiling*
  (denominator of `R_xfer`), a within-example self-patch through a cross-gene-derived subspace
  (`a5_uinfo_replication.json` rank-8 `R_suff_heldout`=0.8747). The **numerator** — `⟨M_xfer−M_empty⟩` using a
  cross-gene SOURCE's coordinates — **has never been measured**; A9 ran cross-example injection but reported only
  payload-semantic correlation, no margin-recovery ratio. Rank-8 cross-gene subspace **overlap is only 0.63**
  (`a5` rank-8 overlap 0.6304, not the headline rank-1 0.970), so true `R_xfer` could be anywhere ~0→~0.87. Every
  SESOI/power claim rests on an **unestimated** effect size. **Fix:** a mandatory discovery-split PILOT that
  measures the cross-example numerator and its per-gene variance BEFORE freeze, with a P3.4-style discovery-MDE
  gate (pre-declare any contrast whose projected dyadic HW > SESOI_R inconclusive).

- **[power] Endpoint B (powered equivalence, dyadic HW < SESOI_R=0.20) is the P3.4 wide-CI-null relapse,
  relabelled.** The SELF denominator is HW 0.0238 **one-way at 320 genes** (`a5`); B3 has far fewer genes and
  **two-way** dyadic inference. Rescaling HW∝1/√n_genes ×√2 for reuse, the denominator alone at ~100 genes
  inflates to ≈0.06, and `R_xfer` is a ratio with a small, per-gene-noisy numerator. The a9b precedent shows this
  class of dyadic CI failing to exclude effects (matched betas HW 0.11–0.15 at n=200; every DML CI includes 0).
  Under two-way clustering + family-adjusted TOST widening, **HW(R_xfer) > 0.20 is the likely outcome**; the
  prereg asserts "reachable" but never projects the dyadic ratio HW. **Fix:** re-scope to a **one-sided POSITIVE
  dose-slope search** as the confirmable endpoint; treat Endpoint B as reachable ONLY if the pilot demonstrates
  projected family-adjusted dyadic HW < 0.20; otherwise route a null to Endpoint C ("inconclusive at floor"),
  never an "honest null." (This is exactly why B3 is NOT P3.4: re-scoped to positives it is confirmable, because
  a large cross-example transfer, if real, fires — whereas P3.4's positives were the *opposite* of its maintained
  hypothesis.)

- **[power/calibration] The R22 dyadic inference for the PRIMARY estimand is NET-NEW, not "reused."**
  `gboot_ratio` (`circuit.py:303-316`) keys every pair by a **single gene** `g` — one-way, cannot represent
  source-reuse-across-targets. `dyadic_ci` (`run_a9b:119-133`) is hard-wired to the linear projection slope
  `β=⟨S,V⟩/⟨V,V⟩`, **NOT** to a **ratio of two different arm means** `R_xfer=⟨M_xfer−M_empty⟩/⟨M_self−M_empty⟩`.
  So the load-bearing R22 inference for `R_xfer` is **net-new code** — add "two-way (source-gene × target-gene)
  Poisson-multiplier dyadic bootstrap for a ratio-of-means" to `net_new`, implement it, and FPR-calibrate it
  before freeze. Do not present it as reused.

- **[calibration] NO planted-positive travels B3's actual `M_xfer` channel.** PP-U (the cited C16 ≈0.9
  sensitivity) patches the U-component into the **final** residual (`run_a5:136-137`, `subspace.patch_component`),
  never the L3 delivery path; PP-xfer as specified reuses A9's `synth=(Vp,u,beta)` shortcut (`run_a9:218-222`)
  that sets `dr=beta·coef·u` with a **single shared population direction** and BYPASSES `Π_U`/per-source content
  entirely. Both cited PPs validate *different halves*; neither exercises the net-new hybrid. **Fix:** add a
  planted-positive that pushes a synthetic source residual through the **real** injection driver (real projection
  + real `deliver`/`head_output`/`inject_capture`, not the synth shortcut) and require dose-monotone recovery;
  **promote M3 (self-ceiling through the actual driver) from an assumed memorial to a hard `--smoke` gate** and
  confirm it clears the C16 ceiling before freezing.

- **[confound/calibration/power] The scrambled-source null `N-scr` is NOT norm-matched, so `Δ_scr>0` is confounded
  with injection magnitude.** Coordinate-permuting a 512-d residual before `Π_U` collapses its in-subspace norm to
  ~√(8/512)·‖r‖ (`subspace.project`, no renorm), so a positive `Δ_scr` can be a pure magnitude artifact — the very
  norm-push confound §5 claims to exclude. **Fix:** renormalize the scrambled projection to `‖Π_U[r_src]‖` before
  injection (exactly as `N-rand` is norm-matched), and add an on-path memorial `‖injected_scr‖ == ‖Π_U[r_src]‖ <
  1e-6`.

- **[confound] Absolute donor-position geometry is missing from the adjustment set (R15/R20 relocation confound).**
  SpliceBERT has **absolute** learned PEs; both L3H1's address (R20: PE-dominated) and the source residual carry
  absolute position. Injecting a position-carrying source residual at the target's DIFFERENT absolute donor
  position is the R15 "relocating vectors that already carry absolute position is forced" pattern. §5's covariate
  list has **no absolute-position term**. **Fix:** add absolute donor position (and PE-band/phase) to the
  match/bracket or residualize it out; add a position-mismatched-but-content-matched control showing it does not
  transfer *more* than a position-matched pair.

---

## Minors (fixable)

- **[identifiability/power] Target-pool `good`-filter is an undocumented post-treatment collider.** The reused A9
  machinery filters targets to `{i : M_self−M_empty > 0.05}` (`run_a9:240`), conditioning on the value-path
  mediator that transfer also uses → biases `R_xfer` upward; honest scope becomes "gene-disjoint donors **with a
  reliable value-path**," narrower than §0/§9's "donors." State it in §1/§5 and scope every generalization
  sentence; for Endpoint B report only within the conditioned population and quantify the retained fraction.
- **[confound] `N-pos` conflates wrong-location with lower-address-weight** (the donor token carries the largest
  clean `a_win` by construction). Norm-match the *delivered* edit so the off-position null differs only in
  location; report per-arm address weight.
- **[power] `β_dose` clustering inconsistent:** §1 L92 says one-way "gene-clustered," §6/§7 require gene-dyadic.
  State it two-way consistently.
- **[power] Two under-specified inference details:** fix the N-permute-pairing permutation count ≥1000 (a9b's
  n_perm=120 gives a 5%±4% FPR estimate — too coarse for a ≤5% gate); project the **family-adjusted** equivalence
  HW in §7, not the raw HW.
- **[identifiability] Match covariates are atlas MLM-packet fields, not coupling-native.** Add partner acceptor
  strength × exact gap and a CORRUPT_ALT effect-size proxy (via `CouplingTask`) to the caliper and |SMD| check;
  prefer the §2 `load_examples` fallback if the adapter cannot supply coupling covariates.
- **[calibration] §0 quotes rank-1 overlap 0.970 but the study transplants rank-8 (overlap 0.63).** Quote both.

---

## Feasibility & MPS (lens 5) — no CUDA/Drive wall; the blocker is scientific + cohort-size, not engineering

- **Kernel composes and already ran in this regime.** `patch_heads` enforces ONLY a source/target shape check
  (`hf_bert.py:166-171`), so cross-example injection is architecturally supported; `interchange()` already handles
  `(i,j)` cross-example, `(i,None)`=empty, `(i,'self')`=self; `frozen_uinfo`/`inject_capture`/`deliver`,
  `subspace.project`/`random_subspace`, `gboot_ratio` all exist and were exercised by A9/A9b at n=240 over ≤400
  pairs. GTF+FASTA resolve. **SESOI_R=0.20 on a recovery RATIO is genuinely 4× coarser than IC2's ±0.05 and 10×
  P3.4's 0.02-nat wall — the equivalence half-width that stalled prior MPS work is not the binding constraint for
  a one-sided positive search.**
- **Cohort is sized on PRE-gate atlas counts.** The short-intron eligibility gate (`GAP∈[60,150]`,
  `circuit.eligible`, `circuit.py:70,221-226`) shrinks the validation donor pool from 2,013 packets to **620**,
  and the primary `single_pair` class from 237 to **153**; after the §5 |SMD|<0.10 + ±15% norm caliper it drops
  further (a9b retained only 53–69 of 600 candidates on FEWER covariates). Dyadic effective-n is **distinct
  genes**, not pairs. **Fix:** re-size on POST-gate counts; make the `--smoke` cohort gate quantitative and
  blocking (≥ n_min matched pairs AND ≥ the distinct-gene count for dyadic HW<SESOI_R); pre-declare a promotion of
  `donor_cluster` (247) / `cassette_exon` (168) into the primary if `single_pair` is short.
- **Worktree artifact absence is operational, not a wall.** The checkpoint and gitignored enriched inputs are
  absent from this worktree but present in the main repo and CPU-regenerable
  (`enrich_atlas_context_packets.py` has no torch/model forward). Name artifact resolution as a `--smoke`
  precondition.
- **Budget envelope 1e4–1e5 forwards** (6 arms × 5 doses × ~200–400 targets, nulls amortized) is comparable to
  A9; confirm N-rand amortization (shared population block vs per-target draw) at pilot.

---

## Independent verification (my checks, not the reviewers')

- **FATAL-1 shape/space:** confirmed `U` from `hidden_states[-1]` clean−corrupt delta → `[8,512]`
  (`run_a9:110-112`); confirmed A9 injects `dr = E[j]["r3cl"] − r3co` (full L3 acceptor-window delta, **not**
  `Π_U`-projected) delivered as `[b,4,32]` head output via `deliver`/`head_output`/`inject_capture`
  (`run_a9:215-227`; `payload.py:40-53`); confirmed `U_info` is a readout metric only there (`run_a9:231`).
  Confirmed C24 states the payload is NOT concentrated in `U_info` and L3H1 CONVERTS (`CLAIMS.md`). ✓
- **FATAL-1 forced-ness of repair (B):** confirmed margin = `cls(final donor residual)` exactly (G0 anchor,
  `run_donor_anchor.py:6,112-116`) ⇒ direct final-residual U-patch is linear-forced. ✓
- **FATAL-2 backdoor:** confirmed `U` = clean−corrupt axis (`run_a9:110`), rank-1 overlap 0.970 / rank-8 overlap
  **0.630** (`a5_uinfo_replication.json`). ✓
- **Power/net-new inference:** confirmed `gboot_ratio` single-gene key (`circuit.py:303-316`); `good`-filter
  `denom>0.05` (`run_a9:240`); `random_subspace`/`project` have no renorm (`subspace.py:38,17-20`). ✓
- **Feasibility:** confirmed `GAP_MIN,GAP_MAX=60,150` and `eligible` gate (`circuit.py:70,221-226`). ✓

---

## RECOMMENDATION (to Cam)

The red-team did its job: **do not freeze/run B3 v1 as written.** As posed, the *primary estimand names an
injection object that cannot physically exist* (a `[8,512]` final-residual `U_info` direction "injected through
the L3H1 `[B,32]` head-output path"), and its headline justification — "we transplant the already-validated
`U_info`, so B3 is lower-risk than B2" — is **false under both coherent repairs**: the machinery delivers the
**acceptor payload** (R22-withdrawn, not `U_info`; C24), while patching the **final-residual `U_info`** directly
is architecture-forced by the `cls(final residual)` readout. Independently, a **generic clean-donor offset inside
`U_info`** rides through all four controls unblocked, so a "successful transfer" can be near-self-injection.

**But this is not P3.4.** The core scientific instinct — a genuine causal variable should support cross-input
interchange (Geiger; Zhang & Nanda; Heimersheim & Nanda) — is correct, the enabling primitive genuinely exists
and is example-agnostic, and the SESOI sits on a coarse recovery ratio, so the fatals are **fixable by redesign,
not a non-identifiable/training-time wall.** The runnable study is the **honest A9-generalization**: transplant
the **acceptor payload** cross-example (drop `Π_U`; drop the `U_info`/lower-risk framing; own R22 as
transferability-of-state, not semantics), add the **same-source-CORRUPT content control** (`clean-source >
corrupt-source`) as the load-bearing transfer estimand, **renorm `N-scr`**, add **absolute-position** matching,
re-scope the confirmable endpoint to a **one-sided positive dose-slope** (demote Endpoint-B equivalence to
pilot-gated), and write + FPR-calibrate the **net-new two-way dyadic ratio-of-means bootstrap**. A **mandatory
discovery pilot** must then measure the actual cross-example numerator and its dyadic HW — because feasibility
currently rests on the *ceiling* (`R_self`=+0.875), not the *effect*, and rank-8 cross-gene overlap is only 0.63.

**Disposition: FIXABLE_THEN_RUNNABLE; survivor = FALSE for this session.** The fixes change the primary object,
add a load-bearing control and net-new inference, and require a pre-freeze pilot to establish that the
confirmatory endpoint clears its floor at the realized gene count — too much to execute cleanly now, and premature
to run before the pilot proves the numerator exists. Retain v1 as design-of-record; produce v2 with the redesign
above, smoke-gate the new driver + PP + M3, run the discovery pilot, and only then freeze. I will not freeze or
run anything without your call.
