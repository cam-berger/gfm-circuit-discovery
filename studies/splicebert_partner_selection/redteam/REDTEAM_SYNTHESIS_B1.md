# B1 — ≈10.63 carrier past attention (closes C12) — 5-lens design red-team synthesis (2026-07-22)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Reviews:** `../preregistrations/PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR.md` v1 (DRAFT, not frozen). Five
independent adversarial lenses, run BEFORE any confirmatory compute (formalize-audit-verify step 2). **Outcome:
the branch is SALVAGEABLE and MPS-feasible, but NOT freezable as written.** The design has TWO structurally
different arms and they earn opposite verdicts:

- The **behavioural arm** (`ΔC(D10)` donor–acceptor coupling under BAND_P10 deletion) is genuinely well-posed,
  identifiable, reuses same-checkpoint machinery, and is the study's real deliverable — but its **equivalence
  power is asserted, never sized** (a second-difference variance / ρ concern two lenses raised; P3.4-FATAL-3
  class).
- The **OV/mediation arm** (`ΔO_h → U_ov^h → S_ov^h` write-onto-`U_info`) is **non-identifiable as an OV-write
  claim as written** — one convergent FATAL across three lenses (confound, identifiability, calibration), plus
  a specificity control (RECIP) that **cannot be constructed** on the 1024nt checkpoint. Unlike P3.4 this fatal
  is **fixable on the frozen checkpoint** by importing the existing P3.3/C35 A-vs-V decomposition — it is a
  redesign of the OV estimand, not a footnote, and not a training-time dead end.

Disposition: **FIXABLE_THEN_RUNNABLE** (not DEFER — the blocker is an estimand redesign + a power pilot, both
MPS-reachable; not a training-time question like P3.4). **Not a this-session survivor:** the must-fixes are a
genuine OV-estimand redesign and a required ΔC power pilot, not a small patch, and the confirmatory OV
equivalence is unreachable at budget (must be re-scoped to positive-only search).

| Lens | Verdict | Headline |
|---|---|---|
| Identifiability | fixable | Behavioural arm clean & separable; OV `ΔO_h` conflates the QK/address column (the ALREADY-assigned C11/C35 role) with the OV column — a ladder column-jump |
| Power & inference | fixable | Primary coupling-equivalence power **asserted, not sized** (second difference; needs ρ≥0.98 at n=256); OV equivalence **unreachable** at budget (CI-HW ≥ SESOI_ov); FWER on the A-OV gate uncontrolled |
| Confound / adjustment-set | **fatal** | `ΔO_h = A_I·V_I − A_D·V_D` — a pure router (exactly C35/R24 for these heads) yields non-zero ΔO_h overlapping `U_info`; the guards control magnitude, not the routing-vs-content channel |
| Calibration ↔ intent | fixable | No planted-positive travels `dO_h→top_k_subspace→S_ov` (modal A-null is a dead-instrument vs true-null ambiguity, P3.4 PP-completion class); RECIP is not energy-matched and **cannot be** on 1024nt; A-null "does not use" contradicted by the arm's own seen +0.026-nat MLM |
| Feasibility / power / MPS | fixable | Runnable core composes on MPS reusing instruments already run at n=250–384; W_O-projected per-head capture is net-new (~10 lines); OV arm fully contingent on an unrun Preflight-R that may return ∅; worktree lacks `data/models` |

---

## The convergent FATAL (verified against code + data)

### FATAL — the OV estimand conflates ROUTING (ΔA) with OV-CONTENT (ΔV); the flagship endpoint is non-identifiable (confound lens fatal; identifiability + calibration lenses concur as major)

- **Intent (§0, §1 B1-OV, Endpoint A-OV):** promote C12 to a **second causal positional mechanism** —
  the ≈10.63 carrier's post-softmax attention **writes carrier-specific OV content onto the donor readout
  `U_info`**.
- **Estimand (§1, lines 74–89):** `ΔO_h = O_h(I) − O_h(D10)`, where `O_h` is head h's attention-weighted,
  W_O-projected output. Algebraically `O_h = A_h · (H_in W_V W_O)`, so `ΔO_h = A_I·V_I − A_D·V_D`.
- **Why fatal (verified):** deleting BAND_P10 from the absolute position embeddings P (D10) changes `H_in`
  **globally**, so it perturbs **both** the attention pattern `A_h` (the QK/address channel these heads were
  ALREADY assigned to by C11 — "the carrier is READ post-softmax") **and** the values `H_in W_V`. A **pure
  router** — precisely what C11 establishes for these heads and what C35/R24 concluded for the period-3 carrier
  ("routes through L3H1's attention address, NOT carrier-dependent changes in its values") — produces a
  **non-zero** `ΔO_h` that overlaps `U_info` and passes `suff_recovery`, because re-pointing attention changes
  the output even with carrier-independent values. This is exactly the `O = A·V` non-separability R23 flagged
  and that the P3.3 program was **built to resolve** with an explicit attention-restore-vs-value-restore
  decomposition — which B1 omits. By the project's own finding, the modal driver of any positive `ΔO_h` is the
  routing change, so **a positive result would relabel a QK/Attention effect as an OV write** — the ladder
  column-jump the mission forbids.
- **The installed guard is for the wrong confound.** The prereg names `M-ovnull` ("beats matched random",
  §5 Risk-4) as "the R23-class forced-identity guard," but `M-ovnull` only rules out a random-direction
  triviality — it does not touch the A·V conflation. `M-ovspec` (RECIP band-vs-band) controls band identity /
  magnitude, not the routing-vs-content channel: random subspaces and other band-deletions are not
  carrier-independent-value perturbations, so neither guard can separate ΔA from ΔV.
- **Calibration cannot reject a router.** There is **no router-only planted-negative**: a head known to only
  route (period-3 L3H1 per C35) is never required to yield `S_ov ≈ null`, so the estimand's ability to reject a
  pure router is untested.
- **Fix (redesign, not footnote — MPS-feasible):** hold attention fixed. Reuse the existing P3.3 A-vs-V
  decomposition (`results/p3_av_decomposition.json` and its driver): capture `O_h` with the **intact-carrier
  attention pattern but D10 values** (and vice versa), and attribute to OV **only** the value-restore
  contribution — only a carrier-dependent-VALUE component may be called an OV write. Add a **router-only
  planted-negative** (run the identical pipeline on period-3 L3H1, established routing-only per C35, and
  **require** `S_ov` in equivalence with the null; if a known router fires, the OV arm must not be interpreted).
  Relabel §1/Endpoint A-OV/§9 to a MEDIATION-through-head-h claim (Attention/mediation column) everywhere the
  value-isolation is not performed.
- **Verified against:** `src/gfmcd/circuits/hf_bert.py:114–139` (`capture_head_outputs` returns pre-W_O
  `[B,H,S,D_head]` — the total head output, not a value-isolated component); `PREREGISTRATION_B1…md:74–89, 242`;
  CLAUDE.md P3.3/C35/R24; `src/gfmcd/splicing/anchor_gates.py:17–20` (`suff_recovery` patches a direction, not
  content).

**This single fatal is why the OV arm cannot be frozen. It is fixable — the machinery exists on this
checkpoint — but the fix is a new estimand + two new calibration legs, and it degrades the likely deliverable
from "second mechanism" to a bounded mediation/positive-only result.**

---

## Additional majors (fixable)

### MAJOR-1 — the RECIPROCAL specificity control cannot be constructed on 1024nt (confound + calibration lenses; VERIFIED)
The prereg specifies RECIP = "adjacent energy-matched 7-bin band, period off ~10.6" (§1 line 68). **Verified
false for this checkpoint:** `run_second_carrier.py` states there is "exactly ONE other high-frequency mode:
bin 96." In `carrier_surgery.json`, every 7-bin control window **not** overlapping bin 96 deletes only
~0.003–0.006 nats (periods ~21–24, bins 14–52; several negative) — 5–9× weaker and spectrally mismatched
(low-frequency drift, not a carrier) — while every window touching bin 96 (bins 92–98…95–101) shows elevated
damage 0.015–0.026 **because it contains the carrier**. So "adjacent energy-matched off-carrier" **does not
exist**: adjacent ⇒ bin-96-contaminated, energy-matched ⇒ smooth drift. The 510nt postsoftmax reciprocal only
worked because that checkpoint had **two** real adjacent carriers (human 10.24 `[49,50,51]` vs multispecies
10.61 `[46,47,48]`) serving as each other's matched control — a natural control 1024nt lacks. Consequently
`M-ovspec` passes for a **raw-magnitude** reason (a stronger deleted mode gives a bigger ΔO_h independent of any
carrier-specific write), so the specificity gate cannot fail for the right reason.
**Fix:** drop the "energy-matched RECIP" framing; state plainly no character-matched off-carrier band exists on
1024nt; base OV specificity on the surgery control-band DISTRIBUTION (periods 21–60, near-zero damage) as an
explicitly magnitude-mismatched null, and lean on an **actually** energy/covariance-matched OV-output null (not
the isotropic `random_subspace`). Verified against: `run_second_carrier.py:11–17`; `carrier_surgery.json`
controls; `run_postsoftmax.py` reciprocal precedent; prereg §1 line 68, `M-ovspec` line 155.

### MAJOR-2 — no planted-positive travels the OV channel; the MODAL outcome (A-null) is uninterpretable (calibration lens; P3.4 PP-completion class)
`PP-U` (§4 lines 169–170) patches the intact `U_info` component and calibrates `suff_recovery/null_recoveries`
**on `U_info` itself** — it never exercises `ΔO_h = O_h(I) − O_h(D10) → top_k_subspace → S_ov^h`. So a null
`S_ov^h` cannot distinguish "carrier writes nothing onto `U_info`" (true null) from "the single-head `ΔO_h` at
k=1,2 is too noisy for the SVD/projection to recover a real write" (dead instrument). The design's own prior is
that A-null is the modal outcome ("Prior: unlikely — coupling already ≈inert," line 242), so **the most likely
result is uninterpretable** — the exact P3.4 PP-completion failure. **Fix:** add a planted-positive that travels
the full `dO_h → top_k_subspace → S_ov` channel and MUST fire (plant a known low-rank write into a reader head's
OV output and recover it), OR drop the OV-equivalence endpoint (A-null) and scope OV to **positive-only** (A-OV,
self-validating via M-ovnull), declaring OV-equivalence non-identifiable rather than a powered null. Verified
against: prereg §4 lines 166–179, §1 lines 84–87; `anchor_gates.py:17`; REDTEAM_SYNTHESIS_P3_4.md lines 118–119.

### MAJOR-3 — PRIMARY coupling-equivalence power is asserted, never sized (power lens; VERIFIED; P3.4-FATAL-3 class)
The A-null primary requires a POWERED equivalence: the gene-clustered CI on the **paired second difference**
`ΔC(D10) = C(I) − C(D10)` must sit wholly inside ±SESOI_C = 0.05. The prereg justifies reachability with
"`carrier_dose` n=256 already resolved D10 cleanly" — but **verified:** `carrier_dose.json` stores only scalar
coupling means per band/alpha (0.9460 intact vs 0.9700 BAND_P10-deleted; `coup_all` computed but not persisted),
**no variance**. The only coupling-margin CI on record (`donor_anchor.json behavior.coupling_ci` HW≈0.135 at
n≈250) implies a per-gene coupling-margin SD≈1.09 nats. `ΔC` is a second difference: its half-width at n=256
clears the effective bar (HW < 0.026, because the seen point estimate ≈ +0.024 is already ~½·SESOI_C, so a clean
TOST needs HW < SESOI_C − |point|) **only if the per-example intact-vs-D10 coupling correlation ρ ≥ 0.98**; at
ρ=0.95 it needs ~670 genes, at ρ=0.90 ~1340. Treating this as established without a `ΔC` variance estimate is
the point-estimate-over-CI error the repo forbids. **Fix:** the smoke/discovery pilot MUST compute the
gene-clustered `ΔC(D10)` CI half-width **and** ρ, and gate equivalence on HW < (SESOI_C − |point|); pre-declare
INCONCLUSIVE (not equivalence) if unmet. If ρ < 0.98 the endpoint is unreachable at n=256 → re-scope to ~700+
genes or a one-sided read. Verified against: `carrier_dose.json` (scalar coupling, `coup_all` absent);
`run_carrier_dose.py`; `donor_anchor.json behavior.coupling_ci`; prereg §5 Risk-3, §7.

### MAJOR-4 — the OV-null half of A-null is unreachable at budget (power lens; VERIFIED)
A-null's OV half needs a POWERED equivalence of `S_ov` to the matched-random null (SESOI_ov = a random rank-k
subspace's ~95th-pct recovery). From `donor_anchor.json` that null scale is tiny (null_max 0.0155 at k=1, 0.053
at k=8) while the `suff_recovery` gene-clustered CI half-widths at n≈250 are ~0.02–0.05 (info.k1.suff_ci
HW≈0.047). Since CI-HW ≥ SESOI_ov, a powered OV equivalence is **not achievable** at budget — the OV-null half
collapses to inconclusive, and §7 gives no OV-equivalence MDE. **Fix:** add an explicit OV-equivalence MDE from
the C16 null spread; either justify a larger SESOI_ov / raise N_NULL·n, or pre-declare the OV-null
**one-sided-only** (`S_ov` does not exceed the matched-random null). Do not claim "not used" on the OV column
without demonstrating HW < SESOI_ov. Verified against: prereg §6 (A-null), §7; `donor_anchor.json`
gates.info.k1 suff_ci / null_max.

### MAJOR-5 — A-null "represented + routed but does NOT use" is contradicted by the arm's own seen data (calibration lens)
A-null's headline "a positional feature the model represents and routes but does NOT use" (§6) is contradicted
by the design's own already-seen secondary: deleting BAND_P10 raises masked-MLM NLL ~0.026 nats uniformly
(exon 0.0265, intron 0.0253 in `carrier_surgery.json`) vs ~0 for inert control bands — the carrier **IS** used.
The honest bounded claim is "not used for donor COUPLING and writes nothing carrier-specific onto `U_info`,"
never unqualified "does not use." Compounding: coupling is chosen as primary because `U_info`/machinery live
there, while the one native output with a seen positive (general MLM) is relegated to "descriptive, never drives
an endpoint" on a power technicality — structurally protecting the "not used" narrative from its own contrary
evidence. **Fix:** rewrite A-null to the qualified claim; pre-commit that a detectable one-sided MLM positive is
reported as "used non-coding-specifically" (the true contrast to period-3's coding-ENRICHED use, exon 0.174 ≫
intron 0.043), not folded into "not used." Verified against: `carrier_surgery.json` (band 94–100 exon
0.0265/intron 0.0253; period-3 exon 0.174/intron 0.043); prereg §6, §1 secondary.

### MAJOR-6 — L1/L2 reader heads: ΔO_h captures upstream P-propagation, not head-h's own carrier reading (confound lens)
Deleting BAND_P10 perturbs P **globally** (`carrier_dose.py` explicitly warns of this). A layer-≥1 reader's
input residual has already absorbed the P-edit through earlier layers, so `ΔO_h` there mixes (a) the head
reading the carrier with (b) the head reading an already-shifted input. The 1024nt QK-reader candidates
(`second_carrier.json`) span L0–L2; only L0 heads have embeddings-only input. **Fix:** restrict per-head "OV
write" claims to L0 readers, OR freeze each head's input residual to the intact run and vary only that head's
carrier contribution; report the layer of every head in `R_1024` and scope accordingly. Verified against:
`carrier_dose.py` (global perturbation warning); `second_carrier.json` readers span L0–L2; prereg §1 lines 76–78.

---

## Minors
- **M-ovnull / M-del mislabelled as forced-identity guards (R23/R21 class).** `M-ovnull`'s null is the
  ISOTROPIC uniform-orthonormal QR draw (`subspace.py:38–42` → `anchor_gates.py:29–35`), **not** energy- or
  covariance-matched, despite the prereg repeatedly calling it "energy-matched random rank-k" (§1, §4 N-randsub,
  §5 Risk-4). Against an anisotropic head-output delta an isotropic null is too easy to beat. `M-del`'s
  `<1e-6` FFT round-trip is architecture-FORCED (irfft of a bin-zeroed rfft reproduces the definition, R21
  class) — only the peak-bin/wrong-axis clause can fail; `M-ovbase` is capture-plumbing-only. **Fix:** relabel
  these as random-direction / plumbing checks; either build a genuinely covariance-matched null or elevate the
  surgery control-band distribution to the load-bearing specificity guard; anchor `M-del` to a *consequence*
  (comb-amplitude collapse).
- **A-OV family-wise error uncontrolled.** A-OV fires if "≥1 of ~8 heads beats the per-head p≤0.05 null";
  FWER ≈ 1−0.95^8 ≈ 0.34. §6 names max-T but never binds it to the per-head `M-ovnull` threshold. **Fix:** apply
  the family (max-T over shared gene-resamples, or BH) adjustment to the per-head add-one-p that drives A-OV;
  state family α (FWER 0.05). Verified against: `postsoftmax.json` (8 Level-A heads); prereg §3, §6.
- **MLM secondary is vacuous at budget even one-sided.** At the OV-arm budget (n≈150–250) the exon MLM CI-HW
  scales to ~0.027–0.032 while the point estimate is only ~0.026, so a one-sided lower bound does not clear 0.
  Report the raw point estimate with its CI and make no enrichment/depletion claim.
- **PP-read is strength-mismatched.** It plants the strong period-3 carrier to prove Preflight-R yields a
  non-empty Level-A set, but the ≈10.63 target is weaker/diffuse; a period-3 pass does not calibrate Preflight-R
  MDE at the target SNR, so `R_1024=∅` is ambiguous ("not read" vs "below sensitivity"). Add an SNR-matched or
  dose-graded reader-sensitivity check, or scope `R_1024=∅` as "not detected at Preflight-R sensitivity S."
- **Cross-layer/position transplant of `U_ov^h` is geometry-vs-function ambiguous (R20-adjacent).** `O_h` is
  captured at reader query positions (`run_second_carrier.py` idx = arange(260,760)) but `suff_recovery` patches
  into the layer-5 DONOR final residual; dimensionally valid, but `S_ov^h` can be positive from shared
  anisotropic absolute-PE geometry, not a carrier→donor write. State the capture position(s) and lean on
  band-specific (not isotropic) controls per the R20 discipline §9 already invokes.
- **T2 control-band reconciliation is correctly handled — noted for completeness.** Band 94–100 IS
  simultaneously BAND_P10 (treatment) and the worst-damage energy-matched inert control in
  `carrier_surgery`/`carrier_dose` (verified: exon damage 0.0265). The drafter's M-ctrl (exclude period 10.6±5%
  from the re-pointed control search) + fresh-cohort discipline is the right fix; the §9 reconciliation
  obligation must be **executed**, not merely declared — and note the seen dose already shows a mild monotone
  **positive** coupling nudge (0.944→0.970), so the band is mildly non-inert **for coupling** even under the
  equivalence reading.

---

## Feasibility / power / MPS — no compute blocker, one launch gotcha
- **The runnable core composes on MPS.** Every reused instrument already executed at the needed n on these
  checkpoints: `run_donor_anchor` (n=250), `run_carrier_surgery` (n=384), `run_carrier_dose` (n=256). The OV
  kernel is present and dimension-agnostic (`subspace.top_k_subspace/subspace_overlap/patch_component`,
  `anchor_gates.suff_recovery/null_recoveries`, `hf_bert.capture_head_outputs/patch_heads`). Estimated
  ~5k–15k cheap forward passes total, within the budget prior runs used.
- **Net-new W_O-projected per-head capture.** `capture_head_outputs` returns the PRE-projection per-head context
  `[B,H,S,D_head=32]`; the estimand needs the W_O-projected 512-d residual contribution `O_h`. Projecting head
  h's 32-d slice through its column-block of `attention.output.dense.weight` is ~10 lines of net-new code on the
  OV measurement path — keep `M-ovbase` (recomposed `O_h` reproduces native residual contribution to <1e-6) as a
  hard smoke gate and verify head-block slice indexing against a full-attention reference. Verified against:
  `hf_bert.py:114–139`.
- **OV arm fully contingent on an unrun Preflight-R that may return ∅.** The 8 cited Level-A readers are on
  510nt (`postsoftmax.json` multispecies, carrier [46,47,48]); on 1024nt the only known BAND_P10 readers are
  **pre-softmax** QK heads, a **disjoint** L0/L1 set (`second_carrier.json`). No evidence any 1024nt head reads
  BAND_P10 post-softmax → `R_1024` may be empty → OV arm = Endpoint B; the coupling prior already predicts
  A-null. The "feasibility HIGH / likely a second mechanism" framing is optimistic; the likely deliverable is a
  bounded "routed but not used," not a second mechanism.
- **Launch gotcha (minor, not scientific).** `DEFAULT_MODEL` is a relative path resolving against the worktree;
  the worktree `data/` holds only `NUCLEOSOME_MAP.md` — the three checkpoints live only in the main checkout.
  Run B1 from the main checkout, symlink `data/models` into the worktree, or pass an absolute `--model`.
- **Preflight-R memory at S=1024.** `run_postsoftmax.profiles()` holds all 6 layers' `[B,16,S,S]` attentions
  (`output_attentions=True`), built at S=510; at S=1024 the footprint is ~4× → set batch 1–2 to avoid MPS OOM.
- **MLM-wall unreachability honestly self-flagged.** The prereg correctly rescopes the ~0.026-nat MLM arm to
  one-sided/at-achieved-precision (C8 anchor HW 0.0198 at n=384), the correct P3.4/IC1 discipline.

---

## Independent verification (my checks, not the reviewers')
- **FATAL A·V:** confirmed `capture_head_outputs` returns pre-W_O `[B,H,S,D_head]` (`hf_bert.py:114–139`); D10
  perturbs P globally so `ΔO_h = A_I·V_I − A_D·V_D` mixes routing and values. C35/R24 (CLAUDE.md) establishes the
  carrier routes through attention, not values → modal driver is ΔA. ✓
- **RECIP non-existence:** confirmed `run_second_carrier.py:11–17` ("exactly ONE other high-frequency mode: bin
  96"); `carrier_surgery.json` off-bin-96 controls damage 0.003–0.006 (bins 14–52) vs 0.026 (bins 94–100). No
  energy+character-matched adjacent band exists on 1024nt. ✓
- **ΔC power:** confirmed `carrier_dose.json` stores scalar coupling only (control rows 0.944→0.970 monotone up,
  no per-example variance); `donor_anchor.json` coupling-margin CI HW≈0.135 at n≈250 ⇒ SD≈1.09 ⇒ ρ≥0.98 needed
  for a second-difference HW<0.026 at n=256. ✓
- **Isotropic null:** confirmed `subspace.random_subspace` is a uniform QR draw (`subspace.py:38–42`), not
  energy-matched, despite the "energy-matched" wording in the prereg. ✓
- **Checkpoint mismatch:** confirmed `postsoftmax.json` multispecies = 510nt, carrier [46,47,48], 8 Level-A on
  L2/L3/L4; `second_carrier.json` 1024nt readers are pre-softmax L0/L1/L2 — disjoint. ✓

---

## RECOMMENDATION (to Cam)

**Do not freeze/run B1 as written; disposition FIXABLE_THEN_RUNNABLE, not this-session survivor.** The branch is
genuinely worth completing — it is the one column right on a ladder already built, and BOTH of its honest
outcomes publish — but the draft as it stands would run a **non-identifiable OV arm** (the `O = A·V` conflation
relabels the carrier's ALREADY-established QK/routing role as a novel OV write — the exact R23 error the P3.3
program was built to avoid, on the exact heads C35/R24 showed to be routers) and would claim a **powered
coupling equivalence it has never sized** (a second difference needing ρ≥0.98 at n=256). Unlike P3.4 this is
**not** a training-time dead end: the OV fix (P3.3 A-vs-V decomposition to isolate ΔV, a router-only
planted-negative, an energy-matched null, and a live-channel planted-positive) all exist on this frozen
checkpoint, and the behavioural arm is clean. But the fixes are a real estimand redesign plus a required power
pilot, and the confirmatory OV-equivalence is unreachable at budget — so the realistic deliverable is a
**behaviour-primary bounded null** ("the ≈10.63 carrier reaches controlled post-softmax attention but is not
used for donor coupling and writes nothing carrier-specific onto `U_info`") plus an **OV positive-only search**
(a "second mechanism" only if it fires against a genuine value-isolated, energy-matched, router-negative gate),
not the "second causal mechanism" headline the draft leads with. Apply the seven must-fixes, run the ΔC + OV
power pilot as a hard pre-freeze gate, then re-submit for freeze; if the OV redesign or the ρ pilot fails,
degrade to the behaviour-only bounded finding and scope C12-OV as unreached. The measurements survive; the
ambitious OV-write endpoint must earn a value-isolated, powered gate before it is written.

**Disposition: FIXABLE_THEN_RUNNABLE. survivor = false** (must-fixes are an OV-estimand redesign + a required
power pilot, not a small patch; confirmatory OV equivalence unreachable → re-scope to positive-only).

**Must-fix before freeze:**
1. **Redesign the OV estimand to isolate ΔV** — import the P3.3/C35 A-vs-V decomposition
   (`results/p3_av_decomposition.json` + driver): attribute an OV write ONLY to the value-restore contribution
   (intact-carrier values on the D10 base, attention held at D10). Relabel every "OV write" the decomposition
   does not license as a MEDIATION-through-head-h (Attention column) claim.
2. **Add a router-only planted-negative** — run the identical OV pipeline on period-3 L3H1 (routing-only per
   C35) and REQUIRE `S_ov` in equivalence with the null; a firing router voids the OV arm.
3. **Add a live-channel planted-positive** that travels `dO_h → top_k_subspace → S_ov` and MUST fire, OR drop
   the OV-equivalence (A-null-OV) endpoint and scope OV to positive-only.
4. **Drop the "energy-matched RECIP" claim** — state no character-matched off-carrier high-freq band exists on
   1024nt; replace the isotropic `random_subspace` with a covariance-matched OV-output null, and use the surgery
   control-band distribution as an explicitly magnitude-mismatched specificity reference.
5. **Size the primary in the pilot** — compute the gene-clustered `ΔC(D10)` CI half-width AND ρ; gate
   equivalence on HW < (SESOI_C − |point estimate|); pre-declare INCONCLUSIVE if unmet; re-scope to ~700+ genes
   or one-sided if ρ<0.98.
6. **Bound A-null wording** — "not used for donor COUPLING and writes nothing carrier-specific onto `U_info`,"
   never unqualified "does not use"; report any one-sided MLM positive as "used non-coding-specifically."
7. **Control FWER on A-OV** (max-T/BH over the ~8-head family bound to the per-head M-ovnull p), restrict
   per-head OV claims to L0 readers (or freeze upstream input residuals), and fix the launch path
   (`data/models` absent in worktree).
