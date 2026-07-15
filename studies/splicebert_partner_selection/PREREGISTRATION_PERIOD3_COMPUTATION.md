# Pre-registration — period-3's computation (thread 2a)

**Status: FROZEN.** Written before `run_period3_computation.py`. The prompt insists we determine
period-3's computation *before* tracing it. Builds on **C8** (deleting the period-3 clock drives
donor–acceptor coupling 0.946→0.027 and costs coding-exon MLM 0.159 nats vs 0.042 in introns)
and on 1e's lead: **the donor coupling is period-3-*dependent* (C8) yet phase-*independent*
(C19)**, which points at positional addressing rather than frame computation. 2a tests the
carrier's general computation directly.

## 0. The failure modes it guards

Two period-3 traps (from the retraction history) bind here. (1) A stride-3 comb rolled by 3
slides onto itself, so raw phase-roll damage sawtooths **on white noise** — never used as
evidence here; the carrier is DELETED (bins zeroed), not rolled. (2) Damage divided by ‖Δout‖
inflates combs (γ≈1.8) — never used. The "energy-matched control band" that was silently the
second (≈10.63) carrier is avoided: controls are bands verified inert.

## 1. Intervention and metric — imported / reused

- **Carrier deletion:** zero the period-3 rfft bins **339–346** of the (mean-removed) position
  table and invert (`run_carrier_surgery.py::make`), swapping the table in place. Restore after.
- **Period-3 benefit** at a masked position `p`: `Δ(p) = logP_true(clean) − logP_true(deleted)`,
  the true-base log-probability lost when the clock is removed. Gene-grouped bootstrap CIs.
- **Per-base codon position** (must-build, extends `gfmcd.splicing.phase`): for a coding base,
  its codon slot 0/1/2, from the containing CDS's frame and offset; validated before use.

## 2. The discriminating tests (the prompt's table)

| Measurement | Outcome → interpretation |
|---|---|
| **Whole-codon vs single-base masking.** Benefit `Δ` at a coding base when only it is masked vs when all 3 of its codon bases are masked. | benefit **disappears** under whole-codon masking → **within-codon completion**; benefit **survives** → **phase-conditioned nucleotide prior**. |
| **CDS-specificity.** `Δ` at CDS (coding) positions vs matched non-CDS (intron/UTR) positions. | coding-specific → **coding-state detector** (refines C8's 0.159 vs 0.042). |
| **Codon-position dependence.** `Δ` stratified by codon slot 0/1/2. | strong variation → phase-conditioned; flat → position-agnostic within the codon. |
| **Intron mod 3 vs annotated phase** (cited from C19 for the donor coupling; re-measured here for the *benefit* if in scope). | strong mod-3 → raw alignment; follows annotated phase → biological boundary phase. |

## 3. FROZEN expectations (each falsifiable)

| Test | PASS/report | Falsifier |
|---|---|---|
| **Sanity** | deleting the clock costs coding MLM more than intronic MLM (reproduces C8 direction) | no coding>intron gap → the deletion or metric is wrong, halt. |
| **P3-Whole-codon** (primary discriminator) | report single-mask `Δ` vs whole-codon-mask `Δ` with CIs; classify within-codon-completion (whole-codon `Δ` CI-high < single-mask CI-low) vs phase-conditioned-prior (whole-codon `Δ` CI-low > 0) | if whole-codon `Δ`≈0 and single≫0 → completion; if whole-codon `Δ`≈single → prior. Both reachable. |
| **P3-CDS** | `Δ` at CDS positions CI-low > `Δ` at non-CDS CI-high → coding-state component | overlap → not CDS-specific. |

## 4. Firewall placement

Characterizes the period-3 **position-table** mechanism's computation (P→behavior); does not by
itself trace it through QK/OV (that is 2b: router vs transporter) or rotate its plane (2c). No
biological claim.

## 5. Post-hoc reflection (REQUIRED)

State per-test pass/fail; the most defensible reading of period-3's computation from the table;
whether it reconciles with 1e (period-3 necessary-but-phase-independent for the donor coupling →
positional/coding-state role rather than frame); and what 2b/2c must still resolve. Name what is
deferred (router-vs-transporter, the plane-rotation phase cut).
