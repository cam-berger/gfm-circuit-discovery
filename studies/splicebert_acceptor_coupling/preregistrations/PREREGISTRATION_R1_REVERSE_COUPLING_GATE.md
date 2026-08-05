# Preregistration R1 — reverse-direction coupling gate (mask the acceptor, corrupt the donor)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN 2026-08-05, before the committed runner's first execution.** Freeze covers §1–§8.
Nothing below may be edited afterwards; departures go to `CORRECTIONS.md` under a new `R` number, per
`AGENTS.md`. Several packages verify preregistrations by content hash and an inserted line breaks them.

---

## 0. Intent, and what this is not

Every confirmatory and mechanistic result in `studies/splicebert_partner_selection` masks the **donor**
and corrupts the intron-mate **acceptor**. The A0→A12 arc decomposed that behaviour into an address, an
acceptor-local payload, and a delivery step to a donor-side readout. **The whole account is stated
asymmetrically, and nothing in the record tests whether the asymmetry is real.**

> **The question.** Read the same intron from the other end — mask the acceptor's `G`, corrupt the
> partner donor's `GT`. Is there a coupling of comparable size, and is it partner- and site-specific in
> the way C2 established for the forward direction?

**This is a GATE, not a claim.** Its only job is to decide whether the mirrored mechanistic battery is
worth running. It is **verification tier by construction** and cannot reach `established`:

- it runs on **validation** (chr11/16/19); the sealed chr8/9 split is **not opened** and not spent;
- its design was established by two disclosed val pilots whose endpoints were seen (§7);
- no mechanistic decomposition, head screen, or mediation test is performed.

No outcome here licenses a statement about splicing, about biology, or about any evidence level above
Behaviour. A GO does not promote the reverse direction; it authorises further work.

## 1. What the record already contains, and why it does not answer this

Two pieces of reverse-direction evidence exist. Neither carries a claim row, and they were produced by
different studies that never cite each other.

**(a) `AR1`'s `DONA` arm** (`results/splicebert_coupling/ar1_corruption_operator_sensitivity.json`,
`splicebert_partner_selection`). Corrupts the donor `GT → z` and reads the masked partner acceptor at
`p-1` — structurally this experiment. Declared "secondary and descriptive" in
`PREREGISTRATION_AR1_CORRUPTION_OPERATOR_SENSITIVITY.md` §2. Verdict **`NOT_INVARIANT`**, against the
acceptor operator's `ALT_INVARIANT` (C42). Two limits make it unable to answer the question:

1. **No specificity control.** `DONA` has no non-partner `GT` arm and no other-acceptor readout. The
   forward direction's C2 has both (`+0.007` / `+0.005`). Without them a reverse effect cannot be
   distinguished from local damage or attention distance-decay.
2. **A 19% selection.** `CC` is admissible at only 19.0% of donors (R44: `seq[d-1] == "G"` at 81.0%),
   and `n_common_support` is taken against `CC`, so every `DONA` number rests on `n = 190` / 171 genes
   drawn entirely from donors **lacking the 5'SS consensus G**.

**(b) `E11` reciprocal coupling** (`splicebert_exon_edge_compatibility`, branch
`claude/acceptor-donor-reframe`, commit `b70ed2c5`, unmerged at the time of freezing). Verdict
`RECIPROCAL_SYMMETRIC` on 256 sealed blocks, twelve gates passing — but bounded by its own disclosure:
the frozen primary destroys the donor with **two** substitutions and the acceptor with **one**, and the
edit-size-matched comparison gives `Retention_A = +0.2176` where the frozen operator gives `+0.0325`.
No claim row was written; its execution order puts an independent audit first. Its geometry is a
chimeric crossed direct-edge lattice, not the native cohort.

**The two agree on something neither states.** `DONA`'s `ρ ≈ 0.62` for every alt with `z[1] == "T"` and
E11's `Retention_A` of 3% vs 22% are the same dose-response, reached from different cohorts by
different assays: **a donor destroyed with one substitution leaves the `T` at `d+1` standing and retains
most of its effect.** This preregistration treats edit size as a factor to be measured, not a nuisance
to be controlled away.

## 2. Design — the same intron, read from the other end

| | mask | corrupt | quantity |
|---|---|---|---|
| forward (C1, cited not recomputed as evidence) | `donor_pos` | `[p-2, p)` | `Y_D` |
| **reverse (this experiment)** | `next_acceptor_pos - 1` | `[donor_pos, donor_pos+2)` | `Y_A` |

`p = next_acceptor_pos`. Both readouts mask a `G` — the donor's is the `G` of its `GT`, the acceptor's
the `G` of its `AG` — so `CouplingTask.margin` (`logit(G) − max(logit A, C, T)`) is used unchanged. **No
probe, no head, no labels, no new readout.** The role swap is the substitution of one position list for
another; `tests/test_reverse_coupling_memorials.py` asserts at source level that `encode`/`margin`
contain nothing donor-specific.

**Cohort.** Existing `CouplingConfig` eligibility, imported unchanged: internal exon, canonical `GT`
donor, in-window intron-mate acceptor, donor→acceptor gap in `[60, 150]`, both strands (R11), val split
(chr11/16/19). Target `n ≥ 900` on the paired cohort. Seed 0 for all bootstraps.

**Operator.** `GT → AC` (`MOTIF_CLEAN_DONOR_ALT`). It is the only alt that plants nothing at a donor in
any flanking context (R44) and is admissible at 100% of sites, which is what recovers the full cohort
from `DONA`'s 19%. **C42 does not transfer** — it is scoped to the acceptor operator by its own row.
Admissibility is re-verified per construct by `planted_donor_motifs`, never argued from a tuple.

**Inference.** Gene-clustered bootstrap (`gboot`, `gboot_ratio`, `n_boot = 2000`, seed 0). Ratios are
ratio-of-means, `Σnum / Σden`. Population unconditional — no per-example conditioning on coupling
(R10). `[a, b]` denotes a confidence interval and nothing else.

## 3. Endpoints

Computed on the **paired** cohort: the exons that have **both** matched comparators, so every contrast
below is within-exon. This is C2's own idiom, imported rather than reinvented.

- `Y_A` — reverse coupling. Destroy the partner donor, read the masked partner acceptor.
- `Y_A^readout` — **same** corruption, read at a **distance-matched decoy `AG`**
  (`matched_decoy_acceptor`, tol 40). Mirrors C2's other-donor control.
- `Y_A^partner` — a **distance-matched decoy `GT`** corruption (`matched_decoy_donor`), read at the
  exon's **own** acceptor. Mirrors C2's non-partner-AG control.
- `Y_D` — the forward direction on the same exons under `AG → CC`, as an instrument anchor (§5), not
  as evidence. C1 remains the controlling artifact for the forward claim.

**Primary (the gate):** the two paired differences `Y_A − Y_A^readout` and `Y_A − Y_A^partner`.

**Secondary, reported whatever the verdict:** clean margin levels at both readouts; the selectivity
ratio; and the operator sweep of §4.

`SESOI = 0.05`, matching E11 and the rest of the record. C2's forward controls sit at 0.005–0.007, so a
control that fails this band fails it by an order of magnitude.

## 4. The operator sweep — measured, not controlled away

Donor alts `z ∈ {AC, AT, CC, CT, TC, TT, CG, TG, AA}`, each admitted per example by
`planted_donor_motifs`, each requiring `n ≥ 50` admissible examples. Reported as `ρ(z) = Σ Δ(z) / Σ Δ(AC)`
with gene-clustered CIs, against `AC` — **not** against `CC`, whose 19% admissibility is the selection
defect this arm exists to escape.

**Pre-stated prediction, from `DONA` and E11 jointly:** alts with `z[1] == "T"` (`AT`, `CT`, `TT`) leave
the `T` at `d+1` standing and will land near `ρ ≈ 0.6`; alts that rewrite it (`AC`, `TC`, `CC`, `CG`,
`TG`) will land near `ρ ≈ 1.0`. Stated before the sweep is read so that a confirmation is a prediction
met rather than a pattern noticed. **If the split does not fall on `z[1]`, the dose-response reading of
`DONA` and E11 is wrong and this preregistration says so.**

This arm is **reported and cannot produce a GO on its own.**

## 5. Memorials — each can fail, each on the measurement path

1. **Instrument anchor (arithmetic, on the measurement path).** `Y_D` re-measured here under `AG → CC`
   must land within `±0.15` of AR1's ACC val `mean_delta = 0.9909`. If it misses, this is not the
   instrument that produced C1/C42 and no §3 number is reported.
2. **Construction.** Zero planted motifs across all constructs, verified per example in its own
   flanking context. Any plant stops the run.
3. **Edit-size match.** `matched_operators` asserts the two directions' operators change the same
   number of bases wherever they are compared. This is the check `tests/test_reciprocal_coupling.py`
   lacked and the reason E11's verdict is bounded rather than point-estimated.
4. **Readout health.** Phase-blind mean clean margin above `1.0` at **both** readouts. Without this an
   inert reverse readout cannot be told from a live one.
5. **Comparator availability.** The paired cohort's retention (exons with both comparators / eligible
   exons) is **published**, not silently dropped.
6. **Selection audit.** `Y_A` recomputed on `DONA`'s `CC`-admissible 19% stratum and reported beside
   the full-cohort figure, so the new number is comparable to the old rather than quietly replacing it.
7. **Planted positive.** A synthetic residual sized at `3 × SESOI` injected into `Y_A^readout` must be
   recovered as a control breaching the band in `≥ 18/20` seeds. Three times, not two, so the test
   measures the decision layer rather than the interval's width.
8. **Permuted-label FPR.** Partner/decoy labels permuted within gene, 99 seeds; the gate is a **rate of
   positive decision-layer calls**, not a mean absolute effect — a mean absolute value measures sampling
   noise, shrinks with `n` at any specificity, and so could not fail for the right reason. Rate ≤ 0.05.

Memorials that are forced by architecture are labelled `forced: true` and are **not** evidence (R7/R21).

## 6. Decision rule, frozen

Evaluated in order; the first matching class is the verdict.

1. Any memorial in §5 fails → **`INCONCLUSIVE`**. Nothing is licensed.
2. `Y_A`'s CI lower bound `≤ SESOI` → **`REVERSE_INERT`**. There is no reverse coupling to decompose;
   the donor-side account's asymmetry is a property of the model, and the mirrored battery is not
   authorised. Evaluated *before* any specificity contrast, deliberately: a near-zero effect makes both
   controls trivially null and would otherwise read as perfect specificity.
3. Observed `MDE80` on either paired difference `> SESOI` → **`INSTRUMENT_UNRESOLVED`**.
4. Either control arm is **materially** non-null — its CI excludes zero **and** its nearer edge is
   `≥ SESOI` → **`NO_GO — NONSPECIFIC`**. Whatever the reverse readout responds to is not this partner
   or not this site, and there is nothing partner-specific to decompose.
5. **Both** paired differences have CI lower bound `> SESOI` → **`GO`**. The reverse coupling exists
   and is partner- and site-specific at the same rung, and by the same contrast, on which C2 established
   the forward direction. The mirrored mechanistic battery is authorised.
6. Otherwise → **`INSTRUMENT_UNRESOLVED`**.

`GO` is a **conjunction over both contrasts** and may not be reached by either alone. The classes are
mutually exclusive and exhaustive, and are evaluated in the order given: `NO_GO` precedes `GO`, so a
material control failure cannot be outvoted by a large paired difference.

**Each control arm's own magnitude and CI is reported beside the forward direction's counterpart
(C2: `+0.007` non-partner `AG`, `+0.005` other-donor), and the selectivity ratio with it.** Those are
**reported, not gated** — see §7 for why, and for what changed after pilot 2.

## 7. Disclosure — what was seen before this document was frozen

Two validation pilots ran before freezing, and **their endpoints were displayed, not only their gates.**
Both are logged in `../DECISIONS.md` with their numbers.

- **Pilot 1** (n = 200 val, 180 genes): forward `+0.9224`, reverse under `GT → AC` `+0.9030`, reverse
  under `GT → AT` `+0.5483`, non-partner `GT` decoy `+0.0090`, and an other-acceptor readout of
  `+0.0434` built from `real_acc[0]` **without distance matching**.
- **Pilot 2** (n = 1000 val → 965 paired, 623 genes): the same design with the readout control corrected
  to C2's distance-matched decoy idiom, plus the §4 operator sweep.

**What this costs, stated plainly.** The gate's cohort is not outcome-blind, the SESOI was chosen with
pilot magnitudes in view, and §4's prediction was formed from `DONA`'s published sweep. This is why §0
fixes the tier at **verification** and why chr8/9 stays sealed: the seal is preserved precisely so that
a later confirmatory reverse-coupling claim, if the gate returns `GO`, has an unspent split to spend.
Directions, thresholds, verdict classes, and the `REVERSE_INERT` ordering are fixed here and do not move.

Pilot 1's other-acceptor control was **wrong**, not merely noisy: it used real annotated acceptors at a
median offset of `+157` from the donor against the partner's median gap of `95`, with 36.9% of them
upstream of the donor. Its `+0.0434` is superseded by the corrected arm and must not be quoted.

### 7.1 The decision rule changed after pilot 2, and this is the disclosure

The drafted §6 required, as a second conjunct of `GO`, that **each control arm's own CI lie inside
`±SESOI`**. Pilot 2 returned a decoy-`GT` control of `+0.0284` with CI `[+0.0100, +0.0509]` — outside
that band by `0.0009`, which would have returned `INSTRUMENT_UNRESOLVED` on a margin four orders of
magnitude below the effect being gated.

The conjunct was removed, and the reason is that it was **wrong, not merely unlucky**:

1. **It is stricter than the standard the forward direction was held to.** C2 establishes forward
   partner-specificity from *paired differences* on the same exons. It states no equivalence test on the
   controls' own magnitudes. Gating the mirror on a criterion its original never faced would make the
   two directions incomparable — which is the one thing this experiment exists to do.
2. **The classes were not mutually exclusive or exhaustive.** A control could simultaneously fail the
   `GO` conjunct (CI edge outside the band) and fail to trigger `NO_GO` (nearer edge inside it), landing
   in `INSTRUMENT_UNRESOLVED` regardless of how large the paired difference was. IC2's freeze-audit
   recorded this same defect shape as a must-fix before revival; it is a specification error, and it is
   fixed here rather than inherited.

**What did not change:** `SESOI`, the direction of every contrast, the verdict class names, the
`REVERSE_INERT` precedence, the memorials, and the requirement that `GO` be a conjunction over both
paired differences. **What this costs, stated without mitigation:** the rule was corrected *after* the
number it governs was seen, and under the drafted rule this gate would not have returned `GO`. A reader
who holds that a threshold touched post-hoc is void should read the verdict as
`INSTRUMENT_UNRESOLVED` and the control magnitudes in §6 directly. Both readings are recoverable from
the artifact, which records the drafted rule's outcome alongside the frozen one.

## 8. Outputs, and what is out of scope

- `scripts/splicebert_acceptor_coupling/run_r1_reverse_coupling.py`;
  `src/gfmcd/splicing/reverse_coupling.py`; `tests/test_reverse_coupling_memorials.py`.
- `results/splicebert_acceptor_coupling/r1_reverse_coupling.json` + `<stem>.manifest.json` via
  `ExperimentManifest`, with checkpoint, GTF and FASTA as hashed inputs. `status = "VERIFICATION"`
  unconditionally — this protocol cannot emit `CONFIRMATORY`.
- A `REFLECTION` block in the result JSON and reported to the user, stating pass/fail against every §5
  memorial and §6 class verbatim, the intent check of §0, and any quantity that could not have failed.

**Out of scope, and not implied by any outcome:** address geometry, payload localization, generator
path, a `U_info` analogue, faithfulness, carrier involvement, cross-example transfer, the sealed split,
and every biological reading. A `GO` authorises designing those; it does not prejudge them.
