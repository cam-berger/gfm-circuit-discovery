# PREREGISTRATION AR1 — corruption-operator sensitivity (is the coupling a CC artifact?)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

> **Status: FROZEN 2026-08-02, before any model forward pass on either split.**
> Freeze covers §1–§9. Nothing below may be edited after the first run; departures go to
> `CORRECTIONS.md` under a new `R` number, per `AGENTS.md`.

## 0. Intent (the north star — re-read at every gate)

Every confirmatory and mechanistic result in the donor-circuit arc corrupts the partner
acceptor with a **single hardcoded dinucleotide**, `CORRUPT_ALT = "CC"`
(`src/gfmcd/splicing/circuit.py:66`). **Does the coupling — and therefore the arc built on
it — depend on that specific choice?**

This is a *robustness* protocol for an existing operator, not a new mechanism claim. It can
bound the operator scope of C1/C2 (and by inheritance the A-series); it cannot strengthen
them.

### 0.1 What the record currently supports

- The panel `ACCEPTOR_ALTS = ("CC","TT","CA","TC")` / `DONOR_ALTS = ("CC","AA","TA","AC")`
  (`coupling.py:55-56`) is justified only as "destroys the motif, cannot create the other one".
  No recorded reason selects `CC` from within it.
- `run_coupling.py` (first protocol) and `run_ed1_architecture_boundary_pairing.py` average
  over all four acceptor alts, then **discard the per-alt breakdown before writing rows**.
- `run_sealed_test.py` (the controlling artifact for **C1 = +0.872 [0.771, 0.969]** and
  **C2**) and every A0–A12 / carrier / GTEx runner use **`CC` only**.
- Therefore **no per-alt effect has ever been reported anywhere in `results/`.**
- `docs/PREREG_CONFORMANCE.md:302` already recorded a single-hardcoded-alt anchor as a
  medium deviation in the (removed) caduceus package.

## 1. The panel is not assumed valid — it is checked per example

Writing dinucleotide `z` over a motif can **create** a motif in a shifted register even when
`z` is not itself the motif. For a substitution at `[q-2, q)` the three overlapping registers
are

| register | bases |
|---|---|
| A | `seq[q-3] + z[0]` |
| B | `z[0] + z[1]` (the substituted pair itself) |
| C | `z[1] + seq[q]` |

`z` is **admissible for that example and arm** iff no register is `AG` (spurious acceptor) and
no register is `GT` or `GC` (spurious canonical or non-canonical U2 donor). Admissibility is a
deterministic function of the reference genome, computed identically on both splits, and is
**frozen here before any model is loaded**.

**Pre-freeze, model-free measurement** (chr1/2/3 train, n = 4,672 eligible internal exons;
`ExperimentManifest` written with the confirmatory record):

- Acceptor arm: `CC`/`TT`/`TC` admissible on **99.5%**; `CA` on **48.5%** — the first exon base
  is `G` in 51.2% of exons, so `CA` puts `AG` in register C, a *rescued acceptor one base
  downstream*. `AC`/`AT` are admissible on 100%.
- Donor arm: the last exon base is `G` in **79.9%** of donors (the 5′SS consensus), so any `z`
  with `z[0] ∈ {C,T}` puts `GC`/`GT` in register A — a *planted donor one base upstream*.
  `CC` is admissible on only **20.1%**, `TA` on **11.8%**, `AA` on 57.4%, `AC` on **100%**.

This is why the panel is re-derived per example rather than taken from the constants.

## 2. Arms and readouts

All readouts are the native masked-LM margin `logit(G) − max(logit A,C,T)` at a masked
position (`CouplingTask.margin`, `circuit.py:131`). No probe, no head, no labels.

| arm | substitution | readout position | donor visible to the model? |
|---|---|---|---|
| **ACC** | partner acceptor `AG → z` at `[p-2, p)` | masked donor `G` at `d` | n/a |
| **DONM** | donor `GT → z` at `[d, d+2)` | masked donor `G` at `d` | **no** — `d` is masked |
| **DONA** | donor `GT → z` at `[d, d+2)` | masked partner acceptor `G` at `p-1` | **yes** |

`p = next_acceptor_pos` (the intron-mate acceptor), `d = donor_pos`. **ACC is the operator the
whole arc uses**; DONM/DONA are secondary and descriptive.

**DONM is expected to be partly degenerate and that is the point.** Masking `d` hides `z[0]`,
so under DONM (i) all `z` sharing `z[1]` are byte-identical constructs, and (ii) every `z` with
`z[1] = T` is byte-identical to *clean*, i.e. destroys nothing the model can see. Both are
**architecture-forced** (R7/R21) and are recorded as forced memorials, never as findings.

## 3. Estimand

Per example *i* and admissible alt *z*, the paired effect

    Δ_i(z) = margin_i(clean) − margin_i(z)

on the same exon against the same clean baseline. The **primary quantity** is the
gene-clustered **ratio of means** against the incumbent

    ρ(z) = mean_i Δ_i(z) / mean_i Δ_i(CC)

with a gene-clustered bootstrap CI (`gboot_ratio`, 2000 draws). **Never mean-of-ratios**
(`circuit.py:303`). Secondary: the paired difference `mean_i[Δ_i(z) − Δ_i(CC)]` via `gboot`.

Cohorts are **UNCONDITIONAL** (`load_examples(..., min_effect=None)`). Conditioning on effect
size would be selection on the outcome, and the ratio denominator here is a cohort mean
(≈0.87), not a per-example value, so no denominator floor is needed. `den_clear_of_zero` is
reported.

### 3.1 Common support

`ρ(z)` is computed **only on examples where both `z` and `CC` are admissible for that arm**.
Comparing a mean over `z`'s admissible set against a mean over `CC`'s would confound the
operator with the cohort. The common-support n is reported per alt, and an alt whose common
support falls below 100 examples is reported as **INSUFFICIENT SUPPORT**, not as a ratio.

## 4. Decision rule (frozen — δ = ±25% relative)

Per arm, over the alts admissible on ≥95% of that arm's confirmatory cohort:

- **ALT-INVARIANT** iff for *every* such `z` the 95% CI for `ρ(z)` ⊆ **[0.75, 1.33]**.
- **NOT INVARIANT** iff some `z` has a CI lying wholly outside `[0.75, 1.33]`.
- **INCONCLUSIVE** otherwise (CI overlaps a bound without containment) — reported as
  inconclusive, **never as agreement**. Achieved CI half-width is reported per alt either way.

Alts admissible on <95% are reported in a separate *contaminated* table and **cannot**
contribute to the verdict.

### 4.1 Donor-side confound probe (DONA only, difference-in-differences)

Does planting a `GC`/`GT` at `[d-1, d]` rescue the donor? Split examples on `seq[d-1] == "G"`
and difference against the 100%-admissible alt `AC`, which plants nothing in either stratum
and so absorbs any intrinsic strength difference between G- and non-G-context donors:

    DiD = [Δ(CC|G) − Δ(CC|¬G)] − [Δ(AC|G) − Δ(AC|¬G)]

- **CONFIRMED** iff the gene-clustered CI excludes 0 with the predicted sign (CC's drop
  *smaller* in the G stratum, i.e. DiD < 0).
- **REFUTED** iff the CI ⊆ ±0.25 × mean Δ(AC).
- **INCONCLUSIVE** otherwise.

## 5. Memorials (on the measurement path; forced ones labelled)

| memorial | can fail? |
|---|---|
| `variant_construction_exact` — built sequence differs from clean at exactly the two target positions and equals `z` there | **yes** |
| `motif_hygiene_recomputed` — registers recomputed from the *built* sequence match the frozen admissibility classification | **yes** |
| `readout_anchor` — masked index carries `G`; `seq[d:d+2]=="GT"`; `seq[p-2:p]=="AG"` | **yes** |
| `batch_invariance` — a construct scored at two different batch offsets gives identical margins | **yes** |
| `identity_noop` — `AG→AG` / `GT→GT` reproduces clean exactly | **FORCED** (byte-identical input); plumbing only, R7/R21 |
| `masked_position_invisibility` — under DONM, constructs sharing `z[1]` give bit-identical margins, and `z[1]=T` equals clean | **FORCED** by masking; recorded as the demonstration of §2, not as evidence |

A memorial failure halts before any verdict is read.

## 6. Calibration (hard preflight gates)

An equivalence design fails in two directions, so both are gated.

- **Sensitivity — planted difference.** Scale one arm's `Δ` by **1.35** (outside the 1.33
  bound) and run §4 unchanged. Must return **NOT INVARIANT** in ≥80% of gene-bootstrap draws.
  A design that cannot detect a planted 35% shift cannot license "they agree".
- **Specificity — planted null.** Scale by **1.00**. Must return **ALT-INVARIANT**.
- **Specificity — permuted-label FPR.** Within each example, permute which score carries which
  alt label (destroys any true alt effect, preserves example structure); recompute §4 over 200
  permutations. Fraction declaring **NOT INVARIANT** must be ≤ **0.05**. If it exceeds 0.05,
  a calibrated critical value is derived and every positive verdict must additionally clear it.

## 7. Splits — the seal is not spent

- **Power/pilot and all mechanics:** `train` (everything outside val/test).
- **Confirmation:** `val` = chr11/16/19.
- **chr8/chr9 remain SEALED and are not touched by this protocol.**

Panel admissibility is a genome-deterministic rule, not a data-selected quantity, so no
selection leak crosses the split. `n` for the confirmatory run is fixed from the train pilot as
the smallest n whose observed CI half-width on `ρ` is ≤ 0.15; if the pilot cannot reach that at
n ≤ 1200, the run proceeds and the verdict may be **INCONCLUSIVE** by §4 — it is not rescued by
raising n after seeing val.

## 8. What a PASS licenses, and what it does not

An **ALT-INVARIANT** result on ACC licenses exactly one sentence: *the C1/C2 coupling effect is
insensitive, within ±25%, to which admissible dinucleotide destroys the partner acceptor, on
this cohort and this checkpoint.* It does **not**:

- promote C1/C2 to any higher evidence level;
- extend to the mechanism results (A0–A12 remain CC-only; alt-invariance at the behaviour level
  is necessary, not sufficient, for alt-invariance of a circuit decomposition);
- say anything about `[MASK]`, indels, or larger edits — all outside the operator class;
- transfer to the GTEx study, which has its own cohort, its own readout, and its own ledger.

A **NOT INVARIANT** result requires an `R` entry bounding C1/C2's operator scope, and a scan of
every ledger row that inherits the operator.

## 9. Outputs

- Confirmatory record: `results/splicebert_coupling/ar1_corruption_operator_sensitivity.json`
  (+ `.manifest.json`), written only for `--split val` with no `--tag`.
- Everything else → `results/splicebert_coupling/development/ar1_<tag>.json`.
- `guard_output_path` refuses to overwrite; `--overwrite-record` is the only escape.
- Reported per alt and arm: n admissible, mean Δ, `ρ`, CI, CI half-width, winsorized Δ, tail
  share of the top decile, plus every memorial and calibration gate — **including the ones that
  fail**.

Runner: `scripts/splicebert/run_ar1_alt_robustness.py`
Model: SpliceBERT.1024nt, checkpoint sha256 `2ad91428…0149d`. `PYTHONHASHSEED=0`.
