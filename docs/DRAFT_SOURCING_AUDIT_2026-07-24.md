# Draft-sourcing audit — ledger vs artifact, 2026-07-24

> **Tier: T2 — provenance.** Supporting record. This document **records** findings; it changes no
> claim, no ledger, and no frozen protocol. Where it disagrees with
> [`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md),
> the ledger remains authoritative until an owner ruling says otherwise.

## What produced this

A Methods-and-Experiments paper draft was sourced against the SpliceBERT donor-circuit record. The
sourcing rule was deliberately strict: **every number came from its controlling immutable artifact,
not from a summary document**, and each was re-checked by an independent adversarial pass. The
manuscript itself is intentionally not committed here.

That exercise is a useful instrument in its own right, because it exercises the record the way an
external reviewer would: it reads a claim, follows it to the artifact, and checks that the artifact
says what the claim says. The findings below are what that check turned up. None invalidates a
conclusion. Several would have propagated into a publication.

## 1. One factual error in a live document

**`figures/splicebert/EVIDENCE.md` § "Why the human-only null is a null and not a blind instrument"**
argues that the null is not a power artifact because *"the best human-only head has `m_T = 0.001`."*

In `results/phase-lattice/fig_report_data.json` the best human-only head is **L2H9 with
`m_T = 0.556`**. No human-only head has `m_T = 0.001`. The `0.001` figure appears to be a confusion
with the preregistered primary head L1H3's *band mass* (`0.00193`, i.e. the 0.19% quoted correctly
two paragraphs earlier), which is a different quantity.

**The document's argument survives, and in fact strengthens, on the correct number.** L2H9's
`p_exact = 0.796` — 318 of its 400 permutations match or exceed it. The head's apparent signal sits
squarely *inside its own permutation null*, rather than being a weak signal that narrowly missed an
FDR threshold. That is a better refutation of the blind-instrument objection than the one currently
written.

**Recommended:** correct EVIDENCE.md. Not done here — it is a figure-of-record evidence document,
and this audit does not edit frozen or of-record material.

## 2. Provenance conflict in a frozen preregistration

`preregistrations/PREREGISTRATION_SEALED.md` **§2** states the validation population gives
`+0.973 [0.919, 1.023]` unconditional and `1.386` conditional, a 1.44× inflation.

Two other records disagree with §2 and agree with each other:

| Source | Unconditional | Conditional | Inflation |
|---|---|---|---|
| `PREREGISTRATION_SEALED.md` §2 | `+0.973 [0.919, 1.023]` | `1.386` | 1.44× |
| `PREREGISTRATION_SEALED.md` §4, Gate A1 | `+0.958 [0.855, 1.059]` | — | — |
| `results/splicebert_coupling/sealed_val.json` | `+0.958 [0.855, 1.059]` | `1.285` | 1.34× |

The prereg is internally inconsistent, and its §4 row matches the archived artifact. **The
provenance of the §2 numbers is unexplained.** The frozen document must not be edited; the conflict
is recorded here so the artifact values are used downstream.

The *point* §2 makes — that conditioning on `min_effect` inflates a behavioural mean and must never
be reported as one — is unaffected and remains correct.

## 3. Units are inconsistent project-wide

The coupling metric is `margin(clean) − margin(corrupt)` where
`margin = logit(G) − max(logit A, C, T)`. **The computed quantity is a raw MLM logit difference.**

Most documents label it and its derived thresholds in **nats**; `PREREGISTRATION_A0` says **logit
units**. Genuinely log-likelihood-based quantities elsewhere in the record (the masked-base exon and
intron damage in `carrier_surgery.json` and `sibling_carrier_bandtransfer.json`) *are* nats, so both
labels are in live use for different quantities under the same word.

This is cosmetic for every internal comparison and non-cosmetic for external readers. No renaming is
proposed here, since SESOI thresholds are frozen in preregistrations under the "nats" label and
changing the word would decouple protocol from result.

## 4. Executed-vs-preregistered deviations

Each is defensible and most are documented in a decision log, but none was filed as a numbered
amendment, and they are not currently visible from the claim ledger.

| Experiment | Preregistered | Executed |
|---|---|---|
| A12 (C33) | iterative backward greedy, recomputing `F` after each removal | static backward-removal rank |
| A12 (C33) | node counts described as interpolated | grid-resolution: smallest evaluated `\|K\|` meeting threshold |
| A9b (C30) | GBR as primary DML nuisance learner | linear ridge (justified by calibration: GBR overfits the low-dim `Z`) |
| P3.2 / P3.3 (C34/C35) | energy/importance-matched control-band (E) arm | not implemented in either runner |
| G1 (C41) | planted-CONFOUND arm pairing each target with its caliper-most-extreme source | never invoked (`match_pairs(extreme=True)` unreachable) |
| G1 (C41) | planted-positive gate: fitted slope's CI must **cover 1** | point check against slope `0.786`, no interval recorded |

Two consequences worth stating plainly:

- **A12's "72 of 103 nodes" is a grid point, not a crossing.** The curve gives `F = 0.899` at 64
  nodes and `F = 0.937` at 72, so the true 90% crossing lies in `(64, 72]`. The no-knee conclusion
  is unaffected; the specific integer is softer than it reads.
- **`Δ_bind` has no null at all.** R23 already establishes it is architecture-forced, and the E-band
  arm that would have supplied an importance-matched null was never run. "Reported, not interpreted"
  is therefore the only available treatment, not a conservative choice among several.

## 5. Assertions carried in prose but absent from the artifact

- **P3.3 plumbing anchors "EXACT 0.0".** Asserted in `CLAIMS.md`, `CORRECTIONS.md` and
  `DECISIONS_A0.md`. `results/p3_av_decomposition.json` has no `memorials` key and records no such
  value. (`p3_carrier_locus.json` *does* carry a memorial,
  `construction_P_hybrid_max_abs_err = 1.49e-08`.) The anchor may well have been checked at runtime;
  it is not in the record.
- **C41 `τ_PAR ≈ 0.18`.** This value appears in `CLAIMS.md` but nowhere in
  `g1_cross_example_transduction_decomp.json`, which records `τ_PAR` matched
  `0.912 [0.790, 1.062]` and natural `0.714 [0.623, 0.827]`. **Unresolved — needs an owner check.**
  The qualitative claim (the generic-aligned component carries the transfer) is supported either way.

## 6. Aggregation and interval-notation mismatches

`CLAIMS.md` renders two C41 quantities in interval notation:
`Δ_spec = −0.33 [−0.345, −0.276]` and `τ_PERP = 0.00 [−0.015, +0.010]`.

In `g1_cross_example_transduction.json` those brackets are the **across-draw ranges over the 8
resampled-source draws**, not confidence intervals. The per-draw dyadic CIs are much wider — draw 0
gives `Δ_spec` CI `[−0.527, −0.220]` against a half-width of 0.153. A reader who takes the ledger
notation literally will substantially understate the uncertainty.

Recomputed across-draw means from the confirmatory artifact, for reference:

| Quantity | `CLAIMS.md` | Artifact (across-draw mean) | Across-draw range |
|---|---|---|---|
| `τ_REAL` | ≈0.65 | 0.659 | [0.591, 0.719] |
| `Δ_spec` | −0.33 | −0.320 (median −0.330) | [−0.345, −0.276], SD 0.023 |
| `τ_PERP` (linear) | 0.00 | −0.0001 | [−0.015, +0.010] |
| `β` dose | 0.27 | **0.243** | [0.215, 0.266] |
| REAL − RANDOM | +0.62 | **0.600** | [0.508, 0.680] |
| REAL − SCRAMBLE | +0.68 | 0.689 | [0.490, 0.900] |
| SELF sufficiency | 0.86 | 0.863 | [0.856, 0.875] |

The two bolded rows are **draw-0 / pilot values quoted as confirmatory summaries**. `β = 0.27`
matches the `H_dev` pilot; the confirmatory `H_conf` mean is 0.243.

## 7. Rounding differences (immaterial, listed for completeness)

| Claim | Ledger | Artifact |
|---|---|---|
| C8 deletion endpoint | `0.946 → 0.027` (CLAIMS); `→ 0.026` (EVIDENCE.md) | `0.94604 → 0.02646` (`carrier_dose.json`) |
| C16 random-subspace bound | `≤0.05` | `null_max = 0.053` (`donor_anchor.json`) |
| A9b composition R² | `CV-R²(S~Z) < 0` (CLAUDE.md) | `−0.071` held-out, `+0.006` discovery |

## 8. Sealed-split reach, restated

Worth surfacing because a reader will assume more than is true: **only the behaviour (C1/C2) and the
`{L3H1, L2H15}` subcircuit arms were measured on the sealed chr8/chr9 split.** A7–A10, A9b, P3.2,
P3.3, IC1 and B3 all ran on validation. P3.2 and P3.3 additionally drew under the `min_effect ≥ 0.3`
outcome-conditioning filter, which R24 already identifies as the likely source of the inflated
residual coupling that failed C34's M1 gate on confirm.

Also worth stating once, plainly: **no external biological assay enters this study anywhere.** There
is no RNA-seq, junction-usage or PSI grounding. Every result is model-native margin work against
annotation-derived labels.

## Recommended actions

| # | Action | Owner call needed? |
|---|---|---|
| 1 | Correct the `m_T = 0.001` claim in `figures/splicebert/EVIDENCE.md` (§1 above) | No — clear error |
| 2 | Resolve C41's `τ_PAR ≈ 0.18` against the decomp artifact (§5) | **Yes** |
| 3 | Re-render C41's two range-notation quantities so they are not read as CIs (§6) | Yes |
| 4 | Reconcile `CLAIMS.md` C41's `β` and REAL−RANDOM with the confirmatory means (§6) | Yes |
| 5 | Note the executed-vs-preregistered deviations against their claims (§4) | Yes |
| 6 | Record the §2/§4 conflict inside `PREREGISTRATION_SEALED.md`'s ledger entry, not the frozen file (§2) | Yes |

Nothing here was applied. This document is additive and touches no frozen record, ledger, figure of
record, or result artifact.
