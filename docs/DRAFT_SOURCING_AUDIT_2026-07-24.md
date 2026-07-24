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

### 5.1 RETRACTED from this audit: the C41 `τ_PAR` item

**An earlier revision of this document listed C41's `τ_PAR ≈ 0.18` as unsupported by any artifact and
flagged it for an owner check. That was an error in the audit, not in the ledger.** It is withdrawn.

`τ_PAR ≈ 0.18` is `tau_PAR_NAT_lin`, the natural-norm parallel component in the linear regime at
`alpha = 0.25`. It is recorded in **every one of the 8 confirmatory draws** in
`results/splicebert_coupling/g1_cross_example_transduction.json`:

```
0.1937  0.1804  0.1683  0.1937  0.1936  0.1783  0.1643  0.1742
mean 0.1808   median 0.1793   range [0.1643, 0.1937]
```

`CLAIMS.md` is correct, and the estimand pairing is the right one: `tau_PAR_NAT_lin ≈ 0.18` against
`PERP_MATCHED_lin ≈ 0.00`, both linear-regime, which is exactly the comparison the generic-axis
verdict rests on.

The audit error was searching `g1_cross_example_transduction_decomp.json` for the wrong estimand:
that file's `tau_PAR_matched` (`0.912`) and `tau_PAR_natural` (`0.714`) are full-dose quantities, not
the linear-regime α = 0.25 one, and its only ≈0.18 value is `tau_PERP_matched = −0.178` — same
magnitude, opposite sign, different quantity. Concluding "absent" from one file when the confirmatory
artifact carries it eight times is precisely the failure mode this audit exists to catch, and it is
recorded here rather than silently deleted.

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
| `τ_PAR` (linear, α=0.25) | ≈0.18 | 0.181 — **ledger correct**, see §5.1 | [0.164, 0.194] |
| `β` dose | 0.27 | **0.243** | [0.215, 0.266] |
| REAL − RANDOM | +0.62 | 0.600 (median 0.595) | [0.508, 0.680] |
| REAL − SCRAMBLE | +0.68 | 0.689 | [0.490, 0.900] |
| SELF sufficiency | 0.86 | 0.863 | [0.856, 0.875] |

One row is a **pilot value quoted as a confirmatory summary**: `β = 0.27` matches the `H_dev` pilot
(`0.2737`), while the confirmatory `H_conf` across-draw mean is `0.243`.

The `REAL − RANDOM` row is a smaller mismatch of unclear provenance rather than a pilot value: the
confirmatory per-draw mean is `0.600` and the median `0.595`, and `+0.62` is not draw 0 (`0.680`) nor
any single draw. There is no across-draw block for this quantity in the artifact. Low priority.

## 7. Rounding differences (immaterial, listed for completeness)

| Claim | Ledger | Artifact |
|---|---|---|
| C8 deletion endpoint | `0.946 → 0.027` (CLAIMS); `→ 0.026` (EVIDENCE.md) | `0.94604 → 0.02646` (`carrier_dose.json`) |
| C16 random-subspace bound | `≤0.05` | `null_max = 0.053` (`donor_anchor.json`) |
| A9b composition R² | `CV-R²(S~Z) < 0` (CLAUDE.md) | `−0.071` held-out, `+0.006` discovery |

## 7.1 IC1 ran at one fifth of its sealed N

`PREREGISTRATION_IC1_DISTAL_INTRON_NECESSITY.md` seals the confirmatory N as a census, explicitly to
remove run-time discretion: *"Confirmatory N = the conservative-constructible census ≈ 1264 clusters
(dose ∧ construction; frozen, no run-time N choice)"* (§5, and again in the freeze table at §192–194:
`dose 0.61 × yield 0.94 × 2205`). Sealed power is `MDE_census ≈ 0.013` at `sd_between = 0.162`.

`ic1_distal_intron_necessity_v8.json` records **`n_clusters = 256`** (259 eligible, 22 excluded
non-constructible), with achieved `MDE = 0.0293`.

That is roughly a fivefold shortfall against a frozen N, and it is not reconciled in any live
document. It matters for how the result reads: at the sealed N the design would have resolved
effects down to `0.013`, comfortably inside the `0.05` SESOI; at the executed N the floor is `0.029`,
which is close enough to the SESOI that the sub-threshold reading rests on a narrower margin than the
protocol intended. The recorded status (`confirmatory: false`, outcome-aware verification) already
prevents over-reading, and this compounds it rather than contradicting it.

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
| 1 | Correct the `m_T = 0.001` claim in `figures/splicebert/EVIDENCE.md` (§1) | No — clear error |
| 2 | **Reconcile IC1's executed `n_clusters = 256` against its sealed census N ≈ 1264 (§7.1)** | **Yes — largest open item** |
| 3 | Re-render C41's `Δ_spec` and `τ_PERP` range-notation so they are not read as CIs (§6) | Yes |
| 4 | Reconcile `CLAIMS.md` C41's `β = 0.27` with the confirmatory mean `0.243` (§6) | Yes |
| 5 | Note the executed-vs-preregistered deviations against their claims (§4) | Yes |
| 6 | Record the §2/§4 conflict inside `PREREGISTRATION_SEALED.md`'s ledger entry, not the frozen file (§2) | Yes |
| 7 | Trace the provenance of C41's `REAL − RANDOM = +0.62` (§6) | Low priority |

**Withdrawn from this list:** an earlier revision asked for an owner check on C41's `τ_PAR ≈ 0.18`.
That was an audit error; the ledger is correct. See §5.1.

Nothing here was applied. This document is additive and touches no frozen record, ledger, figure of
record, or result artifact.
