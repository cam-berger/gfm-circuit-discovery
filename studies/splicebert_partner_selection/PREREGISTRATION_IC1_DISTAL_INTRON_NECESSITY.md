# Pre-registration — IC1: does distal intron interstitial ARRANGEMENT modulate donor–acceptor coupling?

**Status: FROZEN v4 (2026-07-17). Two design-relevant pilot findings flagged for Cam's final sign-off
(§8); confirmatory NOT run until Cam confirms.** First experiment of the intron-coupling arc, opened after
R22 (payload semantic identity unresolved; next target = conditional splice competitiveness, not absolute
strength). **Behaviour-first**: full-model coupling assay only; circuit localization (L3H1 / payload /
`U_info`) is a **separate, contingent** later prereg.

**Lineage:** v1 → 5-lens red-team (11 must-fixes) → v2 → disjoint train pilots (`ic1_pilot.json`,
`ic1_pilot2.json`) sealed thresholds → v3 → 4-lens freeze-audit (pin ORs, seal the last thresholds) → v4.
All numeric thresholds are sealed pre-run from the DISJOINT train split; **no val data is read before the
run.** Every gate below is designed to be able to FAIL and to sit on the measurement path (R7/R21).

---

## 0. INTENT (north star — re-read at every gate)

> **Does the ARRANGEMENT (ordering) of distal intron sequence — with mono- and DI-nucleotide composition,
> both splice-site contexts, the branchpoint/PPT zone (to −70), intron length, and positional address (up
> to triplet-phase; §7 iii) held fixed — modulate donor–acceptor coupling, in minimal internal introns of
> the tested length domain, NET of created cryptic-site competition?**

Honest boundaries baked in: (i) **mono+di composition held EXACTLY fixed**; k≥3-mer composition and
triplet phase are DEDUCTIVE CONSEQUENCES of ordering that co-vary with the shuffle and **cannot** be
separated from pure order — a material effect is attributable to *dinucleotide-composition-controlled
sequence ORDERING (incl. its k≥3-mer/phase content)*, never to "distal content." (ii) The anchored shuffle
can build cryptic 3′SS/PPT even with GT/AG positions fixed (pilot: Δmax-score3 p95=8.6), so the primary
claim is scoped to the **no-new-motif stratum** (§5) and cryptic-site competition is isolated by the
paired arm-difference (§6). (iii) A null = *"no modulation above the achieved MDE, within the tested
minimal-intron (ACC_UP=70, L≳107) arrangement domain"* — NOT "distal content doesn't matter," and NOT a
statement about the protected branchpoint/PPT.

---

## 1. Domain, observable, eligibility, protected zones, DISTAL window

- **Observable (frozen LINEAR functional).** `f = logit(G) − mean{logit A,C,T}` at the masked donor
  (`donor_pos`; intron-start GT). **Linear, NOT the max-margin task metric** — the DiD's additive
  cancellation requires linearity. `A⁺` = native acceptor; `A⁻` = partner acceptor corrupted (**AG→CC** at
  `[a−2,a−1]`, `a=next_acceptor_pos`).
- **Intron domain.** Canonical internal introns (`GT…AG`), native pair, on-strand, in-window; `L∈[60,150]`.
- **Protected zones (bit-identical):** donor 5′SS `[d₀−3 .. d₀+6]`; acceptor-proximal `[a−ACC_UP .. a+3]`,
  **`ACC_UP=70` (SEALED PRIMARY)** — protects 3′SS + PPT + branchpoint **to −70** (past the mapped BP mode
  −19…−37 and most of the tail), so **no BP predictor is used**. Pilot: 2205 eligible genes,
  signed-mean `MDE_census`=0.013 (pre-dose) / ≈0.017 (post-dose ~1345) ≤ 0.05.
- **DISTAL window** `W = [d₀+7 .. a−ACC_UP−1]` (⇒ minimal intron `L ≳ 107`).
- **Conservative robustness subset `ACC_UP=100`** (pilot: 560 genes; L≳137) — **required to sustain any
  positive "arrangement" claim** (guards the residual BP tail beyond −70). A positive primary not holding
  on this subset is **provisional, not confidently distal**.
- **Inference unit = homology cluster** (≥80% id → one cluster). Pilot near-duplicate rate among eligible
  `W` = **0.5%**, so cluster ≈ gene for human internal introns; gene counts below ≈ cluster counts.
- **Eligibility (else EXCLUDE + narrow scope):** (1) `ℓ_dist ≥ 30`; (2) **dose floor** (§5): median altered
  ≥ 8 AND median normalized-Hamming ≥ 0.25 AND all `N_SEED` realizations distinct. Pilot: ~61% pass ⇒
  **~1345 eligible genes** at ACC_UP=70 (≫ N_MIN).

---

## 2. Perturbation — distal interstitial dinucleotide shuffle (`N_SEED=16`)

Perturb **only** `W`; protected zones bit-identical; **whole-intron dinuc multiset preserved exactly**
(unit-tested; `ic1_pilot.json`).
- **PRIMARY (motif-constrained):** anchored Altschul–Erikson shuffle — internal `GT`/`AG` positions FIXED,
  interstitial runs shuffled. (Anchoring fixes AG *positions* but not the PPT it can build — hence §5's
  created-motif stratifier.)
- **SECONDARY (unconstrained):** whole-`W` dinuc shuffle (may move/create cryptic GT/AG) — the
  cryptic-competition arm, run on the SAME introns; interpreted only via the §6 paired contrast.
- `N_SEED=16` per intron per arm (dispersion is a primary readout; pilot: 8→16 dispersion corr 0.90 and
  8-seed underestimates ~11%, so 16). Seeds preregistered (content-independent).

---

## 3. Estimand and controls

Per intron, per seed: `ΔC = [f(I',A⁺) − f(I',A⁻)] − [f(I,A⁺) − f(I,A⁻)]`. Retain ALL per-seed `ΔC`.
- **Two PRIMARY readouts (co-primary; one confirmatory test, §4):**
  - **Signed** `ΔC̄` (seed-mean → cluster mean).
  - **Dispersion** = per-intron across-seed **RMS** of `ΔC` (16 seeds), cluster-aggregated, measured as
    **EXCESS over the sham** `= RMS(ΔC) − RMS(ΔC_sham)` (the sham removes state-dependent damage — pilot:
    RMS_sham 0.059 is ~half of RMS_shuffle 0.112, so the excess, not the raw dispersion, is the signal;
    the exact-0 noop floor is retained only as a mechanics anchor, NOT the comparator).
- **Main effects, symmetric (NO conditioning):** report `Δf⁺=f(I',A⁺)−f(I,A⁺)` and `Δf⁻=f(I',A⁻)−f(I,A⁻)`;
  the pure-damage null IS `Δf⁺=Δf⁻`. Adjustment (if any) = regress `ΔC` on `Δf⁺` over the FULL sample. **The
  `|Δf⁺|`-stratum collider is banned** (a real state-dependent main effect would manufacture a false ΔC).
- **SHAM-acceptor `A⁻'` (SEALED):** an **f-shift-matched** non-partner **PPT purine-swap** (pilot reference
  `substitute(a−10,'AA')`, median f-shift 0.56× of `A⁻`; the confirmatory selects swap position/count
  per-intron to reach **≥0.8× f-shift match**). Reuses the same shuffles. **Material gate:** `ΔC_sham` CI ⊆
  `[−0.05,+0.05]` (pilot mean 0.003 → state-independence holds at population level). Also the dispersion
  comparator above.
- **Dose-response:** report `ΔC` vs realized shuffled-fraction, covaried on `ℓ_dist` and `L`.

---

## 4. Inference (SEALED — one confirmatory test, no forking)

- **Unit:** homology cluster (≥80% id). **Confirmatory N = the post-dose census (~1345 clusters);** no
  run-time N choice. `N_MIN ≥ 80` clusters (≥160 needed for signed-mean MDE≤0.05 at `sd_between`=0.225;
  40 only for the ACC_UP=100 robustness subset, never to license a null).
- **ONE confirmatory test — bootstrap max-statistic over the co-primaries:** cluster (wild/multiplier)
  bootstrap generates the joint null; studentized statistics `z_mean = ΔC̄/se_boot` (**two-sided**) and
  `z_disp = excess_dispersion/se_boot` (**one-sided**, lower-bounded at 0); reject H₀ at **FWER α=0.05**
  iff `max(|z_mean|, z_disp⁺)` exceeds the bootstrap max-null 95th percentile. **CI method = BCa.**
  Secondary arm, dose-response, `Δf`-regression, sham, ACC_UP=100 robustness = explicitly secondary under
  hierarchical gatekeeping (interpreted only if the primary rejects).
- **Sampling stability:** census is default; if ever sub-sampled, multiple **independently seeded** draws,
  report stability. **Instability trigger (SEALED):** if across-draw SD of population `ΔC̄` > `0.5×SESOI =
  0.025`, route to the heterogeneity branch (§6) rather than a mean claim.

---

## 5. Gates / memorials (HARD preflight; run STOPS on failure unless noted)

- **`protected_bit_identical`**, **`length_position_preserved`**, **`whole_intron_dinuc_preserved`**
  (anchor GT/AG positions + whole-window multiset; unit-tested).
- **`perturbation_dose`** (on-path, can fail): per intron — distinct realizations (require all `N_SEED`),
  median altered ≥ 8, median normHam ≥ 0.25; **exclude near-identity introns before inference** ("all
  N_SEED distinct" operationalizes Cam's shuffle-entropy criterion).
- **`no_created_regulatory_signal`** — per intron, per seed: `Δ(max MaxEnt score3 over W)`,
  `Δ(max score5 over W)`, `Δ(max poly-Y run)`. **Sealed band (biologically-motivated tight cutoff): |Δmax
  score3| ≤ 2.0 bits AND |Δmax score5| ≤ 2.0 bits AND |Δ poly-Y run| ≤ 2 nt.** The **no-new-motif
  stratum** = introns whose ALL `N_SEED` primary shuffles stay in-band. This is a **per-intron stratifier
  (NOT a whole-run hard-stop), applied to the PRIMARY arm only** (the secondary arm's cryptic creation is
  characterized, not gated). **The material claim MUST persist in the no-new-motif stratum; report its
  size** (pilot flags this stratum may be a minority — Δmax-score3 p95=8.6 — see §8). MaxEnt has no BP
  model; ACC_UP=70 + the ACC_UP=100 robustness subset cover created-BP.
- **`distal_ood`** — distal-window mean masked-LM NLL over `W`, **per-intron 90th-pctile (p90) tail**,
  within the **empirical band = p95 of NATURAL distal-window NLL** (pilot ≈1.41 nats at ACC_UP=70),
  re-derived on the confirmatory naturals; run on **all four cells** `{I,I'}×{A⁺,A⁻}`, gating the
  doubly-perturbed **`(I',A⁻)`**. **Disposition:** per-intron exclusion; if the excluded fraction exceeds
  `1 − 0.95 = 5%` (i.e. shuffled in-band < natural in-band), escalate to **global redesign** (§6). `|Δf|`
  is a damage report only, never the OOD arbiter. (Pilot: shuffles ~95% in-band, median NLL unchanged.)
- **`sensitivity_onpath`** (SEALED PASS): plant a strong cryptic 3′SS (`"T"×18+"AGG"`) at a frozen mid-`W`
  position and require through-model `|ΔC_plant|` ≥ SESOI. Pilot: absmedian 0.087, 69% ≥ 0.05 ⇒ **PASS**
  — the pipeline detects a distal coupling change of the target size ⇒ the null is on-path licensable.
  (The DISTAL plant only; a proximal BPS/PPT perturbation would NOT license distal sensitivity.) In `--smoke`.
- **`census_MDE` (SEALED, null-licensing):** signed-mean `MDE_census`≈0.017 (post-dose) ≤ 0.05 AND
  **dispersion `MDE_disp_census`=0.004 ≤ 0.05** (pilot2) ⇒ BOTH co-primaries' nulls are licensable.
- **`identity_noop_exact`** (determinism/plumbing per R21, not on-path validity): identity ⇒ `ΔC==0` exact.
- **`native_coupling_present`**: baseline coupling of expected sign/scale on the **eligible subdomain**
  (recomputed there, not the full-domain +0.87).

`--smoke` = mechanics + dose + `no_created_regulatory_signal` + `sensitivity_onpath` on tiny-n.

---

## 6. Decision rule (SEALED; SESOI=0.05 nats, all thresholds numeric)

Eligible (scoped) set, §5 gates PASS:
- **Material effect:** the confirmatory max-stat test rejects at α=0.05 **AND** magnitude ≥ SESOI (signed
  `|ΔĈ|≥0.05`; dispersion excess-over-sham `≥0.05`) **AND** it persists in the **no-new-motif stratum**
  **AND** `ΔC_sham` CI ⊆ `[−0.05,+0.05]` **AND** it holds on the **ACC_UP=100 robustness subset** (else
  *provisional, not confidently distal*). → next prereg: localize where it enters.
- **Bounded null:** the entire signed `ΔC̄` CI ⊆ `[−0.05,+0.05]` **AND** dispersion excess-over-sham < 0.05
  (or indistinguishable from the sham floor) **AND** `sensitivity_onpath` PASSED **AND** achieved MDE ≤
  0.05. Report *"no modulation above [achieved MDE]"*, scoped to ACC_UP=70 / L≳107 arrangement → proceed to
  acceptor-proximal/PPT (IC2) and competitor (IC5) experiments.
- **Inconclusive:** neither (e.g. CI straddles ±0.05).
- **Health/OOD failure** (global, §5): no biological interpretation; redesign a milder perturbation.
- **Heterogeneous** (across-seed or across-draw instability past the sealed 0.025 trigger): characterize
  intron-specific dependence before any mean claim.
- **Cryptic-site localization (secondary, paired):** the per-intron paired contrast
  `ΔC_secondary − ΔC_primary`, tested with the cluster bootstrap (NOT a difference-of-significance across
  two unpaired analyses); per-arm `distal_ood`-clean. A significant paired difference licenses the
  cryptic-competition interpretation.

---

## 7. Scope, caveats, outputs, deferred

- **Behaviour-first**; circuit localization deferred to a contingent prereg.
- **Caveats:** (i) tests dinuc-composition-controlled ORDERING (incl. k≥3-mer/phase), not composition;
  (ii) scoped to eligible minimal introns **L ≳ 107** (ACC_UP=70; robustness subset ACC_UP=100 ⇒ L≳137) —
  intron-definition substrate, not deep-intronic/exon-definition; (iii) **C8 period-3 clock:** dinuc-shuffle
  scrambles triplet phase in `W`, so "positional address held fixed" is true only up to triplet-phase — a
  material `ΔC` may include a content×clock interaction; report `ΔC` vs distal triplet-periodicity change.
- **`ΔC_meaningful`=SESOI=0.05 nats:** above the non-partner-control floor; **always report the achieved MDE**
  and frame any null relative to it.
- **Adversarial verification is a MANUSCRIPT-STAGE requirement** for the full IC chain (design red-teams here
  do not replace it).
- **Outputs:** `scripts/splicebert/run_ic1_distal_intron_necessity.py`; JSON + manifest + preflight; eligible
  `L` + exclusion + no-new-motif-stratum sizes; `ΔC̄`, dispersion (excess-over-sham), `Δf⁺/Δf⁻`, dose-response,
  sham, per-arm + paired contrast, ACC_UP=100 robustness, seed/draw-stability — all with homology-cluster BCa CIs.
- **Provenance:** checkpoint sha256 `2ad91428…0149d`; MaxEnt pinned (`maxentpy 0.0.4`, `8b6d7b46…b7c00`);
  `ushuffle`-class anchored shuffler (unit-tested); all thresholds + seeds preregistered.

---

## 8. SEALED thresholds + the two flagged findings (for Cam's final sign-off)

| Parameter | SEALED | Pilot basis |
|---|---|---|
| ACC_UP primary / robustness | **70 / 100** | 2205 / 560 genes; both MDE ≤ 0.05; BP protected to −70 (no predictor) |
| N_SEED | **16** | dispersion primary; 8↔16 corr 0.90 |
| SESOI | **0.05 nats** | above non-partner floor (Cam) |
| dose floor | median-altered ≥8; normHam ≥0.25; all 16 distinct | ~61% pass ⇒ ~1345 genes |
| confirmatory test | bootstrap **max-stat** over studentized {z_mean 2-sided, z_disp 1-sided}, FWER α=0.05; **BCa** CI; **census** N | pins all v3 ORs |
| signed-mean MDE_census | ≈0.017 (post-dose) ≤ 0.05 | sd_between 0.225 |
| dispersion MDE_census | **0.004 ≤ 0.05** | excess_sd_between 0.071 (pilot2) |
| dispersion comparator | across-seed RMS(16), **excess over sham** | RMS_sham 0.059 ≈ ½ RMS_shuffle 0.112 |
| sham `A⁻'` | f-shift-matched PPT purine-swap (ref `a−10 'AA'`, 0.56×; confirmatory ≥0.8×); gate ΔC_sham CI⊆[−.05,.05] | mean 0.003 (pilot2) |
| on-path positive | plant `"T"×18+"AGG"` mid-W; \|ΔC_plant\|≥0.05 | absmedian 0.087, 69%≥.05 ⇒ PASS |
| no-new-motif band | \|Δmax score3\|≤2 & \|Δmax score5\|≤2 & \|Δ polyY\|≤2 (per-intron stratifier, primary arm) | shuffle Δmax-score3 p95=8.6 ⇒ stratum may be a minority |
| OOD | band=p95 natural distal NLL (~1.41); p90 tail; 4 cells; per-intron exclude, >5% ⇒ global redesign | shuffles ~95% in-band |
| homology | cluster ≈ gene | near-dup ≥80% = 0.5% |
| instability trigger | across-draw SD(ΔC̄) > 0.025 | 0.5×SESOI |

**FLAG 1 — created-cryptic-motif confound is larger than assumed.** Even with GT/AG positions fixed, the
anchored shuffle builds cryptic 3′SS/PPT (Δmax-score3 p95 = 8.6 bits; ΔpolyY p95 = 9). So the primary
"arrangement" claim is scoped to the **no-new-motif stratum**, which may be a **minority** of introns; the
paired anchored-vs-unconstrained arm difference is the principal cryptic-competition readout. *Option if
you want a larger clean stratum:* loosen the band, or add a tighter "conservative-scramble" arm that caps
cryptic-score change by construction.

**FLAG 2 — the sham under-matches f-shift** (0.56× at the piloted `a−10 'AA'`), so excess-over-sham is
mildly anti-conservative. Sealed as an **f-shift-matched procedure (≥0.8×)** for the confirmatory; if you
prefer a fixed sham, say so and I'll seal the fixed edit + accept the documented limitation.

**Power/feasibility note (TRAIN pilot; NOT the confirmatory claim):** the RAW across-seed dispersion at
ACC_UP=70 is ~0.11 and the *excess over sham* ~0.05; the actual decision quantity is the excess, not the
raw dispersion, and it is estimated only on the sealed val set at run time. No val data is read before freeze.
