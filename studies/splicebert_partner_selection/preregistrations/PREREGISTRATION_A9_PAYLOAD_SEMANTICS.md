# Pre-registration — A9: semantic identity of the acceptor payload (donor-graph experiment 13)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** External scorer resolved (Cam chose **A: canonical MaxEntScan**). All rules frozen
per Cam's conditional-go red-lines (2026-07-16). Matched **interchange interventions**
(causal-abstraction test, Geiger et al. 2021 arXiv:2106.02997), reusing the A8 value-path
reconstruct-inject. Freeze the scorer, then smoke → confirmatory.

---

## 0. Why this document exists

A8/C29 localized the payload generator (serial L0→L1→MLP2) but did not name **what the payload
encodes**. A9 asks whether a **preregistered, frozen, sequence-level variable** names it — such that
**equivalence-preserving** interchange leaves the computation intact and **contrastive** interchange
transfers the effect **sign-correctly, reciprocally, monotonically, on held-out genes**. The R20 lesson
(name the variable, not the component). `U_info` is **not** redefined and **no subspace is fit**; the
interchange is on the **full** acceptor payload.

---

## 1. Interchange unit, metric, STAGED readouts

- **Payload unit** = the acceptor 4-token window pre-L3H1 residual `r3_ACC` (A8). Full residual.
- **Cross-example interchange `i ← j`**: deliver j's payload through i's **CLEAN address** `A⁺_i`
  (A8 reconstruct-inject; post-LN linearity): `δr = r3_j[ACC_j] − r3^corr_i[ACC_i]`,
  `o_d = A⁺_i·V⁻_i + Σ_k A⁺_i[d,k] W_V δr_k`, inject on i's corrupt run. Address is i's throughout.
- **Transfer** `T(i←j) = (M'(i←j) − M'(i←∅)) / (M'(i←i) − M'(i←∅))`; 1 = i's own payload, 0 = corrupt.
- **STAGED readouts (report all three; scope a claim to the stage that transfers):**
  **(1)** L3H1 value/output (delivered `‖W_V δr‖`-normalised margin); **(2)** donor-side **frozen**
  `U_info` component (project delivered payload onto the C16 basis — no refit); **(3)** final donor
  margin. `value → U_info → margin` is a mediation chain.

---

## 2. Candidate variables (preregistered, FROZEN — no readout leakage)

- **V1 acceptor identity** (categorical): acceptor-window nt string `[a−3..a]`.
- **V2 local acceptor strength** (ordinal): **MaxEntScan score3** (Yeo–Burge 2004, 23-nt 3′SS
  `[a−20..a+2]`), computed two ways per the red-line: **(2a)** the ordinary 23-nt score; **(2b)** a
  **local counterfactual Δ**: replace only the 4-token acceptor window, flanks fixed, recompute → the
  window's contribution (matches the payload's extent). **Scorer PINNED:** `maxentpy==0.0.4` (git+github kepbod/maxentpy), tables sha256 `8b6d7b46f81788fb0710314867e3233e7f9b7f061b809bbdb11bb120422b7c00`, test `score3('ttccaaacgaacttttgtagGGA')=2.887` (Yeo–Burge).
- **V3 frozen splice-site score** (ordinal, **secondary**): the model's own SpliceBERT acceptor-position
  logit at the site, computed once and frozen. Distinct claim from V2: *model's belief*, not external
  strength; **circular, cannot alone establish biological meaning.**
- **V4 local motif composition** (vector): 3-mer frequency over `[a−15..a+5]`. **Not a "must-fail"
  null** — composition could genuinely be represented; the named variable must **outperform** it in
  matched comparisons (§4).

**Scorer resolved (Cam → A):** canonical MaxEntScan via `maxentpy 0.0.4` (installed from
github kepbod/maxentpy; network was up, the package is off-PyPI). The coarse
`address.acceptor_motif_score` is a **covariate only**, never the primary scorer.

---

## 3. Matched pairs — frozen calipers, reciprocal, common-support

Cross-gene pairs `(i,j)` (different genes; discovery/held-out gene split, seed 0). **Calipers frozen
before any outcome is observed.** The two sets are constructed **differently** (red-line 2):

- **Equivalence set `E_V`**: `V(i)≈V(j)` **tightly** (within caliper), but the **competing** candidates
  deliberately **allowed to differ** (else preservation is just "two near-identical sequences").
- **Contrastive set `C_V`**: `V(i)` vs `V(j)` **substantially different** (≥ preregistered `|ΔV|`
  bins), while the **competing** candidates **tightly matched**.

**Reciprocal** swaps (`i←j` and `j←i`); **cap source reuse** (each example a source ≤ `--max-src`
times); disjoint gene splits; **report common-support N** per set. **If no pairs satisfy the frozen
calipers → the candidate is NON-IDENTIFIABLE, not negative.**

---

## 3.5 Energy stratification (PRIMARY analysis) + null controls

Cross-example swaps vary in perturbation size, a nonlinear confound dominated by off-manifold swaps, so
**stratify (not just regress)** on **both** `‖δr‖` **and** `‖W_V δr‖` (the latter is closer to what L3H1
receives). **Within each energy stratum**, compare candidate-consistent vs candidate-inconsistent swaps.
Regression adjustment on the two energies is **secondary**. **Controls:** (i) same-energy
**label-permutation** (permute V-labels within stratum → FPR); (ii) **random-direction** swap (a random
same-energy `δr` → the distance-only baseline).

---

## 4. Decision standards (frozen; two-sided, reciprocal, monotonic)

On **held-out** genes, energy-stratified:

- **Equivalence** (two-sided, red-line 4): stratum-pooled `T(E_V)` fidelity entirely within
  **`[0.80, 1.20]`** (overshoot is also non-equivalence) at the margin **and** the `U_info` stage.
- **Contrastive**: effect **sign-correct**, **reciprocal** (`i←j` and `j←i` opposite-signed), and
  **monotonic** across preregistered `|ΔV|` bins, CI excluding the same-energy permutation null.
- **Outperform composition**: the named variable's contrastive effect must exceed V4's in **matched**
  (same-energy, competing-controlled) comparisons.
- **Composite fallback**: if several correlated candidates pass but cannot be dissociated (competing
  calipers can't separate them), report a **"score-like composite"**, not a unique variable.
- **Honest null**: if all candidates fail on held-out → **localized (A8) but semantic variable
  unknown**; no subspace fit to rescue.
- **Stage-scoping**: if a candidate transfers only at value **or** only through `U_info` **or** only at
  the margin, scope the claim to that stage.

---

## 4.5 Calibration gate (ESSENTIAL — the memorials test mechanics, not the classifier)

Given the project's history, calibrate the **whole analysis+decision pipeline** before trusting it:

- **Planted-variable positive control (sensitivity):** construct a synthetic payload edit that carries a
  KNOWN variable (e.g. add `c·(strong−weak MaxEnt direction)` to `δr`); the pipeline **must** identify
  it (equivalence + sign-correct contrastive). Failure ⇒ the classifier is underpowered, STOP.
- **Permuted-label FPR (specificity):** run the full decision on **permuted** V-labels; the
  false-positive rate must be **≤ 0.05**. Failure ⇒ the standard is too loose, STOP.

Both are preflight-gated (`--smoke` runs mechanics memorials; the calibration controls run in the
confirmatory and gate the verdict).

---

## 5. Gates and falsifiers (mechanics)

`value_linearity` (2.7e-6, A8); `self_interchange_identity` (`M'(i←i)` reproduces i's clean-payload
margin < 1e-4); `address_fixed` (injected row = `A⁺_i` for all swaps); `positive_control`
(`T(i←i)=1`, `T(i←∅)=0`); `frozen_no_leakage` (scorer params fit on TRAIN, disjoint from val; scorer
sha256 recorded); `both_genes_split`, `both_strands_present`, `checkpoint_recorded`.

---

## 6. Outputs, reflection, outcome table

- `scripts/splicebert/run_a9_payload_semantics.py` (+ a pinned frozen-scorer helper); JSON + manifest +
  preflight. **REFLECTION:** per-variable equivalence/contrastive at all 3 stages, energy-stratified,
  held-out, with the permutation-FPR and planted-positive calibration; the common-support Ns; the
  composition comparison; stage-scoping; the honest null if all fail.

**Outcome table (Cam's, frozen):** MaxEnt+logit pass → externally-grounded splice-strength; only MaxEnt
→ biological motif strength (not the model's current belief); only logit → internal acceptor confidence
(biologically unnamed); composition wins → generic local-sequence; multiple inseparable → composite;
all fail → semantic identity unknown.

**Scope.** Behaviour + mediation, value-path (address held clean), full-payload interchange, held-out.
`U_info` unchanged. Address geometry is the A4-redux sibling, out of scope.

**Provenance.** Checkpoint sha256 `2ad91428…0149d`; payload/metric per A8/C29; `U_info` per C16;
MaxEntScan per Yeo–Burge 2004 (DOI 10.1089/1066527041410418) once pinned.
