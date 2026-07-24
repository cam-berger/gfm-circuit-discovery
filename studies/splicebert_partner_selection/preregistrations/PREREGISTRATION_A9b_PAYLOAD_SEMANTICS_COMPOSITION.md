# Pre-registration — A9b: is the payload LOCAL SPLICE-SITE STRENGTH *net of bulk composition*? (donor-graph 13b)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN (2026-07-17), post-estimand-red-team.** Supersedes A9's composition handling (V4 was
NON-IDENTIFIABLE — 0 matched pairs — so the confound was OPEN). A **5-lens estimand red-team** (before
any build) established the honest framing and forced this spec. Reuses the A9 interchange + value-path
metric + through-model calibration; adds a **composition adjustment set**, a **cross-fitted double-ML**
estimator, **two planted-CONFOUND gates**, and a **dyadic bootstrap**. No new circuit kernel.

> **Amendment (2026-07-17, pre-run, no val data seen).** Re-scoped from the draft's "net of composition"
> to **net of BULK/FREQUENCY composition** (GC, pyrimidine fraction, di/tri-mer frequency, position).
> The red-team established that splice **strength is DEFINITIONALLY INSEPARABLE from the core motif it is
> computed from** (V2b is a deterministic function of the 4 core acceptor nt) — so "strength net of *all*
> composition" is non-identifiable by construction. A9b therefore tests the identifiable question:
> **strength vs generic bulk composition**, and treats core-motif entanglement as intrinsic, not a
> confound to be removed. Primary estimator changed from linear residualization to **cross-fitted DML**
> (nonlinear confounds); added the two confound gates and the dyadic bootstrap; decision is
> **CI-excludes-zero**.

---

## 0. INTENT (north star — re-read at every gate)

> Does splice **STRENGTH**, and not generic **BULK composition**, name the L3H1 acceptor payload? —
> i.e. does strength predict the interchange transfer **net of bulk/frequency composition**, on held-out
> genes AND within composition-matched sets, with adequate common support to identify it? This is the
> bridge from a *causal circuit* (A6–A12) to an *interpretable computation*. **Honest fallback:** if
> strength and bulk composition are collinear / support is thin → report **composite (non-identifiable)**.

The clean alternative a reviewer raises: "the transfer tracks generic local composition (GC/PPT/k-mer),
not splice *strength*." A9b separates the two rather than assuming them distinct.

---

## 1. Interchange, transfer, strength (reuse A9 exactly)

- **Interchange `i←j`** / transfer `T(i←j)` exactly as A9 (deliver j's clean payload through i's clean
  address; A8 reconstruct-inject; `value_linearity` exact). **Signed transfer `S = T − 1`.**
- **Strength (ordinal):** **V2b** = frozen local-4-token MaxEnt contribution (model-independent, PRIMARY);
  **V3** = model's own acceptor logit (secondary/circular). `ΔV = V(j) − V(i)` per contrastive pair.
- Discovery / held-out **gene** split (D = every other gene). Primary decision on **H**.

---

## 2. Bulk-composition adjustment set `Z` (predefined, frozen) and the ADJUSTED ESTIMAND

**`Z(x)`** over the payload region `[a−20 .. a+5]`, all model-independent, computed on the sequence:
- `GC` fraction (region); `PY` pyrimidine (C+T) fraction over the PPT window `[a−20 .. a−3]`;
- `POS` = acceptor absolute position `a` (positional-composition proxy);
- **di-mer** 16-freq ⊕ **3-mer** 64-freq (region) → reduced to the **top-`K_z` principal components**
  (PCA **fit on the TRAIN split only**, frozen `K_z` = smallest k with ≥90% variance; **no val leakage**).

`Z` is deliberately **low-dimensional** (`dim(Z) = 3 + K_z`, target ≤ ~12) to keep `n_pairs ≳ 10·dim(Z)`.
`ΔZ = Z(j) − Z(i)` per pair.

**ADJUSTED ESTIMAND (frozen).** For each strength variable, the composition-adjusted effect is the
partialled slope `β_str` in

    S(i←j)  =  β_str · ΔV_str  +  g(ΔZ)  +  ε

estimated by **cross-fitted double-ML (Robinson partialling)**: with `K=5` **gene-clustered** folds, fit
nuisances `m_S(ΔZ)=E[S|ΔZ]` and `m_V(ΔZ)=E[ΔV|ΔZ]` out-of-fold with a **shallow gradient-boosted
regressor**; then `β_str = Σ S̃·Ṽ / Σ Ṽ²` on the residuals `S̃=S−m_S, Ṽ=ΔV−m_V`. `β_str` = the strength
effect **holding bulk composition fixed**. **Secondary/robustness:** linear-Ridge residualization
`β_str^lin`; must agree in sign. Sign must match the A9 held-out direction (V2b < 0, V3 > 0).

---

## 3. Common support / identifiability (verify BEFORE interpreting) `⟐ RED-TEAM`

`β_str` is interpretable only with adequate overlap (strength and bulk composition are correlated):
- **Collinearity**: **cross-validated R²** of `ΔV_str ~ ΔZ` (5-fold, adjusted for small-n inflation).
  **R² ≥ 0.85 → NON-IDENTIFIABLE (composite)**; do not claim strength.
- **Common support**: count contrastive pairs with **large `|ΔV_str|`** (top tercile) AND **small
  `‖ΔZ‖`** (bottom tercile) — "strength varies, bulk composition matched". Require **≥ `--min-support`
  (default 40)** in **D and H each**; else NON-IDENTIFIABLE.
- Report both regardless of the verdict.

---

## 4. Matched-set variant + held-out (both required)

- **DML (primary):** `β_str` (§2) with dyadic-bootstrap CI on **held-out H**.
- **Matched (secondary):** within pairs whose `‖ΔZ‖` is below a frozen caliper (bottom tercile) AND
  `|ΔV_str|` large — the slope of `S` on `ΔV_str`, on **H**. Strength must hold in BOTH.
- **Out-performs bulk composition:** the composition-only effect (`S ~ ΔZ`, `ΔV` omitted) must NOT
  explain `β_str` away (that is exactly what the DML partialling and confound-gate A test).

---

## 5. Calibration gates (HARD preflight; run STOPS on failure)

Preserved from A9 (through-model), plus two NEW confound gates (semi-synthetic, on the estimator):
- **Planted-positive (A9, sensitivity):** synthetic Vplant-scaled recovering payload delivered through
  the interchange; the (now composition-adjusted) pipeline MUST detect it (`β_str` significant, signed).
- **Permuted-label FPR (A9, specificity):** permute strength labels; `β_str` significance calibrated to
  the permutation null (out-of-sample split), **FPR ≤ 0.05**.
- **Confound-gate A (composition-only → NULL):** synthetic outcome `S* = f(ΔZ) + noise` (NO `ΔV` term,
  β_true=0), built at the observed ΔV–ΔZ collinearity. The DML `β_str` CI **must cover 0** (the
  adjustment does not misattribute composition to strength). Tests *specificity of the adjustment*.
- **Confound-gate B (collinear-strength → RECOVERED):** synthetic outcome `S* = β0·ΔV + f(ΔZ) + noise`,
  known `β0`, at the observed collinearity. The DML `β_str` CI **must cover β0** (the adjustment does not
  over-correct away real, composition-collinear strength). Tests *sensitivity of the adjustment*.
- **Mechanics:** `value_linearity` exact (<1e-4); `self_interchange_identity` (T(i←i)=1);
  `usable_value_path` (fraction with denom>0.05 ≥ 0.4).

`--smoke` = mechanics + all four calibration gates on tiny-n (val sealed; through-model gates relaxed to
code-path only, the two confound gates run for real since they are cheap semi-synthetic).

---

## 6. Decision (frozen)

On **held-out H**, common support adequate (§3), all §5 gates PASS:
- **STRENGTH names the payload NET OF BULK COMPOSITION** iff `β_str` (V2b) **CI excludes 0**, correct
  sign, **and** the matched-set slope (§4) CI excludes 0 — **and** collinearity R² < 0.85 and support ≥ min.
- **COMPOSITE / inseparable** iff collinearity R² ≥ 0.85 **or** common support < min → payload is a
  **splice-strength / bulk-composition composite**; strength not uniquely identified (honest; the
  core-motif entanglement is intrinsic per §0).
- **GENERIC BULK COMPOSITION** iff the composition-only fit (`S~ΔZ`) is significant while `β_str` is null.
- **UNKNOWN** iff none survives.

No subspace fit; `U_info` projected-onto only (C16). The superseded A9 analysis is NOT re-verified — the
adversarial verification (task "(b)") runs against THIS final A9b spec after the confirmatory run.

---

## 7. Outputs, reflection

- `scripts/splicebert/run_a9b_payload_semantics_composition.py` (reuses A9 interchange + calibration;
  adds `Z`, DML, confound gates, dyadic bootstrap, overlap); JSON + manifest + preflight.
- **REFLECTION:** `β_str` (V2b primary, V3 secondary) DML + linear + matched-set, held-out, with dyadic
  CIs; collinearity R² and common-support counts; the verdict; all four calibration gates; the honest
  **composite** outcome if it fires; and a RE-ANCHOR line confirming the result answers §0 (or scopes down).

**Provenance.** Checkpoint sha256 `2ad91428…0149d`; interchange/metric per A9/C30; MaxEnt pinned
(`maxentpy 0.0.4`, tables sha256 `8b6d7b46…b7c00`); seeds logged; DML nuisance = sklearn 1.9 GBR (frozen).
