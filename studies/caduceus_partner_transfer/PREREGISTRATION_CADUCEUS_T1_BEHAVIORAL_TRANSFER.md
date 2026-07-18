# Pre-registration — Caduceus-PS T1: does the partner-specific donor↔acceptor coupling behaviour transfer? (v2)

**Status: FROZEN-CANDIDATE v2 (5-lens red-team integrated). Awaiting Cam fork decisions + compute env → then FREEZE. DO NOT RUN until frozen.**
Supersedes `…DRAFT.md` (pre-red-team). The first transfer experiment (TRANSFER_GUIDE Phase 0 + Phase 1
behavioural qualification + the minimal Phase-2 adapter) to a **second, architecturally-contrasting GFM** —
Caduceus-PS, a BiMamba **state-space** MLM, maximally unlike SpliceBERT's post-LN attention BERT. Ports the
sealed SpliceBERT coupling assay; only the adapter changes. **Behaviour-first**: a positive result retires the
*"SpliceBERT-only behaviour"* concern, NOT the *"SpliceBERT-only mechanism"* concern (Cam) — mechanism
portability needs Phases 3–4 (SSM localization/intervention; no L3H1 / attention-comb / period-3 analog).

**v2 integrates the 5-lens red-team** (`REDTEAM_SYNTHESIS_5lens.md`; verdicts 5× fixable, 0 fatal). The six
HIGH fixes: **(1)** a distance-resolved **long-range transmission-curve** control so the flagship null is
identifiable (SSM leftward flow rides only the reverse Mamba pass — a real range-suppression failure mode);
**(2)** **adapter coordinate correctness** (Caduceus has **prefix_tokens=0**, a **trailing** [SEP], and
**left-padding** by default — the naive SpliceBERT port would silently read the neighbouring nucleotide) +
an **empirical roundtrip memorial under a mixed-length padded batch**; **(3)** effect-size floor on
**ρ_partner** (specificity), not raw C̄; **(4)** **τ sealed as a non-circular, sampling-stable formula**;
**(5)** **joint-power certification** on realized gene-n before any null is called informative; **(6)**
**acceptor-strength confound** stated honestly (distance-matched primary == SpliceBERT A3; score3-caliper
secondary with a pre-committed non-identifiable reading). **Carries every sealed-assay lesson** (B-1 both-strand
indexing, B-3 partner excluded from the null pool, real-base corruption, gene-clustered CIs, determinism).

**⚑ COMPUTE (probe-confirmed 2026-07-18):** Caduceus-PS's HF modeling code **unconditionally imports
`mamba_ssm`** (+ Triton RMSNorm, causal-conv1d) — **no CPU/MPS fallback** (config + single-nt tokenizer load
fine; the model does not). **This Mac cannot run it. A CUDA environment is required (Colab / Algoverse).**

**⚑ FORKS FOR CAM (red-team recommendations applied as defaults; confirm/override before freeze):**
(a) **τ philosophy** → *permuted-partner-null floor* (within-model, non-circular) [default]; alternatives:
fraction-of-ρ_SpliceBERT (cross-model — anchor validity disputed), or existence-only.
(b) **strength confound** → *distance-matched primary + score3-caliper secondary* [default]; alt: strength-primary.
(c) **compute** → *Colab-CUDA primary* [forced by the probe]; Algoverse-CUDA if preferred.
(d) **Caduceus-PH companion** for a biological strand control → *DROP* [default] (strand-specificity is
architecturally out of scope for PS and was never a SpliceBERT estimand).

---

## 0. INTENT (north star)

> **When Caduceus-PS's annotated paired acceptor is selectively corrupted, does its masked-donor prediction
> change more than under matched non-partner and non-donor interventions — i.e., does the partner-specific
> donor↔acceptor coupling behaviour replicate in a single-nucleotide, RC-equivariant state-space model whose
> architecture is maximally unlike SpliceBERT's?**

Behaviour-level, Phase 1. Qualifies → Phases 3–4 (SSM localization, separate preregs). Does-not-qualify
**with positive controls AND the long-range channel certified** → an honest architecture/objective null.
**Does not touch the SpliceBERT seal; held-out on a disjoint Caduceus confirmatory gene split.**

---

## 1. ESTIMANDS (frozen; scale-invariant; ported readout)

Native masked-LM readout, **identical operational definition to SpliceBERT** (`coupling.py`):
`margin_donor(seq) = logit(G) − max(logit A, logit C, logit T)` at the masked donor-G; `margin_acceptor`
at the acceptor AG's G. Corruption = real `CC` for the partner `AG` at `[a−2,a−1]` (never `[MASK]`).
Logit scale is model-specific ⇒ **all confirmatory quantities are scale-invariant**; SpliceBERT's
+0.87 / 37× / 74× / 0.05-nat are **not** imported.

- **Primary coupling** `C = margin_donor(clean) − margin_donor(partner-corrupt)`. `C>0` ⇒ corrupting the
  partner reduces donor confidence. **Existence: `C̄` gene-cluster mean, CI-excludes-0** (BCa, gene-clustered).
- **Anchors (same readout channel, always ≥0):**
  - `A_donor = margin_donor(clean) − margin_donor(donor-GT-destroyed)` — **LOCAL readout + recognition
    sensitivity only** (relabelled per E1; it does NOT certify the long-range channel).
  - `A_lr(d)` = **distance-resolved transmission** (§4): `margin_donor(clean) − margin_donor(downstream
    real-base block-scramble at offset d)`, the SSM downstream→donor channel anchor.
- **Effect sizes (scale-invariant; the intent-bearing quantities):**
  - **`ρ_partner = (C̄ − C̄_nonpartner) / A_donor`** — **PRIMARY meaningfulness estimand** (C1); the
    effect-size **floor `ρ_partner ≥ τ`** sits here, NOT on raw C̄.
  - `ρ_donor = (C̄ − C̄_nondonor) / A_donor`.
  - Report `ρ = C̄ / A_donor` and, where `A_lr` certifies the channel (§4), `ρ_lr = C̄ / A_lr(matched d)` as
    the range-appropriate normaliser. **Scope ρ within-model** unless `A_lr` licenses a cross-model reading.
  - **Ratio CIs by Fieller** (delta-method cross-check), **paired gene-cluster bootstrap** (resample genes
    once; recompute numerator and denominator means on that resample; ratio per replicate); report `A_donor` CV.

**Specificity contrasts (existence = CI-excludes-0 each; paired within-gene):**
2. **Partner specificity** `C̄ − C̄_nonpartner` — vs a **distance-matched, same-side (downstream),
   signed-distance-calipered** non-partner **real AG** with the partner `next_acceptor_pos` **excluded**
   (B-3). **Distance-matched only — IDENTICAL to SpliceBERT's A3** (fair transfer; do NOT hold Caduceus to a
   stricter bar). **The acceptor-strength confound (partner ≈ stronger/more-canonical AG) is an INHERITED
   limitation shared with the SpliceBERT +0.87, stated openly** (A1). **Secondary robustness:** a
   **MaxEnt-`score3` strength-caliper arm** (`|Δscore3| ≤ caliper`, maxentpy already installed) with a
   **pre-committed non-identifiable reading** — matched-contrast ≈ 0 under tight strength-matching ⇒
   *"non-identifiable given partnership≈strength collinearity," NOT "no transfer."*
3. **Donor specificity** `C̄ − C̄_nondonor` — coupling read at the true masked donor-G vs a **distance-matched
   OTHER REAL donor GT** (`other_gt`, per SpliceBERT A2 — **not** an arbitrary non-donor G, which floors
   trivially; A4); CI-excludes-0.

---

## 2. METHOD + ADAPTER (frozen; coordinate-correct per E2/E3/E6)

- **Assay:** import `gfmcd.splicing.coupling` **unchanged** (masked-donor margin, `AG→CC`, `destroy_donor`,
  `perturb_random`, eligibility, both-strand transcript-order indexing incl. B-1). No study assumptions in the adapter.
- **New adapter `src/gfmcd/adapters/caduceus.py`** — Caduceus's tokenizer is NOT SpliceBERT's; freeze:
  - **`prefix_tokens = 0`**; assert exactly **one TRAILING `[SEP]` (id 1)**; `nucleotide_to_token(p) = p`
    (NOT `p+1`). *(E2: `build_inputs_with_special_tokens` returns `ids + [SEP]`, CLS commented out.)*
  - **Force `padding_side='right'`** (Caduceus default is LEFT) — or assert equal-length / bs=1. *(E3: left-pad
    shifts every index under batching.)*
  - **Resolve A/C/G/T/[MASK] ids dynamically** via `convert_tokens_to_ids` at init (A=7,C=8,G=9,T=10,N=11,
    but never hard-code). *(E6.)*
  - Identity: checkpoint sha256 + HF revision + config sha256, tokenizer sha, objective (MLM), context length.
  - Outputs: native per-nucleotide MLM logits with individually-decodable A/C/G/T. Invariants:
    shape/checksum/device/precision/no-mutation. Full run manifest.
- **Context = 1024 nt windows** (match SpliceBERT information content; Caduceus supports 131k, constrained).
- **Cohort:** same eligibility as SpliceBERT (internal exon; in-window intron-mate acceptor; canonical GT
  donor; gap ∈ [60,150]; **both strands**). **Fresh Caduceus discovery / confirmatory CHROMOSOME split**
  (SpliceBERT's chr8/9 seal does not bind Caduceus); **prefer ≥3–4 confirmatory chromosomes** (chromosome
  jackknife). Deterministic (`sorted(set())` + pinned `PYTHONHASHSEED`).

---

## 3. MEMORIALS (HARD preflight; on the measurement path; each CAN fail; STOP on failure)

- **`single_nt_tokenization`** — exactly one token per nucleotide, `prefix_tokens=0`, trailing `[SEP]`;
  `unk` never on `ACGTN`.
- **`coordinate_roundtrip` (E2/E3 — EMPIRICAL, not arithmetic):** on a **deliberately mixed-length,
  right-padded batch**, mask a **known** canonical donor-G and confirm the argmax/margin lands at the intended
  base (not a neighbour). *Fails* → coordinate map wrong (the off-by-one / left-pad trap).
- **`coordinate_sensitive` (D5):** `margin_donor` is high AT the donor index and low at **±1 nt** — catches
  off-by-one that RC symmetry could hide.
- **`margin_readout_decodable`** — A/C/G/T individually decodable; on a known canonical donor, `margin_donor(clean) > 0`.
- **`mlm_health` (D3):** the `[MASK]` id is used; mask-fill top-1 ≫ 25% on random `ACGT`; objective is MLM
  (port `coupling.py::mlm_health`).
- **`real_base_corruption`** — every perturbation substitutes real bases; `[MASK]` never enters a corrupted construct.
- **`donor_destruction_anchor`** — `A_donor > 0`, gene-clustered CI-excludes-0 (also the §4 sensitivity plant).
- **`long_range_transmission` (E1 — NEW, on-path):** the reverse-Mamba downstream→donor channel is alive at
  the biological range (§4). *Fails-to-certify* → the null is an INSTRUMENT-RANGE LIMIT, not biology.
- **`nonpartner_pool_excludes_partner`** (B-3) — `next_acceptor_pos ∉` non-partner pool for every example.
- **`both_strands_present`** (B-1) — **minus-strand share ∈ [0.3, 0.7]** (D6), not merely > 0.
- **`rc_plumbing_consistency` (⚠ ARCHITECTURE-FORCED PLUMBING; A7/E5):** feeding the **nucleotide-level RC**
  (rebuilt + re-tokenised, trailing [SEP] re-appended) reproduces the readout at the **complemented base,
  mirrored index**, compared over **nucleotide positions only**, small tol. Caduceus-PS is RC-*equivariant*
  ⇒ this is guaranteed by construction — it validates adapter RC/coordinate handling, **NOT** biology (§5).
- **`determinism`** — identical cohort + results across `PYTHONHASHSEED ∈ {0,1,2,17}`.
- **`adapter_invariants`** — shape/checksum/device/precision/no-mutation on every forward.

`--smoke` = mechanics + tokenization + `coordinate_roundtrip` + `coordinate_sensitive` +
`margin_readout_decodable` + `mlm_health` + `donor_destruction_anchor` + `nonpartner_pool_excludes_partner`
+ `both_strands_present` + `determinism`, tiny-n, confirmatory sealed.

---

## 4. CALIBRATION + LONG-RANGE CERTIFICATION + POWER (hard gates)

- **Planted-positive (sensitivity):** `A_donor` detected (CI-excludes-0, large) — the model's own donor
  recognition, same readout channel. Fails → underpowered → STOP. *(Local channel only — see below.)*
- **LONG-RANGE TRANSMISSION CURVE (E1 — the fix that makes the null interpretable):** fix the masked donor-G
  readout; apply a **strong generic real-base block perturbation** (scramble/N a 10–20 nt window) at sliding
  **downstream** offset `d ∈ {~10, 30, 60, 100, 150, 200} nt`; measure `|Δmargin_donor(d)|`, gene-clustered CI.
  - **(i) VISIBILITY:** adjacent block (`d≈10`) ⇒ `Δ` large.
  - **(ii) CERTIFICATION:** at `d ∈ [60,150]` (the biological coupling range), `Δ` CI-excludes-0 AND
    non-negligible vs near-distance ⇒ the reverse Mamba pass carries downstream content to the required range.
  - Surgical primary variant: a non-specific 10 nt scramble **at the acceptor locus** (exact channel, minus
    specificity). **Wired to §6.**
- **Null / specificity (FPR ≤ 0.05):**
  1. **Non-site null** — `perturb_random` at a non-site: coupling ≈ 0 (CI ⊆ small bound / includes 0).
  2. **Permuted-partner null** — reassign a random **non-partner** acceptor as nominal "partner", **matched on
     distance + same-side + local composition** (D2/A5/A6); partner-specific structure must **vanish** (≤ FPR).
     **Gene-cluster BLOCK permutation** (B7/D4); the significance threshold is calibrated to this null on a
     **calibrate/test split pinned to DISCOVERY** (out-of-sample FPR).
- **JOINT-POWER CERTIFICATION (B1 — licenses the null):** on discovery, report (1) realized **independent
  GENE count** + top-gene exon share (discovery & confirmatory separately); (2) **per-arm MDE** at 80–90%
  power on gene-n; (3) **joint power of the 4-arm conjunction** by gene-bootstrap simulation at anticipated ρ.
  **A null is "informative" only if joint power ≥ ~0.8**; else widen cohort (relax gap as a LABELED
  robustness arm / add chromosomes) or **demote the strand-subgroup gate to a reported diagnostic**.

---

## 5. IDENTIFIABILITY / common support (frozen)

- **Distance-decay confound** (sealed B-4): blocked by distance-matching the non-partner (estimand 2) and the
  distance-matched other-real-donor (estimand 3). **Predefine overlap on an UNSELECTED pool** (partner vs
  non-partner distance distributions overlap); honest non-identifiable outcome if partnership⊥distance fails.
- **Acceptor-STRENGTH confound (A1 — stated, not hidden):** the distance-matched non-partner is NOT
  strength-matched; the partner is a stronger/more-canonical AG. Estimand-2-positive is therefore
  strength-confounded — **an INHERITED limitation shared with the SpliceBERT +0.87**. The **score3-caliper
  secondary arm** probes it, with the pre-committed non-identifiable reading (§1).
- **Common support / Berkson (A3):** the paired diff is computed only on exons WITH a valid matched control;
  exons lacking one may be the isolated largest-coupling partners. **Per-example caliper-satisfiability check;
  report retention fraction; pre-declared dropped-vs-retained C̄ balance check** — if they differ, report
  Berkson bias.
- **RC-equivariance (declared non-identifiable AS biology for PS):** PS ties the strands by construction ⇒
  forward-vs-RC is architecture-forced ⇒ demoted to the `rc_plumbing_consistency` memorial (§3); NOT strand
  specificity. An independent biological strand control would need Caduceus-PH — **fork (d), default DROP**.

---

## 6. DECISION RULE (frozen; CI-classified; conjunction is an IUT)

Evaluated on the **untouched confirmatory cohort** after §3–4 PASS and τ frozen from discovery. The 4-arm
conjunction is an **intersection-union test (IUT) — FWER-conservative by construction; NO Bonferroni/omnibus**
(B5; differs from IC2's disjunctive omnibus — correct here because the claim IS a conjunction).

- **QUALIFIES-FOR-PHASE-3 (behaviour-only; mechanism unproven)** iff ALL hold (existence = C̄ CI-excludes-0;
  the tautological "ρ CI-excludes-0" gate is DROPPED, B4/C7):
  (a) `C̄` CI-excludes-0, expected sign; (b) partner-specificity `C̄−C̄_nonpartner` CI-excludes-0;
  (c) donor-specificity `C̄−C̄_nondonor` CI-excludes-0; (d) **`ρ_partner ≥ τ`** (frozen from discovery — the
  meaningfulness FLOOR, on the specificity effect size); (e) stability — same sign across the gene bootstrap
  (strand-subgroup concordance is a **reported diagnostic** unless power permits it as a gate, B1);
  (f) all memorials + calibration + **long-range certification** PASS.
  → promote to Phase 3–4 (SSM localization/necessity/mediation; compare *computational motifs*, not head numbers).
- **DOES NOT QUALIFY — honest architecture/objective null:** positive controls PASS, **long-range channel
  CERTIFIED healthy@60–150 nt**, joint power ≥ ~0.8, but coupling/specificity CI includes 0 or `ρ_partner < τ`.
  A real finding: the behaviour does not transfer to this SSM/objective.
- **INSTRUMENT-RANGE LIMIT — NOT a biological null (E1):** coupling-null but the **long-range transmission is
  DEAD at 60–150 nt** (or a positive control / coordinate memorial fails) → the reverse-pass channel cannot
  carry the signal → re-run/redesign, **draw no biological conclusion**.
- **UNDER-POWERED — not informative:** joint power < ~0.8 on realized gene-n → widen cohort or report as
  inconclusive (never as an architecture null).

---

## 7. HELD-OUT + τ (frozen)

Selection occurs (τ, thresholds) ⇒ held-out mandatory. **Discovery cohort:** set/seal τ, certify power/MDE +
joint power, freeze. **Confirmatory cohort** (disjoint, ≥3–4 untouched chromosomes): opened once, after freeze.

**τ is SEALED as a FORMULA before discovery ρ is unsealed (C2/B2 — never the discovery point estimate):**
- **Default (fork a): permuted-partner-null floor** — `τ` = the **upper CI of `ρ_partner` under the
  gene-cluster permuted-partner null** (beat the model's OWN no-partner-identity floor; within-model,
  non-circular, no ρ_SpliceBERT needed).
- **Sampling-stability guard:** if any data-derived component is used, compute `ρ_partner` across **multiple
  independently-seeded discovery draws** and set τ **below the across-seed LOWER envelope**; log the seeded
  spread in the freeze manifest (the R22 discipline — held-out ≠ sampling stability).

---

## 8. GATES + COMPUTE

Global **PASS = AND(memorials §3, calibration + long-range certification + joint-power §4)**; STOP on any failure.
- **Feasibility preflight (Phase-2 HARD gate; DONE 2026-07-18):** config + single-nt tokenizer load; **model
  load FAILS on `mamba_ssm` (no CPU/MPS fallback)** ⇒ **Colab/Algoverse CUDA is PRIMARY compute, not a
  fallback** (E4). **No MPS forcing**; CPU only with logit-for-logit CUDA validation (disabling
  fused_add_norm/rms_norm silently changes normalisation ⇒ not the model's logits — forbidden).
- On CUDA: re-run the preflight (load + forward + `coordinate_roundtrip` + `margin_donor(clean)>0`) before smoke.

---

## 9. SCOPE / provenance

- **Reaches:** Behaviour level, TRANSFER_GUIDE **Phase 1** on a second GFM.
- **Does NOT establish:** mechanism transfer / method portability — needs Phases 3–4 with SSM-appropriate
  interventions (BiMamba state/channel ablation), **no** attention-head / L3H1 / period-3 analog.
- **Inherited limits:** long introns untestable (gap ≤ 150, 1024 nt); one checkpoint; acceptor-strength
  confound (shared with SpliceBERT); **paralog/homology** scope caveat (chromosome jackknife mitigates; B8);
  comparison to SpliceBERT at the level of computational motifs only.
- **Provenance:** `caduceus-ps_seqlen-131k_d_model-256_n_layer-16`, HF revision + config sha256 + tokenizer
  sha recorded before confirmatory unseal; cohort split + seeds + pinned `PYTHONHASHSEED`; run manifest.
  Study dir `studies/caduceus_partner_transfer/`.
