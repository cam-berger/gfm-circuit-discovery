# Pre-registration DRAFT — Caduceus-PS T1: does the partner-specific donor↔acceptor coupling behaviour transfer?

**Status: SUPERSEDED (2026-07-18) by `PREREGISTRATION_CADUCEUS_T1_BEHAVIORAL_TRANSFER.md` v2 — the 5-lens
red-team is integrated there. This DRAFT is the pre-red-team record; do not run it.**

---

**(original DRAFT status:) DRAFT for red-team → integrate → FREEZE. DO NOT RUN until frozen.** The first transfer
experiment (TRANSFER_GUIDE Phase 0 + Phase 1 "behavioral qualification", plus the minimal Phase 2 adapter
needed to run it) to a **second, architecturally-contrasting GFM**. Ports the sealed SpliceBERT coupling
assay verbatim; only the model adapter changes. **Behaviour-first**: this gate is the FIRST milestone of
transfer, not the finish line — a positive behaviour retires the *"SpliceBERT-only behaviour"* concern but
**NOT** the *"SpliceBERT-only mechanism"* concern (per Cam). Method portability is demonstrated only once
model #2 progresses past behaviour into causal localization + intervention (Phases 3–4), which for a
BiMamba SSM must be redesigned from scratch — there is no L3H1 / attention-comb / period-3 analog.

Carries every sealed-assay lesson: B-1 (both-strand transcript-order indexing), B-3 (partner excluded from
the non-partner null pool), unconditional≠conditional coupling, real-base (never `[MASK]`) corruption,
gene-clustered CIs, deterministic cohort (`sorted(set())` + pinned `PYTHONHASHSEED` + determinism test).

---

## 0. INTENT (north star)

> **When Caduceus-PS's annotated paired acceptor is selectively corrupted, does it change its masked-donor
> prediction more than under matched non-partner and non-donor interventions — i.e., does the
> partner-specific donor↔acceptor coupling behaviour replicate in a single-nucleotide, RC-equivariant,
> state-space model whose architecture is maximally unlike SpliceBERT's post-LN attention BERT?**

Model-native observable that would show it: the masked-LM donor-G margin drops selectively when the
*specific* paired acceptor (not a distance/sequence-matched non-partner, not a non-donor site) is destroyed.
This is **behaviour-level, Phase 1**. If it qualifies → Phases 3–4 (SSM localization/necessity/mediation,
separate preregs). If it does not (with positive controls intact) → an honest architecture/objective null.

---

## 1. ESTIMAND (frozen)

Native masked-LM readout, **identical operational definition to SpliceBERT, ported from `coupling.py`**:
`margin_donor(seq) = logit(G) − max(logit A, logit C, logit T)` at the masked donor-G position.
`margin_acceptor` analogously at the acceptor AG's G (`[a−1]`). Corruption = substitute real `CC` for the
partner acceptor `AG` at `[a−2, a−1]` (never `[MASK]`; `CC` cannot re-create an AG).

- **Primary coupling** `C = margin_donor(clean) − margin_donor(paired-acceptor-corrupt)`.
  Units: **Caduceus MLM logit-margin (model-specific scale)**. Sign convention: **C > 0** ⇒ corrupting the
  partner *reduces* donor confidence (the coupling direction). Reference point: `C = 0` (no coupling).

Because logit scale is model-specific, **the confirmatory quantities are all scale-invariant** — we do NOT
import SpliceBERT's +0.87 / 37× / 74× / 0.05-nat SESOI:

1. **Signed coupling** `C̄` — gene-cluster mean; existence + sign via **CI-excludes-0**.
2. **Partner specificity** `C̄ − C̄_nonpartner` — paired, vs a **distance- and local-sequence-matched
   non-partner real AG** (partner `next_acceptor_pos` **excluded** from the pool, B-3); CI-excludes-0.
3. **Donor specificity** `C̄ − C̄_nondonor` — the coupling read at the true masked donor-G vs a
   distance-matched masked **non-donor G**; CI-excludes-0.
4. **Model-relative effect size** `ρ = C̄ / A_donor`, where the **donor-recognition anchor**
   `A_donor = margin_donor(clean) − margin_donor(donor-GT-destroyed)` is Caduceus's own "how strongly do I
   insist on G at a real donor" magnitude (same readout channel, always positive). `ρ` is the
   **logit-scale-invariant** effect size; ratio-of-gene-means with a bootstrap CI (not mean-of-ratios).

**`C̄`/`A_donor` are cohort gene-clustered means; `ρ` is the ratio of those means.** The "predeclared
minimum effect within Caduceus" is `ρ ≥ τ`, with **τ set on the discovery split and sealed** (see §7).

---

## 2. METHOD (frozen; reuse validated machinery)

- **Assay:** import `gfmcd.splicing.coupling` **unchanged** (masked-donor margin, `AG→CC`, `destroy_donor`,
  `perturb_random`, eligibility, both-strand transcript-order indexing incl. B-1 fix, `next_acceptor_pos`).
  Biological assay logic stays in the study package; **no study-specific assumptions enter the adapter**
  (TRANSFER_GUIDE adapter contract).
- **New adapter** `src/gfmcd/adapters/caduceus.py` conforming to the base contract:
  - Identity: checkpoint sha256 + HF revision, tokenizer, objective (MLM), context length, position scheme.
  - Inputs: **deterministic single-nt tokenization** (validated one-token-per-nucleotide, fixed prefix);
    `nucleotide_to_token(p) = p + prefix_tokens` (same coordinate contract as SpliceBERT).
  - Outputs: **native per-nucleotide MLM logits** exposing individually-decodable A/C/G/T tokens so the
    margin readout is computable.
  - Invariants: shape, checksum, device, precision, **no-mutation** checks.
  - Provenance: full run manifest (code revision, data, params, seed, command).
- **Context = 1024 nt windows** (match SpliceBERT's information content, per Cam — "change the model, not
  the biological information available"). Caduceus-PS supports 131k; we constrain. Longer-context
  robustness is deferred.
- **Cohort:** same genomic eligibility as SpliceBERT (internal exon; in-window intron-mate acceptor;
  canonical GT donor; donor→acceptor gap ∈ [60,150]; **both strands**). A **NEW** Caduceus discovery /
  confirmatory chromosome split (SpliceBERT's chr8/9 seal does not bind Caduceus, but we still seal a
  Caduceus-untouched confirmatory cohort). Deterministic (`sorted(set())` + pinned `PYTHONHASHSEED`).

---

## 3. MEMORIALS (HARD preflight; on the measurement path; each CAN fail; STOP on failure)

- **`single_nt_tokenization`** — Caduceus tokenizer yields **exactly one token per nucleotide** with a fixed
  prefix; `unk` never appears on `ACGTN`. *Fails* if vocab is k-mer/BPE → coordinate map wrong. On-path
  (readout position = `p + prefix_tokens`).
- **`margin_readout_decodable`** — A/C/G/T are individual decodable output tokens; on a **known canonical
  donor**, `margin_donor(clean) > 0` (the model recognizes GT). *Fails* if logits are not per-nucleotide.
- **`real_base_corruption`** — every perturbation substitutes real bases; the `[MASK]` id never enters a
  corrupted construct (ported invariant). *Fails* if off-manifold `[MASK]` leaks in.
- **`donor_destruction_anchor`** — destroying the donor GT collapses the donor margin: **`A_donor > 0`,
  gene-clustered CI-excludes-0** (also the sensitivity plant, §4). *Fails* → the model does not recognize
  donors → the premise is void, STOP.
- **`nonpartner_pool_excludes_partner`** (B-3) — assert `next_acceptor_pos ∉` the non-partner pool for
  **every** example. *Fails* → the null equals the treatment (the sealed B-3 defect).
- **`both_strands_present`** (B-1) — minus-strand eligible exons are present (count > 0, ≈ half). *Fails* →
  the strand-indexing regression that silently dropped 41,153/41,153 minus-strand exons.
- **`rc_plumbing_consistency`** — feeding the **properly mirrored + complemented** RC of a window
  reproduces `margin_donor` within numerical tol. **⚠ LABELED ARCHITECTURE-FORCED:** Caduceus-PS is
  RC-*equivariant*, so this equality is *guaranteed by construction* — it is a **PLUMBING memorial that
  validates the adapter's coordinate/RC handling, NOT biological coupling evidence.** (See §5.)
- **`determinism`** — identical cohort + results across `PYTHONHASHSEED ∈ {0,1,2,17}`.
- **`adapter_invariants`** — shape/checksum/device/precision/no-mutation on every forward.

`--smoke` = mechanics + `single_nt_tokenization` + `margin_readout_decodable` + `donor_destruction_anchor`
+ `nonpartner_pool_excludes_partner` + `both_strands_present` + `determinism`, tiny-n, confirmatory sealed.

---

## 4. CALIBRATION (sensitivity + specificity; hard gates)

- **Planted-positive (sensitivity, same channel):** the **donor-destruction anchor** `A_donor` must be
  detected (CI-excludes-0, large) — the pipeline must see the model's *own* donor recognition, which travels
  the identical readout channel as the coupling. If it cannot detect this, the assay is underpowered → STOP.
- **Null / specificity (FPR ≤ 0.05):**
  1. **Non-site null** — corrupt a random **non-site** position (`perturb_random`): coupling ≈ 0 (CI ⊆ small
     bound / includes 0).
  2. **Permuted-partner null** — reassign each donor a random **non-partner** acceptor as its nominal
     "partner" and corrupt that; the partner-specific structure must **vanish** (≤ target FPR). Calibrate
     the significance threshold to this permutation null itself (split calibrate/test for an out-of-sample
     FPR), so "partner-specific" is earned.

---

## 5. IDENTIFIABILITY / common support (frozen)

- **Distance-decay confound** (the sealed B-4 lesson): a nearer acceptor may move the donor more regardless
  of partnership. Blocked by **distance-matching** the non-partner control (estimand 2) and by the
  **distance-matched non-donor** control (estimand 3). Predefine the overlap check on an **unselected** pool
  (partner vs non-partner distance/local-composition distributions must overlap); honest non-identifiable
  outcome if partnership and distance are collinear in the eligible cohort.
- **Local-composition confound:** the matched non-partner AG is matched on local sequence so the contrast is
  not "AG present vs absent" but "*this* AG vs *a* matched AG."
- **RC-equivariance wrinkle (declared non-identifiable AS a biological contrast for PS):** because
  Caduceus-PS ties the two strands by construction, a forward-vs-RC "orientation" comparison **cannot
  separate biology from architecture** — it is architecture-forced. We therefore **demote it to the
  `rc_plumbing_consistency` memorial** (§3) and do **not** report it as strand-specificity evidence. An
  independent biological strand-specificity control would require **Caduceus-PH** (RC-augmented, not
  architecturally tied) or a different construct; flagged as an OPTION, not in primary scope.

---

## 6. DECISION RULE (frozen; CI-classified; mirrors Cam's decision tree)

Evaluated on the **untouched confirmatory cohort** after §3–4 PASS and τ frozen from discovery:

- **QUALIFIES — behaviour transfers** iff ALL hold:
  (a) `C̄` CI-excludes-0 with the expected sign; (b) partner-specificity CI-excludes-0;
  (c) donor-specificity CI-excludes-0; (d) `ρ = C̄/A_donor ≥ τ` (frozen from discovery), ρ CI-excludes-0;
  (e) stability — same sign in both strand subgroups and across the gene bootstrap;
  (f) all memorials + calibration PASS.
  → **promote to Phase 3–4** (SSM causal localization/necessity/mediation; compare *computational motifs*
  with SpliceBERT, not head numbers — positional-routing vs content-routing vs distributed state).
- **DOES NOT QUALIFY — honest architecture/objective null:** positive controls PASS but coupling/specificity
  CI includes 0 or `ρ < τ`. A real, reportable finding: the partner-coupling behaviour does not transfer to
  this SSM/RC-equivariant/objective. (TRANSFER_GUIDE: an honest null in another GFM is a success of the
  method.)
- **INSTRUMENT LIMIT — not a biological null:** a positive control FAILS (e.g. `A_donor` CI includes 0, or
  tokenization/readout memorial fails) → Phase-2 plumbing problem; fix and re-run; draw no biological
  conclusion.

---

## 7. HELD-OUT (frozen)

Selection/calibration occurs (τ, any thresholds), so held-out is mandatory. **Discovery cohort** (calibration
chromosomes): set τ, verify power/MDE for C̄, partner- and donor-specificity, freeze. **Confirmatory cohort**
(disjoint, untouched chromosomes): opened once, after freeze, for the §6 decision. Mirror SpliceBERT
train/val/test discipline. Held-out ≠ sampling stability: report gene-clustered CIs; if any selection is
seed-sensitive, add independently-seeded draws (REFERENCE §5).

---

## 8. GATES

Global **PASS = AND(memorials §3, calibration §4)**; the run **STOPS** on any failure. `--smoke` per §3.
**Feasibility preflight (Phase 2, gates everything before compute):** Caduceus-PS loads via
`AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)` and **runs on the available compute** —
if the modeling code hard-requires `mamba-ssm`/`causal-conv1d` CUDA kernels with no CPU/MPS fallback, fall
back to Colab GPU (precedent: the additive-frozen handoff). 7.7M params ⇒ CPU inference is trivially fast if
kernels are CUDA-only. Confirm single-nt vocab is decodable and MLM per-nt logits are exposed.

---

## 9. SCOPE / provenance

- **Reaches:** Behaviour level, TRANSFER_GUIDE **Phase 1** (behavioral qualification) on a second GFM.
- **Does NOT establish:** mechanism transfer / method portability — that needs Phases 3–4 with
  **SSM-appropriate** interventions (BiMamba state/channel ablation, patching), **no** attention-head /
  L3H1 / period-3 analog. Behaviour ≠ mechanism (Cam's correction, load-bearing).
- **Inherited limits:** long introns untestable (gap ≤ 150, 1024 nt); one checkpoint; comparison to
  SpliceBERT is at the level of computational motifs.
- **Provenance:** `caduceus-ps_seqlen-131k_d_model-256_n_layer-16`, HF revision + config sha256 recorded
  before confirmatory unseal; tokenizer sha; cohort split + seeds + pinned `PYTHONHASHSEED`; run manifest.
- **New study dir:** `studies/caduceus_partner_transfer/` from `studies/TEMPLATE_REAL_GFM_STUDY/`.

---

## OPEN DESIGN DECISIONS (for red-team + Cam)

1. **τ anchoring** — is `ρ = C̄/A_donor` (coupling as a fraction of the model's donor-recognition drop) the
   right scale-invariant effect size, or prefer a standardized effect (C̄ / cross-gene SD)? τ from discovery.
2. **RC control** — accept demoting orientation to a plumbing memorial for PS; add **PH** as a companion for
   an independent biological strand control, or leave strand-specificity out of the primary gate?
3. **Cohort** — reuse SpliceBERT's exact eligible gene set with a fresh discovery/confirmatory chromosome
   split, or draw an independent cohort?
4. **Compute** — attempt local (MPS/CPU) first; pre-authorize Colab GPU fallback if SSM kernels demand CUDA?
