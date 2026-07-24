# Pre-registration — causal acceptor-address transplantation into L3H1 (donor-graph experiment 3)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_acceptor_address.py` is executed. Every candidate
definition, readout, intervention, control, and pass/fail threshold below is fixed here and
imported from `gfmcd.splicing.circuit` / `gfmcd.splicing.address`, not redefined in the
runner. No number produced by the run may revise any threshold in §4.

---

## 0. Why this document exists, and where it sits

The frozen circuit is `CIRCUIT = (READER = L3H1, ADDRESSER = L2H15)`: L3H1 **reads** at the
intron-mate acceptor whose destruction the coupling behaviour measures, and L2H15 and L1H12
contribute upstream through L3H1 without licensing a simple "address versus query" split
(C3, C4, C6, C9). What is *not* yet established is the sharpest causal statement about that read:

> Is the read L3H1 performs at the intron-mate acceptor **specific to the model's own
> partner**, over a distance- and reading-phase-matched generic decoy and — the
> salience-controlled test — a non-partner **real** acceptor; and does relocating the partner's
> **key** re-route L3H1, with routing (key) and content (value) separable?

This experiment tests exactly that. It advances the account from "L2H15 supplies a key that
rescues at the acceptor" (C9, a *restore*-at-the-same-position result) to a
**partner-identity-controlled preference** plus a **key/value dissociation**, or fails to.

> **Amendment (2026-07-15, pre-run, no confirmatory data seen).** The original §0 stated L2H15
> "builds an acceptor key/address" and L3H1 does "a content-addressed read." That re-committed
> the R15-retracted characterization and contradicted §5. Restated mechanism-neutrally above;
> no confirmatory outcome was observed before this edit.

**The failure mode it is built to survive — twofold.**

1. *A preference that is not about the partner.* Any AG at a plausible distance draws some
   attention. The three descriptive preference readouts in §3 (`S_QK`, `S_A`, `S_Y`) are
   licensed **only** by a decoy that is matched on the AG motif, the donor→candidate distance,
   the distance-mod-3 reading phase, and coding state — so the sole remaining difference is
   *is this the model's real intron-mate partner*. A preference that survives that match is a
   partner-identity effect; a preference that does not is distance/motif salience (rival
   explanation), and the descriptive numbers alone cannot separate them.

2. *A relocation that is architecturally forced (R15).* Relocating key/value vectors was
   **retracted once** (R15) because absolute position is already inside the vectors, so a
   "position-independent pointer" reading was forced, not discovered. This experiment does
   **not** re-make that claim. The relocation arm (§4, C-Transplant) is a **one-way copy from a
   different forward pass into a decoy** (R7-safe), and its *only* licensed conclusion is
   *"the intron-mate acceptor's key+value is a transplantable causal carrier of the coupling,"*
   an OV/mediation-level statement, **never** "position-independent" or "content-addressed
   pointer." The valid positional result remains C8. The controls that make the relocation
   able to fail (non-site null, key-only/value-only dissociation, mlm-health) are in §4.

This experiment reaches **no new interpretive-firewall level beyond OV/mediation + Behaviour**
(§5). It does not touch Biology.

---

## 1. The target — imported unchanged from `circuit.py`

Identical to the sealed pre-registration and the donor-anchor; **no redefinition**.

- **Circuit** — `READER = L3H1 = (3, 1)`, `ADDRESSER = L2H15 = (2, 15)` (0-indexed
  `(layer, head)`), `MLP2 = (2, "mlp")` (the layer-2 MLP write), all from `circuit.py`.
- **Readout** — native masked-LM output. Mask the G of the donor GT at the position set by
  `CouplingTask.encode`; score `margin = logit(G) − max(logit A, C, T)`. No probe, head, or
  labels.
- **Corruption** — `substitute(seq, next_acceptor_pos − 2, "CC")`: replace the partner AG at
  `[a−2, a−1]` with real bases (never `[MASK]`; `CC` cannot re-create an AG). A one-way,
  multiset-changing edit, not a swap (R7).
- **Coupling** — `Y(a) = margin(clean) − margin(destroy a)`, per example. The standard
  coupling is `Y(a⁺)` for the intron-mate partner `a⁺ = next_acceptor_pos`.
- **Eligibility** — internal exon; in-window intron-mate acceptor; canonical GT donor;
  donor→acceptor gap in [60, 150] nt; **both strands** (R11). Val split (chr11/16/19).

**Recovery / redirection fractions use ratio-of-means**, `Σ(numerator) / Σ(denominator)`, with
gene-grouped bootstrap CIs (`gboot_ratio`, n_boot = 2000). No per-example conditioning on
coupling for any behavioural magnitude (R10); `min_effect = 0.3` is used **only** as a ratio
denominator and every such number is labelled *conditional*. Size-matched nulls use exact
add-one permutation p `= (1 + #{null ≥ obs}) / (n + 1)` (`empirical_upper_p_value`), corrected
across the scanned family with Benjamini–Hochberg where a family is scanned.

---

## 2. The must-build components and their correctness gates

Two new pieces are built in `src/gfmcd/splicing/address.py` and imported by the runner.

**(a) Two decoys, serving two purposes** (both built in `address.py`; see also
`DECISIONS_ACCEPTOR_ADDRESS.md` for the setup-time reasoning). For each eligible example:

- **Generic-site decoy `a⁻`** — `phase_matched_decoy(x, tol=40)`: a non-partner AG `∈ x.other_ag`
  with `a⁻ > donor + 20`, `|a⁻ − next_acceptor_pos| > 12`, in-window, `|(a⁻ − d) − (a⁺ − d)| ≤ tol`,
  **and** `(a⁻ − d) ≡ (a⁺ − d) (mod 3)`, closest distance. This answers *"partner vs a generic
  distance/phase-matched site."* An intronic AG is a much **weaker** 3′ splice site than a real
  acceptor, so its motif is **not** matched — `motif_gap_sd` is reported, not gated.
- **Real-acceptor decoy `aʳ`** — `real_acceptor_decoy(x, tol = --real-tol, default 60)`: a
  non-partner **real** annotated acceptor `∈ x.real_acc`, matched on reading phase and distance.
  Because both `a⁺` and `aʳ` are genuine 3′ splice sites, splice-site **strength is matched by
  construction**, so a preference here is the salience-controlled **partner-identity** test. Two
  caveats, both reported: (i) real acceptors are sparse (~1 in 5 pairs), so this arm has **modest
  power** (labelled with its `n` and `n_real_decoys`); (ii) non-partner real acceptors tend to sit
  **farther** from the donor than the intron-mate partner (the nearest downstream acceptor), so a
  positive `S_A vs real acceptor` could be attention **distance-decay** rather than identity. The
  **signed** distance gap `dist(aʳ) − dist(a⁺)` distribution is reported (`match_audit.
  real_acceptor_distance`) and the §7.2 reflection must adjudicate identity-vs-distance with it
  (looser `real-tol` trades this confound against power; both tolerances are recorded in the
  manifest).

Covariates recorded per pair: donor→candidate distance for each site, distance-mod-3 (0 by
construction), local 3′-SS motif score `M` (`address.acceptor_motif_score`), GC, coding state
(`example_phase`), and donor→candidate attention mass. **The principal variable is partner
identity; the real-acceptor arm is the one that isolates it from splice-site salience.**

- **G-Match — correctness gate (can fail → STOP the generic-site preference claim):** across the
  pair set, the **systematic** distance bias `dist_bias = |mean dist(a⁺) − mean dist(a⁻)| ≤ 8` nt
  (this is what would bias the aggregate-mean preference via attention distance-decay) and
  distance-mod-3 mismatch rate `= 0` (exact). **Reported, not gated:** `dist_spread` (per-pair
  mean `|Δ|`, which is inherently ~20-30 nt because the nearest phase-matched intronic AG is
  sparse), `motif_gap_sd` (an intronic AG cannot be motif-matched to a real acceptor), and
  `coding_frac` (shared by construction). Because per-pair distance matching is loose, the
  identity claim is not rested on the generic arm's tight matching but on the real-acceptor arm
  with distance stratification (§7.2). If the bias/phase gates fail, the generic decoy is invalid.

**(b) The L3H1 QK-logit readout (must-build; the analytic form is a candidate generator, not
evidence).** `l3h1_qk_logit(ad, exs, key_positions)` returns, per example, the pre-softmax
attention logit from the **donor query token** to a key window, using the head-1 slice
`[32:64]` of L3's query/key projection outputs (`capture_proj`), scaled by `1/√32`:
`ℓ(d, W) = mean_{j∈W} (q_d^{3,1} · k_j^{3,1}) / √32`. `l3h1_attention(ad, exs, key_positions)`
returns the post-softmax attention mass from the donor row to a key window, from
`model(output_attentions=True)` with **`eager=True`** (SDPA returns all-zero attentions — a
silent null). Key windows use `ACC_WINDOW_OFFSETS = (−3, −2, −1, 0)` around a candidate.

- **G-Attn — correctness gate (can fail → STOP):** with `eager=True`, every attention row
  used must sum to `1 ± 1e-3` and no head may be identically zero (the SDPA-zero trap,
  `load_adapter` docstring). If violated, the instrument is broken and no QK/attention number
  is computed.

---

## 3. Preference readouts

Computed once per pair, on the clean sequence, from L3H1's **donor query token** to each
candidate's AG window (`ACC_WINDOW_OFFSETS = (−3, −2, −1, 0)`), with gene-grouped bootstrap CIs.
Against the **generic-site decoy `a⁻`** (the QK→attention→behaviour ladder):

- **`S_QK` = ℓ(d, W⁺) − ℓ(d, W⁻)** — donor-query pre-softmax logit preference, head-1 slice
  `[32:64]` of L3 query/key (`capture_proj`), scaled `1/√32`. (Selection/QK.)
- **`S_A` = A(d, W⁺) − A(d, W⁻)** — post-softmax attention-mass preference, `eager=True`
  (SDPA returns all-zero attentions — a silent null; **G-Attn** gate: rows sum to `1 ± 1e-3`,
  head not identically zero). (Routing/attention.)
- **`S_Y` = Y(a⁺) − Y(a⁻)** — destroying the partner reduces the donor margin more than
  destroying the generic decoy (`Y(a⁻)` corrupts at `a⁻ − 2`). (Behaviour.)

The generic-site preference mixes partner identity with splice-site salience (the decoy is a
weaker site). The **salience-controlled identity** readout is therefore reported alongside:

- **`S_A vs real acceptor` = A(d, W⁺) − A(d, Wʳ)** over the subset with a real-acceptor decoy —
  a preference here cannot be explained by "stronger site" because both are real acceptors. This
  is the partner-identity result; it is lower-powered and labelled with its `n`.

A preference at `S_QK`/`S_A` but **not** `S_Y` (or vs the generic decoy but **not** vs the real
acceptor) is reported as exactly that — evidence never auto-promotes.

---

## 4. Causal tests — FROZEN pass/fail, each with its falsifier

All interventions are one-way copies from a **different forward pass** or component ablations,
never a `(k, v)` swap (R7). Recovery/redirection are ratio-of-means with gene-grouped bootstrap
CIs, population **unconditional** (`min_effect = None`). The **overall verdict** requires
**G-Preference, C-Redirect, and C-Dissociate**; C-Address and C-R15 are **reported** (a
component-attribution question and an R15 guard, respectively), not part of the pass/fail.

| Test | Intervention | PASS threshold | What FALSIFIES it |
|---|---|---|---|
| **G-Preference** (the primary result) | The §3 readouts on the clean run. | `S_QK`, `S_A`, and `S_Y` each have a gene-grouped 95% CI with lower bound `> 0` (partner preferred over the generic matched decoy at all three levels). | any of the three CIs includes 0 → the preference is not established at that level. |
| **C-Redirect** (the key is the address) | On the **corrupt** run, relocate L3H1's **key only** (clean `a⁺` → `a⁻`); measure the **relocation-induced rise** in decoy attention `ΔA = A(d, W⁻)[+key] − A(d, W⁻)[no reloc]` (difference-in-differences at the decoy window; the `A(d, W⁺)` term is dropped because corruption already depresses attention to the destroyed `a⁺`). | `ΔA > 0`, CI lower bound `> 0` — relocating the partner's *key* raises L3H1's donor attention to the decoy site. | `ΔA ≤ 0` → the key does not carry the routing address, or the earlier `W⁻−W⁺` positivity was a corruption artifact. |
| **C-Dissociate** (address ≠ content) | On the corrupt run, recover the donor coupling under relocation of the clean `a⁺` **key only**, **value only**, and **key+value** into `a⁻`; recovery `= Σ(m_reloc − m_co) / Σ(m_cl − m_co)`, gene-grouped. | `k` CI lower bound `> 0`, `k > v`, and `v ≤ 0.15`: routing (key) recovers some coupling but content (value) alone is not read where the query does not look. `kv > k` is **reported** as R15-expected content delivery, **not gated** (kv is position-forced — see C-R15). | `v` alone already recovers (content read position-independently), or `k ≤ v` (routing carries nothing), collapsing the dissociation. |
| **C-Address** (reported, not gated) | Ablate L2H15 (mean); clamp MLP2's write to its L2H15-ablated value; recompute `S_A`. Report `S_A(intact)`, `S_A(L2H15 ablated)`, retained fraction, `Δ_{MLP2}`, and add-one `p` vs the matched-random single-head null (§4.1). | *(reported)* whether L2H15 dominates the **matched-decoy preference** — a different question from its role vs the corrupt partner in `l2_accounting`. | *(reported)* |
| **C-R15** (reported guard, cannot be a "win") | Relocate the clean `a⁺` **key+value** into the decoy `a⁻`, a **non-site** (`control-pos`), and back into `a⁺` (injection). Report all three recoveries and their spread. | *(reported)* recovery is expected to be ~equal at all three positions — relocating a high-affinity key is position-forced (R15). | *(reported; a near-equal spread is the R15 signature and explicitly **blocks** any content-addressing / pointer claim.)* |
| **C-Health** (R8 alive-check, ablation arms) | `mlm_health` on masked non-site probes under the L3H1 and L2H15 mean-ablations. | health ≥ **0.5** under each ablation. | an ablation drops health below 0.5 → its number measures destruction, not mechanism (R12/R13). |

**Amendment protocol.** This document is frozen. Any pre-run correction is recorded inline as
`**Amendment (YYYY-MM-DD, pre-run, no data seen).**`. Post-unblinding corrections go to
`DECISIONS_ACCEPTOR_ADDRESS.md`/`CLAIMS.md`, never here. This §4 was frozen after exploratory
small-n (≤60) diagnostics on val that established the *design* (not any confirmatory number);
those diagnostics are logged in `DECISIONS_ACCEPTOR_ADDRESS.md` and motivated moving the identity
claim to the real-acceptor arm and demoting the key+value transplant to an R15 guard.

### 4.1 Controls and nulls
- **Non-site target** = `build_windows(...).control` — a distance-matched window that misses any
  AG and the corruption (validated by `guards`); used for C-R15's position test.
- **Matched-random head null (C-Address)** = the single non-L2H15 heads **in layers 0–2** (47
  heads; `matched_random_head_sets`, enumerated + de-duplicated — R9 — then restricted to layers
  `< 3`), each mean-ablated; `S_A` drop by add-one `p` over that family.

  > **Amendment (2026-07-15, pre-run, no confirmatory data seen).** The null was originally
  > written as all 95 non-L2H15 heads. `S_A = attentions[3]` depends only on the residual feeding
  > layer 3 (layers 0–2), so the 48 heads in layers ≥ 3 give an exactly-zero drop and would
  > zero-inflate the null and deflate the add-one `p`. Restricted to the 47 causally-live upstream
  > heads. No outcome was observed before this edit.
- **Injection ceiling** = relocating the partner's own clean `(k, v)` back into `a⁺` under
  corruption; the L3H1-only key+value read is a *partial* mediator, so this bounds every
  recovery from above (it is ~0.6, not ~1.0 — L4/L5 transporters carry the rest).

### 4.2 Reachability audit (a control that cannot fail is not a control)
- **G-Preference can fail:** a CI may include 0 at any level; `S_Y` in particular can be null
  while `S_QK`/`S_A` are positive (a QK/attention fact with no behavioural consequence).
- **C-Redirect can fail:** the softmax may refuse to move mass to `a⁻` even with the partner's
  key there (competition), decoupling QK preference from routing.
- **C-Dissociate can fail:** if value is read position-independently, `v` alone recovers and the
  address/content split collapses (this is the R15 boundary the arm is built to detect).
- **C-Health can fail:** R8 is the proof — an unmasked-copy score once *rose* under ablation;
  masked probes make it a real gate.

---

## 5. Interpretive firewall placement

| Level | This experiment |
|---|---|
| P (position table) | not touched |
| QK / selection | `S_QK`, C-Address: L3H1's donor-query scores the partner key above the matched decoy (candidate-level + causal ablation of L2H15/MLP2) |
| Attention / routing | `S_A`, C-Redirect: the preference survives softmax, and relocating the partner **key** re-routes L3H1's donor attention to the decoy site |
| OV / mediation | C-Dissociate: routing (key) and content (value) are separable — key alone re-routes but delivers no content; both together recover the coupling |
| Behaviour | `S_Y` and the recovery fractions use the existing native coupling; no new behaviour is introduced |
| Biology | not touched |

The strongest licensed statement if G-Preference, C-Redirect and C-Dissociate pass: *L3H1's
acceptor read prefers the model's intron-mate partner over a distance/phase-matched generic site
(S_QK/S_A/S_Y) and — the salience-controlled test — over a non-partner **real** acceptor
(S_A vs real acceptor); the partner's **key** carries the routing address (relocating it
re-routes L3H1), and routing and content are separable.* It **does not** license
"position-independent" or "content-addressed pointer": the key+value transplant recovers
~equally at the decoy, a non-site, and `a⁺` (C-R15), which is architecturally forced (R15) and
explicitly blocks any pointer reading. Whether L2H15 *dominates* the matched-decoy preference is
**reported** (C-Address), not claimed; component-share accounting of the address route vs the
corrupt partner is the already-established `l2_accounting.json` (C9/D11), cited not recomputed.
No biological claim.

---

## 6. Data, compute, outputs

- **Data:** val (chr11/16/19), eligibility per `CouplingConfig`; target **n ≥ 250** eligible
  pairs that also have a valid phase-matched decoy; both strands present and counted (R11).
  Seed 0 for all nulls and bootstraps.
- **Compute:** `load_adapter(DEFAULT_MODEL, device, eager=True)` (attention matrices are read).
  SpliceBERT.1024nt is small (6 layers, hidden 512); runtime is minutes on CPU/MPS.
- **Preflight (written in the SAME process, before any §3/§4 number is interpreted):**
  `results/splicebert_coupling/preflight_acceptor_address.json` with top-level
  `{frozen, commit, n_pairs, checks, PASS}`. Named checks (exact keys):
  `probability_invariants` (G-Attn: eager attention rows sum to 1 ± 1e-3, head not all-zero);
  `qk_reconstruction` (G-QK: `softmax(q·k/√32 + mask)` reproduces the model's own L3H1 attention
  to < 1e-3 — locks the analytic `S_QK` readout to the real attention under the absolute-PE/post-LN
  assumptions); `positive_control_partner_coupling` (destroying `a⁺` collapses the donor margin,
  coupling ≥ 0.20); `positive_control_L3H1_ablation` (L3H1 mean-ablation kills ≥ 50% of that coupling);
  `sensitivity_injection` (relocating the partner's own clean `(k, v)` back into `a⁺` under
  corruption recovers ≥ **0.30** — the L3H1-only key+value read is a partial mediator, so this
  ceiling is ~0.6, not ~1.0; it confirms the instrument delivers a known signal);
  `negative_control_value_only` (relocating the partner **value only** into `a⁻` recovers
  ≤ **0.15** — content without routing must not be read; this is the specificity control that
  *can* fail, replacing the R15-forced non-site gate); `diagnostic_kv_nonsite_recovery`
  (key+value into a non-site — **reported, always PASS**: it is position-forced by R15 and cannot
  be a specificity pass/fail); `swap_is_identity` (a symmetric `(k, v)` swap reproduces the clean
  margin to < 1e-4 — the R7 tripwire); `health_under_L3H1_ablation` and
  `health_under_L2H15_ablation` (masked-probe MLM accuracy ≥ 0.5 — R8/R13); `resolution_pair_match`
  (the G-Match audit: distance ≤ 8 nt, phase mismatch 0; `motif_gap_sd` reported);
  `both_strands_present`; `checkpoint_recorded` (checkpoint sha256 for provenance; interventions
  are hooks that never write weights). Global `PASS` is the AND. **If preflight fails, the run
  STOPS and no §3/§4 number is reported** (a `--force` flag exists for DIAGNOSTIC-only inspection
  and marks its output non-confirmatory).
- **Outputs:** `scripts/splicebert/run_acceptor_address.py`; `src/gfmcd/splicing/address.py`
  (reusable pairing + readouts); `results/splicebert_coupling/acceptor_address.json` +
  `<stem>.manifest.json` (via `ExperimentManifest`; `status = "CONFIRMATORY"` **only when
  `preflight_PASS` and `n_pairs ≥ 250`**, else `"DIAGNOSTIC"`; the result JSON carries
  `confirmatory`, `meets_power_target`, `n_genes`, and `preflight_PASS`; checkpoint + GTF + FASTA
  added as hashed inputs); `tests/test_acceptor_address_memorials.py` (phase-matched-decoy matches mod-3, rejects
  the partner neighbourhood; G-Match reports-but-does-not-gate motif, gates distance/phase; pure
  QK/attention readouts match hand-computation). The R7 swap-identity is checked live in the
  preflight and in `tests/test_circuit_memorials.py` for `relocate_proj`.

---

## 7. Post-hoc reflection (REQUIRED — do not skip)

After the run, a `REFLECTION` block (in the result JSON and reported to the user) must state:

1. **Pass/fail** against every §4 threshold and the preflight, verbatim, no softening.
2. **Intent check.** Does the generic-decoy preference (S_QK/S_A/S_Y) survive to the
   **salience-controlled** real-acceptor readout (S_A vs real acceptor), or is it explained by
   the partner being a stronger 3′ splice site? If the identity signal appears only vs the generic
   site and not vs a real acceptor, say so — it is a salience effect, not partner identity.
3. **R15 self-audit.** Confirm C-R15 shows the key+value transplant is position-insensitive, that
   it is reported (not gated), and that **no** "position-independent / content-addressed pointer"
   language appears anywhere in the write-up. The licensed claim stays at preference +
   address/content dissociation. If the result tempts a pointer reading, that is the R15 tell and
   is surfaced, not indulged.
4. **C-Address read.** State whether L2H15 dominates the matched-decoy preference; if it retains
   most of `S_A` under ablation, report that the preference is built more diffusely than the
   corrupt-partner route in `l2_accounting`, and do not force the "L2H15 builds the address" story.
5. **Escaped failure modes:** anything that, in hindsight, could not have failed, or any number
   that agrees to many significant figures (the R7 tell of one intervention wearing two hats).

A failing or ambiguous result is a real result about the address and is reported as such.
