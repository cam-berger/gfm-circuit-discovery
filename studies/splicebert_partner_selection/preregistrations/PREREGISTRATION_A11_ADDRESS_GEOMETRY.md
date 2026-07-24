# Pre-registration — A11: is the address absolute or relative? (donor-graph experiment 15; A4 redesign)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a11_address_geometry.py` runs. Redesign of the confounded A4
(`DECISIONS_A0.md`: A4 clamped ISOLATED-token position embeddings → inconsistent position field →
`S_A` collapsed). A11 uses a **consistent GLOBAL `position_ids` shift** with a **mandatory
intact-`S_A`-survival memorial**. Reuses `address.qk_logit_to_window` / `phase_matched_decoy` /
`windows` (the A1a/A4 `S_A` instrument). Pre-run corrections logged inline.

---

## 0. Why this document exists

C22 (A1a): L3H1's targeting is **position-embedding geometry**, absolute-vs-relative left open. A11
answers it for the donor→acceptor address:

> Is `S_A` (L3H1's donor-query QK preference for the intron-mate acceptor `a⁺` over a phase/distance-
> matched decoy `a⁻`) invariant to a **common translation** of the whole region (⇒ **RELATIVE**, offset
> only) or does it change systematically with absolute position (⇒ **ABSOLUTE**)?

QK/selection level; `S_A` is a mechanistic intermediate; no behavioural claim.

---

## 1. Intervention: consistent GLOBAL `position_ids` shift (not the A4 isolated clamp)

- **PE-table headroom.** SpliceBERT.1024nt: `P=1026`, tokenized `S=1026` (= the whole table ⇒ zero
  headroom). So **truncate** each input to a **window `W=640`** (`[CLS] + first 639 tokens`), keeping
  examples whose `{donor, a⁺ window, a⁻ window}` all fall in `[off, W−8]` — `n ≥ 150`.
- **Common translation.** `position_ids(Δ) = arange(W) + BASE + Δ`, `BASE=192`, `Δ ∈ {−192,−128,−64,0,
  +64,+128,+192}` — every position stays in `[0, 1023] ⊂ [0,P)`. The **whole window's position field
  translates together** (content unchanged) — a consistent field, NOT A4's isolated-token clamp.
- **`S_A(Δ)`** = `qk_logit_to_window(q,k,H3,Dh,donor,a⁺) − (…,a⁻)` on the truncated input with
  `position_ids(Δ)` (L3H1 query/key captured via `capture_proj`; head-1, /√32).
- **`E_common(Δ) = S_A(Δ) − S_A(0)`** (0 = the `BASE` reference).

---

## 2. Gates and falsifiers

**Preflight** (`preflight_a11_address_geometry.json`, global PASS = AND):

- `shift_is_noop_at_native` — `S_A` with `position_ids = arange(W)` (native, no BASE/Δ) reproduces the
  **truncated native forward** `S_A` (no `position_ids` arg) < 1e-4 (the `position_ids` mechanism is a
  faithful no-op at the native field).
- **`intact_S_A_survives` — THE MANDATORY MEMORIAL A4 LACKED:** `S_A(0) ≥ 0.5` (the truncated,
  BASE-shifted preference is substantial) **and** every `S_A(Δ)` on the grid `≥ 0.5 · S_A(0)` (the
  preference **survives** the common shift; a collapse to ≈0 at any Δ ⇒ the shift is confounded — like
  A4 — and the run STOPS). Distinguishes a real geometry signal from a broken position field.
- `truncation_preserves_targeting` — truncated-native `S_A` > 0 and its sign/magnitude track the full-
  length `S_A` (report the correlation; a large mismatch ⇒ truncation altered the mechanism).
- `both_strands_present`, `checkpoint_recorded`, `grid_fits` (n_kept ≥ `--min-n`).

**If preflight fails the run STOPS** (`--force` → DIAGNOSTIC). `--smoke` = memorials only (val sealed).

---

## 3. Decision (frozen; ONLY if `intact_S_A_survives` passes)

- **RELATIVE** (offset-only) iff `max_Δ |E_common(Δ)| / |S_A(0)| < 0.20` and the OLS slope of `S_A` vs Δ
  is CI-consistent with 0 (near-invariance under common translation).
- **ABSOLUTE** iff `E_common(Δ)` is systematic/monotonic with `max_Δ |E_common| / |S_A(0)| > 0.20` and a
  slope CI excluding 0.
- **MIXED / inconclusive** otherwise; report the `E_common(Δ)` curve, `S_A(0)`, and the slope with a
  gene-grouped bootstrap CI regardless.

**Secondary (descriptive, forward-free):** the raw L3H1 positional-QK kernel `K(p,q) = (PE_p
W_Q^{h1})·(PE_q W_K^{h1})/√32` Toeplitz residual fraction (variance along constant-offset diagonals ÷
total). Low ⇒ the raw PE kernel is offset-only — a crude proxy (ignores layers 0–2 and L2H15),
descriptive only; the causal `E_common` is the result.

---

## 4. Outputs, reflection

- `scripts/splicebert/run_a11_address_geometry.py` (reuses the A4/A1a `S_A` instrument + a global
  `position_ids` shift; no new kernel); `results/…/a11_address_geometry.json` + manifest + preflight.
- **REFLECTION:** pass/fail every memorial (esp. `intact_S_A_survives` — the A4 fix); the `E_common(Δ)`
  curve with CIs, `S_A(0)`, slope, `max|E|/|S_A(0)|`; the RELATIVE/ABSOLUTE/MIXED verdict; the
  truncation-preservation correlation; the Toeplitz descriptive with its proxy caveat; escaped failure
  modes (the survival memorial, truncation effects, PE-table edge, whether any Δ could not have moved
  `S_A`).

**Scope.** QK/selection, position-embedding geometry of L3H1's donor→acceptor address (C22). The
**sibling** of the payload arc (A6–A10): payload = *what* (local splice strength); this = *where*
(address geometry). No behavioural claim; no prior claim redefined.

**Provenance.** Checkpoint sha256 `2ad91428…0149d`; `S_A` instrument per A1a/C22 and A4.
