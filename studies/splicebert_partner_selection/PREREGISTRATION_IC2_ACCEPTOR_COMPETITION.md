# Pre-registration — IC2: does SpliceBERT rank COMPETING candidate acceptors by position and payload?

**Status: DRAFT for red-team → pilot-calibrate → FREEZE. DO NOT RUN until frozen.** The compact
acceptor-competition study that supersedes the (preserved, not-pursued) IC2a decomposition. It is the
intended FINAL SpliceBERT experiment before transfer to a second GFM. **Behaviour-first**: full-model
ranking assay; L3H1 / payload / `U_info` tracing is a SEPARATE, contingent follow-up run ONLY if a
competitive-ranking effect is established.

Carries all IC1 lessons: deterministic constructs (`sorted(set())` + pinned `PYTHONHASHSEED` + a
determinism test), CI-based classification, the incremental-within-state OOD noninferiority gate, and
strict outcome-blind discipline (design + pilot + audit + freeze before any val read). **Partner-
specificity is built in** — corrupting a specific candidate isolates its causal support, so no sham
subtraction is needed (this sidesteps IC1's sham-exchangeability problem).

---

## 0. INTENT (north star)

> **When two candidate acceptors compete in a fixed intron background, does the donor's coupling follow
> POSITIONAL ADDRESS, ACCEPTOR-LOCAL PAYLOAD (sequence), their INTERACTION — and does the annotated
> partner still win after position and local sequence are reciprocally exchanged?**

Behaviour-level ranking, measured by candidate-specific causal support. If a ranking effect is
established, trace it into L3H1 / payload / `U_info` (contingent follow-up); otherwise stop and transfer.

---

## 1. Construct, domain, observable

- **Domain:** real internal introns (`L∈[60,150]`, the validated domain) providing (a) a **fixed intron
  background/scaffold** (donor + interior with two candidate SLOTS masked out) and (b) the **annotated
  partner** acceptor window `P` (23-nt 3′SS, `[a−20..a+2]`). Decoys `D` = real acceptor windows drawn from
  a frozen pool (matched or contrastive in MaxEnt `score3`), placed deterministically.
- **Two candidate SLOTS** downstream of the donor, non-overlapping, in-window: `p_near` (short donor–slot
  distance) and `p_far` (long distance), both plausible acceptor positions; the native-partner distance
  anchors the near/far calibration (frozen from the pilot).
- **Observable (frozen LINEAR):** `f = logit(G) − mean{logit A,C,T}` at the masked donor (the sealed
  coupling readout). Corruption of a candidate = **AG→CC** at its `[·−2,·−1]`.

---

## 2. Factorial — 2 sequence × 2 position × candidate-corruption, reciprocal swap `⟐ RED-TEAM`

For each background gene, place sequences `{P, D}` into slots `{near, far}` in BOTH reciprocal
arrangements, and read `f` under every candidate intact/corrupt combination:

- **Arrangement 1:** `P@near, D@far`. **Arrangement 2 (swap):** `P@far, D@near`.
- **Corruption states (per arrangement):** both intact; near-corrupt; far-corrupt; both-corrupt.
- **Candidate-specific causal support** `S(X)` (partner-specific by construction):
  `S(X) = f(only X intact) − f(both corrupt)` — the coupling the donor derives from candidate X alone, at
  X's (sequence, position). The **reliance** drop `Δ(X) = f(both intact) − f(X corrupt, other intact)`
  measures which candidate the donor depends on when both compete.

All constructs deterministic (frozen decoy pool + slot positions + RNG). Whole-scaffold composition and
length held fixed across arrangements; only the two 23-nt slots' content/position differ.

---

## 3. Estimands (frozen; five, per Cam)

Let `S(seq, pos)` be candidate-specific support (§2), `seq∈{P,D}`, `pos∈{near,far}`.
1. **Positional advantage** `β_pos = mean_seq[S(seq,near) − S(seq,far)]` (near vs far, sequence fixed).
2. **Candidate-local sequence advantage** `β_seq = mean_pos[S(strong,pos) − S(weak,pos)]` (strong vs weak
   local sequence, position fixed; strong/weak by frozen `score3`).
3. **Address × payload interaction** `β_int = [S(strong,near)−S(strong,far)] − [S(weak,near)−S(weak,far)]`.
4. **Candidate-specific causal support** — the `S(X)` and reliance `Δ(X)` distributions themselves.
5. **Annotated-partner-wins-after-swap** — in Arrangement 2 (P@far, D@near), the fraction of genes where
   the donor relies more on `P` than `D` (`Δ(P) > Δ(D)`), i.e. sequence beats position; vs the reverse.

All are cluster-mean (homology cluster ≈ gene), **BCa CI**, classified by CI (§6); `β_int` is the
address×payload non-additivity of central interest.

---

## 4. Inference (frozen)

- **Unit:** homology cluster (≥80% id ≈ gene); deterministic cohort (sorted-set + pinned hashseed;
  determinism test required).
- **One FWER-controlled omnibus** over `{β_pos, β_seq, β_int}` (bootstrap max-statistic, two-sided
  α=0.05); interpret the individual estimands (and estimands 4–5) under hierarchical gatekeeping only if
  the omnibus rejects.
- **CIs:** cluster (wild/multiplier) bootstrap, BCa. **SESOI = 0.05 nats** (inherited).
- **Sampling:** census of the eligible deterministic cohort; else multiple seeded draws + stability.

---

## 5. Gates / memorials (HARD preflight; STOP on failure) `⟐ RED-TEAM`

- **`determinism`** (IC1 A2): identical constructs/cohort/results across `PYTHONHASHSEED ∈ {0,1,2,17}`.
- **`native_anchor`**: the native (un-swapped) construct — partner at its real slot — reproduces the sealed
  coupling of expected sign/scale (else the scaffold construction is broken).
- **`corruption_validity`**: AG→CC at each candidate destroys its acceptor (MaxEnt `score3` collapses),
  and `f(both corrupt)` is the uncoupled floor (`f(both corrupt) < f(both intact)` on the eligible set).
- **`distal_ood` (corrected, IC1 A1 form)**: the synthetic 2-candidate construct's window NLL is
  noninferior to a natural-to-natural null (incremental within-state, per-state, gene-clustered; `δ_OOD`
  frozen from the pilot) — a competition construct that is OOD cannot be interpreted.
- **`sensitivity_onpath`**: a planted strong-vs-weak pair (a maximal-`score3` P vs a near-null D) produces a
  detectable ranking (`β_seq ≥ SESOI`) — the null is not licensable if the pipeline cannot detect a clear
  ranking.
- **`native_coupling_present`**, **`identity_noop`**.

`--smoke` = mechanics + determinism + native_anchor + corruption_validity + on-path on tiny-n.

---

## 6. Decision rule (frozen; CI-classified; gatekept by the omnibus)

Omnibus rejects (α=0.05 FWER), gates PASS → interpret each estimand by its CI (SESOI=0.05):
- **material** (CI excludes 0 AND `|·| ≥ 0.05`) / **small-but-detectable** (excludes 0, `|·|<0.05`) /
  **bounded-null** (CI ⊆ ±0.05, includes 0) / **inconclusive** (includes 0, extends beyond ±0.05).
- **Interpretation map:**
  - `β_pos` material → the model uses **positional address** to rank candidates.
  - `β_seq` material → it uses **acceptor-local payload/sequence** to rank.
  - `β_int` material → **address × payload interaction** (non-additive ranking).
  - Estimand 5 (partner-wins-after-swap): partner still wins → **sequence/payload dominates**; near decoy
    wins → **position/address dominates**; mixed → context-dependent.
- **Established competitive ranking** (any of β_pos/β_seq/β_int material, native + on-path + OOD +
  determinism clean) → **promote to the contingent L3H1/payload/`U_info` tracing** follow-up.
- **No ranking** (omnibus null or all bounded/small) → the model does not competitively rank in this assay;
  **stop and transfer to a second GFM** (do not further subdivide SpliceBERT).

---

## 7. Scope, deferred, transfer

- **Behaviour-first.** L3H1 QK / acceptor payload / donor-side `U_info` tracing is a SEPARATE contingent
  prereg, run only if §6 establishes a ranking — via A8 reconstruct-inject (deliver a candidate's payload
  through the other's address) to test whether position or payload drives the mediated support.
- **This is the intended last SpliceBERT experiment.** After IC2 (whatever the outcome), **prioritize
  transfer of the method to a SECOND GFM** over further SpliceBERT subdivision (the method, not the
  catalogue, is the deliverable — CLAUDE.md mission).
- **Biological bridge (if a ranking is established):** whether model ranking predicts MPSA-measured
  competitive-splicing outcomes for natural near/far or strong/weak acceptor pairs.

---

## 8. Pre-freeze pilots required (disjoint TRAIN; no val read)

1. **Scaffold/construct validity:** build the 2-slot competition construct; demonstrate `native_anchor`
   (native reproduces coupling), `corruption_validity`, and adequate eligible-gene count.
2. **Slot geometry:** freeze `p_near`/`p_far` from the native-partner distance distribution; verify both
   slots are viable acceptor positions.
3. **Decoy pool:** freeze the decoy-acceptor pool + the strong/weak `score3` strata (deterministic pick).
4. **Corrected OOD gate** demonstration (natural/identity pass; planted-OOD fails) + `δ_OOD`.
5. **On-path ranking sensitivity** (strong-vs-weak detection) + **MDE / N** for β_pos, β_seq, β_int, omnibus.
6. **Determinism test** across hash seeds.

**Provenance:** checkpoint sha256 `2ad91428…0149d`; MaxEnt pinned; shuffler/decoy-pool + slot positions
pinned; all thresholds + seeds preregistered after the pilots. No val data read before freeze.
