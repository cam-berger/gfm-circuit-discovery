# The same-intron boundary-coupling circuit in SpliceBERT

**Status: partially resolved causal circuit, validated on held-out data (chr11/16/19).**
The sealed test (chr8/chr9) is opened once, against `PREREGISTRATION_SEALED.md`.

All numbers below are **val**, on the corrected **both-strand** dataset. Every number
published before 2026-07-12 was a *plus-strand* number (R11), and several rested on controls
that could not fail (R7–R10). **Read `CLAIMS.md` §Retractions before quoting anything.**

---

## The claim, in one paragraph

SpliceBERT, trained only to fill in masked nucleotides, recognises a 5′ splice site partly by
**reading the 3′ splice site at the other end of the intron**. Destroying the partner
acceptor's `AG` costs the donor **0.958** [0.855, 1.059] of logit margin; destroying a
motif-matched *non-partner* `AG` costs **0.012**, and the same corruption scored at a
*distance-matched other donor* costs **0.031**. A single head — **L3H1** — carries most of it:
ablating it loses 82% of the coupling while retaining 99.8% of the model's masked-prediction
accuracy. The head performs a **conjunctive, content-addressed read**: the acceptor's *key*
selects it and its *value* supplies the content, and neither alone does anything
(key 0.031, value 0.117, key+value **0.680**; interaction **I = +0.532** [0.478, 0.583]).
The address is a point in **representation space, not a coordinate in the sequence** — copy
the partner's (key, value) pair to any other position and the coupling returns just as well
(decoy AG 0.669, non-site 0.656, in place 0.686). An upstream head, **L2H15**, builds that
address, and *everything* it contributes flows through L3H1 (rescue 1.05 [0.97, 1.16]).

## The circuit

```
                the partner acceptor's AG at [a−2, a−1]
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  L2H15 — builds the ADDRESS                 │   ablation: −52% of coupling
        │  the key that makes the acceptor findable   │   rescue via L3H1's output: 0.96
        │  (also via MLP2: 0.69)                      │   → a CLOSED edge (D12)
        └─────────────────────┬───────────────────────┘
                              ▼
        ┌─────────────────────────────────────────────┐
        │  L3H1 — the CONTENT-ADDRESSED READ          │   ablation: −82%, rank 1 of 96
        │                                             │   health retained: 99.8%
        │  reads AT the acceptor:                     │
        │     key   alone   0.031                     │
        │     value alone   0.117                     │
        │     key + value   0.680   I = +0.532        │   the conjunction IS the mechanism
        │                                             │
        │  queries from ~11 positions in the 5′ss     │
        │  neighbourhood, redundantly                 │
        └─────────────────────┬───────────────────────┘
                              │  WRITES into those same 11 tokens
                              │  donor 0.345 · flank ±5 0.495 · acceptor 0.003
                              ▼
        ┌─────────────────────────────────────────────┐
        │  MLP3 → L4 → MLP4 → L5 → MLP5               │   direct effect: 0.023 of 0.818
        │  a DISTRIBUTED CHAIN                        │   0.023→0.18→0.26→0.46→0.53→0.818
        │  every link adds; none suffices             │   no compact readout head exists
        └─────────────────────┬───────────────────────┘
                              ▼
                    donor logit margin
```

## What each arrow is made of

### 1. The behaviour is real, partner-selective, and not a motif artefact
Native masked-LM readout — no probe, no labels. Perturbation substitutes real bases (`CC`),
never `[MASK]` (off-manifold). Unconditional, n=400, gene-clustered bootstrap:

| | effect |
|---|---|
| destroy the **partner** AG | **+0.958** [0.855, 1.059] |
| destroy a **non-partner** AG (partner now excluded from the pool) | +0.012 → paired diff **+0.945** [0.846, 1.045] |
| same corruption, scored at a **distance-matched other donor** | +0.031 → paired diff **+0.927** [0.817, 1.041] |

Selectivity **15.6×** [8.4, 39.9]. The effect is right-skewed (median **0.714**), so the mean
is not the typical exon — say so, and never quote the conditional mean (1.285) as the
behavioural magnitude (R10).

### 2. L3H1 is necessary, and it is not merely breaking the model
Exhaustive 96-head ablation, both strands: **L3H1 0.830** [0.654, 0.994] is #1; **L2H15
0.515** [0.396, 0.625] is #2. **L1H12 (0.266) is a persistent third and is NOT in C.**

| ablate | coupling lost | MLM health retained (**masked** probes) |
|---|---|---|
| L3H1 | 82.3% | 99.8% |
| L2H15 | 52.2% | 98.2% |
| **C = {L3H1, L2H15}** | **86.8%** | **97.8%** |

Against 400 **distinct** matched-random 2-head sets: circuit drop +1.063 vs random mean
+0.006 (max +0.400), **p = 0.0025**. Patching C back restores **83%** [78%, 87%].

The health column only became meaningful once the control could fail: the previous metric
scored the model *copying unmasked input*, returned ~0.99 for everything, and **rose** with
all 96 heads ablated (R8). Baseline is 0.53, not 0.99.

### 3. The read is conjunctive — and the conjunction is the mechanism
Patch L3H1's key/value at the acceptor `AG` window, clean → corrupt. Acceptor corruption
collapses the head's attention to its partner by **94%**.

| patch | coupling restored | attention restored |
|---|---|---|
| key only | 0.031 | 77% |
| value only | 0.117 | 0% |
| **key + value** | **0.680** | 77% |
| query @ donor | 0.003 | — |
| key+value @ control position | −0.000 | 0% |

**I = +0.532** [0.478, 0.583]. The key restores *selection* but not the conclusion; the value
restores neither alone. Only together does the head both **find** the partner and **read** it.

### 4. The address is separable from the content — and it is not a coordinate
Destroy the partner's AG, then **copy** its clean key and/or value into a *different*
position. A **swap** is an algebraic identity and measures nothing (R7); a one-way copy
changes the multiset of (key, value) pairs and can fail.

| copy the partner's… | to | coupling restored |
|---|---|---|
| value only | decoy AG | **0.004** — content the head is not looking at is worthless |
| key only | decoy AG | 0.238 — routes to the decoy, retrieves the decoy's *generic* AG content |
| **key + value** | **decoy AG** | **0.669** |
| key + value | a **non-site** position | 0.656 |
| key + value | back **in place** (reference) | 0.686 |

Relocation costs **2%**. The head finds its partner *wherever its key vector is put*. That is
what content-addressing means, measured rather than asserted.

### 5. L2H15 → L3H1 is a closed edge
With L2H15 ablated, restoring **L3H1's whole output** to its L2H15-intact value rescues
**0.959** [0.875, 1.052] of what L2H15 supplies — a CI that covers 1.0. All 16 layer-3 heads together give the same — no other L3 head is involved. The calibration row (restore the entire layer-2 output) returns
**1.000** exactly, as it must by construction; had it not, no other row would be interpretable.

The earlier "~60% of L2H15 unaccounted for" was an artefact of only ever patching a 4-token
window at the acceptor (which gives 0.43). Nothing was missing.

### 6. The readout is a chain, not a head
Patch L3H1 clean into the corrupt run and clamp every downstream residual write to its
unpatched-corrupt value. **The direct effect is 0.023** of a 0.818 total — L3H1's write is a
*signal*, not a logit. Releasing the stack bottom-up: 0.023 → 0.18 (MLP3) → 0.26 (L4-attn) → 0.46 (MLP4)
→ 0.53 (L5-attn) → 0.818. Every link adds; no single block mediates alone
(best: MLP3, +0.158). **There is no compact readout head that we failed to find.**

---

## What this does NOT show

- **Completeness is untested, and the obvious test has no power.** Ablating everything outside
  C destroys the coupling — but so does ablating the same *number* of random heads (p = 0.47
  for 32 heads; p = 0.97 for 47). Large ablations break the language model and the coupling
  dies whatever is deleted; at matched health the circuit-preserving ablation is *gentler*
  than random. We claimed layers 4–5 were specifically required and the null retracted it
  within the hour (R12). **C is the dominant sub-circuit, not "the circuit."**
- **Only short introns.** The donor→acceptor gap must fit the 1024 nt context (60–150 nt here).
  Typical human introns are orders of magnitude longer. Nothing here generalises to them, and
  this architecture cannot be made to test it.
- **One model.** No cross-model claim is made or supported.
- **L1H12 (0.266) is real and unexplained.**
- **The donor's query is not necessary in isolation.** Transplanting it collapses donor→partner
  attention by 95% yet costs only 29% of the coupling — because ~11 query positions in the 5′ss
  neighbourhood read the same partner redundantly. Do not say "the donor asks about its
  partner": the 5′ss *neighbourhood* does, and no single position is load-bearing.

## How to read the claim ladder

| Rung | Status |
|---|---|
| Behaviour is real, matched controls | ✅ |
| Components are necessary | ✅ (ablation + distinct matched-random null + a health control that can fail) |
| The proposed path mediates the effect | ✅ (K×V conjunction, relocated read, closed L2H15→L3H1 edge) |
| Interventions are specific | ✅ (non-partner AG, distance-matched donor, non-site key routing, control windows) |
| **Causal circuit exists** | ✅ ← **this is the current label** |
| Replicated on sealed data | ⏳ one shot, pre-registered |
| Faithfulness / completeness quantified | ❌ **the test has no power** (D15) |
| Fully characterised circuit | ❌ |
