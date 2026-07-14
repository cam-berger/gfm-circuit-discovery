# SpliceBERT — findings of record

**Everything we know, everything we withdrew, and what is still open.**
Written 2026-07-12, after the sealed test. Every number below is traceable to a
manifest-stamped artifact in `results/`. This is the document to argue from.

> **Directory name is a misnomer.** This study is called `splicebert_partner_selection` for
> historical reasons — it is the study ID that v1's configs, manifests and result paths all
> reference. The partner-*selection* task was withdrawn (R2/R3). This is the **coupling**
> study. The directory was deliberately **not** renamed, because rewriting those manifests
> would falsify provenance to fix a cosmetic name.

---

## 0. Decision summary

**What you can say, and defend:**

> SpliceBERT contains a **causally validated, sealed-replicated circuit** for same-intron
> donor–acceptor coupling, centred on **L3H1**, with **L2H15** and **L1H12** as upstream
> feeders. Its completeness we cannot quantify, and the available test for that has no power.

**What you cannot say:** anything about pointers, binding, or content-addressing. Three
separate attempts to establish it all failed (§4). The circuit is real; the *interpretation*
of how it addresses its target is not.

**One genuinely new, separate finding** fell out of probing L1H12: SpliceBERT has a **learned
codon-phase attention lattice** (42 of 96 heads). Structurally striking, **functionally
unproven** (§5). It is a different study.

**Model:** SpliceBERT.1024nt — BertForMaskedLM, 6 layers × 16 heads, hidden 512, single-
nucleotide tokenisation, absolute position embeddings.

---

## 1. The behaviour

**Native masked-LM readout — no probe, no readout head, no labels, so there is no task to win
and nothing to leak.** Mask the G of a donor `GT`; score `margin = logit(G) − max(logit A,C,T)`.
Destroy the partner acceptor's `AG` at `[a−2, a−1]` by substituting **real bases** (`CC`) —
never `[MASK]`, which is off-manifold. `coupling = margin(clean) − margin(corrupt)`.

**Sealed test, chr8/chr9, n=400, unconditional, gene-clustered bootstrap:**

| | effect |
|---|---|
| **destroy the PARTNER acceptor's AG** | **+0.872 [0.771, 0.969]** |
| destroy a motif-matched **NON-partner** AG | +0.007 |
| same corruption, scored at a **distance-matched other donor** | +0.005 |
| **partner selectivity** | **19.6× [12.0, 37.2]** |

The effect is **right-skewed** — median **0.638** vs mean 0.872. The mean is not the typical
exon; say so. Never quote the conditional mean (1.198) as the behavioural magnitude (**R10**).

---

## 2. The circuit

```
        the partner acceptor's AG at [a−2, a−1]
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
┌──────────────┐              ┌──────────────┐
│    L2H15     │              │    L1H12     │     two INDEPENDENT upstream feeders
│  −48% abl.   │              │  −18% abl.   │     (each nested in L3H1, not in each other)
│ denoise 0.00 │              │ denoise 0.00 │     necessity without carriage = MACHINERY
└──────┬───────┘              └──────┬───────┘
       │  rescue via L3H1: 0.96      │  rescue via L3H1: 1.12
       └──────────────┬──────────────┘     ← BOTH edges are CLOSED
                      ▼
        ┌─────────────────────────────┐
        │   L3H1 — the READER         │   rank 1 of 96 · −79% ablation
        │   conjunctive K×V read      │   MLM health retained: 100.6%
        │   key 0.032 · value 0.101   │
        │   key+value 0.663           │   I = +0.531 [0.477, 0.583]
        └─────────────┬───────────────┘
                      │  writes into 11 of 1026 tokens (the 5′ss neighbourhood)
                      ▼
        ┌─────────────────────────────┐
        │  MLP3 → L4 → MLP4 → L5      │   direct effect 0.023 of 0.818
        │  a DISTRIBUTED CHAIN        │   every link adds; none suffices
        └─────────────┬───────────────┘
                      ▼
              donor logit margin
```

### 2.1 Necessity — sealed test

Exhaustive 96-head ablation, both strands, val: **L3H1 0.830 [0.654, 0.994]** (#1),
**L2H15 0.515 [0.396, 0.625]** (#2), **L1H12 0.266 [0.178, 0.367]** (#3), L2H4 0.105 (#4).

| ablate | coupling lost | MLM health retained (**masked** probes) |
|---|---|---|
| L3H1 | **78.7%** | 100.6% |
| L2H15 | 47.7% | 99.8% |
| **C = {L3H1, L2H15}** | **81.6%** | **99.2%** |

**Matched-random null:** 400 **distinct** 2-head sets. Circuit drop **+0.879** vs random mean
+0.000, max +0.346. **p = 0.0025.**

The health column only became meaningful once the control could fail — the previous metric
scored the model *copying unmasked input* and **rose** when all 96 heads were ablated (**R8**).

### 2.2 Mediation — the conjunctive read (sealed test)

Acceptor corruption collapses L3H1's attention to its partner by **93%** (0.566 → 0.038).

| patch L3H1's… at the acceptor | coupling restored | attention restored |
|---|---|---|
| key only | 0.032 | 79% |
| value only | 0.101 | 0% |
| **key + value** | **0.663 [0.603, 0.716]** | 79% |
| query @ donor | 0.001 | — |
| key+value @ control position | −0.000 | 0% |

**K×V interaction I = +0.531 [0.477, 0.583]** — and it reproduced from val (+0.532) to test
(+0.531) **to three decimals, on chromosomes the circuit had never seen.** The key restores
*selection* but not the conclusion; the value restores neither alone. **The conjunction is the
mechanism.**

### 2.3 Where L3H1 writes — 11 of 1026 tokens

| region | tokens | recovery |
|---|---|---|
| donor | 1 | **0.345** |
| **donor flank (±5)** | **10** | **0.495** |
| acceptor | 11 | 0.003 |
| intron body | 88 | 0.034 |
| upstream exon | 319 | 0.007 |
| downstream | 597 | −0.001 |
| all positions | 1026 | 0.812 |

**The head READS at the acceptor and WRITES at the 5′ splice site.** The acceptor it reads
contributes 0.003 of its own causal write.

This dissolves the donor-query puzzle: transplanting L3H1's query at the donor collapses
donor→partner attention **94%** (0.648 → 0.037) but costs only **34%** of the coupling, and the
clean margin barely moves (5.233 → 4.859, so the transplant did not break the donor). Reason:
**~11 redundant query positions in the 5′ss neighbourhood read the same partner.** No single
position is load-bearing. **Do not write "the donor asks about its partner."**

### 2.4 The readout is a chain, not a head

Patch L3H1 clean into the corrupt run and clamp every downstream residual write to its
unpatched-corrupt value:

```
direct (all frozen)   0.023      ← L3H1's write is a SIGNAL, not a logit
  + MLP3              0.181
  + L4-attn           0.258
  + MLP4              0.445
  + L5-attn           0.506
  all live            0.818
```

No single block mediates (best alone: MLP3, +0.158). **There is no compact readout head that
we failed to find — there is no compact readout head.**

### 2.5 Both upstream edges are CLOSED

| | supplies | rescue via **L3H1's whole output** | calibration |
|---|---|---|---|
| **L2H15** | 0.661 | **0.959 [0.875, 1.052]** | 1.000 ✅ |
| **L1H12** | 0.271 | **1.124 [0.929, 1.435]** | 1.000 ✅ |

Nothing either head contributes reaches the behaviour except through L3H1. All 16 layer-3
heads together give 0.994 — no other L3 head is involved. (The calibration row restores the
entire preceding layer output and must return exactly 1.000 by construction; it does.)

**But the routing within L3H1 is diffuse, and we decline to tidy it.** The attractive
symmetry — "L2H15 builds the ADDRESS, L1H12 builds the QUESTION" — is **not supported**:

| restore, with the feeder ablated | L2H15 | L1H12 |
|---|---|---|
| L3H1 key @ acceptor | 0.426 | 0.434 |
| L3H1 query @ donor | 0.183 | 0.285 |
| L3H1 value | ~0 | **−0.086** |
| via MLP (2 / 1) | 0.651 | −0.322 |

Both feeders move the key *and* the query. Neither is a pure address-builder. The one clean
shared fact: **neither routes through the VALUE channel.**

### 2.6 C needs no fourth head

L1H12 was probed **as a candidate 4th component** and is not one: it adds **+0.002
[−0.027, +0.031]** on top of C. It is 95% nested in L3H1 and **independent of L2H15**
(+0.284 [0.221, 0.348] on top of it). It never looks at the partner (attention 0.003 vs L3H1's
0.62) and its attention does not move when the partner is destroyed.

**The signature shared by L2H15 and L1H12 — real necessity, ZERO denoise recovery (0.002 /
0.003) — means machinery, not carriage.** They are what the reader needs, not what it reads.

### 2.7 Faithfulness

Ablate C → **82%** of the coupling is lost. Patch C → **79% [74%, 83%]** is restored.

---

## 3. What the completeness test cannot tell us

**The obvious test has no power, and we proved it rather than reporting the flattering read.**

| ablate everything outside C | coupling retained | add-one p |
|---|---|---|
| layer-3 others (15 heads) | 124% *(it goes UP)* | 1.00 |
| layer-4+5 (32 heads) | 19% | **0.47** |
| L3-others + L4 + L5 (47 heads) | 8% | **0.97** |

Deleting 47 heads breaks the language model (health 0.53 → 0.45) and the coupling dies
**whatever you delete**. At **matched health**, the circuit-preserving ablation is *gentler* on
the coupling than a random one. This neither supports nor refutes completeness.

**We briefly claimed layers 4–5 were "specifically required" on the strength of the 19% figure,
and retracted it within the hour when the size-matched null came back (R12).**

---

## 4. What we tried to establish and COULD NOT

This section is the reason to trust §2. Four serious attempts, four failures, all recorded.

| # | Claim attempted | Why it failed |
|---|---|---|
| 1 | **"Address and content are separable" via a key/value SWAP** | **Algebraic identity.** Transposing (k,v) *pairs* permutes the summation index of `out_i = Σ_j softmax(q_i·k_j)·v_j` — output unchanged **exactly**. Reproduced on a **randomly-initialised** model (\|Δ\| = 1.2e-06). The gate passed on untrained weights. **R7.** |
| 2 | **"The address is not a coordinate" via a one-way COPY to a decoy** | **Architecturally forced.** BERT positions are absolute; the score `q_i·k_j` has no term in the slot index `j`. A (k,v) pair carried to a new slot attracts *exactly* the attention it attracted at home (measured: **9.9e-07** difference). "Relocation costs 2%" is architecture, not biology. **Retracted.** |
| 3 | **"The address is content" via transplanting the acceptor SEQUENCE** | **Null.** Pasting the partner's 25 nt acceptor context elsewhere recovers **0.031** near the true distance, **0.000** far away — barely above the shuffled control. The head needs the *original site*. (Weak null: 25 nt may not constitute a functional acceptor — no branch point, no polypyrimidine tract.) |
| 4 | **"L1H12 builds L3H1's QUERY" (a parallel-feeder story)** | **Not supported.** It moves query (0.285) and key (0.434) about equally. Diffuse. |

**Net: we have no evidence about HOW the circuit addresses its target.** What survives is that
the read is **conjunctive** (I = +0.531) and that **destroying the AG at a fixed position
collapses the attention 93%** — i.e. the key's match depends on the acceptor's *content*. That
is the floor. It does **not** license "pointer", "binding", or "content-addressed lookup".

Worth noting against a content-only story: **non-partner AGs elsewhere in the window produce
~0 coupling (+0.007)**. If the address were purely content ("find an AG"), those would be read.
They are not. **Distance is very likely part of the address**, and we have not measured how much.

---

## 5. A separate finding: seven codon combs (**not** a 42-head lattice — see R14)

Fell out of probing L1H12. **This is a different study and should be treated as one.**

> ### ⛔ RETRACTED — R14 (2026-07-12). The "42-head codon-phase lattice" does not exist.
>
> This section previously claimed: *"**42 of 96 heads** put >60% of their local attention mass on
> a single phase (mod 3); seven sit at **1.000**. All three phases are covered and near-balanced:
> **14 / 16 / 12**. **Absent at random initialisation: 0/96** → it was LEARNED."*
>
> **The metric was measuring SHARPNESS, not periodicity.**
>
> "Phase purity" = fraction of a head's local attention mass on one residue class mod 3, against
> a chance baseline of ⅓. **That statistic is maximised by a head that attends to exactly ONE
> position** — one offset has one remainder, so a single-offset head scores **1.000 by
> arithmetic.**
>
> `corr(purity, effective support) = **−0.767**` — purity rises as a head gets *narrower*.
>
> | the "purest" heads | purity | effective support | actually attends to |
> |---|---|---|---|
> | L0H4 | 1.000 | **1.00** | offset **−1** (−1 ≡ 2 mod 3) |
> | L0H12 | 1.000 | **1.00** | offset **+1** |
> | L0H5 | 1.000 | **1.00** | offset **+2** |
> | L0H14 | 1.000 | **1.00** | offset **−2** |
>
> **24 of the 42 have effective support < 2.** The 54 *non*-lattice heads have median effective
> support **37.4** — they are the *diffuse* ones, which spread over all three residue classes and
> therefore score at chance. The metric ranked heads by how *narrow* they were and called the
> narrow ones periodic.
>
> Every supporting pillar falls with it:
> - **"Three phases covered, balanced 14/16/12"** — sharp heads attend to −1, −2, +1, +2, −3, +3…,
>   and those offsets fall into the three residue classes roughly evenly. **A balanced mod-3
>   spread is exactly what phase-INDIFFERENT sharp heads produce.** The balance I read as a
>   three-channel coordinate system is evidence of nothing.
> - **"Absent at random init (0/96, mean 0.344)"** — random init has *diffuse* attention. What
>   training produced is **locality**, not phase.
> - **"Survives sequence shuffling"** — true of any positional copier.
> - **"Concentrated in L0–L4, gone by L5"** — layer 0 is where the sharp local heads live.
>
> Evidence: `results/phase-lattice/comb_check.json`.

### What actually survives — and it is better

**Seven genuine stride-3 combs.** A comb = attention spread over ≥3 offsets (effective support
`exp(entropy)` ≥ 3) whose teeth are **all congruent mod 3**:

| head | purity | eff. support | teeth | gaps |
|---|---|---|---|---|
| **L1H12** | 0.915 | 21.2 | −23, −20, −17, −14, −11 | **3, 3, 3, 3** |
| **L1H10** | 0.878 | 16.8 | −7, −4, −1, +2, +5 | **3, 3, 3, 3** |
| **L2H15** | 0.842 | 27.3 | −16, −13, −10, −7, −4 | **3, 3, 3, 3** |
| L0H11 | 0.747 | 21.0 | +6, +9, +12, +15, +18 | **3, 3, 3, 3** |
| L0H0 | 0.676 | 36.8 | −20, −17, −14, −11, −8 | **3, 3, 3, 3** |
| L2H2 | 0.674 | 35.8 | −12, −9, +6, +9, +12 | 3, (15), 3, 3 |
| L0H13 | 0.659 | 22.9 | −18, −15, −12, −9, −6 | **3, 3, 3, 3** |

**Layers 0–2 only. None in layers 3, 4 or 5.**

**L2H15 is the ADDRESSER** — the coupling circuit's upstream head (D12/D16), found by an ablation
scan on a *splicing* task that knew nothing about periodicity. Its appearance as the third-
strongest comb is an **independent convergence** and is the striking part. **L1H12 is not
independent evidence**: the entire periodicity thread began by probing it.

**One further structural fact, independent of the head analysis:** SpliceBERT's **position-
embedding matrix is itself weakly period-3**. Self-similarity between position embeddings at lags
3/6/9/12 averages **+0.648** against **+0.552** at other lags. The position code has genuine mod-3
structure.

### Function — still OPEN

The heads are **content-blind in their routing** (the shuffle test), but that constrains only the
QK circuit. Attention is `out_i = Σ_j a_ij·v_j`; the shuffle test moves the sequence and finds
`a_ij` unchanged, which says **nothing** about `v_j = W_V·x_j`. **An earlier version of this
document said the heads "read no bases." That was wrong** — the OV circuit was never tested. A
comb with a frozen attention pattern can still be a **phase-selective nucleotide aggregator**:
fixed routing, content-bearing payload.

A relative lattice has no global origin, so it cannot encode an absolute reading frame by
itself. But with all three phases present, **every nucleotide gets its neighbourhood
decomposed into three interleaved combs** — its own codon phase (±3, ±6…), the next, and the
one after. That is a phase-decomposed view of local triplet structure, built positionally and
for free in layer 0. Frame would then have to be pinned by *content* downstream.

### Function — OPEN. (The test that said "no" is **VOID** — see R13.)

The model's predictions **are** codon-periodic, and specifically in coding sequence:

| region | period-3 enrichment (1.0 = none) |
|---|---|
| **exon (coding)** | **3.34×** |
| intron | 1.33× |
| ratio | **2.50×** |

> ### ⛔ RETRACTED — R13 (2026-07-12)
>
> This section previously read: *"**But the lattice is not what produces it.** Ablating all 42
> lattice heads does not reduce that periodicity — it **rises** (3.34 → 4.52) and sits inside
> the random null: 9 of 15 size-matched random 42-head ablations did as much. Add-one
> p = 0.625. So: a striking structure, and a failed link to the obvious function."*
>
> **That negative had no power. The metric was read off a model that had lost the behaviour.**
>
> | arm | mean log-prob of the true base | probability |
> |---|---|---|
> | intact, exon | −1.120 | 0.326 |
> | **42 lattice heads ablated** | **−1.377** | **0.252** |
> | the 15 random 42-head nulls | −1.23 … −1.40 | ~0.25–0.29 |
> | **chance, ln(¼)** | **−1.386** | **0.250** |
>
> The intact model's entire headroom over chance is **0.076 nats**. The ablation removes
> **0.074 of it — 97%.** Every arm, treatment and null alike, was at chance. The period-3
> statistic was computed on the output of a model that was no longer predicting anything, and
> the *rise* in P3 is the tell: a wrecked model falls back on its position prior, which is
> trivially periodic. That is also why 9 of 15 random nulls "did as much" — they were all
> equally dead.
>
> `run_phase_function.py` never checked MLM health. The log-prob was printed in its own output
> and misread as a sanity number rather than as ln(¼).
>
> **Third instance of one failure mode** (R8: probes never masked, so the model was scored
> copying its own input; R12: an ablation compared against an unmatched-health null; now R13).
> The rule, now standing beside *"a control that cannot fail is not a control"*:
>
> > **An intervention that destroys the behaviour cannot measure the mechanism.**
> > Every arm must be shown to retain the behaviour *before* its metric is interpreted.
>
> **The function of the lattice is OPEN, not closed.** Structure (above) is unaffected.
> Replacement design: `studies/codon_phase_lattice/PREREGISTRATION_PHASE_SHIFT.md` — a
> phase-shift intervention that *rolls the routing and keeps the values*, leaving the model
> intact and the behaviour measurable.

**Where this leaves the codon periodicity.** It is real (3.34× in exons vs 1.33× in introns).
Whether the lattice produces it is **unknown**. The competing explanation — that it comes from
*content* (learned codon-biased base statistics) rather than from the phase-comb attention —
remains live, but it was never actually tested, only asserted on the strength of a void null.

*(A statistical bug was caught here too: the first version used period-3 power as a **fraction
of total power**, which is length-confounded — shorter segments have fewer Fourier bins, so
shorter exons get a higher score for free. It manufactured a "1.96× coding-specific"
periodicity out of nothing. The corrected, length-independent enrichment is reported above.)*

---

## 6. Retractions — read before quoting anything pre-2026-07-12

| ID | Withdrawn | Why |
|---|---|---|
| **R7** | D9 — the competing-acceptor "binding double dissociation" | An algebraic identity; passes on random weights. **This was the result the team was told to lead with.** |
| **R8** | MLM health 0.9958 → 0.9953 | The control scored the model **copying unmasked input**. It returns ~0.99 for everything and **rises to 0.999 with all 96 heads ablated**. It could not fail. Real baseline: **0.53**. |
| **R9** | matched-random null **p = 0.000999** | 1000 draws with replacement from a **95**-head universe at k=1 are 95 distinct values. The p was literally `1/(n_null+1)` — a function of a CLI flag. Honest value: **1/96 = 0.0104**. Evidence overstated 10.4×. |
| **R10** | coupling **1.359** [1.266, 1.450] | A **conditional** mean (filtered to ≥0.3, which conditions on the outcome) reported as a behavioural magnitude. Unconditional: **0.973** [0.919, 1.023]. |
| **R11** | Everything published before 2026-07-12 | **The study was PLUS-STRAND ONLY and never said so.** `exons` is sorted by genomic start, so on the minus strand the transcript-order next exon is `exons[i-1]`, not `exons[i+1]`. **41,153/41,153** minus-strand exons on chr8 were silently dropped. Eligible counts roughly doubled on the fix (test 735 → 1,568). **The circuit is unchanged on corrected data.** |
| **R12** | "Ablating layers 4–5 destroys the coupling, so a downstream stage is specifically required" | **Our own claim, retracted the same day.** Collateral damage: p = 0.47; 46/100 random 32-head sets did as much, and at matched health the circuit-preserving ablation is *gentler*. |
| R1–R6 | v1/v2 partner-selection task, exon-definition reading, "single dedicated head", "33% of causal mass", "refines key and value" | See `CLAIMS.md`. |

**Three of the four pre-flight blockers were controls that could not fail.** Standing rule
adopted: **a control that cannot fail is not a control** — state what would falsify a gate and
verify that result is reachable, *before* pre-registering it. Two gates were **deleted rather
than re-tuned**, because a threshold on a quantity that is constant by construction cannot be
repaired by moving the threshold. Enforced by `tests/test_circuit_memorials.py` (the
swap-is-an-identity test needs no genome and no model — it is algebra, and it runs in 0.4s).

---

## 7. Limits that must travel with every statement

1. **Only short introns.** The donor→acceptor gap must fit the 1024 nt context (60–150 nt here).
   Typical human introns are **orders of magnitude longer**. Nothing here generalises to them,
   and this architecture cannot be made to test it. This is the single biggest limitation.
2. **Completeness is unquantified** and the obvious test has **no power** (§3).
3. **C is the dominant sub-circuit, not "the circuit."** L1H12 (0.266) is real, outside C, and
   only partially explained.
4. **One model.** No cross-model claim is made or supported.
5. **We do not know how the circuit addresses its target** (§4).
6. The effect is **right-skewed**; the mean is not the typical exon.

---

## 8. Where this sits on the claim ladder

| Rung | Status |
|---|---|
| Candidate circuit | ✅ exceeded |
| Behaviour real, matched controls | ✅ |
| Components necessary | ✅ ablation + **distinct** matched-random null + a health control that **can** fail |
| Proposed path mediates | ✅ K×V conjunction; both upstream edges closed |
| Interventions specific | ✅ non-partner AG, distance-matched donor, non-site key routing, control windows |
| **Causal circuit exists** | ✅ ← **current label** |
| **Replicated on sealed data, no reselection** | ✅ **13/13, chr8/chr9, one shot** |
| Faithfulness / completeness quantified | ❌ **the test has no power** |
| Fully characterised circuit | ❌ |
| Biological predictor | ❌ not started |

---

## 9. The decision

**Three options, with honest costs.**

| | What it buys | Cost | Risk |
|---|---|---|---|
| **A. Ship the circuit** | A sealed-replicated causal circuit in a real genomic FM, with an unusually clean retraction trail. Defensible today. | Write-up only. | The "so what" is narrow: short introns, one model. Reviewers will ask about long introns and you have no answer. |
| **B. Chase the phase lattice** | Potentially the bigger result — a *learned* codon-phase organisation across ~44% of the model, connecting to your GFM-nano frame-counter work. | ⚠️ **Updated 2026-07-12.** The function link has **not** failed — the test that said so is **VOID (R13)**: it read its metric off a model that ablation had driven to chance. The question is **open and untested**. Next step is the **phase-shift intervention** (`studies/codon_phase_lattice/`), which rolls the routing and keeps the values, so the model survives the intervention and the behaviour stays measurable. | It may still be an inert positional habit — but nothing yet run has the power to say so. |
| **C. Port the assay to a longer-context model** | Attacks the **biggest** limitation head-on (long introns), and tests whether the circuit motif generalises. The leak-proof assay + memorial tests port unchanged. | New model integration. | The circuit may not exist elsewhere — which is itself a result. |

**My read:** the strongest *scientific* asset is not L3H1 — it is the **assay and the
discipline** (a behaviour with no task to win, plus five falsifiers that each killed a story we
wanted). That ports. **C** attacks the limitation that will otherwise define every review of
this work; **B** is higher-variance and has already failed its first functional test.

## Provenance

- Sealed protocol + code hashes (recorded pre-unblinding): `PREREGISTRATION_SEALED.md`
- Sealed result: `SEALED_TEST_RESULT.md`, `results/splicebert_coupling/sealed_test.json`
- Claims + retractions: `CLAIMS.md` · Decision log: `DECISIONS.md` · Mechanism: `MECHANISM.md`
- Phase lattice: `results/phase-lattice/`
- The retracted binding artifact is quarantined in `results/splicebert_coupling/retracted/`
  with its own README, so it cannot be re-cited.
- Figures 1–3 with sha256 provenance: `figures/splicebert/`
- 163 tests pass, including 5 memorial tests — one per blocker.
